"""
src/decoding.py

Recurrence-Risk Decoding for repetition-loop control, plus standard baselines.

Strategies:
  - Greedy
  - Temperature sampling
  - Top-k filtering
  - Nucleus / top-p filtering          [Holtzman et al., 2020]
  - Typical sampling                   [Meister et al., 2023]
  - Repetition penalty                 [Keskar et al., 2019]
  - No-repeat n-gram blocking          [Paulus et al., 2018]  -- hard constraint
  - Mirostat v2                        [Basu et al., 2021]
  - LookBackDecoder (LZ77-style)       -- history-aware baseline
  - RecurrenceRiskDecoder              -- soft exponential penalty (this work)

Theory (RecurrenceRiskDecoder):
  The decoder is a KL projection of the model distribution p onto the set of
  distributions whose expected recurrence risk is bounded:

      minimize  KL( q || p )
      subject to  E_q[ risk(v) ] <= epsilon

  By Lagrangian duality the optimal q has the closed form

      q(v)  proportional to  p(v) * exp( -lambda * risk(v) )

  i.e. we subtract (lambda * risk(v)) from each logit before softmax.
  This is "minimum-distortion": among all distributions meeting the risk
  bound, q changes p as little as possible (in KL). Two limits:
    * lambda -> 0     recovers the unmodified model distribution p
    * lambda -> inf   recovers hard no-repeat-ngram (any risky token banned)
  The method is softer than hard no-repeat and more local than repetition
  penalty, which discounts a token everywhere in the history regardless of
  whether a repeated n-gram is actually imminent.

Implementation note on recurrence risk:
  risk(v) is computed from an INCREMENTALLY MAINTAINED hash map from each
  (n-1)-gram context to the set of tokens that have followed it. At every
  step we (a) read the set for the current context in O(1) per n, and
  (b) register the newly generated (n-1)-gram -> token edge. We never
  rescan the full history. This matches the O(1)-amortised-per-step claim
  in the paper. See RecurrenceRiskDecoder._register / _risk_scores.
"""

import math
import time
import torch
import torch.nn.functional as F
from collections import Counter, deque, defaultdict


def top_k_filtering(logits, k):
    if k <= 0:
        return logits
    k = min(k, logits.shape[-1])   # clamp to vocab size
    v, _ = torch.topk(logits, k)
    thresh = v[..., -1, None]
    return torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)


def top_p_filtering(logits, p):
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum   = torch.cumsum(probs, dim=-1)
    mask  = cum > p
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    return torch.empty_like(sorted_logits).scatter(-1, sorted_idx, sorted_logits)


def typical_filtering(logits, mass=0.9):
    if mass >= 1.0:
        return logits
    probs        = torch.softmax(logits, dim=-1)
    log_probs    = torch.log(probs + 1e-10)
    entropy      = -(probs * log_probs).sum(dim=-1, keepdim=True)
    surprisal_d  = torch.abs(-log_probs - entropy)
    sorted_d, si = torch.sort(surprisal_d, dim=-1)
    cum          = torch.cumsum(probs.gather(-1, si), dim=-1)
    mask         = cum > mass
    mask[..., 0] = False
    orig_mask    = torch.zeros_like(mask).scatter(-1, si, mask)
    return logits.masked_fill(orig_mask.bool(), float("-inf"))


def repetition_penalty_filtering(logits, generated_ids, penalty=1.3):
    """Global history-based penalty (Keskar et al., 2019).
    Divides the logit of every previously generated token. This is a GLOBAL
    penalty (scans set(generated_ids)), in contrast to the local
    recurrence-risk penalty."""
    if penalty == 1.0 or not generated_ids:
        return logits
    for tid in set(generated_ids):
        if logits[..., tid] > 0:
            logits[..., tid] /= penalty
        else:
            logits[..., tid] *= penalty
    return logits


def no_repeat_ngram_filtering(logits, generated_ids, n=4, follower_map=None):
    """Hard constraint — bans tokens that would create a repeated n-gram.
    This is the infinite-penalty limit of RecurrenceRiskDecoder. Kept SEPARATE
    in all analysis because it directly constrains the measured failure event.
    If follower_map is given (context -> set of followers), uses O(1) lookup."""
    if n <= 0 or len(generated_ids) < n - 1:
        return logits
    context = tuple(generated_ids[-(n - 1):])
    if follower_map is not None:
        banned = follower_map.get(context, None)
    else:
        banned = set()
        for i in range(len(generated_ids) - (n - 1)):
            if tuple(generated_ids[i : i + n - 1]) == context:
                banned.add(generated_ids[i + n - 1])
    if banned:
        lc = logits.clone()
        for tid in banned:
            lc[..., tid] = float("-inf")
        return lc
    return logits


class MirostatSampler:
    """Mirostat v2 (Basu et al., 2021). Keeps surprise near tau bits."""
    def __init__(self, tau=3.0, eta=0.1, vocab_size=256):
        self.tau, self.eta = tau, eta
        self.mu = 2 * tau
        self.vocab_size = vocab_size

    def sample(self, logits):
        sl, si = torch.sort(logits, descending=True)
        probs  = torch.softmax(sl, dim=-1)
        surp   = -torch.log2(probs + 1e-10)
        cutoff = max(1, int((surp <= self.mu).sum().item()))
        tp     = probs[:cutoff] / probs[:cutoff].sum()
        local  = int(torch.multinomial(tp, 1).item())
        chosen = int(si[local].item())
        self.mu -= self.eta * (-math.log2(float(probs[local].item()) + 1e-10) - self.tau)
        return chosen

    def reset(self):
        self.mu = 2 * self.tau


class LookBackDecoder:
    """
    LZ77-style look-back baseline (history-aware).

    Finds the longest suffix of the generated sequence that matches earlier
    in the history, and penalises tokens that would extend that match.
    This explicitly SCANS the history for the longest match (no hash table) --
    cost O(history) per step, reported honestly in the runtime comparison.
    It is the natural compression-style baseline against which the
    recurrence-risk hash method is compared.
    """
    def __init__(self, temperature=0.8, top_p=0.95, alpha=3.0,
                 max_history=400, ref_len=20):
        self.temperature = temperature
        self.top_p       = top_p
        self.alpha       = alpha
        self.max_history = max_history
        self.ref_len     = ref_len

    def reset(self):
        pass

    def _longest_suffix_match(self, generated_ids):
        n    = len(generated_ids)
        hist = generated_ids[max(0, n - self.max_history):]
        h    = len(hist)
        best = 0
        for start in range(h - 1):
            length = 0
            while (start + length < h - 1 and
                   length < h - start - 1 and
                   hist[start + length] == hist[h - 1 - length]):
                length += 1
                best = max(best, length)
        return best

    def step(self, logits, generated_ids):
        if len(generated_ids) < 2:
            logits = logits / max(self.temperature, 1e-6)
            probs  = torch.softmax(top_p_filtering(logits.unsqueeze(0), self.top_p).squeeze(0), dim=-1)
            return int(torch.multinomial(probs, 1).item())

        match_len = self._longest_suffix_match(generated_ids)
        if match_len > 0:
            hist    = generated_ids[max(0, len(generated_ids) - self.max_history):]
            h       = len(hist)
            penalty = self.alpha * match_len / self.ref_len
            suffix  = tuple(hist[h - match_len:])
            risky   = set()
            for i in range(h - match_len):
                if tuple(hist[i : i + match_len]) == suffix and i + match_len < h:
                    risky.add(hist[i + match_len])
            if risky:
                lc = logits.clone()
                for tid in risky:
                    lc[tid] -= penalty
                logits = lc

        logits = logits / max(self.temperature, 1e-6)
        logits = top_p_filtering(logits.unsqueeze(0), self.top_p).squeeze(0)
        probs  = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())


class RecurrenceRiskDecoder:
    """
    Recurrence-Risk Decoding (this work).

    KL projection of p onto {q : E_q[risk(v)] <= eps}, giving
        q(v)  proportional to  p(v) * exp(-alpha * risk(v))
    where
        risk(v) = (1/N) * sum_{n=n_min}^{n_max}
                  1[ token v has followed the current (n-1)-gram context before ]

    risk(v) is read from an incrementally maintained hash map
    (context -> set of follower tokens), one map per n. After each token is
    emitted we register the new edge. No full-history rescan, so per-step
    cost is O(N) hash ops plus O(#risky) logit writes.

    Modes:
      * adaptive=True  (main configuration): alpha adapts online from recent
        repetition rate and entropy.
      * adaptive=False (risk-only ablation): alpha fixed at alpha_base; this
        isolates the recurrence-risk signal.

    Hard no-repeat-ngram is the limit alpha -> infinity.
    """
    def __init__(self, temperature=0.8, top_p=0.95, n_min=3, n_max=6,
                 alpha_base=2.0, alpha_max=8.0, lambda_rep=10.0, lambda_ent=1.0,
                 rep_target=0.05, ent_target=3.5, window=100, adaptive=True):
        self.temperature = temperature
        self.top_p       = top_p
        self.n_min       = n_min
        self.n_max       = n_max
        self.n_sizes     = n_max - n_min + 1
        self.alpha_base  = alpha_base
        self.alpha_max   = alpha_max
        self.lambda_rep  = lambda_rep
        self.lambda_ent  = lambda_ent
        self.rep_target  = rep_target
        self.ent_target  = ent_target
        self.window      = window
        self.adaptive    = adaptive
        self._recent     = deque(maxlen=window)
        self._all_ids    = []
        self._followers  = {n: defaultdict(set) for n in range(n_min, n_max + 1)}
        self.alpha_history = []

    def reset(self):
        self._recent.clear()
        self._all_ids = []
        self._followers = {n: defaultdict(set) for n in range(self.n_min, self.n_max + 1)}
        self.alpha_history = []

    def prime(self, prompt_ids):
        """Register the prompt's n-gram edges so risk is correct from step 1."""
        for tid in prompt_ids:
            self._register(tid)

    def _register(self, token):
        """Add n-gram edges ending at the newly appended token, O(N)."""
        self._all_ids.append(token)
        L = len(self._all_ids)
        for n in range(self.n_min, self.n_max + 1):
            if L >= n:
                context  = tuple(self._all_ids[L - n : L - 1])
                follower = self._all_ids[L - 1]
                self._followers[n][context].add(follower)
        self._recent.append(token)

    def _rep_rate(self):
        seq = list(self._recent)
        if len(seq) < 5:
            return 0.0
        grams = [tuple(seq[i:i+5]) for i in range(len(seq)-4)]
        c = Counter(grams)
        return sum(v-1 for v in c.values() if v > 1) / len(grams)

    def _entropy(self):
        seq = list(self._recent)
        if not seq:
            return 0.0
        c = Counter(seq); t = len(seq)
        return -sum((v/t)*math.log2(v/t) for v in c.values())

    def _current_alpha(self):
        if not self.adaptive:
            return self.alpha_base
        alpha = (self.alpha_base
                 + self.lambda_rep * max(0.0, self._rep_rate() - self.rep_target)
                 - self.lambda_ent * max(0.0, self._entropy() - self.ent_target))
        return float(max(0.0, min(self.alpha_max, alpha)))

    def _risk_scores(self, vocab_size):
        """O(N) hash lookups using incrementally maintained follower maps."""
        risk = torch.zeros(vocab_size)
        L = len(self._all_ids)
        if L < self.n_min - 1:
            return risk
        for n in range(self.n_min, self.n_max + 1):
            if L < n - 1:
                continue
            context   = tuple(self._all_ids[L - (n - 1):])
            followers = self._followers[n].get(context)
            if followers:
                inc = 1.0 / self.n_sizes
                for tid in followers:
                    risk[tid] += inc
        return risk

    def step(self, logits, generated_ids=None):
        vocab_size = logits.shape[-1]
        alpha      = self._current_alpha()
        self.alpha_history.append(alpha)

        risk   = self._risk_scores(vocab_size).to(logits.device)
        logits = logits - alpha * risk
        logits = logits / max(self.temperature, 1e-6)
        logits = top_p_filtering(logits.unsqueeze(0), self.top_p).squeeze(0)
        probs  = torch.softmax(logits, dim=-1)
        token  = int(torch.multinomial(probs, 1).item())

        self._register(token)
        return token


# Backwards-compatible aliases
RecurrenceAwareDecoder = RecurrenceRiskDecoder
LZRepetitionDecoder    = LookBackDecoder


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=0, top_p=1.0,
             typical_p=1.0, rep_penalty=1.0, no_repeat_ngram=0,
             mirostat_tau=0.0, mirostat_eta=0.1, adaptive=None,
             lz_decoder=None, measure_time=False):
    """
    Unified autoregressive generation.
    Returns (token_tensor, chars_per_sec). chars_per_sec is None unless
    measure_time=True. Priority: adaptive > lz_decoder > mirostat > greedy >
    stochastic. The stochastic no_repeat_ngram path uses an incrementally
    maintained follower map (O(1) per step).
    """
    model.eval()
    device = next(model.parameters()).device
    idx    = idx.to(device)
    B      = idx.shape[0]

    mirostat      = None
    generated_ids = idx[0].tolist() if B == 1 else []

    if mirostat_tau > 0.0 and adaptive is None and lz_decoder is None:
        assert B == 1
        mirostat = MirostatSampler(mirostat_tau, mirostat_eta, model.vocab_size)

    if adaptive is not None:
        adaptive.reset()
        adaptive.prime(generated_ids)
    if lz_decoder is not None:
        lz_decoder.reset()

    nr_followers = None
    if no_repeat_ngram > 0 and adaptive is None and lz_decoder is None and mirostat is None:
        nr_followers = defaultdict(set)
        ids = generated_ids
        for i in range(len(ids) - (no_repeat_ngram - 1)):
            ctx = tuple(ids[i : i + no_repeat_ngram - 1])
            nr_followers[ctx].add(ids[i + no_repeat_ngram - 1])

    t0 = time.perf_counter() if measure_time else None

    for _ in range(max_new_tokens):
        idx_cond  = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits    = logits[:, -1, :]

        if adaptive is not None:
            assert B == 1
            next_id = adaptive.step(logits[0], generated_ids)
            generated_ids.append(next_id)
            idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
            continue

        if lz_decoder is not None:
            assert B == 1
            next_id = lz_decoder.step(logits[0], generated_ids)
            generated_ids.append(next_id)
            idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
            continue

        if mirostat is not None:
            next_id = mirostat.sample(logits[0])
            generated_ids.append(next_id)
            idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
            continue

        if temperature == 0.0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            if B == 1:
                generated_ids.append(int(next_id[0, 0].item()))
            idx = torch.cat([idx, next_id], dim=1)
            continue

        logits = logits / temperature
        if rep_penalty != 1.0 and B == 1:
            logits[0] = repetition_penalty_filtering(logits[0], generated_ids, rep_penalty)
        if no_repeat_ngram > 0 and B == 1:
            logits[0] = no_repeat_ngram_filtering(
                logits[0], generated_ids, no_repeat_ngram, follower_map=nr_followers)
        logits  = top_k_filtering(logits, top_k)
        logits  = top_p_filtering(logits, top_p)
        logits  = typical_filtering(logits, typical_p)
        probs   = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        tid = int(next_id[0, 0].item())
        if B == 1:
            generated_ids.append(tid)
            if nr_followers is not None and len(generated_ids) >= no_repeat_ngram:
                ctx = tuple(generated_ids[-no_repeat_ngram:-1])
                nr_followers[ctx].add(generated_ids[-1])
        idx = torch.cat([idx, next_id], dim=1)

    cps = None
    if measure_time:
        elapsed = time.perf_counter() - t0
        cps = round(max_new_tokens / elapsed, 1) if elapsed > 0 else 0.0

    return idx, cps