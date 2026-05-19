"""
src/decoding.py

Decoding strategies for character-level LM:
  - Greedy
  - Temperature sampling
  - Top-k filtering
  - Nucleus / top-p filtering          [Holtzman et al., 2020]
  - Typical sampling                   [Meister et al., 2023]
  - Repetition penalty                 [Keskar et al., 2019]
  - No-repeat n-gram blocking          [Paulus et al., 2018]  -- hard constraint
  - Mirostat v2                        [Basu et al., 2021]
  - RecurrenceAwareDecoder             -- soft exponential penalty (novel)
  - LZRepetitionDecoder                -- LZ77-style history-aware baseline

Theoretical note on RecurrenceAwareDecoder:
  The soft penalty can be derived as the solution to a minimum-distortion
  recurrence-risk control problem:

      minimize  KL( q || p )
      subject to  E_q[ risk(v) ] <= epsilon

  By Lagrangian duality the optimal q has the form:
      q(v)  proportional to  p(v) * exp( -lambda * risk(v) )

  which is equivalent to subtracting (lambda * risk(v)) from the log-prob
  before softmax — exactly what RecurrenceAwareDecoder does with alpha=lambda.
  The hard no-repeat constraint is the limit as lambda -> infinity.
  This gives the exponential penalty form a clean theoretical justification.
"""

import math
import time
import torch
import torch.nn.functional as F
from collections import Counter, deque


def top_k_filtering(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    v, _ = torch.topk(logits, k)
    thresh = v[..., -1, None]
    return torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)


def top_p_filtering(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum   = torch.cumsum(probs, dim=-1)
    mask  = cum > p
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    return torch.empty_like(sorted_logits).scatter(-1, sorted_idx, sorted_logits)


def typical_filtering(logits: torch.Tensor, mass: float = 0.9) -> torch.Tensor:
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


def repetition_penalty_filtering(
    logits: torch.Tensor, generated_ids: list, penalty: float = 1.3
) -> torch.Tensor:
    if penalty == 1.0 or not generated_ids:
        return logits
    for tid in set(generated_ids):
        if logits[..., tid] > 0:
            logits[..., tid] /= penalty
        else:
            logits[..., tid] *= penalty
    return logits


def no_repeat_ngram_filtering(
    logits: torch.Tensor, generated_ids: list, n: int = 4
) -> torch.Tensor:
    """Hard constraint — bans tokens that would create a repeated n-gram.
    This is the infinite-penalty limit of RecurrenceAwareDecoder.
    Reported as a hard-constraint baseline only; it mechanically eliminates
    the repetition metric it wins on."""
    if n <= 0 or len(generated_ids) < n - 1:
        return logits
    context = tuple(generated_ids[-(n - 1):])
    banned  = set()
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
    def __init__(self, tau: float = 3.0, eta: float = 0.1, vocab_size: int = 256):
        self.tau, self.eta = tau, eta
        self.mu = 2 * tau
        self.vocab_size = vocab_size

    def sample(self, logits: torch.Tensor) -> int:
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


class LZRepetitionDecoder:
    """
    LZ77-style history-aware baseline decoder.

    At each step, finds the longest suffix of the current generated sequence
    that matches anywhere in the history. The length of this match (normalised
    by a reference length) gives a per-step repetition risk. Candidates that
    would extend this match are penalised proportionally.

    This is a principled look-back baseline: it penalises not individual
    repeated n-grams (like no-repeat-ngram) but continuation of the longest
    currently active repeated suffix — closer to how LZ77 compression detects
    redundancy.

    Parameters:
        temperature   : sampling temperature
        top_p         : nucleus filter after penalty
        alpha         : penalty strength (logit units per unit match length)
        max_history   : how many past tokens to search for matches
        ref_len       : normalisation constant for match length (typ. n_chars/4)
    """
    def __init__(
        self,
        temperature : float = 0.8,
        top_p       : float = 0.95,
        alpha       : float = 3.0,
        max_history : int   = 400,
        ref_len     : int   = 20,
    ):
        self.temperature = temperature
        self.top_p       = top_p
        self.alpha       = alpha
        self.max_history = max_history
        self.ref_len     = ref_len

    def reset(self):
        pass

    def _longest_suffix_match(self, generated_ids: list) -> int:
        """
        Finds the longest suffix of generated_ids that also appears
        earlier in generated_ids (within max_history tokens).
        Returns the length of that suffix (0 if no match).
        """
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

    def step(self, logits: torch.Tensor, generated_ids: list) -> int:
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


class RecurrenceAwareDecoder:
    """
    Online recurrence-aware decoder with soft exponential penalty.

    Derived from minimum-distortion recurrence-risk control:
      q(v) proportional to p(v) * exp(-alpha * risk(v))

    where risk(v) = fraction of n-gram sizes in {n_min..n_max} for which
    appending v creates a repeated suffix. alpha adapts online:
      alpha_t = alpha_base
                + lambda_rep * max(0, rep_rate_window - rep_target)
                - lambda_ent * max(0, entropy_window  - ent_target)

    Hard no-repeat-ngram is the limit as alpha -> infinity.
    """
    def __init__(
        self,
        temperature : float = 0.8,
        top_p       : float = 0.95,
        n_min       : int   = 3,
        n_max       : int   = 6,
        alpha_base  : float = 2.0,
        alpha_max   : float = 8.0,
        lambda_rep  : float = 10.0,
        lambda_ent  : float = 1.0,
        rep_target  : float = 0.05,
        ent_target  : float = 3.5,
        window      : int   = 100,
    ):
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
        self._recent: deque = deque(maxlen=window)
        self.alpha_history: list = []

    def reset(self):
        self._recent.clear()
        self.alpha_history.clear()

    def _rep_rate(self) -> float:
        seq = list(self._recent)
        if len(seq) < 5:
            return 0.0
        grams = [tuple(seq[i:i+5]) for i in range(len(seq)-4)]
        c = Counter(grams)
        return sum(v-1 for v in c.values() if v > 1) / len(grams)

    def _entropy(self) -> float:
        seq = list(self._recent)
        if not seq:
            return 0.0
        c = Counter(seq); t = len(seq)
        return -sum((v/t)*math.log2(v/t) for v in c.values())

    def _current_alpha(self) -> float:
        alpha = (self.alpha_base
                 + self.lambda_rep * max(0.0, self._rep_rate() - self.rep_target)
                 - self.lambda_ent * max(0.0, self._entropy() - self.ent_target))
        return float(max(0.0, min(self.alpha_max, alpha)))

    def _risk_scores(self, generated_ids: list, vocab_size: int) -> torch.Tensor:
        risk = torch.zeros(vocab_size)
        if len(generated_ids) < self.n_min:
            return risk
        for n in range(self.n_min, self.n_max + 1):
            if len(generated_ids) < n - 1:
                continue
            context    = tuple(generated_ids[-(n - 1):])
            seen_after = set()
            for i in range(len(generated_ids) - (n - 1)):
                if tuple(generated_ids[i : i + n - 1]) == context:
                    seen_after.add(generated_ids[i + n - 1])
            for tid in seen_after:
                risk[tid] += 1.0 / self.n_sizes
        return risk

    def step(self, logits: torch.Tensor, generated_ids: list) -> int:
        vocab_size = logits.shape[-1]
        alpha      = self._current_alpha()
        self.alpha_history.append(alpha)
        risk   = self._risk_scores(generated_ids, vocab_size).to(logits.device)
        logits = logits - alpha * risk
        logits = logits / max(self.temperature, 1e-6)
        logits = top_p_filtering(logits.unsqueeze(0), self.top_p).squeeze(0)
        probs  = torch.softmax(logits, dim=-1)
        token  = int(torch.multinomial(probs, 1).item())
        self._recent.append(token)
        return token


@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens  : int,
    temperature     : float = 1.0,
    top_k           : int   = 0,
    top_p           : float = 1.0,
    typical_p       : float = 1.0,
    rep_penalty     : float = 1.0,
    no_repeat_ngram : int   = 0,
    mirostat_tau    : float = 0.0,
    mirostat_eta    : float = 0.1,
    adaptive        = None,
    lz_decoder      = None,
    measure_time    : bool  = False,
) -> tuple:
    """
    Unified autoregressive generation.

    Returns (token_tensor, chars_per_sec) where chars_per_sec is None
    unless measure_time=True.

    Priority: adaptive > lz_decoder > mirostat > greedy > stochastic.
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
    if lz_decoder is not None:
        lz_decoder.reset()

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
            logits[0] = no_repeat_ngram_filtering(logits[0], generated_ids, no_repeat_ngram)
        logits  = top_k_filtering(logits, top_k)
        logits  = top_p_filtering(logits, top_p)
        logits  = typical_filtering(logits, typical_p)
        probs   = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        if B == 1:
            generated_ids.append(int(next_id[0, 0].item()))
        idx = torch.cat([idx, next_id], dim=1)

    cps = None
    if measure_time:
        elapsed = time.perf_counter() - t0
        cps = round(max_new_tokens / elapsed, 1) if elapsed > 0 else 0.0

    return idx, cps