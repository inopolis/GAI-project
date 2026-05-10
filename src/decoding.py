"""
src/decoding.py — Decoding strategies for character-level LM.

Strategies:
  - Greedy
  - Temperature sampling
  - Top-k filtering
  - Nucleus (top-p) filtering     [Holtzman et al., 2020]
  - Typical sampling              [Meister et al., 2023]
  - Repetition penalt             [Keskar et al., 2019]
  - No-repeat n-gram blocking      [Paulus et al., 2018]
  - Mirostat v2                      [Basu et al., 2021]
  - RecurrenceAwareDecoder (new)       — soft per-candidate penalty
                                         based on online rep/entropy signals
"""

import math
import torch
import torch.nn.functional as F
from collections import Counter, deque


# Basic filters

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
    Reported as a baseline only; it mechanically eliminates the repetition
    metric it wins on and is not a probabilistic quality improvement."""
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


# Mirostat v2 

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
        tp     = probs[:cutoff]
        tp     = tp / tp.sum()
        local  = torch.multinomial(tp, 1).item()
        chosen = int(si[local].item())
        self.mu -= self.eta * (-math.log2(float(probs[local].item()) + 1e-10) - self.tau)
        return chosen

    def reset(self):
        self.mu = 2 * self.tau


# Recurrence-Aware Adaptive Decoder

class RecurrenceAwareDecoder:
    """
    Online recurrence-aware decoder that softly penalises candidate tokens
    which would extend repeated n-grams, and adapts penalty strength based
    on recent repetition rate and entropy.

    Design:
      At each step t, before sampling:
        1. Compute per-candidate recurrence risk score r(v):
             r(v) = (number of n-gram sizes in {n_min..n_max} for which
                     appending v to the last (n-1) generated tokens creates
                     a suffix that has already appeared) / n_sizes
           This is a soft version of no-repeat-ngram: instead of hard-banning,
           we subtract alpha * r(v) from the logit of v.

        2. Adapt alpha online:
             alpha_t = alpha_base + lambda_rep  * (rep_rate_window - rep_target)
                                  - lambda_ent  * (entropy_window  - ent_target)
             clamped to [0, alpha_max].
           High recent repetition  => increase penalty.
           High recent entropy     => decrease penalty (model is healthy).

        3. Apply temperature and optionally top-p after the recurrence penalty.

    Parameters:
        temperature : base sampling temperature
        top_p : nucleus filtering after penalty (1.0 = disabled)
        n_min, n_max : n-gram range to check for recurrence risk
        alpha_base   : base penalty strength (logit units)
        alpha_max    : maximum allowed penalty
        lambda_rep   : sensitivity to recent repetition rate
        lambda_ent   : sensitivity to recent entropy
        rep_target   : target repetition rate (below = no extra penalty)
        ent_target    : target entropy (above = reduce penalty)
        window         : number of recent tokens used to compute rep/ent signals
    """
    def __init__(
        self,
        temperature : float = 0.8,
        top_p       : float = 1.0,
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

        # Runtime state
        self._recent: deque = deque(maxlen=window)
        self.alpha_history: list = []

    def reset(self):
        self._recent.clear()
        self.alpha_history.clear()

    #Online signals

    def _rep_rate(self) -> float:
        """Repetition rate of 5-grams in recent window."""
        seq = list(self._recent)
        if len(seq) < 5:
            return 0.0
        grams = [tuple(seq[i:i+5]) for i in range(len(seq)-4)]
        c = Counter(grams)
        return sum(v-1 for v in c.values() if v > 1) / len(grams)

    def _entropy(self) -> float:
        """Unigram entropy of recent window (proxy for diversity)."""
        seq = list(self._recent)
        if not seq:
            return 0.0
        c = Counter(seq)
        t = len(seq)
        return -sum((v/t)*math.log2(v/t) for v in c.values())

    def _current_alpha(self) -> float:
        rep  = self._rep_rate()
        ent  = self._entropy()
        alpha = (self.alpha_base
                 + self.lambda_rep * max(0.0, rep - self.rep_target)
                 - self.lambda_ent * max(0.0, ent - self.ent_target))
        return float(max(0.0, min(self.alpha_max, alpha)))

    #  Per-candidate recurrence risk

    def _risk_scores(self, generated_ids: list, vocab_size: int) -> torch.Tensor:
        """
        Returns a (vocab_size,) float tensor of recurrence risk in [0, 1].
        risk[v] = fraction of n-gram sizes for which appending v creates a repeat.
        """
        risk = torch.zeros(vocab_size)
        if len(generated_ids) < self.n_min:
            return risk

        # Build suffix lookup for each n in [n_min, n_max]
        for n in range(self.n_min, self.n_max + 1):
            if len(generated_ids) < n - 1:
                continue
            context = tuple(generated_ids[-(n - 1):])  # last n-1 tokens
            # Find all tokens that have followed this context before
            seen_after = set()
            for i in range(len(generated_ids) - (n - 1)):
                if tuple(generated_ids[i : i + n - 1]) == context:
                    seen_after.add(generated_ids[i + n - 1])
            for tid in seen_after:
                risk[tid] += 1.0 / self.n_sizes

        return risk  # in [0, 1]

    # Step

    def step(self, logits: torch.Tensor, generated_ids: list) -> int:
        """
        Apply recurrence-aware penalty and sample one token.
        logits: (vocab_size,) raw logits from the model.
        Returns sampled token id (int).
        """
        vocab_size = logits.shape[-1]
        alpha      = self._current_alpha()
        self.alpha_history.append(alpha)

        # 1. Soft recurrence penalty
        risk   = self._risk_scores(generated_ids, vocab_size).to(logits.device)
        logits = logits - alpha * risk

        # 2. Temperature
        logits = logits / max(self.temperature, 1e-6)

        # 3. Nucleus filter
        logits = top_p_filtering(logits.unsqueeze(0), self.top_p).squeeze(0)

        # 4. Sample
        probs  = torch.softmax(logits, dim=-1)
        token  = int(torch.multinomial(probs, 1).item())

        # 5. Update window
        self._recent.append(token)
        return token


# generate() — unified entry point
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
    adaptive        : "RecurrenceAwareDecoder | None" = None,
) -> torch.Tensor:
    """
    Unified autoregressive generation.

    For the recurrence-aware decoder pass an initialised RecurrenceAwareDecoder
    instance as `adaptive`. All other parameters are ignored when adaptive is set.

    idx: (B, T) int64. B=1 required for stateful strategies.
    Returns: (B, T + max_new_tokens) int64.
    """
    model.eval()
    device = next(model.parameters()).device
    idx    = idx.to(device)
    B      = idx.shape[0]

    mirostat      = None
    generated_ids = idx[0].tolist() if B == 1 else []

    if mirostat_tau > 0.0 and adaptive is None:
        assert B == 1
        mirostat = MirostatSampler(mirostat_tau, mirostat_eta, model.vocab_size)

    if adaptive is not None:
        adaptive.reset()

    for _ in range(max_new_tokens):
        idx_cond  = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits    = logits[:, -1, :]  # (B, V)

        # Recurrence-aware adaptive decoder
        if adaptive is not None:
            assert B == 1
            next_id = adaptive.step(logits[0], generated_ids)
            generated_ids.append(next_id)
            idx = torch.cat([idx,
                             torch.tensor([[next_id]], device=device)], dim=1)
            continue

        # Mirostat
        if mirostat is not None:
            next_id = mirostat.sample(logits[0])
            generated_ids.append(next_id)
            idx = torch.cat([idx,
                             torch.tensor([[next_id]], device=device)], dim=1)
            continue

        # Greedy
        if temperature == 0.0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            if B == 1:
                generated_ids.append(int(next_id[0, 0].item()))
            idx = torch.cat([idx, next_id], dim=1)
            continue

        # Stochastic path 
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

    return idx