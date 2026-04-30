# Decoding strategies for character-level LM
import math
import torch
import torch.nn.functional as F
from collections import Counter


# Basic filters 
def top_k_filtering(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only top-k logits, set rest to -inf."""
    if k <= 0:
        return logits
    v, _ = torch.topk(logits, k)
    thresh = v[..., -1, None]
    return torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)


def top_p_filtering(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering: keep smallest set of tokens whose cumulative prob >= p."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum   = torch.cumsum(probs, dim=-1)

    mask = cum > p
    mask[..., 0] = False  # always keep at least one token

    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    unsorted = torch.empty_like(sorted_logits).scatter(-1, sorted_idx, sorted_logits)
    return unsorted


def typical_filtering(logits: torch.Tensor, mass: float = 0.9) -> torch.Tensor:
    """
    Locally typical sampling (Meister et al., 2023).
    Keeps tokens whose information content is closest to the conditional entropy,
    which tends to avoid both degenerate repetition and incoherent randomness.
    """
    if mass >= 1.0:
        return logits

    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)

    # Conditional entropy H = -sum p * log p
    entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)

    # |log p(x) - H|: how far each token is from the expected information
    surprisal_diff = torch.abs(-log_probs - entropy)

    # Sort by closeness to entropy (ascending diff)
    sorted_diff, sorted_idx = torch.sort(surprisal_diff, dim=-1)
    sorted_probs = probs.gather(-1, sorted_idx)
    cum_probs    = torch.cumsum(sorted_probs, dim=-1)

    mask = cum_probs > mass
    mask[..., 0] = False  # keep at least one

    # Map mask back to original order
    orig_mask = torch.zeros_like(mask).scatter(-1, sorted_idx, mask)
    logits = logits.masked_fill(orig_mask.bool(), float("-inf"))
    return logits


def repetition_penalty_filtering(
    logits: torch.Tensor,
    generated_ids: list,
    penalty: float = 1.3,
) -> torch.Tensor:
    """
    Repetition penalty (Keskar et al., 2019).
    Divides logits of already-generated tokens by `penalty` (>1 reduces their prob).
    """
    if penalty == 1.0 or not generated_ids:
        return logits
    for token_id in set(generated_ids):
        if logits[..., token_id] > 0:
            logits[..., token_id] /= penalty
        else:
            logits[..., token_id] *= penalty
    return logits


def no_repeat_ngram_filtering(
    logits: torch.Tensor,
    generated_ids: list,
    n: int = 4,
) -> torch.Tensor:
    """
    No-repeat n-gram blocking (Paulus et al., 2018 / Fairseq).
    Forbids generating any token that would create a repeated n-gram.
    """
    if n <= 0 or len(generated_ids) < n - 1:
        return logits
    # The last (n-1) tokens form the context we check
    context = tuple(generated_ids[-(n - 1):])
    banned  = set()
    for i in range(len(generated_ids) - (n - 1)):
        if tuple(generated_ids[i : i + n - 1]) == context:
            banned.add(generated_ids[i + n - 1])
    if banned:
        logits_clone = logits.clone()
        for token_id in banned:
            logits_clone[..., token_id] = float("-inf")
        return logits_clone
    return logits


#Mirostat v2

class MirostatSampler:
    """
    Mirostat v2 (Basu et al., 2021)
    Adaptively adjusts the sampling threshold to maintain a target surprise
    level (tau bits), which keeps perplexity approximately constant.

    Args:
        tau: target surprise in bits (default 3.0; lower = more focused)
        eta: learning rate for mu update (default 0.1)
        vocab_size: vocabulary size
    """
    def __init__(self, tau: float = 3.0, eta: float = 0.1, vocab_size: int = 256):
        self.tau        = tau
        self.eta        = eta
        self.mu         = 2 * tau          # initial mu
        self.vocab_size = vocab_size

    def sample(self, logits: torch.Tensor) -> int:
        """
        Given logits (1D, vocab_size), return a single sampled token id
        and update internal state.
        """
        # Sort descending
        sorted_logits, sorted_ids = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)

        # Truncate at mu: keep tokens whose estimated surprise <= mu
        # Surprise of token i ≈ -log2(p_i)  (approximation for Mirostat v2)
        surprises = -torch.log2(probs + 1e-10)
        cutoff    = int((surprises <= self.mu).sum().item())
        cutoff    = max(1, cutoff)

        trunc_probs = probs[:cutoff]
        trunc_probs = trunc_probs / trunc_probs.sum()  # renormalize

        # Sample
        chosen_local = torch.multinomial(trunc_probs, num_samples=1).item()
        chosen_id    = sorted_ids[chosen_local].item()

        # Update mu
        observed_surprise = -math.log2(float(probs[chosen_local].item()) + 1e-10)
        error       = observed_surprise - self.tau
        self.mu    -= self.eta * error

        return int(chosen_id)

    def reset(self):
        self.mu = 2 * self.tau


#Core generate(

@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens: int,
    temperature:    float = 1.0,
    top_k:          int   = 0,
    top_p:          float = 1.0,
    typical_p:      float = 1.0,
    rep_penalty:    float = 1.0,
    no_repeat_ngram: int  = 0,
    mirostat_tau:   float = 0.0,   # 0.0 = disabled; >0 activates Mirostat v2
    mirostat_eta:   float = 0.1,
) -> torch.Tensor:
    """
    Autoregressive generation with all decoding strategies.

    idx: (B, T) int64 prompt tensor. Only B=1 is supported for stateful
         strategies (mirostat, repetition_penalty, no_repeat_ngram).

    Returns: (B, T + max_new_tokens) int64 tensor.
    """
    model.eval()
    device = next(model.parameters()).device
    idx    = idx.to(device)
    B      = idx.shape[0]

    # State for stateful strategies (batch=1 only)
    mirostat = None
    if mirostat_tau > 0.0:
        assert B == 1, "Mirostat requires batch size 1"
        mirostat = MirostatSampler(
            tau=mirostat_tau, eta=mirostat_eta,
            vocab_size=model.vocab_size
        )

    generated_ids = idx[0].tolist() if B == 1 else []

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # (B, V)

        # Mirostat (handles its own sampling internally) 
        if mirostat is not None:
            next_id = mirostat.sample(logits[0])
            generated_ids.append(next_id)
            idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
            continue

        # Greedy
        if temperature == 0.0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            if B == 1:
                generated_ids.append(int(next_id[0, 0].item()))
            idx = torch.cat([idx, next_id], dim=1)
            continue

        #Stochastic path
        logits = logits / temperature

        # Repetition penalty (B=1 only)
        if rep_penalty != 1.0 and B == 1:
            logits[0] = repetition_penalty_filtering(logits[0], generated_ids, rep_penalty)

        # No-repeat n-gram (B=1 only)
        if no_repeat_ngram > 0 and B == 1:
            logits[0] = no_repeat_ngram_filtering(logits[0], generated_ids, no_repeat_ngram)

        # Standard filters
        logits = top_k_filtering(logits, top_k)
        logits = top_p_filtering(logits, top_p)
        logits = typical_filtering(logits, typical_p)

        probs   = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if B == 1:
            generated_ids.append(int(next_id[0, 0].item()))

        idx = torch.cat([idx, next_id], dim=1)

    return idx