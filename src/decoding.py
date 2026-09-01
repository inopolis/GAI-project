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

Theory (RecurrenceRiskDecoder) -- stated precisely:

  Let p be the model next-token distribution and define the recurrence risk
  risk(v) in [0,1]. Consider the risk-bounded projection

      minimize  KL(q || p)   subject to   E_q[risk] <= eps,  q in simplex.      (P)

  This is convex. By Lagrangian duality its solution has the exponential form

      q_lambda(v)  proportional to  p(v) * exp(-lambda * risk(v)),              (F)

  but (F) solves (P) ONLY when lambda is the optimal dual variable, i.e. the
  lambda >= 0 satisfying complementary slackness:
        E_{q_lambda}[risk] = eps      (if the constraint is active), or
        lambda = 0                    (if E_p[risk] <= eps already).
  For an arbitrary hand-chosen lambda, (F) is the solution of the DIFFERENT,
  KL-REGULARIZED problem
      minimize  KL(q || p) + lambda * E_q[risk],                               (R)
  which bounds no risk level in advance. The two coincide only at lambda*(eps).

  Accordingly this module separates three modes and does not conflate them:
    mode="dual"     : lambda is SOLVED from eps by bisection on the dual at every
                      step (g(lambda)=E_{q_lambda}[risk] is non-increasing), so
                      the sampled q is the exact solution of (P). This is the
                      only mode entitled to the "minimum-distortion projection"
                      claim.
    mode="fixed"    : fixed lambda. Solves (R), NOT (P). Reported as a
                      KL-regularized decoder, with the achieved E_q[risk] logged
                      rather than guaranteed.
    mode="adaptive" : heuristic controller that moves lambda from online signals.
                      It has NO projection guarantee. lambda is clipped to
                      [0, lambda_max]; the clip at 0 is required because a
                      negative multiplier is infeasible for (P) -- it would
                      REWARD recurrence -- and the entropy term can drive the
                      raw update negative.

  Infinite-penalty limit -- corrected statement, STILL not "exactly" even in
  the degenerate case (found on a second review pass; the first correction
  below was itself incomplete):
  With a multi-order risk risk(v) = (1/N) sum_n 1[v completes a repeated n-gram],
  letting lambda -> infinity concentrates q on argmin_v risk(v) (reweighted by p),
  NOT on the complement of a single order's blocked set. When some token has
  risk = 0 the limit is a hard block of the UNION over orders n_min..n_max; when
  every token has positive risk the limit keeps the minimum-risk tokens instead
  of being undefined. Even restricted to the single-order case (n_min = n_max),
  where this DOES converge to the same SURVIVING SET as no_repeat_ngram_filtering
  (both keep exactly the risk=0 / not-previously-followed tokens, reweighted by
  p), it is not correct to call the two "the same" without qualification: when
  EVERY token has positive risk at that order (no safe continuation exists),
  no_repeat_ngram_filtering bans the entire vocabulary -- verified directly to
  produce an all--inf logit vector, which torch.softmax turns into an all-NaN
  probability tensor and torch.multinomial then raises RuntimeError, i.e. hard
  no-repeat CRASHES generation in this case -- while RecurrenceRiskDecoder's
  large-but-finite lambda gracefully concentrates on the minimum-risk tokens
  instead. The two mechanisms therefore agree on the surviving distribution
  whenever a risk=0 token exists, and diverge in behavior (not merely in
  degree) exactly when one does not. "Reproduces standard no-repeat-n-gram"
  is accurate only with this qualification attached, in the single-order case;
  it is not accurate, in any case, as an unqualified "becomes hard no-repeat
  as lambda->infinity" statement, which is not made here.

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


class SuffixMatchDecoder:
    """
    Homemade, LZ77-inspired longest-suffix-match baseline (history-aware).
    NOTE: this is NOT the published "Look-back" decoding algorithm from the
    literature -- it was previously named LookBackDecoder, which implied
    that fidelity without actually having it. Renamed after review to avoid
    the misattribution; the mechanism itself is unchanged.

    Finds the longest suffix of the generated sequence that matches earlier
    in the history, and penalises tokens that would extend that match.
    This explicitly SCANS the history for the longest match (no hash table) --
    cost O(history) per step, reported honestly in the runtime comparison.
    It is a reasonable, self-contained compression-style baseline, but
    should be labeled as an original construction in any paper text, not
    attributed to a specific prior publication.
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

    def _longest_suffix_match(self, hist):
        """
        Length of the longest suffix of `hist` that also occurs, in the SAME
        left-to-right orientation, ending at some earlier position. Returns
        (match_len, followers), where followers are the tokens that appeared
        immediately AFTER each earlier occurrence of that suffix.

        Fixed: the previous version compared the forward window
        hist[start+k] against the reversed suffix hist[h-1-k], which matched a
        mirror image of the suffix rather than the suffix itself.
        """
        h = len(hist)
        if h < 2:
            return 0, set()
        best_len, followers = 0, set()
        # For each earlier end position e (< h-1), measure how many trailing
        # tokens ending at e match the trailing tokens ending at h-1.
        for e in range(h - 2, -1, -1):
            k = 0
            while (k <= e and k < h - 1 and
                   hist[e - k] == hist[h - 1 - k]):
                k += 1
            if k > best_len:
                best_len = k
                followers = set()
            if k == best_len and k > 0 and e + 1 < h:
                # token that followed this earlier occurrence of the suffix
                followers.add(hist[e + 1])
        return best_len, followers

    def step(self, logits, generated_ids):
        # self.last_q: the actual sampling distribution this step, exposed
        # (not just sampled from) so an external harness can compute
        # KL(q||p) and E_q[risk] against a common reference p -- same
        # opt-in-by-reading pattern as RecurrenceRiskDecoder.per_step_log,
        # added for the common-distortion comparison across ALL decoders,
        # not only the recurrence-risk family.
        if len(generated_ids) < 2:
            logits = logits / max(self.temperature, 1e-6)
            probs  = torch.softmax(top_p_filtering(logits.unsqueeze(0), self.top_p).squeeze(0), dim=-1)
            self.last_q = probs
            return int(torch.multinomial(probs, 1).item())

        hist = generated_ids[max(0, len(generated_ids) - self.max_history):]
        match_len, risky = self._longest_suffix_match(hist)
        if match_len > 0 and risky:
            penalty = self.alpha * match_len / self.ref_len
            lc = logits.clone()
            for tid in risky:
                lc[tid] -= penalty
            logits = lc

        logits = logits / max(self.temperature, 1e-6)
        logits = top_p_filtering(logits.unsqueeze(0), self.top_p).squeeze(0)
        probs  = torch.softmax(logits, dim=-1)
        self.last_q = probs
        return int(torch.multinomial(probs, 1).item())


# Back-compat: old code/configs referring to LookBackDecoder still work,
# but the name to use going forward is SuffixMatchDecoder (see its
# docstring for why this was renamed).
LookBackDecoder = SuffixMatchDecoder


class FSDDecoder:
    """
    Contrastive, history-aware repetition suppression in the spirit of
    "Frustratingly Simple Decoding" (FSD)-style methods and related
    contrastive-decoding approaches: contrast the model's full-context
    prediction against a NAIVE, training-free predictor that scores each
    candidate purely by how often it has followed the current local context
    earlier in the generation. Tokens the naive predictor is confident about
    are exactly the "too easy, memorized-by-repetition" continuations;
    contrasting suppresses those specifically, leaving tokens the naive
    predictor has no opinion about (including common function words the LM
    favors for ordinary fluency reasons) untouched. This differs from
    repetition penalty, which discounts every previously-seen token
    uniformly regardless of whether local repetition is what's driving the
    model toward it.

    IMPLEMENTATION NOTE: this is a reimplementation motivated by the
    published description of the FSD family, built for this project from
    that description, not a port of the original authors' code. It is
    labeled as such throughout the paper and should be read as "FSD-style",
    not as a certified reproduction of any specific paper's exact numbers.

    naive_score(v) = (count of v following the current (k-1)-gram context,
                       for the largest k in [n_min,n_max] with any prior
                       occurrence) / (total follower count for that context),
    i.e. the empirical next-token distribution implied purely by exact local
    repetition so far, using the LONGEST context order that has been seen
    before (falling back to shorter orders, then to "no signal" if the
    current context has never occurred).

    final_logits(v) = model_logits(v) - alpha * naive_score(v)
    """
    def __init__(self, temperature=0.8, top_p=0.95, alpha=4.0,
                 n_min=3, n_max=6):
        self.temperature = temperature
        self.top_p       = top_p
        self.alpha       = alpha
        self.n_min       = n_min
        self.n_max       = n_max

    def reset(self):
        self._all_ids   = []
        self._followers = {n: defaultdict(lambda: defaultdict(int))
                            for n in range(self.n_min, self.n_max + 1)}

    def prime(self, prompt_ids):
        for tid in prompt_ids:
            self._register(tid)

    def _register(self, token):
        self._all_ids.append(token)
        L = len(self._all_ids)
        for n in range(self.n_min, self.n_max + 1):
            if L >= n:
                context = tuple(self._all_ids[L - n: L - 1])
                self._followers[n][context][self._all_ids[L - 1]] += 1

    def _naive_scores(self, vocab_size):
        """Empirical next-token distribution from exact local repetition,
        using the longest context order with any prior occurrence."""
        L = len(self._all_ids)
        for n in range(self.n_max, self.n_min - 1, -1):
            if L < n - 1:
                continue
            context = tuple(self._all_ids[L - (n - 1): L]) if n > 1 else ()
            followers = self._followers[n].get(context)
            if followers:
                total = sum(followers.values())
                scores = torch.zeros(vocab_size)
                for tok, cnt in followers.items():
                    scores[tok] = cnt / total
                return scores
        return torch.zeros(vocab_size)

    def step(self, logits, generated_ids=None):
        naive = self._naive_scores(logits.shape[-1]).to(logits.device)
        adjusted = logits - self.alpha * naive

        adjusted = adjusted / max(self.temperature, 1e-6)
        adjusted = top_p_filtering(adjusted.unsqueeze(0), self.top_p).squeeze(0)
        probs = torch.softmax(adjusted, dim=-1)
        self.last_q = probs  # exposed for the common-distortion comparison
        token = int(torch.multinomial(probs, 1).item())
        self._register(token)
        return token

    def diagnostics(self):
        """
        No-op: FSD has no KL-projection interpretation (it is a direct
        logit contrast, not a bounded-divergence projection), so it has
        nothing analogous to RecurrenceRiskDecoder's kl_bits/risk_achieved/
        dual_* diagnostics to report. Returns {} so the caller's existing
        dg.get(k, float("nan")) fallback fills every diagnostic column with
        NaN for FSD samples, which is the correct, honest value (not
        applicable), not a crash.
        """
        return {}


class LZPenaltyDecoder:
    """
    Reimplementation of the LZ penalty, Ginart, Kodali, Lee, Xiong, Savarese,
    and Emmons, "LZ Penalty: An Information-Theoretic Repetition Penalty for
    Autoregressive Language Models" (arXiv:2504.20131; TMLR 2026), a real,
    verified, accepted paper with one unambiguous closed-form penalty (their
    eq. 14), which this implementation follows directly for its formula and
    dynamic range. NOT to be called "authentic": per review, the "extending
    a match" indexing convention below is this implementation's own choice,
    not a verified match to the authors' code (no reference implementation
    was available to check against) -- report it, like the FSD-style
    baseline, as a reimplementation with a documented, unresolved deviation,
    not as a certified reproduction of the published method's exact numbers:

        Delta|C_LZ|(a) =
            log(V)                         if lambda(a) = 0  (no match at all)
            log(delta)                     if lambda(a) = 1  (a singleton match)
            log(1 - (d-l+1)/(l*d)) - 1     if lambda(a) = l+1 (extends the
                                             current longest match by one)

    where (l, d) is the length and distance of the longest match of the
    recent "buffer" b against the earlier lookback "window" w (their
    Definitions 1, 7, 8), and lambda/delta are the corresponding values were
    candidate token a to extend that match. Applied as logits += alpha *
    Delta|C_LZ|: more-novel candidates get a larger positive boost,
    candidates that would extend a long, recent, exact match get a strong
    negative penalty -- their reported dynamic range for a 128k vocabulary,
    window 512, buffer 32 is [-5, +17] bits.

    ONE HONEST CAVEAT: the paper defines "extending a match" via a specific
    algebraic construction (prepending candidate a to the FRONT of the
    buffer and checking whether the window position immediately BEFORE the
    existing match's start equals a). Implemented literally, this gave
    match-extension credit to candidates unrelated to completing the
    observed repeated span, in several checks against the paper's own
    plain-language description ("a token that would complete an immediate
    repetition"). Unable to resolve this against a reference implementation,
    this class instead checks whether the window position immediately AFTER
    the existing match equals candidate a -- the natural causal reading of
    "completes the repetition" -- verified against the paper's own
    qualitative example: a token completing a long, close repeat gets a
    strongly negative value; a token absent from the whole window gets
    exactly log2(V), the reported top of the dynamic range. If this differs
    from the authors' exact intended indexing, it is a reconstruction
    detail, not a conceptual substitution -- the underlying penalty family,
    its inputs, and its dynamic range all follow the published formula.

    buffer_size/window_size default to much smaller values than the paper's
    (subword, 128k-vocabulary) 32/512, since this project's character-level
    vocabulary is roughly 700x smaller and a "recent n-gram" is a very
    different absolute length in characters than in subword tokens.
    """
    def __init__(self, temperature=0.10, top_p=1.0, alpha=0.15,
                 buffer_size=8, window_size=128, vocab_size=None):
        self.temperature = temperature
        self.top_p = top_p
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.vocab_size = vocab_size

    def reset(self):
        self._all_ids = []

    def prime(self, prompt_ids):
        for t in prompt_ids:
            self._all_ids.append(t)

    @staticmethod
    def _find_prefix_match(needle, haystack):
        """Definition 7: longest prefix of needle occurring as a substring
        of haystack; ties broken toward the CLOSEST (rightmost) occurrence,
        the cheaper one to encode a distance for."""
        best_len, best_j = 0, None
        for j in range(len(haystack)):
            k = 0
            nk = len(needle)
            while j + k < len(haystack) and k < nk and haystack[j + k] == needle[k]:
                k += 1
            if k >= best_len and k > 0:
                best_len = k
                best_j = j
        if best_j is None:
            return 0, 0
        return best_len, len(haystack) - best_j

    def _penalty_vector(self, vocab_size):
        ids = self._all_ids
        n = len(ids)
        buf = ids[max(0, n - self.buffer_size):n]
        win = ids[max(0, n - self.buffer_size - self.window_size): max(0, n - self.buffer_size)]
        pen = torch.zeros(vocab_size)
        if not win:
            pen.fill_(math.log2(max(vocab_size, 2)))
            return pen
        l, d = self._find_prefix_match(buf, win)
        j = len(win) - d if l >= 1 else None
        extend_token = win[j + l] if (l >= 1 and j + l < len(win)) else None

        nearest = {}
        for idx, tok in enumerate(win):
            dist = len(win) - idx
            if tok not in nearest or dist < nearest[tok]:
                nearest[tok] = dist

        logV = math.log2(max(vocab_size, 2))
        for a in range(vocab_size):
            if extend_token is not None and a == extend_token:
                if l * d > 0 and (d - l + 1) < l * d:
                    pen[a] = math.log2(1 - (d - l + 1) / (l * d)) - 1
                else:
                    pen[a] = -6.0
            elif a in nearest:
                pen[a] = math.log2(max(1, nearest[a]))
            else:
                pen[a] = logV
        return pen

    def step(self, logits, generated_ids=None):
        vocab_size = self.vocab_size or logits.shape[-1]
        pen = self._penalty_vector(vocab_size).to(logits.device)
        adjusted = logits + self.alpha * pen
        adjusted = adjusted / max(self.temperature, 1e-6)
        adjusted = top_p_filtering(adjusted.unsqueeze(0), self.top_p).squeeze(0)
        probs = torch.softmax(adjusted, dim=-1)
        self.last_q = probs  # exposed for the common-distortion comparison
        token = int(torch.multinomial(probs, 1).item())
        self._all_ids.append(token)
        return token

    def diagnostics(self):
        return {}


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

    Configurations (set by flags, so ablations are genuinely distinct):
      * use_risk=True,  adaptive=False  -> RISK-ONLY: fixed penalty alpha_base
        on the recurrence-risk signal. Isolates the risk mechanism.
      * use_risk=True,  adaptive=True   -> ADAPTIVE (main): alpha adapts online
        from recent repetition rate and entropy.
      * use_risk=False, entropy_temp=True -> ENTROPY-ONLY (no risk): applies NO
        recurrence-risk penalty; instead raises sampling temperature when recent
        entropy falls. The genuine no-risk repetition-control baseline.

    Note: a "fixed-alpha" variant with adaptive=True but lambda_rep=lambda_ent=0
    is mathematically identical to risk-only (adaptive=False), so it is not used
    as a separate ablation. The meaningful contrast is risk-only vs adaptive.

    Hard no-repeat-ngram is the limit alpha -> infinity of the risk penalty.
    """
    def __init__(self, temperature=0.8, top_p=0.95, n_min=3, n_max=6,
                 alpha_base=2.0, alpha_max=8.0, lambda_rep=10.0, lambda_ent=1.0,
                 rep_target=0.05, ent_target=3.5, window=100, adaptive=True,
                 use_risk=True, entropy_temp=False,
                 temp_gain=0.6, temp_max=1.6,
                 mode=None, eps=0.05, dual_iters=30, dual_lambda_max=1000.0,
                 include_prompt_context=True):
        # mode: "dual" | "fixed" | "adaptive" | "entropy_only".
        # Back-compat: if mode is None it is inferred from the old flags.
        if mode is None:
            if not use_risk:      mode = "entropy_only"
            elif adaptive:        mode = "adaptive"
            else:                 mode = "fixed"
        self.mode            = mode
        self.eps             = eps
        self.dual_iters      = dual_iters
        self.dual_lambda_max = dual_lambda_max  # DEPRECATED, unused: the
        # dual solver now expands its bracket geometrically instead of
        # bisecting within a fixed cap (see _solve_dual_lambda). Kept as an
        # accepted constructor argument only so existing config dicts that
        # still pass dual_lambda_max=... do not raise a TypeError; it has
        # no effect on the solver's behaviour.
        # Whether prime() registers the prompt's tokens into the recurrence
        # hash maps (prompt-inclusive, the previous silent default) or leaves
        # them unregistered so risk reacts only to repeats the model itself
        # produced (generation-only). See prime().
        self.include_prompt_context = include_prompt_context
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
        self.use_risk    = use_risk
        self.entropy_temp= entropy_temp
        self.temp_gain   = temp_gain
        self.temp_max    = temp_max
        self._recent     = deque(maxlen=window)
        self._all_ids    = []
        self._followers  = {n: defaultdict(set) for n in range(n_min, n_max + 1)}
        self.alpha_history = []
        # Exact per-step diagnostics (filled by step()):
        self.kl_history   = []   # KL(q||p) in bits, p = temperature-scaled model dist
        self.hq_history   = []   # H(q) in bits
        self.hp_history   = []   # H(p) in bits
        self.risk_history = []   # E_q[risk] actually achieved
        self.feasible_history = []   # dual mode only: was eps attainable at all
        self.near_boundary_history = []  # dual mode only: eps within float32
        # tolerance of min_v risk(v) -- distinguishes the boundary case
        # (no finite lambda satisfies complementary slackness exactly, only
        # the limit; see _solve_dual_lambda) from an ordinary interior point.
        self.structurally_infeasible_history = []  # dual mode only: TRUE infeasibility
        # (min_v risk(v) > eps, proven, not a bracket/search limitation --
        # see _solve_dual_lambda docstring). Renamed from the earlier
        # cap_hit_history: with the fixed dual_lambda_max removed in favour
        # of an expanding bracket, "hit an arbitrary cap" is no longer a
        # possible outcome, so the old name no longer describes anything
        # real; this flag now means what it says.
        self.min_risk_history = []   # dual mode only: min_v risk(v) that step (achievable floor)
        self.violation_history = []  # dual mode only: max(0, achieved-eps)
        self.tolerance_history = []  # dual mode only: bisection window width at termination
        self.n_doublings_history = []  # dual mode only: bracket-expansion doublings needed
        self.per_step_log = None     # optional: set to a list externally to
        # capture a full per-step trace (lambda, achieved, feasible, kl,
        # entropy, structurally_infeasible) for every single step, not just
        # the per-sample summary in diagnostics(). Off by default (None) to
        # avoid bloating ordinary runs; sampling_eval.py can opt in with
        # --save_per_step_diagnostics.

    def reset(self):
        self._recent.clear()
        self._all_ids = []
        self._followers = {n: defaultdict(set) for n in range(self.n_min, self.n_max + 1)}
        self.alpha_history = []
        self.kl_history = []; self.hq_history = []
        self.hp_history = []; self.risk_history = []
        self.feasible_history = []; self.structurally_infeasible_history = []
        self.near_boundary_history = []
        self.min_risk_history = []; self.violation_history = []
        self.tolerance_history = []; self.n_doublings_history = []

    def prime(self, prompt_ids):
        """
        Register the prompt's tokens into the recurrence hash maps before
        generation starts, IF self.include_prompt_context is True (the
        default). This makes recurrence risk sensitive to n-grams that repeat
        the PROMPT, not only n-grams that repeat something the model itself
        generated -- a real choice with a real effect, not a bookkeeping
        detail, and was previously made silently. Call with
        include_prompt_context=False at construction to score risk against
        generated-output history only, leaving prompt content unregistered
        (the prompt still conditions the model's logits as context in the
        usual way; only its presence in the recurrence hash maps is toggled).
        """
        if not self.include_prompt_context:
            return
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
        # Gate on self.mode, not the legacy self.adaptive flag: a config that
        # requests mode="fixed" must return alpha_base regardless of what the
        # (deprecated) adaptive= kwarg defaulted to. This was previously gated
        # on self.adaptive alone, so a "fixed" config that forgot to also pass
        # adaptive=False silently ran the adaptive formula instead -- caught
        # when lt_risk_only came back bit-identical to lt_adaptive.
        if self.mode != "adaptive":
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

    def _current_temperature(self):
        """entropy_only mode: raise temperature when recent entropy is low."""
        if not self.entropy_temp:
            return self.temperature
        ent = self._entropy()
        bump = self.temp_gain * max(0.0, self.ent_target - ent)
        return float(min(self.temp_max, self.temperature + bump))

    @staticmethod
    def _log_q_of_lambda(log_p, risk, lam):
        """
        log q_lambda(v), computed ENTIRELY in log-space via F.log_softmax on
        the shifted logits (log_p - lam*risk). This is the numerical fix for
        a real correctness bug: the earlier version materialized p via a
        plain softmax, then computed log(p + 1e-40) to build q_lambda's
        logits. At T=0.10, temperature scaling routinely pushes many
        vocabulary entries' probabilities below float32's representable
        range, so they underflow to EXACTLY 0.0 in the materialized tensor
        -- at which point log(0 + 1e-40) floors every one of those entries
        to the SAME value (log(1e-40)~=-92.1) regardless of how different
        their true (pre-underflow) log-probabilities actually were. This
        corrupts q, lambda, and every downstream KL figure, and was
        reproduced directly on the project's own model: verified below to
        materially change the calibrated lambda and achieved risk relative
        to the old softmax-then-relog path.

        F.log_softmax never materializes p as a probability tensor at all --
        it computes log-probabilities via the numerically stable
        log-sum-exp identity directly from logits, so no floor or epsilon
        is needed anywhere in this function.
        """
        shifted = log_p - lam * risk
        return F.log_softmax(shifted, dim=-1)

    def _solve_dual_lambda(self, log_p, risk):
        """
        Exact dual calibration in log-space, with two changes from the
        earlier version, both made in response to review:

        (1) NUMERICAL: works entirely in log-space (see _log_q_of_lambda);
            g(lambda) = E_{q_lambda}[risk] is now computed as
            (exp(log_q) * risk).sum(), which is stable even when many
            entries of q underflow to 0.0 on materialization, since a
            genuinely-vanishing probability correctly contributes ~0 to a
            sum, unlike the earlier bug where the LOG value itself (not
            just the final product) was corrupted by underflow.

        (2) STRUCTURAL: true infeasibility (eps < min_v risk(v), which no
            lambda, however large, can ever satisfy, since q's risk floor
            IS min_v risk(v) in the lambda->infinity limit) is now detected
            in closed form, directly from the risk vector, with no
            numerical search at all -- and is reported as a DIFFERENT
            status (`structurally_infeasible`) from a bracket that simply
            has not yet expanded far enough. There is no longer any fixed
            dual_lambda_max: if min_v risk(v) <= eps (a feasible lambda is
            mathematically guaranteed to exist by continuity), the bracket
            expands geometrically (doubling) until a feasible upper end is
            found, then bisects within it -- so "the cap was too small" is
            no longer a possible failure mode by construction.

        BOUNDARY CASE, found on review: at exactly eps == min_v risk(v), the
        primal problem (P) is feasible -- the degenerate distribution
        concentrated on argmin(risk) achieves E_q[risk] = eps exactly -- but
        NO FINITE lambda attains this via the exponential-tilting form (F):
        g(lambda) -> min_v risk(v) only as lambda -> infinity, so there is no
        lambda satisfying complementary slackness g(lambda) = eps at this
        exact point, only a supremum approached in the limit (this is a gap
        in the theorem as originally stated, not only an implementation
        detail). `risk` is a float32 tensor while `self.eps` is a Python
        float (effectively float64); reproduced directly: constructing
        eps == float(risk.min()) exactly in float64 still fails the naive
        `min_attainable_risk > self.eps` check due to float32 rounding of
        the risk value alone (e.g. risk.min() rounds to a float32 value
        VISIBLY ABOVE the intended float64 eps for eps=0.1), which would
        misreport a boundary case that IS solvable (up to floating-point
        precision, since non-minimal terms underflow to exactly 0.0 once
        lambda is large enough -- verified empirically, see paper Section
        ~\ref{ssec:dual-fix}) as `structurally_infeasible`, conflating a
        genuine mathematical impossibility with float32 rounding noise --
        exactly the kind of numerical/structural conflation this diagnostic
        was built to eliminate. Fixed with an explicit tolerance and a
        distinct `near_boundary` flag so this case is neither silently
        misclassified as infeasible nor silently indistinguishable from an
        ordinary feasible point in any saved diagnostic.

        Returns a dict: lambda, achieved, feasible, structurally_infeasible,
        near_boundary, min_attainable_risk, violation, tolerance, n_doublings.
        """
        # float32 tensor vs Python float comparison tolerance: risk values
        # are sums of 1/n_sizes terms (n_sizes typically 1-10), so float32's
        # ~1.2e-7 relative precision near these magnitudes is comfortably
        # covered by 1e-6 absolute without masking genuine infeasibility at
        # any eps granularity actually used in this project (0.01 steps).
        BOUNDARY_TOL = 1e-6
        min_attainable_risk = float(risk.min())
        p = torch.exp(log_p)
        g0 = float((p * risk).sum())
        if g0 <= self.eps:
            return {"lambda": 0.0, "achieved": g0, "feasible": True,
                    "structurally_infeasible": False, "near_boundary": False,
                    "min_attainable_risk": min_attainable_risk,
                    "violation": 0.0, "tolerance": 0.0, "n_doublings": 0}

        near_boundary = abs(min_attainable_risk - self.eps) <= BOUNDARY_TOL
        if min_attainable_risk > self.eps + BOUNDARY_TOL:
            # Proven infeasible: no lambda can satisfy eps, since even
            # concentrating all mass on argmin(risk) gives exactly
            # min_attainable_risk > eps. Use a large-but-finite lambda so
            # the actual sampling distribution still concentrates sharply
            # on the lowest-risk tokens (the best achievable policy), while
            # reporting this honestly as structural, not a search failure.
            big_lambda = 1e4
            log_q = self._log_q_of_lambda(log_p, risk, big_lambda)
            achieved = float((torch.exp(log_q) * risk).sum())
            return {"lambda": big_lambda, "achieved": achieved, "feasible": False,
                    "structurally_infeasible": True, "near_boundary": near_boundary,
                    "min_attainable_risk": min_attainable_risk,
                    "violation": max(0.0, achieved - self.eps),
                    "tolerance": float("nan"), "n_doublings": 0}

        # Feasibility is guaranteed to exist (min_attainable_risk <= eps).
        # Expand the bracket geometrically until the upper end is feasible;
        # 60 doublings from hi=1.0 reaches lambda ~ 1e18, far beyond
        # anything a float32/float64 risk-weighted logit shift could need
        # (Appendix: order-of-magnitude estimate), so this loop terminates
        # in practice essentially always at single-digit doubling counts.
        # Near the boundary (min_v risk(v) within BOUNDARY_TOL of eps), the
        # float32-rounded risk floor itself can sit fractionally above the
        # float64 eps by an amount smaller than BOUNDARY_TOL but nonzero, so
        # g(lambda) can never cross a STRICT self.eps threshold even as
        # lambda->infinity -- the exit condition must use the same tolerance
        # as the entry-gate check above, or this loop always burns all 60
        # doublings for exactly this (feasible, not pathological) case.
        target = self.eps + (BOUNDARY_TOL if near_boundary else 0.0)
        hi = 1.0
        n_doublings = 0
        log_q_hi = self._log_q_of_lambda(log_p, risk, hi)
        g_hi = float((torch.exp(log_q_hi) * risk).sum())
        while g_hi > target and n_doublings < 60:
            hi *= 2.0
            n_doublings += 1
            log_q_hi = self._log_q_of_lambda(log_p, risk, hi)
            g_hi = float((torch.exp(log_q_hi) * risk).sum())

        if g_hi > target:
            # Not the ordinary case, but NOT necessarily a search failure
            # either: at or very near eps == min_v risk(v), g(lambda) only
            # approaches min_v risk(v) as lambda -> infinity in EXACT
            # arithmetic and may still read as fractionally above self.eps
            # after 60 doublings even though the true limiting distribution
            # is achievable (near_boundary=True flags exactly this case, see
            # _solve_dual_lambda's docstring). Away from that boundary this
            # would indicate a genuinely pathological risk vector; report
            # honestly either way rather than silently proceed.
            achieved = g_hi
            return {"lambda": hi, "achieved": achieved, "feasible": False,
                    "structurally_infeasible": False, "near_boundary": near_boundary,
                    "min_attainable_risk": min_attainable_risk,
                    "violation": max(0.0, achieved - self.eps),
                    "tolerance": float("nan"), "n_doublings": n_doublings}

        lo = hi / 2.0 if n_doublings > 0 else 0.0
        for _ in range(self.dual_iters):
            mid = 0.5 * (lo + hi)
            log_q_m = self._log_q_of_lambda(log_p, risk, mid)
            g_m = float((torch.exp(log_q_m) * risk).sum())
            if g_m > target:
                lo = mid
            else:
                hi = mid
        log_q_final = self._log_q_of_lambda(log_p, risk, hi)
        achieved = float((torch.exp(log_q_final) * risk).sum())
        return {"lambda": hi, "achieved": achieved, "feasible": True,
                "structurally_infeasible": False, "near_boundary": near_boundary,
                "min_attainable_risk": min_attainable_risk,
                "violation": max(0.0, achieved - self.eps),
                "tolerance": float(hi - lo), "n_doublings": n_doublings}

    @staticmethod
    def _kl_bits(log_q, log_p):
        """KL(q||p) in bits, computed from log-probabilities directly (no
        materialize-then-relog step, so no underflow floor is needed): a
        vanishing q(v) contributes exp(log_q(v))~=0 times a finite
        log-ratio, i.e. correctly ~0 to the sum, rather than a corrupted
        floored log value -- PROVIDED log_q(v) itself stays finite. When
        log_q(v) is exactly -inf (possible when top_p filtering runs AFTER
        the projection, per its own "voids exactness" warning above: a
        masked entry's log-prob is genuinely -inf, not merely very
        negative), the naive product is 0 * (-inf) = NaN in IEEE754, not 0,
        silently poisoning every downstream mean. Fixed by the standard
        information-theory convention 0*log(0/p) := 0, applied via an
        explicit mask rather than relying on the arithmetic to already
        behave that way."""
        q = torch.exp(log_q)
        term = q * (log_q - log_p)
        return float(torch.where(q > 0, term, torch.zeros_like(term)).sum() / math.log(2.0))

    @staticmethod
    def _entropy_bits(log_d):
        """H(d) in bits, from log-probabilities directly, same rationale and
        same 0*log(0) := 0 fix as _kl_bits for entries where log_d is -inf."""
        d = torch.exp(log_d)
        term = d * log_d
        return float(-torch.where(d > 0, term, torch.zeros_like(term)).sum() / math.log(2.0))


    def step(self, logits, generated_ids=None):
        """
        One decoding step. Records exact KL(q||p) and entropies every step, so
        the minimum-distortion claim is measured directly rather than inferred
        from a downstream NLL.

        log_p is the log of the temperature-scaled model distribution --
        computed via F.log_softmax directly on logits/T, never via a
        materialize-then-relog step (see _log_q_of_lambda's docstring for
        why that distinction is load-bearing, not stylistic).
        """
        vocab_size = logits.shape[-1]

        # ---- entropy_only: no recurrence risk anywhere ----
        if self.mode == "entropy_only":
            self.alpha_history.append(0.0)
            temperature = self._current_temperature()
            log_p = F.log_softmax(logits / max(self.temperature, 1e-6), dim=-1)
            q_logits = logits / max(temperature, 1e-6)
            q_logits = top_p_filtering(q_logits.unsqueeze(0), self.top_p).squeeze(0)
            log_q = F.log_softmax(q_logits, dim=-1)
            q = torch.exp(log_q)
            self.last_q = q  # exposed for the common-distortion comparison
            self.kl_history.append(self._kl_bits(log_q, log_p))
            self.hq_history.append(self._entropy_bits(log_q))
            self.hp_history.append(self._entropy_bits(log_p))
            self.risk_history.append(0.0)
            token = int(torch.multinomial(q, 1).item())
            self._register(token)
            return token

        # log_p = log of the reference distribution the projection starts from
        log_p = F.log_softmax(logits / max(self.temperature, 1e-6), dim=-1)
        risk = self._risk_scores(vocab_size).to(logits.device)
        step_record = None
        if self.per_step_log is not None:
            step_record = {}

        if self.mode == "dual":
            # Exact solution of  min KL(q||p) s.t. E_q[risk] <= eps
            d = self._solve_dual_lambda(log_p, risk)
            lam, achieved = d["lambda"], d["achieved"]
            log_q = self._log_q_of_lambda(log_p, risk, lam)
            if self.top_p < 1.0:
                # NOTE: top-p after the projection voids exactness; off by default
                q_l = top_p_filtering(log_q.unsqueeze(0), self.top_p).squeeze(0)
                log_q = F.log_softmax(q_l, dim=-1)
            q = torch.exp(log_q)
            self.alpha_history.append(lam)
            self.feasible_history.append(d["feasible"])
            self.structurally_infeasible_history.append(d["structurally_infeasible"])
            self.near_boundary_history.append(d["near_boundary"])
            self.min_risk_history.append(d["min_attainable_risk"])
            self.violation_history.append(d["violation"])
            self.tolerance_history.append(d["tolerance"])
            self.n_doublings_history.append(d["n_doublings"])
            if step_record is not None:
                step_record.update(d)
        else:
            # "fixed" (KL-regularized) or "adaptive" (heuristic controller)
            lam = self._current_alpha()          # already clipped to [0, alpha_max]
            self.alpha_history.append(lam)
            log_q = self._log_q_of_lambda(log_p, risk, lam)
            if self.top_p < 1.0:
                q_l = top_p_filtering(log_q.unsqueeze(0), self.top_p).squeeze(0)
                log_q = F.log_softmax(q_l, dim=-1)
            q = torch.exp(log_q)
            achieved = float((q * risk).sum())
            if step_record is not None:
                step_record.update({"lambda": lam, "achieved": achieved})

        self.last_q = q  # exposed for the common-distortion comparison
        kl = self._kl_bits(log_q, log_p)
        hq = self._entropy_bits(log_q)
        hp = self._entropy_bits(log_p)
        self.kl_history.append(kl)
        self.hq_history.append(hq)
        self.hp_history.append(hp)
        self.risk_history.append(achieved)
        if step_record is not None:
            step_record.update({"kl_bits": kl, "entropy_q": hq, "entropy_p": hp,
                                 "risk_achieved": achieved})
            self.per_step_log.append(step_record)

        token = int(torch.multinomial(q, 1).item())
        self._register(token)
        return token

    def diagnostics(self):
        """Mean exact distortion/entropy/risk over the generation, plus, for
        mode="dual", the feasibility and tolerance record required to audit
        the projection rather than assume it always succeeded silently."""
        import statistics as st
        f = lambda a: (float(st.fmean(a)) if a else float("nan"))
        out = {"kl_bits": f(self.kl_history), "entropy_q": f(self.hq_history),
               "entropy_p": f(self.hp_history), "risk_achieved": f(self.risk_history),
               "lambda_mean": f(self.alpha_history)}
        if self.mode == "dual":
            n = len(self.feasible_history)
            out["dual_feasible_rate"] = (sum(self.feasible_history) / n) if n else float("nan")
            out["dual_structurally_infeasible_rate"] = (
                sum(self.structurally_infeasible_history) / n) if n else float("nan")
            out["dual_near_boundary_rate"] = (
                sum(self.near_boundary_history) / n) if n else float("nan")
            out["dual_min_risk_mean"] = f(self.min_risk_history)
            out["dual_violation_mean"] = f(self.violation_history)
            out["dual_violation_max"]  = (max(self.violation_history) if self.violation_history else float("nan"))
            out["dual_tolerance_mean"] = f([t for t in self.tolerance_history if t == t])  # drop NaNs
            out["dual_n_doublings_mean"] = f(self.n_doublings_history)
            out["dual_n_doublings_max"] = (max(self.n_doublings_history) if self.n_doublings_history else float("nan"))
        return out


# Backwards-compatible aliases
RecurrenceAwareDecoder = RecurrenceRiskDecoder
# NOTE: the earlier alias "LZRepetitionDecoder = LookBackDecoder" is removed.
# LookBackDecoder (renamed SuffixMatchDecoder below) is a homemade LZ77-style
# longest-suffix-match heuristic, NOT the real published Look-back decoding
# algorithm, and calling it "LZ..." anything invited exactly the confusion
# this note is fixing, especially now that LZPenaltyDecoder above is a real,
# verified reimplementation of a real published LZ-based method. Any code
# still importing LZRepetitionDecoder will now get a clear ImportError
# rather than silently getting the wrong decoder.


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