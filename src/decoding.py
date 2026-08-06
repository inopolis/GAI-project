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

  Infinite-penalty limit -- corrected statement:
  With a multi-order risk risk(v) = (1/N) sum_n 1[v completes a repeated n-gram],
  letting lambda -> infinity concentrates q on argmin_v risk(v) (reweighted by p),
  NOT on the complement of a single order's blocked set. When some token has
  risk = 0 the limit is a hard block of the UNION over orders n_min..n_max; when
  every token has positive risk the limit keeps the minimum-risk tokens instead
  of being undefined. The limit therefore reproduces STANDARD no-repeat-n-gram
  only in the single-order case n_min = n_max. This is stated as such; the
  earlier blanket claim was wrong.

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
        if len(generated_ids) < 2:
            logits = logits / max(self.temperature, 1e-6)
            probs  = torch.softmax(top_p_filtering(logits.unsqueeze(0), self.top_p).squeeze(0), dim=-1)
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
                 mode=None, eps=0.05, dual_iters=30, dual_lambda_max=50.0,
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
        self.dual_lambda_max = dual_lambda_max
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
        self.feasible_history = []   # dual mode only: was eps attainable under lambda_max
        self.cap_hit_history  = []   # dual mode only: did lambda hit dual_lambda_max
        self.min_risk_history = []   # dual mode only: min_v risk(v) that step (achievable floor)
        self.violation_history = []  # dual mode only: max(0, achieved-eps)
        self.tolerance_history = []  # dual mode only: bisection window width at termination

    def reset(self):
        self._recent.clear()
        self._all_ids = []
        self._followers = {n: defaultdict(set) for n in range(self.n_min, self.n_max + 1)}
        self.alpha_history = []
        self.kl_history = []; self.hq_history = []
        self.hp_history = []; self.risk_history = []
        self.feasible_history = []; self.cap_hit_history = []
        self.min_risk_history = []; self.violation_history = []
        self.tolerance_history = []

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
    def _q_of_lambda(p, risk, lam):
        """q_lambda(v) proportional to p(v)*exp(-lam*risk(v)), computed stably."""
        logits = torch.log(p + 1e-40) - lam * risk
        return torch.softmax(logits, dim=-1)

    def _solve_dual_lambda(self, p, risk):
        """
        Exact dual calibration: return the smallest lambda >= 0 with
        E_{q_lambda}[risk] <= eps.  g(lambda) = E_{q_lambda}[risk] is
        non-increasing, so bisection on [0, lambda_max] is valid.

        Returns a dict with everything needed to audit the step, not just the
        multiplier:
          lambda        : the calibrated multiplier.
          achieved       : E_{q_lambda}[risk] at that multiplier.
          feasible       : whether the target eps was reachable within
                           dual_lambda_max at all.
          cap_hit        : True iff lambda == dual_lambda_max was returned
                           (whether or not that also happens to be feasible --
                           logged separately from `feasible` since a run can
                           hit the cap and still land under eps by coincidence
                           at the cap, which should still be visible).
          min_attainable_risk : min_v risk(v) reweighted by p among the
                           lowest-risk tokens -- the floor E_q[risk] cannot
                           go below no matter how large lambda is, since mass
                           always concentrates on argmin risk, never to zero
                           if the minimum risk itself is >0.
          violation      : max(0, achieved - eps), i.e. how much the
                           constraint is actually violated when infeasible;
                           0 when feasible.
          tolerance       : the bisection window width at termination
                           (hi - lo), the numerical precision the calibration
                           is accurate to.
        If E_p[risk] <= eps the constraint is inactive and lambda* = 0 (q = p,
        zero distortion, reported as feasible with tolerance 0).
        """
        min_attainable_risk = float(risk.min())
        g0 = float((p * risk).sum())
        if g0 <= self.eps:
            return {"lambda": 0.0, "achieved": g0, "feasible": True,
                    "cap_hit": False, "min_attainable_risk": min_attainable_risk,
                    "violation": 0.0, "tolerance": 0.0}

        hi = self.dual_lambda_max
        g_hi = float((self._q_of_lambda(p, risk, hi) * risk).sum())
        if g_hi > self.eps:
            # Infeasible under the cap: report the violation honestly instead
            # of silently returning as if the constraint had been met.
            return {"lambda": hi, "achieved": g_hi, "feasible": False,
                    "cap_hit": True, "min_attainable_risk": min_attainable_risk,
                    "violation": max(0.0, g_hi - self.eps), "tolerance": float("nan")}

        lo = 0.0
        for _ in range(self.dual_iters):
            mid = 0.5 * (lo + hi)
            g_m = float((self._q_of_lambda(p, risk, mid) * risk).sum())
            if g_m > self.eps:
                lo = mid
            else:
                hi = mid
        achieved = float((self._q_of_lambda(p, risk, hi) * risk).sum())
        return {"lambda": hi, "achieved": achieved, "feasible": True,
                "cap_hit": (hi >= self.dual_lambda_max - 1e-9),
                "min_attainable_risk": min_attainable_risk,
                "violation": max(0.0, achieved - self.eps),
                "tolerance": float(hi - lo)}

    @staticmethod
    def _kl_bits(q, p):
        m = q > 0
        return float((q[m] * (torch.log2(q[m] + 1e-40) - torch.log2(p[m] + 1e-40))).sum())

    @staticmethod
    def _entropy_bits(d):
        m = d > 0
        return float(-(d[m] * torch.log2(d[m] + 1e-40)).sum())

    def step(self, logits, generated_ids=None):
        """
        One decoding step. Records exact KL(q||p) and entropies every step, so
        the minimum-distortion claim is measured directly rather than inferred
        from a downstream NLL.

        p is the temperature-scaled model distribution -- the reference the
        projection is taken from.
        """
        vocab_size = logits.shape[-1]

        # ---- entropy_only: no recurrence risk anywhere ----
        if self.mode == "entropy_only":
            self.alpha_history.append(0.0)
            temperature = self._current_temperature()
            p = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
            q_logits = logits / max(temperature, 1e-6)
            q_logits = top_p_filtering(q_logits.unsqueeze(0), self.top_p).squeeze(0)
            q = torch.softmax(q_logits, dim=-1)
            self.kl_history.append(self._kl_bits(q, p))
            self.hq_history.append(self._entropy_bits(q))
            self.hp_history.append(self._entropy_bits(p))
            self.risk_history.append(0.0)
            token = int(torch.multinomial(q, 1).item())
            self._register(token)
            return token

        # p = reference distribution the projection starts from
        p    = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        risk = self._risk_scores(vocab_size).to(logits.device)

        if self.mode == "dual":
            # Exact solution of  min KL(q||p) s.t. E_q[risk] <= eps
            d = self._solve_dual_lambda(p, risk)
            lam, achieved = d["lambda"], d["achieved"]
            q = self._q_of_lambda(p, risk, lam)
            if self.top_p < 1.0:
                # NOTE: top-p after the projection voids exactness; off by default
                q_l = torch.log(q + 1e-40)
                q_l = top_p_filtering(q_l.unsqueeze(0), self.top_p).squeeze(0)
                q = torch.softmax(q_l, dim=-1)
            self.alpha_history.append(lam)
            self.feasible_history.append(d["feasible"])
            self.cap_hit_history.append(d["cap_hit"])
            self.min_risk_history.append(d["min_attainable_risk"])
            self.violation_history.append(d["violation"])
            self.tolerance_history.append(d["tolerance"])
        else:
            # "fixed" (KL-regularized) or "adaptive" (heuristic controller)
            lam = self._current_alpha()          # already clipped to [0, alpha_max]
            self.alpha_history.append(lam)
            q = self._q_of_lambda(p, risk, lam)
            if self.top_p < 1.0:
                q_l = torch.log(q + 1e-40)
                q_l = top_p_filtering(q_l.unsqueeze(0), self.top_p).squeeze(0)
                q = torch.softmax(q_l, dim=-1)
            achieved = float((q * risk).sum())

        self.kl_history.append(self._kl_bits(q, p))
        self.hq_history.append(self._entropy_bits(q))
        self.hp_history.append(self._entropy_bits(p))
        self.risk_history.append(achieved)

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
            out["dual_cap_hit_rate"]  = (sum(self.cap_hit_history) / n) if n else float("nan")
            out["dual_min_risk_mean"] = f(self.min_risk_history)
            out["dual_violation_mean"] = f(self.violation_history)
            out["dual_violation_max"]  = (max(self.violation_history) if self.violation_history else float("nan"))
            out["dual_tolerance_mean"] = f([t for t in self.tolerance_history if t == t])  # drop NaNs
        return out


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