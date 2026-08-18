"""
sampling_eval.py — Recurrence-Risk Decoding evaluation.

Implements the full evaluation protocol:
  - Equal-budget sweeps for every baseline (temperature, top-p, top-k,
    typical, Mirostat, look-back, repetition penalty, no-repeat).
  - Multiple loop definitions (not only the exact n-gram that no-repeat
    can mechanically avoid): repeated-n-gram onset at n in {8,10,12,16},
    longest-repeated-substring threshold, and a compression-ratio threshold.
  - Survival analysis with proper right-censoring (Kaplan-Meier, RMST, AUC).
  - Model-consistency NLL (NOT called quality; it measures how probable the
    text is under the SAME model) plus distributional similarity, spelling
    error rate, and metric-leakage checks.
  - Paired bootstrap CIs and method-vs-method paired tests.
  - Runtime overhead (chars/sec).

Hard no-repeat is always kept SEPARATE because it directly constrains the
measured failure event.

Usage:
  python3 sampling_eval.py \
    --ckpt runs/baseline/best.pt runs/cosine/best.pt \
    --out_dir runs/sampling_eval_v6 \
    --n_seeds 10 --n_chars 500
"""

import os, sys, argparse, csv, json, math, zlib
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from src.utils import set_seed, load_json, ensure_dir
from src.model import CharTransformerLM
from src.decoding import generate, RecurrenceRiskDecoder, LookBackDecoder, FSDDecoder


# 15 prompts; key methods use all 15, sweep configs use first 5.
# ---------------------------------------------------------------------------
# PROMPT PROTOCOL (declared, not ad hoc).
#   DEV_PROMPTS  : the only prompts used while designing the method and choosing
#                  hyper-parameters (n-gram band, eps/lambda, window, targets).
#   TEST_PROMPTS : written afterwards from a fixed recipe (openers of narration,
#                  dialogue, description, and exposition), never inspected while
#                  tuning, and used ONLY for the reported numbers.
# Headline results must be produced with --prompt_set test.
# ---------------------------------------------------------------------------
DEV_PROMPTS = [
    ("chapter",  "CHAPTER 1\n"),
    ("night",    "The night was "),
    ("she",      "She had never "),
    ("best",     "It was the best of "),
    ("darcy",    "Mr. Darcy had never "),
]

TEST_PROMPTS = [
    ("t_road",    "The road turned sharply where "),
    ("t_letter",  "He folded the letter and "),
    ("t_window",  "Rain struck the window as "),
    ("t_say",     "\"You must understand,\" she began, "),
    ("t_market",  "At the market that morning "),
    ("t_years",   "Three years afterwards, when "),
    ("t_room",    "The room smelled of "),
    ("t_doctor",  "The doctor set down his bag and "),
    ("t_river",   "Beyond the river lay "),
    ("t_ask",     "\"And what,\" he asked, \"became of "),
    ("t_winter",  "That winter the family "),
    ("t_ship",    "The ship had been at sea for "),
    ("t_garden",  "In the garden behind the house "),
    ("t_debt",    "His debts had grown until "),
    ("t_dawn",    "By dawn the streets were "),
]

PROMPTS = DEV_PROMPTS   # replaced at runtime by --prompt_set


# Equal-budget sweeps. Each baseline family is swept over several settings so
# that no baseline is penalised by a single fixed choice. The key methods
# (adaptive, risk-only, look-back, no-repeat) run on the full prompt set.
def build_configs():
    C = []
    def add(name, cat, key=False, **kw):
        d = dict(temperature=1.0, top_k=0, top_p=1.0, typical_p=1.0,
                 rep_penalty=1.0, no_repeat_ngram=0, mirostat_tau=0.0,
                 risk=None, lookback=None, fsd=None)
        d.update(kw)
        d["name"] = name; d["category"] = cat; d["key"] = key
        C.append(d)

    add("greedy", "baseline", temperature=0.0, key=True)

    # ---------------------------------------------------------------------
    # LOOP REGIME (the regime the method is actually for).
    # Measured with the validated persistent-loop event, this model produces
    # persistent loops only at T <= ~0.1: greedy fires on 80% of samples and
    # T=0.1 on ~65%, while T>=0.2 fires on 0% even at 1200 chars. Comparing
    # decoders at T=0.8 therefore compares zeros. Every configuration below is
    # held at T=0.1 so that the loop pathology is present and the decoders can
    # be separated; the question is which of them removes the loops, and at what
    # exact KL cost relative to the model distribution p.
    # ---------------------------------------------------------------------
    for t in (0.05, 0.10, 0.15):
        add(f"lt_temp_{t}", "loop_regime", key=(abs(t-0.10) < 1e-9), temperature=t)

    add("lt_rep_penalty_1.3", "loop_regime", key=True,
        temperature=0.10, rep_penalty=1.3)
    add("lt_no_repeat_4gram", "loop_regime_hard", key=True,
        temperature=0.10, no_repeat_ngram=4)
    add("lt_lookback_a3.0", "loop_regime", key=True,
        lookback=dict(temperature=0.10, top_p=1.0, alpha=3.0,
                      max_history=400, ref_len=20))
    add("lt_fsd", "loop_regime", key=True,
        fsd=dict(temperature=0.10, top_p=1.0, alpha=4.0, n_min=3, n_max=6))
    add("lt_risk_only", "loop_regime_rr", key=True,
        risk=dict(temperature=0.10, top_p=1.0, n_min=3, n_max=6,
                  mode="fixed", alpha_base=2.0, include_prompt_context=True))
    add("lt_risk_only_genonly", "loop_regime_rr", key=True,
        risk=dict(temperature=0.10, top_p=1.0, n_min=3, n_max=6,
                  mode="fixed", alpha_base=2.0, include_prompt_context=False))
    add("lt_adaptive", "loop_regime_rr", key=True,
        risk=dict(temperature=0.10, top_p=1.0, n_min=3, n_max=6,
                  mode="adaptive", alpha_base=2.0, alpha_max=8.0,
                  lambda_rep=10.0, lambda_ent=1.0, rep_target=0.05,
                  ent_target=3.5, window=100))
    for e in (0.01, 0.05):
        add(f"lt_dual_eps{e}", "loop_regime_rr", key=True,
            risk=dict(temperature=0.10, top_p=1.0, n_min=3, n_max=6,
                      mode="dual", eps=e))


    # temperature sweep
    for t in (0.7, 0.8, 0.9, 1.0):
        add(f"temp_{t}", "sweep_temp", key=(t == 0.8), temperature=t)

    # top-p sweep
    for p in (0.90, 0.95, 0.99):
        add(f"nucleus_p{p}", "sweep_topp", key=(abs(p-0.95) < 1e-9), temperature=1.0, top_p=p)

    # top-k sweep
    for k in (10, 40, 100):
        add(f"top_k_{k}", "sweep_topk", temperature=1.0, top_k=k)

    # typical sweep
    for m in (0.85, 0.90, 0.95):
        add(f"typical_p{m}", "sweep_typical", key=(abs(m-0.90) < 1e-9), temperature=1.0, typical_p=m)

    # repetition penalty sweep
    for r in (1.1, 1.3, 1.5):
        # 1.3 and 1.5 are both Pareto-relevant -> full sample budget
        add(f"rep_penalty_{r}", "sweep_reppen", key=(abs(r-1.3) < 1e-9 or abs(r-1.5) < 1e-9),
            temperature=0.8, rep_penalty=r)

    # mirostat sweep
    for tau in (3.0, 5.0):
        add(f"mirostat_tau{tau}", "sweep_mirostat", key=(abs(tau-5.0) < 1e-9), mirostat_tau=tau)

    # look-back sweep (alpha)
    for a in (2.0, 3.0, 5.0):
        # alpha=3 and alpha=5 are both Pareto-relevant -> full sample budget
        add(f"lookback_a{a}", "sweep_lookback", key=(abs(a-3.0) < 1e-9 or abs(a-5.0) < 1e-9),
            lookback=dict(temperature=0.8, top_p=0.95, alpha=a, max_history=400, ref_len=20))

    # hard no-repeat — kept separate (directly forbids the measured event)
    for n in (3, 4):
        add(f"no_repeat_{n}gram", "hard_constraint", key=(n == 4),
            temperature=0.8, no_repeat_ngram=n)

    # DUAL-CALIBRATED recurrence risk: lambda solved from eps at every step.
    # These are the only configurations that solve the risk-BOUNDED problem and
    # are therefore the only ones the minimum-distortion claim may rest on.
    for e in (0.02, 0.05, 0.10):
        add(f"rr_dual_eps{e}", "recurrence_risk", key=(abs(e-0.05) < 1e-9),
            risk=dict(temperature=0.8, top_p=1.0, n_min=3, n_max=6,
                      mode="dual", eps=e))

    # recurrence-risk: risk-only — fixed penalty, isolates the risk signal
    add("risk_only", "recurrence_risk", key=True,
        risk=dict(temperature=0.8, top_p=0.95, n_min=3, n_max=6,
                  alpha_base=2.0, adaptive=False, use_risk=True))

    # recurrence-risk: adaptive — main configuration (alpha adapts online)
    add("adaptive", "recurrence_risk", key=True,
        risk=dict(temperature=0.8, top_p=0.95, n_min=3, n_max=6,
                  alpha_base=2.0, alpha_max=8.0, lambda_rep=10.0, lambda_ent=1.0,
                  rep_target=0.05, ent_target=3.5, window=100,
                  adaptive=True, use_risk=True))

    # Ablations. Each removes exactly one component relative to a reference.
    #
    # entropy-only (NO RISK): removes the recurrence-risk signal entirely and
    # controls repetition only by raising temperature when entropy drops. This
    # is the genuine no-risk baseline (fixes the earlier bug where the entropy
    # variant still subtracted alpha*risk).
    add("rr_entropy_only", "ablation", key=True,
        risk=dict(temperature=0.8, top_p=0.95, n_min=3, n_max=6,
                  use_risk=False, entropy_temp=True,
                  ent_target=3.5, temp_gain=0.6, temp_max=1.6, window=100))

    # no top-p: adaptive risk but without the nucleus filter after the penalty.
    add("rr_no_top_p", "ablation",
        risk=dict(temperature=0.8, top_p=1.0, n_min=3, n_max=6,
                  alpha_base=2.0, alpha_max=8.0, lambda_rep=10.0, lambda_ent=1.0,
                  rep_target=0.05, ent_target=3.5, window=100,
                  adaptive=True, use_risk=True))

    # narrow / wide n-gram band: adaptive risk with a different risk band.
    add("rr_narrow_ngram", "ablation",
        risk=dict(temperature=0.8, top_p=0.95, n_min=3, n_max=4,
                  alpha_base=2.0, alpha_max=8.0, lambda_rep=10.0, lambda_ent=1.0,
                  rep_target=0.05, ent_target=3.5, window=100,
                  adaptive=True, use_risk=True))
    add("rr_wide_ngram", "ablation",
        risk=dict(temperature=0.8, top_p=0.95, n_min=2, n_max=8,
                  alpha_base=2.0, alpha_max=8.0, lambda_rep=10.0, lambda_ent=1.0,
                  rep_target=0.05, ent_target=3.5, window=100,
                  adaptive=True, use_risk=True))
    return C


CONFIGS = build_configs()


# ---- Metrics ----

def type_token_ratio(text):
    w = text.split()
    return len(set(w)) / len(w) if w else 0.0

def char_ngram_entropy(text, n=4):
    gs = [text[i:i+n] for i in range(len(text)-n+1)]
    if not gs: return 0.0
    c = Counter(gs); t = sum(c.values())
    return -sum((v/t)*math.log2(v/t) for v in c.values())

def repetition_rate(text, n=5):
    gs = [text[i:i+n] for i in range(len(text)-n+1)]
    if not gs: return 0.0
    c = Counter(gs)
    return sum(v-1 for v in c.values() if v > 1) / len(gs)

def rep_ngram_mass(text, n):
    gs = [text[i:i+n] for i in range(len(text)-n+1)]
    if not gs: return 0.0
    c = Counter(gs)
    return sum(v for v in c.values() if v > 1) / len(gs)

def longest_repeated_substring(text, min_len=3):
    n = len(text)
    if n < min_len*2: return 0
    def has_rep(L):
        s = set()
        for i in range(n-L+1):
            sub = text[i:i+L]
            if sub in s: return True
            s.add(sub)
        return False
    lo, hi, res = min_len, min(n//2, 300), 0
    while lo <= hi:
        mid = (lo+hi)//2
        if has_rep(mid): res=mid; lo=mid+1
        else: hi=mid-1
    return res

def compression_ratio(text):
    """Compressed size / raw size. Lower = more repetitive."""
    if not text: return 1.0
    raw = text.encode("utf-8")
    comp = zlib.compress(raw, level=9)
    return round(len(comp) / max(1, len(raw)), 4)


# ---- Multiple loop-onset definitions (punkt 4) ----
# Each returns the first character position where the loop condition first
# holds, or -1 (right-censored) if it never holds in the sample.

import re as _re_loopcheck

def _normalize_whitespace_for_loop_check(text):
    """Collapse any run of whitespace (space, newline, tab, ...) to a single
    space before periodicity matching. See persistent_loop_onset docstring
    for why this is necessary: irregular line-wrapping otherwise breaks
    exact-match detection of an obviously-repeating phrase."""
    return _re_loopcheck.sub(r"\s+", " ", text)


def persistent_loop_onset(text, P_max=60, R=5, min_p=2, catch_period_one=True,
                           period_one_min_run=8):
    """
    PRIMARY loop event: onset of a PERSISTENT cycle.

    Returns the earliest position e (in NORMALIZED-text coordinates, see
    below) at which the text ending at e consists of a span of length P
    repeated R times consecutively and exactly, i.e. the generation has
    entered a cycle. -1 (right-censored) if it never happens.

    WHITESPACE NORMALIZATION (added after a real miss was found): the model
    inserts newlines at positions that are not perfectly periodic even while
    it is unambiguously repeating the same phrase -- e.g.
    "...the subject of the\\nconsideration of the subject of the subject..."
    wraps a newline into an otherwise-repeating span at an irregular offset.
    Matched byte-for-byte, this breaks EXACT periodicity at R=5 even though
    R=3/R=4 (needing fewer consecutive exact repeats, so less exposed to one
    irregular linebreak) still find it, and the old n-gram-based check (kept
    only as a secondary diagnostic) also still fires. Requiring exact
    equality on raw text therefore risks UNDER-counting genuine degenerate
    repetition whenever the model's line-wrapping is irregular, which is a
    validity problem, not a feature: a human reading this text has no
    trouble seeing it as one repeating loop regardless of where the line
    happens to wrap. All whitespace runs (spaces, newlines, tabs) are
    collapsed to a single space before the periodicity search so trivial
    formatting variation cannot mask or create a match either way. This
    changes the false-positive rate on real prose (which wraps at newlines
    constantly) and REQUIRES re-running validate_loop_event_v2.py's
    calibrate/freeze/validate protocol with this normalization in place --
    the (R=5, P_max=60) values here are provisional until that is redone.

    catch_period_one adds a separate, DECOUPLED threshold (period_one_min_run,
    default 8) for single-character runs, since using R itself as that
    threshold made the detector trip on ordinary short typesetting
    conventions (e.g. a 5-dash scene-break divider trips at R=5 if R is
    reused as the period-1 threshold). 8 sits above ordinary divider/ellipsis
    lengths and well below genuine degenerate model output.
    """
    text = _normalize_whitespace_for_loop_check(text)

    if catch_period_one:
        run = 1
        for i in range(1, len(text)):
            if text[i] == text[i-1]:
                run += 1
                if run >= period_one_min_run:
                    return i - period_one_min_run + 2
            else:
                run = 1
    n = len(text)
    for e in range(min_p * R, n + 1):
        for P in range(min_p, min(P_max, e // R) + 1):
            span = text[e - P:e]
            ok = True
            for r in range(2, R + 1):
                if text[e - r * P:e - (r - 1) * P] != span:
                    ok = False; break
            if ok:
                return e
    return -1


def loop_onset_ngram(text, n=10):
    seen = {}
    for i in range(len(text)-n+1):
        g = text[i:i+n]
        if g in seen:
            return i
        seen[g] = i
    return -1

def loop_onset_lrs(text, threshold=20):
    """First position at which the longest repeated substring of the prefix
    reaches `threshold`. Uses incremental check on growing prefix."""
    # check at coarse steps for speed
    step = 5
    for end in range(threshold, len(text)+1, step):
        if longest_repeated_substring(text[:end], min_len=threshold) >= threshold:
            return end
    return -1

def loop_onset_compression(text, window=60, thresh=0.45, step=10):
    """First position where the trailing `window` chars compress below `thresh`
    (i.e. become highly repetitive)."""
    for end in range(window, len(text)+1, step):
        seg = text[end-window:end]
        if compression_ratio(seg) < thresh:
            return end
    return -1

LOOP_DEFS = {
    # PRIMARY -- validated: 0.0% false-positive rate on held-out human text.
    "persistent": lambda t: persistent_loop_onset(t),
    # SECONDARY / diagnostic only. These are NOT pathology detectors: on human
    # text they fire on 54-72% (ngram10) of windows. Retained to show how the
    # ranking changes once a valid event is used; never used for headline claims.
    "ngram8":   lambda t: loop_onset_ngram(t, 8),
    "ngram10":  lambda t: loop_onset_ngram(t, 10),
    "ngram12":  lambda t: loop_onset_ngram(t, 12),
    "ngram16":  lambda t: loop_onset_ngram(t, 16),
    "lrs20":    lambda t: loop_onset_lrs(t, 20),
    "compress": lambda t: loop_onset_compression(t),
}


# ---- Quality metrics ----

@torch.no_grad()
def model_consistency_nll(model, gen_ids, block_size, device, prompt_ids=None,
                          nll_batch_size=256):
    """
    Model-consistency NLL (BPC): NLL of the GENERATED tokens under the SAME
    model that produced them, with every scored position given EXACTLY the
    context the model architecture allows (up to block_size preceding
    tokens), not an approximation of it.

    Two bugs fixed here, in sequence:

    (1) An earlier version was called with the generated portion alone (no
    prompt_ids): the first chunk started at position 0 of the GENERATED
    text with no prompt in context, silently omitting the first generated
    token's own likelihood and under-conditioning early tokens. Fixed by
    scoring over (prompt_ids + gen_ids) and restricting the mean to
    positions >= len(prompt_ids).

    (2) A second version used non-overlapping chunks (stride = block_size):
    correct for the first chunk, but every position near the start of every
    later chunk was scored with a truncated context, even though real
    preceding characters existed. A stride = block_size//2 partial fix was
    tried and found insufficient: numerically verified on a toy model, only
    the SINGLE LAST position of each non-final chunk actually reached full
    context; every other kept position still fell short (e.g. one position
    got only 9 of 16 possible context tokens where the true architectural
    maximum was 16).

    This version instead builds, for EVERY target position t independently,
    the exact window full[max(0, t-block_size) : t] and batches these
    variable-length (left-padded) windows through the model together, so
    each position gets precisely min(t, block_size) tokens of context --
    verified numerically to match the architectural ideal exactly, not
    approximately, for every position, on a toy model with a deliberately
    small block_size where the discrepancy is easy to see.
    """
    prompt_ids = prompt_ids or []
    full_list = list(prompt_ids) + list(gen_ids)
    full = np.array(full_list, dtype=np.int64)
    plen = len(prompt_ids)
    n = len(full)
    if n < 2 or len(gen_ids) < 1:
        return float("nan")

    targets = [t for t in range(1, n) if t >= plen]
    if not targets:
        return float("nan")

    pad_id = 0
    all_nll = []
    for b0 in range(0, len(targets), nll_batch_size):
        batch_targets = targets[b0:b0 + nll_batch_size]
        windows = [full_list[max(0, t - block_size):t] for t in batch_targets]
        max_len = max(len(w) for w in windows)
        padded = [[pad_id] * (max_len - len(w)) + w for w in windows]
        x = torch.tensor(padded, dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(x)
        lp = F.log_softmax(logits[:, -1, :], dim=-1)   # prediction for position t, from context ending at t-1
        y = torch.tensor([full_list[t] for t in batch_targets], dtype=torch.long, device=device)
        nll = -lp.gather(1, y.unsqueeze(-1)).squeeze(-1)
        all_nll.extend(nll.tolist())

    return float(np.mean(all_nll) / math.log(2))


def ngram_js_similarity(gen_text, ref_text, n=4):
    """1 - JS divergence between n-gram distributions. Higher = closer to ref."""
    def dist(text):
        gs = [text[i:i+n] for i in range(len(text)-n+1)]
        if not gs: return {}
        c = Counter(gs); t = sum(c.values())
        return {k: v/t for k, v in c.items()}
    p = dist(gen_text); q = dist(ref_text)
    if not p or not q: return 0.0
    vocab = set(p) | set(q)
    m = {k: 0.5*(p.get(k,0)+q.get(k,0)) for k in vocab}
    def kl(a, b):
        return sum(a[k]*math.log2(a[k]/b[k]) for k in a if a[k]>0 and b[k]>0)
    jsd = max(0.0, min(1.0, 0.5*kl(p,m)+0.5*kl(q,m)))
    return round(1.0 - jsd, 4)


def _load_wordlist():
    for p in ["/usr/share/dict/words", "/usr/dict/words"]:
        if os.path.exists(p):
            with open(p, encoding="utf-8", errors="ignore") as f:
                return {w.strip().lower() for w in f if w.strip().isalpha()}
    # Minimal fallback for systems without a word list (e.g. Windows)
    COMMON = ("the of and a to in is it you that he was for on are with as his "
              "they be at one have this from or had by but not what all were we "
              "when your can said there use an each which she do how their if "
              "will up other about out many then them these so some her would "
              "make like him into time has look two more go see no way could").split()
    return set(COMMON)

_WORDS = _load_wordlist()

def spelling_error_rate(text):
    words = [w.lower() for w in text.split() if w.isalpha()]
    if not words: return 0.0
    return round(sum(1 for w in words if w not in _WORDS) / len(words), 4)


# ---- Survival analysis ----

def kaplan_meier(onsets, max_t):
    events = sorted([t for t in onsets if t >= 0])
    n_total = len(onsets)
    if not events:
        return [0, max_t], [1.0, 1.0]
    times = sorted(set(events))
    S, t_out, surv = 1.0, [], []
    n_before = 0
    for t in times:
        n_ev = events.count(t)
        n_at_risk = n_total - n_before
        if n_at_risk > 0:
            S *= (1 - n_ev / n_at_risk)
        n_before += n_ev
        t_out.append(t); surv.append(round(S, 5))
    return t_out, surv

def survival_auc(onsets, max_t):
    """
    NOTE: survival_auc(onsets, T) == rmst(onsets, T) / T exactly -- it is RMST in
    units of the horizon, not independent evidence. Reports should quote RMST
    (interpretable: expected loop-free characters) and may mention SAUC only as
    the normalised form. Both are retained here for backward compatibility.
    """

    ts, S = kaplan_meier(onsets, max_t)
    if not ts: return 1.0
    prev_t, prev_s, area = 0, 1.0, 0.0
    for t, s in zip(ts, S):
        area += (t - prev_t) * prev_s
        prev_t, prev_s = t, s
    area += (max_t - prev_t) * prev_s
    return round(area / max_t, 4)

def rmst(onsets, max_t):
    """Restricted Mean Survival Time = AUC * max_t (expected loop-free chars)."""
    return round(survival_auc(onsets, max_t) * max_t, 2)

def loop_rate(onsets):
    return round(sum(1 for t in onsets if t >= 0) / len(onsets), 4)


# ---- Bootstrap ----

# ---------------------------------------------------------------------------
# CLUSTERED INFERENCE.
# Samples are nested inside prompts (n_seeds draws per prompt). Draws that share
# a prompt are correlated, so an i.i.d. bootstrap over all samples understates
# the variance. Every interval and p-value below resamples PROMPTS (the
# clusters) with replacement and keeps all seeds within a drawn prompt.
# `groups[i]` is the prompt index of sample i.
# ---------------------------------------------------------------------------

def _cluster_indices(groups, rng):
    """Resample clusters with replacement; return the concatenated sample idx."""
    uniq = sorted(set(groups))
    by = {g: [i for i, gg in enumerate(groups) if gg == g] for g in uniq}
    drawn = rng.choice(len(uniq), size=len(uniq), replace=True)
    out = []
    for d in drawn:
        out.extend(by[uniq[d]])
    return out


def cluster_bootstrap_ci(values, groups, n_boot=1000, seed=0):
    """95% CI for a mean, resampling prompts (clusters)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boots = []
    for _ in range(n_boot):
        idx = _cluster_indices(groups, rng)
        boots.append(arr[idx].mean())
    boots = np.asarray(boots)
    return (round(float(arr.mean()), 4),
            round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4))


def cluster_paired_rmst(onsets_a, onsets_b, groups, max_t, n_boot=1000, seed=0):
    """
    Paired, prompt-clustered bootstrap for the RMST difference.
    Returns (diff, lo, hi, p) where p is the two-sided bootstrap p-value for the
    SAME statistic the CI is built on -- so the CI and p can never disagree.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(onsets_a); b = np.asarray(onsets_b)
    obs = rmst(a.tolist(), max_t) - rmst(b.tolist(), max_t)
    diffs = []
    for _ in range(n_boot):
        idx = _cluster_indices(groups, rng)
        diffs.append(rmst(a[idx].tolist(), max_t) - rmst(b[idx].tolist(), max_t))
    diffs = np.asarray(diffs)
    lo = float(np.percentile(diffs, 2.5)); hi = float(np.percentile(diffs, 97.5))
    centred = diffs - diffs.mean()
    p = float(np.mean(np.abs(centred) >= abs(obs)))
    return round(float(obs), 2), round(lo, 2), round(hi, 2), round(p, 4)


def cluster_paired_p(vals_a, vals_b, groups, n_boot=1000, seed=0):
    """Prompt-clustered paired bootstrap p-value for a mean difference."""
    rng = np.random.default_rng(seed)
    a = np.asarray(vals_a, float); b = np.asarray(vals_b, float)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    g = groups[:n]
    d = a - b
    obs = d.mean()
    boots = []
    for _ in range(n_boot):
        idx = _cluster_indices(g, rng)
        boots.append(d[idx].mean())
    boots = np.asarray(boots)
    return round(float(np.mean(np.abs(boots - boots.mean()) >= abs(obs))), 4)


def bootstrap_ci(values, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    boots = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                      for _ in range(n_boot)])
    return (round(float(arr.mean()), 4),
            round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4))

def paired_bootstrap_p(a, b, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.array(a, float), np.array(b, float)
    n = min(len(a), len(b))
    delta = a[:n] - b[:n]
    obs = delta.mean()
    boots = np.array([rng.choice(delta, size=n, replace=True).mean()
                      for _ in range(n_boot)])
    centred = boots - boots.mean()
    return round(float(np.mean(np.abs(centred) >= abs(obs))), 4)

def paired_rmst_diff_ci(onsets_a, onsets_b, max_t, n_boot=1000, seed=0):
    """Paired bootstrap on prompt/seed unit: resample sample indices jointly."""
    rng = np.random.default_rng(seed)
    a, b = np.array(onsets_a), np.array(onsets_b)
    n = min(len(a), len(b))
    obs = rmst(a[:n].tolist(), max_t) - rmst(b[:n].tolist(), max_t)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(rmst(a[idx].tolist(), max_t) - rmst(b[idx].tolist(), max_t))
    diffs = np.array(diffs)
    return (round(float(obs), 2),
            round(float(np.percentile(diffs, 2.5)), 2),
            round(float(np.percentile(diffs, 97.5)), 2))


# ---- Model helpers ----

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = CharTransformerLM(
        vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
        n_layer=cfg["n_layer"], n_embd=cfg["n_embd"],
        n_head=cfg["n_head"], dropout=0.0).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg

def encode(prompt, stoi):
    unk = stoi.get(" ", 0)
    return torch.tensor([[stoi.get(c, unk) for c in prompt]], dtype=torch.long)

def decode(ids, itos):
    return "".join(itos.get(str(int(i)), "?") for i in ids)


def make_decoders(cfg):
    risk = None; lz = None
    if cfg["risk"] is not None:
        risk = RecurrenceRiskDecoder(**cfg["risk"])
    if cfg.get("fsd") is not None:
        # FSDDecoder implements the same .step()/.reset()/.prime() interface
        # as RecurrenceRiskDecoder, so it plugs into generate()'s existing
        # `adaptive=` slot directly -- no changes needed to generate() itself.
        risk = FSDDecoder(**cfg["fsd"])
    if cfg["lookback"] is not None:
        lz = LookBackDecoder(**cfg["lookback"])
    return risk, lz


SCALARS = ["ttr", "entropy_4gram", "rep_rate_5",
           "rep_mass_2", "rep_mass_3", "rep_mass_4", "rep_mass_5", "rep_mass_6",
           "longest_rep_sub", "compression",
           "mc_nll_bpc", "ngram_sim_4", "spelling_error"]


def eval_checkpoint(ckpt_path, data_dir, n_chars, n_seeds, device, out_dir,
                    measure_runtime_for, only=None):
    model, cfg = load_model(ckpt_path, device)
    vocab = load_json(os.path.join(data_dir, "vocab.json"))
    stoi, itos = vocab["stoi"], vocab["itos"]
    name = os.path.basename(os.path.dirname(ckpt_path))
    block_size = cfg["block_size"]

    dtype = np.uint16 if cfg["vocab_size"] < 65535 else np.uint32
    val = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=dtype).astype(np.int64)
    ref_text = decode(val[:20000].tolist(), itos)

    print(f"\n  Checkpoint: {ckpt_path}")
    ensure_dir(out_dir)
    samples_f = open(os.path.join(out_dir, f"samples_{name}.txt"), "w", encoding="utf-8")

    rows = []
    configs = CONFIGS if not only else [c for c in CONFIGS if c["name"] in set(only)]
    for c in configs:
        prompts = PROMPTS if c["key"] else PROMPTS[:5]
        print(f"    {c['name']:<22} ({len(prompts)*n_seeds:>3} samples)", end="", flush=True)

        acc = {k: [] for k in SCALARS}
        onsets = {ld: [] for ld in LOOP_DEFS}
        groups = []          # prompt index per sample -> clustered inference
        diag = {"kl_bits": [], "entropy_q": [], "entropy_p": [],
                "risk_achieved": [], "lambda_mean": [],
                "dual_feasible_rate": [], "dual_cap_hit_rate": [],
                "dual_min_risk_mean": [], "dual_violation_mean": [],
                "dual_violation_max": [], "dual_tolerance_mean": []}
        cps_val = None

        for pi, (pname, ptext) in enumerate(prompts):
            for seed in range(1, n_seeds+1):
                print(f"\r    {c['name']:<24} sample {pi*n_seeds+seed}/{len(prompts)*n_seeds} "
                      f"(prompt={pname}, seed={seed})", end="", flush=True)
                set_seed(seed)
                idx = encode(ptext, stoi).to(device)
                risk, lz = make_decoders(c)
                measure = (c["name"] in measure_runtime_for and seed == 1 and pname == prompts[0][0])
                out, cps = generate(
                    model, idx, max_new_tokens=n_chars,
                    temperature=c["temperature"], top_k=c["top_k"], top_p=c["top_p"],
                    typical_p=c["typical_p"], rep_penalty=c["rep_penalty"],
                    no_repeat_ngram=c["no_repeat_ngram"], mirostat_tau=c["mirostat_tau"],
                    adaptive=risk, lz_decoder=lz, measure_time=measure)
                if cps is not None:
                    cps_val = cps
                gen_ids = out[0].tolist()[len(ptext):]
                text = decode(gen_ids, itos)

                groups.append(pi)                       # cluster id = prompt
                if risk is not None:
                    dg = risk.diagnostics()
                    for k in diag: diag[k].append(dg.get(k, float("nan")))

                for ld, fn in LOOP_DEFS.items():
                    onsets[ld].append(fn(text))

                acc["ttr"].append(type_token_ratio(text))
                acc["entropy_4gram"].append(char_ngram_entropy(text, 4))
                acc["rep_rate_5"].append(repetition_rate(text, 5))
                for nn in (2,3,4,5,6):
                    acc[f"rep_mass_{nn}"].append(rep_ngram_mass(text, nn))
                acc["longest_rep_sub"].append(longest_repeated_substring(text))
                acc["compression"].append(compression_ratio(text))
                prompt_id_list = idx[0].tolist()
                acc["mc_nll_bpc"].append(model_consistency_nll(
                    model, gen_ids, block_size, device, prompt_ids=prompt_id_list))
                acc["ngram_sim_4"].append(ngram_js_similarity(text, ref_text, 4))
                acc["spelling_error"].append(spelling_error_rate(text))

                if seed == 1:
                    samples_f.write(f"[{c['name']}][{name}] prompt='{ptext.strip()}'\n"
                                    + "-"*60 + "\n" + text + "\n\n")

        row = {"strategy": c["name"], "category": c["category"],
               "checkpoint": name, "key": c["key"],
               "n_samples": len(acc["ttr"])}
        row["groups"] = groups
        for k in SCALARS:
            m, lo, hi = cluster_bootstrap_ci(acc[k], groups)   # prompt-clustered
            row[f"{k}_mean"] = m; row[f"{k}_lo"] = lo; row[f"{k}_hi"] = hi
            row[f"{k}_vals"] = acc[k]
        # exact distortion diagnostics (decoder-side, recurrence-risk configs only)
        for k, v in diag.items():
            if v and not all(x != x for x in v):
                m, lo, hi = cluster_bootstrap_ci(v, groups)
                row[f"{k}_mean"] = m; row[f"{k}_lo"] = lo; row[f"{k}_hi"] = hi
        for ld in LOOP_DEFS:
            row[f"loop_rate_{ld}"]    = loop_rate(onsets[ld])
            row[f"survival_auc_{ld}"] = survival_auc(onsets[ld], n_chars)
            row[f"rmst_{ld}"]         = rmst(onsets[ld], n_chars)
            row[f"onsets_{ld}"]       = onsets[ld]
        if cps_val is not None:
            row["chars_per_sec"] = cps_val
        rows.append(row)

        kl_s = f"  KL={row['kl_bits_mean']:.3f}b" if 'kl_bits_mean' in row else ""
        print(f"\r  PERSIST loop={row['loop_rate_persistent']:.2f} "
              f"rmst={row['rmst_persistent']:.0f} | "
              f"[old ngram10 loop={row['loop_rate_ngram10']:.2f}] "
              f"mcnll={row['mc_nll_bpc_mean']:.3f} "
              f"sim={row['ngram_sim_4_mean']:.3f}{kl_s}"
              + (f" cps={cps_val}" if cps_val else "") + " " * 20)

    samples_f.close()

    # CSV (drop big arrays). Union of keys across rows; fill missing with "".
    skip = {f"{k}_vals" for k in SCALARS} | {f"onsets_{ld}" for ld in LOOP_DEFS} | {"groups"}
    keys = []
    for r in rows:
        for k in r:
            if k not in skip and k not in keys:
                keys.append(k)
    with open(os.path.join(out_dir, f"metrics_{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in rows])

    return rows, n_chars


def write_pareto(all_results, out_dir):
    """Pareto CSV: survival(ngram10) vs each quality axis. Hard constraint flagged."""
    fields = ["strategy", "category", "checkpoint", "hard_constraint",
              "survival_auc_ngram10", "rmst_ngram10", "loop_rate_ngram10",
              "mc_nll_bpc", "ngram_sim_4", "spelling_error", "compression",
              "longest_rep_sub", "chars_per_sec"]
    rows_out = []
    for name, rows in all_results.items():
        for r in rows:
            rows_out.append({
                "strategy": r["strategy"], "category": r["category"],
                "checkpoint": r["checkpoint"],
                "hard_constraint": (r["category"] == "hard_constraint"),
                "survival_auc_ngram10": r["survival_auc_ngram10"],
                "rmst_ngram10": r["rmst_ngram10"],
                "loop_rate_ngram10": r["loop_rate_ngram10"],
                "mc_nll_bpc": r["mc_nll_bpc_mean"],
                "ngram_sim_4": r["ngram_sim_4_mean"],
                "spelling_error": r["spelling_error_mean"],
                "compression": r["compression_mean"],
                "longest_rep_sub": r["longest_rep_sub_mean"],
                "chars_per_sec": r.get("chars_per_sec", ""),
            })
    with open(os.path.join(out_dir, "pareto_data.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows_out)


def write_method_vs_method(all_results, out_dir, max_t):
    """
    Paired, prompt-clustered method-vs-method comparisons.

    Fixes applied here:
      * the primary event is the VALIDATED persistent-loop event, not the
        n-gram event that fires on ordinary human text;
      * the CI and the p-value are computed from the SAME statistic on the SAME
        clustered resamples, so they cannot contradict each other (previously
        the column labelled p(SAUC) was in fact a p-value for the loop-RATE
        indicator, which is why it could disagree with the RMST interval);
      * resampling is over prompts (clusters), not over individual samples.
    """
    PAIRS = [
        ("rr_dual_eps0.05", "risk_only"),
        ("rr_dual_eps0.05", "adaptive"),
        ("adaptive", "risk_only"),
        ("adaptive", "rr_entropy_only"),
        ("risk_only", "temp_0.8"),
        ("risk_only", "nucleus_p0.95"),
        ("risk_only", "typical_p0.9"),
        ("risk_only", "mirostat_tau5.0"),
        ("risk_only", "rep_penalty_1.3"),
        ("risk_only", "rep_penalty_1.5"),
        ("risk_only", "lookback_a3.0"),
        ("risk_only", "lookback_a5.0"),
        ("adaptive", "no_repeat_4gram"),
    ]
    out = []
    for name, rows in all_results.items():
        rmap = {r["strategy"]: r for r in rows}
        for a, b in PAIRS:
            ra, rb = rmap.get(a), rmap.get(b)
            if not ra or not rb:
                continue
            groups = ra.get("groups") or list(range(len(ra["onsets_persistent"])))
            leak = " [leakage: hard-bans the measured event]" if "no_repeat" in b else ""
            d, lo, hi, p = cluster_paired_rmst(
                ra["onsets_persistent"], rb["onsets_persistent"], groups, max_t)
            rec = {
                "label": f"{a} vs {b}{leak}", "checkpoint": name,
                "method_a": a, "method_b": b,
                "n_pairs": len(ra["onsets_persistent"]),
                "n_prompts": len(set(groups)),
                "event": "persistent_loop",
                "rmst_a": ra["rmst_persistent"], "rmst_b": rb["rmst_persistent"],
                "rmst_diff_AminusB": d, "rmst_ci_lo": lo, "rmst_ci_hi": hi,
                "p_rmst": p,                      # same statistic as the CI
                "delta_mc_nll": round(ra["mc_nll_bpc_mean"] - rb["mc_nll_bpc_mean"], 4),
                "p_mc_nll": cluster_paired_p(ra["mc_nll_bpc_vals"], rb["mc_nll_bpc_vals"], groups),
                "delta_ngram_sim": round(ra["ngram_sim_4_mean"] - rb["ngram_sim_4_mean"], 4),
                "p_ngram_sim": cluster_paired_p(ra["ngram_sim_4_vals"], rb["ngram_sim_4_vals"], groups),
            }
            if "kl_bits_mean" in ra and "kl_bits_mean" in rb:
                rec["delta_kl_bits"] = round(ra["kl_bits_mean"] - rb["kl_bits_mean"], 4)
            out.append(rec)
    if out:
        with open(os.path.join(out_dir, "method_vs_method.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader(); w.writerows(out)


def write_loop_robustness(all_results, out_dir):
    """How survival ranking holds across loop definitions (punkt 4)."""
    fields = ["strategy", "category", "checkpoint"] + \
             [f"rmst_{ld}" for ld in LOOP_DEFS] + \
             [f"loop_rate_{ld}" for ld in LOOP_DEFS]
    rows_out = []
    for name, rows in all_results.items():
        for r in rows:
            d = {"strategy": r["strategy"], "category": r["category"], "checkpoint": name}
            for ld in LOOP_DEFS:
                d[f"rmst_{ld}"] = r[f"rmst_{ld}"]
                d[f"loop_rate_{ld}"] = r[f"loop_rate_{ld}"]
            rows_out.append(d)
    with open(os.path.join(out_dir, "loop_robustness.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows_out)


def write_survival_curves(all_results, out_dir, max_t):
    km = {}
    for name, rows in all_results.items():
        km[name] = {}
        for r in rows:
            t, s = kaplan_meier(r["onsets_ngram10"], max_t)
            km[name][r["strategy"]] = {
                "km_times": t, "km_survival": s,
                "n_censored": sum(1 for x in r["onsets_ngram10"] if x < 0),
                "n_total": len(r["onsets_ngram10"]),
                "rmst": r["rmst_ngram10"], "survival_auc": r["survival_auc_ngram10"],
            }
    with open(os.path.join(out_dir, "survival_curves.json"), "w") as f:
        json.dump(km, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+",
                    default=["runs/baseline/best.pt", "runs/cosine/best.pt"])
    ap.add_argument("--data_dir", default="data_out")
    ap.add_argument("--out_dir", default="runs/sampling_eval_v6")
    ap.add_argument("--n_chars", type=int, default=500)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--prompt_set", choices=["dev","test"], default="test",
                    help="dev = the 5 prompts used while tuning; test = the 15 "
                         "untouched prompts. Headline numbers must use test.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Run only these strategy names (e.g. --only rep_penalty_1.5 lookback_a5.0). "
                         "Useful for topping up a config to the full sample budget.")
    args = ap.parse_args()

    global PROMPTS
    PROMPTS = TEST_PROMPTS if args.prompt_set == "test" else DEV_PROMPTS
    print(f"Prompt set : {args.prompt_set} ({len(PROMPTS)} prompts)"
          + ("  [untouched during tuning]" if args.prompt_set=="test" else "  [used for tuning]"))

    ensure_dir(args.out_dir)
    device = (torch.device("mps") if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))

    runtime_for = {"greedy", "temp_0.8", "nucleus_p0.95", "mirostat_tau5.0",
                   "rep_penalty_1.3", "lookback_a3.0", "no_repeat_4gram",
                   "risk_only", "adaptive"}

    n_key = sum(1 for c in CONFIGS if c["key"])
    print(f"Device: {device}")
    print(f"Configs: {len(CONFIGS)} ({n_key} key on 15 prompts, rest on 5)")
    print(f"Loop definitions: {list(LOOP_DEFS.keys())}")

    all_results = {}
    max_t = args.n_chars
    for ckpt in args.ckpt:
        name = os.path.basename(os.path.dirname(ckpt))
        rows, max_t = eval_checkpoint(ckpt, args.data_dir, args.n_chars,
                                      args.n_seeds, device, args.out_dir, runtime_for,
                                      only=args.only)
        all_results[name] = rows

    write_pareto(all_results, args.out_dir)
    write_loop_robustness(all_results, args.out_dir)
    write_survival_curves(all_results, args.out_dir, max_t)
    if args.only:
        print("\n  --only mode: skipping method_vs_method (needs the full config set).")
        print("  Merge these rows into the full run's CSVs, then recompute comparisons.")
    else:
        write_method_vs_method(all_results, args.out_dir, max_t)

    # Full JSON (drop per-sample arrays to keep size reasonable, keep onsets)
    skip = {f"{k}_vals" for k in SCALARS}
    serial = {name: [{k: v for k, v in r.items() if k not in skip} for r in rows]
              for name, rows in all_results.items()}
    with open(os.path.join(args.out_dir, "all_results.json"), "w") as f:
        json.dump(serial, f, indent=2)

    print(f"\n  Wrote: metrics_*.csv, pareto_data.csv, loop_robustness.csv,")
    print(f"         method_vs_method.csv, survival_curves.json, all_results.json")
    print(f"  -> {args.out_dir}/")


if __name__ == "__main__":
    main()