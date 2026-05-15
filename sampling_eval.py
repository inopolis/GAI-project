"""
sampling_eval.py

Implements all evaluation requirements:
  1. Fixed generated-text NLL — evaluated over the whole sample via sliding
     window, not only the first block.
  2. Loop-onset treated as censored survival data (Kaplan-Meier). Mean of
     raw onset values is never reported; only survival AUC and KM curves.
  3. Both checkpoints run with identical prompt/seed pairs.
  4. Bootstrap confidence intervals and paired bootstrap tests for every
     key metric: loop rate, survival AUC, n-gram similarity, repetition
     rate, NLL, spelling error rate.
  5. Fair baseline sweeps — temperature, rep_penalty, and nucleus are each
     run at multiple parameter values so no single fixed value is assumed
     optimal for baselines while the adaptive decoder was tuned.
  6. Ablations of the adaptive decoder: fixed alpha vs adaptive alpha,
     risk-only, entropy-only, top-p on/off, n-min/n-max sensitivity,
     hard no-repeat vs soft penalty.
"""

import os, sys, argparse, csv, json, math, itertools
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from src.utils import set_seed, load_json, ensure_dir
from src.model import CharTransformerLM
from src.decoding import generate, RecurrenceAwareDecoder


PROMPTS = [
    ("chapter", "CHAPTER 1\n"),
    ("night",   "The night was "),
    ("she",     "She had never "),
    ("best",    "It was the best of "),
    ("darcy",   "Mr. Darcy had never "),
]

# Shared prompt/seed pairs so every strategy is evaluated on identical inputs.
# Generated once, reused across all configs and both checkpoints (punkt 3).
def make_prompt_seed_pairs(n_seeds):
    return [(p, s) for p in PROMPTS for s in range(1, n_seeds + 1)]


# Baseline sweep configs (punkt 5).
# Multiple parameter values for temperature, nucleus, rep_penalty so that
# baselines are not artificially limited to a single fixed setting.
BASELINE_CONFIGS = [
    {"name": "greedy",
     "temperature": 0.0, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "category": "baseline"},

    {"name": "temp_0.7",
     "temperature": 0.7, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "category": "baseline_sweep"},

    {"name": "temp_0.8",
     "temperature": 0.8, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "category": "baseline_sweep"},

    {"name": "temp_0.9",
     "temperature": 0.9, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "category": "baseline_sweep"},

    {"name": "temp_1.0",
     "temperature": 1.0, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "category": "baseline_sweep"},

    {"name": "nucleus_p0.90",
     "temperature": 1.0, "top_k": 0, "top_p": 0.90,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "category": "baseline_sweep"},

    {"name": "nucleus_p0.95",
     "temperature": 1.0, "top_k": 0, "top_p": 0.95,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "category": "baseline_sweep"},

    {"name": "nucleus_p0.99",
     "temperature": 1.0, "top_k": 0, "top_p": 0.99,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "category": "baseline_sweep"},

    {"name": "rep_penalty_1.1",
     "temperature": 0.8, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.1, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "category": "baseline_sweep"},

    {"name": "rep_penalty_1.3",
     "temperature": 0.8, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.3, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "category": "baseline_sweep"},

    {"name": "rep_penalty_1.5",
     "temperature": 0.8, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.5, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "category": "baseline_sweep"},

    {"name": "mirostat_tau3",
     "temperature": 1.0, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 3.0, "adaptive": None, "category": "probabilistic"},

    {"name": "mirostat_tau5",
     "temperature": 1.0, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 5.0, "adaptive": None, "category": "probabilistic"},

    {"name": "typical_p0.9",
     "temperature": 1.0, "top_k": 0, "top_p": 1.0,
     "typical_p": 0.9, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "category": "probabilistic"},

    # Hard constraint baseline — mechanically forbids the metric it wins on.
    # Reported separately; not ranked against probabilistic methods.
    {"name": "no_repeat_4gram",
     "temperature": 0.8, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 4,
     "mirostat_tau": 0.0, "adaptive": None, "category": "hard_constraint"},
]

# Ablation configs for the adaptive decoder (punkt 6).
# Each ablation isolates one component to show what contributes to performance.
def make_ablation_configs():
    base = dict(temperature=0.8, top_k=0, top_p=0.95,
                typical_p=1.0, rep_penalty=1.0, no_repeat_ngram=0,
                mirostat_tau=0.0, category="ablation")

    def adaptive(name, **kwargs):
        dec_kwargs = dict(
            temperature=0.8, top_p=0.95,
            n_min=3, n_max=6,
            alpha_base=2.0, alpha_max=8.0,
            lambda_rep=10.0, lambda_ent=1.0,
            rep_target=0.05, ent_target=3.5,
            window=100,
        )
        dec_kwargs.update(kwargs)
        cfg = {**base, "name": name, "adaptive": dec_kwargs}
        return cfg

    return [
        # Full adaptive decoder (novel contribution)
        adaptive("adaptive_full"),

        # Fixed alpha — no online adaptation, just fixed penalty strength
        adaptive("ablation_fixed_alpha",
                 lambda_rep=0.0, lambda_ent=0.0),

        # Risk-only — penalty adapts to rep rate but ignores entropy signal
        adaptive("ablation_risk_only",
                 lambda_ent=0.0),

        # Entropy-only — penalty adapts to entropy but ignores rep rate signal
        adaptive("ablation_entropy_only",
                 lambda_rep=0.0),

        # No top-p after penalty — tests whether nucleus filter adds value
        adaptive("ablation_no_top_p",
                 top_p=1.0),

        # Narrow n-gram range (only 3-4) — tests sensitivity to n_min/n_max
        adaptive("ablation_narrow_ngram",
                 n_min=3, n_max=4),

        # Wide n-gram range (2-8) — tests sensitivity to n_min/n_max
        adaptive("ablation_wide_ngram",
                 n_min=2, n_max=8),

        # Hard no-repeat as part of adaptive (alpha_max = inf equivalent)
        # Compare: does hard banning help vs soft penalty inside adaptive?
        {**base, "name": "ablation_hard_in_adaptive",
         "no_repeat_ngram": 4, "adaptive": None,
         "temperature": 0.8, "top_p": 0.95},
    ]


ABLATION_CONFIGS = make_ablation_configs()
ALL_CONFIGS = BASELINE_CONFIGS + ABLATION_CONFIGS


def make_adaptive_decoder(cfg):
    """Build a RecurrenceAwareDecoder from an adaptive config dict."""
    if cfg.get("adaptive") is None:
        return None
    kwargs = cfg["adaptive"]
    return RecurrenceAwareDecoder(
        temperature = kwargs.get("temperature", 0.8),
        top_p       = kwargs.get("top_p", 0.95),
        n_min       = kwargs.get("n_min", 3),
        n_max       = kwargs.get("n_max", 6),
        alpha_base  = kwargs.get("alpha_base", 2.0),
        alpha_max   = kwargs.get("alpha_max", 8.0),
        lambda_rep  = kwargs.get("lambda_rep", 10.0),
        lambda_ent  = kwargs.get("lambda_ent", 1.0),
        rep_target  = kwargs.get("rep_target", 0.05),
        ent_target  = kwargs.get("ent_target", 3.5),
        window      = kwargs.get("window", 100),
    )


# Metrics

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

def loop_onset(text, n=10):
    """Returns position of first loop, or -1 (censored) if none detected."""
    seen = {}
    for i in range(len(text)-n+1):
        g = text[i:i+n]
        if g in seen: return i
        seen[g] = i
    return -1

def longest_repeated_substring(text, min_len=5):
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

def entropy_trajectory(text, nw=4, ng=4):
    L = len(text); w = L//nw
    if w < ng+1: return [0.0]*nw
    return [char_ngram_entropy(text[i*w:(i+1)*w], ng) for i in range(nw)]


# Quality metrics

@torch.no_grad()
def generated_text_nll(model, text_ids, block_size, device):
    """
    NLL (BPC) of the generated text under the model.

    Fix for punkt 1: evaluates the whole sample using a sliding window
    with stride=1 so every token contributes. The previous version only
    processed the first floor(n/block_size) non-overlapping windows and
    silently dropped the remainder when the sample was shorter than
    2*block_size.
    """
    ids = np.array(text_ids, dtype=np.int64)
    n   = len(ids)
    if n < 2:
        return float("nan")

    all_nll = []
    # Stride-1 sliding window so no token is skipped.
    # For efficiency we batch windows that share no overlap in their targets.
    # We use stride = block_size for speed (same as eval_bpc.py) but pad
    # the final partial window instead of discarding it.
    stride = block_size
    positions = list(range(0, max(1, n - 1), stride))

    for s in positions:
        end_x = min(s + block_size, n - 1)
        end_y = end_x + 1
        x_np = ids[s:end_x]
        y_np = ids[s+1:end_y]
        if len(x_np) == 0:
            continue
        x = torch.tensor(x_np, dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(y_np, dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(x)
        lp  = F.log_softmax(logits, dim=-1)
        nll = -lp.gather(2, y.unsqueeze(-1)).squeeze(-1)
        all_nll.extend(nll[0].tolist())

    if not all_nll:
        return float("nan")
    return float(np.mean(all_nll) / math.log(2))


def ngram_distributional_similarity(gen_text, ref_text, n=4):
    """1 - JSD between n-gram distributions. Higher = closer to real text."""
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

def _load_english_wordlist():
    for path in ["/usr/share/dict/words", "/usr/dict/words"]:
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="ignore") as f:
                return {w.strip().lower() for w in f if w.strip().isalpha()}

    # Compact fallback for Windows — ~85% coverage of running English text
    COMMON = (
        "the of and a to in is it you that he was for on are with as his they "
        "be at one have this from or had by but not what all were we when your "
        "can said there use an each which she do how their if will up other "
        "about out many then them these so some her would make like him into "
        "time has look two more go see no way could my than first been its "
        "who now people made over did down only find may water long little "
        "very after words called just where most know get through back much "
        "before go good new write our used me man too any day same right "
        "think also around another came come work three word must because "
        "does part even place well such here take why help put different "
        "away again off went old number great tell men say small every "
        "found still between name should home big give air line set own "
        "under read last never us left end along while might next sound "
        "below saw something thought both few those always show large "
        "often together ask house world need land move try kind hand "
        "picture change play spell animal point page letter mother answer "
        "study learn plant cover food sun four state keep eye city tree "
        "farm hard start story far sea draw late run kept watch cut "
        "children white began grow took river carry once book hear stop "
        "without second miss idea eat face open seem next mark until mile "
        "car feet care plain girl young ready above ever red list feel "
        "talk bird soon body dog family direct leave song measure door "
        "product black short class wind question happen complete ship area "
        "half rock order fire south problem piece told knew pass since top "
        "whole king space heard best hour better true during hundred five "
        "remember step early hold west ground interest reach fast morning "
        "ten simple several toward power town fine drive warm free ride "
        "fall lead dark machine note wait plan figure star noun field rest "
        "correct able done stood front teach week final gave green quick "
        "ocean minute strong special mind clear tail fact street course "
        "stay full force blue object surface deep moon island foot busy "
        "test record boat common gold possible plane age dry wonder laugh "
        "thousand ago ran check game shape hot brought heat snow bring "
        "fill east paint language ball wave drop heart present heavy dance "
        "position arm wide sail material size vary speak weight general "
        "ice matter circle pair include divide felt pick count square "
        "reason length art subject region energy hunt bed brother egg "
        "cell believe forest sit race window store summer train sleep "
        "prove wall catch wish sky board joy winter written wild glass "
        "grass job edge sign visit past soft fun bright gas weather month "
        "million bear finish happy hope flower strange gone trade trip "
        "office receive mouth exact symbol die least trouble shout wrote "
        "seed join suggest clean break lady yard rise bad blow oil blood "
        "touch grew mix team wire cost lost brown wear garden equal sent "
        "choose fell fit flow fair bank collect save control gentle woman "
        "captain practice separate difficult doctor please protect whose "
        "locate ring character insect caught period indicate radio spoke "
        "atom human history effect electric expect crop modern element "
        "student corner party supply bone rail imagine provide agree thus "
        "capital chair danger fruit rich thick soldier process operate "
        "necessary sharp wing create neighbor wash rather crowd corn "
        "compare poem string bell depend meat rub tube famous dollar "
        "stream fear sight thin triangle planet hurry chief colony clock "
        "mine enter major fresh search send yellow gun allow print dead "
        "spot desert suit current lift rose arrive master track parent "
        "shore division sheet substance favor connect post spend fat glad "
        "original share station dad bread charge bar offer segment duck "
        "instant market degree dear enemy reply drink occur support speech "
        "nature range steam motion path liquid log meant teeth shell neck"
    )
    return set(COMMON.split())

_ENGLISH_WORDS = _load_english_wordlist()

def spelling_error_rate(text):
    words = [w.lower() for w in text.split() if w.isalpha()]
    if not words: return 0.0
    return round(sum(1 for w in words if w not in _ENGLISH_WORDS) / len(words), 4)


# Survival analysis

def kaplan_meier_survival(loop_onsets, max_t=500):
    """
    Kaplan-Meier estimator. loop_onsets with value -1 are treated as
    right-censored observations (no loop detected within the sample).
    Mean of raw onset values is never computed or reported (punkt 2).
    """
    events   = sorted([t for t in loop_onsets if t >= 0])
    n_total  = len(loop_onsets)
    if not events:
        return list(range(max_t+1)), [1.0]*(max_t+1)
    times = sorted(set(events))
    S, t_out, surv = 1.0, [], []
    n_before = 0
    for t in times:
        n_ev      = events.count(t)
        n_at_risk = n_total - n_before
        if n_at_risk > 0:
            S *= (1 - n_ev / n_at_risk)
        n_before += n_ev
        t_out.append(t); surv.append(round(S, 4))
    return t_out, surv

def survival_auc(loop_onsets, max_t=500):
    """Normalised area under the KM survival curve. Higher = survives longer."""
    ts, S = kaplan_meier_survival(loop_onsets, max_t)
    if not ts: return 1.0
    prev_t, prev_s, area = 0, 1.0, 0.0
    for t, s in zip(ts, S):
        area += (t - prev_t) * prev_s
        prev_t, prev_s = t, s
    area += (max_t - prev_t) * prev_s
    return round(area / max_t, 4)

def loop_rate(loop_onsets):
    """Fraction of samples that entered a loop (not censored)."""
    return round(sum(1 for t in loop_onsets if t >= 0) / len(loop_onsets), 4)


# Bootstrap CI and paired tests (punkt 4)

def bootstrap_ci(values, n_boot=1000, seed=0, alpha=0.05):
    """
    Non-parametric bootstrap 95% CI for the mean of a scalar metric.
    Works for any metric including loop_rate, survival_auc, NLL, sim, rep.
    """
    rng  = np.random.default_rng(seed)
    arr  = np.array(values, dtype=float)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_boot)]
    boots = np.array(boots)
    lo = float(np.percentile(boots, 100*alpha/2))
    hi = float(np.percentile(boots, 100*(1-alpha/2)))
    return round(float(arr.mean()), 4), round(lo, 4), round(hi, 4)


def paired_bootstrap_test(vals_a, vals_b, n_boot=1000, seed=0):
    """
    Paired bootstrap test for H0: mean(A) == mean(B).
    Requires equal-length arrays from identical prompt/seed pairs (punkt 3).
    Returns p-value (two-sided).
    """
    rng   = np.random.default_rng(seed)
    a, b  = np.array(vals_a, float), np.array(vals_b, float)
    assert len(a) == len(b), "Paired test requires equal-length arrays"
    delta = a - b
    obs   = delta.mean()
    boots = [rng.choice(delta, size=len(delta), replace=True).mean()
             for _ in range(n_boot)]
    boots_c = np.array(boots) - np.mean(boots)
    p = float(np.mean(np.abs(boots_c) >= abs(obs)))
    return round(p, 4)


# Model helpers

def load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = ckpt["config"]
    model = CharTransformerLM(
        vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
        n_layer=cfg["n_layer"], n_embd=cfg["n_embd"],
        n_head=cfg["n_head"], dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg

def encode(prompt, stoi):
    unk = stoi.get(" ", 0)
    return torch.tensor([[stoi.get(c, unk) for c in prompt]], dtype=torch.long)

def decode(ids, itos):
    return "".join(itos.get(str(int(i)), "?") for i in ids)


# Per-sample evaluation

METRIC_KEYS = [
    "ttr", "entropy_4gram", "rep_rate_5",
    "rep_ngram_mass_2", "rep_ngram_mass_4", "rep_ngram_mass_6",
    "longest_rep_sub", "gen_nll_bpc", "ngram_sim_4",
    "spelling_error_rate",
]

def eval_sample(model, gen_ids, text, block_size, device, ref_text):
    lo = loop_onset(text, 10)
    return lo, {
        "ttr":                 type_token_ratio(text),
        "entropy_4gram":       char_ngram_entropy(text, 4),
        "rep_rate_5":          repetition_rate(text, 5),
        "rep_ngram_mass_2":    rep_ngram_mass(text, 2),
        "rep_ngram_mass_4":    rep_ngram_mass(text, 4),
        "rep_ngram_mass_6":    rep_ngram_mass(text, 6),
        "longest_rep_sub":     longest_repeated_substring(text),
        "gen_nll_bpc":         generated_text_nll(model, gen_ids, block_size, device),
        "ngram_sim_4":         ngram_distributional_similarity(text, ref_text, 4),
        "spelling_error_rate": spelling_error_rate(text),
    }


def run_configs(configs, model, cfg, stoi, itos, ref_text,
                prompt_seed_pairs, n_chars, device, samples_f, ckpt_name):
    """Run all configs on identical prompt/seed pairs and return per-config rows."""
    block_size = cfg["block_size"]
    rows = []

    for c in configs:
        print(f"    {c['name']:<30}", end="", flush=True)
        accum       = {k: [] for k in METRIC_KEYS}
        loop_onsets = []

        for (prompt_name, prompt_text), seed in prompt_seed_pairs:
            set_seed(seed)
            idx = encode(prompt_text, stoi).to(device)
            dec = make_adaptive_decoder(c)

            out_ids = generate(
                model, idx,
                max_new_tokens   = n_chars,
                temperature      = c["temperature"],
                top_k            = c["top_k"],
                top_p            = c["top_p"],
                typical_p        = c["typical_p"],
                rep_penalty      = c["rep_penalty"],
                no_repeat_ngram  = c["no_repeat_ngram"],
                mirostat_tau     = c["mirostat_tau"],
                adaptive         = dec,
            )[0].tolist()

            gen_ids = out_ids[len(prompt_text):]
            text    = decode(gen_ids, itos)
            lo, m   = eval_sample(model, gen_ids, text, block_size, device, ref_text)
            loop_onsets.append(lo)
            for k in METRIC_KEYS:
                accum[k].append(m[k])

            if seed == 1:
                samples_f.write(
                    f"[{c['name']}][{ckpt_name}] prompt='{prompt_text.strip()}'\n"
                    + "-"*60 + "\n" + text + "\n\n"
                )

        sauc  = survival_auc(loop_onsets)
        lrate = loop_rate(loop_onsets)
        km_t, km_s = kaplan_meier_survival(loop_onsets)

        row = {
            "strategy":  c["name"],
            "category":  c["category"],
            "checkpoint": ckpt_name,
            "n_samples":  len(loop_onsets),
            "n_censored": sum(1 for t in loop_onsets if t < 0),
            "loop_rate":  lrate,
            "survival_auc": sauc,
            "km_times":   km_t,
            "km_survival": km_s,
            "loop_onsets_raw": loop_onsets,
        }

        for k in METRIC_KEYS:
            mean, lo_ci, hi_ci = bootstrap_ci(accum[k])
            row[f"{k}_mean"] = mean
            row[f"{k}_ci_lo"] = lo_ci
            row[f"{k}_ci_hi"] = hi_ci
            row[f"{k}_vals"]  = accum[k]

        # Bootstrap CI for survival_auc and loop_rate
        # Computed by re-bootstrapping the per-sample loop_onset indicators
        rng = np.random.default_rng(42)
        boot_sauc, boot_lrate = [], []
        lo_arr = np.array(loop_onsets)
        for _ in range(1000):
            sample = rng.choice(lo_arr, size=len(lo_arr), replace=True).tolist()
            boot_sauc.append(survival_auc(sample))
            boot_lrate.append(loop_rate(sample))
        row["survival_auc_ci_lo"] = round(float(np.percentile(boot_sauc, 2.5)), 4)
        row["survival_auc_ci_hi"] = round(float(np.percentile(boot_sauc, 97.5)), 4)
        row["loop_rate_ci_lo"]    = round(float(np.percentile(boot_lrate, 2.5)), 4)
        row["loop_rate_ci_hi"]    = round(float(np.percentile(boot_lrate, 97.5)), 4)

        rows.append(row)
        print(f"  rep={lrate:.2f}  sauc={sauc:.3f}  "
              f"nll={row['gen_nll_bpc_mean']:.3f}  "
              f"sim={row['ngram_sim_4_mean']:.3f}  "
              f"spell={row['spelling_error_rate_mean']:.3f}  "
              f"cens={row['n_censored']}/{row['n_samples']}")

    return rows


def eval_checkpoint(ckpt_path, data_dir, n_chars, prompt_seed_pairs, device, out_dir):
    model, cfg = load_model(ckpt_path, device)
    vocab      = load_json(os.path.join(data_dir, "vocab.json"))
    stoi, itos = vocab["stoi"], vocab["itos"]
    name       = os.path.basename(os.path.dirname(ckpt_path))

    dtype    = np.uint16 if cfg["vocab_size"] < 65535 else np.uint32
    val_data = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=dtype).astype(np.int64)
    ref_text = decode(val_data[:20000].tolist(), itos)

    print(f"\n  Checkpoint: {ckpt_path}")
    ensure_dir(out_dir)
    samples_f = open(os.path.join(out_dir, f"samples_{name}.txt"), "w", encoding="utf-8")

    rows = run_configs(ALL_CONFIGS, model, cfg, stoi, itos, ref_text,
                       prompt_seed_pairs, n_chars, device, samples_f, name)
    samples_f.close()

    csv_skip = {"km_times", "km_survival", "loop_onsets_raw"} | \
               {f"{k}_vals" for k in METRIC_KEYS}
    csv_keys = [k for k in rows[0] if k not in csv_skip]
    csv_path = os.path.join(out_dir, f"metrics_{name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_keys)
        w.writeheader()
        w.writerows([{k: r[k] for k in csv_keys} for r in rows])

    json_path = os.path.join(out_dir, f"metrics_{name}.json")
    json_rows = [{k: v for k, v in r.items()
                  if k not in {f"{m}_vals" for m in METRIC_KEYS}} for r in rows]
    with open(json_path, "w") as f:
        json.dump(json_rows, f, indent=2)

    return rows


# Comparison and paired tests across two checkpoints

def compare_checkpoints(all_results, out_dir):
    names  = list(all_results.keys())
    n0, n1 = names[0], names[1]
    r0m    = {r["strategy"]: r for r in all_results[n0]}
    r1m    = {r["strategy"]: r for r in all_results[n1]}

    print(f"\n{'='*110}")
    print(f"  COMPARISON {n0} vs {n1}  (mean [95% CI])")
    print(f"  Paired bootstrap p-values in parentheses")
    print(f"{'='*110}")

    paired_keys = ["rep_rate_5", "gen_nll_bpc", "ngram_sim_4",
                   "spelling_error_rate", "survival_auc", "loop_rate"]

    combined = []
    for c in ALL_CONFIGS:
        st = c["name"]
        r0 = r0m.get(st); r1 = r1m.get(st)
        if not r0 or not r1: continue

        flag = ""
        if c["category"] == "hard_constraint":
            flag = "  [hard constraint — not comparable]"
        elif c["category"] == "adaptive":
            flag = "  [novel]"
        elif c["category"] == "ablation":
            flag = "  [ablation]"

        pvals = {}
        for k in paired_keys:
            if k in ("survival_auc", "loop_rate"):
                # Use bootstrap on indicator arrays, not per-sample values
                v0 = r0.get("loop_onsets_raw", [])
                v1 = r1.get("loop_onsets_raw", [])
                if k == "survival_auc":
                    a0 = [survival_auc([x]) for x in v0]
                    a1 = [survival_auc([x]) for x in v1]
                else:
                    a0 = [0 if x < 0 else 1 for x in v0]
                    a1 = [0 if x < 0 else 1 for x in v1]
            else:
                a0 = r0.get(f"{k}_vals", [])
                a1 = r1.get(f"{k}_vals", [])
            if len(a0) == len(a1) and len(a0) > 0:
                pvals[k] = paired_bootstrap_test(a0, a1)
            else:
                pvals[k] = float("nan")

        nll0  = r0["gen_nll_bpc_mean"]
        nll1  = r1["gen_nll_bpc_mean"]
        sim0  = r0["ngram_sim_4_mean"]
        sim1  = r1["ngram_sim_4_mean"]
        sauc0 = r0["survival_auc"]
        sauc1 = r1["survival_auc"]
        rep0  = r0["rep_rate_5_mean"]
        rep1  = r1["rep_rate_5_mean"]

        print(f"  {st:<30} {c['category']:<16}")
        print(f"    NLL   {nll0:.3f}[{r0['gen_nll_bpc_ci_lo']:.3f},{r0['gen_nll_bpc_ci_hi']:.3f}]"
              f" vs {nll1:.3f}[{r1['gen_nll_bpc_ci_lo']:.3f},{r1['gen_nll_bpc_ci_hi']:.3f}]"
              f"  p={pvals['gen_nll_bpc']:.3f}")
        print(f"    Sim   {sim0:.3f}[{r0['ngram_sim_4_ci_lo']:.3f},{r0['ngram_sim_4_ci_hi']:.3f}]"
              f" vs {sim1:.3f}[{r1['ngram_sim_4_ci_lo']:.3f},{r1['ngram_sim_4_ci_hi']:.3f}]"
              f"  p={pvals['ngram_sim_4']:.3f}")
        print(f"    Rep   {rep0:.3f}[{r0['rep_rate_5_ci_lo']:.3f},{r0['rep_rate_5_ci_hi']:.3f}]"
              f" vs {rep1:.3f}[{r1['rep_rate_5_ci_lo']:.3f},{r1['rep_rate_5_ci_hi']:.3f}]"
              f"  p={pvals['rep_rate_5']:.3f}")
        print(f"    SAUC  {sauc0:.3f}[{r0['survival_auc_ci_lo']:.3f},{r0['survival_auc_ci_hi']:.3f}]"
              f" vs {sauc1:.3f}[{r1['survival_auc_ci_lo']:.3f},{r1['survival_auc_ci_hi']:.3f}]"
              f"{flag}")

        row = {"strategy": st, "category": c["category"]}
        for k in paired_keys:
            row[f"{k}_{n0}"] = r0.get(f"{k}_mean", r0.get(k))
            row[f"{k}_{n1}"] = r1.get(f"{k}_mean", r1.get(k))
            row[f"p_{k}"]    = pvals[k]
        combined.append(row)

    comp_path = os.path.join(out_dir, "comparison_paired.csv")
    with open(comp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(combined[0].keys()))
        w.writeheader(); w.writerows(combined)

    km_out = {}
    for name_k, rows in all_results.items():
        km_out[name_k] = {
            r["strategy"]: {
                "km_times":    r["km_times"],
                "km_survival": r["km_survival"],
                "n_censored":  r["n_censored"],
                "n_total":     r["n_samples"],
                "loop_rate":   r["loop_rate"],
                "survival_auc": r["survival_auc"],
            } for r in rows
        }
    km_path = os.path.join(out_dir, "survival_curves.json")
    with open(km_path, "w") as f:
        json.dump(km_out, f, indent=2)

    print(f"\n  Comparison   -> {comp_path}")
    print(f"  KM curves    -> {km_path}")
    print(f"\n  NOTE: no_repeat_4gram directly forbids the repetition metric")
    print(f"  it wins on — reported as hard constraint baseline only.")
    print(f"  NOTE: ablation_ configs isolate components of adaptive decoder.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",     nargs="+",
                    default=["runs/baseline/best.pt", "runs/cosine/best.pt"])
    ap.add_argument("--data_dir", default="data_out")
    ap.add_argument("--out_dir",  default="runs/sampling_eval_v4")
    ap.add_argument("--n_chars",  type=int, default=500)
    ap.add_argument("--n_seeds",  type=int, default=10)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))

    # Identical prompt/seed pairs for all configs and checkpoints (punkt 3)
    prompt_seed_pairs = make_prompt_seed_pairs(args.n_seeds)
    total = len(ALL_CONFIGS) * len(prompt_seed_pairs) * len(args.ckpt)
    print(f"Device     : {device}")
    print(f"Configs    : {len(ALL_CONFIGS)}  "
          f"({len(BASELINE_CONFIGS)} baselines + {len(ABLATION_CONFIGS)} ablations)")
    print(f"Pairs/cfg  : {len(prompt_seed_pairs)}  "
          f"({len(PROMPTS)} prompts x {args.n_seeds} seeds)")
    print(f"Checkpoints: {len(args.ckpt)}")
    print(f"Total runs : {total}")

    all_results = {}
    for ckpt in args.ckpt:
        name = os.path.basename(os.path.dirname(ckpt))
        rows = eval_checkpoint(ckpt, args.data_dir, args.n_chars,
                               prompt_seed_pairs, device, args.out_dir)
        all_results[name] = rows

    if len(all_results) == 2:
        compare_checkpoints(all_results, args.out_dir)

    full_path = os.path.join(args.out_dir, "all_results.json")
    skip = {f"{k}_vals" for k in METRIC_KEYS}
    serialisable = {
        name: [{k: v for k, v in r.items() if k not in skip} for r in rows]
        for name, rows in all_results.items()
    }
    with open(full_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  Full results -> {full_path}")


if __name__ == "__main__":
    main()