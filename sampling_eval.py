"""
sampling_eval.py

Evaluation requirements addressed:
  1. Method-vs-method paired comparisons: adaptive_full vs risk_only,
     adaptive_full vs entropy_only, risk_only vs best baselines,
     soft methods vs hard no-repeat.
  2. Survival properly: RMST and survival-AUC differences with CIs,
     using prompt/seed as the paired unit.
  3. Key methods run on extended prompt set (15 prompts) for higher power.
  4. LZRepetitionDecoder added as stronger close baseline.
  5. Metric-leakage checks: longest repeated substring, compression ratio
     (zlib), repeated suffix length, rep_ngram_mass at n=2,3,4,5,6.
  6. Pareto data saved for survival vs NLL, survival vs sim, survival vs
     spelling. Hard no-repeat flagged separately.
  7. Matched qualitative examples: same prompt+seed through all key methods.
  8. Runtime overhead measured in chars/sec per strategy.
"""

import os, sys, argparse, csv, json, math, zlib
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from src.utils import set_seed, load_json, ensure_dir
from src.model import CharTransformerLM
from src.decoding import (generate, RecurrenceAwareDecoder, LZRepetitionDecoder)


PROMPTS_STANDARD = [
    ("chapter", "CHAPTER 1\n"),
    ("night",   "The night was "),
    ("she",     "She had never "),
    ("best",    "It was the best of "),
    ("darcy",   "Mr. Darcy had never "),
]

PROMPTS_EXTENDED = PROMPTS_STANDARD + [
    ("he",      "He looked at "),
    ("it",      "It was a dark "),
    ("from",    "From the moment "),
    ("years",   "Many years ago "),
    ("said",    "She said nothing "),
    ("morning", "The morning light "),
    ("door",    "The door opened "),
    ("london",  "In the streets of London "),
    ("silence", "A long silence "),
    ("truth",   "The truth was "),
]

KEY_METHODS = {
    "adaptive_full", "ablation_risk_only", "ablation_entropy_only",
    "rep_penalty_1.3", "mirostat_tau5", "temp_0.8",
    "nucleus_p0.95", "no_repeat_4gram", "lz_decoder",
}


def make_prompt_seed_pairs(prompts, n_seeds):
    return [(p, s) for p in prompts for s in range(1, n_seeds + 1)]


BASELINE_CONFIGS = [
    {"name": "greedy",
     "temperature": 0.0, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "lz": False, "category": "baseline"},
    {"name": "temp_0.7",
     "temperature": 0.7, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "lz": False, "category": "baseline_sweep"},
    {"name": "temp_0.8",
     "temperature": 0.8, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "lz": False, "category": "baseline_sweep"},
    {"name": "temp_0.9",
     "temperature": 0.9, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "lz": False, "category": "baseline_sweep"},
    {"name": "temp_1.0",
     "temperature": 1.0, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "lz": False, "category": "baseline_sweep"},
    {"name": "nucleus_p0.90",
     "temperature": 1.0, "top_k": 0, "top_p": 0.90,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "lz": False, "category": "baseline_sweep"},
    {"name": "nucleus_p0.95",
     "temperature": 1.0, "top_k": 0, "top_p": 0.95,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "lz": False, "category": "baseline_sweep"},
    {"name": "nucleus_p0.99",
     "temperature": 1.0, "top_k": 0, "top_p": 0.99,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "lz": False, "category": "baseline_sweep"},
    {"name": "rep_penalty_1.1",
     "temperature": 0.8, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.1, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "lz": False, "category": "baseline_sweep"},
    {"name": "rep_penalty_1.3",
     "temperature": 0.8, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.3, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "lz": False, "category": "baseline_sweep"},
    {"name": "rep_penalty_1.5",
     "temperature": 0.8, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.5, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "lz": False, "category": "baseline_sweep"},
    {"name": "mirostat_tau3",
     "temperature": 1.0, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 3.0, "adaptive": None, "lz": False, "category": "probabilistic"},
    {"name": "mirostat_tau5",
     "temperature": 1.0, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 5.0, "adaptive": None, "lz": False, "category": "probabilistic"},
    {"name": "typical_p0.9",
     "temperature": 1.0, "top_k": 0, "top_p": 1.0,
     "typical_p": 0.9, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "lz": False, "category": "probabilistic"},
    {"name": "lz_decoder",
     "temperature": 0.8, "top_k": 0, "top_p": 0.95,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0,
     "mirostat_tau": 0.0, "adaptive": None, "lz": True, "category": "strong_baseline"},
    {"name": "no_repeat_4gram",
     "temperature": 0.8, "top_k": 0, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 4,
     "mirostat_tau": 0.0, "adaptive": None, "lz": False, "category": "hard_constraint"},
]


def make_ablation_configs():
    base = dict(temperature=0.8, top_k=0, top_p=0.95,
                typical_p=1.0, rep_penalty=1.0, no_repeat_ngram=0,
                mirostat_tau=0.0, lz=False, category="ablation")

    def adaptive(name, **kwargs):
        dec = dict(temperature=0.8, top_p=0.95, n_min=3, n_max=6,
                   alpha_base=2.0, alpha_max=8.0, lambda_rep=10.0,
                   lambda_ent=1.0, rep_target=0.05, ent_target=3.5, window=100)
        dec.update(kwargs)
        return {**base, "name": name, "adaptive": dec}

    return [
        adaptive("adaptive_full"),
        adaptive("ablation_fixed_alpha",  lambda_rep=0.0, lambda_ent=0.0),
        adaptive("ablation_risk_only",    lambda_ent=0.0),
        adaptive("ablation_entropy_only", lambda_rep=0.0),
        adaptive("ablation_no_top_p",     top_p=1.0),
        adaptive("ablation_narrow_ngram", n_min=3, n_max=4),
        adaptive("ablation_wide_ngram",   n_min=2, n_max=8),
        {**base, "name": "ablation_hard_in_adaptive",
         "no_repeat_ngram": 4, "adaptive": None, "top_p": 0.95},
    ]


ABLATION_CONFIGS = make_ablation_configs()
ALL_CONFIGS      = BASELINE_CONFIGS + ABLATION_CONFIGS


def make_decoder(cfg):
    if cfg.get("lz"):
        return None, LZRepetitionDecoder(temperature=0.8, top_p=0.95)
    if cfg.get("adaptive") is not None:
        return RecurrenceAwareDecoder(**cfg["adaptive"]), None
    return None, None


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

def compression_ratio(text):
    """zlib compression ratio — lower = more repetitive (leakage check)."""
    raw = text.encode("utf-8")
    if not raw: return 1.0
    return round(len(zlib.compress(raw, level=6)) / len(raw), 4)

def repeated_suffix_length(text, n=10):
    """Length of longest suffix of text that also appears earlier in text."""
    if len(text) < n: return 0
    for length in range(min(len(text)//2, 200), n-1, -1):
        suffix = text[-length:]
        if text[:-length].find(suffix) >= 0:
            return length
    return 0


@torch.no_grad()
def generated_text_nll(model, text_ids, block_size, device):
    """Full-sample NLL: stride=block_size windows covering all tokens."""
    ids = np.array(text_ids, dtype=np.int64)
    n   = len(ids)
    if n < 2: return float("nan")
    all_nll = []
    for s in range(0, max(1, n-1), block_size):
        end_x = min(s+block_size, n-1)
        x_np, y_np = ids[s:end_x], ids[s+1:end_x+1]
        if len(x_np) == 0: continue
        x = torch.tensor(x_np, dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(y_np, dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(x)
        lp  = F.log_softmax(logits, dim=-1)
        nll = -lp.gather(2, y.unsqueeze(-1)).squeeze(-1)
        all_nll.extend(nll[0].tolist())
    if not all_nll: return float("nan")
    return float(np.mean(all_nll) / math.log(2))

def ngram_distributional_similarity(gen_text, ref_text, n=4):
    def dist(text):
        gs = [text[i:i+n] for i in range(len(text)-n+1)]
        if not gs: return {}
        c = Counter(gs); t = sum(c.values())
        return {k: v/t for k, v in c.items()}
    p = dist(gen_text); q = dist(ref_text)
    if not p or not q: return 0.0
    vocab = set(p)|set(q)
    m = {k: 0.5*(p.get(k,0)+q.get(k,0)) for k in vocab}
    def kl(a, b):
        return sum(a[k]*math.log2(a[k]/b[k]) for k in a if a[k]>0 and b.get(k,0)>0)
    jsd = max(0.0, min(1.0, 0.5*kl(p,m)+0.5*kl(q,m)))
    return round(1.0-jsd, 4)

def _load_english_wordlist():
    for path in ["/usr/share/dict/words", "/usr/dict/words"]:
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="ignore") as f:
                return {w.strip().lower() for w in f if w.strip().isalpha()}
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
    return round(sum(1 for w in words if w not in _ENGLISH_WORDS)/len(words), 4)


def kaplan_meier_survival(loop_onsets, max_t=500):
    events  = sorted([t for t in loop_onsets if t >= 0])
    n_total = len(loop_onsets)
    if not events:
        return list(range(max_t+1)), [1.0]*(max_t+1)
    times = sorted(set(events))
    S, t_out, surv = 1.0, [], []
    n_before = 0
    for t in times:
        n_ev = events.count(t)
        n_at = n_total - n_before
        if n_at > 0: S *= (1 - n_ev/n_at)
        n_before += n_ev
        t_out.append(t); surv.append(round(S, 4))
    return t_out, surv

def survival_auc(loop_onsets, max_t=500):
    ts, S = kaplan_meier_survival(loop_onsets, max_t)
    if not ts: return 1.0
    prev_t, prev_s, area = 0, 1.0, 0.0
    for t, s in zip(ts, S):
        area += (t-prev_t)*prev_s; prev_t, prev_s = t, s
    area += (max_t-prev_t)*prev_s
    return round(area/max_t, 4)

def rmst(loop_onsets, tau=500):
    """Restricted Mean Survival Time up to tau (expected loop-free chars)."""
    ts, S = kaplan_meier_survival(loop_onsets, max_t=tau)
    if not ts: return float(tau)
    prev_t, prev_s, area = 0, 1.0, 0.0
    for t, s in zip(ts, S):
        area += (t-prev_t)*prev_s; prev_t, prev_s = t, s
    area += (tau-prev_t)*prev_s
    return round(area, 2)

def loop_rate(loop_onsets):
    return round(sum(1 for t in loop_onsets if t >= 0)/len(loop_onsets), 4)

def bootstrap_ci(values, n_boot=1000, seed=0, alpha=0.05):
    rng   = np.random.default_rng(seed)
    arr   = np.array(values, dtype=float)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_boot)]
    boots = np.array(boots)
    return (round(float(arr.mean()), 4),
            round(float(np.percentile(boots, 100*alpha/2)), 4),
            round(float(np.percentile(boots, 100*(1-alpha/2))), 4))

def paired_bootstrap_test(vals_a, vals_b, n_boot=1000, seed=0):
    rng   = np.random.default_rng(seed)
    a, b  = np.array(vals_a, float), np.array(vals_b, float)
    n     = min(len(a), len(b))
    delta = a[:n] - b[:n]
    obs   = delta.mean()
    boots = [rng.choice(delta, size=n, replace=True).mean()
             for _ in range(n_boot)]
    boots_c = np.array(boots) - np.mean(boots)
    return round(float(np.mean(np.abs(boots_c) >= abs(obs))), 4)

def bootstrap_rmst_diff(lo_a, lo_b, tau=500, n_boot=1000, seed=0):
    """Bootstrap CI for RMST(A)-RMST(B) using prompt/seed as paired unit."""
    rng = np.random.default_rng(seed)
    n   = min(len(lo_a), len(lo_b))
    obs = rmst(lo_a[:n], tau) - rmst(lo_b[:n], tau)
    idx_all = np.arange(n)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        boots.append(rmst([lo_a[i] for i in idx], tau) -
                     rmst([lo_b[i] for i in idx], tau))
    boots = np.array(boots)
    return (round(float(obs), 2),
            round(float(np.percentile(boots, 2.5)), 2),
            round(float(np.percentile(boots, 97.5)), 2))


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


SCALAR_KEYS = [
    "ttr", "entropy_4gram", "rep_rate_5",
    "rep_ngram_mass_2", "rep_ngram_mass_3", "rep_ngram_mass_4",
    "rep_ngram_mass_5", "rep_ngram_mass_6",
    "longest_rep_sub", "compression_ratio", "repeated_suffix_len",
    "gen_nll_bpc", "ngram_sim_4", "spelling_error_rate",
]

def eval_sample(model, gen_ids, text, block_size, device, ref_text):
    lo = loop_onset(text, 10)
    return lo, {
        "ttr":                type_token_ratio(text),
        "entropy_4gram":      char_ngram_entropy(text, 4),
        "rep_rate_5":         repetition_rate(text, 5),
        "rep_ngram_mass_2":   rep_ngram_mass(text, 2),
        "rep_ngram_mass_3":   rep_ngram_mass(text, 3),
        "rep_ngram_mass_4":   rep_ngram_mass(text, 4),
        "rep_ngram_mass_5":   rep_ngram_mass(text, 5),
        "rep_ngram_mass_6":   rep_ngram_mass(text, 6),
        "longest_rep_sub":    longest_repeated_substring(text),
        "compression_ratio":  compression_ratio(text),
        "repeated_suffix_len":repeated_suffix_length(text),
        "gen_nll_bpc":        generated_text_nll(model, gen_ids, block_size, device),
        "ngram_sim_4":        ngram_distributional_similarity(text, ref_text, 4),
        "spelling_error_rate":spelling_error_rate(text),
    }


def run_configs(configs, model, cfg, stoi, itos, ref_text,
                pairs_std, pairs_ext, n_chars, device, samples_f, ckpt_name):
    block_size = cfg["block_size"]
    rows = []

    for c in configs:
        is_key = c["name"] in KEY_METHODS
        pairs  = pairs_ext if is_key else pairs_std
        print(f"    {c['name']:<30}{'[ext]' if is_key else '     '}", end="", flush=True)

        accum       = {k: [] for k in SCALAR_KEYS}
        loop_onsets = []
        cps_list    = []

        for (prompt_name, prompt_text), seed in pairs:
            set_seed(seed)
            idx        = encode(prompt_text, stoi).to(device)
            adec, ldec = make_decoder(c)

            out, cps = generate(
                model, idx, max_new_tokens=n_chars,
                temperature=c["temperature"], top_k=c["top_k"],
                top_p=c["top_p"], typical_p=c["typical_p"],
                rep_penalty=c["rep_penalty"],
                no_repeat_ngram=c["no_repeat_ngram"],
                mirostat_tau=c["mirostat_tau"],
                adaptive=adec, lz_decoder=ldec,
                measure_time=True,
            )
            if cps is not None:
                cps_list.append(cps)

            gen_ids = out[0].tolist()[len(prompt_text):]
            text    = decode(gen_ids, itos)
            lo, m   = eval_sample(model, gen_ids, text, block_size, device, ref_text)
            loop_onsets.append(lo)
            for k in SCALAR_KEYS:
                accum[k].append(m[k])

            if seed == 1:
                samples_f.write(
                    f"[{c['name']}][{ckpt_name}] prompt='{prompt_text.strip()}'\n"
                    + "-"*60 + "\n" + text + "\n\n"
                )

        sauc   = survival_auc(loop_onsets, max_t=n_chars)
        lrate  = loop_rate(loop_onsets)
        rmst_v = rmst(loop_onsets, tau=n_chars)
        km_t, km_s = kaplan_meier_survival(loop_onsets, max_t=n_chars)

        row = {
            "strategy":     c["name"],
            "category":     c["category"],
            "checkpoint":   ckpt_name,
            "n_samples":    len(loop_onsets),
            "n_censored":   sum(1 for t in loop_onsets if t < 0),
            "loop_rate":    lrate,
            "survival_auc": sauc,
            "rmst":         rmst_v,
            "chars_per_sec": round(float(np.mean(cps_list)), 1) if cps_list else None,
            "km_times":     km_t,
            "km_survival":  km_s,
            "loop_onsets_raw": loop_onsets,
        }

        for k in SCALAR_KEYS:
            mean, lo_ci, hi_ci = bootstrap_ci(accum[k])
            row[f"{k}_mean"]  = mean
            row[f"{k}_ci_lo"] = lo_ci
            row[f"{k}_ci_hi"] = hi_ci
            row[f"{k}_vals"]  = accum[k]

        rng = np.random.default_rng(42)
        lo_arr = np.array(loop_onsets)
        boot_sauc, boot_lrate, boot_rmst = [], [], []
        for _ in range(1000):
            s = rng.choice(lo_arr, size=len(lo_arr), replace=True).tolist()
            boot_sauc.append(survival_auc(s, max_t=n_chars))
            boot_lrate.append(loop_rate(s))
            boot_rmst.append(rmst(s, tau=n_chars))
        row["survival_auc_ci_lo"] = round(float(np.percentile(boot_sauc, 2.5)), 4)
        row["survival_auc_ci_hi"] = round(float(np.percentile(boot_sauc, 97.5)), 4)
        row["loop_rate_ci_lo"]    = round(float(np.percentile(boot_lrate, 2.5)), 4)
        row["loop_rate_ci_hi"]    = round(float(np.percentile(boot_lrate, 97.5)), 4)
        row["rmst_ci_lo"]         = round(float(np.percentile(boot_rmst, 2.5)), 2)
        row["rmst_ci_hi"]         = round(float(np.percentile(boot_rmst, 97.5)), 2)

        rows.append(row)
        cps_s = f"  cps={row['chars_per_sec']}" if row['chars_per_sec'] else ""
        print(f"  lr={lrate:.2f}  sauc={sauc:.3f}  rmst={rmst_v:.0f}"
              f"  nll={row['gen_nll_bpc_mean']:.3f}"
              f"  sim={row['ngram_sim_4_mean']:.3f}"
              f"  comp={row['compression_ratio_mean']:.3f}"
              f"  cens={row['n_censored']}/{row['n_samples']}{cps_s}")

    return rows


def run_method_vs_method_tests(all_results, out_dir, n_chars):
    """Method-vs-method paired comparisons using prompt/seed as paired unit."""
    by_key = {}
    for ckpt_name, rows in all_results.items():
        for r in rows:
            by_key[(ckpt_name, r["strategy"])] = r

    comparisons = []
    ckpt_names  = list(all_results.keys())

    def compare(ckpt, a, b, label):
        ra = by_key.get((ckpt, a))
        rb = by_key.get((ckpt, b))
        if not ra or not rb: return
        lo_a = ra["loop_onsets_raw"]
        lo_b = rb["loop_onsets_raw"]
        n    = min(len(lo_a), len(lo_b))
        rmst_d, rmst_lo, rmst_hi = bootstrap_rmst_diff(lo_a[:n], lo_b[:n], tau=n_chars)
        sauc_a = [survival_auc([x], max_t=n_chars) for x in lo_a[:n]]
        sauc_b = [survival_auc([x], max_t=n_chars) for x in lo_b[:n]]
        p_sauc = paired_bootstrap_test(sauc_a, sauc_b)
        row = {
            "label": label, "checkpoint": ckpt,
            "method_a": a, "method_b": b, "n_pairs": n,
            "rmst_a": ra["rmst"], "rmst_b": rb["rmst"],
            "rmst_diff_AminusB": rmst_d,
            "rmst_ci_lo": rmst_lo, "rmst_ci_hi": rmst_hi,
            "p_survival_auc": p_sauc,
        }
        for k in ["gen_nll_bpc", "ngram_sim_4", "rep_rate_5", "spelling_error_rate"]:
            va = ra.get(f"{k}_vals", [])[:n]
            vb = rb.get(f"{k}_vals", [])[:n]
            if va and vb:
                row[f"p_{k}"]     = paired_bootstrap_test(va, vb)
                row[f"delta_{k}"] = round(float(np.mean(va))-float(np.mean(vb)), 4)
        comparisons.append(row)
        print(f"    {a:<26} vs {b:<26}  "
              f"RMST_diff={rmst_d:+.1f}[{rmst_lo:+.1f},{rmst_hi:+.1f}]  "
              f"p_sauc={p_sauc:.3f}")

    print(f"\n  Method-vs-method paired tests")
    print(f"  {'='*90}")
    for ckpt in ckpt_names:
        print(f"\n  [{ckpt}]")
        compare(ckpt, "adaptive_full",     "ablation_risk_only",   "adaptive vs risk_only")
        compare(ckpt, "adaptive_full",     "ablation_entropy_only","adaptive vs entropy_only")
        compare(ckpt, "ablation_risk_only","temp_0.8",             "risk_only vs temp_0.8")
        compare(ckpt, "ablation_risk_only","nucleus_p0.95",        "risk_only vs nucleus_0.95")
        compare(ckpt, "ablation_risk_only","typical_p0.9",         "risk_only vs typical_0.9")
        compare(ckpt, "ablation_risk_only","mirostat_tau5",        "risk_only vs mirostat_tau5")
        compare(ckpt, "ablation_risk_only","rep_penalty_1.3",      "risk_only vs rep_penalty_1.3")
        compare(ckpt, "adaptive_full",     "rep_penalty_1.3",      "adaptive vs rep_penalty")
        compare(ckpt, "adaptive_full",     "mirostat_tau5",        "adaptive vs mirostat")
        compare(ckpt, "adaptive_full",     "lz_decoder",           "adaptive vs lz_decoder")
        compare(ckpt, "adaptive_full",     "no_repeat_4gram",
                "adaptive vs hard_no_repeat [leakage: hard bans measured event]")
        compare(ckpt, "rep_penalty_1.3",   "no_repeat_4gram",
                "rep_penalty vs hard_no_repeat [leakage]")

    if comparisons:
        path = os.path.join(out_dir, "method_vs_method.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(comparisons[0].keys()))
            w.writeheader(); w.writerows(comparisons)
        print(f"\n  Saved -> {path}")
    return comparisons


def save_pareto_data(all_results, out_dir):
    """Pareto plot data: survival vs NLL/sim/spelling. Hard no-repeat flagged."""
    pareto = []
    for ckpt_name, rows in all_results.items():
        for r in rows:
            pareto.append({
                "strategy":        r["strategy"],
                "category":        r["category"],
                "checkpoint":      ckpt_name,
                "survival_auc":    r["survival_auc"],
                "rmst":            r["rmst"],
                "gen_nll_bpc":     r.get("gen_nll_bpc_mean"),
                "ngram_sim_4":     r.get("ngram_sim_4_mean"),
                "spelling_error":  r.get("spelling_error_rate_mean"),
                "rep_rate_5":      r.get("rep_rate_5_mean"),
                "compression":     r.get("compression_ratio_mean"),
                "chars_per_sec":   r.get("chars_per_sec"),
                "hard_constraint": r["category"] == "hard_constraint",
            })
    path = os.path.join(out_dir, "pareto_data.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pareto[0].keys()))
        w.writeheader(); w.writerows(pareto)
    print(f"  Pareto data -> {path}")


def save_qualitative_examples(all_results, model_map, stoi_map,
                               itos_map, cfg_map, n_chars, device, out_dir):
    """
    Matched qualitative examples using same prompt+seed.
    Shows: greedy loops, hard no-repeat avoids loop but changes text,
    soft adaptive avoids loop while preserving quality.
    """
    qual_methods = [
        "greedy", "temp_0.8", "rep_penalty_1.3",
        "no_repeat_4gram", "adaptive_full", "lz_decoder",
    ]
    example_triples = [
        ("chapter", "CHAPTER 1\n", 1),
        ("best",    "It was the best of ", 2),
        ("darcy",   "Mr. Darcy had never ", 3),
    ]
    cfg_by_name = {c["name"]: c for c in ALL_CONFIGS}
    path = os.path.join(out_dir, "qualitative_examples.txt")

    with open(path, "w", encoding="utf-8") as f:
        for ckpt_name, model in model_map.items():
            stoi = stoi_map[ckpt_name]
            itos = itos_map[ckpt_name]
            f.write(f"\n{'#'*80}\nCheckpoint: {ckpt_name}\n{'#'*80}\n\n")

            for prompt_name, prompt_text, seed in example_triples:
                f.write(f"{'='*80}\n")
                f.write(f"PROMPT: '{prompt_text.strip()}'  SEED: {seed}\n")
                f.write(f"{'='*80}\n\n")
                for method_name in qual_methods:
                    c = cfg_by_name.get(method_name)
                    if not c: continue
                    set_seed(seed)
                    idx        = encode(prompt_text, stoi).to(device)
                    adec, ldec = make_decoder(c)
                    out, _     = generate(
                        model, idx, max_new_tokens=n_chars,
                        temperature=c["temperature"], top_k=c["top_k"],
                        top_p=c["top_p"], typical_p=c["typical_p"],
                        rep_penalty=c["rep_penalty"],
                        no_repeat_ngram=c["no_repeat_ngram"],
                        mirostat_tau=c["mirostat_tau"],
                        adaptive=adec, lz_decoder=ldec,
                    )
                    gen_ids = out[0].tolist()[len(prompt_text):]
                    text    = decode(gen_ids, itos)
                    lo      = loop_onset(text, 10)
                    lo_str  = f"loop@{lo}" if lo >= 0 else "no_loop"
                    f.write(f"[{method_name}]  [{lo_str}]\n")
                    f.write(text[:400] + ("..." if len(text)>400 else "") + "\n\n")
    print(f"  Qualitative examples -> {path}")


def eval_checkpoint(ckpt_path, data_dir, n_chars, pairs_std, pairs_ext, device, out_dir):
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
                       pairs_std, pairs_ext, n_chars, device, samples_f, name)
    samples_f.close()

    skip = {"km_times", "km_survival", "loop_onsets_raw"} | \
           {f"{k}_vals" for k in SCALAR_KEYS}
    csv_keys = [k for k in rows[0] if k not in skip]
    with open(os.path.join(out_dir, f"metrics_{name}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_keys)
        w.writeheader()
        w.writerows([{k: r[k] for k in csv_keys} for r in rows])

    json_rows = [{k: v for k, v in r.items()
                  if k not in {f"{m}_vals" for m in SCALAR_KEYS}} for r in rows]
    with open(os.path.join(out_dir, f"metrics_{name}.json"), "w") as f:
        json.dump(json_rows, f, indent=2)

    return rows, model, stoi, itos, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",     nargs="+",
                    default=["runs/baseline/best.pt", "runs/cosine/best.pt"])
    ap.add_argument("--data_dir", default="data_out")
    ap.add_argument("--out_dir",  default="runs/sampling_eval_v5")
    ap.add_argument("--n_chars",  type=int, default=500)
    ap.add_argument("--n_seeds",  type=int, default=10)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))

    pairs_std = make_prompt_seed_pairs(PROMPTS_STANDARD, args.n_seeds)
    pairs_ext = make_prompt_seed_pairs(PROMPTS_EXTENDED, args.n_seeds)

    print(f"Device     : {device}")
    print(f"Configs    : {len(ALL_CONFIGS)}")
    print(f"Std pairs  : {len(pairs_std)} per standard config")
    print(f"Ext pairs  : {len(pairs_ext)} per key config")
    print(f"Checkpoints: {len(args.ckpt)}")

    all_results = {}
    model_map, stoi_map, itos_map, cfg_map = {}, {}, {}, {}

    for ckpt in args.ckpt:
        name = os.path.basename(os.path.dirname(ckpt))
        rows, model, stoi, itos, cfg = eval_checkpoint(
            ckpt, args.data_dir, args.n_chars,
            pairs_std, pairs_ext, device, args.out_dir)
        all_results[name] = rows
        model_map[name]   = model
        stoi_map[name]    = stoi
        itos_map[name]    = itos
        cfg_map[name]     = cfg

    run_method_vs_method_tests(all_results, args.out_dir, args.n_chars)
    save_pareto_data(all_results, args.out_dir)
    save_qualitative_examples(all_results, model_map, stoi_map,
                              itos_map, cfg_map, args.n_chars, device, args.out_dir)

    skip = {f"{k}_vals" for k in SCALAR_KEYS}
    serialisable = {
        name: [{k: v for k, v in r.items() if k not in skip} for r in rows]
        for name, rows in all_results.items()
    }
    with open(os.path.join(args.out_dir, "all_results.json"), "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n  Done -> {args.out_dir}")


if __name__ == "__main__":
    main()