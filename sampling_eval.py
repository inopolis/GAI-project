"""
sampling_eval.py — Rigorous decoding evaluation with quality metrics and
recurrence-aware adaptive decoder.

New vs previous version:
  - AdaptiveDecoder added as a strategy
  - Quality metrics: generated-text NLL, n-gram distributional similarity
  - Loop-onset survival curve (censored Kaplan-Meier style)
  - Both checkpoints evaluated and compared
  - no_repeat_4gram clearly flagged as hard constraint baseline
"""

import os, sys, argparse, csv, json, math, numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from src.utils import set_seed, load_json, ensure_dir
from src.model import CharTransformerLM
from src.decoding import generate, RecurrenceAwareDecoder


# Configs

CONFIGS = [
    {"name":"greedy", "temperature":0.0, "top_k":0, "top_p":1.0,
     "typical_p":1.0, "rep_penalty":1.0, "no_repeat_ngram":0, "mirostat_tau":0.0,
     "adaptive":False, "category":"baseline"},

    {"name":"temp_0.8", "temperature":0.8, "top_k":0, "top_p":1.0,
     "typical_p":1.0, "rep_penalty":1.0, "no_repeat_ngram":0, "mirostat_tau":0.0,
     "adaptive":False, "category":"baseline"},

    {"name":"nucleus_p0.95", "temperature":1.0, "top_k":0, "top_p":0.95,
     "typical_p":1.0, "rep_penalty":1.0, "no_repeat_ngram":0, "mirostat_tau":0.0,
     "adaptive":False, "category":"baseline"},

    {"name":"rep_penalty_1.3","temperature":0.8, "top_k":0, "top_p":1.0,
     "typical_p":1.0, "rep_penalty":1.3, "no_repeat_ngram":0, "mirostat_tau":0.0,
     "adaptive":False, "category":"probabilistic"},

    {"name":"mirostat_tau5", "temperature":1.0, "top_k":0, "top_p":1.0,
     "typical_p":1.0, "rep_penalty":1.0, "no_repeat_ngram":0, "mirostat_tau":5.0,
     "adaptive":False, "category":"probabilistic"},

    {"name":"typical_p0.9", "temperature":1.0, "top_k":0, "top_p":1.0,
     "typical_p":0.9, "rep_penalty":1.0, "no_repeat_ngram":0, "mirostat_tau":0.0,
     "adaptive":False, "category":"probabilistic"},

    # Hard constraint baseline — not a fair probabilistic comparison
    {"name":"no_repeat_4gram","temperature":0.8, "top_k":0, "top_p":1.0,
     "typical_p":1.0, "rep_penalty":1.0, "no_repeat_ngram":4, "mirostat_tau":0.0,
     "adaptive":False, "category":"hard_constraint"},

    # Recurrence-aware adaptive decoder (novel contribution)
    {"name":"adaptive", "temperature":0.8, "top_k":0, "top_p":0.95,
     "typical_p":1.0, "rep_penalty":1.0, "no_repeat_ngram":0, "mirostat_tau":0.0,
     "adaptive":True,  "category":"adaptive"},
]

PROMPTS = [
    ("chapter", "CHAPTER 1\n"),
    ("night", "The night was "),
    ("she", "She had never "),
    ("best", "It was the best of "),
    ("darcy", "Mr. Darcy had never "),
]


# Degeneration metrics

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
    return -1   # censored — no loop detected

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
def generated_text_nll(model, text_ids: list, block_size: int, device) -> float:
    """
    NLL of the generated text under the model itself.
    Lower = model assigns higher probability to what it generated
    (proxy for coherence and staying in-distribution).

    We use non-overlapping windows, same as eval_bpc.py.
    """
    if len(text_ids) < block_size + 1:
        # Short text: single forward pass
        x = torch.tensor([text_ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([text_ids[1:]],  dtype=torch.long, device=device)
        if x.shape[1] == 0:
            return float("nan")
        # Pad/truncate to block_size
        T = min(x.shape[1], block_size)
        x, y = x[:, :T], y[:, :T]
        logits, _ = model(x)
        lp  = F.log_softmax(logits, dim=-1)
        nll = -lp.gather(2, y.unsqueeze(-1)).squeeze(-1).mean().item()
        return float(nll / math.log(2))   # nats -> BPC

    data = np.array(text_ids, dtype=np.int64)
    n    = len(data)
    n_win = (n - 1) // block_size
    all_nll = []
    for start in range(0, n_win * block_size, block_size):
        s = start
        if s + block_size + 1 > n: break
        x = torch.tensor([data[s : s+block_size]],       dtype=torch.long, device=device)
        y = torch.tensor([data[s+1 : s+block_size+1]],   dtype=torch.long, device=device)
        logits, _ = model(x)
        lp  = F.log_softmax(logits, dim=-1)
        nll = -lp.gather(2, y.unsqueeze(-1)).squeeze(-1).mean().item()
        all_nll.append(nll)
    if not all_nll:
        return float("nan")
    return float(np.mean(all_nll) / math.log(2))  # BPC


def ngram_distributional_similarity(
    gen_text: str, ref_text: str, n: int = 4
) -> float:
    """
    Jensen-Shannon divergence between n-gram distributions of generated
    text and reference text. Returns 1 - JSD so that higher = more similar.

    JSD in [0, 1] (using log2). Similarity = 1 means identical distributions.
    """
    def dist(text):
        gs = [text[i:i+n] for i in range(len(text)-n+1)]
        if not gs: return {}
        c = Counter(gs); t = sum(c.values())
        return {k: v/t for k, v in c.items()}

    p = dist(gen_text)
    q = dist(ref_text)
    if not p or not q:
        return 0.0

    vocab = set(p) | set(q)
    m = {k: 0.5*(p.get(k,0) + q.get(k,0)) for k in vocab}

    def kl(a, b):
        return sum(a[k]*math.log2(a[k]/b[k]) for k in a if a[k] > 0 and b[k] > 0)

    jsd = 0.5*kl(p, m) + 0.5*kl(q, m)
    jsd = max(0.0, min(1.0, jsd))
    return round(1.0 - jsd, 4)


# Spelling error rate

def _load_english_wordlist() -> set:
    """
    Loads a word list without external dependencies.
    Uses /usr/share/dict/words if available (macOS/Linux standard).
    Falls back to a compact built-in set of the 5000 most common English words.
    """
    import os
    for path in ["/usr/share/dict/words", "/usr/dict/words"]:
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="ignore") as f:
                return {w.strip().lower() for w in f if w.strip().isalpha()}

    # Compact fallback — most frequent English words covering +-85% of running text (for windows users)
    COMMON = (
        "the of and a to in is it you that he was for on are with as his they "
        "be at one have this from or had by but not what all were we when your "
        "can said there use an each which she do how their if will up other "
        "about out many then them these so some her would make like him into "
        "time has look two more go see no way could my than first been its "
        "who now people my made over did down only way find use may water long "
        "little very after words called just where most know get through back "
        "much before go good new write our used me man too any day same right "
        "look think also around another came come work three word must because "
        "does part even place well such here take why help put different away "
        "again off went old number great tell men say small every found still "
        "between name should home big give air line set own under read last "
        "never us left end along while might next sound below saw something "
        "thought both few those always show large often together ask house "
        "world below asked went men read need land different home us move "
        "try kind hand picture change play spell air away animal house point "
        "page letter mother answer found study still learn plant cover food "
        "sun four between state keep eye never last let thought city tree "
        "cross farm hard start might story saw far sea draw left late run "
        "kept watch cut children white sea began grow took river four carry "
        "state once book hear stop without second late miss idea eat face "
        "watch far enough near open seem together next white children begin "
        "got walk example ease paper often always music those both mark book "
        "letter until mile river car feet care second enough plain girl usual "
        "young ready above ever red list though feel talk bird soon body dog "
        "family direct pose leave song measure door product black short numeral "
        "class wind question happen complete ship area half rock order fire "
        "south problem piece told knew pass since top whole king space heard "
        "best hour better true during hundred five remember step early hold "
        "west ground interest reach fast verb sing listen six table travel "
        "less morning ten simple several vowel toward power town fine drive "
        "warm free ride yes proper fall lead dark machine note wait plan "
        "figure star box noun field rest correct able pound done beauty "
        "drive stood contain front teach week final gave green oh quick "
        "develop ocean warm free minute strong special mind behind clear "
        "tail produce fact street inch multiply nothing course stay wheel "
        "full force blue object decide surface deep moon island foot yet "
        "busy test record boat common gold possible plane age dry wonder "
        "laugh thousand ago ran check game shape equate hot miss brought "
        "heat snow tire bring yes distant fill east paint language among "
        "grand ball yet wave drop heart am present heavy dance engine "
        "position arm wide sail material size vary settle speak weight "
        "general ice matter circle pair include divide syllable felt perhaps "
        "pick sudden count square reason length represent art subject region "
        "energy hunt probable bed brother egg ride cell believe fraction "
        "forest sit race window store summer train sleep prove lone leg "
        "exercise wall catch mount wish sky board joy winter sat written "
        "wild instrument kept glass grass cow job edge sign visit past "
        "soft fun bright gas weather month million bear finish happy hope "
        "flower clothe strange gone trade melody trip office receive row "
        "mouth exact symbol die least trouble shout except wrote seed tone "
        "join suggest clean break lady yard rise bad blow oil blood touch "
        "grew cent mix team wire cost lost brown wear garden equal sent "
        "choose fell fit flow fair bank collect save control decimal gentle "
        "woman captain practice separate difficult doctor please protect noon "
        "whose locate ring character insect caught period indicate radio spoke "
        "atom human history effect electric expect crop modern element hit "
        "student corner party supply bone rail imagine provide agree thus "
        "capital chair danger fruit rich thick soldier process operate guess "
        "necessary sharp wing create neighbor wash bat rather crowd corn "
        "compare poem string bell depend meat rub tube famous dollar stream "
        "fear sight thin triangle planet hurry chief colony clock mine tie "
        "enter major fresh search send yellow gun allow print dead spot "
        "desert suit current lift rose arrive master track parent shore "
        "division sheet substance favor connect post spend chord fat glad "
        "original share station dad bread charge proper bar offer segment "
        "slave duck instant market degree populate chick dear enemy reply "
        "drink occur support speech nature range steam motion path liquid "
        "log meant quotient teeth shell neck"
    )
    return set(COMMON.split())

# Build once at module load
_ENGLISH_WORDS: set = _load_english_wordlist()


def spelling_error_rate(text: str) -> float:
    """
    Fraction of alphabetic word tokens not found in the English word list.

    Only purely alphabetic tokens are checked (punctuation, numbers ignored).
    Case-insensitive. Lower is better — 0.0 means all words are valid English.

    Interpretation:
      - Greedy (loops): high rate because repeated garbage chars form non-words
      - Stochastic methods: moderate rate from rare/invented words
      - Adaptive: should be low if it stays close to model distribution
    """
    words = [w.lower() for w in text.split() if w.isalpha()]
    if not words:
        return 0.0
    errors = sum(1 for w in words if w not in _ENGLISH_WORDS)
    return round(errors / len(words), 4)


# Survival curve

def kaplan_meier_survival(loop_onsets: list, max_t: int = 500):
    """
    Kaplan-Meier survival estimate for loop-onset times.

    loop_onsets: list of ints where -1 means censored (no loop observed).
    Returns: (times, survival) arrays suitable for plotting.

    Censored observations (-1) contribute to the at-risk set for all t
    but do not count as events — this is correct survival analysis,
    unlike treating -1 as a large value or ignoring those samples.
    """
    events    = sorted([t for t in loop_onsets if t >= 0])
    n_censored = sum(1 for t in loop_onsets if t < 0)
    n_total    = len(loop_onsets)

    if not events:
        return list(range(max_t+1)), [1.0]*(max_t+1)

    times    = sorted(set(events))
    S        = 1.0
    survival = []
    t_out    = []

    # At each event time, n_at_risk = total - (events and censored before t)
    n_before = 0
    for t in times:
        n_events_at_t = events.count(t)
        n_at_risk     = n_total - n_before
        if n_at_risk > 0:
            S *= (1 - n_events_at_t / n_at_risk)
        n_before += n_events_at_t
        t_out.append(t)
        survival.append(round(S, 4))

    return t_out, survival


def survival_auc(loop_onsets: list, max_t: int = 500) -> float:
    """
    Area under the survival curve up to max_t, normalised to [0,1].
    Higher = model survives longer without looping.
    """
    ts, S = kaplan_meier_survival(loop_onsets, max_t)
    if not ts:
        return 1.0
    # Trapezoidal integral from 0 to max_t
    prev_t, prev_s, area = 0, 1.0, 0.0
    for t, s in zip(ts, S):
        area   += (t - prev_t) * prev_s
        prev_t, prev_s = t, s
    area += (max_t - prev_t) * prev_s
    return round(area / max_t, 4)


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


# Eval one checkpoint

SCALAR_KEYS = [
    "ttr", "entropy_4gram", "rep_rate_5",
    "rep_ngram_mass_2", "rep_ngram_mass_4", "rep_ngram_mass_6",
    "loop_onset", "longest_rep_sub",
    "gen_nll_bpc", "ngram_sim_4", "spelling_error_rate",
    "survival_auc",
]

def eval_checkpoint(ckpt_path, data_dir, n_chars, n_seeds, device, out_dir):
    model, cfg = load_model(ckpt_path, device)
    vocab = load_json(os.path.join(data_dir, "vocab.json"))
    stoi, itos = vocab["stoi"], vocab["itos"]
    name = os.path.basename(os.path.dirname(ckpt_path))
    block_size = cfg["block_size"]

    # Load reference text (val.bin decoded) for distributional similarity
    dtype = np.uint16 if cfg["vocab_size"] < 65535 else np.uint32
    val_data = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=dtype).astype(np.int64)
    # Used first 20K chars as reference (fast)
    ref_text = decode(val_data[:20000].tolist(), itos)

    print(f"\n  Checkpoint : {ckpt_path}")
    ensure_dir(out_dir)
    samples_f = open(os.path.join(out_dir, f"samples_{name}.txt"), "w", encoding="utf-8")

    rows = []
    for cfg_s in CONFIGS:
        print(f" {cfg_s['name']:<22}", end="", flush=True)

        accum = {k: [] for k in SCALAR_KEYS}
        loop_onsets = []

        # Build adaptive decoder once per strategy (reset each sample)
        adaptive_dec = RecurrenceAwareDecoder(
            temperature=0.8, top_p=0.95,
            n_min=3, n_max=6,
            alpha_base=2.0, alpha_max=8.0,
            lambda_rep=10.0, lambda_ent=1.0,
            rep_target=0.05, ent_target=3.5,
            window=100,
        ) if cfg_s["adaptive"] else None

        for prompt_name, prompt_text in PROMPTS:
            for seed in range(1, n_seeds+1):
                set_seed(seed)
                idx = encode(prompt_text, stoi).to(device)

                out_ids = generate(
                    model, idx,
                    max_new_tokens   = n_chars,
                    temperature      = cfg_s["temperature"],
                    top_k            = cfg_s["top_k"],
                    top_p            = cfg_s["top_p"],
                    typical_p        = cfg_s["typical_p"],
                    rep_penalty      = cfg_s["rep_penalty"],
                    no_repeat_ngram  = cfg_s["no_repeat_ngram"],
                    mirostat_tau     = cfg_s["mirostat_tau"],
                    adaptive         = adaptive_dec,
                )[0].tolist()

                gen_ids  = out_ids[len(prompt_text):]
                text     = decode(gen_ids, itos)

                # Degeneration metrics
                lo = loop_onset(text, 10)
                loop_onsets.append(lo)

                m = {
                    "ttr"                : type_token_ratio(text),
                    "entropy_4gram"      : char_ngram_entropy(text, 4),
                    "rep_rate_5"         : repetition_rate(text, 5),
                    "rep_ngram_mass_2"   : rep_ngram_mass(text, 2),
                    "rep_ngram_mass_4"   : rep_ngram_mass(text, 4),
                    "rep_ngram_mass_6"   : rep_ngram_mass(text, 6),
                    "loop_onset"         : lo,
                    "longest_rep_sub"    : longest_repeated_substring(text),
                    # Quality metrics
                    "gen_nll_bpc"        : generated_text_nll(model, gen_ids, block_size, device),
                    "ngram_sim_4"        : ngram_distributional_similarity(text, ref_text, 4),
                    "spelling_error_rate": spelling_error_rate(text),
                    "survival_auc"       : 0.0,  # filled below
                }
                for k in SCALAR_KEYS[:-1]:  # survival_auc filled after loop
                    accum[k].append(m[k])

                if seed == 1:
                    samples_f.write(
                        f"[{cfg_s['name']}][{name}] prompt='{prompt_text.strip()}'\n"
                        + "-"*60 + "\n" + text + "\n\n"
                    )

        # Survival AUC over all loop_onsets for this strategy
        sauc = survival_auc(loop_onsets, max_t=n_chars)
        # Fill survival_auc with the same value for each sample (aggregate)
        accum["survival_auc"] = [sauc] * len(loop_onsets)

        # Kaplan-Meier curve
        km_t, km_s = kaplan_meier_survival(loop_onsets, max_t=n_chars)

        row = {
            "strategy"   : cfg_s["name"],
            "category"   : cfg_s["category"],
            "checkpoint" : name,
        }
        for k in SCALAR_KEYS:
            vals = accum[k]
            row[f"{k}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{k}_std"]  = round(float(np.std(vals)),  4)

        row["n_censored"]   = sum(1 for x in loop_onsets if x < 0)
        row["n_total"]      = len(loop_onsets)
        row["km_times"]     = km_t
        row["km_survival"]  = km_s

        rows.append(row)

        print(f"  ttr={row['ttr_mean']:.3f}  "
              f"rep5={row['rep_rate_5_mean']:.3f}  "
              f"nll={row['gen_nll_bpc_mean']:.3f}  "
              f"sim={row['ngram_sim_4_mean']:.3f}  "
              f"spell={row['spelling_error_rate_mean']:.3f}  "
              f"sauc={sauc:.3f}  "
              f"cens={row['n_censored']}/{row['n_total']}")

    samples_f.close()

    # Save CSV (without KM curve — too wide)
    csv_keys = [k for k in rows[0] if k not in ("km_times","km_survival")]
    csv_path = os.path.join(out_dir, f"metrics_{name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_keys)
        w.writeheader()
        w.writerows([{k: r[k] for k in csv_keys} for r in rows])

    # Save JSON (includes KM curves)
    json_path = os.path.join(out_dir, f"metrics_{name}.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    return rows


# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",     nargs="+",
                    default=["runs/baseline/best.pt","runs/cosine/best.pt"])
    ap.add_argument("--data_dir", default="data_out")
    ap.add_argument("--out_dir",  default="runs/sampling_eval_v3")
    ap.add_argument("--n_chars",  type=int, default=500)
    ap.add_argument("--n_seeds",  type=int, default=10)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))

    total = len(CONFIGS)*len(PROMPTS)*args.n_seeds*len(args.ckpt)
    print(f"Device: {device}  |  Strategies: {len(CONFIGS)}  |  "
          f"Total samples: {total}")

    all_results = {}
    for ckpt in args.ckpt:
        name             = os.path.basename(os.path.dirname(ckpt))
        rows             = eval_checkpoint(ckpt, args.data_dir,
                                           args.n_chars, args.n_seeds,
                                           device, args.out_dir)
        all_results[name] = rows

    # Comparison table
    if len(all_results) == 2:
        names  = list(all_results.keys())
        n0, n1 = names[0], names[1]
        r0m    = {r["strategy"]: r for r in all_results[n0]}
        r1m    = {r["strategy"]: r for r in all_results[n1]}

        print(f"\n{'='*100}")
        print(f"  COMPARISON  {n0}  vs  {n1}")
        print(f"  Metrics: TTR(↑)  RepRate(↓)  GenNLL-BPC(↓)  NgramSim(↑)  SpellErr(↓)  SurvivalAUC(↑)")
        print(f"{'='*100}")

        combined = []
        for cfg_s in CONFIGS:
            st  = cfg_s["name"]
            cat = cfg_s["category"]
            r0  = r0m.get(st); r1 = r1m.get(st)
            if not r0 or not r1: continue

            flag = "  [hard constraint]" if cat=="hard_constraint" else \
                   "  [adaptive - novel]" if cat=="adaptive" else ""

            print(f"  {st:<22} {cat:<14}"
                  f"  TTR {r0['ttr_mean']:.3f}/{r1['ttr_mean']:.3f}"
                  f"  Rep {r0['rep_rate_5_mean']:.3f}/{r1['rep_rate_5_mean']:.3f}"
                  f"  NLL {r0['gen_nll_bpc_mean']:.2f}/{r1['gen_nll_bpc_mean']:.2f}"
                  f"  Sim {r0['ngram_sim_4_mean']:.3f}/{r1['ngram_sim_4_mean']:.3f}"
                  f"  Spell {r0['spelling_error_rate_mean']:.3f}/{r1['spelling_error_rate_mean']:.3f}"
                  f"  SAUC {r0['survival_auc_mean']:.3f}/{r1['survival_auc_mean']:.3f}"
                  f"{flag}")

            combined.append({
                "strategy": st, "category": cat,
                f"ttr_{n0}":   r0["ttr_mean"],                f"ttr_{n1}":   r1["ttr_mean"],
                f"rep_{n0}":   r0["rep_rate_5_mean"],         f"rep_{n1}":   r1["rep_rate_5_mean"],
                f"nll_{n0}":   r0["gen_nll_bpc_mean"],        f"nll_{n1}":   r1["gen_nll_bpc_mean"],
                f"sim_{n0}":   r0["ngram_sim_4_mean"],        f"sim_{n1}":   r1["ngram_sim_4_mean"],
                f"spell_{n0}": r0["spelling_error_rate_mean"],f"spell_{n1}": r1["spelling_error_rate_mean"],
                f"sauc_{n0}":  r0["survival_auc_mean"],       f"sauc_{n1}":  r1["survival_auc_mean"],
            })

        print(f"\n  NOTE: no_repeat_4gram directly forbids the repetition metric "
              f"it wins on — not comparable to probabilistic methods.")
        print(f"  NOTE: adaptive decoder is the novel contribution — compare "
              f"against rep_penalty and mirostat on ALL metrics including NLL and sim.")

        comp_path = os.path.join(args.out_dir, "comparison.csv")
        with open(comp_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(combined[0].keys()))
            w.writeheader(); w.writerows(combined)

        # Save KM curves for plotting
        km_out = {}
        for name_k, rows in all_results.items():
            km_out[name_k] = {
                r["strategy"]: {
                    "km_times":    r["km_times"],
                    "km_survival": r["km_survival"],
                    "n_censored":  r["n_censored"],
                    "n_total":     r["n_total"],
                } for r in rows
            }
        km_path = os.path.join(args.out_dir, "survival_curves.json")
        with open(km_path, "w") as f:
            json.dump(km_out, f, indent=2)
        print(f"\n  Survival curves -> {km_path}")
        print(f"  Comparison      -> {comp_path}")

    full_path = os.path.join(args.out_dir, "all_results.json")
    with open(full_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Full results    -> {full_path}")


if __name__ == "__main__":
    main()