"""
generalize_gpt2.py — Generalization experiment for Recurrence-Risk Decoding.

Tests whether the method transfers beyond the 848K-parameter character model
to a PRETRAINED SUBWORD model (DistilGPT-2 by default, GPT-2 optional).

This addresses the concern that recurrence-risk decoding might be tuned to
one small character model and one loop metric. The recurrence-risk principle
is token-agnostic: with subword tokens, risk(v) is defined over subword
n-grams instead of character n-grams, and the same KL-projection penalty
  q(v) proportional to p(v) * exp(-alpha * risk(v))
applies unchanged.

Requires:
  pip install transformers torch

Usage:
  python3 generalize_gpt2.py --model distilgpt2 --n_seeds 5 --n_tokens 200 \
      --out runs/generalize/distilgpt2.json

  # also try GPT-2 small:
  python3 generalize_gpt2.py --model gpt2 --n_seeds 5 --n_tokens 200 \
      --out runs/generalize/gpt2.json

Notes:
  - Loop onset is measured in SUBWORD-TOKEN positions, and separately in
    characters after detokenisation, so the metric is not tied to one unit.
  - Hard no-repeat (subword 3-gram) is included but kept SEPARATE because it
    directly forbids the measured failure event.
  - model-consistency NLL is computed under the SAME pretrained model.
"""

import os, sys, argparse, json, math, time, zlib
import numpy as np
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---- Subword recurrence-risk decoder ----

class SubwordRecurrenceRiskDecoder:
    """
    Recurrence-Risk Decoding for subword LMs.

    Identical KL-projection penalty as the character version, but risk(v) is
    computed over subword n-grams via an incrementally maintained hash map
    (context -> set of follower token ids). One map per n in [n_min, n_max].

    adaptive=True : alpha adapts from recent repetition rate and entropy
                    (main configuration).
    adaptive=False: alpha fixed at alpha_base (risk-only ablation).
    """
    def __init__(self, temperature=0.9, top_p=0.95, n_min=2, n_max=4,
                 alpha_base=3.0, alpha_max=12.0, lambda_rep=15.0, lambda_ent=1.0,
                 rep_target=0.05, ent_target=4.0, window=64, adaptive=True):
        self.temperature = temperature; self.top_p = top_p
        self.n_min = n_min; self.n_max = n_max
        self.n_sizes = n_max - n_min + 1
        self.alpha_base = alpha_base; self.alpha_max = alpha_max
        self.lambda_rep = lambda_rep; self.lambda_ent = lambda_ent
        self.rep_target = rep_target; self.ent_target = ent_target
        self.window = window; self.adaptive = adaptive
        self.reset()

    def reset(self):
        from collections import deque
        self._recent = deque(maxlen=self.window)
        self._all = []
        self._fol = {n: defaultdict(set) for n in range(self.n_min, self.n_max+1)}

    def prime(self, ids):
        for t in ids:
            self._register(t)

    def _register(self, token):
        self._all.append(token)
        L = len(self._all)
        for n in range(self.n_min, self.n_max+1):
            if L >= n:
                ctx = tuple(self._all[L-n:L-1])
                self._fol[n][ctx].add(self._all[L-1])
        self._recent.append(token)

    def _rep_rate(self):
        s = list(self._recent)
        if len(s) < 3: return 0.0
        g = [tuple(s[i:i+3]) for i in range(len(s)-2)]
        c = Counter(g)
        return sum(v-1 for v in c.values() if v>1) / len(g)

    def _entropy(self):
        s = list(self._recent)
        if not s: return 0.0
        c = Counter(s); t = len(s)
        return -sum((v/t)*math.log2(v/t) for v in c.values())

    def _alpha(self):
        if not self.adaptive:
            return self.alpha_base
        a = (self.alpha_base
             + self.lambda_rep*max(0.0, self._rep_rate()-self.rep_target)
             - self.lambda_ent*max(0.0, self._entropy()-self.ent_target))
        return float(max(0.0, min(self.alpha_max, a)))

    def _risk(self, vocab):
        r = torch.zeros(vocab)
        L = len(self._all)
        for n in range(self.n_min, self.n_max+1):
            if L < n-1: continue
            ctx = tuple(self._all[L-(n-1):])
            fol = self._fol[n].get(ctx)
            if fol:
                inc = 1.0/self.n_sizes
                for tid in fol:
                    r[tid] += inc
        return r

    def step(self, logits):
        vocab = logits.shape[-1]
        a = self._alpha()
        risk = self._risk(vocab).to(logits.device)
        logits = logits - a*risk
        logits = logits / max(self.temperature, 1e-6)
        logits = _top_p(logits, self.top_p)
        probs = torch.softmax(logits, dim=-1)
        tok = int(torch.multinomial(probs, 1).item())
        self._register(tok)
        return tok


def _top_p(logits, p):
    if p >= 1.0: return logits
    sl, si = torch.sort(logits, descending=True)
    cum = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
    mask = cum > p; mask[0] = False
    sl = sl.masked_fill(mask, float("-inf"))
    return torch.empty_like(sl).scatter(0, si, sl)


# ---- Baseline generation ----

@torch.no_grad()
def gen_baseline(model, input_ids, n_tokens, device, mode, **kw):
    ids = input_ids.to(device)
    nr_fol = defaultdict(set) if mode == "no_repeat" else None
    n_nr = kw.get("no_repeat_n", 3)
    if nr_fol is not None:
        seq = ids[0].tolist()
        for i in range(len(seq)-(n_nr-1)):
            nr_fol[tuple(seq[i:i+n_nr-1])].add(seq[i+n_nr-1])
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        out = model(ids)
        logits = out.logits[:, -1, :].squeeze(0)
        if mode == "greedy":
            tok = int(torch.argmax(logits).item())
        else:
            T = kw.get("temperature", 1.0)
            logits = logits / T
            if mode == "rep_penalty":
                pen = kw.get("penalty", 1.3)
                for tid in set(ids[0].tolist()):
                    if logits[tid] > 0: logits[tid] /= pen
                    else: logits[tid] *= pen
            if mode == "no_repeat":
                ctx = tuple(ids[0].tolist()[-(n_nr-1):])
                for tid in nr_fol.get(ctx, ()):
                    logits[tid] = float("-inf")
            if mode in ("nucleus", "rep_penalty", "no_repeat", "temp"):
                logits = _top_p(logits, kw.get("top_p", 1.0))
            probs = torch.softmax(logits, dim=-1)
            tok = int(torch.multinomial(probs, 1).item())
        ids = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=1)
        if nr_fol is not None:
            seq = ids[0].tolist()
            if len(seq) >= n_nr:
                nr_fol[tuple(seq[-n_nr:-1])].add(seq[-1])
    cps = n_tokens / (time.perf_counter() - t0)
    return ids[0].tolist(), cps


@torch.no_grad()
def gen_risk(model, input_ids, n_tokens, device, adaptive):
    ids = input_ids.to(device)
    dec = SubwordRecurrenceRiskDecoder(adaptive=adaptive)
    dec.reset(); dec.prime(ids[0].tolist())
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        out = model(ids)
        logits = out.logits[:, -1, :].squeeze(0)
        tok = dec.step(logits)
        ids = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=1)
    cps = n_tokens / (time.perf_counter() - t0)
    return ids[0].tolist(), cps


# ---- Metrics ----

def loop_onset_tokens(ids, n=3):
    seen = {}
    for i in range(len(ids)-n+1):
        g = tuple(ids[i:i+n])
        if g in seen: return i
        seen[g] = i
    return -1

def loop_onset_chars(text, n=20):
    seen = {}
    for i in range(len(text)-n+1):
        g = text[i:i+n]
        if g in seen: return i
        seen[g] = i
    return -1

def rep_rate_tokens(ids, n=3):
    g = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    if not g: return 0.0
    c = Counter(g)
    return sum(v-1 for v in c.values() if v>1)/len(g)

def distinct_n(ids, n=3):
    g = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    if not g: return 0.0
    return len(set(g))/len(g)

def compression_ratio(text):
    if not text: return 1.0
    raw = text.encode("utf-8")
    return len(zlib.compress(raw, 9))/max(1, len(raw))

@torch.no_grad()
def mc_nll_bits(model, full_ids, prompt_len, device):
    """
    Model-consistency NLL in BITS per token, scored on the GENERATED tokens
    only (positions >= prompt_len). The prompt provides context but its tokens
    are not counted, so the value reflects the continuation the decoder
    produced, not the fixed prompt.
    """
    if len(full_ids) < prompt_len + 2:
        return float("nan")
    x = torch.tensor([full_ids[:-1]], device=device)
    y = torch.tensor([full_ids[1:]], device=device)
    out = model(x)
    lp = F.log_softmax(out.logits, dim=-1)
    tok_nll = -lp.gather(2, y.unsqueeze(-1)).squeeze(-1)[0]    # nats per position
    gen_nll = tok_nll[prompt_len - 1:]                        # generated only
    if gen_nll.numel() == 0:
        return float("nan")
    return float(gen_nll.mean().item() / math.log(2))          # nats -> bits


def survival_auc(onsets, max_t):
    events = sorted([t for t in onsets if t >= 0])
    n = len(onsets)
    if not events: return 1.0
    S, prev_t, area, before = 1.0, 0, 0.0, 0
    for t in sorted(set(events)):
        nev = events.count(t); risk = n - before
        area += (t - prev_t) * S
        if risk > 0: S *= (1 - nev/risk)
        before += nev; prev_t = t
    area += (max_t - prev_t) * S
    return round(area/max_t, 4)

def loop_rate(onsets):
    return round(sum(1 for t in onsets if t>=0)/len(onsets), 4)


PROMPTS = [
    "The history of the city begins",
    "She opened the door and",
    "In recent years, scientists have",
    "The most important thing to remember is",
    "Once upon a time there was",
    "The economy of the region depends on",
    "He looked at the sky and thought",
    "According to the report, the",
]

METHODS = [
    ("greedy",          dict(mode="greedy")),
    ("temp_0.9",        dict(mode="temp", temperature=0.9, top_p=1.0)),
    ("nucleus_p0.95",   dict(mode="nucleus", temperature=1.0, top_p=0.95)),
    ("rep_penalty_1.3", dict(mode="rep_penalty", temperature=0.9, penalty=1.3, top_p=1.0)),
    ("no_repeat_3gram", dict(mode="no_repeat", temperature=0.9, no_repeat_n=3, top_p=1.0)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2", help="distilgpt2 or gpt2")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--n_tokens", type=int, default=200)
    ap.add_argument("--out", default="runs/generalize/distilgpt2.json")
    args = ap.parse_args()

    device = (torch.device("mps") if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Model: {args.model}  Device: {device}")
    print(f"Seeds: {args.n_seeds}  Tokens: {args.n_tokens}  Prompts: {len(PROMPTS)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    max_t_tok = args.n_tokens
    results = {}

    def run(name, genfn):
        ot, oc = [], []
        rr, dn, comp, nll = [], [], [], []
        gen_char_lens = []
        cps_v = None
        for p in PROMPTS:
            enc = tok(p, return_tensors="pt").input_ids
            prompt_len = enc.shape[1]
            for seed in range(1, args.n_seeds+1):
                torch.manual_seed(seed); np.random.seed(seed)
                ids, cps = genfn(enc)
                cps_v = cps
                gen = ids[prompt_len:]
                text = tok.decode(gen)
                gen_char_lens.append(len(text))
                ot.append(loop_onset_tokens(gen, 3))
                oc.append(loop_onset_chars(text, 20))
                rr.append(rep_rate_tokens(gen, 3))
                dn.append(distinct_n(gen, 3))
                comp.append(compression_ratio(text))
                nll.append(mc_nll_bits(model, ids, prompt_len, device))
        # Character horizon: shortest generated character length, the common
        # window every sample reaches. Any loop onsetting beyond it is censored.
        # (Fixes SAUC>1: char onsets were previously normalised by the TOKEN
        #  horizon, ~4x smaller than the character lengths.)
        max_t_char = int(min(gen_char_lens)) if gen_char_lens else (max_t_tok * 4)
        oc_censored = [o if (0 <= o < max_t_char) else -1 for o in oc]
        return {
            "loop_rate_tok3": loop_rate(ot),
            "survival_auc_tok3": survival_auc(ot, max_t_tok),
            "loop_rate_char20": loop_rate(oc_censored),
            "survival_auc_char20": survival_auc(oc_censored, max_t_char),
            "char_horizon": max_t_char,
            "rep_rate_tok3": round(float(np.mean(rr)), 4),
            "distinct_3": round(float(np.mean(dn)), 4),
            "compression": round(float(np.mean(comp)), 4),
            "mc_nll_bits_gen": round(float(np.nanmean(nll)), 4),
            "chars_per_sec": round(cps_v, 1) if cps_v else None,
            "n_samples": len(ot),
        }

    for name, kw in METHODS:
        print(f"  {name:<20}", end="", flush=True)
        r = run(name, lambda enc, kw=kw: gen_baseline(model, enc, args.n_tokens, device, **kw))
        r["category"] = "hard_constraint" if "no_repeat" in name else "baseline"
        results[name] = r
        print(f"  loop_tok={r['loop_rate_tok3']:.2f}  SAUC={r['survival_auc_tok3']:.3f}  "
              f"rep={r['rep_rate_tok3']:.3f}  mcnll={r['mc_nll_bits_gen']:.3f}")

    for name, adaptive in [("risk_only", False), ("adaptive", True)]:
        print(f"  {name:<20}", end="", flush=True)
        r = run(name, lambda enc, a=adaptive: gen_risk(model, enc, args.n_tokens, device, a))
        r["category"] = "recurrence_risk"
        results[name] = r
        print(f"  loop_tok={r['loop_rate_tok3']:.2f}  SAUC={r['survival_auc_tok3']:.3f}  "
              f"rep={r['rep_rate_tok3']:.3f}  mcnll={r['mc_nll_bits_gen']:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "n_seeds": args.n_seeds,
                   "n_tokens": args.n_tokens, "results": results}, f, indent=2)
    print(f"\n  Saved -> {args.out}")
    print("\n  NOTE: no_repeat_3gram directly forbids the token-3gram failure event;")
    print("  keep it separate. Compare adaptive/risk_only against the soft baselines.")


if __name__ == "__main__":
    main()

    