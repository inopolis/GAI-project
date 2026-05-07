"""
sampling_eval.py — Rigorous decoding strategy evaluation across both checkpoints.

Runs all strategies on BOTH baseline and cosine checkpoints so results
can be directly compared. Outputs per-checkpoint CSVs and a combined
comparison table.

Key fixes vs original:
  - Both checkpoints evaluated (not just one)
  - 10 seeds x 5 prompts = 50 samples per strategy (was 3)
  - Mean +/- std reported for every metric
  - Stronger degeneration metrics: loop_onset, longest_rep_sub,
    rep_ngram_mass_2/4/6, entropy_trajectory
  - New strategies: typical_p, rep_penalty, no_repeat_ngram, mirostat
  - no_repeat_ngram is flagged as a hard constraint, not a fair
    probabilistic comparison (it directly forbids the metric it wins on)

Usage:
  python3 sampling_eval.py \
      --ckpt runs/baseline/best.pt runs/cosine/best.pt \
      --out_dir runs/sampling_eval_v2 \
      --n_seeds 10 --n_chars 500
"""

import os
import sys
import argparse
import csv
import json
import math
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from src.utils import set_seed, load_json, ensure_dir
from src.model import CharTransformerLM
from src.decoding import generate


# Decoding config
CONFIGS = [
    # --- Original strategies ---
    {"name": "greedy",
     "temperature":0.0, "top_k":0, "top_p":1.0,
     "typical_p":1.0, "rep_penalty":1.0, "no_repeat_ngram":0, "mirostat_tau":0.0,
     "category": "baseline"},
    {"name": "temp_0.6",
     "temperature":0.6, "top_k":0, "top_p":1.0,
     "typical_p":1.0, "rep_penalty":1.0, "no_repeat_ngram":0, "mirostat_tau":0.0,
     "category": "baseline"},
    {"name": "temp_0.8",
     "temperature":0.8, "top_k":0, "top_p":1.0,
     "typical_p":1.0, "rep_penalty":1.0, "no_repeat_ngram":0, "mirostat_tau":0.0,
     "category": "baseline"},
    {"name": "temp_1.0",
     "temperature":1.0, "top_k":0, "top_p":1.0,
     "typical_p":1.0, "rep_penalty":1.0, "no_repeat_ngram":0, "mirostat_tau":0.0,
     "category": "baseline"},
    {"name": "top_k_10",
     "temperature":1.0, "top_k":10, "top_p":1.0,
     "typical_p":1.0, "rep_penalty":1.0, "no_repeat_ngram":0, "mirostat_tau":0.0,
     "category": "baseline"},
    {"name": "nucleus_p0.95",
     "temperature":1.0, "top_k":0, "top_p":0.95,
     "typical_p":1.0, "rep_penalty":1.0, "no_repeat_ngram":0, "mirostat_tau":0.0,
     "category": "baseline"},

    # --- Stronger strategies (new) ---
    {"name": "typical_p0.9",
     "temperature":1.0, "top_k":0, "top_p":1.0,
     "typical_p":0.9, "rep_penalty":1.0, "no_repeat_ngram":0, "mirostat_tau":0.0,
     "category": "probabilistic"},
    {"name": "rep_penalty_1.3",
     "temperature":0.8, "top_k":0, "top_p":1.0,
     "typical_p":1.0, "rep_penalty":1.3, "no_repeat_ngram":0, "mirostat_tau":0.0,
     "category": "probabilistic"},
    {"name": "mirostat_tau3",
     "temperature":1.0, "top_k":0, "top_p":1.0,
     "typical_p":1.0, "rep_penalty":1.0, "no_repeat_ngram":0, "mirostat_tau":3.0,
     "category": "probabilistic"},
    {"name": "mirostat_tau5",
     "temperature":1.0, "top_k":0, "top_p":1.0,
     "typical_p":1.0, "rep_penalty":1.0, "no_repeat_ngram":0, "mirostat_tau":5.0,
     "category": "probabilistic"},

    # --- Hard constraint (reported separately — not a fair comparison) ---
    # NOTE: no_repeat_ngram directly forbids the metric it wins on (repetition
    # rate). It mechanically eliminates repeated n-grams regardless of model
    # probability, so it trivially achieves rep_rate=0. This is not a
    # probabilistic improvement and should not be ranked against other methods.
    {"name": "no_repeat_4gram",
     "temperature":0.8, "top_k":0, "top_p":1.0,
     "typical_p":1.0, "rep_penalty":1.0, "no_repeat_ngram":4, "mirostat_tau":0.0,
     "category": "hard_constraint"},
]

PROMPTS = [
    ("chapter",  "CHAPTER 1\n"),
    ("night",    "The night was "),
    ("she",      "She had never "),
    ("best",     "It was the best of "),
    ("darcy",    "Mr. Darcy had never "),
]


#Metrics

def type_token_ratio(text):
    words = text.split()
    return len(set(words)) / len(words) if words else 0.0

def char_ngram_entropy(text, n=4):
    grams = [text[i:i+n] for i in range(len(text)-n+1)]
    if not grams:
        return 0.0
    c = Counter(grams)
    t = sum(c.values())
    return -sum((v/t)*math.log2(v/t) for v in c.values())

def repetition_rate(text, n=5):
    grams = [text[i:i+n] for i in range(len(text)-n+1)]
    if not grams:
        return 0.0
    c = Counter(grams)
    return sum(v-1 for v in c.values() if v > 1) / len(grams)

def rep_ngram_mass(text, n):
    grams = [text[i:i+n] for i in range(len(text)-n+1)]
    if not grams:
        return 0.0
    c = Counter(grams)
    return sum(v for v in c.values() if v > 1) / len(grams)

def loop_onset(text, n=10):
    seen = {}
    for i in range(len(text)-n+1):
        g = text[i:i+n]
        if g in seen:
            return i
        seen[g] = i
    return -1

def longest_repeated_substring(text, min_len=5):
    n = len(text)
    if n < min_len * 2:
        return 0
    def has_rep(L):
        seen = set()
        for i in range(n-L+1):
            s = text[i:i+L]
            if s in seen:
                return True
            seen.add(s)
        return False
    lo, hi, res = min_len, min(n//2, 300), 0
    while lo <= hi:
        mid = (lo+hi)//2
        if has_rep(mid):
            res = mid; lo = mid+1
        else:
            hi = mid-1
    return res

def entropy_trajectory(text, n_windows=4, ngram_size=4):
    L = len(text)
    w = L // n_windows
    if w < ngram_size+1:
        return [0.0]*n_windows
    return [char_ngram_entropy(text[i*w:(i+1)*w], ngram_size)
            for i in range(n_windows)]

def all_metrics(text):
    return {
        "ttr":              type_token_ratio(text),
        "entropy_4gram":    char_ngram_entropy(text, 4),
        "rep_rate_5":       repetition_rate(text, 5),
        "rep_ngram_mass_2": rep_ngram_mass(text, 2),
        "rep_ngram_mass_4": rep_ngram_mass(text, 4),
        "rep_ngram_mass_6": rep_ngram_mass(text, 6),
        "loop_onset":       loop_onset(text, 10),
        "longest_rep_sub":  longest_repeated_substring(text),
        "entropy_traj":     entropy_trajectory(text, 4),
    }


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


# Run eval for one checkpoint

SCALAR_KEYS = ["ttr","entropy_4gram","rep_rate_5",
               "rep_ngram_mass_2","rep_ngram_mass_4","rep_ngram_mass_6",
               "loop_onset","longest_rep_sub"]

def eval_checkpoint(ckpt_path, data_dir, n_chars, n_seeds, device, out_dir):
    model, cfg = load_model(ckpt_path, device)
    vocab      = load_json(os.path.join(data_dir, "vocab.json"))
    stoi, itos = vocab["stoi"], vocab["itos"]
    name       = os.path.basename(os.path.dirname(ckpt_path))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Checkpoint : {ckpt_path}  ({n_params:,} params)")

    rows      = []
    samples_f = open(os.path.join(out_dir, f"samples_{name}.txt"), "w", encoding="utf-8")

    for cfg_s in CONFIGS:
        print(f"    {cfg_s['name']:<22}", end="", flush=True)
        accum = {k: [] for k in SCALAR_KEYS}
        traj  = [[] for _ in range(4)]

        for prompt_name, prompt_text in PROMPTS:
            for seed in range(1, n_seeds+1):
                set_seed(seed)
                idx = encode(prompt_text, stoi).to(device)
                out = generate(
                    model, idx,
                    max_new_tokens   = n_chars,
                    temperature      = cfg_s["temperature"],
                    top_k            = cfg_s["top_k"],
                    top_p            = cfg_s["top_p"],
                    typical_p        = cfg_s["typical_p"],
                    rep_penalty      = cfg_s["rep_penalty"],
                    no_repeat_ngram  = cfg_s["no_repeat_ngram"],
                    mirostat_tau     = cfg_s["mirostat_tau"],
                )[0].tolist()

                text = decode(out, itos)[len(prompt_text):]
                m    = all_metrics(text)
                for k in SCALAR_KEYS:
                    accum[k].append(m[k])
                for i, v in enumerate(m["entropy_traj"]):
                    traj[i].append(v)

                if seed == 1:
                    samples_f.write(f"[{cfg_s['name']}] [{name}] prompt='{prompt_text.strip()}'\n")
                    samples_f.write("-"*60 + "\n")
                    samples_f.write(text + "\n\n")

        row = {"strategy":cfg_s["name"], "category":cfg_s["category"], "checkpoint":name}
        for k in SCALAR_KEYS:
            row[f"{k}_mean"] = round(float(np.mean(accum[k])), 4)
            row[f"{k}_std"]  = round(float(np.std(accum[k])),  4)
        for i in range(4):
            row[f"entropy_traj_{i}_mean"] = round(float(np.mean(traj[i])), 4)
        rows.append(row)

        # inline progress
        print(f"  ttr={row['ttr_mean']:.3f}+-{row['ttr_std']:.2f}"
              f"  rep5={row['rep_rate_5_mean']:.3f}+-{row['rep_rate_5_std']:.2f}"
              f"  loop={row['loop_onset_mean']:.0f}")

    samples_f.close()

    # Save per-checkpoint CSV
    csv_path = os.path.join(out_dir, f"metrics_{name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows


#Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",      nargs="+",
                    default=["runs/baseline/best.pt", "runs/cosine/best.pt"])
    ap.add_argument("--data_dir",  default="data_out")
    ap.add_argument("--out_dir",   default="runs/sampling_eval_v2")
    ap.add_argument("--n_chars",   type=int, default=500)
    ap.add_argument("--n_seeds",   type=int, default=10)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Device   : {device}")
    print(f"Seeds    : {args.n_seeds}  |  Chars/sample : {args.n_chars}")
    print(f"Ckpts    : {args.ckpt}")
    total = len(CONFIGS) * len(PROMPTS) * args.n_seeds * len(args.ckpt)
    print(f"Total samples: {total}\n")

    all_rows = {}
    for ckpt in args.ckpt:
        name          = os.path.basename(os.path.dirname(ckpt))
        rows          = eval_checkpoint(ckpt, args.data_dir,
                                        args.n_chars, args.n_seeds, device,
                                        args.out_dir)
        all_rows[name] = rows

    # Combined comparison table
    if len(all_rows) == 2:
        names  = list(all_rows.keys())
        n0, n1 = names[0], names[1]
        rows0  = {r["strategy"]: r for r in all_rows[n0]}
        rows1  = {r["strategy"]: r for r in all_rows[n1]}

        print(f"\n{'='*90}")
        print(f"  COMPARISON: {n0}  vs  {n1}")
        print(f"  (mean +/- std across {args.n_seeds} seeds x {len(PROMPTS)} prompts)")
        print(f"{'='*90}")

        hdr = f"  {'Strategy':<22} {'Cat':<14}"
        hdr += f" {'TTR':>12}{'':>12} {'RepRate':>12}{'':>12} {'LoopOnset':>10}{'':>10}"
        print(hdr)
        print(f"  {'':22} {'':14}"
              f" {n0:>12} {n1:>11}"
              f" {n0:>12} {n1:>11}"
              f" {n0:>10} {n1:>9}")
        print("  " + "-"*88)

        combined = []
        for strat in [c["name"] for c in CONFIGS]:
            r0 = rows0.get(strat)
            r1 = rows1.get(strat)
            if not r0 or not r1:
                continue
            cat  = r0["category"]
            note = "  [hard constraint — not comparable]" if cat == "hard_constraint" else ""
            line = (f"  {strat:<22} {cat:<14}"
                    f" {r0['ttr_mean']:>6.3f}+-{r0['ttr_std']:.2f}"
                    f" {r1['ttr_mean']:>6.3f}+-{r1['ttr_std']:.2f}"
                    f" {r0['rep_rate_5_mean']:>6.3f}+-{r0['rep_rate_5_std']:.2f}"
                    f" {r1['rep_rate_5_mean']:>6.3f}+-{r1['rep_rate_5_std']:.2f}"
                    f" {r0['loop_onset_mean']:>9.0f}"
                    f" {r1['loop_onset_mean']:>9.0f}"
                    f"{note}")
            print(line)
            combined.append({
                "strategy"     : strat,
                "category"     : cat,
                f"ttr_{n0}"    : r0["ttr_mean"],
                f"ttr_{n1}"    : r1["ttr_mean"],
                f"rep_{n0}"    : r0["rep_rate_5_mean"],
                f"rep_{n1}"    : r1["rep_rate_5_mean"],
                f"loop_{n0}"   : r0["loop_onset_mean"],
                f"loop_{n1}"   : r1["loop_onset_mean"],
            })

        print(f"\n  NOTE: no_repeat_4gram (hard_constraint) directly forbids the")
        print(f"  metric it wins on. Its rep_rate=0 is mechanically guaranteed,")
        print(f"  not a property of generation quality. Do not rank it with")
        print(f"  probabilistic methods.")

        # Save combined CSV
        comp_path = os.path.join(args.out_dir, "metrics_comparison.csv")
        with open(comp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(combined[0].keys()))
            writer.writeheader()
            writer.writerows(combined)
        print(f"\n  Comparison saved -> {comp_path}")

    # Save combined JSON
    json_path = os.path.join(args.out_dir, "all_metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"  Full results  -> {json_path}")


if __name__ == "__main__":
    main()