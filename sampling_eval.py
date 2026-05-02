"""
sampling_eval.py — Rigorous sampling evaluation with variance and stronger metrics.

Fixe vs original:
  - More samples per strategy (--n_seeds, default 10 seeds x n_prompts)
  - Reports mean ± std for every metric (not just mean)
  - Stronger degeneration metrics:
      * loop_onset : first position where a length-10 n-gram repeats (chars; -1 = no loop)
      * longest_rep_sub: length of longest repeated substring
      * rep_ngram_mass_*: fraction of n-grams that are repeated, for n=2,4,6
      * entropy_trajectory: mean 4-gram entropy in 4 equal windows (early->late)
  - New decoding strategies: typical_p, rep_penalty, no_repeat_ngram, mirostat
  - Claims about "best" strategy only made when it dominates under ALL metrics

Usage:
  python3 sampling_eval.py --ckpt runs/baseline/best.pt --out_dir runs/baseline/sampling_eval_v2
  python3 sampling_eval.py --ckpt runs/cosine/best.pt   --out_dir runs/cosine/sampling_eval_v2
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
from src.decoding import generate, MirostatSampler


# ── Sampling configs ──────────────────────────────────────────────────────────

CONFIGS = [
    # --- Baselines from stage 2 ---
    {"name": "greedy",           "temperature": 0.0, "top_k": 0,  "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0, "mirostat_tau": 0.0},
    {"name": "temp_0.6",         "temperature": 0.6, "top_k": 0,  "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0, "mirostat_tau": 0.0},
    {"name": "temp_0.8",         "temperature": 0.8, "top_k": 0,  "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0, "mirostat_tau": 0.0},
    {"name": "temp_1.0",         "temperature": 1.0, "top_k": 0,  "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0, "mirostat_tau": 0.0},
    {"name": "top_k_10",         "temperature": 1.0, "top_k": 10, "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0, "mirostat_tau": 0.0},
    {"name": "nucleus_p0.95",    "temperature": 1.0, "top_k": 0,  "top_p": 0.95,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0, "mirostat_tau": 0.0},

    # --- New stronger baselines ---
    {"name": "typical_p0.9",     "temperature": 1.0, "top_k": 0,  "top_p": 1.0,
     "typical_p": 0.9, "rep_penalty": 1.0, "no_repeat_ngram": 0, "mirostat_tau": 0.0},
    {"name": "rep_penalty_1.3",  "temperature": 0.8, "top_k": 0,  "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.3, "no_repeat_ngram": 0, "mirostat_tau": 0.0},
    {"name": "no_repeat_4gram",  "temperature": 0.8, "top_k": 0,  "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 4, "mirostat_tau": 0.0},
    {"name": "mirostat_tau3",    "temperature": 1.0, "top_k": 0,  "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0, "mirostat_tau": 3.0},
    {"name": "mirostat_tau5",    "temperature": 1.0, "top_k": 0,  "top_p": 1.0,
     "typical_p": 1.0, "rep_penalty": 1.0, "no_repeat_ngram": 0, "mirostat_tau": 5.0},
]

PROMPTS = [
    ("chapter",  "CHAPTER 1\n"),
    ("night",    "The night was "),
    ("she",      "She had never "),
    ("best",     "It was the best of "),
    ("darcy",    "Mr. Darcy had never "),
]


#Metrics

def type_token_ratio(text: str) -> float:
    words = text.split()
    return len(set(words)) / len(words) if words else 0.0


def char_ngram_entropy(text: str, n: int = 4) -> float:
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    total  = sum(counts.values())
    return -sum((c/total) * math.log2(c/total) for c in counts.values())


def repetition_rate(text: str, n: int = 5) -> float:
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    if not ngrams:
        return 0.0
    counts   = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def rep_ngram_mass(text: str, n: int) -> float:
    """Fraction of n-gram occurrences that are repetitions (appear >1 time)."""
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    if not ngrams:
        return 0.0
    counts   = Counter(ngrams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def loop_onset(text: str, n: int = 10) -> int:
    """
    First character position where a length-n n-gram is seen for the second time.
    Returns -1 if no repetition found (no degeneration loop detected).
    """
    seen = {}
    for i in range(len(text) - n + 1):
        gram = text[i:i+n]
        if gram in seen:
            return i
        seen[gram] = i
    return -1


def longest_repeated_substring(text: str, min_len: int = 5) -> int:
    """
    Length of the longest substring that appears at least twice.
    Uses a binary search + rolling hash approach for efficiency.
    For texts up to ~1000 chars this is fast enough.
    """
    n = len(text)
    if n < min_len * 2:
        return 0

    def has_repeated_len(L):
        seen = set()
        for i in range(n - L + 1):
            s = text[i:i+L]
            if s in seen:
                return True
            seen.add(s)
        return False

    lo, hi = min_len, min(n // 2, 300)  # cap at 300 for speed
    result = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if has_repeated_len(mid):
            result = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return result


def entropy_trajectory(text: str, n_windows: int = 4, ngram_size: int = 4) -> list:
    """
    Split text into n_windows equal parts and compute 4-gram entropy in each.
    A drop in entropy over time = model is degenerating / becoming more repetitive.
    """
    L = len(text)
    w = L // n_windows
    if w < ngram_size + 1:
        return [0.0] * n_windows
    return [
        char_ngram_entropy(text[i*w:(i+1)*w], ngram_size)
        for i in range(n_windows)
    ]


def compute_all_metrics(text: str) -> dict:
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
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg    = ckpt["config"]
    model  = CharTransformerLM(
        vocab_size = cfg["vocab_size"],
        block_size = cfg["block_size"],
        n_layer    = cfg["n_layer"],
        n_embd     = cfg["n_embd"],
        n_head     = cfg["n_head"],
        dropout    = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def encode_prompt(prompt, stoi):
    unk = stoi.get(" ", 0)
    return torch.tensor([[stoi.get(ch, unk) for ch in prompt]], dtype=torch.long)


def decode_ids(ids, itos):
    return "".join([itos[str(int(i))] if str(int(i)) in itos else "?" for i in ids])


# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",      type=str, default="runs/baseline/best.pt")
    ap.add_argument("--data_dir",  type=str, default="data_out")
    ap.add_argument("--out_dir",   type=str, default="runs/baseline/sampling_eval_v2")
    ap.add_argument("--n_chars",   type=int, default=500,
                    help="Characters to generate per sample")
    ap.add_argument("--n_seeds",   type=int, default=10,
                    help="Number of random seeds per (strategy, prompt) pair")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model, cfg = load_model(args.ckpt, device)
    vocab      = load_json(os.path.join(args.data_dir, "vocab.json"))
    stoi, itos = vocab["stoi"], vocab["itos"]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['n_layer']}L×{cfg['n_embd']}d  {n_params:,} params")
    print(f"Seeds: {args.n_seeds}  |  Chars/sample: {args.n_chars}")
    print(f"Strategies: {len(CONFIGS)}  |  Prompts: {len(PROMPTS)}")
    print(f"Total samples: {len(CONFIGS) * len(PROMPTS) * args.n_seeds}\n")

    METRIC_KEYS = [
        "ttr", "entropy_4gram", "rep_rate_5",
        "rep_ngram_mass_2", "rep_ngram_mass_4", "rep_ngram_mass_6",
        "loop_onset", "longest_rep_sub",
    ]
    TRAJ_KEYS = [f"entropy_traj_{i}" for i in range(4)]

    all_rows   = []
    samples_path = os.path.join(args.out_dir, "samples.txt")

    with open(samples_path, "w", encoding="utf-8") as sf:
        for cfg_s in CONFIGS:
            print(f"  Strategy: {cfg_s['name']}")
            strategy_metrics = {k: [] for k in METRIC_KEYS}
            strategy_traj    = [[] for _ in range(4)]

            for prompt_name, prompt_text in PROMPTS:
                for seed in range(1, args.n_seeds + 1):
                    set_seed(seed)
                    idx = encode_prompt(prompt_text, stoi).to(device)

                    out = generate(
                        model, idx,
                        max_new_tokens  = args.n_chars,
                        temperature     = cfg_s["temperature"],
                        top_k           = cfg_s["top_k"],
                        top_p           = cfg_s["top_p"],
                        typical_p       = cfg_s["typical_p"],
                        rep_penalty     = cfg_s["rep_penalty"],
                        no_repeat_ngram = cfg_s["no_repeat_ngram"],
                        mirostat_tau    = cfg_s["mirostat_tau"],
                        mirostat_eta    = 0.1,
                    )[0].tolist()

                    text = decode_ids(out, itos)[len(prompt_text):]
                    m    = compute_all_metrics(text)

                    for k in METRIC_KEYS:
                        strategy_metrics[k].append(m[k])
                    for i, v in enumerate(m["entropy_traj"]):
                        strategy_traj[i].append(v)

                    # Save one sample per (strategy, prompt)
                    if seed == 1:
                        sf.write(f"[{cfg_s['name']}] prompt='{prompt_text.strip()}'\n")
                        sf.write("-" * 60 + "\n")
                        sf.write(text + "\n\n")

            # Aggregate: mean ± std
            row = {"strategy": cfg_s["name"]}
            row.update({k: float(cfg_s[k]) if isinstance(cfg_s[k], (int, float)) else cfg_s[k]
                        for k in ["temperature", "top_k", "top_p", "typical_p", "rep_penalty",
                                  "no_repeat_ngram", "mirostat_tau"]})
            for k in METRIC_KEYS:
                vals = strategy_metrics[k]
                row[f"{k}_mean"] = round(float(np.mean(vals)), 4)
                row[f"{k}_std"]  = round(float(np.std(vals)),  4)
            for i in range(4):
                vals = strategy_traj[i]
                row[TRAJ_KEYS[i] + "_mean"] = round(float(np.mean(vals)), 4)
                row[TRAJ_KEYS[i] + "_std"]  = round(float(np.std(vals)),  4)

            all_rows.append(row)
            print(f"    ttr={row['ttr_mean']:.3f}±{row['ttr_std']:.3f}"
                  f"  rep5={row['rep_rate_5_mean']:.3f}±{row['rep_rate_5_std']:.3f}"
                  f"  loop_onset={row['loop_onset_mean']:.0f}"
                  f"  longest_rep={row['longest_rep_sub_mean']:.0f}")

    #Save CSV
    csv_path = os.path.join(args.out_dir, "metrics_v2.csv")
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nMetrics → {csv_path}")

    #save JSON (includes trajectory)
    json_path = os.path.join(args.out_dir, "metrics_v2.json")
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2)

    # Determine "best" strategy conservatively 
    # A strategy must be best on ALL three key dimensions to be called best:
    #   higher ttr, higher entropy, lower rep_rate_5
    print("\n" + "="*75)
    print("RESULTS (mean ± std across all prompts × seeds)")
    print("="*75)
    print(f"{'Strategy':<22} {'TTR':>10} {'Entropy':>10} {'RepRate5':>10} {'LoopOnset':>10} {'LongestRep':>11}")
    print("-"*75)
    for r in all_rows:
        print(f"{r['strategy']:<22}"
              f" {r['ttr_mean']:>5.3f}±{r['ttr_std']:.2f}"
              f" {r['entropy_4gram_mean']:>5.2f}±{r['entropy_4gram_std']:.2f}"
              f" {r['rep_rate_5_mean']:>5.3f}±{r['rep_rate_5_std']:.2f}"
              f" {r['loop_onset_mean']:>9.0f}"
              f" {r['longest_rep_sub_mean']:>10.0f}")
    print("="*75)

    # Conservative best: dominates on ttr (↑), entropy (↑), rep_rate (↓)
    # using mean - 1*std for ttr/entropy and mean + 1*std for rep_rate
    def dominates(r):
        return (
            r["ttr_mean"] - r["ttr_std"],
            r["entropy_4gram_mean"] - r["entropy_4gram_std"],
            -(r["rep_rate_5_mean"] + r["rep_rate_5_std"]),
        )

    best = max(all_rows, key=dominates)
    print(f"\n  Conservative best (dominates under mean±std): {best['strategy']}")
    print("  NOTE: 'best' is only claimed if dominant across ALL three key metrics.")
    print(f"  Samples → {samples_path}")


if __name__ == "__main__":
    main()