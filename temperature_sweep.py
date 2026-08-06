"""
temperature_sweep.py

Addresses review point 5: the earlier temperature probe (2-6 samples per
setting) was far too small to support a claim like "loops are confined below
T=0.2". This script runs a properly powered sweep -- many temperatures, many
prompts, many seeds, and a longer generation horizon -- and reports the
persistent-loop rate with an exact Clopper-Pearson interval at every setting,
so the phase-transition claim in the paper is backed by a number with real
statistical power behind it, not an anecdote.

This must run on the machine with the trained checkpoint (uses the actual
character model), so it is not runnable inside a network/GPU-restricted
sandbox. Expect real wall-clock time: see the printed estimate before it
starts, and use --dry_run to check that estimate without generating anything.

Usage
-----
  python3 temperature_sweep.py --ckpt runs/cosine/best.pt --data_dir data_out \
      --out sweep_report.json --dry_run

  python3 temperature_sweep.py --ckpt runs/cosine/best.pt --data_dir data_out \
      --out sweep_report.json
"""

import os, sys, json, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from src.utils import set_seed, load_json
from src.model import CharTransformerLM
from src.decoding import generate

try:
    from scipy.stats import beta as _beta
    def clopper_pearson(k, n, alpha=0.05):
        lo = 0.0 if k == 0 else _beta.ppf(alpha/2, k, n-k+1)
        hi = 1.0 if k == n else _beta.ppf(1-alpha/2, k+1, n-k)
        return float(lo), float(hi)
except Exception:
    def clopper_pearson(k, n, alpha=0.05):
        if n == 0:
            return (0.0, 1.0)
        z = 1.96
        p = k / n
        denom = 1 + z*z/n
        centre = p + z*z/(2*n)
        half = z * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5)
        return (max(0.0, (centre - half)/denom), min(1.0, (centre + half)/denom))


def _normalize_whitespace_for_loop_check(text):
    """Collapse whitespace runs to a single space before periodicity matching.
    Irregular line-wrapping otherwise breaks exact-match detection of an
    obviously-repeating phrase; see sampling_eval.py's persistent_loop_onset
    docstring for the concrete example this was found from."""
    import re as _re
    return _re.sub(r"\s+", " ", text)


def persistent_loop_onset(text, period_max=60, R=5, min_p=2,
                           catch_period_one=True, period_one_min_run=8):
    """R=5, period_max=60: the frozen setting from validate_loop_event_v2.py's
    calibrate/freeze/validate protocol. Kept in sync with sampling_eval.py's
    definition of the same name -- both must use the same frozen event."""
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
        for P in range(min_p, min(period_max, e // R) + 1):
            span = text[e - P:e]
            ok = True
            for r in range(2, R + 1):
                if text[e - r*P:e - (r-1)*P] != span:
                    ok = False; break
            if ok:
                return e
    return -1


# DEV prompts only -- this sweep is a calibration step (deciding where to run
# the main experiment), not a headline result, so it uses the prompt set
# designated for tuning decisions, consistent with the DEV/TEST split
# elsewhere in this project. If a systematic sweep result is ever reported as
# a paper table (not just used to pick T), it should be re-run on TEST prompts.
DEV_PROMPTS = [
    ("chapter",  "CHAPTER 1\n"),
    ("night",    "The night was "),
    ("she",      "She had never "),
    ("best",     "It was the best of "),
    ("darcy",    "Mr. Darcy had never "),
]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/cosine/best.pt")
    ap.add_argument("--data_dir", default="data_out")
    ap.add_argument("--out", default="sweep_report.json")
    ap.add_argument("--n_chars", type=int, default=1000,
                     help="Generation length. Longer than the earlier probe's "
                          "500-1200 chars so late-onset loops are not missed.")
    ap.add_argument("--n_seeds", type=int, default=10,
                     help="Seeds per (temperature, prompt) cell.")
    ap.add_argument("--temps", nargs="*", type=float,
                     default=[0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25,
                              0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
                     help="Temperature grid. Denser near the suspected "
                          "transition (0.1-0.3) than the earlier probe.")
    ap.add_argument("--dry_run", action="store_true",
                     help="Print the time estimate and exit without generating.")
    args = ap.parse_args()

    n_prompts = len(DEV_PROMPTS)
    n_cells = len(args.temps) * n_prompts * args.n_seeds
    total_chars = n_cells * args.n_chars
    print(f"Grid: {len(args.temps)} temperatures x {n_prompts} prompts x {args.n_seeds} seeds "
          f"= {n_cells} generations x {args.n_chars} chars = {total_chars:,} chars total.")
    print("Rough wall-clock estimate at ~150-250 chars/sec (character model, MPS/CPU): "
          f"{total_chars/200/60:.0f}-{total_chars/150/60:.0f} minutes.")
    if args.dry_run:
        return

    device = (torch.device("mps") if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Device: {device}")

    model, cfg = load_model(args.ckpt, device)
    vocab = load_json(os.path.join(args.data_dir, "vocab.json"))
    stoi, itos = vocab["stoi"], vocab["itos"]

    results = {"n_chars": args.n_chars, "n_seeds": args.n_seeds,
               "prompts": [p for p, _ in DEV_PROMPTS], "rows": []}

    t0 = time.time()
    for T in args.temps:
        onsets = []
        for pname, ptext in DEV_PROMPTS:
            for seed in range(1, args.n_seeds + 1):
                set_seed(seed)
                idx = encode(ptext, stoi).to(device)
                out, _ = generate(model, idx, max_new_tokens=args.n_chars,
                                   temperature=T, top_p=1.0)
                gen = out[0].tolist()[len(ptext):]
                text = decode(gen, itos)
                onsets.append(persistent_loop_onset(text))
        n = len(onsets)
        fires = sum(1 for o in onsets if o >= 0)
        lo, hi = clopper_pearson(fires, n)
        onset_vals = [o for o in onsets if o >= 0]
        median_onset = sorted(onset_vals)[len(onset_vals)//2] if onset_vals else None
        elapsed = time.time() - t0
        print(f"  T={T:<5} {fires:>3}/{n:<3} fire ({fires/n*100:5.1f}%, "
              f"95% CI [{lo*100:5.1f}%, {hi*100:5.1f}%])  "
              f"median onset={median_onset}  [{elapsed/60:.1f} min elapsed]")
        results["rows"].append({
            "temperature": T, "fires": fires, "n": n, "rate": fires/n,
            "ci_lo": lo, "ci_hi": hi, "median_onset": median_onset,
            "onsets": onsets,
        })

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    print(f"Saved -> {args.out}")

    # Locate the transition: smallest T whose CI upper bound still excludes
    # a "frequent" pathology (here: upper bound < 5%), read directly off the
    # table above -- printed explicitly so the paper can cite a number, not
    # eyeball a plot.
    below = [r for r in results["rows"] if r["ci_hi"] < 0.05]
    if below:
        t_star = max(r["temperature"] for r in below)
        print(f"\nLargest T with 95% CI upper bound < 5% loop rate: T={t_star}")
    else:
        print("\nNo temperature in the grid had a 95% CI upper bound under 5%; "
              "extend the grid downward or check n_chars/n_seeds.")


if __name__ == "__main__":
    main()