"""
pretrained_subword_pilot.py

Cheap pilot for a genuinely different, pretrained SUBWORD model, before
committing to a full comparison. Per explicit review guidance: first confirm
there IS a measurable low-temperature repetition regime on this model before
running anything expensive, rather than assuming the character-model's
T-regime (or its very existence) transfers.

This uses GPT-2 (124M, standard HuggingFace checkpoint) by default -- small
enough to run on CPU in reasonable time, large enough to be a genuinely
different architecture/training/tokenization regime from the from-scratch
848K-parameter character model used everywhere else in this project.

WHAT THIS PILOT DOES NOT DO: it does not reuse the character-level R=5/
period_max=60 event at all. Token-level repetition in a ~50k-vocabulary
subword model is a different phenomenon at a different natural granularity
(a repeated PHRASE is typically a handful of tokens, not tens of characters),
so this pilot uses a token-level persistent-cycle event with its OWN
calibration, run identically in spirit to validate_loop_event_v2.py's
protocol but parameterized for tokens instead of characters. Skipping this
recalibration and reusing the character-level thresholds would repeat
exactly the mistake already found and corrected once in this project
(assuming a threshold transfers instead of checking).

Usage
-----
  pip install transformers --break-system-packages
  python3 pretrained_subword_pilot.py --model gpt2 --n_prompts 5 --n_seeds 5

Then read the printed temperature-vs-loop-rate table. If NO temperature in
the tested range shows a measurable (say >10%) loop rate at n=25, this model
may not exhibit the pathology at all in this configuration, and the full
comparison should not proceed until either the temperature range is
extended downward or a different, smaller/weaker pretrained model is tried
(smaller models degenerate more readily, matching the character model's own
848K-parameter scale better than GPT-2's 124M).
"""
import argparse, json, math, re, hashlib
from collections import Counter

import torch


def persistent_loop_onset_tokens(token_ids, R=5, period_max=20, min_p=1):
    """
    Token-level analogue of the character-level persistent-cycle event:
    earliest position where a span of PERIOD_MAX or fewer TOKENS repeats
    exactly R times consecutively. period_max defaults far smaller than the
    character-level event's 60, since a repeated span here is measured in
    tokens (each carrying several characters' worth of information), not
    characters -- this default is a starting point for calibration, not
    assumed correct without the confirmatory check below.
    """
    n = len(token_ids)
    for e in range(min_p * R, n + 1):
        for P in range(min_p, min(period_max, e // R) + 1):
            span = token_ids[e - P:e]
            ok = True
            for r in range(2, R + 1):
                if token_ids[e - r * P:e - (r - 1) * P] != span:
                    ok = False
                    break
            if ok:
                return e
    return -1


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return float("nan"), float("nan")
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


PILOT_PROMPTS = [
    "The report concluded that",
    "In the meeting, she said",
    "According to the study,",
    "He walked into the room and",
    "The committee decided to",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2",
                     help="HuggingFace model id. gpt2 (124M) by default; "
                          "consider a smaller/older model if gpt2 shows no "
                          "measurable degeneration at any tested temperature "
                          "-- larger, better-trained models may simply not "
                          "exhibit this pathology at reachable temperatures.")
    ap.add_argument("--temperatures", nargs="*", type=float,
                     default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--n_prompts", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=150)
    ap.add_argument("--R", type=int, default=5)
    ap.add_argument("--period_max", type=int, default=20)
    ap.add_argument("--out", default="pretrained_pilot_results.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    prompts = PILOT_PROMPTS[:args.n_prompts]
    results = {"model": args.model, "R": args.R, "period_max": args.period_max,
               "rows": []}

    print(f"\n{'T':>6} {'loop rate':>12} {'95% CI':>20} {'n':>5}")
    print("-" * 50)
    for T in args.temperatures:
        onsets = []
        for seed in range(args.n_seeds):
            for p in prompts:
                # stable_seed, NOT Python's hash(): hash() is randomized
                # per-process, which is exactly the reproducibility bug
                # found and fixed elsewhere in this project (see
                # validate_loop_event_v2.py's stable_seed) -- reproduced
                # here deliberately so this script does not reintroduce it.
                p_hash = int(hashlib.md5(p.encode()).hexdigest()[:8], 16) % 1000
                torch.manual_seed(seed * 1000 + p_hash)
                inputs = tok(p, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens,
                        do_sample=(T > 0), temperature=max(T, 1e-6),
                        top_p=1.0, top_k=0,
                        pad_token_id=tok.eos_token_id)
                gen_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
                onset = persistent_loop_onset_tokens(
                    gen_ids, R=args.R, period_max=args.period_max)
                onsets.append(onset)
        n = len(onsets)
        k = sum(1 for o in onsets if o >= 0)
        rate = k / n if n else float("nan")
        try:
            lo, hi = clopper_pearson(k, n)
            ci_str = f"[{lo*100:.1f}%, {hi*100:.1f}%]"
        except ImportError:
            ci_str = "(scipy not available)"
        print(f"{T:>6.2f} {rate*100:>11.1f}% {ci_str:>20} {n:>5}")
        results["rows"].append({"T": T, "fires": k, "n": n, "rate": rate})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.out}")

    max_rate = max(r["rate"] for r in results["rows"])
    if max_rate < 0.10:
        print("\nWARNING: no tested temperature reached even a 10% loop rate.")
        print("Before running the full comparison, either:")
        print("  (a) extend --temperatures further down (e.g. 0.01, 0.02), or")
        print("  (b) try a smaller/older pretrained model (GPT-2 124M is "
              "reasonably robust; distilled or smaller models degenerate "
              "more readily and may better match this project's regime).")
        print("Running the full comparison on a model with no measurable "
              "pathology would not test anything -- there would be nothing "
              "for any decoder to fix.")
    else:
        best_T = max(results["rows"], key=lambda r: r["rate"])["T"]
        print(f"\nMeasurable repetition regime found (peak {max_rate*100:.1f}% "
              f"at T={best_T}). Proceed to the full comparison at a "
              f"temperature from this regime, with a properly CALIBRATED "
              f"(not just this pilot's placeholder R/period_max) token-level "
              f"event -- this pilot's R={args.R}, period_max={args.period_max} "
              f"were not validated against held-out non-degenerate text and "
              f"must not be used for a reported comparison as-is.")


if __name__ == "__main__":
    main()