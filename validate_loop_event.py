"""
validate_loop_event.py

Validates that the loop event used for the survival analysis is a PATHOLOGY
detector and not a detector of ordinary language.

Protocol
--------
A loop event is admissible only if it (a) essentially never fires on held-out
HUMAN text and (b) fires on generations that are visibly degenerate (greedy).
We measure both on the same window length used for the generations.

Why this exists
---------------
The event used in earlier versions -- "the first repeated n-gram" -- fails (a)
badly: on held-out human prose it fires on the majority of windows, so a decoder
could score "better than Dickens" on it. Any survival ranking built on it is
therefore uninterpretable. This script reproduces that finding and calibrates
the replacement event.

Usage
-----
  python3 validate_loop_event.py --data_dir data_out \
      --samples runs/<run>/samples_cosine.txt --n_chars 500
"""

import os, argparse, json
import numpy as np


def loop_onset_ngram(t, n):
    seen = set()
    for i in range(len(t) - n + 1):
        g = t[i:i + n]
        if g in seen:
            return i
        seen.add(g)
    return -1


def persistent_loop_onset(text, P_max=60, R=3, min_p=2):
    """Onset of a span of length P repeated R times consecutively and exactly."""
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


def human_windows(bin_path, itos, n_chars, n_win):
    data = np.fromfile(bin_path, dtype=np.uint16).astype(np.int64)
    text = "".join(itos.get(str(int(i)), "?") for i in data[:n_chars * n_win * 2].tolist())
    return [text[i * n_chars:(i + 1) * n_chars] for i in range(n_win)]


def grab_samples(path, method, ckpt):
    if not path or not os.path.exists(path):
        return []
    txt = open(path, encoding="utf-8").read()
    out, i = [], 0
    while True:
        i = txt.find(f"[{method}][{ckpt}] prompt=", i)
        if i < 0:
            break
        lines = txt[i:].split("\n"); body = []
        for l in lines[2:]:
            if l.strip() == "" and body:
                break
            body.append(l)
        out.append(" ".join(body).strip()); i += 10
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_out")
    ap.add_argument("--samples", default=None, help="samples_<ckpt>.txt from a run")
    ap.add_argument("--ckpt_name", default="cosine")
    ap.add_argument("--n_chars", type=int, default=500)
    ap.add_argument("--n_win", type=int, default=150)
    ap.add_argument("--out", default="loop_event_validation.json")
    args = ap.parse_args()

    itos = json.load(open(os.path.join(args.data_dir, "vocab.json")))["itos"]
    report = {"n_chars": args.n_chars, "n_windows": args.n_win, "human": {}, "generated": {}}

    print(f"Window = {args.n_chars} chars, {args.n_win} windows per source\n")
    print("A) FALSE-POSITIVE RATE ON HELD-OUT HUMAN TEXT  (must be ~0 to be admissible)")
    print(f"   {'source':<16}" + "".join(f"{k:>12}" for k in
          ["ngram8", "ngram10", "ngram12", "ngram16", "persist R=2", "persist R=3"]))
    for label, fn in [("val (Doyle)", "val.bin"), ("test (Dickens)", "test.bin")]:
        p = os.path.join(args.data_dir, fn)
        if not os.path.exists(p):
            continue
        wins = human_windows(p, itos, args.n_chars, args.n_win)
        rates = {
            "ngram8":  sum(1 for w in wins if loop_onset_ngram(w, 8) >= 0) / len(wins),
            "ngram10": sum(1 for w in wins if loop_onset_ngram(w, 10) >= 0) / len(wins),
            "ngram12": sum(1 for w in wins if loop_onset_ngram(w, 12) >= 0) / len(wins),
            "ngram16": sum(1 for w in wins if loop_onset_ngram(w, 16) >= 0) / len(wins),
            "persist_R2": sum(1 for w in wins if persistent_loop_onset(w, R=2) >= 0) / len(wins),
            "persist_R3": sum(1 for w in wins if persistent_loop_onset(w, R=3) >= 0) / len(wins),
        }
        report["human"][label] = rates
        print(f"   {label:<16}" + "".join(f"{100*v:>11.1f}%" for v in rates.values()))

    print("\n   Verdict: an event that fires on a large share of human prose is not a")
    print("   pathology detector. Only persistent R=3 is admissible below.\n")

    if args.samples:
        print("B) DETECTION RATE ON GENERATED TEXT (persistent R=3)")
        for m in ["greedy", "temp_0.8", "nucleus_p0.95", "rep_penalty_1.3",
                  "lookback_a3.0", "risk_only", "adaptive", "rr_dual_eps0.05"]:
            s = grab_samples(args.samples, m, args.ckpt_name)
            if not s:
                continue
            f = [persistent_loop_onset(x) for x in s]
            rate = sum(1 for x in f if x >= 0) / len(f)
            ons = sorted([x for x in f if x >= 0])
            med = ons[len(ons) // 2] if ons else None
            report["generated"][m] = {"rate": rate, "n": len(s), "median_onset": med}
            print(f"   {m:<18} fires {100*rate:5.1f}%  (n={len(s)}"
                  + (f", median onset {med})" if med is not None else ")"))
        print("\n   If every stochastic decoder sits at 0% while greedy fires, the horizon")
        print("   is too short to separate them and the generation length must be raised")
        print("   before any survival claim is made.")

    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()