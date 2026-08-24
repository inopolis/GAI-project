"""
aggregate_human_eval.py

Aggregates one or more result blobs produced by human_eval_tool.html (each
participant pastes back exactly one JSON object, saved to its own file) into
pooled loop-detection-agreement and pairwise-preference statistics.

Usage
-----
  python3 aggregate_human_eval.py results_participant1.json results_participant2.json ...

Each input file must contain exactly the JSON object the tool's "Copy
results" step produces:
  {"participant_id": ..., "task1": [...], "task2": [...], ...}
"""
import argparse, json, math, sys
from collections import Counter, defaultdict


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return float("nan"), float("nan")
    try:
        from scipy.stats import beta
    except ImportError:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def load_results(paths):
    blobs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if "task1" not in d or "task2" not in d:
            print(f"  WARNING: {p} does not look like a human_eval_tool.html "
                  f"result blob (missing task1/task2); skipping.", file=sys.stderr)
            continue
        d["_source_file"] = p
        blobs.append(d)
    return blobs


def summarize_task1(blobs):
    print("\n=== Part 1: loop-detection agreement ===")
    all_records = []
    for b in blobs:
        pid = b.get("participant_id", b["_source_file"])
        correct = sum(1 for r in b["task1"] if r["correct"])
        n = len(b["task1"])
        print(f"  {pid:<20} {correct}/{n} correct ({100*correct/n:.1f}%)"
              if n else f"  {pid:<20} no task1 responses")
        for r in b["task1"]:
            r = dict(r); r["participant_id"] = pid
            all_records.append(r)

    if not all_records:
        print("  (no data)")
        return

    total_correct = sum(1 for r in all_records if r["correct"])
    total_n = len(all_records)
    lo, hi = clopper_pearson(total_correct, total_n)
    print(f"\n  POOLED accuracy: {total_correct}/{total_n} "
          f"({100*total_correct/total_n:.1f}%), 95% CI "
          f"[{lo*100:.1f}%, {hi*100:.1f}%]")

    print("\n  By category:")
    by_cat = defaultdict(lambda: [0, 0])
    for r in all_records:
        by_cat[r["category"]][0] += int(r["correct"])
        by_cat[r["category"]][1] += 1
    for cat, (c, n) in sorted(by_cat.items()):
        print(f"    {cat:<12} {c}/{n} ({100*c/n:.1f}%)")

    # Confusion: how often participants say "loop" when ground truth is/isn't
    tp = sum(1 for r in all_records if r["ground_truth_is_loop"] and r["participant_says_loop"])
    fn = sum(1 for r in all_records if r["ground_truth_is_loop"] and not r["participant_says_loop"])
    fp = sum(1 for r in all_records if not r["ground_truth_is_loop"] and r["participant_says_loop"])
    tn = sum(1 for r in all_records if not r["ground_truth_is_loop"] and not r["participant_says_loop"])
    print(f"\n  Confusion matrix (pooled): TP={tp} FN={fn} FP={fp} TN={tn}")

    if len(blobs) >= 2:
        # Pairwise inter-participant agreement on identical items, if item
        # sets overlap (they will, since all participants see all 18 items,
        # just in a different order).
        by_item = defaultdict(list)
        for r in all_records:
            by_item[r["item_index"]].append(r["participant_says_loop"])
        agree, total_pairs = 0, 0
        for votes in by_item.values():
            for i in range(len(votes)):
                for j in range(i + 1, len(votes)):
                    total_pairs += 1
                    agree += int(votes[i] == votes[j])
        if total_pairs:
            print(f"  Inter-participant agreement rate: {agree}/{total_pairs} "
                  f"({100*agree/total_pairs:.1f}%)")


def summarize_task2(blobs):
    print("\n=== Part 2: pairwise preference (dual vs suffixmatch) ===")
    all_records = []
    for b in blobs:
        pid = b.get("participant_id", b["_source_file"])
        c = Counter(r["winner"] for r in b["task2"])
        print(f"  {pid:<20} dual={c.get('dual',0)} suffixmatch={c.get('suffixmatch',0)} "
              f"tie={c.get('tie',0)}")
        for r in b["task2"]:
            r = dict(r); r["participant_id"] = pid
            all_records.append(r)

    if not all_records:
        print("  (no data)")
        return

    c = Counter(r["winner"] for r in all_records)
    n_dual, n_sfx, n_tie = c.get("dual", 0), c.get("suffixmatch", 0), c.get("tie", 0)
    n_total = n_dual + n_sfx + n_tie
    print(f"\n  POOLED: dual={n_dual}  suffixmatch={n_sfx}  tie={n_tie}  "
          f"(n={n_total})")

    # Two-sided sign test on decisive (non-tie) comparisons: is dual
    # preferred more often than chance among cases with a stated preference?
    n_decisive = n_dual + n_sfx
    if n_decisive > 0:
        p_dual = n_dual / n_decisive
        lo, hi = clopper_pearson(n_dual, n_decisive)
        print(f"  Among decisive comparisons (excl. ties): dual preferred "
              f"{n_dual}/{n_decisive} ({100*p_dual:.1f}%), 95% CI "
              f"[{lo*100:.1f}%, {hi*100:.1f}%]")
        print("  (50% = no preference either way; this is a small-n pilot "
              "statistic, not a claim of significance on its own.)")
    else:
        print("  No decisive comparisons (all ties).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="One JSON file per participant.")
    ap.add_argument("--out", default=None,
                    help="Optional path to write the combined raw records as JSON.")
    args = ap.parse_args()

    blobs = load_results(args.results)
    print(f"Loaded {len(blobs)} participant result file(s).")
    if not blobs:
        return

    summarize_task1(blobs)
    summarize_task2(blobs)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(blobs, f, indent=2)
        print(f"\nWrote combined raw records -> {args.out}")


if __name__ == "__main__":
    main()
