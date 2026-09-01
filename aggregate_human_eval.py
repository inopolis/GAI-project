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
    print("\n=== Part 2: pairwise preference (all method pairings) ===")
    all_records = []
    for b in blobs:
        pid = b.get("participant_id", b["_source_file"])
        c = Counter(r["winner"] for r in b["task2"])
        summary = "  ".join(f"{k}={v}" for k, v in sorted(c.items()))
        print(f"  {pid:<20} {summary}")
        for r in b["task2"]:
            r = dict(r); r["participant_id"] = pid
            all_records.append(r)

    if not all_records:
        print("  (no data)")
        return

    # Group by pair-TYPE (method_a, method_b), order-independent, since older
    # result blobs (pre-expansion) may only have dual/suffixmatch and lack
    # method_a/method_b -- back-fill those from the only two winners possible
    # in that older format so old and new result files can still be combined.
    for r in all_records:
        if "method_a" not in r:
            r["method_a"], r["method_b"] = "dual", "suffixmatch"

    by_pair = defaultdict(list)
    for r in all_records:
        key = tuple(sorted((r["method_a"], r["method_b"])))
        by_pair[key].append(r)

    print(f"\n  By pairing (n={len(all_records)} total comparisons, "
          f"{len(by_pair)} distinct pairings):")
    for (m1, m2), records in sorted(by_pair.items()):
        c = Counter(r["winner"] for r in records)
        n1, n2, tie = c.get(m1, 0), c.get(m2, 0), c.get("tie", 0)
        n_decisive = n1 + n2
        if n_decisive > 0:
            lo, hi = clopper_pearson(n1, n_decisive)
            print(f"    {m1} vs {m2}: {m1}={n1} {m2}={n2} tie={tie}  "
                  f"({m1} preferred {100*n1/n_decisive:.0f}% of decisive votes, "
                  f"95% CI [{lo*100:.0f}%, {hi*100:.0f}%])")
        else:
            print(f"    {m1} vs {m2}: {m1}={n1} {m2}={n2} tie={tie}  (all ties)")

    # Pooled per-method win rate across every pairing it appeared in -- a
    # simple tally, NOT a Bradley-Terry or other model-based ranking; useful
    # as an at-a-glance summary only, and only as informative as the (small,
    # pilot-scale) n behind it.
    wins, appearances = Counter(), Counter()
    for r in all_records:
        appearances[r["method_a"]] += 1
        appearances[r["method_b"]] += 1
        if r["winner"] != "tie":
            wins[r["winner"]] += 1
    print("\n  Pooled win rate per method (simple tally across all its pairings, "
          "not a fitted ranking model):")
    for method in sorted(appearances):
        n = appearances[method]
        w = wins.get(method, 0)
        print(f"    {method:<14} won {w}/{n} comparisons it appeared in "
              f"({100*w/n:.0f}%)")
    print("  (This is a small-n pilot statistic throughout; no significance "
          "claim is made from it alone.)")


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
