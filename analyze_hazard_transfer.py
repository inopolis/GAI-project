"""
analyze_hazard_transfer.py

Context-clustered analysis of the real-model hazard-transfer estimator.

Single-token design (hazard_dataset.json from build_hazard_dataset.py):
  per context, the mean hazard label of its "risky" candidates (choosing the
  token now completes an already-seen n-gram) and of its "novel" candidates;
  the per-context difference risky - novel is the unit of analysis.

Multi-step commitment probe (multistep_hazard_probe.json from
test_multistep_hazard.py): per context, one risky and one novel candidate,
for each commitment length L.

For each design: mean paired difference, a percentile bootstrap interval from
resampling WHOLE CONTEXTS, and an exact sign-flip p-value on the per-context
differences (all 2^k patterns when k <= 20, otherwise 100,000 random flips).
For the multi-step probe the p-values are also Bonferroni-adjusted across the
commitment lengths tested.

Usage
-----
  python3 analyze_hazard_transfer.py --dataset hazard_dataset.json \
      --probe multistep_hazard_probe.json --out hazard_transfer_analysis.json
"""
import argparse, itertools, json
import numpy as np


def paired_stats(diffs, n_boot=10000, seed=0):
    d = np.asarray(diffs, dtype=float)
    k = len(d)
    if k == 0:
        return {"n_contexts": 0}
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, k, k)].mean() for _ in range(n_boot)]
    obs = abs(d.mean())
    if k <= 20:
        signs = np.array(list(itertools.product([1, -1], repeat=k)))
        p = float((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).mean())
    else:
        signs = rng.choice([1, -1], size=(100000, k))
        p = float((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).mean())
    return {"n_contexts": k, "mean_diff": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_signflip": p}


def single_token(dataset_path):
    rows = json.load(open(dataset_path))
    by_ctx = {}
    for r in rows:
        if r["hazard_label"] != r["hazard_label"]:
            continue
        key = "risky" if r["candidate_immediate_risk"] > 0 else "novel"
        by_ctx.setdefault(r["context_idx"], {"risky": [], "novel": []})[key].append(r["hazard_label"])
    diffs = [np.mean(v["risky"]) - np.mean(v["novel"])
             for v in by_ctx.values() if v["risky"] and v["novel"]]
    n_risky = sum(len(v["risky"]) for v in by_ctx.values())
    n_novel = sum(len(v["novel"]) for v in by_ctx.values())
    out = paired_stats(diffs)
    out.update({"n_risky_examples": n_risky, "n_novel_examples": n_novel,
                "mean_hazard_risky": float(np.mean([x for v in by_ctx.values() for x in v["risky"]])),
                "mean_hazard_novel": float(np.mean([x for v in by_ctx.values() for x in v["novel"]]))})
    return out


def multi_step(probe_path):
    summary = json.load(open(probe_path))
    out = []
    for s in summary:
        risky = {x["context_idx"]: x["hazard"] for x in s["risky"]}
        novel = {x["context_idx"]: x["hazard"] for x in s["novel"]}
        common = sorted(set(risky) & set(novel))
        stats = paired_stats([risky[c] - novel[c] for c in common])
        stats["L"] = s["L"]
        out.append(stats)
    m = len(out)
    for st in out:
        if "p_signflip" in st:
            st["p_bonferroni"] = min(1.0, st["p_signflip"] * m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hazard_dataset.json")
    ap.add_argument("--probe", default="multistep_hazard_probe.json")
    ap.add_argument("--out", default="hazard_transfer_analysis.json")
    args = ap.parse_args()
    result = {"single_token": single_token(args.dataset),
              "multi_step": multi_step(args.probe)}
    print(json.dumps(result, indent=2))
    json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
