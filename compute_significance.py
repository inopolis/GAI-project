"""
compute_significance.py

Produces the paper's significance table (Table 7) as a SAVED, reproducible
artifact, replacing the ad-hoc inline Python re-run by hand at every check-in
of this project. Two procedures, both applied to the SAME underlying
prompt-level statistic so they cannot disagree about what they are testing:

  PRIMARY: exact sign-flip (randomization) test. For k=15 prompts, all
  2^15=32768 sign patterns are enumerated exactly (not Monte Carlo).
  SENSITIVITY: cluster bootstrap (3000 resamples of prompts).

Usage
-----
  python3 compute_significance.py \
      --all_results runs/v11_loop_regime/all_results.json \
      --checkpoint cosine --max_t 500 \
      --out significance_table7.json --out_csv significance_table7.csv
"""
import json, csv, argparse, itertools
import numpy as np


def rmst(onsets, mt):
    ev = sorted([t for t in onsets if t >= 0])
    n = len(onsets)
    if not ev:
        return float(mt)
    S, pt, area, before = 1.0, 0, 0.0, 0
    for t in sorted(set(ev)):
        nev = ev.count(t); risk = n - before
        area += (t - pt) * S
        if risk > 0:
            S *= (1 - nev / risk)
        before += nev; pt = t
    area += (mt - pt) * S
    return area


def prompt_level_rmst(onsets, groups, mt):
    uniq = sorted(set(groups))
    out = {}
    for g in uniq:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        out[g] = rmst([onsets[i] for i in idx], mt)
    return out


def exact_sign_flip(diffs):
    diffs = np.array(diffs)
    k = len(diffs)
    obs = diffs.mean()
    if k <= 20:
        extreme, total = 0, 0
        for bits in itertools.product([1, -1], repeat=k):
            s = np.array(bits)
            stat = (s * diffs).mean()
            total += 1
            if abs(stat) >= abs(obs) - 1e-12:
                extreme += 1
        return extreme / total, obs, True
    rng = np.random.default_rng(0)
    nb = 20000
    cnt = 0
    for _ in range(nb):
        s = rng.choice([1, -1], size=k)
        stat = (s * diffs).mean()
        if abs(stat) >= abs(obs) - 1e-12:
            cnt += 1
    return cnt / nb, obs, False


def cluster_idx(groups, rng):
    uniq = sorted(set(groups))
    by = {g: [i for i, gg in enumerate(groups) if gg == g] for g in uniq}
    drawn = rng.choice(len(uniq), size=len(uniq), replace=True)
    out = []
    for d in drawn:
        out.extend(by[uniq[d]])
    return out


def cluster_ci(oa, ob, groups, mt, nb=3000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.array(oa), np.array(ob)
    diffs = []
    for _ in range(nb):
        idx = cluster_idx(groups, rng)
        diffs.append(rmst(a[idx].tolist(), mt) - rmst(b[idx].tolist(), mt))
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# The exact comparisons reported in the paper's Table 7. Edit here, not in a
# throwaway script, so the paper and this file cannot drift apart.
PAIRS = [
    ("lt_risk_only", "lt_temp_0.1"),
    ("lt_risk_only", "lt_rep_penalty_1.3"),
    ("lt_risk_only", "lt_suffixmatch_a3.0"),
    ("lt_risk_only", "lt_dual_eps0.05"),
    ("lt_risk_only", "lt_adaptive"),
    ("lt_adaptive", "lt_dual_eps0.05"),
    ("lt_adaptive", "lt_rep_penalty_1.3"),
    ("lt_dual_eps0.01", "lt_dual_eps0.05"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_results", required=True)
    ap.add_argument("--checkpoint", default="cosine")
    ap.add_argument("--max_t", type=int, default=500)
    ap.add_argument("--out", default="significance_table7.json")
    ap.add_argument("--out_csv", default="significance_table7.csv")
    args = ap.parse_args()

    R = json.load(open(args.all_results))
    rows = {r["strategy"]: r for r in R[args.checkpoint]}

    results = []
    print(f"{'comparison':<36}{'obs dRMST':>10}{'p (exact)':>11}{'exact?':>8}{'cluster CI':>22}")
    print("-" * 90)
    for a, b in PAIRS:
        if a not in rows or b not in rows:
            print(f"  SKIP {a} vs {b}: missing from all_results.json")
            continue
        ra, rb = rows[a], rows[b]
        groups = ra["groups"]
        pra = prompt_level_rmst(ra["onsets_persistent"], groups, args.max_t)
        prb = prompt_level_rmst(rb["onsets_persistent"], groups, args.max_t)
        common = sorted(set(pra) & set(prb))
        diffs = [pra[g] - prb[g] for g in common]
        p, obs, is_exact = exact_sign_flip(diffs)
        lo, hi = cluster_ci(ra["onsets_persistent"], rb["onsets_persistent"], groups, args.max_t)
        print(f"{a+' vs '+b:<36}{obs:>10.1f}{p:>11.4f}{str(is_exact):>8}{f'[{lo:.1f}, {hi:.1f}]':>22}")
        results.append({
            "comparison": f"{a} vs {b}", "method_a": a, "method_b": b,
            "n_prompts": len(common), "obs_dRMST": round(obs, 2),
            "p_exact_signflip": round(p, 4), "is_exact_enumeration": is_exact,
            "cluster_ci_lo": round(lo, 2), "cluster_ci_hi": round(hi, 2),
        })

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    if results:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)
    print(f"\nSaved -> {args.out}, {args.out_csv}")


if __name__ == "__main__":
    main()
    