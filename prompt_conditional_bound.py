"""
prompt_conditional_bound.py

Prompt-conditional version of the event-level bound, from the sample-level
records written by common_kl_comparison.py (common_kl_samples.csv).

For each prompt x the decoder samples from Q_x and the reference
(temperature sampling) from P_x, and the chain-rule bound applies prompt by
prompt:   KL(Q_x || P_x) >= d_Ber(Q_x(E) || P_x(E)).
With fixed prompt weights w_x = 1/#prompts the quantity of interest is
      K  = sum_x w_x * (mean total path KL on prompt x)
      B  = sum_x w_x * d_Ber( Q~_x(E) || P~_x(E) )
and the excess  K - B  is the estimated distortion of the conditional shape of
the generation (what is not needed to move the event probability).  Pooling
prompts first, d_Ber(mean Q || mean P), can only be smaller than B (joint
convexity of d_Ber), so a pooled bound overstates the excess.

Estimation:
  * P_x(E), Q_x(E) are loop fractions over the seeds of one prompt (10), so
    the plug-in d_Ber is infinite whenever P_x(E) is 0 or 1 and Q_x(E) is not
    equal to it.  The primary analysis therefore uses Jeffreys-smoothed
    fractions (k+0.5)/(n+1) for BOTH P_x and Q_x; plug-in and Laplace
    smoothings are reported as sensitivity analyses.
  * Uncertainty: percentile bootstrap that resamples WHOLE PROMPTS with
    replacement (the same resampled prompts for the reference and the
    decoder), keeping every sample's loop event and path KL together.
  * All inputs are sample estimates; the outputs are descriptive plug-in
    estimates, not bounds with guaranteed coverage.

Also reported, per decoder-free reference quantity: the cost of rejection
sampling from the reference (resample a whole generation until the detector
does not fire): per prompt it attains Q_x(E)=0 exactly, with
KL = -log2(1 - P_x(E)) bits and 1/(1 - P_x(E)) expected generations.

Usage
-----
  python3 prompt_conditional_bound.py \
      --samples runs/common_kl_comparison_v4_samples/common_kl_samples.csv \
      --out runs/common_kl_comparison_v4_samples/prompt_conditional_bound.json
"""
import argparse, csv, json, math
from collections import defaultdict
import numpy as np

REF = "temperature_0.10"


def d_ber(q, p):
    v = 0.0
    if q > 0:
        v += q * math.log2(q / p) if p > 0 else math.inf
    if q < 1:
        v += (1 - q) * math.log2((1 - q) / (1 - p)) if p < 1 else math.inf
    return v


def smooth(k, n, a):
    return (k + a) / (n + 2 * a)


def load(path):
    data = defaultdict(lambda: defaultdict(list))   # strategy -> prompt -> [(event, kl_total)]
    for r in csv.DictReader(open(path)):
        data[r["strategy"]][int(r["prompt_index"])].append(
            (int(r["loop_event"]), float(r["kl_bits_total"])))
    return data


def estimate(ref, dec, prompts, a):
    """Returns (K, B, B_pool, rejection_kl) for the given prompt multiset."""
    K = B = rej = 0.0
    kr = nr = kd = nd = 0
    for x in prompts:
        re_, de_ = ref[x], dec[x]
        k_ref, n_ref = sum(e for e, _ in re_), len(re_)
        k_dec, n_dec = sum(e for e, _ in de_), len(de_)
        p, q = smooth(k_ref, n_ref, a), smooth(k_dec, n_dec, a)
        K += np.mean([kl for _, kl in de_])
        B += d_ber(q, p)
        rej += -math.log2(1 - p)
        kr += k_ref; nr += n_ref; kd += k_dec; nd += n_dec
    m = len(prompts)
    B_pool = d_ber(smooth(kd, nd, a), smooth(kr, nr, a))
    return K / m, B / m, B_pool, rej / m


def analyze(data, n_boot, seed):
    ref = data[REF]
    prompts = sorted(ref)
    rng = np.random.default_rng(seed)
    out = {}
    for strat, dec in data.items():
        if strat == REF:
            continue
        K, B, Bp, rej = estimate(ref, dec, prompts, 0.5)
        boots = []
        for _ in range(n_boot):
            sel = rng.choice(prompts, size=len(prompts), replace=True).tolist()
            k, b, _, _ = estimate(ref, dec, sel, 0.5)
            boots.append((k, b, k - b))
        boots = np.array(boots)
        ci = lambda col: [float(np.percentile(boots[:, col], 2.5)),
                           float(np.percentile(boots[:, col], 97.5))]
        sens = {}
        for label, a in [("laplace", 1.0), ("jeffreys", 0.5), ("plug_in", 0.0)]:
            try:
                _, b_a, bp_a, _ = estimate(ref, dec, prompts, a)
            except ZeroDivisionError:
                b_a, bp_a = math.inf, math.inf
            sens[label] = {"B_prompt_conditional": b_a, "B_pooled": bp_a}
        k_loop = sum(e for ps in dec.values() for e, _ in ps)
        n_all = sum(len(ps) for ps in dec.values())
        out[strat] = {
            "loop_events": k_loop, "n_samples": n_all,
            "path_kl_bits": K, "path_kl_ci95": ci(0),
            "bound_prompt_conditional": B, "bound_ci95": ci(1),
            "excess_bits": K - B, "excess_ci95": ci(2),
            "bound_pooled": Bp, "ratio_descriptive": (K / B if B > 0 else math.inf),
            "sensitivity": sens,
        }
    k_ref = sum(e for ps in ref.values() for e, _ in ps)
    n_ref = sum(len(ps) for ps in ref.values())
    p_x = {x: smooth(sum(e for e, _ in ref[x]), len(ref[x]), 0.5) for x in prompts}
    out["_reference"] = {
        "loop_events": k_ref, "n_samples": n_ref,
        "per_prompt_P_smoothed": p_x,
        "rejection_kl_bits_mean": float(np.mean([-math.log2(1 - p) for p in p_x.values()])),
        "rejection_expected_generations_mean": float(np.mean([1 / (1 - p) for p in p_x.values()])),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", default="prompt_conditional_bound.json")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res = analyze(load(args.samples), args.n_boot, args.seed)
    ref = res.pop("_reference")
    print(f"reference: {ref['loop_events']}/{ref['n_samples']} loop events; rejection sampling "
          f"costs {ref['rejection_kl_bits_mean']:.3f} bits and "
          f"{ref['rejection_expected_generations_mean']:.2f} expected generations per sample "
          f"(prompt-averaged, Jeffreys)\n")
    print(f"{'strategy':<20}{'loops':>7}{'path KL':>10}{'bound B':>9}{'excess':>10}  "
          f"{'excess 95% CI':<24}{'pooled B':>9}")
    for s, r in res.items():
        print(f"{s:<20}{r['loop_events']:>7}{r['path_kl_bits']:>10.1f}{r['bound_prompt_conditional']:>9.3f}"
              f"{r['excess_bits']:>10.1f}  [{r['excess_ci95'][0]:.1f}, {r['excess_ci95'][1]:.1f}]"
              f"{'':<6}{r['bound_pooled']:>9.3f}")
    res["_reference"] = ref
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
