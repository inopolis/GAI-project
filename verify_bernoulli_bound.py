"""
verify_bernoulli_bound.py

Two checks, both from already-saved artifacts, no new generation:

1. Confirms the FSM oracle (synthetic_fsm_hazard.py, Table 7 in the paper)
   is exactly the minimum-KL policy implied by the general Bernoulli-KL
   decomposition D(Q||P) = d_Ber(Q(E)||P(E)) + ... , by comparing the
   oracle's own KL (backward-recursion identity) against d_Ber(eps||p) with
   p computed directly from the untilted hazard table.

2. Applies the same bound to every decoder in the primary-model
   common-distortion comparison (common_kl_comparison.py), using only the
   loop rate and per-step mean KL already in that CSV: total path KL is the
   per-step mean times the fixed generation length (500 characters), and
   the bound is d_Ber(decoder's own loop rate || the T=0.10 reference's
   loop rate).

3. Functional-form ablation on the same FSM (new): the optimal causal kernel
   uses the exact event-to-go hazard h in the AFFINE form
       q(v|h_t) ~ P(v|h_t) * [1 - (1 - 2^-lam) * h(v)]
   with ONE multiplier lam for the whole generation.  Two variants that use
   the same exact hazard are compared at the same sequence-level target:
   an EXPONENTIAL tilt P * 2^(-lam*h) with one global lam, and the per-step
   projection with a re-solved one-step budget (the "hazard-informed local"
   controller of the paper).  This separates the effect of the functional
   form from the effect of re-solving a per-step budget.

Usage
-----
  python3 verify_bernoulli_bound.py \
      --commonkl_csv runs/common_kl_comparison_v4_samples/common_kl_comparison.csv
"""
import argparse, csv, math

import synthetic_fsm_hazard as fsm


def d_ber_bits(eps, p):
    val = 0.0
    if eps > 0:
        val += eps * math.log2(eps / p)
    if eps < 1:
        val += (1 - eps) * math.log2((1 - eps) / (1 - p))
    return val


def check_oracle():
    P_base = fsm.build_base_matrix()
    P = fsm.temper(P_base, 0.6)
    f_table = fsm.build_hazard_table(P, fsm.H, fsm.W)
    start_win = (0,)
    p_exact = f_table[fsm.H][(start_win[-1], start_win)]
    print(f"Exact untilted loop-by-H probability p = {p_exact:.6f}\n")

    eps_grid = [0.30, 0.20, 0.10, 0.05, 0.02, 0.01]
    print(f"{'eps':>6} {'oracle_KL_total':>16} {'dBer(eps||p)':>14} {'diff':>12}")
    for eps in eps_grid:
        orc = fsm.solve_oracle_lambda(P, fsm.H, fsm.W, start=0, target_eps=eps)
        okl = orc["kl_bits_per_step"] * fsm.H
        bound = d_ber_bits(orc["eps_achieved"], p_exact)
        print(f"{eps:>6.2f} {okl:>16.10f} {bound:>14.10f} {okl - bound:>12.2e}")
        assert abs(okl - bound) < 1e-9, "oracle KL does not match Bernoulli-KL bound"
    print("\nAll six grid points match to <1e-9 (paper reports <1e-14 observed).")


def global_tilt_forward(P, f_table, lam, form):
    """Exact forward propagation of a causal kernel built from the exact
    event-to-go hazard with ONE global multiplier lam.  Returns
    (loop probability, total path KL in bits)."""
    S, H, W = fsm.S, fsm.H, fsm.W
    dist = {(0, (0,)): 1.0}
    looped = kl_total = 0.0
    for t in range(H):
        remaining = H - t - 1
        new = {}
        for (s, win), m in dist.items():
            p_row = P[s, :]
            h = [1.0 if sp in win else f_table[remaining][(sp, (win + (sp,))[-W:])]
                 for sp in range(S)]
            h = [min(max(x, 0.0), 1.0) for x in h]
            if form == "affine":
                w = [p_row[i] * (1.0 - (1.0 - 2.0 ** (-lam)) * h[i]) for i in range(S)]
            else:
                w = [p_row[i] * 2.0 ** (-lam * h[i]) for i in range(S)]
            tot = sum(w)
            q = [x / tot for x in w]
            for i in range(S):
                if q[i] <= 0:
                    continue
                kl_total += m * q[i] * math.log2(q[i] / p_row[i])
                if i in win:
                    looped += m * q[i]
                else:
                    key = (i, (win + (i,))[-W:])
                    new[key] = new.get(key, 0.0) + m * q[i]
        dist = new
    return looped, kl_total


def solve_global_tilt(P, f_table, target, form, lam_hi=60.0, iters=80):
    lo, hi = 0.0, lam_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        eps, _ = global_tilt_forward(P, f_table, mid, form)
        if eps > target:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    eps, kl = global_tilt_forward(P, f_table, lam, form)
    return lam, eps, kl


def check_functional_form():
    P_base = fsm.build_base_matrix()
    P = fsm.temper(P_base, 0.6)
    f_table = fsm.build_hazard_table(P, fsm.H, fsm.W)
    p_exact = f_table[fsm.H][(0, (0,))]
    print("\nFunctional-form ablation (total path KL in bits; H = %d)" % fsm.H)
    print(f"{'eps':>6} {'affine/global':>14} {'d_Ber':>10} {'exp/global':>12} "
          f"{'hazard, per-step budget':>24}")
    rows = []
    for eps in [0.30, 0.20, 0.10, 0.05, 0.02, 0.01]:
        _, e_aff, kl_aff = solve_global_tilt(P, f_table, eps, "affine")
        _, e_exp, kl_exp = solve_global_tilt(P, f_table, eps, "exp")
        haz = fsm.solve_step_eps_for_target(P, fsm.W, fsm.H, "hazard", f_table, eps)
        kl_haz = haz["kl_bits_per_step"] * fsm.H
        bound = d_ber_bits(eps, p_exact)
        print(f"{eps:>6.2f} {kl_aff:>14.6f} {bound:>10.6f} {kl_exp:>12.6f} {kl_haz:>24.6f}")
        assert abs(kl_aff - bound) < 1e-7 and abs(e_aff - eps) < 1e-7
        rows.append({"eps": eps, "affine_global_kl": kl_aff, "d_ber": bound,
                      "exp_global_kl": kl_exp, "exp_global_eps": e_exp,
                      "hazard_per_step_budget_kl": kl_haz})
    return rows


def check_real_decoders(commonkl_csv, n_chars=500):
    rows = {r["strategy"]: r for r in csv.DictReader(open(commonkl_csv))}
    p = float(rows["temperature_0.10"]["loop_rate"])
    print(f"\nReference (T=0.10) loop rate p = {p:.4f}, n_chars = {n_chars}\n")
    print(f"{'strategy':<20}{'loop%':>8}{'path_KL':>12}{'bound':>10}{'ratio':>10}")
    order = ["rr_fixed", "rr_adaptive", "lzpenalty", "rep_penalty_1.3",
              "suffixmatch_a3.0", "fsd", "no_repeat_4gram",
              "rr_dual_eps0.05", "rr_dual_eps0.01"]
    for name in order:
        r = rows[name]
        eps = float(r["loop_rate"])
        kl_total = float(r["kl_bits_mean"]) * n_chars
        bound = d_ber_bits(eps, p)
        ratio = kl_total / bound if bound > 0 else float("inf")
        print(f"{name:<20}{eps*100:>7.1f}%{kl_total:>12.1f}{bound:>10.3f}{ratio:>9.1f}x")
        assert kl_total >= bound - 1e-6, f"{name}: measured KL below the bound"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commonkl_csv",
                     default="runs/common_kl_comparison_v4_samples/common_kl_comparison.csv")
    args = ap.parse_args()
    check_oracle()
    rows = check_functional_form()
    import json
    json.dump(rows, open("functional_form_ablation.json", "w"), indent=2)
    check_real_decoders(args.commonkl_csv)


if __name__ == "__main__":
    main()
