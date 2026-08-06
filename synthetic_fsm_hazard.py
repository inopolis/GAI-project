"""
synthetic_fsm_hazard.py  (v2 -- bounded-window DP, tractable)

First verification of finite-horizon loop-hazard control on a finite-state
model where the TRUE loop probability is exactly computable.

Design note vs. v1: "visited" is now a bounded trailing WINDOW of the last w
states (not the unbounded full history). This is more faithful to how the
persistent-loop detector actually works in the main paper (period <= 60,
not "ever in the whole generation") and it makes the hazard table finite and
REUSABLE across the whole Monte Carlo run: the DP state is
(current_state, trailing window, steps_remaining), computed once bottom-up
for r = 0..H, then looked up (not recomputed) at every simulated step.

Model
-----
6 states. {0,1,5} diffuse. State 2 = trap entry: not itself a repeat, but
from 2 the chain moves with high probability into a tight 2-cycle {3,4}. A
decoder that only reacts to an IMMEDIATE window-repeat cannot see this
coming -- it only fires once already inside the 3<->4 cycle. A decoder using
the TRUE finite-horizon hazard can downweight state 2 pre-emptively.

"Loop within the next H steps" = the next state entered is already present
in the trailing window of the last w states, at any point in the next H
transitions.
"""

import numpy as np
import json
import time

S = 20        # states
W = 2         # trailing-window length ("period" bound)
H = 6         # horizon (transitions)
N_MC = 3000
TRAP, CYC_A, CYC_B = S - 3, S - 2, S - 1   # gateway, then a tight 2-cycle
P_GATEWAY, P_ENTER, P_CYCLE = 0.05, 0.75, 0.80


def build_base_matrix():
    """
    Most states form a large, mutually-diffuse 'safe' region: from any of
    them, only a SMALL probability (P_GATEWAY) leads to the gateway state, so
    most candidates at any step are genuinely low-hazard. The gateway itself,
    once entered, funnels with HIGH probability (P_ENTER) into a tight
    2-cycle (P_CYCLE both ways). This asymmetry is what a one-step-only proxy
    cannot see: the gateway is not itself a repeat when first chosen, but
    choosing it is what actually carries the risk.
    """
    P = np.full((S, S), 0.5 / S)
    diffuse = [s for s in range(S) if s not in (TRAP, CYC_A, CYC_B)]
    for s in diffuse:
        P[s, :] = (1 - P_GATEWAY) / (S - 1)
        P[s, TRAP] = P_GATEWAY
    P[TRAP, :] = (1 - P_ENTER) / (S - 1)
    P[TRAP, CYC_A] = P_ENTER
    P[CYC_A, :] = (1 - P_CYCLE) / (S - 1)
    P[CYC_A, CYC_B] = P_CYCLE
    P[CYC_B, :] = (1 - P_CYCLE) / (S - 1)
    P[CYC_B, CYC_A] = P_CYCLE
    return P / P.sum(axis=1, keepdims=True)


def temper(P, T):
    logP = np.log(P)
    Q = np.exp(logP / T)
    return Q / Q.sum(axis=1, keepdims=True)


def windows_of_length(k):
    if k == 0:
        return [()]
    out = []
    def rec(pre):
        if len(pre) == k:
            out.append(tuple(pre)); return
        for s in range(S):
            rec(pre + [s])
    rec([])
    return out


def build_hazard_table(P, horizon, w):
    all_windows = []
    for k in range(0, w + 1):
        all_windows.extend(windows_of_length(k))

    f = {0: {}}
    for s in range(S):
        for win in all_windows:
            f[0][(s, win)] = 0.0

    for r in range(1, horizon + 1):
        f[r] = {}
        for s in range(S):
            row = P[s, :]
            for win in all_windows:
                total = 0.0
                for sp in range(S):
                    p = row[sp]
                    if p <= 0:
                        continue
                    if sp in win:
                        total += p * 1.0
                    else:
                        new_win = (win + (sp,))[-w:]
                        total += p * f[r - 1][(sp, new_win)]
                f[r][(s, win)] = total
    return f


def dual_calibrate(p_row, risk_row, eps, lam_max=200.0, iters=40):
    def q_of(lam):
        wgt = p_row * np.exp(-lam * risk_row)
        return wgt / wgt.sum()
    g0 = float((p_row * risk_row).sum())
    if g0 <= eps:
        return 0.0, g0, True
    lo, hi = 0.0, lam_max
    g_hi = float((q_of(hi) * risk_row).sum())
    feasible = g_hi <= eps
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        g_m = float((q_of(mid) * risk_row).sum())
        if g_m > eps:
            lo = mid
        else:
            hi = mid
    return hi, float((q_of(hi) * risk_row).sum()), feasible


def kl(q, p):
    m = q > 0
    return float((q[m] * (np.log2(q[m] + 1e-300) - np.log2(p[m] + 1e-300))).sum())


def simulate(P, decoder, eps, f_table, n, horizon, w, start=0, seed=0):
    rr = np.random.default_rng(seed)
    loops = 0
    kls = []
    states_range = np.arange(S)
    for _ in range(n):
        s = start
        window = (s,)
        looped = False
        step_kls = []
        for t in range(horizon):
            p_row = P[s, :]
            remaining = horizon - t - 1
            if decoder == "raw":
                q_row = p_row
            else:
                if decoder == "local":
                    risk_row = np.array([1.0 if sp in window else 0.0 for sp in range(S)])
                elif decoder == "hazard":
                    risk_row = np.array([
                        1.0 if sp in window else f_table[remaining][(sp, (window + (sp,))[-w:])]
                        for sp in range(S)
                    ])
                else:
                    raise ValueError(decoder)
                lam, ach, feas = dual_calibrate(p_row, risk_row, eps)
                q_row = p_row * np.exp(-lam * risk_row)
                q_row = q_row / q_row.sum()
                step_kls.append(kl(q_row, p_row))

            s_next = int(rr.choice(states_range, p=q_row))
            if s_next in window:
                looped = True
            window = (window + (s_next,))[-w:]
            s = s_next
        if looped:
            loops += 1
        if step_kls:
            kls.append(float(np.mean(step_kls)))
    return loops / n, (float(np.mean(kls)) if kls else 0.0)


def main():
    t0 = time.time()
    P_base = build_base_matrix()
    T0 = 0.6
    P = temper(P_base, T0)

    print(f"Building bounded-window hazard table (S={S}, W={W}, H={H})...")
    f_table = build_hazard_table(P, H, W)
    print(f"  done in {time.time()-t0:.1f}s\n")

    raw_rate, _ = simulate(P, "raw", None, f_table, N_MC, H, W, seed=0)
    print(f"RAW baseline (T={T0}): empirical loop-by-{H} rate = {raw_rate:.3f}\n")

    eps_grid = [0.30, 0.20, 0.10, 0.05, 0.02, 0.01]
    results = {"raw_loop_rate": raw_rate, "T0": T0, "H": H, "W": W, "S": S,
               "eps_grid": eps_grid, "rows": []}

    print(f"{'eps':>6} | {'local: loop%':>13} {'KL/step':>9} | {'hazard: loop%':>14} {'KL/step':>9}")
    print("-" * 66)
    for eps in eps_grid:
        lr, lkl = simulate(P, "local", eps, f_table, N_MC, H, W, seed=1)
        hr, hkl = simulate(P, "hazard", eps, f_table, N_MC, H, W, seed=2)
        print(f"{eps:>6.2f} | {lr*100:>12.1f}% {lkl:>9.4f} | {hr*100:>13.1f}% {hkl:>9.4f}")
        results["rows"].append({"eps": eps, "local_loop_rate": lr, "local_kl": lkl,
                                 "hazard_loop_rate": hr, "hazard_kl": hkl})

    ref = [r for r in results["rows"] if r["eps"] == 0.05][0]
    target_kl = ref["hazard_kl"]
    print(f"\nMatched-distortion check: hazard@eps=0.05 spends {target_kl:.4f} bits/step "
          f"for {ref['hazard_loop_rate']*100:.1f}% loop rate.")
    print("Scanning local-risk eps for the closest KL/step match...")
    best = None
    for eps in np.linspace(0.001, 0.5, 40):
        lr, lkl = simulate(P, "local", float(eps), f_table, 1200, H, W, seed=3)
        d = abs(lkl - target_kl)
        if best is None or d < best[0]:
            best = (d, eps, lr, lkl)
    _, eps_m, lr_m, lkl_m = best
    print(f"  local-risk  @ eps={eps_m:.3f}: KL/step={lkl_m:.4f}, loop rate={lr_m*100:.1f}%")
    print(f"  true-hazard @ eps=0.050: KL/step={target_kl:.4f}, loop rate={ref['hazard_loop_rate']*100:.1f}%")
    results["matched_distortion_check"] = {
        "target_kl_per_step": target_kl, "local_eps_matched": float(eps_m),
        "local_loop_rate": lr_m, "local_kl": lkl_m,
        "hazard_loop_rate": ref["hazard_loop_rate"],
    }

    with open("synthetic_fsm_results.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("Saved -> synthetic_fsm_results.json")


if __name__ == "__main__":
    main()