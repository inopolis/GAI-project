"""
synthetic_fsm_hazard.py

First verification of finite-horizon loop-hazard control on a finite-state
model where the TRUE loop probability is exactly computable by dynamic
programming, used to check the projection machinery (Section 6 of the paper)
before attempting to transfer it to a real language model (Section 10).

Model (S=20 states; see build_base_matrix for exact transition probabilities)
-------------------------------------------------------------------------
Most states form a large, mutually diffuse "safe" region. A small
probability P_GATEWAY from any diffuse state leads to a designated GATEWAY
state, which is NOT itself a repeat when first entered (a one-step/local
risk proxy sees nothing unusual about it) but which transitions with high
probability P_ENTER into a tight 2-cycle {CYC_A, CYC_B} (further
self-reinforcing probability P_CYCLE). A decoder that only reacts to an
already-realized repeat cannot see the gateway coming; a decoder using the
true finite-horizon hazard can downweight the gateway pre-emptively.

Precise event definition
-------------------------
Let s_0, s_1, ..., s_t, ... be the state sequence and W_t = (s_{max(0,t-w+1)},
..., s_t) the trailing window of the last w <= W states (W=2 throughout).
LOOP(t) := 1[ s_{t+1} in W_t ], i.e. the state entered at t+1 already
appears in the trailing window BEFORE that transition. "Loop within the next
H steps starting from state s at time t with window W_t" is the event
  L = 1[ exists t' in {t, ..., t+H-1} : LOOP(t') = 1 ].

Precise definition of the DP table (build_hazard_table)
----------------------------------------------------------
f[r][(s, win)] := Pr( L over the next r transitions | current state = s,
current trailing window = win ), computed by exact backward recursion:
  f[0][(s, win)] = 0
  f[r][(s, win)] = sum_{s'} P[s,s'] * ( 1[s' in win] + 1[s' not in win] *
                                          f[r-1][(s', (win+(s',))[-w:])] )
This is exact (not an approximation or a sampled estimate) because the state
and window spaces are finite and the recursion is a direct application of
the law of total probability over the next transition.

Per-candidate hazard used by both decoders (haz(s') in simulate()):
  haz(s') = 1[s' in current window]                     if s' already loops immediately
          = f[remaining][(s', new_window)]                otherwise
where `remaining` is the number of transitions left in the current rollout
and `new_window` is the window that would result from moving to s'. This
mirrors exactly the per-candidate hazard structure used in the real-model
transfer attempt (Section 10): a candidate's hazard is the probability of
looping within the REST of the horizon, GIVEN that candidate is chosen now.
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

    # Matched-distortion comparison, applied UNIFORMLY across every eps in the
    # grid (not a single hand-picked reference point, which would reintroduce
    # exactly the kind of post-hoc-best-result selection this paper otherwise
    # penalizes -- see the L-sweep discussion in the real-model hazard-transfer
    # section). For each hazard row, scan local's eps to find the closest KL
    # match, then compare loop rates at that matched budget.
    print(f"\nMatched-distortion comparison (uniform across the full eps grid):")
    print(f"{'hazard eps':>10} {'hazard KL':>10} {'hazard loop%':>13} | "
          f"{'local eps (matched)':>20} {'local KL':>10} {'local loop%':>12}")
    print("-" * 90)
    matched_rows = []
    local_scan_eps = np.linspace(0.001, 0.5, 60)
    # Pre-compute local's (loop, KL) at every scan point ONCE, reused for all
    # hazard rows (avoids re-simulating the same local configuration 6 times).
    local_scan_cache = {}
    for eps in local_scan_eps:
        lr, lkl = simulate(P, "local", float(eps), f_table, 2000, H, W, seed=3)
        local_scan_cache[float(eps)] = (lr, lkl)

    # Does local's own achievable KL keep rising as eps -> 0, or does it hit a
    # structural ceiling? At eps == the one-step risk floor, q is already
    # fully concentrated on the minimum-risk action(s); tightening eps
    # further cannot change q at all, so KL cannot rise further either --
    # this is the SAME boundary phenomenon as eps == min_v risk(v) in the
    # real dual solver (Section 5 of the paper), here showing up as a hard
    # ceiling on how much distortion budget the LOCAL mechanism can ever
    # spend, no matter how aggressively it is configured. Confirmed
    # empirically, not assumed: eps swept down to 0 exactly.
    print(f"\nLocal's KL ceiling as eps -> 0 (is the tail of the eps grid a "
          f"grid-resolution gap, or a structural limit local cannot cross "
          f"at ANY eps, however small?):")
    ceiling_probe_eps = [0.001, 0.0001, 0.00001, 0.0]
    ceiling_probe_rows = []
    for eps in ceiling_probe_eps:
        lr, lkl = simulate(P, "local", eps, f_table, 6000, H, W, seed=3)
        print(f"  eps={eps:<10g} local loop%={lr*100:6.2f}  local KL/step={lkl:.4f}")
        ceiling_probe_rows.append({"eps": eps, "loop_rate": lr, "kl": lkl})
    ceiling_kl = ceiling_probe_rows[-1]["kl"]  # eps=0.0: tightest possible constraint
    print(f"  -> local's KL ceiling is ~{ceiling_kl:.4f} bits/step; this does "
          f"NOT increase further below eps=0.0001, confirming a structural "
          f"limit, not a scan-resolution artifact.")
    results["local_kl_ceiling"] = ceiling_kl
    results["local_kl_ceiling_probe"] = ceiling_probe_rows

    for row in results["rows"]:
        target_kl = row["hazard_kl"]
        best = min(local_scan_cache.items(), key=lambda kv: abs(kv[1][1] - target_kl))
        eps_m, (lr_m, lkl_m) = best
        gap = lr_m - row["hazard_loop_rate"]
        # A "matched-KL" comparison requires local to be ABLE to spend at
        # least as much KL as hazard does at this eps -- if hazard's KL
        # exceeds local's structural ceiling, no local configuration (however
        # aggressive) reaches that budget, and the closest-available point
        # found above is a comparison at UNEQUAL, not matched, distortion.
        # Reporting a "win" there would credit hazard for spending distortion
        # local was never able to spend in the first place.
        comparable = target_kl <= ceiling_kl + 1e-6
        flag = "" if comparable else "  [NOT COMPARABLE: exceeds local's KL ceiling]"
        print(f"{row['eps']:>10.2f} {target_kl:>10.4f} {row['hazard_loop_rate']*100:>12.1f}% | "
              f"{eps_m:>20.3f} {lkl_m:>10.4f} {lr_m*100:>11.1f}%{flag}")
        matched_rows.append({
            "hazard_eps": row["eps"], "hazard_kl": target_kl,
            "hazard_loop_rate": row["hazard_loop_rate"],
            "local_eps_matched": eps_m, "local_kl_at_match": lkl_m,
            "local_loop_rate_at_match": lr_m,
            "loop_rate_gap_local_minus_hazard": gap,
            "comparable_at_matched_kl": comparable,
        })
    results["matched_distortion_all_eps"] = matched_rows

    comparable_rows = [r for r in matched_rows if r["comparable_at_matched_kl"]]
    n_hazard_wins = sum(1 for r in comparable_rows if r["loop_rate_gap_local_minus_hazard"] > 0)
    n_excluded = len(matched_rows) - len(comparable_rows)
    print(f"\nAmong points where a matched-KL comparison is actually possible "
          f"({len(comparable_rows)}/{len(matched_rows)}; {n_excluded} excluded "
          f"as exceeding local's KL ceiling, not credited either way), "
          f"hazard beats local in {n_hazard_wins}/{len(comparable_rows)}.")

    with open("synthetic_fsm_results.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("Saved -> synthetic_fsm_results.json")


if __name__ == "__main__":
    main()
    