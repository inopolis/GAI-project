"""
budget_ladder.py

Exact (no simulation) experiments in the small system of synthetic_fsm_hazard.py
that separate three things the earlier comparison confounded when a per-step
controller re-solves a fixed absolute risk budget at every step:

  (a) INFEASIBILITY  - at nodes where the budget is below the achievable floor
                       min_v H(h, v) the one-step problem has no solution and
                       the controller collapses onto the safest token;
  (b) ALLOCATION     - how a feasible rule spreads the total loop-probability
                       reduction over histories (which histories get how much
                       one-step distortion);
  (c) FUNCTIONAL FORM - exponential vs affine use of the same hazard.

All controllers use the same exact event-to-go hazard H(h, v) as their risk
signal and are calibrated by bisection on ONE scalar so that their exact
sequence-level loop probability equals the target eps:

  abs     fixed absolute budget      E_q[H] <= b          (the controller of Table 1;
                                                           infeasible nodes collapse)
  rel     headroom-relative budget   E_q[H] <= floor(h) + rho * (E_p[H] - floor(h))
                                      always feasible; one scalar rho for the whole run
  sched   time-dependent absolute budgets b_1..b_H (history independent), optimised
  lam     fixed multiplier           q ~ p * exp(-lam * H)          (global lam)
  affine  optimal rule               q ~ p * (1 - (1-2^-lam) * H)   (global lam)

Per-node records give, for every controller, the quantities used by the
allocation analysis in the paper (Section 4):

  w_Q(h)   probability that the controlled process visits history h (not yet looped)
  w_P(h)   the same under the reference
  sigma2   Var_{p(.|h)}[H(h, v)]
  delta    one-step hazard reduction  E_p[H] - E_q[H]
  lam_h    implied local multiplier   delta / sigma2   (nats)

and two exact identities are asserted:
  (I1)  P(E) - Q(E) = sum_h w_Q(h) delta(h)                 (performance difference)
  (I2)  sum_h w_P(h) sigma2(h) = p (1 - p)                   (variance of a martingale)

Usage
-----
  python3 budget_ladder.py --out budget_ladder.json
"""
import argparse
import json
import math
import time

import numpy as np

import synthetic_fsm_hazard as fsm
from one_step_projection import project_batch, SolverFailure

LN2 = math.log(2.0)


def d_ber_bits(eps, p):
    """Bernoulli KL in bits; evaluated in 40-digit arithmetic when |p-eps| is tiny, where the two
    terms of the float64 formula cancel."""
    if abs(p - eps) < 1e-3:
        try:
            import mpmath as mp
            mp.mp.dps = 40
            e, q = mp.mpf(eps), mp.mpf(p)
            return float((e * mp.log(e / q) + (1 - e) * mp.log((1 - e) / (1 - q))) / mp.log(2))
        except ImportError:
            pass
    v = 0.0
    if eps > 0:
        v += eps * math.log2(eps / p)
    if eps < 1:
        v += (1 - eps) * math.log2((1 - eps) / (1 - p))
    return v


class System:
    """Finite-state chain of synthetic_fsm_hazard.py with a loop event of persistence R:
    the event occurs when the chain enters a state already in the trailing window of the
    last W states R times in a row (R=1 is the event of Section 5 of the paper)."""

    def __init__(self, gateway=0.05, enter=0.75, cycle=0.80, temp=0.6, horizon=6, window=2, R=1):
        fsm.P_GATEWAY, fsm.P_ENTER, fsm.P_CYCLE = gateway, enter, cycle
        self.S = fsm.S
        self.W = window
        self.Hh = horizon
        self.R = R
        self.P = fsm.temper(fsm.build_base_matrix(), temp)
        self._build_f()
        self.p = self.f[horizon][(0, (0,), 0)]
        self._hcache = {}
        self.wP = self._reference_weights()

    def _step(self, s, win, c, sp):
        """Returns (looped_now, new_key) for moving from (s, win, c) to state sp."""
        if sp in win:
            c2 = c + 1
            if c2 >= self.R:
                return True, None
        else:
            c2 = 0
        return False, (sp, (win + (sp,))[-self.W:], c2)

    def _build_f(self):
        wins = []
        for k in range(0, self.W + 1):
            wins.extend(fsm.windows_of_length(k))
        f = {0: {(s, w, c): 0.0 for s in range(self.S) for w in wins for c in range(self.R)}}
        for r in range(1, self.Hh + 1):
            f[r] = {}
            for (s, w, c) in f[0]:
                tot = 0.0
                for sp in range(self.S):
                    pr = self.P[s, sp]
                    looped, key = self._step(s, w, c, sp)
                    tot += pr * (1.0 if looped else f[r - 1][key])
                f[r][(s, w, c)] = tot
        self.f = f

    def hvec(self, remaining, s, win, c):
        key = (remaining, s, win, c)
        v = self._hcache.get(key)
        if v is None:
            out = np.empty(self.S)
            for sp in range(self.S):
                looped, k2 = self._step(s, win, c, sp)
                out[sp] = 1.0 if looped else self.f[remaining][k2]
            self._hcache[key] = out
            v = out
        return v

    def _reference_weights(self):
        wP = {}
        dist = {(0, (0,), 0): 1.0}
        for t in range(self.Hh):
            new = {}
            for (s, win, c), m in dist.items():
                wP[(t, s, win, c)] = m
                for sp in range(self.S):
                    looped, key = self._step(s, win, c, sp)
                    if looped:
                        continue
                    new[key] = new.get(key, 0.0) + m * self.P[s, sp]
            dist = new
        return wP

    # ------------------------------------------------------------------
    def forward(self, rule, param, record=False):
        """Exact forward propagation. rule in {abs, rel, sched, lam, affine, ref}.
        Returns dict(eps, kl_bits, infeasible_mass, kl_infeasible, kl_active, records)."""
        S, Hh = self.S, self.Hh
        dist = {(0, (0,), 0): 1.0}
        looped = kl_tot = infeas_mass = kl_infeas = 0.0
        recs = []
        for t in range(Hh):
            rem = Hh - t - 1
            keys = list(dist.keys())
            N = len(keys)
            Prow = np.stack([self.P[k[0]] for k in keys])
            Hm = np.stack([self.hvec(rem, *k) for k in keys])
            m_p = (Prow * Hm).sum(1)
            floor = Hm.min(1)
            mean_p = m_p
            sig2 = (Prow * Hm ** 2).sum(1) - m_p ** 2
            infeas = np.zeros(N, dtype=bool)
            lam = np.zeros(N)
            Q = None
            if rule == "ref":
                pass
            elif rule in ("abs", "abs_skip", "rel", "sched"):
                if rule in ("abs", "abs_skip"):
                    bud = np.full(N, float(param))
                elif rule == "sched":
                    bud = np.full(N, float(param[t]))
                else:
                    bud = floor + float(param) * (m_p - floor)
                # one shared solver with an expanding bracket and an explicit final check
                # (raises SolverFailure instead of reporting success); lowest-risk rule when infeasible
                Q, lam, status, _ = project_batch(Prow, Hm, bud, abstain_when_infeasible=(rule == "abs_skip"))
                infeas = status == "infeasible"
            elif rule == "lam":
                lam[:] = float(param)
            if rule == "affine":
                c = 1.0 - 2.0 ** (-float(param))
                wts = Prow * (1.0 - c * Hm)
                Q = wts / wts.sum(1, keepdims=True)
            elif Q is None:
                logw = np.log(Prow) - lam[:, None] * Hm
                logw -= logw.max(1, keepdims=True)
                Q = np.exp(logw)
                Q /= Q.sum(1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                klrow = np.where(Q > 0, Q * (np.log2(Q) - np.log2(Prow)), 0.0).sum(1)
            m_q = (Q * Hm).sum(1)
            delta = m_p - m_q
            new = {}
            for i, key0 in enumerate(keys):
                s, win, c = key0
                mass = dist[key0]
                kl_tot += mass * klrow[i]
                if infeas[i]:
                    infeas_mass += mass
                    kl_infeas += mass * klrow[i]
                if record:
                    recs.append((t, s, win, mass, self.wP.get((t, s, win, c), 0.0),
                                 sig2[i], delta[i], klrow[i], bool(infeas[i]), floor[i], m_p[i]))
                for sp in range(S):
                    q = Q[i, sp]
                    if q <= 0:
                        continue
                    lp, key = self._step(s, win, c, sp)
                    if lp:
                        looped += mass * q
                    else:
                        new[key] = new.get(key, 0.0) + mass * q
            dist = new
        return {"eps": looped, "kl_bits": kl_tot, "infeasible_mass": infeas_mass / Hh,
                "kl_infeasible": kl_infeas, "records": recs}

    # ------------------------------------------------------------------
    def calibrate(self, rule, eps, lo, hi, iters=60):
        """Bisect the single scalar so the exact loop probability equals eps.
        Monotone increasing in the scalar for abs/rel; decreasing for lam/affine."""
        increasing = rule in ("abs", "abs_skip", "rel")
        a, b = lo, hi
        for _ in range(iters):
            mid = 0.5 * (a + b)
            e = self.forward(rule, mid)["eps"]
            if (e > eps) == increasing:
                b = mid
            else:
                a = mid
        # return the side that satisfies eps from below (loop prob <= eps)
        cand = a if increasing else b
        return cand, self.forward(rule, cand, record=True)


def allocation_stats(sys_, res, p):
    """Exact identities and the implied-multiplier dispersion for one controller."""
    recs = res["records"]
    wQ = np.array([r[3] for r in recs])
    wP = np.array([r[4] for r in recs])
    sig2 = np.array([r[5] for r in recs])
    delta = np.array([r[6] for r in recs])
    infeas = np.array([r[8] for r in recs])
    kl = np.array([r[7] for r in recs])
    i1 = float((wQ * delta).sum())
    out = {"I1_residual": abs(i1 - (p - res["eps"]))}
    ok = sig2 > 1e-12
    # quadratic-regime prediction (weights w_P): KL ~ sum w_P delta^2 / (2 sigma2 ln2)
    kq = float((wP[ok] * delta[ok] ** 2 / (2 * sig2[ok] * LN2)).sum())
    D = float((wP * delta).sum())
    var_sum = float((wP * sig2).sum())
    R_pred = float((wP[ok] * delta[ok] ** 2 / sig2[ok]).sum() * var_sum / D ** 2) if D > 0 else float("nan")
    # dispersion of the implied multiplier lam_h = delta / sigma2, weights pi ~ w_P sigma2
    lam_h = np.where(ok, delta / np.where(ok, sig2, 1.0), 0.0)
    pi = wP * sig2
    pi = pi / pi.sum()
    mu = float((pi * lam_h).sum())
    cv2 = float((pi * lam_h ** 2).sum() / mu ** 2 - 1.0) if mu > 0 else float("nan")
    # same, restricted to nodes that are feasible (exclude collapsed nodes)
    feas = ~infeas
    pi_f = pi * feas
    cv2_feas = float("nan")
    if pi_f.sum() > 0:
        pf = pi_f / pi_f.sum()
        muf = float((pf * lam_h).sum())
        if muf > 0:
            cv2_feas = float((pf * lam_h ** 2).sum() / muf ** 2 - 1.0)
    active_mass = float(((np.abs(delta) > 1e-9) * wQ).sum())
    phi = float((wP * sig2 * (np.abs(delta) > 1e-12)).sum() / var_sum)   # share of event variance on touched nodes
    out.update({"quad_kl_pred": kq, "D_first_order": D, "var_sum_wP_sigma2": var_sum,
                "R_pred_quadratic": R_pred, "cv2_lambda": cv2, "cv2_lambda_feasible": cv2_feas,
                "mean_lambda_h": mu, "kl_share_infeasible": res["kl_infeasible"] / res["kl_bits"]
                if res["kl_bits"] > 0 else 0.0,
                "mass_active_over_total": active_mass / float(wQ.sum()),
                "active_variance_share": phi, "R_lower_1_over_phi": 1.0 / phi if phi > 0 else float("nan")})
    return out


SUITE = {
    "base (R=1, H=6)":          dict(),
    "persistent R=2, H=6":      dict(R=2, horizon=6),
    "persistent R=2, H=8":      dict(R=2, horizon=8),
    "persistent R=3, H=8":      dict(R=3, horizon=8),
    "rarer gateway (0.02)":     dict(gateway=0.02),
    "commoner gateway (0.10)":  dict(gateway=0.10),
    "weaker funnel (enter .5)": dict(enter=0.5),
    "temperature 0.8":          dict(temp=0.8),
}
RULES = {"abs": (0.0, 1.0), "abs_skip": (0.0, 1.0), "rel": (0.0, 1.0),
         "lam": (0.0, 60.0), "affine": (0.0, 60.0)}


def run_system(name, kw, fractions, with_sched, absolute_grid=None, verbose=True):
    sy = System(**kw)
    p = sy.p
    var_ref = sum(r[4] * r[5] for r in sy.forward("ref", 0.0, record=True)["records"])
    assert abs(var_ref - p * (1 - p)) < 1e-9, (name, var_ref, p * (1 - p))
    grid = absolute_grid if absolute_grid is not None else [round(p * f, 4) for f in fractions]
    rows = []
    if verbose:
        print(f"\n== {name}: p = {p:.6f}   (identity I2: sum w_P sigma^2 = p(1-p) holds)")
        print(f"{'eps':>7} {'rule':>9} {'KL':>8} {'KL/dBer':>8} {'infeas%':>8} {'KL@inf%':>8} {'CV2':>9} {'Rpred':>8}")
    for eps in grid:
        opt = d_ber_bits(eps, p)
        row = {"eps": eps, "d_ber": opt, "rules": {}}
        for rule, (lo, hi) in RULES.items():
            param, res = sy.calibrate(rule, eps, lo, hi)
            if abs(res["eps"] - eps) > 1e-6:      # rule cannot reach this target
                row["rules"][rule] = {"reachable": False, "eps_achieved": res["eps"]}
                continue
            st = allocation_stats(sy, res, p)
            assert st["I1_residual"] < 1e-9, st["I1_residual"]
            if rule == "affine":   # Proposition 2: sequential computation equals the closed form
                gap = abs(res["kl_bits"] - d_ber_bits(res["eps"], p))
                assert gap < 1e-9, (name, eps, gap)
            e = {"reachable": True, "param": param, "eps_achieved": res["eps"],
                 "kl_bits": res["kl_bits"], "ratio": res["kl_bits"] / opt,
                 "infeasible_mass": res["infeasible_mass"]}
            e.update(st)
            row["rules"][rule] = e
            if verbose:
                print(f"{eps:>7.3f} {rule:>9} {res['kl_bits']:>8.4f} {res['kl_bits']/opt:>8.3f} "
                      f"{100*res['infeasible_mass']:>8.1f} {100*st['kl_share_infeasible']:>8.1f} "
                      f"{st['cv2_lambda']:>9.3f} {st['R_pred_quadratic']:>8.3f}")
        if with_sched:
            e = optimise_schedule(sy, eps, opt, row["rules"]["abs"].get("param", 0.2))
            row["rules"]["sched"] = e
            if verbose:
                print(f"{eps:>7.3f} {'sched':>9} {e['kl_bits']:>8.4f} {e['ratio']:>8.3f}")
        rows.append(row)
    return {"name": name, "kwargs": kw, "p": p, "grid": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="budget_ladder.json")
    ap.add_argument("--skip_sched", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    fractions = [0.85, 0.7, 0.5, 0.3, 0.15, 0.05, 0.02]
    out = {"systems": []}
    for i, (name, kw) in enumerate(SUITE.items()):
        if i == 0:
            # base system on the same absolute grid as Tables 1-3 of the paper, plus
            # three near-reference targets used to check the small-distortion law
            r = run_system(name, kw, fractions, not args.skip_sched,
                           absolute_grid=[0.68, 0.67, 0.65, 0.60, 0.50, 0.40, 0.30, 0.20,
                                          0.10, 0.05, 0.02, 0.01])
        else:
            r = run_system(name, kw, fractions, False)
        out["systems"].append(r)
    base = System()
    conv = []
    for D in [0.1, 0.03, 0.01, 0.003, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        eps = base.p - D
        opt = d_ber_bits(eps, base.p)
        entry = {"D": D, "min_kl_bits": opt}
        for rule, (lo, hi) in [("abs", (0.0, 1.0)), ("rel", (0.0, 1.0)), ("lam", (0.0, 60.0))]:
            param, res = base.calibrate(rule, eps, lo, hi, iters=90)
            st = allocation_stats(base, res, base.p)
            entry[rule] = {"exact_ratio": res["kl_bits"] / opt, "kl_bits": res["kl_bits"],
                           "R_pred": st["R_pred_quadratic"], "R_lower_1_over_phi": st["R_lower_1_over_phi"]}
        conv.append(entry)
    # Limit of the fixed-budget ratio as D -> 0: only the histories at the maximal hazard level are
    # touched, so the second-order ratio tends to 1/phi_max (finite), not to infinity.
    ref = base.forward("ref", 0.0, record=True)["records"]
    level = max(r[10] for r in ref)
    top = [r for r in ref if abs(r[10] - level) < 1e-12]
    phi_max = sum(r[4] * r[5] for r in top) / (base.p * (1 - base.p))
    out["fixed_budget_limit"] = {"max_hazard_level": level, "n_histories_at_max": len(top),
                                 "phi_max": phi_max, "one_over_phi_max": 1.0 / phi_max}
    out["small_distortion"] = conv
    from scipy.stats import spearmanr
    xs, ys = [], []
    for sy in out["systems"]:
        for r in sy["grid"]:
            for k in ["abs", "rel", "lam", "affine"]:
                e = r["rules"].get(k)
                if e and e.get("reachable", True) and "ratio" in e and e["ratio"] > 0:
                    xs.append(math.log(1 + e["cv2_lambda"])); ys.append(math.log(e["ratio"]))
    out["spearman_logratio_vs_log1pCV2"] = {"n": len(xs), "rho": float(spearmanr(xs, ys)[0])}
    out["elapsed_s"] = time.time() - t0
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\ndone in {time.time()-t0:.0f}s -> {args.out}")


def optimise_schedule(sys_, eps, opt, b0):
    from scipy.optimize import minimize
    H = sys_.Hh
    cache = {}

    def ev(x):
        key = tuple(np.round(x, 12))
        if key not in cache:
            cache[key] = sys_.forward("sched", np.clip(x, 0.0, 1.0))
        return cache[key]

    cons = {"type": "eq", "fun": lambda x: ev(x)["eps"] - eps}
    best = None
    for start in [np.full(H, b0), np.linspace(1.5 * b0, 0.5 * b0, H), np.linspace(0.5 * b0, 1.5 * b0, H)]:
        r = minimize(lambda x: ev(x)["kl_bits"], np.clip(start, 1e-3, 1.0), method="SLSQP",
                     bounds=[(0.0, 1.0)] * H, constraints=[cons],
                     options={"maxiter": 200, "ftol": 1e-10})
        e = ev(r.x)
        if abs(e["eps"] - eps) < 1e-6 and (best is None or e["kl_bits"] < best[1]["kl_bits"]):
            best = (r.x, e)
    if best is None:
        return {"kl_bits": float("nan"), "ratio": float("nan")}
    return {"budgets": [float(v) for v in best[0]], "eps_achieved": best[1]["eps"],
            "kl_bits": best[1]["kl_bits"], "ratio": best[1]["kl_bits"] / opt,
            "infeasible_mass": best[1]["infeasible_mass"]}


if __name__ == "__main__":
    main()
