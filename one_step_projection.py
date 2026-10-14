"""
one_step_projection.py -- the one-step KL projection used by every exact experiment.

Problem (P):   min_q KL(q || p)   s.t.   E_q[r] <= b        (p > 0 everywhere)

Rule (stated in the paper, Section 3; identical in src/decoding.py):

  * inactive    E_p[r] <= b          -> q = p.
  * interior    floor + tol < b < E_p[r]  -> q = p exp(-lam r)/Z with the smallest lam found by an
                EXPANDING bracket (doubling from lam=1 until E_q[r] <= b, no fixed cap other than
                LAM_CAP) followed by bisection; the final q is CHECKED: E_q[r] <= b + tol.
                If the check fails the call raises SolverFailure -- it is never reported as success.
  * lowest-risk b <= floor + tol     -> q = p restricted to the minimum-risk actions
                {v : r(v) <= floor + tol}, renormalized (the lam -> infinity limit of the projection and
                the minimum-KL distribution among those attaining the smallest achievable risk).
                The step is flagged `infeasible` if b < floor - tol (constraint violated by floor - b)
                and `boundary` if |b - floor| <= tol (constraint met with equality up to tol).

floor = min_v r(v); tol = 1e-12 (risk is computed in float64 here; the float32 language-model decoder
uses its own tolerance of 1e-6, Appendix C).
"""
import numpy as np

TOL = 1e-12
LAM_CAP = 1e9


class SolverFailure(RuntimeError):
    """The expanding bracket reached LAM_CAP, or the final distribution violates the budget."""


def _tilt(logp, R, lam):
    lw = logp - lam[:, None] * R
    lw -= lw.max(1, keepdims=True)
    q = np.exp(lw)
    return q / q.sum(1, keepdims=True)


def project_batch(P, R, b, abstain_when_infeasible=False, iters=100):
    """Rows are independent one-step problems. Returns (Q, lam, status, achieved) where status is one
    of 'inactive', 'interior', 'boundary', 'infeasible' (strings), lam is np.inf for the lowest-risk
    rule, and achieved = E_q[r]. Raises SolverFailure on any numerical failure."""
    P = np.asarray(P, float)
    R = np.asarray(R, float)
    b = np.asarray(b, float)
    N = P.shape[0]
    m = (P * R).sum(1)
    floor = R.min(1)
    status = np.full(N, "inactive", dtype=object)
    active = m > b
    infeas = active & (b < floor - TOL)
    boundary = active & ~infeas & (b <= floor + TOL)
    interior = active & ~infeas & ~boundary
    status[infeas] = "infeasible"
    status[boundary] = "boundary"
    status[interior] = "interior"
    Q = P.copy()
    lam = np.zeros(N)
    logp = np.log(P)

    # lowest-risk rule
    low = boundary | (infeas & (not abstain_when_infeasible))
    if low.any():
        mask = (R[low] <= floor[low, None] + TOL).astype(float)
        q = P[low] * mask
        Q[low] = q / q.sum(1, keepdims=True)
        lam[low] = np.inf

    # interior rows: expanding bracket, then bisection
    idx = np.where(interior)[0]
    if idx.size:
        Pi, Ri, bi, lpi = P[idx], R[idx], b[idx], logp[idx]
        hi = np.ones(idx.size)
        while True:
            g = (_tilt(lpi, Ri, hi) * Ri).sum(1)
            bad = g > bi
            if not bad.any():
                break
            if (hi[bad] >= LAM_CAP).any():
                raise SolverFailure(f"lam cap {LAM_CAP:g} reached with E_q[r]-b = "
                                    f"{float((g - bi)[bad].max()):.3e} (b - floor = {float((bi - floor[idx])[bad].min()):.3e})")
            hi[bad] *= 2.0
        lo = np.where(hi > 1.0, hi / 2.0, 0.0)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            over = (_tilt(lpi, Ri, mid) * Ri).sum(1) > bi
            lo = np.where(over, mid, lo)
            hi = np.where(over, hi, mid)
        qi = _tilt(lpi, Ri, hi)
        gi = (qi * Ri).sum(1)
        if (gi > bi + TOL).any():
            raise SolverFailure(f"final check failed: max E_q[r]-b = {float((gi - bi).max()):.3e}")
        Q[idx] = qi
        lam[idx] = hi

    achieved = (Q * R).sum(1)
    ok = (achieved <= b + TOL) | (status == "infeasible") | (m <= b)
    if not ok.all():
        raise SolverFailure("budget not met at a node that was classified feasible")
    return Q, lam, status, achieved


def project_one_step(p_row, r_row, b, abstain_when_infeasible=False):
    Q, lam, st, ach = project_batch(np.asarray(p_row)[None, :], np.asarray(r_row)[None, :],
                                    np.asarray([b]), abstain_when_infeasible)
    return Q[0], float(lam[0]), str(st[0]), float(ach[0])
