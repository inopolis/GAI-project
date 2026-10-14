"""One rule for a one-step budget that cannot be met, in the exact code and in the decoder.

Checks (1) the rule itself, (2) that the language-model decoder (float32, src/decoding.py) and the exact
solver (float64, one_step_projection.py) return the same distribution on random problems in every regime
(inactive, interior, boundary, infeasible), and (3) that failures are reported, not treated as success.
"""
import math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from one_step_projection import project_one_step, project_batch, SolverFailure
from src.decoding import RecurrenceRiskDecoder


def test_lowest_risk_rule():
    p = np.array([0.5, 0.3, 0.15, 0.05])
    r = np.array([0.40, 0.40, 0.45, 0.90])          # floor 0.40 attained by two actions
    q, lam, st, ach = project_one_step(p, r, 0.2)    # budget below the floor
    assert st == "infeasible" and math.isinf(lam)
    assert np.allclose(q, [0.5 / 0.8, 0.3 / 0.8, 0.0, 0.0]) and abs(ach - 0.4) < 1e-12
    q, lam, st, ach = project_one_step(p, r, 0.4)    # exactly at the floor: boundary, same distribution
    assert st == "boundary" and np.allclose(q, [0.5 / 0.8, 0.3 / 0.8, 0.0, 0.0])


def test_interior_meets_budget_even_for_tiny_headroom():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(3, 30)
        p = rng.dirichlet(np.ones(n))
        r = rng.random(n)
        floor, mean = r.min(), float((p * r).sum())
        b = floor + 10.0 ** rng.uniform(-9, -0.1) * (mean - floor)
        q, lam, st, ach = project_one_step(p, r, b)
        assert st in ("interior", "boundary", "inactive") and ach <= b + 1e-12


def test_solver_failure_is_reported():
    import one_step_projection as op
    old = op.LAM_CAP
    op.LAM_CAP = 4.0                                  # an artificially small cap must raise, not "succeed"
    try:
        p = np.array([0.5, 0.5]); r = np.array([0.0, 1.0])
        try:
            project_one_step(p, r, 1e-6)
        except SolverFailure:
            return
        raise AssertionError("SolverFailure was not raised")
    finally:
        op.LAM_CAP = old


def test_decoder_matches_exact_solver():
    rng = np.random.default_rng(1)
    torch.manual_seed(0)
    n_checked = {"inactive": 0, "interior": 0, "infeasible": 0}
    for trial in range(300):
        n = int(rng.integers(5, 40))
        p = rng.dirichlet(np.ones(n) * 0.3)
        p = np.maximum(p, 1e-9); p /= p.sum()
        levels = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        r = levels[rng.integers(0, 5, n)]
        if trial % 3 == 0:
            r = np.maximum(r, 0.25)                  # no zero-risk action: small budgets are infeasible
        b = float(rng.choice([0.01, 0.05, 0.3, 0.6, 0.9]))
        dec = RecurrenceRiskDecoder(temperature=1.0, mode="dual", eps=b, n_min=3, n_max=6)
        log_p = torch.log(torch.tensor(p, dtype=torch.float32))
        risk = torch.tensor(r, dtype=torch.float32)
        d = dec._solve_dual_lambda(log_p, risk)
        q_dec = (torch.exp(d["log_q"]) if "log_q" in d else
                 torch.exp(dec._log_q_of_lambda(log_p, risk, d["lambda"]))).numpy()
        q_ex, lam, st, ach = project_one_step(p, r, b)
        assert np.abs(q_dec - q_ex).max() < 2e-3, (trial, st, np.abs(q_dec - q_ex).max())
        assert (st == "infeasible") == bool(d["structurally_infeasible"]), (trial, st)
        n_checked[st if st in n_checked else "interior"] += 1
    assert all(v > 10 for v in n_checked.values()), n_checked


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok:", name)
    print("OK: one rule for an unmet budget; decoder and exact solver agree; failures are raised.")
