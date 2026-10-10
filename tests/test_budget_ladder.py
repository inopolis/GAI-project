"""Checks of budget_ladder.py against independent computations (exact, fast)."""
import math, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import budget_ladder as bl
import synthetic_fsm_hazard as fsm


def test_reference_probability_matches_original_dp():
    s = bl.System()
    f = fsm.build_hazard_table(s.P, fsm.H, fsm.W)
    assert abs(s.p - f[fsm.H][(0, (0,))]) < 1e-12


def test_fixed_budget_matches_original_forward_propagation():
    s = bl.System()
    f_orig = fsm.build_hazard_table(s.P, fsm.H, fsm.W)
    for b in (0.05, 0.2, 0.4):
        r = s.forward("abs", b)
        e, k, i = fsm.controlled_forward_exact(s.P, fsm.W, fsm.H, "hazard", b, f_orig)
        assert abs(r["eps"] - e) < 1e-10 and abs(r["kl_bits"] - k) < 1e-8 and abs(r["infeasible_mass"] - i / fsm.H) < 1e-10


def test_accounting_identities():
    for kw in (dict(), dict(R=2, horizon=6), dict(R=3, horizon=8)):
        s = bl.System(**kw)
        ref = s.forward("ref", 0.0, record=True)
        assert abs(sum(r[4] * r[5] for r in ref["records"]) - s.p * (1 - s.p)) < 1e-9      # Prop. 3(b)
        for rule, par in (("abs", 0.2), ("rel", 0.3), ("lam", 2.0)):
            res = s.forward(rule, par, record=True)
            lhs = s.p - res["eps"]
            rhs = sum(r[3] * r[6] for r in res["records"])
            assert abs(lhs - rhs) < 1e-9                                                    # Prop. 3(a)


def test_affine_rule_attains_bernoulli_bound():
    s = bl.System()
    for eps in (0.5, 0.2, 0.05):
        _, res = s.calibrate("affine", eps, 0.0, 60.0)
        assert abs(res["kl_bits"] - bl.d_ber_bits(res["eps"], s.p)) < 1e-9                  # Prop. 1-2


def test_waste_law_is_exact_in_the_limit():
    s = bl.System()
    eps = s.p - 3e-4
    opt = bl.d_ber_bits(eps, s.p)
    _, res = s.calibrate("rel", eps, 0.0, 1.0, iters=70)
    st = bl.allocation_stats(s, res, s.p)
    assert abs(res["kl_bits"] / opt / st["R_pred_quadratic"] - 1.0) < 0.01                  # Prop. 4(ii)-(iii)
    _, res = s.calibrate("abs", eps, 0.0, 1.0, iters=70)
    st = bl.allocation_stats(s, res, s.p)
    assert st["R_pred_quadratic"] >= st["R_lower_1_over_phi"] - 1e-9                        # R >= 1/phi


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok:", name)
    print("OK: budget_ladder.py agrees with the original DP and satisfies Propositions 1-4.")
