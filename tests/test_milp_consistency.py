"""MILP-vs-enumeration consistency test (SPEC_modelo_ofertas.md §10).

Builds an independent closed-form "enumeration" oracle for SCO and checks the
MILP-based SCOStrategy reproduces it exactly on several small instances. The
oracle is precisely the closed form of the "aware" (declared surplus) clearing
model with MAV/technical constraints switched off: full dispatch whenever a
period is in-the-money (lambda >= P^V), day accepted iff the declared surplus
covers TF, real costs applied only afterwards on the fixed dispatch.
"""

import numpy as np
import pytest

from bidding.config import ResolvedGrid, RiskObjective
from reporting.optimizer import build_sco_model_aware, extract_sco_results, solve_model
from bidding.orders.sco import SCOStrategy, evaluate_sco_theta

from .conftest import avail_matrix, make_tech

STRAT = SCOStrategy()
W = RiskObjective()


def enumerate_sco(tech, lambda_matrix, avail, pv, tf):
    """Independent closed-form oracle for the revised aware SCO semantics.

    Once the day is accepted by the day-level MIC, the entire daily profile is
    dispatched in every period. The market does not screen hours one-by-one for
    period-level profitability; the only acceptance gate is the day-level
    declared surplus.
    """
    cost = np.asarray(tech.cost_array(lambda_matrix.shape[1]), dtype=float)
    mic_surplus = np.sum((lambda_matrix - pv) * avail, axis=1) - tf
    u = (mic_surplus >= 0.0).astype(float)
    q = avail * u[:, None]
    margin = lambda_matrix - cost[None, :]
    profits = np.sum(margin * q, axis=1)
    if tech.startup_cost > 0.0:
        profits = profits - tech.startup_cost * u
    return q, u, profits


@pytest.mark.parametrize(
    "prices,pv,tf,cost,startup",
    [
        (np.full((3, 24), 50.0), 10.0, 1_000.0, 0.0, 0.0),
        (np.full((2, 24), 20.0), 10.0, 50_000.0, 0.0, 0.0),   # MIC unmet -> reject
        # step price with a strictly positive margin in every in-the-money period,
        # to avoid a zero-margin tie where the MILP and closed form may legitimately
        # pick different (equally optimal) dispatch levels
        ([[5.0] * 12 + [100.0] * 12], 0.0, 0.0, 0.0, 0.0),
        (np.full((4, 24), 60.0), 0.0, 0.0, 0.0, 1000.0),       # startup cost
        # mixed ITM/OTM day, mav_frac=0: the OTM half (lambda=5 < pv=40)
        # contributes negatively to the declared surplus, so the aware MILP
        # leaves it at 0 -- matching the oracle's price-based delta rule.
        ([[60.0] * 12 + [5.0] * 12], 40.0, 0.0, 30.0, 0.0),
    ],
)
def test_milp_matches_enumeration_oracle(prices, pv, tf, cost, startup):
    tech = make_tech([100.0] * 24, cost=cost, startup=startup, family=3)
    lam = np.asarray(prices, dtype=float)
    S = lam.shape[0]
    probs = np.ones(S) / S
    avail = avail_matrix(tech, S)

    grid = ResolvedGrid(price_levels=[pv], mar_levels=[0.5], mav_fraction_levels=[0.0], tf_levels=[tf])
    result = STRAT.evaluate(tech, lam, avail, probs, grid, W, 0.95)

    q_oracle, u_oracle, profits_oracle = enumerate_sco(tech, lam, avail, pv, tf)

    np.testing.assert_allclose(result["dispatch"], q_oracle, atol=1e-6)
    np.testing.assert_allclose(result["matched"], u_oracle, atol=1e-6)
    np.testing.assert_allclose(result["profits"], profits_oracle, atol=1e-6)


class TestClosedFormMatchesAwareMILP:
    """The production aware path is the closed form (_clear_sco_aware); the
    MILP formulation (build_sco_model_aware) is kept as an independent
    cross-check. Instances avoid lambda == P^V ties, where the MILP's choice
    is legitimately arbitrary (declared contribution 0), and include the MAV
    floor and the technical minimum so those constraints are exercised too."""

    @pytest.mark.parametrize("mav_frac,enforce_qmin", [(0.5, False), (0.9, True)])
    def test_small_instances(self, mav_frac, enforce_qmin):
        tech = make_tech(
            [100.0] * 24, cost=30.0, startup=1_000.0, family=3,
            technical_min=40.0 if enforce_qmin else 0.0,
            enforce_technical_constraints=enforce_qmin,
        )
        lam = np.array([
            [50.0] * 12 + [5.0] * 12,   # surplus (50-25)*100*12 = 30 000 >= TF -> accepted
            [26.0] * 24,                # surplus (26-25)*100*24 =  2 400 <  TF -> rejected
            [24.0] * 24,                # every hour OTM -> rejected
        ])
        S, T = lam.shape
        probs = np.ones(S) / S
        avail = avail_matrix(tech, S)
        theta = {"price_variable": 25.0, "fixed_term": 3_000.0, "mav_fraction": mav_frac}

        q_cf, u_cf, _ = evaluate_sco_theta(tech, lam, avail, probs, theta, 0.95, W, "aware")

        model = build_sco_model_aware(tech, lam, avail, probs, theta)
        solve_model(model)
        q_milp, u_milp = extract_sco_results(model, S, T)
        u_milp = (q_milp.sum(axis=1) > 1e-9).astype(float)  # same no-dispatch guard

        np.testing.assert_allclose(q_cf, q_milp, atol=1e-6)
        np.testing.assert_allclose(u_cf, u_milp)
