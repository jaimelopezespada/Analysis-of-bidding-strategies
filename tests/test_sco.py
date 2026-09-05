"""Unit tests for SCO acceptance rule (MIC / full-day reject)."""

import numpy as np
import pytest

from bidding.config import ResolvedGrid
from bidding.orders.sco import SCOStrategy

from .conftest import W, avail_matrix, make_tech

STRAT = SCOStrategy()


def single_combo_grid(pv: float, tf: float, mav_frac: float) -> ResolvedGrid:
    """Grid with exactly one combination — isolates a specific SCO scenario."""
    return ResolvedGrid(
        price_levels=[pv],
        mar_levels=[0.5],
        mav_fraction_levels=[mav_frac],
        tf_levels=[tf],
    )


class TestMICCondition:
    def test_mic_met_order_accepted(self):
        """Revenue (100-10)*100*24 = 216 000 >> TF=1 000 → u^s=1."""
        tech = make_tech([100.0] * 24, family=3)
        lam = np.full((2, 24), 100.0)
        probs = np.ones(2) / 2
        grid = single_combo_grid(pv=10, tf=1_000, mav_frac=0.5)
        result = STRAT.evaluate(tech, lam, avail_matrix(tech, 2), probs, grid, W, 0.95)

        assert result["match_probability"] == pytest.approx(1.0)
        np.testing.assert_allclose(result["dispatch"], 100.0)

    def test_mic_not_met_full_day_rejected(self):
        """Revenue (20-10)*100*24 = 24 000 < TF=50 000 → u^s=0."""
        tech = make_tech([100.0] * 24, family=3)
        lam = np.full((1, 24), 20.0)
        probs = np.array([1.0])
        grid = single_combo_grid(pv=10, tf=50_000, mav_frac=0.5)
        result = STRAT.evaluate(tech, lam, avail_matrix(tech, 1), probs, grid, W, 0.95)

        np.testing.assert_allclose(result["dispatch"], 0.0)
        assert result["match_probability"] == pytest.approx(0.0)
        assert result["expected_profit"] == pytest.approx(0.0)

    def test_mic_boundary_exact_zero_accepted(self):
        """MIC surplus = 0 exactly → accepted (condition is ≥ 0)."""
        tech = make_tech([100.0] * 24, family=3)
        # TF = (50-10)*100*24 = 96 000 exactly
        tf_exact = (50.0 - 10.0) * 100.0 * 24
        lam = np.full((1, 24), 50.0)
        probs = np.array([1.0])
        grid = single_combo_grid(pv=10, tf=tf_exact, mav_frac=0.5)
        result = STRAT.evaluate(tech, lam, avail_matrix(tech, 1), probs, grid, W, 0.95)

        assert result["match_probability"] == pytest.approx(1.0)

    def test_price_below_pv_never_active(self):
        """Market price < P^V → δ_t^s=0 → MIC=−TF<0 → rejected even with TF=0."""
        tech = make_tech([100.0] * 24, family=3)
        lam = np.full((2, 24), 5.0)
        probs = np.ones(2) / 2
        grid = single_combo_grid(pv=20, tf=0, mav_frac=0.0)
        result = STRAT.evaluate(tech, lam, avail_matrix(tech, 2), probs, grid, W, 0.95)

        np.testing.assert_allclose(result["dispatch"], 0.0)

    def test_mixed_scenarios_partial_match(self):
        """2 scenarios: s0 price=100 (MIC met), s1 price=5 (MIC not met)."""
        tech = make_tech([100.0] * 24, family=3)
        lam = np.array([
            [100.0] * 24,  # s0: met
            [5.0] * 24,    # s1: not met
        ])
        probs = np.array([0.5, 0.5])
        grid = single_combo_grid(pv=10, tf=1_000, mav_frac=0.5)
        result = STRAT.evaluate(tech, lam, avail_matrix(tech, 2), probs, grid, W, 0.95)

        assert result["match_probability"] == pytest.approx(0.5)
        np.testing.assert_allclose(result["dispatch"][0], 100.0)
        np.testing.assert_allclose(result["dispatch"][1], 0.0)


class TestMixedITMOTM:
    """How OTM periods (P^V > lambda_t^s) clear depends on the clearing model:
    under "naive" (perfect-knowledge benchmark) an OTM period is not
    structurally barred from dispatch — it is only bounded by the aggregate
    MIC and its MAV_t floor, and clears whenever it is profitable at real
    cost; under "aware" (declared surplus, default) it is never dispatched,
    because its contribution to the declared surplus is negative — matching
    real price-based clearing."""

    def test_naive_otm_period_dispatches_up_to_mic_bound(self):
        """period0 ITM (lambda=50), period1 OTM (lambda=10), pv=25, tf=0.
        With cost=0, dispatching the OTM period earns real money, and the
        naive solver (which sees that cost) uses it: aggregate surplus
        = (50-25)*100 - (25-10)*q1 >= 0 -> q1 <= 2500/15. MAV1 = 0.5*200
        = 100 <= 166.67, so the OTM period dispatches up to the MIC bound."""
        tech = make_tech([100.0, 200.0], cost=0.0, family=3)
        lam = np.array([[50.0, 10.0]])
        probs = np.array([1.0])
        grid = single_combo_grid(pv=25.0, tf=0.0, mav_frac=0.5)

        result = STRAT.evaluate(
            tech, lam, avail_matrix(tech, 1), probs, grid, W, 0.95, sco_model="naive"
        )

        q = result["dispatch"][0]
        assert result["match_probability"] == pytest.approx(1.0)
        assert q[0] == pytest.approx(100.0)
        assert q[1] == pytest.approx(2500.0 / 15.0, rel=1e-4)
        assert q[1] >= 100.0 - 1e-6  # respects MAV_1 = 0.5 * 200
        assert q[1] < 200.0  # strictly below its own availability cap

    def test_aware_otm_period_dispatches_with_day_level_mic(self):
        """With the revised aware SCO, an accepted day dispatches the full
        declared profile, including OTM periods. The day-level MIC is the only
        gate; moneyness is not used to zero out an OTM hour ex ante."""
        tech = make_tech([100.0, 200.0], cost=0.0, family=3)
        lam = np.array([[50.0, 10.0]])
        probs = np.array([1.0])
        grid = single_combo_grid(pv=15.0, tf=0.0, mav_frac=0.5)

        result = STRAT.evaluate(tech, lam, avail_matrix(tech, 1), probs, grid, W, 0.95)

        q = result["dispatch"][0]
        assert result["match_probability"] == pytest.approx(1.0)
        assert q[0] == pytest.approx(100.0)
        assert q[1] == pytest.approx(200.0, abs=1e-6)

    @pytest.mark.parametrize("sco_model", ["naive", "aware"])
    def test_otm_period_stays_in_day_profile_when_mav_floor_is_met(self, sco_model):
        """Under the revised day-level semantics, the OTM period is not
        dropped just because it is negative on a period-by-period declared
        surplus basis. Once the day clears and the MAV floor is satisfied, the
        accepted day keeps its full declared profile."""
        tech = make_tech([100.0, 200.0], cost=0.0, family=3)
        lam = np.array([[50.0, 10.0]])
        probs = np.array([1.0])
        grid = single_combo_grid(pv=15.0, tf=0.0, mav_frac=0.9)

        result = STRAT.evaluate(
            tech, lam, avail_matrix(tech, 1), probs, grid, W, 0.95, sco_model=sco_model
        )

        q = result["dispatch"][0]
        assert result["match_probability"] == pytest.approx(1.0)
        assert q[0] == pytest.approx(100.0)
        assert q[1] == pytest.approx(200.0, abs=1e-6)


class TestClearingModelContrast:
    """The economic correction that motivates the two models: the aware
    clearing cannot see the generator's real costs, so with TF=0 it must
    commit to days that lose money at real cost — the declared MIC (TF, P^V)
    is the ONLY protection. The naive benchmark self-protects using cost
    knowledge the market does not have."""

    def test_aware_commits_losing_day_naive_self_protects(self):
        """lambda=40 between pv=30 and real cost=50. Declared surplus
        (40-30)*100*24 = 24 000 >= TF=0 -> aware MUST clear and lose
        (40-50)*100*24 = -24 000. Naive sees the real cost and stays out."""
        tech = make_tech([100.0] * 24, cost=50.0, family=3)
        lam = np.full((1, 24), 40.0)
        probs = np.array([1.0])
        grid = single_combo_grid(pv=30.0, tf=0.0, mav_frac=0.0)

        r_aware = STRAT.evaluate(tech, lam, avail_matrix(tech, 1), probs, grid, W, 0.95)
        r_naive = STRAT.evaluate(
            tech, lam, avail_matrix(tech, 1), probs, grid, W, 0.95, sco_model="naive"
        )

        assert r_aware["expected_profit"] == pytest.approx(-24_000.0)
        np.testing.assert_allclose(r_aware["dispatch"], 100.0)
        assert r_naive["expected_profit"] == pytest.approx(0.0)
        np.testing.assert_allclose(r_naive["dispatch"], 0.0)

    def test_aware_tf_becomes_the_protection(self):
        """Same losing day, but the grid offers TF=100 000 as an alternative:
        the declared surplus 24 000 < 100 000 makes the MIC reject the day
        (profit 0 beats -24 000), so the optimiser now picks a strictly
        positive TF — TF is no longer trivially 0 under the aware model."""
        tech = make_tech([100.0] * 24, cost=50.0, family=3)
        lam = np.full((1, 24), 40.0)
        probs = np.array([1.0])
        grid = ResolvedGrid(
            price_levels=[30.0],
            mar_levels=[0.5],
            mav_fraction_levels=[0.0],
            tf_levels=[0.0, 100_000.0],
        )

        result = STRAT.evaluate(tech, lam, avail_matrix(tech, 1), probs, grid, W, 0.95)

        assert result["optimal_params"]["fixed_term"] == pytest.approx(100_000.0)
        assert result["expected_profit"] == pytest.approx(0.0)
        np.testing.assert_allclose(result["dispatch"], 0.0)


class TestSCOOptimality:
    def test_lower_tf_wins_when_revenue_borderline(self):
        """Between TF=0 and TF=500 000, TF=0 should win when revenue is borderline."""
        tech = make_tech([100.0] * 24, cost=0.0, family=3)
        lam = np.full((5, 24), 15.0)   # low market price
        probs = np.ones(5) / 5
        grid = ResolvedGrid(
            price_levels=[10],
            mar_levels=[0.5],
            mav_fraction_levels=[0.5],
            tf_levels=[0, 500_000],
        )
        result = STRAT.evaluate(tech, lam, avail_matrix(tech, 5), probs, grid, W, 0.95)

        assert result["optimal_params"]["fixed_term"] == pytest.approx(0.0)

    def test_startup_cost_deducted_from_accepted_scenarios(self):
        """Startup cost should reduce profit in scenarios where order is accepted."""
        tech = make_tech([100.0] * 24, cost=0.0, startup=5_000.0, family=3)
        lam = np.full((1, 24), 50.0)
        probs = np.array([1.0])
        grid = single_combo_grid(pv=0, tf=0, mav_frac=0.0)
        result = STRAT.evaluate(tech, lam, avail_matrix(tech, 1), probs, grid, W, 0.95)

        gross = 50.0 * 100.0 * 24           # 120 000
        expected = gross - 5_000.0           # 115 000
        assert result["expected_profit"] == pytest.approx(expected, rel=1e-9)
