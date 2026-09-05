"""Unit tests for SBO welfare condition and block dispatch."""

import numpy as np
import pytest

from bidding.config import ResolvedGrid
from bidding.orders.sbo import SBOStrategy

from .conftest import W, avail_matrix, make_tech

STRAT = SBOStrategy()


def single_combo_grid(pb: float, mar: float = 0.5) -> ResolvedGrid:
    return ResolvedGrid(
        price_levels=[pb],
        mar_levels=[mar],
        mav_fraction_levels=[0.5],
        tf_levels=[0],
    )


class TestWelfareCondition:
    def test_welfare_positive_block_accepted(self):
        """Σ(λ−P^B)·Q̂ = (50−40)·100·24 > 0 → u^s=1."""
        tech = make_tech([100.0] * 24)
        lam = np.full((2, 24), 50.0)
        probs = np.ones(2) / 2
        result = STRAT.evaluate(
            tech, lam, avail_matrix(tech, 2), probs, single_combo_grid(40.0), W, 0.95
        )

        assert result["match_probability"] == pytest.approx(1.0)
        np.testing.assert_allclose(result["dispatch"], 100.0)

    def test_welfare_negative_block_rejected(self):
        """Σ(λ−P^B)·Q̂ = (30−40)·100·24 < 0 → u^s=0."""
        tech = make_tech([100.0] * 24)
        lam = np.full((2, 24), 30.0)
        probs = np.ones(2) / 2
        result = STRAT.evaluate(
            tech, lam, avail_matrix(tech, 2), probs, single_combo_grid(40.0), W, 0.95
        )

        assert result["match_probability"] == pytest.approx(0.0)
        np.testing.assert_allclose(result["dispatch"], 0.0)

    def test_welfare_exactly_zero_accepted(self):
        """Welfare = 0 boundary: spec requires ≥ 0 → accepted."""
        tech = make_tech([100.0] * 24)
        lam = np.full((1, 24), 40.0)
        probs = np.array([1.0])
        result = STRAT.evaluate(
            tech, lam, avail_matrix(tech, 1), probs, single_combo_grid(40.0), W, 0.95
        )

        assert result["match_probability"] == pytest.approx(1.0)

    def test_non_uniform_availability(self):
        """Solar profile: Q̂_t varies; welfare uses actual Q̂_t per period."""
        avail = [0, 0, 0, 0, 0, 0, 10, 30, 60, 80, 90, 95,
                 95, 90, 80, 60, 40, 15, 5, 0, 0, 0, 0, 0]
        tech = make_tech(avail)
        lam = np.full((1, 24), 50.0)
        probs = np.array([1.0])
        # welfare = (50 - P^B) * sum(avail); sum=750
        # P^B=40 → welfare=(50-40)*750=7 500 > 0 → accepted
        result = STRAT.evaluate(
            tech, lam, avail_matrix(tech, 1), probs, single_combo_grid(40.0), W, 0.95
        )

        assert result["match_probability"] == pytest.approx(1.0)
        expected_profit = (50.0 - 0.0) * sum(avail)   # cost=0
        assert result["expected_profit"] == pytest.approx(expected_profit, rel=1e-9)


class TestSBOOptimality:
    def test_lower_block_price_wins_low_market(self):
        """With consistently low market prices, lower P^B matches more → higher E[Π]."""
        tech = make_tech([100.0] * 24, cost=0.0)
        lam = np.full((10, 24), 25.0)
        probs = np.ones(10) / 10
        grid = ResolvedGrid(
            price_levels=[0, 10, 20, 30, 40],
            mar_levels=[0.5],
            mav_fraction_levels=[0.5],
            tf_levels=[0],
        )
        result = STRAT.evaluate(tech, lam, avail_matrix(tech, 10), probs, grid, W, 0.95)

        # P^B=20 → welfare=(25-20)*100*24>0, accepted; P^B=30 → rejected
        assert result["optimal_params"]["block_price"] <= 20.0

    def test_startup_cost_applied_once_per_accepted_scenario(self):
        """Each accepted scenario pays startup cost once."""
        tech = make_tech([100.0] * 24, cost=0.0, startup=2_000.0)
        lam = np.full((3, 24), 60.0)
        probs = np.ones(3) / 3
        result = STRAT.evaluate(
            tech, lam, avail_matrix(tech, 3), probs, single_combo_grid(0.0), W, 0.95
        )

        gross = 60.0 * 100.0 * 24       # 144 000
        expected = gross - 2_000.0       # 142 000
        assert result["expected_profit"] == pytest.approx(expected, rel=1e-9)

    def test_mixed_scenarios_expected_profit(self):
        """s0: price=80 (accepted, profit>0), s1: price=5 (rejected, profit=0)."""
        tech = make_tech([100.0] * 24, cost=0.0)
        lam = np.array([[80.0] * 24, [5.0] * 24])
        probs = np.array([0.5, 0.5])
        grid = single_combo_grid(pb=10.0)
        result = STRAT.evaluate(tech, lam, avail_matrix(tech, 2), probs, grid, W, 0.95)

        profit_s0 = 80.0 * 100.0 * 24      # 192 000
        profit_s1 = 0.0
        expected = 0.5 * profit_s0 + 0.5 * profit_s1
        assert result["expected_profit"] == pytest.approx(expected, rel=1e-9)
        assert result["match_probability"] == pytest.approx(0.5)
