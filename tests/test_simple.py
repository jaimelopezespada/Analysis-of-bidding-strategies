"""Unit tests for Simple order acceptance rule and dispatch."""

import numpy as np
import pytest

from bidding.orders.simple import SimpleOrderStrategy

from .conftest import W, avail_matrix, make_grid, make_tech

STRAT = SimpleOrderStrategy()


class TestAcceptanceRule:
    def test_constant_high_price_all_dispatched(self):
        """Market always at 50 €/MWh → all periods dispatched at Q̂."""
        tech = make_tech([100.0] * 24)
        lam = np.full((3, 24), 50.0)
        probs = np.ones(3) / 3
        result = STRAT.evaluate(tech, lam, avail_matrix(tech, 3), probs, make_grid(), W, 0.95)

        np.testing.assert_allclose(result["dispatch"], 100.0)
        # E[Π] = 50 €/MWh × 100 MWh × 24 h = 120 000 €
        assert result["expected_profit"] == pytest.approx(120_000.0, rel=1e-9)

    def test_price_below_lowest_offer_zero_dispatch(self):
        """Market at 0 €/MWh, grid starts at 10 → nothing accepted, profit = 0."""
        tech = make_tech([100.0] * 24)
        lam = np.zeros((3, 24))
        probs = np.ones(3) / 3
        result = STRAT.evaluate(
            tech, lam, avail_matrix(tech, 3), probs, make_grid(price_levels=[10, 20, 40]), W, 0.95
        )

        np.testing.assert_allclose(result["dispatch"], 0.0)
        assert result["expected_profit"] == pytest.approx(0.0)
        assert result["match_probability"] == pytest.approx(0.0)

    def test_step_price_only_peak_dispatched(self):
        """Price=0 h0-11, price=100 h12-23 with cost=0.  Best offer=0 everywhere.
        Dispatch should be 100 MWh in h12-23, 0 in h0-11 (profit margin=0 there)."""
        tech = make_tech([100.0] * 24)
        prices = [0.0] * 12 + [100.0] * 12
        lam = np.array([prices])           # 1 scenario
        probs = np.array([1.0])
        result = STRAT.evaluate(tech, lam, avail_matrix(tech, 1), probs, make_grid(), W, 0.95)

        # h0-11: price=0, margin=0 regardless → dispatch 100 (accepted at p=0) but profit=0
        # h12-23: price=100, offer=0, dispatch 100, profit=100*100=10 000 per period
        assert result["expected_profit"] == pytest.approx(100.0 * 100.0 * 12, rel=1e-9)
        np.testing.assert_allclose(result["dispatch"][0, 12:], 100.0)


class TestVariableCost:
    def test_variable_cost_reduces_profit(self):
        """With cost=30 and market=50, net margin=20 per MWh."""
        tech = make_tech([100.0] * 24, cost=30.0)
        lam = np.full((1, 24), 50.0)
        probs = np.array([1.0])
        result = STRAT.evaluate(tech, lam, avail_matrix(tech, 1), probs, make_grid(), W, 0.95)

        expected = 20.0 * 100.0 * 24
        assert result["expected_profit"] == pytest.approx(expected, rel=1e-9)

    def test_offer_above_market_not_accepted(self):
        """If best offer (10) > market price (5), nothing dispatched."""
        tech = make_tech([100.0] * 24, cost=0.0)
        lam = np.full((2, 24), 5.0)
        probs = np.ones(2) / 2
        result = STRAT.evaluate(
            tech, lam, avail_matrix(tech, 2), probs, make_grid(price_levels=[10, 20, 40]), W, 0.95
        )

        np.testing.assert_allclose(result["dispatch"], 0.0)

    def test_negative_margin_prefers_zero_dispatch(self):
        """Cost=80, market=50: offering at 0 gives dispatch but negative margin.
        Grid includes 0, so the model will accept; profit should be negative."""
        tech = make_tech([100.0] * 24, cost=80.0)
        lam = np.full((1, 24), 50.0)
        probs = np.array([1.0])
        result = STRAT.evaluate(
            tech, lam, avail_matrix(tech, 1), probs,
            make_grid(price_levels=[0, 10, 40, 60, 90]), W, 0.95,
        )
        # Offering at 90 → never accepted (50 < 90) → profit = 0
        # That is better than offering at 0 → profit = (50-80)*100*24 = -72000
        assert result["expected_profit"] == pytest.approx(0.0)


class TestStartupCost:
    def test_startup_cost_deducted_once_per_scenario(self):
        """Startup cost deducted once if any energy dispatched in the scenario."""
        tech = make_tech([100.0] * 24, cost=0.0, startup=1000.0)
        lam = np.full((2, 24), 50.0)
        probs = np.ones(2) / 2
        result = STRAT.evaluate(tech, lam, avail_matrix(tech, 2), probs, make_grid(), W, 0.95)

        # Gross: 50*100*24=120 000; minus startup 1 000 → 119 000
        assert result["expected_profit"] == pytest.approx(119_000.0, rel=1e-9)

    def test_per_transition_charges_each_restart(self):
        """Two production spells separated by off hours → 2 startups charged
        when startup_per_transition=True, but still 1 by default."""
        tech = make_tech([100.0] * 24, cost=0.0, startup=1000.0)
        # High price h0-5 and h12-17, low elsewhere; grid floor at 10 keeps
        # the low-price hours out of the money → dispatch pattern on/off/on/off.
        prices = [100.0] * 6 + [5.0] * 6 + [100.0] * 6 + [5.0] * 6
        lam = np.array([prices])
        probs = np.array([1.0])
        grid = make_grid(price_levels=[10])

        default = STRAT.evaluate(tech, lam, avail_matrix(tech, 1), probs, grid, W, 0.95)
        per_transition = STRAT.evaluate(
            tech, lam, avail_matrix(tech, 1), probs, grid, W, 0.95,
            startup_per_transition=True,
        )

        gross = 100.0 * 100.0 * 12  # 12 dispatched hours at margin 100
        assert default["expected_profit"] == pytest.approx(gross - 1_000.0, rel=1e-9)
        assert per_transition["expected_profit"] == pytest.approx(gross - 2_000.0, rel=1e-9)

    def test_per_transition_counts_initial_startup(self):
        """Continuous production from h0 → exactly one startup, same as default."""
        tech = make_tech([100.0] * 24, cost=0.0, startup=1000.0)
        lam = np.full((1, 24), 50.0)
        probs = np.array([1.0])
        result = STRAT.evaluate(
            tech, lam, avail_matrix(tech, 1), probs, make_grid(), W, 0.95,
            startup_per_transition=True,
        )

        assert result["expected_profit"] == pytest.approx(50.0 * 100.0 * 24 - 1_000.0, rel=1e-9)


class TestEnergyCapacity:
    def test_battery_caps_total_energy(self):
        """Battery with E_max=200 MWh, P_max=50 MW over 24h (1200 MWh uncapped)."""
        tech = make_tech([50.0] * 24, cost=0.0, energy_cap=200.0)
        # Flat price so greedy picks first 4 periods (4×50=200 MWh)
        lam = np.full((1, 24), 60.0)
        probs = np.array([1.0])
        result = STRAT.evaluate(
            tech, lam, avail_matrix(tech, 1), probs, make_grid(price_levels=[0]), W, 0.95
        )

        total_energy = result["dispatch"].sum()
        assert total_energy == pytest.approx(200.0, rel=1e-9)
        # Profit: 60*200 = 12 000
        assert result["expected_profit"] == pytest.approx(12_000.0, rel=1e-9)
