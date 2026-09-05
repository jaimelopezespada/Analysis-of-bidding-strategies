"""Unit tests for technical_min enforcement in the SCO MILP.

ramp_limit is deliberately NOT enforced: the SDAC "nueva tipologia" (SBO, SCO,
EXBO, LSBO) has no bid parameter to express a ramp/gradient condition — that
was a feature of the discontinued pre-2025 OMIE *ofertas complejas clasicas*
(see optimizer.py's module docstring). ramp_limit remains a TechnologyConfig
field only for the technology's qualitative characterisation.
"""

import numpy as np
import pytest

from bidding.config import ResolvedGrid, RiskObjective
from bidding.orders.sco import SCOStrategy

from .conftest import avail_matrix, make_tech

STRAT = SCOStrategy()
W = RiskObjective()


def _grid(pv, tf, mav_frac=0.0):
    return ResolvedGrid(price_levels=[pv], mar_levels=[0.5], mav_fraction_levels=[mav_frac], tf_levels=[tf])


class TestTechnicalMinimum:
    def test_dispatch_never_between_zero_and_qmin(self):
        """When active, q_t^s must be 0 or >= technical_min — never strictly between."""
        tech = make_tech(
            [100.0] * 24, cost=50.0, family=3,
            technical_min=40.0, enforce_technical_constraints=True,
        )
        # Varying prices so some periods are in-the-money, some not.
        prices = [60.0] * 12 + [10.0] * 12
        lam = np.array([prices])
        probs = np.array([1.0])
        result = STRAT.evaluate(tech, lam, avail_matrix(tech, 1), probs, _grid(pv=20, tf=0), W, 0.95)

        q = result["dispatch"][0]
        for value in q:
            assert value == pytest.approx(0.0, abs=1e-6) or value >= 40.0 - 1e-6


class TestRampNotEnforced:
    def test_ramp_limit_has_no_effect_on_dispatch(self):
        """ramp_limit must NOT constrain dispatch, even with sharp price swings
        that would violate it if it were (wrongly) enforced."""
        tech_with_ramp = make_tech(
            [100.0] * 24, cost=10.0, family=3,
            technical_min=0.0, ramp_limit=20.0, enforce_technical_constraints=True,
        )
        tech_without_ramp = make_tech(
            [100.0] * 24, cost=10.0, family=3,
            technical_min=0.0, ramp_limit=None, enforce_technical_constraints=True,
        )
        prices = [50.0] * 6 + [5.0] * 6 + [50.0] * 6 + [5.0] * 6  # sharp swings
        lam = np.array([prices])
        probs = np.array([1.0])
        # sco_model="naive": the per-period MILP drops the lambda=5 hours
        # (below cost=10), so dispatch actually swings 0 <-> 100 and ramps
        # are observable. The revised day-level "aware" model dispatches the
        # full daily profile once accepted, so it can never produce intra-day
        # swings to test ramp against.
        grid = _grid(pv=20, tf=0)

        result_with = STRAT.evaluate(tech_with_ramp, lam, avail_matrix(tech_with_ramp, 1), probs, grid, W, 0.95, sco_model="naive")
        result_without = STRAT.evaluate(tech_without_ramp, lam, avail_matrix(tech_without_ramp, 1), probs, grid, W, 0.95, sco_model="naive")

        np.testing.assert_allclose(result_with["dispatch"], result_without["dispatch"])
        # Sharp swings (0 <-> 100 MWh between adjacent periods) exceed the
        # declared ramp_limit=20, confirming it was not applied as a constraint.
        diffs = np.abs(np.diff(result_with["dispatch"][0]))
        assert np.any(diffs > 20.0)


class TestNoRegressionWhenDisabled:
    def test_disabled_matches_hand_derived_profit(self):
        """enforce_technical_constraints=False (default) must reproduce the exact
        pre-MILP closed-form numbers already hand-derived in test_sco.py."""
        tech = make_tech([100.0] * 24, family=3)  # enforce_technical_constraints=False
        lam = np.full((2, 24), 100.0)
        probs = np.ones(2) / 2
        grid = _grid(pv=10, tf=1_000, mav_frac=0.5)
        result = STRAT.evaluate(tech, lam, avail_matrix(tech, 2), probs, grid, W, 0.95)

        assert result["match_probability"] == pytest.approx(1.0)
        np.testing.assert_allclose(result["dispatch"], 100.0)
