"""Unit tests for EXBO (exclusive group of blocks) acceptance logic."""

import numpy as np
import pytest

from bidding.config import AvailabilityConfig, BlockConfig, ExboGroupConfig
from bidding.orders.exbo import EXBOStrategy

from .conftest import W, make_grid, make_tech

STRAT = EXBOStrategy()


def _block(id_, price, avail):
    # make_tech sets price_reference=1.0, so these pct levels read as absolute €/MWh.
    return BlockConfig(id=id_, price_levels_pct=[price], availability=AvailabilityConfig(values=avail))


class TestMutualExclusivity:
    def test_only_best_welfare_block_selected(self):
        """Two blocks both qualify (positive welfare); the higher-welfare one wins.

        Distinguishable availability profiles (100 vs 60 MWh) so the resulting
        dispatch/profit reveals which block was actually selected.
        """
        low_price_block = _block("low", 10.0, [100.0] * 24)    # welfare=(50-10)*100*24=96000
        high_price_block = _block("high", 40.0, [60.0] * 24)   # welfare=(50-40)*60*24=14400
        tech = make_tech([100.0] * 24)
        tech.exbo_groups = [ExboGroupConfig(group_id="g", blocks=[low_price_block, high_price_block])]

        lam = np.full((1, 24), 50.0)
        probs = np.array([1.0])
        result = STRAT.evaluate(tech, lam, lam.copy(), probs, make_grid(), W, 0.95)

        assert result["optimal_params"]["block_prices"] == {"low": 10.0, "high": 40.0}
        # Higher-welfare ("low") block's profile (100 MWh) should be dispatched, not
        # the 60 MWh one — profit = (50-0)*100*24 = 120000, not (50-0)*60*24=72000.
        np.testing.assert_allclose(result["dispatch"], 100.0)
        assert result["expected_profit"] == pytest.approx(120_000.0, rel=1e-9)

    def test_all_negative_welfare_nothing_matches(self):
        b1 = _block("a", 60.0, [100.0] * 24)
        b2 = _block("b", 80.0, [100.0] * 24)
        tech = make_tech([100.0] * 24)
        tech.exbo_groups = [ExboGroupConfig(group_id="g", blocks=[b1, b2])]

        lam = np.full((1, 24), 50.0)  # below both block prices -> negative welfare
        probs = np.array([1.0])
        result = STRAT.evaluate(tech, lam, lam.copy(), probs, make_grid(), W, 0.95)

        assert result["match_probability"] == pytest.approx(0.0)
        np.testing.assert_allclose(result["dispatch"], 0.0)

    def test_never_dispatches_more_than_one_block(self):
        """Sum of dispatched blocks per scenario never exceeds one block's profile."""
        b1 = _block("a", 10.0, [100.0] * 24)
        b2 = _block("b", 15.0, [80.0] * 24)
        tech = make_tech([100.0] * 24)
        tech.exbo_groups = [ExboGroupConfig(group_id="g", blocks=[b1, b2])]

        lam = np.full((3, 24), 50.0)
        probs = np.ones(3) / 3
        result = STRAT.evaluate(tech, lam, lam.copy(), probs, make_grid(), W, 0.95)

        # Every scenario's dispatch must match exactly one of the two declared profiles
        for s in range(3):
            total = result["dispatch"][s].sum()
            assert total == pytest.approx(0.0) or total == pytest.approx(2400.0) or total == pytest.approx(1920.0)

    def test_missing_exbo_groups_raises(self):
        tech = make_tech([100.0] * 24)  # no exbo_groups set
        lam = np.full((1, 24), 50.0)
        probs = np.array([1.0])
        with pytest.raises(ValueError, match="exbo_groups"):
            STRAT.evaluate(tech, lam, lam.copy(), probs, make_grid(), W, 0.95)
