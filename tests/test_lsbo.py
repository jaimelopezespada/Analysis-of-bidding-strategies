"""Unit tests for LSBO (linked block orders) parent/child acceptance logic."""

import numpy as np
import pytest

from bidding.config import AvailabilityConfig, BlockConfig, LsboFamilyConfig
from bidding.orders.lsbo import LSBOStrategy

from .conftest import W, make_grid, make_tech

STRAT = LSBOStrategy()


def _block(id_, price, avail):
    # make_tech sets price_reference=1.0, so these pct levels read as absolute €/MWh.
    return BlockConfig(id=id_, price_levels_pct=[price], availability=AvailabilityConfig(values=avail))


class TestParentChildLinking:
    def test_child_never_dispatched_without_parent(self):
        """If the family is rejected (parent not accepted), no child dispatches either."""
        parent = _block("p", 60.0, [100.0] * 24)   # welfare=(30-60)*100*24 <0
        child = _block("c", 10.0, [50.0] * 24)      # welfare=(30-10)*50*24 >0 on its own
        tech = make_tech([100.0] * 24)
        tech.lsbo_families = [LsboFamilyConfig(family_id="f", parent=parent, children=[child])]

        lam = np.full((1, 24), 30.0)
        probs = np.array([1.0])
        result = STRAT.evaluate(tech, lam, lam.copy(), probs, make_grid(), W, 0.95)

        # parent welfare = -72000, child welfare = +24000, family = -48000 < 0 -> reject all
        assert result["match_probability"] == pytest.approx(0.0)
        np.testing.assert_allclose(result["dispatch"], 0.0)

    def test_parent_deficit_compensated_by_children(self):
        """Parent has slightly negative welfare but children's surplus makes the family net positive."""
        parent = _block("p", 52.0, [100.0] * 24)   # welfare=(50-52)*100*24=-4800
        child = _block("c", 10.0, [50.0] * 24)      # welfare=(50-10)*50*24=48000
        tech = make_tech([100.0] * 24)
        tech.lsbo_families = [LsboFamilyConfig(family_id="f", parent=parent, children=[child])]

        lam = np.full((1, 24), 50.0)
        probs = np.array([1.0])
        result = STRAT.evaluate(tech, lam, lam.copy(), probs, make_grid(), W, 0.95)

        # family welfare = -4800 + 48000 = 43200 >= 0 -> family accepted
        assert result["match_probability"] == pytest.approx(1.0)
        np.testing.assert_allclose(result["dispatch"], 150.0)  # 100 (parent) + 50 (child)

    def test_child_never_accepted_with_negative_own_welfare(self):
        """Even if the family total is positive, a child with negative own welfare is excluded."""
        parent = _block("p", 10.0, [100.0] * 24)   # welfare=(50-10)*100*24=96000 (big surplus)
        bad_child = _block("c", 90.0, [50.0] * 24)  # welfare=(50-90)*50*24=-48000 (negative on its own)
        tech = make_tech([100.0] * 24)
        tech.lsbo_families = [LsboFamilyConfig(family_id="f", parent=parent, children=[bad_child])]

        lam = np.full((1, 24), 50.0)
        probs = np.array([1.0])
        result = STRAT.evaluate(tech, lam, lam.copy(), probs, make_grid(), W, 0.95)

        # Family welfare would be 96000-48000=48000>=0 if the child were included, but the
        # child's own welfare is negative so it must be excluded regardless.
        assert result["match_probability"] == pytest.approx(1.0)
        np.testing.assert_allclose(result["dispatch"], 100.0)  # parent only, not 150

    def test_missing_lsbo_families_raises(self):
        tech = make_tech([100.0] * 24)  # no lsbo_families set
        lam = np.full((1, 24), 50.0)
        probs = np.array([1.0])
        with pytest.raises(ValueError, match="lsbo_families"):
            STRAT.evaluate(tech, lam, lam.copy(), probs, make_grid(), W, 0.95)
