"""Unit tests for pydantic config validation."""

import pytest
from pydantic import ValidationError

from bidding.config import (
    AvailabilityConfig,
    BlockConfig,
    CandidateGrid,
    ExboGroupConfig,
    LsboFamilyConfig,
    RiskObjective,
    RunConfig,
    TechnologyConfig,
)


class TestRiskObjective:
    def test_beta_in_bounds(self):
        RiskObjective(beta=0.0)
        RiskObjective(beta=1.0)
        RiskObjective(beta=0.5)

    def test_beta_out_of_bounds_rejected(self):
        with pytest.raises(ValidationError):
            RiskObjective(beta=1.5)
        with pytest.raises(ValidationError):
            RiskObjective(beta=-0.1)


class TestAvailabilityConfig:
    def test_static_requires_values(self):
        with pytest.raises(ValidationError):
            AvailabilityConfig(source="static", values=None)

    def test_static_with_values_ok(self):
        cfg = AvailabilityConfig(source="static", values=[1.0, 2.0])
        assert cfg.values == [1.0, 2.0]

    def test_renewable_csv_requires_resource_and_capacity(self):
        with pytest.raises(ValidationError):
            AvailabilityConfig(source="renewable_csv", resource="solar")
        with pytest.raises(ValidationError):
            AvailabilityConfig(source="renewable_csv", nameplate_capacity_mw=100.0)

    def test_renewable_csv_with_fields_ok(self):
        cfg = AvailabilityConfig(source="renewable_csv", resource="solar", nameplate_capacity_mw=100.0)
        assert cfg.resource == "solar"


class TestCandidateGrid:
    def test_defaults_non_empty(self):
        grid = CandidateGrid()
        assert len(grid.price_levels_pct) > 0
        assert len(grid.mar_levels) > 0
        assert len(grid.mav_fraction_levels) > 0
        assert len(grid.tf_levels) > 0

    def test_absolute_price_levels_rejected(self):
        """Stale YAMLs still declaring absolute price_levels must fail loudly."""
        with pytest.raises(ValidationError):
            CandidateGrid(price_levels=[0, 10, 20])


class TestExboGroupConfig:
    def test_requires_at_least_two_blocks(self):
        block = BlockConfig(id="a", availability=AvailabilityConfig(values=[1.0]))
        with pytest.raises(ValidationError):
            ExboGroupConfig(group_id="g", blocks=[block])

    def test_two_blocks_ok(self):
        b1 = BlockConfig(id="a", availability=AvailabilityConfig(values=[1.0]))
        b2 = BlockConfig(id="b", availability=AvailabilityConfig(values=[2.0]))
        group = ExboGroupConfig(group_id="g", blocks=[b1, b2])
        assert len(group.blocks) == 2


class TestLsboFamilyConfig:
    def test_requires_at_least_one_child(self):
        parent = BlockConfig(id="p", availability=AvailabilityConfig(values=[1.0]))
        with pytest.raises(ValidationError):
            LsboFamilyConfig(family_id="f", parent=parent, children=[])

    def test_one_child_ok(self):
        parent = BlockConfig(id="p", availability=AvailabilityConfig(values=[1.0]))
        child = BlockConfig(id="c", availability=AvailabilityConfig(values=[0.5]))
        family = LsboFamilyConfig(family_id="f", parent=parent, children=[child])
        assert len(family.children) == 1


class TestTechnologyConfigCostArray:
    def test_scalar_cost_broadcasts(self):
        tech = TechnologyConfig(
            name="t", family=1, side="sell", variable_cost=10.0,
            availability=AvailabilityConfig(values=[1.0] * 24),
        )
        assert tech.cost_array(24) == [10.0] * 24

    def test_list_cost_wrong_length_raises(self):
        tech = TechnologyConfig(
            name="t", family=1, side="sell", variable_cost=[1.0, 2.0],
            price_reference=1.0,
            availability=AvailabilityConfig(values=[1.0] * 24),
        )
        with pytest.raises(ValueError):
            tech.cost_array(24)


class TestPriceReference:
    def test_zero_cost_without_reference_rejected(self):
        with pytest.raises(ValidationError):
            TechnologyConfig(
                name="t", family=1, side="sell", variable_cost=0.0,
                availability=AvailabilityConfig(values=[1.0] * 24),
            )

    def test_list_cost_without_reference_rejected(self):
        with pytest.raises(ValidationError):
            TechnologyConfig(
                name="t", family=1, side="sell", variable_cost=[1.0] * 24,
                availability=AvailabilityConfig(values=[1.0] * 24),
            )

    def test_negative_reference_rejected(self):
        with pytest.raises(ValidationError):
            TechnologyConfig(
                name="t", family=1, side="sell", variable_cost=0.0, price_reference=-5.0,
                availability=AvailabilityConfig(values=[1.0] * 24),
            )

    def test_reference_defaults_to_variable_cost(self):
        tech = TechnologyConfig(
            name="t", family=1, side="sell", variable_cost=85.0,
            availability=AvailabilityConfig(values=[1.0] * 24),
        )
        assert tech.price_reference_value == 85.0

    def test_explicit_reference_wins(self):
        tech = TechnologyConfig(
            name="t", family=1, side="sell", variable_cost=0.0, price_reference=10.0,
            availability=AvailabilityConfig(values=[1.0] * 24),
        )
        assert tech.price_reference_value == 10.0

    def test_monthly_source_forbids_reference(self):
        with pytest.raises(ValidationError):
            TechnologyConfig(
                name="t", family=2, side="sell",
                variable_cost_source="monthly_price_mean", price_reference=10.0,
                availability=AvailabilityConfig(values=[1.0] * 24),
            )

    def test_unresolved_monthly_cost_raises(self):
        tech = TechnologyConfig(
            name="t", family=2, side="sell",
            variable_cost_source="monthly_price_mean",
            availability=AvailabilityConfig(values=[1.0] * 24),
        )
        with pytest.raises(RuntimeError):
            tech.cost_array(24)
        with pytest.raises(RuntimeError):
            _ = tech.price_reference_value

    def test_tech_fraction_at_tech_level_rejected(self):
        with pytest.raises(ValidationError):
            TechnologyConfig(
                name="t", family=1, side="sell", variable_cost=10.0,
                availability=AvailabilityConfig(source="tech_fraction", values=[1.0] * 24),
            )


class TestTechFractionAvailability:
    def test_fractions_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            AvailabilityConfig(source="tech_fraction", values=[0.5, 1.5])
        with pytest.raises(ValidationError):
            AvailabilityConfig(source="tech_fraction", values=[-0.1, 0.5])

    def test_valid_fractions_ok(self):
        cfg = AvailabilityConfig(source="tech_fraction", values=[0.0, 0.5, 1.0])
        assert cfg.values == [0.0, 0.5, 1.0]


class TestResolvedGrid:
    def test_mar_level_below_technical_min_floor_raises(self):
        tech = TechnologyConfig(
            name="t", family=3, side="sell", technical_min=40.0, variable_cost=10.0,
            availability=AvailabilityConfig(values=[100.0] * 24),
            candidate_grid=CandidateGrid(mar_levels=[0.0, 0.5, 1.0], mav_fraction_levels=[0.5, 1.0]),
        )
        with pytest.raises(ValueError):
            tech.resolved_grid(CandidateGrid())

    def test_mav_level_below_technical_min_floor_raises(self):
        tech = TechnologyConfig(
            name="t", family=3, side="sell", technical_min=40.0, variable_cost=10.0,
            availability=AvailabilityConfig(values=[100.0] * 24),
            candidate_grid=CandidateGrid(mar_levels=[0.5, 1.0], mav_fraction_levels=[0.0, 1.0]),
        )
        with pytest.raises(ValueError):
            tech.resolved_grid(CandidateGrid())

    def test_levels_at_or_above_floor_ok(self):
        tech = TechnologyConfig(
            name="t", family=3, side="sell", technical_min=40.0, variable_cost=10.0,
            availability=AvailabilityConfig(values=[100.0] * 24),
            candidate_grid=CandidateGrid(mar_levels=[0.4, 0.5, 1.0], mav_fraction_levels=[0.4, 1.0]),
        )
        grid = tech.resolved_grid(CandidateGrid())
        assert grid.mar_levels == [0.4, 0.5, 1.0]

    def test_zero_technical_min_allows_zero_level(self):
        tech = TechnologyConfig(
            name="t", family=1, side="sell", technical_min=0.0, variable_cost=10.0,
            availability=AvailabilityConfig(values=[100.0] * 24),
            candidate_grid=CandidateGrid(mar_levels=[0.0, 1.0], mav_fraction_levels=[0.0, 1.0]),
        )
        grid = tech.resolved_grid(CandidateGrid())
        assert grid.mar_levels == [0.0, 1.0]

    def test_pct_levels_scale_with_reference(self):
        tech = TechnologyConfig(
            name="t", family=1, side="sell", variable_cost=50.0,
            availability=AvailabilityConfig(values=[100.0] * 24),
            candidate_grid=CandidateGrid(price_levels_pct=[0.0, 0.5, 1.0, 2.0]),
        )
        grid = tech.resolved_grid(CandidateGrid())
        assert grid.price_levels == [0.0, 25.0, 50.0, 100.0]


class TestRunConfig:
    def test_order_types_nonempty_required(self):
        with pytest.raises(ValidationError):
            RunConfig(order_types=[], prices={"csv_path": "x.csv"})

    def test_order_types_accepts_exbo_lsbo(self):
        cfg = RunConfig(order_types=["exbo", "lsbo"], prices={"csv_path": "x.csv"})
        assert cfg.order_types == ["exbo", "lsbo"]
