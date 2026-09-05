"""Tests for the monthly partition, dynamic water-value resolution and the
cross-month aggregate ranking."""

import numpy as np
import pandas as pd
import pytest

from bidding.cli import evaluate_technology, iter_month_runs
from bidding.config import AvailabilityConfig, RiskObjective, RunConfig, TechnologyConfig
from bidding.monthly import monthly_mean_price, resolve_tech_for_month, split_by_month
from bidding.orders.base import block_avail_matrix
from bidding.config import BlockConfig
from bidding.ranking import build_aggregate_ranking

from .conftest import make_tech


class TestSplitByMonth:
    def test_partitions_and_preserves_chronological_order(self):
        labels = ["2025-06-29", "2025-06-30", "2025-07-01", "2025-07-02", "2025-07-03"]
        months = split_by_month(labels)
        assert [m for m, _ in months] == ["2025-06", "2025-07"]
        assert months[0][1].tolist() == [0, 1]
        assert months[1][1].tolist() == [2, 3, 4]

    def test_single_month_is_one_partition(self):
        labels = [f"2025-01-{d:02d}" for d in range(1, 32)]
        months = split_by_month(labels)
        assert len(months) == 1
        assert months[0][0] == "2025-01"
        assert len(months[0][1]) == 31


class TestMonthlyMeanPrice:
    def test_mean_over_all_hours_of_the_month(self):
        lam = np.vstack([np.full(24, 10.0), np.full(24, 30.0), np.full(24, 100.0)])
        assert monthly_mean_price(lam, np.array([0, 1])) == pytest.approx(20.0)
        assert monthly_mean_price(lam, np.array([2])) == pytest.approx(100.0)


class TestResolveTechForMonth:
    def _hydro(self):
        return TechnologyConfig(
            name="hydro", family=2, side="sell",
            variable_cost_source="monthly_price_mean",
            availability=AvailabilityConfig(values=[200.0] * 24),
        )

    def test_static_tech_passes_through(self):
        tech = make_tech([100.0] * 24, cost=85.0)
        assert resolve_tech_for_month(tech, "2025-01", 60.0) is tech

    def test_dynamic_tech_resolves_to_month_mean(self):
        resolved = resolve_tech_for_month(self._hydro(), "2025-07", 70.47)
        assert resolved.variable_cost == pytest.approx(70.47)
        assert resolved.variable_cost_source == "static"
        assert resolved.price_reference_value == pytest.approx(70.47)
        assert resolved.cost_array(24) == [pytest.approx(70.47)] * 24

    def test_pct_grid_scales_with_the_month(self):
        from bidding.config import CandidateGrid

        hydro = self._hydro().model_copy(
            update={"candidate_grid": CandidateGrid(
                price_levels_pct=[0.0, 0.5, 1.0, 2.0],
                mar_levels=[1.0], mav_fraction_levels=[1.0], tf_levels=[0],
            )}
        )
        g_jan = resolve_tech_for_month(hydro, "2025-01", 100.30).resolved_grid(CandidateGrid())
        g_may = resolve_tech_for_month(hydro, "2025-05", 17.47).resolved_grid(CandidateGrid())
        assert g_jan.price_levels == pytest.approx([0.0, 50.15, 100.30, 200.60])
        assert g_may.price_levels == pytest.approx([0.0, 8.735, 17.47, 34.94])

    def test_non_positive_month_mean_raises(self):
        with pytest.raises(ValueError, match="<= 0"):
            resolve_tech_for_month(self._hydro(), "2025-04", -3.0)


def _write_price_csv(path, day_prices: dict[str, float]) -> None:
    rows = [
        {"date": day, "period": p, "price": price}
        for day, price in day_prices.items()
        for p in range(1, 25)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _two_month_cfg(tmp_path, order_types=("simple",), mode="stochastic"):
    csv_path = tmp_path / "prices.csv"
    _write_price_csv(csv_path, {
        "2025-01-01": 100.0,
        "2025-01-02": 100.0,
        "2025-02-01": 20.0,
        "2025-02-02": 20.0,
        "2025-02-03": 20.0,
    })
    return RunConfig(
        mode=mode,
        order_types=list(order_types),
        prices={"csv_path": str(csv_path)},
        candidate_grid={
            "price_levels_pct": [0.0, 0.5, 1.0],
            "mar_levels": [0.5],
            "mav_fraction_levels": [0.0],
            "tf_levels": [0],
        },
    )


class TestIterMonthRuns:
    def test_two_months_with_resolved_hydro_cost(self, tmp_path):
        cfg = _two_month_cfg(tmp_path)
        hydro = TechnologyConfig(
            name="hydro", family=2, side="sell",
            variable_cost_source="monthly_price_mean",
            availability=AvailabilityConfig(values=[200.0] * 24),
        )
        runs = list(iter_month_runs(hydro, cfg))
        assert [r["month"] for r in runs] == ["2025-01", "2025-02"]
        assert [r["n_days"] for r in runs] == [2, 3]
        assert runs[0]["variable_cost"] == pytest.approx(100.0)
        assert runs[1]["variable_cost"] == pytest.approx(20.0)
        # pct grid resolved against each month's water value
        assert runs[0]["grid"].price_levels == pytest.approx([0.0, 50.0, 100.0])
        assert runs[1]["grid"].price_levels == pytest.approx([0.0, 10.0, 20.0])
        # probabilities renormalized uniform within the month
        assert runs[1]["probs"].tolist() == pytest.approx([1 / 3] * 3)

    def test_deterministic_collapses_each_month_to_one_mean_day(self, tmp_path):
        cfg = _two_month_cfg(tmp_path, mode="deterministic")
        tech = make_tech([100.0] * 24, cost=10.0)
        runs = list(iter_month_runs(tech, cfg))
        assert all(r["lambda_matrix"].shape[0] == 1 for r in runs)
        assert [r["n_days"] for r in runs] == [2, 3]  # real day counts preserved


class TestAggregateRanking:
    def test_day_weighted_expected_profit(self, tmp_path):
        cfg = _two_month_cfg(tmp_path)
        tech = make_tech([100.0] * 24, cost=10.0)
        monthly, aggregate = evaluate_technology(tech, cfg)

        assert len(monthly) == 2
        # simple order, cost 10: Jan days (lambda 100) profit = (100-10)*100*24;
        # Feb days (lambda 20) profit = (20-10)*100*24
        jan = monthly[0]["ranking"].iloc[0]["expected_profit"]
        feb = monthly[1]["ranking"].iloc[0]["expected_profit"]
        agg = aggregate.iloc[0]["expected_profit"]
        assert agg == pytest.approx((2 * jan + 3 * feb) / 5)
        # optimal_params is a {month: params} map in the aggregate
        assert "2025-01" in aggregate.iloc[0]["optimal_params"]

    def test_weights_sum_to_one(self):
        monthly = [
            {"month": "2025-01", "n_days": 2, "results": [{
                "order_type": "simple",
                "profits": np.array([100.0, 200.0]),
                "matched": np.array([1.0, 1.0]),
                "dispatch": np.ones((2, 24)),
                "optimal_params": {"p": 1},
            }]},
            {"month": "2025-02", "n_days": 3, "results": [{
                "order_type": "simple",
                "profits": np.array([10.0, 20.0, 30.0]),
                "matched": np.array([1.0, 0.0, 1.0]),
                "dispatch": np.ones((3, 24)),
                "optimal_params": {"p": 2},
            }]},
        ]
        df = build_aggregate_ranking(monthly, 0.95, RiskObjective(beta=0.0), 100.0)
        expected = (100 + 200) / 5 + (10 + 20 + 30) / 5
        assert df.iloc[0]["expected_profit"] == pytest.approx(expected)


class TestBlockAvailMatrix:
    def test_static_tiles_profile(self):
        block = BlockConfig(
            id="b", availability=AvailabilityConfig(values=[50.0] * 24)
        )
        avail = np.random.default_rng(0).uniform(0, 100, size=(3, 24))
        out = block_avail_matrix(block, avail)
        assert out.shape == (3, 24)
        np.testing.assert_allclose(out, 50.0)

    def test_tech_fraction_tracks_daily_resource(self):
        fractions = [0.5] * 12 + [1.0] * 12
        block = BlockConfig(
            id="b",
            availability=AvailabilityConfig(source="tech_fraction", values=fractions),
        )
        avail = np.vstack([np.full(24, 80.0), np.full(24, 20.0)])  # good vs bad day
        out = block_avail_matrix(block, avail)
        np.testing.assert_allclose(out[0, :12], 40.0)
        np.testing.assert_allclose(out[0, 12:], 80.0)
        np.testing.assert_allclose(out[1, :12], 10.0)   # low-resource day scales down
        np.testing.assert_allclose(out[1, 12:], 20.0)
