"""Tests for the SCO out-of-sample (train/test) validation."""

import numpy as np
import pandas as pd
import pytest

from bidding.config import RunConfig
from reporting.validation import split_scenarios_chronological, validate_sco_oos

from .conftest import make_tech


class TestChronologicalSplit:
    def test_basic_70_30(self):
        train, test = split_scenarios_chronological(10, 0.7)
        assert train.tolist() == [0, 1, 2, 3, 4, 5, 6]
        assert test.tolist() == [7, 8, 9]

    def test_always_leaves_at_least_one_test_day(self):
        train, test = split_scenarios_chronological(2, 0.95)
        assert train.tolist() == [0]
        assert test.tolist() == [1]

    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError):
            split_scenarios_chronological(10, 1.0)


def _write_price_csv(path, day_prices: dict[str, float]) -> None:
    """One flat price per day, 24 hourly periods, OMIE-style columns."""
    rows = [
        {"date": day, "period": p, "price": price}
        for day, price in day_prices.items()
        for p in range(1, 25)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


class TestValidateScoOOS:
    def test_theta_from_train_applied_mechanically_on_test(self, tmp_path):
        """3 good train days (lambda=60) and 1 bad test day (lambda=5): the
        single-candidate theta (pv=10, tf=0) is accepted in-sample, while on
        the held-out day every period is OTM -> mechanical rejection, profit 0.
        """
        csv_path = tmp_path / "prices.csv"
        _write_price_csv(csv_path, {
            "2025-01-01": 60.0,
            "2025-01-02": 60.0,
            "2025-01-03": 60.0,
            "2025-01-04": 5.0,
        })
        tech = make_tech([100.0] * 24, cost=20.0, family=3)
        cfg = RunConfig(
            order_types=["sco"],
            prices={"csv_path": str(csv_path)},
            candidate_grid={
                # pct of the tech's price reference (= variable_cost 20) -> 10 €/MWh
                "price_levels_pct": [0.5],
                "mar_levels": [0.5],
                "mav_fraction_levels": [0.0],
                "tf_levels": [0.0],
            },
        )

        result = validate_sco_oos(tech, cfg, train_fraction=0.75)

        # theta* is now keyed by month (all 4 days are 2025-01)
        assert result["theta"] == {
            "2025-01": {"price_variable": 10.0, "fixed_term": 0.0, "mav_fraction": 0.0},
        }
        metrics = result["metrics"].set_index("split")
        assert (metrics["month"] == "2025-01").all()
        assert metrics.loc["train", "n_days"] == 3
        assert metrics.loc["test", "n_days"] == 1
        assert metrics.loc["train", "date_from"] == "2025-01-01"
        assert metrics.loc["test", "date_from"] == "2025-01-04"
        # train: (60-20)*100*24 per day, accepted every day
        assert metrics.loc["train", "expected_profit"] == pytest.approx(96_000.0)
        assert metrics.loc["train", "match_probability"] == pytest.approx(1.0)
        assert metrics.loc["train", "expected_matched_periods"] == pytest.approx(24.0)
        # test: lambda=5 < pv=10 in every period -> mechanically rejected
        assert metrics.loc["test", "expected_profit"] == pytest.approx(0.0)
        assert metrics.loc["test", "match_probability"] == pytest.approx(0.0)
        assert metrics.loc["test", "expected_matched_energy"] == pytest.approx(0.0)
        assert metrics.loc["test", "expected_matched_periods"] == pytest.approx(0.0)
