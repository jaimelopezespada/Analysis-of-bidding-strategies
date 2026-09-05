"""Unit tests for price/availability CSV loading and validation."""

import numpy as np
import pandas as pd
import pytest

from bidding.availability import load_renewable_availability
from bidding.config import AvailabilityConfig, PricesConfig, TechnologyConfig
from bidding.prices import load_price_matrix


def _write_csv(path, rows, columns):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


class TestLoadPriceMatrix:
    def test_missing_column_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        _write_csv(path, [["2025-01-01", 1]], ["date", "period"])
        cfg = PricesConfig(csv_path=str(path))
        with pytest.raises(ValueError, match="missing required columns"):
            load_price_matrix(cfg, resolution=1)

    def test_wrong_period_count_raises(self, tmp_path):
        path = tmp_path / "bad_periods.csv"
        rows = [["2025-01-01", 1, 10.0], ["2025-01-01", 2, 11.0]]
        _write_csv(path, rows, ["date", "period", "price"])
        cfg = PricesConfig(csv_path=str(path))
        with pytest.raises(ValueError, match="wrong period count"):
            load_price_matrix(cfg, resolution=1)

    def test_valid_csv_pivots_correctly(self, tmp_path):
        path = tmp_path / "good.csv"
        rows = [
            ["2025-01-01", 1, 10.0], ["2025-01-01", 2, 20.0],
            ["2025-01-02", 1, 30.0], ["2025-01-02", 2, 40.0],
        ]
        _write_csv(path, rows, ["date", "period", "price"])
        cfg = PricesConfig(csv_path=str(path))
        matrix, labels = load_price_matrix(cfg, resolution=2)
        assert matrix.shape == (2, 2)
        assert labels == ["2025-01-01", "2025-01-02"]
        np.testing.assert_allclose(matrix, [[10.0, 20.0], [30.0, 40.0]])

    def test_exact_duplicate_rows_deduplicated(self, tmp_path):
        """Chunked-download artefact: an exact duplicate (date, period) row is dropped."""
        path = tmp_path / "dup.csv"
        rows = [
            ["2025-01-01", 1, 10.0], ["2025-01-01", 1, 10.0],  # exact duplicate
            ["2025-01-01", 2, 20.0],
        ]
        _write_csv(path, rows, ["date", "period", "price"])
        cfg = PricesConfig(csv_path=str(path))
        matrix, labels = load_price_matrix(cfg, resolution=2)
        assert matrix.shape == (1, 2)
        np.testing.assert_allclose(matrix, [[10.0, 20.0]])

    def test_conflicting_duplicate_rows_raise(self, tmp_path):
        path = tmp_path / "conflict.csv"
        rows = [
            ["2025-01-01", 1, 10.0], ["2025-01-01", 1, 99.0],  # conflicting values
            ["2025-01-01", 2, 20.0],
        ]
        _write_csv(path, rows, ["date", "period", "price"])
        cfg = PricesConfig(csv_path=str(path))
        with pytest.raises(ValueError, match="conflicting"):
            load_price_matrix(cfg, resolution=2)


class TestLoadRenewableAvailability:
    def _tech(self, nameplate=100.0):
        return TechnologyConfig(
            name="Solar test",
            family=1,
            side="sell",
            price_reference=10.0,  # required with the default variable_cost=0
            availability=AvailabilityConfig(
                source="renewable_csv", resource="solar", nameplate_capacity_mw=nameplate
            ),
        )

    def test_scaling_arithmetic(self, tmp_path):
        path = tmp_path / "gen.csv"
        rows = [
            ["2025-01-01", 1, 50.0], ["2025-01-01", 2, 100.0],
            ["2025-01-02", 1, 25.0], ["2025-01-02", 2, 200.0],  # max = 200
        ]
        _write_csv(path, rows, ["date", "period", "solar_mwh"])
        prices_cfg = PricesConfig(csv_path="unused.csv", renewable_csv_path=str(path))
        tech = self._tech(nameplate=100.0)
        result = load_renewable_availability(
            tech, prices_cfg, resolution=2, scenario_labels=["2025-01-01", "2025-01-02"]
        )
        expected = np.array([[50.0, 100.0], [25.0, 200.0]]) / 200.0 * 100.0
        np.testing.assert_allclose(result, expected)

    def test_missing_dates_raise(self, tmp_path):
        path = tmp_path / "gen.csv"
        rows = [["2025-01-01", 1, 50.0], ["2025-01-01", 2, 100.0]]
        _write_csv(path, rows, ["date", "period", "solar_mwh"])
        prices_cfg = PricesConfig(csv_path="unused.csv", renewable_csv_path=str(path))
        tech = self._tech()
        with pytest.raises(ValueError, match="do not match"):
            load_renewable_availability(
                tech, prices_cfg, resolution=2,
                scenario_labels=["2025-01-01", "2025-01-02"],
            )

    def test_missing_renewable_csv_path_raises(self):
        prices_cfg = PricesConfig(csv_path="unused.csv")
        tech = self._tech()
        with pytest.raises(ValueError, match="renewable_csv_path"):
            load_renewable_availability(tech, prices_cfg, resolution=2, scenario_labels=["2025-01-01"])
