"""Unit tests for verano/invierno season-comparison utilities."""

from pathlib import Path

import pandas as pd
import pytest

from bidding.cli import tech_output_dir
from bidding.config import RunConfig
from reporting.family import run_family_seasons
from reporting.seasons import evaluate_tech_seasons

TECH_YAML = """
name: "Test tech"
family: 1
side: sell
variable_cost: 0.0
price_reference: 10.0
startup_cost: 0.0
availability:
  values: [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
           10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
candidate_grid:
  price_levels_pct: [0.0, 0.5]
  mar_levels: [0.5]
  mav_fraction_levels: [0.0]
  tf_levels: [0]
"""


def _write_price_csv(path: Path, dates: list[str], price: float) -> None:
    rows = []
    for d in dates:
        for t in range(1, 25):
            rows.append({"date": d, "period": t, "price": price})
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_run_yaml(path: Path, csv_path: Path, season: str) -> None:
    path.write_text(
        f"""
mode: stochastic
resolution: 24
order_types: [simple, sco, sbo]
prices:
  csv_path: {csv_path.as_posix()}
  scenario_mode: per_day
objective:
  beta: 0.0
cvar_alpha: 0.95
output_dir: results_test
season: {season}
seed: 42
"""
    )


class TestTechOutputDir:
    def test_namespaced_when_season_set(self):
        cfg = RunConfig(prices={"csv_path": "x.csv"}, season="verano")
        assert tech_output_dir(cfg, "CCGT 400 MW") == Path("results/verano/ccgt_400_mw")

    def test_not_namespaced_when_season_unset(self):
        cfg = RunConfig(prices={"csv_path": "x.csv"})
        assert tech_output_dir(cfg, "CCGT 400 MW") == Path("results/ccgt_400_mw")

    def test_verano_and_invierno_never_collide(self):
        cfg_a = RunConfig(prices={"csv_path": "x.csv"}, season="verano")
        cfg_b = RunConfig(prices={"csv_path": "x.csv"}, season="invierno")
        assert tech_output_dir(cfg_a, "Nuclear") != tech_output_dir(cfg_b, "Nuclear")


class TestEvaluateTechSeasons:
    def test_combined_dataframe_has_season_column_and_both_labels(self, tmp_path):
        tech_path = tmp_path / "tech.yaml"
        tech_path.write_text(TECH_YAML)

        csv_verano = tmp_path / "verano.csv"
        csv_invierno = tmp_path / "invierno.csv"
        _write_price_csv(csv_verano, ["2025-06-01", "2025-06-02"], price=50.0)
        _write_price_csv(csv_invierno, ["2025-12-01", "2025-12-02"], price=20.0)

        run_verano = tmp_path / "run_verano.yaml"
        run_invierno = tmp_path / "run_invierno.yaml"
        _write_run_yaml(run_verano, csv_verano, "verano")
        _write_run_yaml(run_invierno, csv_invierno, "invierno")

        df = evaluate_tech_seasons(str(tech_path), str(run_verano), str(run_invierno))

        assert set(df["season"]) == {"verano", "invierno"}
        assert set(df["order_type"]) == {"simple", "sco", "sbo"}
        # one row per (order_type, season) combination
        assert len(df) == 3 * 2

    def test_higher_price_season_has_higher_expected_profit(self, tmp_path):
        tech_path = tmp_path / "tech.yaml"
        tech_path.write_text(TECH_YAML)

        csv_verano = tmp_path / "verano.csv"
        csv_invierno = tmp_path / "invierno.csv"
        _write_price_csv(csv_verano, ["2025-06-01"], price=50.0)
        _write_price_csv(csv_invierno, ["2025-12-01"], price=5.0)

        run_verano = tmp_path / "run_verano.yaml"
        run_invierno = tmp_path / "run_invierno.yaml"
        _write_run_yaml(run_verano, csv_verano, "verano")
        _write_run_yaml(run_invierno, csv_invierno, "invierno")

        df = evaluate_tech_seasons(str(tech_path), str(run_verano), str(run_invierno))

        verano_simple = df[(df["season"] == "verano") & (df["order_type"] == "simple")]["expected_profit"].iloc[0]
        invierno_simple = df[(df["season"] == "invierno") & (df["order_type"] == "simple")]["expected_profit"].iloc[0]
        assert verano_simple > invierno_simple


class TestRunFamilySeasons:
    def test_writes_combined_comparison_csv(self, tmp_path, monkeypatch):
        yaml_dir = tmp_path / "yaml"
        yaml_dir.mkdir()
        (yaml_dir / "tech.yaml").write_text(TECH_YAML)

        csv_verano = tmp_path / "verano.csv"
        csv_invierno = tmp_path / "invierno.csv"
        _write_price_csv(csv_verano, ["2025-06-01"], price=50.0)
        _write_price_csv(csv_invierno, ["2025-12-01"], price=20.0)

        run_verano = tmp_path / "run_verano.yaml"
        run_invierno = tmp_path / "run_invierno.yaml"
        _write_run_yaml(run_verano, csv_verano, "verano")
        _write_run_yaml(run_invierno, csv_invierno, "invierno")

        monkeypatch.chdir(tmp_path)
        run_family_seasons("1", str(run_verano), str(run_invierno), yaml_dir=str(yaml_dir))

        comparison_path = tmp_path / "results" / "family_1_estacional" / "comparison.csv"
        assert comparison_path.exists()
        df = pd.read_csv(comparison_path)
        assert set(df["season"]) == {"verano", "invierno"}
        assert set(df["technology"]) == {"Test tech"}
