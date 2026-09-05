"""Smoke/regression tests against the real winter/summer OMIE + ESIOS datasets.

Marked slow: exercises the full CLI-style pipeline (real CSVs, MILP for SCO)
end to end for one representative technology per data-availability path.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bidding.availability import load_renewable_availability, static_availability_matrix
from bidding.config import RunConfig, TechnologyConfig
from bidding.metrics import objective_value
from bidding.orders import STRATEGIES
from bidding.prices import load_price_matrix
from bidding.ranking import build_ranking

REPO_ROOT = Path(__file__).resolve().parents[1]
EDA_STATS = REPO_ROOT / "results" / "eda" / "estadisticos.csv"

pytestmark = pytest.mark.slow


def _evaluate(tech_path: str, run_path: str):
    tech = TechnologyConfig.from_yaml(REPO_ROOT / tech_path)
    cfg = RunConfig.from_yaml(REPO_ROOT / run_path)
    lambda_matrix, labels = load_price_matrix(cfg.prices, cfg.resolution)
    S, _ = lambda_matrix.shape

    if tech.availability.source == "static":
        avail_matrix = static_availability_matrix(tech, S)
    else:
        avail_matrix = load_renewable_availability(tech, cfg.prices, cfg.resolution, labels)

    probs = np.ones(S) / S
    results = []
    for order_type in cfg.order_types:
        strategy = STRATEGIES[order_type]()
        grid = tech.resolved_grid(cfg.candidate_grid)
        result = strategy.evaluate(tech, lambda_matrix, avail_matrix, probs, grid, cfg.objective, cfg.cvar_alpha)
        result["objective_value"] = objective_value(result["profits"], probs, cfg.cvar_alpha, cfg.objective)
        results.append(result)
    return lambda_matrix, build_ranking(results)


class TestRealDataEndToEnd:
    def test_wind_summer_no_errors_and_reasonable_ranking(self):
        lambda_matrix, ranking = _evaluate("yaml/eolica.yaml", "yaml/run_verano.yaml")
        assert not ranking.isnull().values.any()
        assert (ranking["expected_profit"] > 0).all()

    def test_wind_winter_no_errors_and_reasonable_ranking(self):
        lambda_matrix, ranking = _evaluate("yaml/eolica.yaml", "yaml/run_invierno.yaml")
        assert not ranking.isnull().values.any()
        assert (ranking["expected_profit"] > 0).all()


class TestPriceStatsConsistencyWithEda:
    @pytest.mark.skipif(not EDA_STATS.exists(), reason="EDA stats not generated yet")
    def test_summer_mean_std_match_eda(self):
        stats = pd.read_csv(EDA_STATS, index_col=0)
        lambda_matrix, _ = load_price_matrix(
            RunConfig.from_yaml(REPO_ROOT / "yaml" / "run_verano.yaml").prices, 24
        )
        assert lambda_matrix.mean() == pytest.approx(stats.loc["mean", "Verano 2025"], rel=0.02)
        assert lambda_matrix.std() == pytest.approx(stats.loc["std", "Verano 2025"], rel=0.05)

    @pytest.mark.skipif(not EDA_STATS.exists(), reason="EDA stats not generated yet")
    def test_winter_mean_std_match_eda(self):
        stats = pd.read_csv(EDA_STATS, index_col=0)
        lambda_matrix, _ = load_price_matrix(
            RunConfig.from_yaml(REPO_ROOT / "yaml" / "run_invierno.yaml").prices, 24
        )
        assert lambda_matrix.mean() == pytest.approx(stats.loc["mean", "Invierno 2025-2026"], rel=0.02)
        assert lambda_matrix.std() == pytest.approx(stats.loc["std", "Invierno 2025-2026"], rel=0.05)
