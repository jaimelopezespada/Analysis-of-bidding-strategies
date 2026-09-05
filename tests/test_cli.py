"""End-to-end regression test for the CLI pipeline against the example CSV."""

import sys
from pathlib import Path

import numpy as np

import bidding.cli as cli
from bidding.availability import static_availability_matrix
from bidding.config import ResolvedGrid, RunConfig, TechnologyConfig
from bidding.metrics import objective_value
from bidding.monthly import resolve_tech_for_month
from bidding.orders import STRATEGIES
from bidding.prices import load_price_matrix
from bidding.ranking import build_ranking

REPO_ROOT = Path(__file__).resolve().parents[1]


def _evaluate(tech_path: str, run_path: str, grid_override: ResolvedGrid | None = None):
    tech = TechnologyConfig.from_yaml(REPO_ROOT / tech_path)
    cfg = RunConfig.from_yaml(REPO_ROOT / run_path)
    lambda_matrix, _ = load_price_matrix(cfg.prices, cfg.resolution)
    S, _ = lambda_matrix.shape
    avail_matrix = static_availability_matrix(tech, S)
    probs = np.ones(S) / S
    if tech.variable_cost_source == "monthly_price_mean":
        # Single-period evaluation: resolve the dynamic water value against the
        # whole dataset's mean price (the CLI does this per month).
        tech = resolve_tech_for_month(tech, "all", float(lambda_matrix.mean()))

    results = []
    for order_type in cfg.order_types:
        strategy = STRATEGIES[order_type]()
        grid = grid_override if grid_override is not None else tech.resolved_grid(cfg.candidate_grid)
        result = strategy.evaluate(tech, lambda_matrix, avail_matrix, probs, grid, cfg.objective, cfg.cvar_alpha)
        result["objective_value"] = objective_value(result["profits"], probs, cfg.cvar_alpha, cfg.objective)
        results.append(result)
    return build_ranking(results)


class TestRegressionAgainstExampleCsv:
    def test_battery_ranking_is_deterministic_and_stable(self):
        """Same inputs must always produce the same ranking (no hidden randomness)."""
        ranking1 = _evaluate("yaml/bateria.yaml", "yaml/run.yaml")
        ranking2 = _evaluate("yaml/bateria.yaml", "yaml/run.yaml")

        assert ranking1["order_type"].tolist() == ranking2["order_type"].tolist()
        assert (ranking1["expected_profit"] == ranking2["expected_profit"]).all()

    def test_ranking_sorted_descending_by_expected_profit(self):
        ranking = _evaluate("yaml/bateria.yaml", "yaml/run.yaml")
        profits = ranking["expected_profit"].tolist()
        assert profits == sorted(profits, reverse=True)

    def test_nuclear_ranking_has_all_order_types(self):
        # Small grid override: nuclear's own YAML grid (168 SCO combos with
        # enforce_technical_constraints=True) is representative but slow —
        # this test only checks the pipeline wiring, not grid-search quality.
        small_grid = ResolvedGrid(
            price_levels=[10, 15], mar_levels=[0.5], mav_fraction_levels=[0.0], tf_levels=[0, 200_000],
        )
        ranking = _evaluate("yaml/nuclear.yaml", "yaml/run.yaml", grid_override=small_grid)
        assert set(ranking["order_type"]) == {"simple", "sco", "sbo", "exbo", "lsbo"}


class TestStartupPerTransitionConfig:
    def test_defaults_to_false(self):
        cfg = RunConfig.from_yaml(REPO_ROOT / "yaml" / "run.yaml")
        assert cfg.startup_per_transition is False

    def test_parsed_from_yaml(self):
        # run_verano.yaml opts in explicitly (startup_per_transition: true),
        # exercising that the value comes from the YAML, not the default.
        cfg = RunConfig.from_yaml(REPO_ROOT / "yaml" / "run_verano.yaml")
        assert cfg.startup_per_transition is True


class TestDiscoverTechYamls:
    def test_finds_only_technology_yamls(self):
        """Every YAML with a `family:` key qualifies; run configs are excluded."""
        paths = cli.discover_tech_yamls(str(REPO_ROOT / "yaml"))
        names = {p.name for p in paths}
        assert "ccgt.yaml" in names
        assert "solar_fv.yaml" in names
        assert not any(n.startswith("run") for n in names)


class TestRunDispatch:
    """Argparse wiring of `run`: run_command is stubbed to record its calls."""

    def _capture(self, monkeypatch, argv):
        calls = []
        monkeypatch.setattr(
            cli, "run_command",
            lambda tech, run, **kw: calls.append((tech, run, kw)),
        )
        monkeypatch.chdir(REPO_ROOT)
        monkeypatch.setattr(sys, "argv", ["bidding", *argv])
        cli.main()
        return calls

    def test_tech_all_two_seasons_runs_every_pair(self, monkeypatch):
        calls = self._capture(monkeypatch, [
            "run", "--tech", "all",
            "--run-verano", "yaml/run_verano.yaml",
            "--run-invierno", "yaml/run_invierno.yaml",
        ])
        n_techs = len(cli.discover_tech_yamls("yaml"))
        assert len(calls) == 2 * n_techs
        assert {run for _, run, _ in calls} == {"yaml/run_verano.yaml", "yaml/run_invierno.yaml"}
        # Batch mode must not abort the sweep on a single failure
        assert all(kw["exit_on_error"] is False for _, _, kw in calls)

    def test_single_tech_single_run_keeps_exit_on_error(self, monkeypatch):
        calls = self._capture(monkeypatch, [
            "run", "--tech", "yaml/ccgt.yaml", "--run", "yaml/run.yaml",
        ])
        assert len(calls) == 1
        assert calls[0][2]["exit_on_error"] is True
        assert calls[0][2]["startup_override"] is None

    def test_startup_flag_forwarded(self, monkeypatch):
        calls = self._capture(monkeypatch, [
            "run", "--tech", "yaml/ccgt.yaml", "--run", "yaml/run.yaml",
            "--startup-per-transition",
        ])
        assert calls[0][2]["startup_override"] is True

    def test_no_startup_flag_forwarded(self, monkeypatch):
        calls = self._capture(monkeypatch, [
            "run", "--tech", "yaml/ccgt.yaml", "--run", "yaml/run.yaml",
            "--no-startup-per-transition",
        ])
        assert calls[0][2]["startup_override"] is False
