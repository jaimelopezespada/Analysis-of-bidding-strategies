"""Annual ranking vs beta: how the order-type ranking moves with risk aversion.

For each technology and each beta in the sweep, every order type is
re-optimised per month under RiskObjective(beta) (theta* legitimately shifts
with beta) and the per-scenario profits are aggregated across months with
day-count weights — the exact same aggregation as ranking.aggregate_results —
so the annual ranking at beta=0.5 reproduces the main run's ranking.csv.

Outputs, per technology:
    results/<season>/<tech_slug>/ranking_vs_beta.csv
    results/<season>/<tech_slug>/figs/6_ranking_vs_beta.png
and cross-technology:
    results/<season>/figs/winner_by_beta.csv
    results/<season>/figs/winner_by_beta.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from bidding.cli import (
    _tech_slug,
    discover_tech_yamls,
    iter_month_runs,
    tech_output_dir,
)
from bidding.config import RiskObjective, RunConfig, TechnologyConfig
from bidding.orders import STRATEGIES
from bidding.plots import plot_ranking_vs_beta, plot_winner_by_beta
from bidding.ranking import build_aggregate_ranking

from .summary import _tech_label, season_base_dir

DEFAULT_BETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def sweep_tech(
    tech: TechnologyConfig, cfg: RunConfig, betas: list[float], verbose: bool = True
) -> pd.DataFrame:
    """Annual ranking of every order type at each beta, for one technology."""
    month_runs = list(iter_month_runs(tech, cfg))
    order_types = [ot for ot in cfg.order_types if ot in STRATEGIES]

    frames = []
    for beta in betas:
        objective = RiskObjective(beta=beta)
        monthly = []
        for mr in month_runs:
            results = []
            for order_type in order_types:
                strategy = STRATEGIES[order_type]()
                result = strategy.evaluate(
                    tech=mr["tech"],
                    lambda_matrix=mr["lambda_matrix"],
                    avail_matrix=mr["avail_matrix"],
                    probs=mr["probs"],
                    grid=mr["grid"],
                    objective=objective,
                    cvar_alpha=cfg.cvar_alpha,
                    startup_per_transition=cfg.startup_per_transition,
                    sco_model=cfg.sco_model,
                )
                results.append(result)
            monthly.append({**mr, "results": results})

        ranking = build_aggregate_ranking(
            monthly, cfg.cvar_alpha, objective, tech.installed_capacity_mw
        )
        ranking = ranking.reset_index()  # rank back as a column
        ranking.insert(0, "beta", beta)
        frames.append(ranking)
        if verbose:
            winner = ranking.loc[ranking["rank"] == 1, "order_type"].iloc[0]
            print(f"    beta = {beta:.1f}  ->  gana {winner.upper()}", flush=True)

    return pd.concat(frames, ignore_index=True)


def crossover_betas(df: pd.DataFrame) -> list[float]:
    """Betas at which the rank-1 order type differs from the previous beta's."""
    winners = df[df["rank"] == 1].sort_values("beta")
    crossings = []
    prev = None
    for _, row in winners.iterrows():
        if prev is not None and row["order_type"] != prev:
            crossings.append(float(row["beta"]))
        prev = row["order_type"]
    return crossings


def run_beta_ranking(
    run_path: str,
    tech: str = "all",
    yaml_dir: str = "yaml",
    betas: list[float] | None = None,
) -> None:
    cfg = RunConfig.from_yaml(run_path)
    betas = betas if betas else DEFAULT_BETAS

    if tech == "all":
        tech_paths = discover_tech_yamls(yaml_dir)
        if not tech_paths:
            print(f"[ERROR] No hay YAML de tecnología en {yaml_dir}/", file=sys.stderr)
            sys.exit(1)
    else:
        tech_paths = [Path(tech)]

    winner_frames = []
    for tech_path in tech_paths:
        tech_cfg = TechnologyConfig.from_yaml(tech_path)
        print(f"\n  Barrido de beta — {tech_cfg.name} ({len(betas)} valores)")
        df = sweep_tech(tech_cfg, cfg, betas)

        out_dir = tech_output_dir(cfg, tech_cfg.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "ranking_vs_beta.csv"
        df.to_csv(csv_path, index=False)

        try:
            plot_ranking_vs_beta(
                df, out_dir / "figs" / "6_ranking_vs_beta.png", tech_name=tech_cfg.name
            )
        except Exception as exc:
            print(f"  [WARN] Figura ranking-vs-beta no generada: {exc}", file=sys.stderr)

        crossings = crossover_betas(df)
        note = (
            f"cambia de ganadora en beta = {crossings}" if crossings
            else "misma ganadora en todo el barrido"
        )
        print(f"  Guardado {csv_path}  ({note})")

        winners = df[df["rank"] == 1][
            ["beta", "order_type", "objective_value_per_mw", "expected_profit_per_mw", "cvar"]
        ].copy()
        winners.insert(0, "technology", _tech_label(_tech_slug(tech_cfg.name)))
        winner_frames.append(winners)

    figs_dir = season_base_dir(cfg) / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    winners_all = pd.concat(winner_frames, ignore_index=True)
    winners_all.to_csv(figs_dir / "winner_by_beta.csv", index=False)
    try:
        plot_winner_by_beta(winners_all, figs_dir / "winner_by_beta.png",
                            season_label=cfg.season or "")
    except Exception as exc:
        print(f"  [WARN] Figura winner-by-beta no generada: {exc}", file=sys.stderr)
    print(f"\n  Resumen orden ganadora vs beta : {figs_dir / 'winner_by_beta.png'}")
