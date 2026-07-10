"""Compare a single technology's results between two run configs (e.g. verano/invierno)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .cli import _ensure_utf8_stdout, _tech_slug, evaluate_technology
from .config import RunConfig, TechnologyConfig
from .plots import plot_season_comparison


def _season_label(cfg: RunConfig, run_path: str) -> str:
    """cfg.season if set, otherwise fall back to the run YAML's filename stem."""
    return cfg.season or Path(run_path).stem


def evaluate_tech_seasons(tech_path: str, run_a_path: str, run_b_path: str) -> pd.DataFrame:
    """
    Evaluate one technology against two run configs and return a combined
    ranking DataFrame tagged with a `season` column (one row per order type
    per season).
    """
    tech = TechnologyConfig.from_yaml(tech_path)

    rows = []
    for run_path in (run_a_path, run_b_path):
        cfg = RunConfig.from_yaml(run_path)
        np.random.seed(cfg.seed)
        _, ranking, _ = evaluate_technology(tech, cfg)
        if ranking is None:
            continue
        tagged = ranking.copy()
        tagged.insert(0, "season", _season_label(cfg, run_path))
        rows.append(tagged)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_compare_seasons(tech_path: str, run_verano_path: str, run_invierno_path: str) -> None:
    _ensure_utf8_stdout()
    tech = TechnologyConfig.from_yaml(tech_path)

    sep = "=" * 62
    print(f"\n{sep}\n  Comparacion estacional - {tech.name}\n{sep}")

    comparison = evaluate_tech_seasons(tech_path, run_verano_path, run_invierno_path)
    if comparison.empty:
        print("[ERROR] No se obtuvieron resultados para ninguna temporada.", file=sys.stderr)
        sys.exit(1)

    display_cols = ["season", "order_type", "expected_profit", "cvar", "match_probability"]
    print(comparison[display_cols].to_string(index=False))

    output_dir = Path("results") / "comparacion_estacional" / _tech_slug(tech.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\n  Comparativa guardada en : {comparison_path}")

    try:
        plot_season_comparison(comparison, output_dir, tech_name=tech.name)
        print(f"  Grafica en             : {output_dir / 'figs' / 'season_comparison.png'}")
    except Exception as exc:
        print(f"  [WARN] Grafica no generada: {exc}", file=sys.stderr)

    print(f"\n{sep}\n")
