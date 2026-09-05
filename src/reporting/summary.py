"""Cross-technology summary: best objective value per month, one figure.

Reads the ranking_by_month.csv files already written by `bidding run` under
results/<season>/<tech_slug>/ — no re-optimization — and draws every
technology on one plot (color = technology, marker = winning order type,
y-axis in €/MW installed).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bidding.config import RunConfig
from bidding.plots import plot_best_expected_profit_by_month, plot_best_objective_by_month

_ACRONYMS = {"mw": "MW", "mwh": "MWh", "fv": "FV", "ccgt": "CCGT"}


def _tech_label(slug: str) -> str:
    """'eolica_50_mw' → 'Eolica 50 MW' (inverse of cli._tech_slug, best-effort)."""
    return " ".join(_ACRONYMS.get(w, w.capitalize()) for w in slug.split("_"))


def season_base_dir(cfg: RunConfig) -> Path:
    base = Path(cfg.output_dir)
    return base / cfg.season if cfg.season else base


def run_summary(run_path: str) -> None:
    """Build results/<season>/figs/best_objective_by_month_per_mw.{png,csv}."""
    cfg = RunConfig.from_yaml(run_path)
    base = season_base_dir(cfg)

    frames = []
    for csv_path in sorted(base.glob("*/ranking_by_month.csv")):
        df = pd.read_csv(csv_path)
        if "objective_value_per_mw" not in df.columns or df["objective_value_per_mw"].isna().all():
            print(f"[WARN] {csv_path} sin objective_value_per_mw (re-ejecutar "
                  f"'bidding run' para esa tecnología) — omitida.")
            continue
        cols = ["month", "order_type", "objective_value_per_mw"]
        if "expected_profit_per_mw" in df.columns:
            cols.append("expected_profit_per_mw")
        else:
            print(f"[WARN] {csv_path} sin expected_profit_per_mw (re-ejecutar "
                  f"'bidding run' para esa tecnología) — omitida en la figura de E[Π].")
        best = df[df["rank"] == 1][cols].copy()
        best.insert(0, "technology", _tech_label(csv_path.parent.name))
        frames.append(best)

    if not frames:
        print(f"[WARN] Ningún ranking_by_month.csv utilizable en {base}\\*\\ — "
              f"resumen no generado.")
        return

    best = pd.concat(frames, ignore_index=True)
    figs_dir = base / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    best.to_csv(figs_dir / "best_objective_by_month_per_mw.csv", index=False)
    plot_best_objective_by_month(
        best, figs_dir / "best_objective_by_month_per_mw.png",
        season_label=cfg.season or "",
    )
    print(f"  Resumen mejor oferta/mes  : {figs_dir / 'best_objective_by_month_per_mw.png'}")

    if "expected_profit_per_mw" in best.columns and best["expected_profit_per_mw"].notna().any():
        plot_best_expected_profit_by_month(
            best, figs_dir / "best_expected_profit_by_month_per_mw.png",
            season_label=cfg.season or "",
        )
        print(f"  Resumen E[Π] mejor oferta : {figs_dir / 'best_expected_profit_by_month_per_mw.png'}")
