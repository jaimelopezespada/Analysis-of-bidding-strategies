"""Theta grid-convergence test: nested refinement of the candidate grid.

Re-evaluates ONE technology over the full run with progressively finer
candidate grids and checks that (1) the optimal theta* stabilises and (2) the
winning product (rank-1 order type of the annual aggregate ranking) does not
change with grid resolution — evidence that the exhaustive-search rankings
are not an artifact of the grid coarseness.

Refinement is by bisection of consecutive levels (refine_levels): with
factors that divide each other (1|2|4|8) every coarse grid is a SUBSET of the
finer one. Because each strategy is an exhaustive argmax over its grid, the
per-month objective is then monotone non-decreasing in the factor, so the
observed convergence is real and not resampling noise. The ANNUAL aggregate
objective (CVaR re-scored over concatenated per-day profits,
ranking.aggregate_results) is monotone in practice but not by construction —
this script reports deltas, it never assumes annual monotonicity.

Note on MAV: under sco_model="aware" (the default) mav_fraction never enters
the clearing (_clear_sco_aware ignores it), every MAV level ties and the
strict argmax keeps the first one — its stability is trivial, so it is
excluded from the theta-stability criterion and --no-refine-mav skips
refining it (x6 cheaper at factor 8).

Note on ties: for SBO/SCO the profit is a STEP function of the declared
price(s) — it only changes where the acceptance pattern changes — so the
objective has plateaus and theta* within a plateau is not unique. A finer
grid inserts equivalent candidates earlier in the enumeration and the strict
argmax picks them even though the coarse theta* is still optimal. Stability
is therefore judged by optimality, not identity: theta* is "stable" at a
transition iff no month's objective improved (the previous theta*, still
present in the nested grid, remains an argmax). The raw parameter drift is
reported as a diagnostic (theta_linf_delta).

Usage:
    python -m reporting.convergence --tech yaml/ccgt.yaml --run yaml/run_full_year.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bidding.cli import _ensure_utf8_stdout, evaluate_technology, tech_output_dir
from bidding.config import CandidateGrid, RunConfig, TechnologyConfig

DEFAULT_FACTORS = [1, 2, 4, 8]
DEFAULT_ORDERS = ["simple", "sbo", "sco"]

# Scalar theta components compared exactly between refinement levels.
# mav_fraction is deliberately excluded (inert under sco_model="aware").
_SCALAR_PARAMS = {"sbo": ["block_price", "mar"], "sco": ["price_variable", "fixed_term"]}

_COLORS = {"simple": "#2196F3", "sbo": "#FF9800", "sco": "#4CAF50"}


# ---------------------------------------------------------------------------
# Grid refinement (pure functions)
# ---------------------------------------------------------------------------

def refine_levels(levels: list[float], factor: int) -> list[float]:
    """Insert factor-1 equispaced points between each pair of consecutive levels.

    factor=1 returns sorted(set(levels)). Length is factor*(n-1)+1. Endpoints
    are preserved exactly, and grids are NESTED whenever the factors divide
    each other (values are rounded to 9 decimals so the same midpoint computed
    at different factors compares equal).
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    base = sorted({float(x) for x in levels})
    if factor == 1 or len(base) < 2:
        return base
    out: list[float] = []
    for lo, hi in zip(base[:-1], base[1:]):
        seg = np.linspace(lo, hi, factor + 1)[:-1]
        out.extend(round(float(v), 9) for v in seg)
    out.append(round(base[-1], 9))
    return out


def refine_grid(base: CandidateGrid, factor: int, refine_mav: bool = True) -> CandidateGrid:
    """Refined copy of a CandidateGrid (same ranges, denser levels)."""
    return CandidateGrid(
        price_levels_pct=refine_levels(base.price_levels_pct, factor),
        mar_levels=refine_levels(base.mar_levels, factor),
        mav_fraction_levels=(
            refine_levels(base.mav_fraction_levels, factor)
            if refine_mav
            else refine_levels(base.mav_fraction_levels, 1)
        ),
        tf_levels=refine_levels(base.tf_levels, factor),
    )


def n_combos(grid: CandidateGrid, order_type: str) -> int:
    """Theta candidates enumerated per month by one order type on this grid."""
    p = len(grid.price_levels_pct)
    if order_type == "simple":
        return p
    if order_type == "sbo":
        return p * len(grid.mar_levels)
    if order_type == "sco":
        return p * len(grid.tf_levels) * len(grid.mav_fraction_levels)
    raise ValueError(f"Unsupported order type for convergence test: {order_type!r}")


# ---------------------------------------------------------------------------
# Evaluation per refinement level
# ---------------------------------------------------------------------------

def run_level(
    tech: TechnologyConfig, cfg: RunConfig, grid: CandidateGrid
) -> tuple[pd.DataFrame, list[dict], float]:
    """Evaluate the tech on one refined grid; return (aggregate, monthly, elapsed_s).

    The tech-level candidate_grid must be overridden too because it takes
    priority over the run-level one in TechnologyConfig.resolved_grid().
    """
    tech_f = tech.model_copy(update={"candidate_grid": grid})
    cfg_f = cfg.model_copy(update={"candidate_grid": grid})
    t0 = time.perf_counter()
    monthly, aggregate = evaluate_technology(tech_f, cfg_f, verbose=False)
    elapsed = time.perf_counter() - t0
    if aggregate is None:
        raise RuntimeError("evaluate_technology produced no results — check the run config.")
    return aggregate.reset_index(), monthly, elapsed


def theta_records(monthly: list[dict], factor: int) -> list[dict]:
    """Long-format theta* rows: (factor, order_type, month, param, value).

    Lists (the simple order's price_profile) are serialised as JSON, the same
    convention as ranking.build_ranking.
    """
    rows = []
    for m in monthly:
        for r in m["results"]:
            for param, value in r["optimal_params"].items():
                if isinstance(value, list):
                    value = json.dumps(value)
                rows.append(
                    {
                        "factor": factor,
                        "order_type": r["order_type"],
                        "month": m["month"],
                        "param": param,
                        "value": value,
                    }
                )
    return rows


def monthly_records(monthly: list[dict], factor: int) -> list[dict]:
    """Per-month UNROUNDED objective rows: (factor, order_type, month, objective_value).

    This is where the nested-grid monotonicity guarantee lives (per-month
    argmax over a superset of candidates), and what the theta-stability
    criterion is computed from.
    """
    rows = []
    for m in monthly:
        for r in m["results"]:
            rows.append(
                {
                    "factor": factor,
                    "order_type": r["order_type"],
                    "month": m["month"],
                    "objective_value": float(r["objective_value"]),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Stability diagnostics between consecutive levels
# ---------------------------------------------------------------------------

def _scalar_theta(theta_df: pd.DataFrame, factor: int, order_type: str) -> pd.DataFrame:
    params = _SCALAR_PARAMS.get(order_type, [])
    sub = theta_df[
        (theta_df["factor"] == factor)
        & (theta_df["order_type"] == order_type)
        & (theta_df["param"].isin(params))
    ]
    return sub.pivot(index="month", columns="param", values="value").astype(float)


def _profile_matrix(theta_df: pd.DataFrame, factor: int) -> np.ndarray:
    """Concatenated annual price_profile of the simple order (sorted by month)."""
    sub = theta_df[
        (theta_df["factor"] == factor)
        & (theta_df["order_type"] == "simple")
        & (theta_df["param"] == "price_profile")
    ].sort_values("month")
    return np.concatenate([np.asarray(json.loads(v), dtype=float) for v in sub["value"]])


def stability_between(
    theta_df: pd.DataFrame, prev_factor: int, factor: int, order_type: str
) -> dict:
    """Compare theta* of one order type between two refinement levels.

    For sbo/sco: exact per-month equality of the scalar params (exact is the
    right test with nested grids — coarse points survive refinement and the
    strict argmax keeps them on ties). For simple: L-inf distance and fraction
    of hours whose optimal price changed, as diagnostics only.
    """
    if order_type == "simple":
        prev = _profile_matrix(theta_df, prev_factor)
        cur = _profile_matrix(theta_df, factor)
        delta = np.abs(cur - prev)
        return {
            "theta_stable": None,
            "profile_linf_delta": float(delta.max()),
            "profile_pct_hours_changed": float((delta > 1e-9).mean()),
        }
    prev = _scalar_theta(theta_df, prev_factor, order_type)
    cur = _scalar_theta(theta_df, factor, order_type)
    stable = prev.shape == cur.shape and bool(
        np.allclose(prev.sort_index().values, cur.sort_index().values, atol=1e-9)
    )
    return {
        "theta_stable": stable,
        "profile_linf_delta": None,
        "profile_pct_hours_changed": None,
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def summarize_convergence(summary_df: pd.DataFrame, tol: float) -> dict:
    """Convergence verdict over the per-level summary table.

    "Converged at factor F" = at the transition to F: same winner as the
    previous level, |relative delta| of the winner's objective < tol, and the
    winner's theta* is stable (criterion skipped for simple, whose theta is a
    price profile that converges trivially — see stability_between).
    """
    factors = sorted(summary_df["factor"].unique())
    winners = {
        int(f): summary_df.loc[
            (summary_df["factor"] == f) & (summary_df["rank"] == 1), "order_type"
        ].iloc[0]
        for f in factors
    }
    winner_stable_all = len(set(winners.values())) == 1

    converged_at = None
    for prev_f, f in zip(factors[:-1], factors[1:]):
        if winners[f] != winners[prev_f]:
            continue
        row = summary_df[
            (summary_df["factor"] == f) & (summary_df["order_type"] == winners[f])
        ].iloc[0]
        obj_ok = abs(row["delta_obj_rel"]) < tol
        theta_ok = row["theta_stable"] is None or bool(row["theta_stable"])
        if obj_ok and theta_ok:
            converged_at = int(f)
            break

    last = factors[-1]
    last_row = summary_df[
        (summary_df["factor"] == last) & (summary_df["order_type"] == winners[last])
    ].iloc[0]
    return {
        "winner_by_factor": winners,
        "winner_stable": winner_stable_all,
        "obj_converged": bool(abs(last_row["delta_obj_rel"]) < tol) if len(factors) > 1 else None,
        "theta_stable": (
            None if last_row["theta_stable"] is None else bool(last_row["theta_stable"])
        ),
        "converged_at_factor": converged_at,
        "tol": tol,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_convergence_objective(
    summary_df: pd.DataFrame, out: Path, tech_name: str, tol: float
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for order_type, sub in summary_df.groupby("order_type"):
        sub = sub.sort_values("factor")
        ax.plot(
            sub["n_combos"],
            sub["objective_value"],
            marker="o",
            color=_COLORS.get(order_type, "#607D8B"),
            label=order_type.upper(),
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"x{int(row['factor'])}",
                (row["n_combos"], row["objective_value"]),
                textcoords="offset points",
                xytext=(0, 6),
                fontsize=8,
                ha="center",
            )
    winner = summary_df[summary_df["rank"] == 1].sort_values("factor").iloc[-1]
    band = tol * abs(winner["objective_value"])
    ax.axhspan(
        winner["objective_value"] - band,
        winner["objective_value"] + band,
        color="grey",
        alpha=0.15,
        label=f"ganadora nivel mas fino ± tol ({tol:g})",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Candidatos de θ evaluados por mes (log)", fontsize=11)
    ax.set_ylabel("Valor objetivo anual (€/día)", fontsize=11)
    ax.set_title(f"Convergencia del objetivo con la resolución del grid — {tech_name}", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_convergence_theta(
    theta_df: pd.DataFrame, order_type: str, out: Path, tech_name: str
) -> None:
    params = _SCALAR_PARAMS[order_type]
    factors = sorted(theta_df["factor"].unique())
    fig, axes = plt.subplots(1, len(params), figsize=(6 * len(params), 4.5), squeeze=False)
    for ax, param in zip(axes[0], params):
        per_month = {}
        for f in factors:
            pivot = _scalar_theta(theta_df, f, order_type)
            if param in pivot.columns:
                per_month[f] = pivot[param]
        wide = pd.DataFrame(per_month)  # index: month, columns: factor
        for month, series in wide.iterrows():
            ax.plot(factors, series.values, color="grey", alpha=0.35, linewidth=0.8)
        ax.plot(
            factors,
            wide.mean(axis=0).values,
            color=_COLORS.get(order_type, "#607D8B"),
            linewidth=2.5,
            marker="o",
            label="media anual",
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(factors)
        ax.set_xticklabels([f"x{f}" for f in factors])
        ax.set_xlabel("Factor de refinamiento", fontsize=10)
        ax.set_title(param, fontsize=11)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("θ* mensual (líneas grises) y media anual", fontsize=10)
    axes[0][0].legend(fontsize=9)
    fig.suptitle(f"Estabilidad de θ* ({order_type.upper()}) — {tech_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_convergence(
    tech_path: str,
    run_path: str,
    orders: list[str] | None = None,
    factors: list[int] | None = None,
    tol: float = 1e-3,
    refine_mav: bool = True,
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    orders = list(orders) if orders else list(DEFAULT_ORDERS)
    factors = sorted(set(factors)) if factors else list(DEFAULT_FACTORS)
    for prev_f, f in zip(factors[:-1], factors[1:]):
        if f % prev_f != 0:
            print(
                f"[WARN] factor {f} no es múltiplo de {prev_f}: los grids no serán "
                f"anidados y el objetivo puede no ser monótono.",
                file=sys.stderr,
            )

    tech = TechnologyConfig.from_yaml(tech_path)
    cfg = RunConfig.from_yaml(run_path)
    cfg = cfg.model_copy(update={"order_types": orders, "beta_sweep": None})
    base_grid = tech.candidate_grid if tech.candidate_grid is not None else cfg.candidate_grid

    out_dir = (
        Path(out_dir) if out_dir else tech_output_dir(cfg, tech.name) / "convergence"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        sep = "=" * 62
        print(f"\n{sep}\n  Test de convergencia de θ — {tech.name}\n{sep}")
        print(f"  Órdenes: {', '.join(orders)}  |  factores: {factors}  |  tol: {tol:g}")

    summary_rows: list[dict] = []
    theta_rows: list[dict] = []
    for factor in factors:
        grid = refine_grid(base_grid, factor, refine_mav)
        aggregate, monthly, elapsed = run_level(tech, cfg, grid)
        theta_rows.extend(theta_records(monthly, factor))

        for _, row in aggregate.iterrows():
            summary_rows.append(
                {
                    "factor": factor,
                    "n_price": len(grid.price_levels_pct),
                    "n_mar": len(grid.mar_levels),
                    "n_mav": len(grid.mav_fraction_levels),
                    "n_tf": len(grid.tf_levels),
                    "order_type": row["order_type"],
                    "n_combos": n_combos(grid, row["order_type"]),
                    "rank": int(row["rank"]),
                    "objective_value": float(row["objective_value"]),
                    "objective_value_per_mw": float(row["objective_value_per_mw"]),
                    "expected_profit": float(row["expected_profit"]),
                    "cvar": float(row["cvar"]),
                    "elapsed_s": round(elapsed, 2),
                }
            )
        if verbose:
            winner = aggregate.loc[aggregate["rank"] == 1, "order_type"].iloc[0]
            combos_max = max(n_combos(grid, ot) for ot in orders)
            print(
                f"  x{factor:<3d} combos/mes máx = {combos_max:>7,d}   "
                f"t = {elapsed:6.1f} s   gana {winner.upper()}",
                flush=True,
            )

    summary_df = pd.DataFrame(summary_rows)
    theta_df = pd.DataFrame(theta_rows)

    # Deltas and theta stability vs the previous level, per order type.
    summary_df["delta_obj_rel"] = np.nan
    summary_df["theta_stable"] = None
    summary_df["profile_linf_delta"] = np.nan
    summary_df["profile_pct_hours_changed"] = np.nan
    for order_type in orders:
        for prev_f, f in zip(factors[:-1], factors[1:]):
            mask = (summary_df["factor"] == f) & (summary_df["order_type"] == order_type)
            prev_mask = (summary_df["factor"] == prev_f) & (
                summary_df["order_type"] == order_type
            )
            obj = summary_df.loc[mask, "objective_value"].iloc[0]
            obj_prev = summary_df.loc[prev_mask, "objective_value"].iloc[0]
            summary_df.loc[mask, "delta_obj_rel"] = (obj - obj_prev) / max(1.0, abs(obj_prev))
            stab = stability_between(theta_df, prev_f, f, order_type)
            summary_df.loc[mask, "theta_stable"] = stab["theta_stable"]
            summary_df.loc[mask, "profile_linf_delta"] = stab["profile_linf_delta"]
            summary_df.loc[mask, "profile_pct_hours_changed"] = stab[
                "profile_pct_hours_changed"
            ]

    verdict = summarize_convergence(summary_df, tol)

    summary_df.to_csv(out_dir / "convergence_summary.csv", index=False)
    theta_df.to_csv(out_dir / "convergence_theta.csv", index=False)
    with open(out_dir / "convergence_verdict.json", "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, ensure_ascii=False, indent=2)

    figs_dir = out_dir / "figs"
    figs_dir.mkdir(exist_ok=True)
    try:
        plot_convergence_objective(
            summary_df, figs_dir / "convergence_objective.png", tech.name, tol
        )
        for order_type in orders:
            if order_type in _SCALAR_PARAMS:
                plot_convergence_theta(
                    theta_df,
                    order_type,
                    figs_dir / f"convergence_theta_{order_type}.png",
                    tech.name,
                )
    except Exception as exc:  # pragma: no cover - plotting must never kill the run
        print(f"  [WARN] Figuras de convergencia no generadas: {exc}", file=sys.stderr)

    if verbose:
        print(f"\n  Ganadora por nivel : {verdict['winner_by_factor']}")
        if verdict["converged_at_factor"] is not None:
            print(
                f"  VEREDICTO: CONVERGE en factor x{verdict['converged_at_factor']} "
                f"(|Δobj| rel < {tol:g}, ganadora y θ* estables)."
            )
        else:
            print(
                f"  VEREDICTO: NO converge dentro de los factores probados "
                f"(tol {tol:g}) — considerar factores mayores."
            )
        print(f"  Resultados en: {out_dir}")

    return verdict


def main() -> None:
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(
        prog="python -m reporting.convergence",
        description=(
            "Test de convergencia de θ: re-evalúa una tecnología con grids anidados "
            "cada vez más finos y comprueba que θ* y el producto ganador son estables."
        ),
    )
    parser.add_argument("--tech", default="yaml/ccgt.yaml", help="YAML de la tecnología")
    parser.add_argument("--run", default="yaml/run_full_year.yaml", help="YAML del run")
    parser.add_argument(
        "--orders",
        default=",".join(DEFAULT_ORDERS),
        help="Tipos de orden separados por comas (exbo/lsbo desaconsejadas: coste combinatorio)",
    )
    parser.add_argument(
        "--factors",
        default=",".join(str(f) for f in DEFAULT_FACTORS),
        help="Factores de refinamiento separados por comas; deben dividirse entre sí",
    )
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerancia relativa de Δobjetivo")
    parser.add_argument(
        "--no-refine-mav",
        action="store_true",
        help="No refinar mav_fraction_levels (inerte bajo sco_model=aware; ahorra coste)",
    )
    parser.add_argument("--out", default=None, help="Directorio de salida (defecto: junto al ranking)")
    args = parser.parse_args()

    run_convergence(
        tech_path=args.tech,
        run_path=args.run,
        orders=[o.strip() for o in args.orders.split(",") if o.strip()],
        factors=[int(f) for f in args.factors.split(",") if f.strip()],
        tol=args.tol,
        refine_mav=not args.no_refine_mav,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
