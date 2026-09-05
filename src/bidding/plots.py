"""Comparative plots for the bidding strategy optimizer."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
_FIGSIZE_WIDE = (12, 5)
_FIGSIZE_STD = (9, 5)
_DPI = 150

# Cross-technology summary: color = technology (fixed alphabetical assignment,
# CVD-safe ordering), marker = order type.
_TECH_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
                "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
_ORDER_MARKERS = {"simple": "o", "sco": "s", "sbo": "^", "exbo": "D", "lsbo": "v"}


def plot_all(
    results: list[dict],
    lambda_matrix: np.ndarray,
    output_dir: Path,
    tech_name: str = "",
    capacity_mw: float | None = None,
) -> None:
    figs_dir = output_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    plot_expected_profit_bars(results, figs_dir, tech_name)
    plot_profit_distribution(results, figs_dir, tech_name)
    plot_offer_vs_price(results, lambda_matrix, figs_dir, tech_name, capacity_mw)
    plot_dispatch_profiles(results, figs_dir, tech_name)

    plt.close("all")


# ---------------------------------------------------------------------------
# 1. Bar chart: E[Π] with CVaR overlay
# ---------------------------------------------------------------------------

def plot_expected_profit_bars(
    results: list[dict], out: Path, tech_name: str
) -> None:
    n = len(results)
    labels = [r["order_type"].upper() for r in results]
    ep = [r["expected_profit"] for r in results]
    cv = [r["cvar"] for r in results]
    colors = _COLORS[:n]

    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)
    x = np.arange(n)
    ax.bar(x, ep, color=colors, alpha=0.85, label="E[Π] (beneficio esperado)")
    ax.bar(x, cv, color="crimson", alpha=0.35, label="CVaR₀.₉₅")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("€", fontsize=11)
    ax.set_title(
        f"Beneficio esperado por tipo de orden"
        + (f" — {tech_name}" if tech_name else ""),
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    fig.savefig(out / "1_expected_profit_bars.png", dpi=_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Boxplot: profit distribution across scenarios
# ---------------------------------------------------------------------------

def plot_profit_distribution(
    results: list[dict], out: Path, tech_name: str
) -> None:
    n = len(results)
    data = [r["profits"] for r in results]
    labels = [r["order_type"].upper() for r in results]

    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], _COLORS[:n]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Beneficio por escenario (€)", fontsize=11)
    ax.set_title(
        f"Distribución del beneficio entre escenarios"
        + (f" — {tech_name}" if tech_name else ""),
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    fig.savefig(out / "2_profit_distribution.png", dpi=_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Offer price profile vs mean market price
# ---------------------------------------------------------------------------

def plot_offer_vs_price(
    results: list[dict],
    lambda_matrix: np.ndarray,
    out: Path,
    tech_name: str,
    capacity_mw: float | None = None,
) -> None:
    T = lambda_matrix.shape[1]
    mean_price = lambda_matrix.mean(axis=0)
    periods = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.plot(periods, mean_price, "k--", linewidth=2, label="Precio medio mercado")

    for r, color in zip(results, _COLORS):
        params = r["optimal_params"]
        otype = r["order_type"]

        if otype == "simple":
            offer = params["price_profile"]
            ax.step(periods, offer, where="post", color=color, linewidth=1.8,
                    label=f"SIMPLE P^V_t (por periodo)")
        elif otype == "sco":
            pv = params["price_variable"]
            if capacity_mw is not None:
                mav_label = f"MAV = {params['mav_fraction'] * capacity_mw:.1f} MW"
            else:
                mav_label = f"MAV = {params['mav_fraction']:.1%}"
            ax.axhline(pv, color=color, linewidth=1.8,
                       label=f"SCO  P^V = {pv:.1f} €/MWh  |  TF = {params['fixed_term']:.0f} €  |  {mav_label}")
        elif otype == "sbo":
            pb = params["block_price"]
            ax.axhline(pb, color=color, linewidth=1.8, linestyle="-.",
                       label=f"SBO  P^B = {pb:.1f} €/MWh  |  MAR = {params['mar']:.1%}")

    ax.set_xlabel("Periodo (MTU)", fontsize=11)
    ax.set_ylabel("€/MWh", fontsize=11)
    ax.set_title(
        f"Precio de oferta óptimo vs. precio medio de mercado"
        + (f" — {tech_name}" if tech_name else ""),
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "3_offer_vs_price.png", dpi=_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3b. Offer price profile, with block-restricted segments for EXBO/LSBO
# ---------------------------------------------------------------------------

_BLOCK_LINESTYLES = [":", "--", "-."]


def _block_active_mask(block) -> np.ndarray:
    """Periods where a BlockConfig's own declared availability is non-zero."""
    values = np.asarray(block.availability.values, dtype=float)
    return values > 0.0


def plot_block_offers(
    results: list[dict],
    lambda_matrix: np.ndarray,
    tech,
    out: Path,
    tech_name: str = "",
    capacity_mw: float | None = None,
) -> None:
    """Like plot_offer_vs_price, but also renders exbo/lsbo block prices.

    plot_offer_vs_price silently skips exbo/lsbo (it only has branches for
    simple/sco/sbo). A block's declared price cannot be drawn as a full-width
    axhline, though, since EXBO/LSBO blocks routinely cover only a subset of
    the day's periods — this draws each block's price restricted to its own
    active hours instead. The block hour-windows aren't in optimal_params
    (only prices/ids are), so this needs ``tech`` to look them up via
    tech.exbo_groups / tech.lsbo_families.
    """
    T = lambda_matrix.shape[1]
    mean_price = lambda_matrix.mean(axis=0)
    periods = np.arange(1, T + 1)

    exbo_groups = {g.group_id: g for g in (tech.exbo_groups or [])}
    lsbo_families = {f.family_id: f for f in (tech.lsbo_families or [])}

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.plot(periods, mean_price, "k--", linewidth=2, label="Precio medio mercado")

    for r, color in zip(results, _COLORS):
        params = r["optimal_params"]
        otype = r["order_type"]

        if otype == "simple":
            ax.step(periods, params["price_profile"], where="post", color=color,
                     linewidth=1.8, label="SIMPLE P^V_t (por periodo)")
        elif otype == "sco":
            pv = params["price_variable"]
            mav_label = (f"MAV = {params['mav_fraction'] * capacity_mw:.1f} MW"
                         if capacity_mw is not None else f"MAV = {params['mav_fraction']:.1%}")
            ax.axhline(pv, color=color, linewidth=1.8,
                       label=f"SCO  P^V = {pv:.1f} €/MWh  |  TF = {params['fixed_term']:.0f} €  |  {mav_label}")
        elif otype == "sbo":
            pb = params["block_price"]
            ax.axhline(pb, color=color, linewidth=1.8, linestyle="-.",
                       label=f"SBO  P^B = {pb:.1f} €/MWh  |  MAR = {params['mar']:.1%}")
        elif otype == "exbo":
            group = exbo_groups.get(params["group_id"])
            if group is None:
                continue
            blocks_by_id = {b.id: b for b in group.blocks}
            for i, (block_id, price) in enumerate(params["block_prices"].items()):
                block = blocks_by_id.get(block_id)
                if block is None or not _block_active_mask(block).any():
                    continue
                seg = np.where(_block_active_mask(block), price, np.nan)
                ax.plot(periods, seg, color=color, linewidth=2.4,
                        linestyle=_BLOCK_LINESTYLES[i % len(_BLOCK_LINESTYLES)],
                        marker="_", markersize=10,
                        label=f"EXBO {block_id}  P={price:.1f} €/MWh")
        elif otype == "lsbo":
            family = lsbo_families.get(params["family_id"])
            if family is None:
                continue
            segs = [(family.parent.id, params["parent_price"], family.parent)]
            segs += [
                (c.id, params["child_prices"][c.id], c)
                for c in family.children if c.id in params["child_prices"]
            ]
            for i, (block_id, price, block) in enumerate(segs):
                if not _block_active_mask(block).any():
                    continue
                seg = np.where(_block_active_mask(block), price, np.nan)
                ax.plot(periods, seg, color=color, linewidth=2.4,
                        linestyle=_BLOCK_LINESTYLES[i % len(_BLOCK_LINESTYLES)],
                        marker="_", markersize=10,
                        label=f"LSBO {block_id}  P={price:.1f} €/MWh")

    ax.set_xlabel("Periodo (MTU)", fontsize=11)
    ax.set_ylabel("€/MWh", fontsize=11)
    ax.set_title(
        "Precio de oferta óptimo vs. precio medio de mercado"
        + (f" — {tech_name}" if tech_name else ""),
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "3b_block_offers.png", dpi=_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Expected dispatch profile per order type
# ---------------------------------------------------------------------------

def plot_dispatch_profiles(
    results: list[dict], out: Path, tech_name: str
) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, r, color in zip(axes, results, _COLORS):
        dispatch = r["dispatch"]          # (S, T)
        T = dispatch.shape[1]
        periods = np.arange(1, T + 1)

        mean_d = dispatch.mean(axis=0)
        p10 = np.percentile(dispatch, 10, axis=0)
        p90 = np.percentile(dispatch, 90, axis=0)

        ax.fill_between(periods, p10, p90, color=color, alpha=0.25, label="P10-P90")
        ax.plot(periods, mean_d, color=color, linewidth=2, label="Media")

        ax.set_title(r["order_type"].upper(), fontsize=12)
        ax.set_xlabel("Periodo", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Energía casada (MWh)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Perfil de energía casada esperada por tipo de orden"
        + (f" — {tech_name}" if tech_name else ""),
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "4_dispatch_profiles.png", dpi=_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Risk-return frontier: E[Pi] vs CVaR as beta sweeps (H4, sensitivity analysis)
# ---------------------------------------------------------------------------

def plot_risk_frontier(frontier_df, out: Path, tech_name: str = "") -> None:
    import pandas as pd

    out = Path(out)
    figs_dir = out / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)
    for color, (order_type, group) in zip(_COLORS, frontier_df.groupby("order_type")):
        group = group.sort_values("beta")
        sizes = 30 + 120 * group["beta"]
        ax.plot(group["expected_profit"], group["cvar"], color=color, linewidth=1.2, alpha=0.6)
        ax.scatter(
            group["expected_profit"], group["cvar"], s=sizes, color=color,
            label=order_type.upper(), edgecolors="black", linewidths=0.5,
        )

    ax.set_xlabel("E[Π] (€)", fontsize=11)
    ax.set_ylabel("CVaR_α (€)", fontsize=11)
    ax.set_title(
        "Frontera beneficio-riesgo (tamaño del punto ∝ β)"
        + (f" — {tech_name}" if tech_name else ""),
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figs_dir / "5_risk_frontier.png", dpi=_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Ranking vs beta: annual objective per order type as risk aversion sweeps
# ---------------------------------------------------------------------------

_ORDER_COLORS = {
    "simple": _COLORS[0], "sco": _COLORS[1], "sbo": _COLORS[2],
    "exbo": _COLORS[3], "lsbo": _COLORS[4],
}


def plot_ranking_vs_beta(df, out_path: Path, tech_name: str = "") -> None:
    """df: columns beta, order_type, objective_value_per_mw (annual aggregate)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)
    for order_type, group in df.groupby("order_type"):
        group = group.sort_values("beta")
        ax.plot(group["beta"], group["objective_value_per_mw"],
                color=_ORDER_COLORS.get(order_type, "#52514e"),
                marker=_ORDER_MARKERS.get(order_type, "o"),
                markersize=5, linewidth=1.8, label=order_type.upper())

    ax.set_xlabel("β (aversión al riesgo)", fontsize=11)
    ax.set_ylabel("Valor objetivo (€/MW instalado)", fontsize=11)
    ax.set_title(
        "Valor objetivo anual por tipo de orden según β"
        + (f" — {tech_name}" if tech_name else ""),
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)


def plot_winner_by_beta(winners, out_path: Path, season_label: str = "") -> None:
    """Categorical map: winning order type per (technology, beta).

    winners: DataFrame with columns technology, beta, order_type.
    """
    from matplotlib.patches import Patch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    techs = sorted(winners["technology"].unique())
    betas = sorted(winners["beta"].unique())
    x_of = {b: i for i, b in enumerate(betas)}
    y_of = {t: i for i, t in enumerate(techs)}

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(techs) + 1.8))
    for _, row in winners.iterrows():
        ax.scatter(x_of[row["beta"]], y_of[row["technology"]],
                   marker="s", s=340,
                   color=_ORDER_COLORS.get(row["order_type"], "#52514e"))

    order_types = [ot for ot in _ORDER_COLORS if ot in set(winners["order_type"])]
    handles = [Patch(color=_ORDER_COLORS[ot], label=ot.upper()) for ot in order_types]
    ax.legend(handles=handles, title="Orden ganadora", fontsize=8, title_fontsize=9,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))

    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f"{b:g}" for b in betas], fontsize=9)
    ax.set_yticks(range(len(techs)))
    ax.set_yticklabels(techs, fontsize=9)
    ax.set_xlabel("β (aversión al riesgo)", fontsize=11)
    ax.set_title(
        "Orden ganadora por tecnología según β"
        + (f" — {season_label}" if season_label else ""),
        fontsize=12,
    )
    ax.set_xlim(-0.5, len(betas) - 0.5)
    ax.set_ylim(-0.5, len(techs) - 0.5)
    ax.invert_yaxis()
    ax.grid(alpha=0.15)

    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Family-level comparison: best order type per technology within a family
# ---------------------------------------------------------------------------

def plot_family_comparison(per_tech_rankings: dict, family_label: str, out: Path) -> None:
    out = Path(out)
    figs_dir = out / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    tech_names = list(per_tech_rankings.keys())
    best_profit_per_mw = [df["expected_profit_per_mw"].max() for df in per_tech_rankings.values()]
    best_order = [
        df.loc[df["expected_profit_per_mw"].idxmax(), "order_type"] for df in per_tech_rankings.values()
    ]

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    x = np.arange(len(tech_names))
    bars = ax.bar(x, best_profit_per_mw, color=_COLORS[: len(tech_names)] if len(tech_names) <= len(_COLORS)
                  else [_COLORS[i % len(_COLORS)] for i in range(len(tech_names))])
    for bar, order_type in zip(bars, best_order):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), order_type.upper(),
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(tech_names, fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("Mejor E[Π] por MW instalado entre tipos de orden (€/MW)", fontsize=11)
    ax.set_title(f"Comparativa por tecnología — {family_label}", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    fig.savefig(figs_dir / "family_comparison.png", dpi=_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Season comparison: one technology, verano vs invierno, grouped by order type
# ---------------------------------------------------------------------------

def plot_season_comparison(comparison, out: Path, tech_name: str = "") -> None:
    """comparison: DataFrame with columns season, order_type, expected_profit, cvar."""
    out = Path(out)
    figs_dir = out / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    seasons = list(comparison["season"].unique())
    order_types = list(comparison["order_type"].unique())
    x = np.arange(len(order_types))
    width = 0.8 / max(len(seasons), 1)

    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)
    for i, (season, color) in enumerate(zip(seasons, _COLORS)):
        sub = comparison[comparison["season"] == season].set_index("order_type")
        profits = [sub.loc[ot, "expected_profit"] if ot in sub.index else 0.0 for ot in order_types]
        offset = (i - (len(seasons) - 1) / 2) * width
        ax.bar(x + offset, profits, width=width, color=color, label=season, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([ot.upper() for ot in order_types], fontsize=10)
    ax.set_ylabel("E[Π] (€)", fontsize=11)
    ax.set_title(
        "Comparación estacional del beneficio esperado"
        + (f" — {tech_name}" if tech_name else ""),
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    fig.savefig(figs_dir / "season_comparison.png", dpi=_DPI)
    plt.close(fig)


def _plot_best_by_month(
    best, value_col: str, ylabel: str, title: str, out_path: Path, season_label: str = ""
) -> None:
    """One point per (technology, month) — the month's winning order type.

    ``best``: DataFrame with columns technology, month, order_type and
    ``value_col``. Color encodes the technology (fixed alphabetical
    assignment) and the marker shape encodes the winning order type, each with
    its own legend.
    """
    from matplotlib.lines import Line2D

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best = best.dropna(subset=[value_col])
    months = sorted(best["month"].unique())
    x_of = {mo: i for i, mo in enumerate(months)}
    techs = sorted(best["technology"].unique())
    order_types = [ot for ot in _ORDER_MARKERS if ot in set(best["order_type"])]

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    for tech, color in zip(techs, _TECH_COLORS):
        sub = best[best["technology"] == tech].sort_values("month")
        x = sub["month"].map(x_of)
        ax.plot(x, sub[value_col], color=color, linewidth=1.2, alpha=0.5)
        for otype, marker in _ORDER_MARKERS.items():
            pts = sub[sub["order_type"] == otype]
            if pts.empty:
                continue
            ax.scatter(pts["month"].map(x_of), pts[value_col],
                       marker=marker, s=60, color=color, zorder=3,
                       edgecolors="white", linewidths=1.0)

    tech_handles = [
        Line2D([], [], color=color, marker="o", linestyle="-", markersize=7,
               markeredgecolor="white", label=tech)
        for tech, color in zip(techs, _TECH_COLORS)
    ]
    marker_handles = [
        Line2D([], [], color="#52514e", marker=_ORDER_MARKERS[ot], linestyle="none",
               markersize=7, label=ot.upper())
        for ot in order_types
    ]
    leg_tech = ax.legend(handles=tech_handles, title="Tecnología", fontsize=8,
                         title_fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(leg_tech)
    leg_orders = ax.legend(handles=marker_handles, title="Mejor oferta", fontsize=8,
                           title_fontsize=9, loc="lower left", bbox_to_anchor=(1.01, 0.0))

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(
        title + (f" — {season_label}" if season_label else ""),
        fontsize=12,
    )
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight",
                bbox_extra_artists=(leg_tech, leg_orders))
    plt.close(fig)


def plot_best_objective_by_month(best, out_path: Path, season_label: str = "") -> None:
    """Best objective value per month for every technology in one figure."""
    _plot_best_by_month(
        best,
        value_col="objective_value_per_mw",
        ylabel="Mejor valor objetivo (€/MW instalado)",
        title="Mejor valor objetivo por mes y tecnología",
        out_path=out_path,
        season_label=season_label,
    )


def plot_best_expected_profit_by_month(best, out_path: Path, season_label: str = "") -> None:
    """E[Π]/MW of the month's objective-winning order, per technology."""
    _plot_best_by_month(
        best,
        value_col="expected_profit_per_mw",
        ylabel="E[Π] de la mejor oferta (€/MW instalado)",
        title="Beneficio esperado de la mejor oferta por mes y tecnología",
        out_path=out_path,
        season_label=season_label,
    )


def plot_family_season_comparison(comparison, family_label: str, out: Path) -> None:
    """comparison: DataFrame with columns technology, season, order_type, expected_profit."""
    out = Path(out)
    figs_dir = out / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    best = (
        comparison.loc[comparison.groupby(["technology", "season"])["expected_profit_per_mw"].idxmax()]
        .reset_index(drop=True)
    )
    tech_names = list(best["technology"].unique())
    seasons = list(best["season"].unique())
    x = np.arange(len(tech_names))
    width = 0.8 / max(len(seasons), 1)

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    for i, (season, color) in enumerate(zip(seasons, _COLORS)):
        sub = best[best["season"] == season].set_index("technology")
        profits = [sub.loc[t, "expected_profit_per_mw"] if t in sub.index else 0.0 for t in tech_names]
        order_types = [sub.loc[t, "order_type"] if t in sub.index else "" for t in tech_names]
        offset = (i - (len(seasons) - 1) / 2) * width
        bars = ax.bar(x + offset, profits, width=width, color=color, label=season, alpha=0.85)
        for bar, ot in zip(bars, order_types):
            if ot:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), ot.upper(),
                        ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(tech_names, fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("Mejor E[Π] por MW instalado entre tipos de orden (€/MW)", fontsize=11)
    ax.set_title(f"Comparativa estacional por tecnología — {family_label}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    fig.savefig(figs_dir / "family_season_comparison.png", dpi=_DPI)
    plt.close(fig)
