"""Comparative plots for the bidding strategy optimizer."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
_FIGSIZE_WIDE = (12, 5)
_FIGSIZE_STD = (9, 5)
_DPI = 150


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
        labels=labels,
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
