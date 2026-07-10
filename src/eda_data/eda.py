"""Análisis exploratorio (EDA) de precios OMIE y generación renovable.

Compara el régimen de verano (2025-06-01 a 2025-08-31) con el de invierno
(2025-12-01 a 2026-02-28) a resolución horaria, generando las figuras y la
tabla de estadísticos usados en la sección de EDA de la memoria.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
OUT_DIR = ROOT_DIR / "results" / "eda"
FIGS_DIR = OUT_DIR / "figs"

SEASONS = {
    "verano": {
        "label": "Verano 2025",
        "price_csv": DATA_DIR / "precios_omie_2025-06-01_2025-08-31_hour.csv",
        "gen_csv": DATA_DIR / "generacion_renovable_2025-06-01_2025-08-31_hour.csv",
        "color": "#FF9800",
    },
    "invierno": {
        "label": "Invierno 2025-2026",
        "price_csv": DATA_DIR / "precios_omie_2025-12-01_2026-02-28_hour.csv",
        "gen_csv": DATA_DIR / "generacion_renovable_2025-12-01_2026-02-28_hour.csv",
        "color": "#2196F3",
    },
}

_DPI = 150
_FIGSIZE_WIDE = (10, 5)
_FIGSIZE_STD = (7, 5)


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_season(key: str) -> pd.DataFrame:
    """Carga precio + generación renovable de un régimen y los une por (date, period)."""
    cfg = SEASONS[key]
    price = pd.read_csv(cfg["price_csv"], parse_dates=["date"])
    gen = pd.read_csv(cfg["gen_csv"], parse_dates=["date"])

    df = price.merge(
        gen[["date", "period", "wind_mwh", "solar_mwh", "renewable_total_mwh"]],
        on=["date", "period"],
        how="left",
    )
    df["season"] = key
    return df


# ---------------------------------------------------------------------------
# Estadísticos descriptivos (Tabla ~\ref{tab:estadisticos})
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame) -> dict[str, float]:
    p = df["price"]
    return {
        "mean": p.mean(),
        "median": p.median(),
        "std": p.std(),
        "p5": p.quantile(0.05),
        "p95": p.quantile(0.95),
        "min": p.min(),
        "max": p.max(),
        "n_le0": int((p <= 0).sum()),
        "pct_le0": 100 * (p <= 0).mean(),
        "n": len(p),
    }


def save_stats_csv(stats_by_season: dict[str, dict], path: Path) -> pd.DataFrame:
    table = pd.DataFrame(
        {SEASONS[k]["label"]: v for k, v in stats_by_season.items()}
    )
    table.to_csv(path)
    return table


def save_stats_tex(stats_by_season: dict[str, dict], path: Path) -> None:
    order = list(SEASONS.keys())
    v, i = stats_by_season[order[0]], stats_by_season[order[1]]
    labels = [SEASONS[k]["label"] for k in order]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Estadísticos descriptivos del precio horario por tramo.}",
        r"\label{tab:estadisticos}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Estadístico} & \textbf{" + labels[0] + r"} & \textbf{" + labels[1] + r"} \\",
        r"\midrule",
        rf"Media (\euro/MWh)          & {v['mean']:.2f} & {i['mean']:.2f} \\",
        rf"Mediana (\euro/MWh)        & {v['median']:.2f} & {i['median']:.2f} \\",
        rf"Desviación típica          & {v['std']:.2f} & {i['std']:.2f} \\",
        rf"Percentil 5 / 95           & {v['p5']:.2f} / {v['p95']:.2f} & {i['p5']:.2f} / {i['p95']:.2f} \\",
        rf"Mínimo / Máximo            & {v['min']:.2f} / {v['max']:.2f} & {i['min']:.2f} / {i['max']:.2f} \\",
        rf"Horas con precio $\le 0$   & {v['n_le0']} ({v['pct_le0']:.1f}\%) & {i['n_le0']} ({i['pct_le0']:.1f}\%) \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Figura 1: perfil intradiario medio con banda P10-P90 (fig:perfil)
# ---------------------------------------------------------------------------

def plot_perfil_intradiario(dfs: dict[str, pd.DataFrame], out: Path) -> None:
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)

    for key, df in dfs.items():
        g = df.groupby("period")["price"]
        mean = g.mean()
        p10 = g.quantile(0.10)
        p90 = g.quantile(0.90)
        color = SEASONS[key]["color"]

        ax.fill_between(mean.index, p10, p90, color=color, alpha=0.2)
        ax.plot(mean.index, mean, color=color, linewidth=2, label=SEASONS[key]["label"])

    ax.set_xlabel("Hora del día (periodo 1-24)", fontsize=11)
    ax.set_ylabel("Precio (€/MWh)", fontsize=11)
    ax.set_title("Perfil intradiario medio del precio, con banda P10-P90", fontsize=12)
    ax.set_xticks(range(1, 25, 2))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    fig.savefig(out / "perfil_intradiario.pdf", dpi=_DPI)
    fig.savefig(out / "perfil_intradiario.png", dpi=_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figura 2: distribución del precio horario (fig:distrib)
# ---------------------------------------------------------------------------

def plot_distribucion_precios(dfs: dict[str, pd.DataFrame], out: Path) -> None:
    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)

    all_prices = pd.concat([df["price"] for df in dfs.values()])
    bins = np.linspace(all_prices.min(), all_prices.max(), 60)

    for key, df in dfs.items():
        color = SEASONS[key]["color"]
        ax.hist(df["price"], bins=bins, density=True, color=color, alpha=0.45,
                label=SEASONS[key]["label"])
        ax.axvline(df["price"].mean(), color=color, linewidth=1.5, linestyle="--")

    ax.set_xlabel("Precio (€/MWh)", fontsize=11)
    ax.set_ylabel("Densidad", fontsize=11)
    ax.set_title("Distribución del precio horario por tramo", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "distribucion_precios.pdf", dpi=_DPI)
    fig.savefig(out / "distribucion_precios.png", dpi=_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figura 3: volatilidad diaria (fig:volatilidad)
# ---------------------------------------------------------------------------

def plot_volatilidad_diaria(dfs: dict[str, pd.DataFrame], out: Path) -> None:
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)

    for key, df in dfs.items():
        daily_std = df.groupby("date")["price"].std().sort_index()
        color = SEASONS[key]["color"]
        day_idx = np.arange(len(daily_std))

        ax.plot(day_idx, daily_std.values, color=color, linewidth=1.2, alpha=0.85,
                label=f"{SEASONS[key]['label']} (std diaria)")
        ax.axhline(daily_std.mean(), color=color, linewidth=1.5, linestyle="--",
                   label=f"Media {SEASONS[key]['label']}: {daily_std.mean():.1f} €/MWh")

    ax.set_xlabel("Día dentro del periodo", fontsize=11)
    ax.set_ylabel("Desviación típica diaria del precio (€/MWh)", fontsize=11)
    ax.set_title("Volatilidad diaria del precio por tramo", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "volatilidad_diaria.pdf", dpi=_DPI)
    fig.savefig(out / "volatilidad_diaria.png", dpi=_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figura 4: correlación precio vs. generación renovable (fig:correlacion)
# ---------------------------------------------------------------------------

def plot_correlacion_precio_renovable(dfs: dict[str, pd.DataFrame], out: Path) -> dict[str, float]:
    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)
    correlaciones = {}

    for key, df in dfs.items():
        d = df.dropna(subset=["renewable_total_mwh"])
        color = SEASONS[key]["color"]
        ax.scatter(d["renewable_total_mwh"], d["price"], s=8, color=color, alpha=0.35,
                   label=SEASONS[key]["label"])
        r = d["renewable_total_mwh"].corr(d["price"])
        correlaciones[key] = r

    ax.set_xlabel("Generación renovable eólica + solar (MWh/h)", fontsize=11)
    ax.set_ylabel("Precio (€/MWh)", fontsize=11)
    ax.set_title("Relación entre el precio horario y la generación renovable", fontsize=12)
    ax.legend(
        [f"{SEASONS[k]['label']} (r = {correlaciones[k]:.2f})" for k in dfs],
        fontsize=9,
    )
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "correlacion_precio_renovable.pdf", dpi=_DPI)
    fig.savefig(out / "correlacion_precio_renovable.png", dpi=_DPI)
    plt.close(fig)

    return correlaciones


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    dfs = {key: load_season(key) for key in SEASONS}
    stats_by_season = {key: compute_stats(df) for key, df in dfs.items()}

    stats_table = save_stats_csv(stats_by_season, OUT_DIR / "estadisticos.csv")
    save_stats_tex(stats_by_season, OUT_DIR / "tabla_estadisticos.tex")

    plot_perfil_intradiario(dfs, FIGS_DIR)
    plot_distribucion_precios(dfs, FIGS_DIR)
    plot_volatilidad_diaria(dfs, FIGS_DIR)
    correlaciones = plot_correlacion_precio_renovable(dfs, FIGS_DIR)

    print("=" * 70)
    print("Estadísticos descriptivos del precio horario por tramo")
    print("=" * 70)
    print(stats_table.to_string())

    print("\nCorrelación precio vs. generación renovable (Pearson r):")
    for key, r in correlaciones.items():
        print(f"  {SEASONS[key]['label']}: r = {r:.3f}")

    print("\nHora (periodo 1-24) de precio medio mínimo / máximo por tramo:")
    for key, df in dfs.items():
        mean_profile = df.groupby("period")["price"].mean()
        print(
            f"  {SEASONS[key]['label']}: mínimo en periodo {mean_profile.idxmin()} "
            f"({mean_profile.min():.2f} €/MWh), máximo en periodo {mean_profile.idxmax()} "
            f"({mean_profile.max():.2f} €/MWh)"
        )

    print(f"\nFiguras guardadas en: {FIGS_DIR}")
    print(f"Tabla de estadísticos guardada en: {OUT_DIR / 'estadisticos.csv'}")
    print(f"Tabla LaTeX guardada en: {OUT_DIR / 'tabla_estadisticos.tex'}")


if __name__ == "__main__":
    main()
