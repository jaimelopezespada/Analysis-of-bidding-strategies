"""Ejemplos ilustrativos: un dia concreto en el que SBO, EXBO y LSBO ganan.

El ranking anual real (yaml/run_full_year.yaml) nunca corona a SBO, EXBO ni
LSBO: Simple o SCO ganan siempre (Seccion "Resultados" de la memoria). La
Seccion 4.7 ("Valor de uso de SBO, EXBO y LSBO mas alla del ranking")
argumenta en prosa que estos productos si aportan valor en otros contextos.
Este script construye, para cada uno de los tres productos, un escenario de
precio de 24h (S dias/realizaciones equiprobables o ponderados, en parte
teoricos) en el que ese producto especifico supera a Simple y a SCO como
minimo.

Bajo el modelo cerrado del optimizador, con la MISMA rejilla de candidatos
para todos los productos, SCO domina debilmente a SBO (puede replicar su
regla de aceptacion con TF=0) y SBO/SCO dominan debilmente a LSBO (pueden
replicar el welfare conjunto de padre+hijo). Por eso, en estos ejemplos
ilustrativos, cada producto puede recibir su PROPIA rejilla de candidatos
(igual que EXBO/LSBO ya usan price_levels_pct propios por bloque en el
modelo real) -- una eleccion de diseno explicita, documentada aqui y en la
memoria, que representa agentes con distinta capacidad de calibracion/
expresividad de oferta, no una manipulacion oculta del resultado.

El mecanismo real (no dependiente de la rejilla) que hace ganar a cada
producto es distinto en cada escenario:
  - SBO: Simple busca su precio optimo periodo a periodo SIN restar el coste
    de arranque en esa busqueda (solo se resta despues, sobre el despacho ya
    fijado) por lo que puede aceptar un dia cuyo margen bruto no cubre el
    arranque. SBO explora tambien el mismo abanico de precios pero elige el
    que maximiza el beneficio REAL (con arranque incluido), evitando ese dia.
  - EXBO: sus bloques declaran su propia ventana horaria, independiente del
    perfil de disponibilidad de la tecnologia -- puede limitarse a UNA
    ventana rentable sin arrastrar la otra, pagando como mucho un arranque
    por dia (exclusividad, Sum x_b <= 1), mientras Simple puede arrancar dos
    veces en el mismo dia si ambas ventanas superan su umbral por separado.
  - LSBO: el bloque hijo declara su propia disponibilidad, independiente del
    array de la tecnologia -- puede declarar CAPACIDAD ADICIONAL (un segundo
    paquete/activo) que Simple/SBO/SCO, limitados al array de la tecnologia,
    no pueden ver ni despachar nunca.

Uso:
    python -m reporting.showcase_examples [--scenario {sbo,exbo,lsbo,all}]
        [--output-dir results/showcase] [--betas 0.0,0.5] [--no-assert]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bidding.cli import _ensure_utf8_stdout
from bidding.config import (
    AvailabilityConfig,
    BlockConfig,
    CandidateGrid,
    ExboGroupConfig,
    LsboFamilyConfig,
    RiskObjective,
    TechnologyConfig,
)
from bidding.metrics import objective_value
from bidding.orders import STRATEGIES
from bidding.plots import (
    plot_block_offers,
    plot_dispatch_profiles,
    plot_expected_profit_bars,
    plot_profit_distribution,
)
from bidding.ranking import build_ranking

_CVAR_ALPHA = 0.95


def _pct(prices: list[float], reference: float) -> list[float]:
    """Absolute EUR/MWh candidates -> sorted, deduplicated fractions of ``reference``."""
    return sorted({round(p / reference, 6) for p in prices})


@dataclass
class Scenario:
    label: str
    story: str
    tech: TechnologyConfig
    lambda_matrix: np.ndarray
    avail_matrix: np.ndarray
    probs: np.ndarray
    grids: dict[str, CandidateGrid]
    betas: list[float] = field(default_factory=lambda: [0.0, 0.5])
    order_types: list[str] = field(
        default_factory=lambda: ["simple", "sco", "sbo", "exbo", "lsbo"]
    )
    expected_winner: str = "simple"
    must_beat: tuple[str, ...] = ("simple", "sco")
    startup_per_transition: bool = False


# ---------------------------------------------------------------------------
# Escenario 1 -- SBO: "robustez sin prevision"
# ---------------------------------------------------------------------------

def build_sbo_scenario() -> Scenario:
    cost = 40.0
    startup = 11_000.0
    capacity = 80.0

    tech = TechnologyConfig(
        name="Cogeneracion rigida 80 MW",
        family=2,
        side="sell",
        variable_cost=cost,
        startup_cost=startup,
        availability=AvailabilityConfig(source="static", values=[capacity] * 24),
    )

    # Realizaciones de precio (plano en el dia) equiprobables. 45 EUR/MWh es
    # el dia "trampa": margen bruto diario = (45-40)*80*24 = 9.600 EUR, por
    # debajo del arranque (11.000 EUR) -> aceptarlo es una perdida neta.
    daily_prices = [20.0, 30.0, 45.0, 47.0, 50.0, 90.0]
    S = len(daily_prices)
    lambda_matrix = np.array([[p] * 24 for p in daily_prices])
    avail_matrix = np.tile(np.array(tech.availability.values), (S, 1))
    probs = np.full(S, 1.0 / S)

    generic = CandidateGrid(
        price_levels_pct=_pct([0, 20, 30, 40, 50, 60, 80], cost),
        tf_levels=[0, 3_000, 7_000, 15_000, 35_000],
        mar_levels=[0.0, 1.0],
        mav_fraction_levels=[0.0, 1.0],
    )
    sbo_own = CandidateGrid(
        price_levels_pct=_pct(daily_prices + [0.0], cost),
        mar_levels=[0.0, 1.0],
    )

    return Scenario(
        label="sbo_robustez_sin_prevision",
        story=(
            "Cogeneracion con perfil rigido de 80 MW y un coste de arranque real "
            "de 11.000 EUR. El agente desconoce el precio del dia siguiente y se "
            "enfrenta a 6 posibles realizaciones equiprobables (20, 30, 45, 47, 50 "
            "y 90 EUR/MWh, planas en el dia). La oferta Simple elige su precio "
            "optimo periodo a periodo SIN descontar el coste de arranque en esa "
            "busqueda, por lo que tambien acepta el dia marginal de 45 EUR/MWh, "
            "cuyo margen bruto diario (9.600 EUR) no cubre el arranque: un dia de "
            "perdida neta que Simple no ve venir. El SBO, con una rejilla de "
            "precio propia (mas fina que la generica), encuentra el precio de "
            "bloque exacto (47 EUR/MWh) que excluye ese dia marginal conservando "
            "los dias rentables."
        ),
        tech=tech,
        lambda_matrix=lambda_matrix,
        avail_matrix=avail_matrix,
        probs=probs,
        grids={"simple": generic, "sco": generic, "sbo": sbo_own},
        order_types=["simple", "sco", "sbo"],
        expected_winner="sbo",
        must_beat=("simple", "sco"),
    )


# ---------------------------------------------------------------------------
# Escenario 2 -- EXBO: "ventana movil / exclusividad"
# ---------------------------------------------------------------------------

def build_exbo_scenario() -> Scenario:
    cost = 30.0
    startup = 900.0
    capacity = 50.0

    manana_mask = [0] * 6 + [1] * 4 + [0] * 14   # horas 6-9
    tarde_mask = [0] * 18 + [1] * 4 + [0] * 2     # horas 18-21

    tech = TechnologyConfig(
        name="Almacenamiento 50 MW (ventana diaria unica)",
        family=4,
        side="sell",
        variable_cost=cost,
        startup_cost=startup,
        availability=AvailabilityConfig(source="static", values=[capacity] * 24),
        exbo_groups=[
            ExboGroupConfig(
                group_id="ventana_diaria",
                blocks=[
                    BlockConfig(
                        id="manana",
                        price_levels_pct=_pct([36, 42, 48, 54, 60, 66], cost),
                        availability=AvailabilityConfig(
                            source="static",
                            values=[capacity * m for m in manana_mask],
                        ),
                    ),
                    BlockConfig(
                        id="tarde",
                        price_levels_pct=_pct([36, 42, 48, 54, 60, 66], cost),
                        availability=AvailabilityConfig(
                            source="static",
                            values=[capacity * m for m in tarde_mask],
                        ),
                    ),
                ],
            )
        ],
    )

    def day(manana_price: float, tarde_price: float, base: float = 15.0) -> np.ndarray:
        arr = np.full(24, base)
        arr[np.array(manana_mask, dtype=bool)] = manana_price
        arr[np.array(tarde_mask, dtype=bool)] = tarde_price
        return arr

    lambda_matrix = np.array([
        day(80.0, 15.0),   # pico_manana: solo la ventana de manana es rentable
        day(15.0, 80.0),   # pico_tarde: solo la ventana de tarde es rentable
        day(80.0, 32.0),   # doble_debil_A: manana fuerte, tarde apenas rentable
        day(32.0, 80.0),   # doble_debil_B: espejo de la anterior
        day(15.0, 15.0),   # plano_malo: no hay mercado ese dia
    ])
    avail_matrix = np.tile(np.array(tech.availability.values), (5, 1))
    probs = np.array([0.20, 0.20, 0.25, 0.25, 0.10])

    generic = CandidateGrid(
        price_levels_pct=_pct([0, 15, 22.5, 30, 32, 45, 60, 80], cost),
        tf_levels=[0, 500, 1_000, 2_000, 5_000],
        mar_levels=[0.0, 1.0],
        mav_fraction_levels=[0.0, 1.0],
    )

    return Scenario(
        label="exbo_ventana_movil",
        story=(
            "Almacenamiento de 50 MW que descarga en una unica ventana diaria, "
            "pero sin saber de antemano si manana el pico de precio se dara por "
            "la manana (horas 6-9) o por la tarde (horas 18-21). En los dias en "
            "que ambas ventanas resultan minimamente atractivas a la vez (precio "
            "alto en una, apenas rentable en la otra), la oferta Simple -- con "
            "arranque por transicion -- prueba a arrancar en las dos, pagando dos "
            "arranques (900 EUR cada uno) por apenas 400 EUR de margen adicional "
            "en la ventana debil: un error que el bloque exclusivo EXBO no puede "
            "cometer, pues solo puede casar una de sus dos ventanas por dia "
            "(Sum x_b <= 1) y siempre elige la de mayor excedente."
        ),
        tech=tech,
        lambda_matrix=lambda_matrix,
        avail_matrix=avail_matrix,
        probs=probs,
        grids={"simple": generic, "sco": generic, "sbo": generic, "exbo": generic},
        order_types=["simple", "sco", "sbo", "exbo"],
        expected_winner="exbo",
        must_beat=("simple", "sco"),
        startup_per_transition=True,
    )


# ---------------------------------------------------------------------------
# Escenario 3 -- LSBO: "paquete incremental, aproximacion en unidad unica"
# ---------------------------------------------------------------------------

def build_lsbo_scenario() -> Scenario:
    cost = 35.0
    startup = 8_000.0
    base_capacity = 70.0
    extra_capacity = 30.0

    tech = TechnologyConfig(
        name="Planta con paquete incremental 70+30 MW",
        family=2,
        side="sell",
        variable_cost=cost,
        startup_cost=startup,
        # Disponibilidad a nivel de tecnologia: SOLO los 70 MW "base". Simple/
        # SBO/SCO usan siempre este array -- nunca pueden ver ni despachar los
        # 30 MW adicionales del bloque hijo (representa un segundo paquete o
        # activo que solo se puede enlazar via LSBO).
        availability=AvailabilityConfig(source="static", values=[base_capacity] * 24),
        lsbo_families=[
            LsboFamilyConfig(
                family_id="base_mas_incremental",
                parent=BlockConfig(
                    id="base",
                    price_levels_pct=_pct([35, 37.8, 40.25, 45.5, 50.75], cost),
                    availability=AvailabilityConfig(
                        source="static", values=[base_capacity] * 24
                    ),
                ),
                children=[
                    BlockConfig(
                        id="incremental",
                        price_levels_pct=_pct([42, 49, 50, 56, 63], cost),
                        availability=AvailabilityConfig(
                            source="static", values=[extra_capacity] * 24
                        ),
                    )
                ],
            )
        ],
    )

    daily_prices = [20.0, 38.0, 50.0, 65.0, 90.0]
    S = len(daily_prices)
    lambda_matrix = np.array([[p] * 24 for p in daily_prices])
    avail_matrix = np.tile(np.array(tech.availability.values), (S, 1))
    probs = np.full(S, 1.0 / S)

    generic = CandidateGrid(
        price_levels_pct=_pct([0, 17.5, 26.25, 35, 40.25, 45.5, 50, 56, 65, 90], cost),
        tf_levels=[0, 2_000, 4_000, 8_000, 16_000],
        mar_levels=[0.0, 1.0],
        mav_fraction_levels=[0.0, 1.0],
    )
    lsbo_own = CandidateGrid(
        price_levels_pct=_pct(
            [0, 35, 37.8, 40.25, 42, 45.5, 49, 50, 56, 63, 65, 90], cost
        ),
    )

    return Scenario(
        label="lsbo_paquete_incremental",
        story=(
            "Planta cuya capacidad 'oficial' (declarada a Simple/SBO/SCO) es de "
            "70 MW, pero que dispone de un segundo paquete incremental de 30 MW "
            "solo activable mediante una LSBO (padre = base de 70 MW, hijo = "
            "paquete adicional de 30 MW, ambos con el mismo coste variable). En "
            "los dias en que el precio supera tambien el umbral del hijo, la "
            "LSBO despacha 100 MW mientras Simple/SBO/SCO permanecen limitados "
            "estructuralmente a los 70 MW declarados: no es una ventaja de "
            "calibracion sino de capacidad -- el hijo declara una disponibilidad "
            "propia que el array de la tecnologia jamas contiene."
        ),
        tech=tech,
        lambda_matrix=lambda_matrix,
        avail_matrix=avail_matrix,
        probs=probs,
        grids={"simple": generic, "sco": generic, "sbo": generic, "lsbo": lsbo_own},
        order_types=["simple", "sco", "sbo", "lsbo"],
        expected_winner="lsbo",
        must_beat=("simple", "sco"),
    )


BUILDERS = {
    "sbo": build_sbo_scenario,
    "exbo": build_exbo_scenario,
    "lsbo": build_lsbo_scenario,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def plot_scenario_prices(scn: Scenario, out_path: Path) -> None:
    """Curvas de precio construidas + ventanas de bloque declaradas (bespoke)."""
    T = scn.lambda_matrix.shape[1]
    periods = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for s in range(scn.lambda_matrix.shape[0]):
        ax.plot(
            periods, scn.lambda_matrix[s], marker="o", markersize=3, linewidth=1.5,
            label=f"Escenario {s + 1} (p={scn.probs[s]:.2f})",
        )

    blocks: list[tuple[str, "BlockConfig"]] = []
    for g in (scn.tech.exbo_groups or []):
        blocks.extend((b.id, b) for b in g.blocks)
    for fam in (scn.tech.lsbo_families or []):
        blocks.append((fam.parent.id, fam.parent))
        blocks.extend((c.id, c) for c in fam.children)

    y_lo, y_hi = float(scn.lambda_matrix.min()), float(scn.lambda_matrix.max())
    y_span = (y_hi - y_lo) or 1.0
    seen: set[str] = set()
    n_labeled = 0
    for i, (block_id, block) in enumerate(blocks):
        if block_id in seen:
            continue
        seen.add(block_id)
        mask = np.asarray(block.availability.values, dtype=float) > 0.0
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        ax.axvspan(idx.min() + 0.5, idx.max() + 1.5, color=f"C{i}", alpha=0.10)
        y_label = y_hi + y_span * (0.04 + 0.07 * n_labeled)
        ax.text(idx.mean() + 1, y_label, block_id, ha="center", va="bottom", fontsize=8, alpha=0.8)
        n_labeled += 1

    if n_labeled:
        ax.set_ylim(y_lo - y_span * 0.05, y_hi + y_span * (0.10 + 0.07 * n_labeled))

    ax.set_xlabel("Periodo (hora)", fontsize=11)
    ax.set_ylabel("EUR/MWh", fontsize=11)
    ax.set_title(f"Escenarios de precio construidos -- {scn.label}", fontsize=12)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_scenario(
    scn: Scenario, output_dir: Path, assert_winner: bool = True
) -> dict[float, pd.DataFrame]:
    print(f"\n{'=' * 78}\n{scn.label}\n{'=' * 78}")
    print(scn.story)

    scn_dir = output_dir / scn.label
    scn_dir.mkdir(parents=True, exist_ok=True)

    rankings: dict[float, pd.DataFrame] = {}
    any_mismatch = False

    for beta in scn.betas:
        objective = RiskObjective(beta=beta)
        results = []
        for order_type in scn.order_types:
            strategy = STRATEGIES[order_type]()
            grid = scn.tech.resolved_grid(scn.grids[order_type])
            result = strategy.evaluate(
                tech=scn.tech,
                lambda_matrix=scn.lambda_matrix,
                avail_matrix=scn.avail_matrix,
                probs=scn.probs,
                grid=grid,
                objective=objective,
                cvar_alpha=_CVAR_ALPHA,
                startup_per_transition=scn.startup_per_transition,
            )
            result["objective_value"] = objective_value(
                result["profits"], scn.probs, _CVAR_ALPHA, objective
            )
            result["objective_value_per_mw"] = (
                result["objective_value"] / scn.tech.installed_capacity_mw
            )
            results.append(result)

        ranking = build_ranking(results)
        rankings[beta] = ranking

        beta_dir = scn_dir / f"beta_{beta:g}"
        beta_dir.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(beta_dir / "ranking.csv")

        print(f"\n--- beta = {beta} ---")
        print(ranking.drop(columns=["optimal_params"]).to_string())

        winner = ranking.iloc[0]["order_type"]
        winner_obj = ranking.iloc[0]["objective_value"]
        gaps = {}
        for ot in scn.must_beat:
            row = ranking[ranking["order_type"] == ot]
            if not row.empty:
                gaps[ot] = winner_obj - row.iloc[0]["objective_value"]

        ok = winner == scn.expected_winner and all(g > 0 for g in gaps.values())
        if not ok:
            any_mismatch = True
        print(
            f"Ganador esperado: {scn.expected_winner.upper()}  |  "
            f"Ganador real: {winner.upper()}  |  {'OK' if ok else '**MISMATCH**'}"
        )
        for ot, g in gaps.items():
            print(f"  gap vs {ot.upper()}: {g:+.2f}")

        figs_dir = beta_dir / "figs"
        figs_dir.mkdir(parents=True, exist_ok=True)
        try:
            plot_expected_profit_bars(results, figs_dir, scn.label)
            plot_profit_distribution(results, figs_dir, scn.label)
            plot_dispatch_profiles(results, figs_dir, scn.label)
            plot_block_offers(
                results, scn.lambda_matrix, scn.tech, figs_dir, scn.label,
                scn.tech.installed_capacity_mw,
            )
        except Exception as exc:  # pragma: no cover - illustrative script
            print(f"[WARN] fallo al generar figuras ({beta=}): {exc}", file=sys.stderr)

        if assert_winner:
            assert winner == scn.expected_winner, (
                f"[{scn.label}, beta={beta}] esperado ganador "
                f"'{scn.expected_winner}', obtenido '{winner}'"
            )
            for ot, g in gaps.items():
                assert g > 0, (
                    f"[{scn.label}, beta={beta}] '{scn.expected_winner}' no "
                    f"supera a '{ot}' (gap={g:.2f})"
                )

    try:
        plot_scenario_prices(scn, scn_dir / "scenario_prices.png")
    except Exception as exc:  # pragma: no cover - illustrative script
        print(f"[WARN] fallo al generar scenario_prices: {exc}", file=sys.stderr)

    if any_mismatch and not assert_winner:
        print(f"[WARN] {scn.label}: al menos un beta no cumplio el objetivo esperado.")

    return rankings


def _write_summary_table(all_rankings: dict[str, dict[float, pd.DataFrame]], output_dir: Path) -> None:
    rows = []
    for label, by_beta in all_rankings.items():
        for beta, ranking in by_beta.items():
            winner = ranking.iloc[0]
            row = {
                "scenario": label,
                "beta": beta,
                "winner": winner["order_type"],
                "winner_objective": winner["objective_value"],
            }
            for ot in ("simple", "sco"):
                sub = ranking[ranking["order_type"] == ot]
                row[f"{ot}_objective"] = sub.iloc[0]["objective_value"] if not sub.empty else float("nan")
                row[f"gap_vs_{ot}"] = (
                    winner["objective_value"] - sub.iloc[0]["objective_value"]
                    if not sub.empty else float("nan")
                )
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = output_dir / "summary_table.csv"
    df.to_csv(out_path, index=False)
    print(f"\nTabla resumen guardada en: {out_path}")
    print(df.to_string(index=False))


def main() -> None:
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", choices=["sbo", "exbo", "lsbo", "all"], default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/showcase"))
    parser.add_argument(
        "--betas", type=str, default=None,
        help="Lista separada por comas, p.ej. '0.0,0.5'. Por defecto usa la del escenario.",
    )
    parser.add_argument(
        "--no-assert", action="store_true",
        help="No abortar si un escenario no cumple su objetivo (modo exploracion).",
    )
    args = parser.parse_args()

    labels = list(BUILDERS) if args.scenario == "all" else [args.scenario]
    betas_override = (
        [float(b) for b in args.betas.split(",")] if args.betas else None
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rankings: dict[str, dict[float, pd.DataFrame]] = {}
    for label in labels:
        scn = BUILDERS[label]()
        if betas_override is not None:
            scn.betas = betas_override
        all_rankings[label] = run_scenario(
            scn, args.output_dir, assert_winner=not args.no_assert
        )

    _write_summary_table(all_rankings, args.output_dir)


if __name__ == "__main__":
    main()
