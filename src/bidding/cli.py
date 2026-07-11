"""Command-line interface: python -m bidding run --tech <yaml> --run <yaml>"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml as pyyaml

from .availability import load_renewable_availability, static_availability_matrix
from .config import RunConfig, TechnologyConfig
from .frontier import sweep_beta
from .metrics import objective_value
from .orders import STRATEGIES
from .plots import plot_all, plot_risk_frontier
from .prices import load_price_matrix
from .ranking import build_ranking


def _tech_slug(name: str) -> str:
    """Convert a technology name to a filesystem-safe directory name.

    "CCGT 400 MW"      → "ccgt_400_mw"
    "Solar FV 100 MW"  → "solar_fv_100_mw"
    """
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def _ensure_utf8_stdout() -> None:
    """Windows consoles default to cp1252, which cannot encode Π/€ combos."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def tech_output_dir(cfg: RunConfig, tech_name: str) -> Path:
    """results/<season>/<tech_slug>/ when cfg.season is set, else results/<tech_slug>/.

    Namespacing by season keeps two run configs that share the same
    output_dir (e.g. run_verano.yaml / run_invierno.yaml) from overwriting
    each other's ranking.csv/figs when run against the same technology.
    """
    base = Path(cfg.output_dir)
    if cfg.season:
        base = base / cfg.season
    return base / _tech_slug(tech_name)


def discover_tech_yamls(yaml_dir: str = "yaml") -> list[Path]:
    """All technology YAMLs in yaml_dir — those with a `family:` key, which
    run/execution configs lack by construction (same rule as
    family.discover_family_techs, defined here to avoid a circular import)."""
    matches = []
    for path in sorted(Path(yaml_dir).glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            data = pyyaml.safe_load(f)
        if isinstance(data, dict) and "family" in data:
            matches.append(path)
    return matches


def load_scenarios(tech: TechnologyConfig, cfg: RunConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load (lambda_matrix, avail_matrix, probs, scenario_labels) for a tech+run pair."""
    lambda_matrix, scenario_labels = load_price_matrix(cfg.prices, cfg.resolution)
    S, T = lambda_matrix.shape

    if tech.availability.source == "static":
        n_avail = len(tech.availability.values)
        if n_avail != T:
            raise ValueError(
                f"Technology availability has {n_avail} periods but "
                f"resolution={cfg.resolution} and the CSV has {T} periods per day."
            )
        avail_matrix = static_availability_matrix(tech, S)
    else:
        avail_matrix = load_renewable_availability(tech, cfg.prices, cfg.resolution, scenario_labels)

    if cfg.mode == "deterministic":
        lambda_matrix = lambda_matrix.mean(axis=0, keepdims=True)
        avail_matrix = avail_matrix.mean(axis=0, keepdims=True)
        scenario_labels = ["mean"]
        S = 1

    probs = np.ones(S) / S
    return lambda_matrix, avail_matrix, probs, scenario_labels


def evaluate_technology(tech: TechnologyConfig, cfg: RunConfig):
    """
    Pure computation: evaluate every configured order type for one technology
    and return (results, ranking, lambda_matrix). No printing, no file I/O —
    reused by both the single-technology `run` command and `family`.
    """
    lambda_matrix, avail_matrix, probs, _ = load_scenarios(tech, cfg)

    results: list[dict] = []
    for order_type in cfg.order_types:
        if order_type not in STRATEGIES:
            continue
        strategy = STRATEGIES[order_type]()
        grid = tech.effective_grid(cfg.candidate_grid)
        result = strategy.evaluate(
            tech=tech,
            lambda_matrix=lambda_matrix,
            avail_matrix=avail_matrix,
            probs=probs,
            grid=grid,
            objective=cfg.objective,
            cvar_alpha=cfg.cvar_alpha,
            startup_per_transition=cfg.startup_per_transition,
            sco_model=cfg.sco_model,
        )
        result["objective_value"] = objective_value(
            result["profits"], probs, cfg.cvar_alpha, cfg.objective
        )
        results.append(result)

    ranking = build_ranking(results) if results else None
    return results, ranking, lambda_matrix


def run_command(
    tech_path: str,
    run_path: str,
    mode_override: str | None = None,
    startup_override: bool | None = None,
    sco_model_override: str | None = None,
    exit_on_error: bool = True,
) -> None:
    _ensure_utf8_stdout()
    tech = TechnologyConfig.from_yaml(tech_path)
    cfg = RunConfig.from_yaml(run_path)

    if mode_override:
        cfg = cfg.model_copy(update={"mode": mode_override})
    if startup_override is not None:
        cfg = cfg.model_copy(update={"startup_per_transition": startup_override})
    if sco_model_override:
        cfg = cfg.model_copy(update={"sco_model": sco_model_override})

    np.random.seed(cfg.seed)

    try:
        lambda_matrix, avail_matrix, probs, scenario_labels = load_scenarios(tech, cfg)
    except ValueError as exc:
        if not exit_on_error:
            raise
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    S = lambda_matrix.shape[0]

    # ---------------------------------------------------------------- header
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Tecnología : {tech.name}")
    print(f"  Modo       : {cfg.mode}  |  Resolución : {cfg.resolution} MTU")
    print(f"  Escenarios : {S}  |  Tipos de orden : {', '.join(cfg.order_types)}")
    if "sco" in cfg.order_types:
        print(f"  Modelo SCO : {cfg.sco_model}")
    if cfg.startup_per_transition:
        print("  Arranques  : uno por cada transición cero→producción")
    print(sep)

    # --------------------------------------------------------- evaluate each order type
    results: list[dict] = []
    for order_type in cfg.order_types:
        if order_type not in STRATEGIES:
            print(f"[WARN] Unknown order type '{order_type}' — skipped.", file=sys.stderr)
            continue

        strategy = STRATEGIES[order_type]()
        print(f"\n  Evaluando {order_type.upper()}...", end="  ", flush=True)

        grid = tech.effective_grid(cfg.candidate_grid)
        result = strategy.evaluate(
            tech=tech,
            lambda_matrix=lambda_matrix,
            avail_matrix=avail_matrix,
            probs=probs,
            grid=grid,
            objective=cfg.objective,
            cvar_alpha=cfg.cvar_alpha,
            startup_per_transition=cfg.startup_per_transition,
            sco_model=cfg.sco_model,
        )
        result["objective_value"] = objective_value(
            result["profits"], probs, cfg.cvar_alpha, cfg.objective
        )
        results.append(result)

        print(
            f"E[Π] = {result['expected_profit']:>10,.0f} €   "
            f"CVaR = {result['cvar']:>10,.0f} €   "
            f"P_match = {result['match_probability']:.1%}   "
            f"E[Π]/MW = {result['expected_profit_per_mw']:>8,.1f} €/MW"
        )

    if not results:
        print("[ERROR] No valid order types evaluated.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------ ranking
    ranking = build_ranking(results)

    print(f"\n{sep}")
    print("  RANKING (por valor objetivo)")
    print(sep)
    display_cols = ["order_type", "objective_value", "expected_profit", "cvar",
                    "match_probability", "expected_matched_energy", "expected_profit_per_mw"]
    print(ranking[display_cols].to_string())
    print()

    # ------------------------------------------------------------ output
    output_dir = tech_output_dir(cfg, tech.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranking_path = output_dir / "ranking.csv"
    ranking.to_csv(ranking_path)
    print(f"  Ranking guardado en : {ranking_path}")

    try:
        plot_all(results, lambda_matrix, output_dir, tech_name=tech.name,
                 capacity_mw=tech.installed_capacity_mw)
        print(f"  Gráficas en        : {output_dir / 'figs'}")
    except Exception as exc:
        print(f"  [WARN] Gráficas no generadas: {exc}", file=sys.stderr)

    # ------------------------------------------------------- risk frontier (optional)
    if cfg.beta_sweep:
        print(f"\n  Calculando frontera beneficio-riesgo (beta = {cfg.beta_sweep})...")
        grid = tech.effective_grid(cfg.candidate_grid)
        frontier_frames = [
            sweep_beta(
                tech, lambda_matrix, avail_matrix, probs, grid, cfg.cvar_alpha,
                ot, cfg.beta_sweep, startup_per_transition=cfg.startup_per_transition,
                sco_model=cfg.sco_model,
            )
            for ot in cfg.order_types
            if ot in STRATEGIES
        ]
        import pandas as pd
        frontier_df = pd.concat(frontier_frames, ignore_index=True)
        frontier_path = output_dir / "frontier.csv"
        frontier_df.to_csv(frontier_path, index=False)
        print(f"  Frontera guardada en : {frontier_path}")
        try:
            plot_risk_frontier(frontier_df, output_dir, tech_name=tech.name)
            print(f"  Gráfica en           : {output_dir / 'figs' / '5_risk_frontier.png'}")
        except Exception as exc:
            print(f"  [WARN] Gráfica de frontera no generada: {exc}", file=sys.stderr)

    print(f"\n{sep}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bidding",
        description="Optimizador de estrategias de oferta — mercado diario eléctrico",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Evaluar tipos de orden y generar ranking")
    run_p.add_argument("--tech", required=True, metavar="YAML|all",
                       help="Fichero YAML de tecnología (ej: yaml/solar_fv.yaml), "
                            "o 'all' para todas las tecnologías de --yaml-dir")
    run_group = run_p.add_mutually_exclusive_group(required=True)
    run_group.add_argument("--run", metavar="YAML",
                           help="Fichero YAML de ejecución (ej: yaml/run.yaml)")
    run_group.add_argument("--run-verano", metavar="YAML",
                           help="Run YAML de verano (usar junto a --run-invierno)")
    run_p.add_argument("--run-invierno", metavar="YAML",
                       help="Run YAML de invierno (requerido si se usa --run-verano)")
    run_p.add_argument("--yaml-dir", default="yaml",
                       help="Directorio con los YAML de tecnologia para --tech all (por defecto: yaml/)")
    run_p.add_argument("--mode", choices=["deterministic", "stochastic"],
                       help="Sobreescribe mode del run YAML")
    run_p.add_argument("--startup-per-transition", action=argparse.BooleanOptionalAction,
                       default=None,
                       help="Sobreescribe startup_per_transition del run YAML: cobra un "
                            "arranque por cada transición cero→producción (solo ofertas simples)")
    run_p.add_argument("--sco-model", choices=["naive", "aware"], default=None,
                       help="Sobreescribe sco_model del run YAML: 'aware' (casación con "
                            "el excedente declarado, como Euphemia) o 'naive' (casación "
                            "con el beneficio real — cota de conocimiento perfecto)")

    family_p = sub.add_parser("family", help="Evaluar todas las tecnologias de una familia")
    family_p.add_argument("--family-num", required=True,
                          help="Numero de familia (1-4) o 'all'")
    family_group = family_p.add_mutually_exclusive_group(required=True)
    family_group.add_argument("--run", metavar="YAML",
                              help="Fichero YAML de ejecucion (una sola temporada/dataset)")
    family_group.add_argument("--run-verano", metavar="YAML",
                              help="Run YAML de verano (usar junto a --run-invierno)")
    family_p.add_argument("--run-invierno", metavar="YAML",
                          help="Run YAML de invierno (requerido si se usa --run-verano)")
    family_p.add_argument("--yaml-dir", default="yaml",
                          help="Directorio con los YAML de tecnologia (por defecto: yaml/)")

    oos_p = sub.add_parser(
        "validate-oos",
        help="Validacion out-of-sample de la SCO: optimiza theta* en train "
             "(dias iniciales) y lo evalua mecanicamente en test (dias finales)",
    )
    oos_p.add_argument("--tech", required=True, metavar="YAML",
                       help="Fichero YAML de tecnologia")
    oos_p.add_argument("--run", required=True, metavar="YAML",
                       help="Fichero YAML de ejecucion")
    oos_p.add_argument("--train-fraction", type=float, default=0.7,
                       help="Fraccion cronologica inicial de escenarios usada como train "
                            "(por defecto: 0.7)")
    oos_p.add_argument("--sco-model", choices=["naive", "aware"], default=None,
                       help="Sobreescribe sco_model del run YAML")

    seasons_p = sub.add_parser(
        "compare-seasons", help="Comparar una tecnologia entre verano e invierno"
    )
    seasons_p.add_argument("--tech", required=True, metavar="YAML",
                           help="Fichero YAML de tecnologia")
    seasons_p.add_argument("--run-verano", required=True, metavar="YAML",
                           help="Run YAML de verano")
    seasons_p.add_argument("--run-invierno", required=True, metavar="YAML",
                           help="Run YAML de invierno")

    args = parser.parse_args()

    if args.command == "run":
        if args.run_verano and not args.run_invierno:
            parser.error("--run-invierno es requerido junto a --run-verano")
        run_paths = [args.run] if args.run else [args.run_verano, args.run_invierno]

        if args.tech == "all":
            tech_paths = discover_tech_yamls(args.yaml_dir)
            if not tech_paths:
                parser.error(f"No se encontraron YAML de tecnologia en {args.yaml_dir}/")
        else:
            tech_paths = [Path(args.tech)]

        batch = len(tech_paths) > 1 or len(run_paths) > 1
        for run_path in run_paths:
            for tech_path in tech_paths:
                try:
                    run_command(
                        str(tech_path), str(run_path),
                        mode_override=args.mode,
                        startup_override=args.startup_per_transition,
                        sco_model_override=args.sco_model,
                        exit_on_error=not batch,
                    )
                except ValueError as exc:
                    print(f"[WARN] {tech_path} ({run_path}): {exc} — omitida.", file=sys.stderr)
    elif args.command == "family":
        from .family import run_family, run_family_seasons
        if args.run:
            run_family(args.family_num, args.run, yaml_dir=args.yaml_dir)
        else:
            if not args.run_invierno:
                parser.error("--run-invierno es requerido junto a --run-verano")
            run_family_seasons(
                args.family_num, args.run_verano, args.run_invierno, yaml_dir=args.yaml_dir
            )
    elif args.command == "validate-oos":
        from .validation import run_validate_oos
        run_validate_oos(
            args.tech, args.run,
            train_fraction=args.train_fraction,
            sco_model_override=args.sco_model,
        )
    elif args.command == "compare-seasons":
        from .seasons import run_compare_seasons
        run_compare_seasons(args.tech, args.run_verano, args.run_invierno)
