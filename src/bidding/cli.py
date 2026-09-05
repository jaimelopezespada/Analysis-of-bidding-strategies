"""Command-line interface: python -m bidding run --tech <yaml> --run <yaml>"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml as pyyaml

from reporting.frontier import sweep_beta

from .availability import load_renewable_availability, static_availability_matrix
from .config import RunConfig, TechnologyConfig
from .metrics import objective_value
from .monthly import monthly_mean_price, resolve_tech_for_month, split_by_month
from .orders import STRATEGIES
from .plots import plot_all, plot_risk_frontier
from .prices import load_price_matrix
from .ranking import aggregate_results, build_aggregate_ranking, build_ranking


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
    """Load (lambda_matrix, avail_matrix, probs, scenario_labels) for a tech+run pair.

    Always returns the full per-day matrices; deterministic-mode collapsing
    happens per month inside iter_month_runs (one mean day per month).
    """
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

    probs = np.ones(S) / S
    return lambda_matrix, avail_matrix, probs, scenario_labels


def iter_month_runs(tech: TechnologyConfig, cfg: RunConfig):
    """Yield one optimization input set per calendar month of the run.

    The run is partitioned by month because a month-dependent cost
    (variable_cost_source="monthly_price_mean") requires one bid curve per
    month. Per month: the mean hourly price is computed BEFORE any
    deterministic collapse, the technology is resolved to a static-cost copy,
    matrices are sliced (and collapsed to a single mean day in deterministic
    mode), probabilities are renormalized uniform, and the candidate grid is
    resolved to absolute prices against the month's price reference.
    """
    lambda_matrix, avail_matrix, _, scenario_labels = load_scenarios(tech, cfg)

    for month, idx in split_by_month(scenario_labels):
        month_mean = monthly_mean_price(lambda_matrix, idx)
        tech_m = resolve_tech_for_month(tech, month, month_mean)
        lam_m = lambda_matrix[idx]
        avail_m = avail_matrix[idx]
        n_days = len(idx)
        if cfg.mode == "deterministic":
            lam_m = lam_m.mean(axis=0, keepdims=True)
            avail_m = avail_m.mean(axis=0, keepdims=True)
        probs_m = np.ones(lam_m.shape[0]) / lam_m.shape[0]
        grid_m = tech_m.resolved_grid(cfg.candidate_grid)
        cost = tech_m.variable_cost
        yield {
            "month": month,
            "n_days": n_days,
            "tech": tech_m,
            "grid": grid_m,
            "lambda_matrix": lam_m,
            "avail_matrix": avail_m,
            "probs": probs_m,
            "variable_cost": float(np.mean(cost)) if isinstance(cost, list) else float(cost),
        }


def evaluate_technology(tech: TechnologyConfig, cfg: RunConfig, verbose: bool = False):
    """
    Evaluate every configured order type for one technology, one optimization
    per calendar month, and return (monthly, aggregate_ranking).

    ``monthly`` is a list of per-month dicts (month, n_days, variable_cost,
    tech, grid, matrices, results, ranking); ``aggregate_ranking`` re-scores
    the concatenated per-day outcomes across months (see
    ranking.build_aggregate_ranking). No file I/O — reused by the
    single-technology `run` command, `family` and `compare-seasons`.
    """
    dynamic_cost = tech.variable_cost_source == "monthly_price_mean"

    monthly: list[dict] = []
    for mr in iter_month_runs(tech, cfg):
        if verbose:
            cost_note = f"  |  C_V resuelto = {mr['variable_cost']:.2f} €/MWh" if dynamic_cost else ""
            print(f"\n  --- {mr['month']} ({mr['n_days']} días){cost_note} ---")

        results: list[dict] = []
        for order_type in cfg.order_types:
            if order_type not in STRATEGIES:
                if verbose:
                    print(f"[WARN] Unknown order type '{order_type}' — skipped.", file=sys.stderr)
                continue
            strategy = STRATEGIES[order_type]()
            if verbose:
                print(f"  Evaluando {order_type.upper()}...", end="  ", flush=True)
            result = strategy.evaluate(
                tech=mr["tech"],
                lambda_matrix=mr["lambda_matrix"],
                avail_matrix=mr["avail_matrix"],
                probs=mr["probs"],
                grid=mr["grid"],
                objective=cfg.objective,
                cvar_alpha=cfg.cvar_alpha,
                startup_per_transition=cfg.startup_per_transition,
                sco_model=cfg.sco_model,
            )
            result["objective_value"] = objective_value(
                result["profits"], mr["probs"], cfg.cvar_alpha, cfg.objective
            )
            result["objective_value_per_mw"] = (
                result["objective_value"] / tech.installed_capacity_mw
            )
            results.append(result)
            if verbose:
                print(
                    f"E[Π] = {result['expected_profit']:>10,.0f} €   "
                    f"CVaR = {result['cvar']:>10,.0f} €   "
                    f"P_match = {result['match_probability']:.1%}   "
                    f"E[Π]/MW = {result['expected_profit_per_mw']:>8,.1f} €/MW"
                )

        monthly.append(
            {**mr, "results": results, "ranking": build_ranking(results) if results else None}
        )

    if not monthly or not monthly[0]["results"]:
        return monthly, None

    aggregate = build_aggregate_ranking(
        monthly, cfg.cvar_alpha, cfg.objective, tech.installed_capacity_mw
    )
    return monthly, aggregate


def ranking_by_month(monthly: list[dict]):
    """Concatenate the per-month rankings, tagged with month/n_days/variable_cost."""
    import pandas as pd

    frames = []
    for m in monthly:
        if m["ranking"] is None:
            continue
        tagged = m["ranking"].copy()
        tagged.insert(0, "variable_cost", round(m["variable_cost"], 2))
        tagged.insert(0, "n_days", m["n_days"])
        tagged.insert(0, "month", m["month"])
        frames.append(tagged)
    return pd.concat(frames) if frames else None


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

    # ---------------------------------------------------------------- header
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Tecnología : {tech.name}")
    print(f"  Modo       : {cfg.mode}  |  Resolución : {cfg.resolution} MTU")
    print(f"  Tipos de orden : {', '.join(cfg.order_types)}  (una optimización por mes)")
    if "sco" in cfg.order_types:
        print(f"  Modelo SCO : {cfg.sco_model}")
    if cfg.startup_per_transition:
        print("  Arranques  : uno por cada transición cero→producción")
    if tech.variable_cost_source == "monthly_price_mean":
        print("  Coste var. : dinámico — valor del agua = precio medio mensual del mercado")
    print(sep)

    # ------------------------------------------- evaluate (one run per month)
    try:
        monthly, aggregate = evaluate_technology(tech, cfg, verbose=True)
    except ValueError as exc:
        if not exit_on_error:
            raise
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    if aggregate is None:
        print("[ERROR] No valid order types evaluated.", file=sys.stderr)
        sys.exit(1)

    n_days_total = sum(m["n_days"] for m in monthly)
    print(f"\n{sep}")
    print(f"  RANKING AGREGADO (por valor objetivo — {len(monthly)} mes(es), {n_days_total} días)")
    print(sep)
    display_cols = ["order_type", "objective_value", "expected_profit", "cvar",
                    "match_probability", "expected_matched_energy",
                    "expected_matched_periods", "expected_profit_per_mw"]
    print(aggregate[display_cols].to_string())
    print()

    # ------------------------------------------------------------ output
    import pandas as pd

    output_dir = tech_output_dir(cfg, tech.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    for m in monthly:
        month_dir = output_dir / m["month"]
        month_dir.mkdir(parents=True, exist_ok=True)
        if m["ranking"] is not None:
            m["ranking"].to_csv(month_dir / "ranking.csv")
        try:
            plot_all(m["results"], m["lambda_matrix"], month_dir,
                     tech_name=f"{tech.name} — {m['month']}",
                     capacity_mw=tech.installed_capacity_mw)
        except Exception as exc:
            print(f"  [WARN] Gráficas de {m['month']} no generadas: {exc}", file=sys.stderr)

    # Figuras agregadas a nivel de tecnología: mismos 4 plots que por mes, con
    # los resultados de todos los meses concatenados y los parámetros óptimos
    # promediados entre meses.
    try:
        agg_results = aggregate_results(
            monthly, cfg.cvar_alpha, cfg.objective, tech.installed_capacity_mw
        )
        plot_results = [{**r, "optimal_params": r["mean_params"]} for r in agg_results]
        lambda_full = np.concatenate([m["lambda_matrix"] for m in monthly], axis=0)
        plot_all(plot_results, lambda_full, output_dir,
                 tech_name=f"{tech.name} — media de {len(monthly)} mes(es)",
                 capacity_mw=tech.installed_capacity_mw)
    except Exception as exc:
        print(f"  [WARN] Gráficas agregadas no generadas: {exc}", file=sys.stderr)

    by_month = ranking_by_month(monthly)
    if by_month is not None:
        by_month_path = output_dir / "ranking_by_month.csv"
        by_month.to_csv(by_month_path)
        print(f"  Ranking mensual en  : {by_month_path}")

    ranking_path = output_dir / "ranking.csv"
    aggregate.to_csv(ranking_path)
    print(f"  Ranking agregado en : {ranking_path}")
    print(f"  Resultados por mes  : {output_dir}\\<YYYY-MM>\\")

    # ------------------------------------------------------- risk frontier (optional)
    if cfg.beta_sweep:
        print(f"\n  Calculando frontera beneficio-riesgo (beta = {cfg.beta_sweep})...")
        frontier_frames_all = []
        for m in monthly:
            frontier_frames = [
                sweep_beta(
                    m["tech"], m["lambda_matrix"], m["avail_matrix"], m["probs"],
                    m["grid"], cfg.cvar_alpha, ot, cfg.beta_sweep,
                    startup_per_transition=cfg.startup_per_transition,
                    sco_model=cfg.sco_model,
                )
                for ot in cfg.order_types
                if ot in STRATEGIES
            ]
            frontier_df = pd.concat(frontier_frames, ignore_index=True)
            month_dir = output_dir / m["month"]
            month_dir.mkdir(parents=True, exist_ok=True)
            frontier_df.to_csv(month_dir / "frontier.csv", index=False)
            try:
                plot_risk_frontier(frontier_df, month_dir,
                                   tech_name=f"{tech.name} — {m['month']}")
            except Exception as exc:
                print(f"  [WARN] Gráfica de frontera de {m['month']} no generada: {exc}",
                      file=sys.stderr)
            frontier_df.insert(0, "month", m["month"])
            frontier_frames_all.append(frontier_df)
        frontier_path = output_dir / "frontier_by_month.csv"
        pd.concat(frontier_frames_all, ignore_index=True).to_csv(frontier_path, index=False)
        print(f"  Frontera guardada en : {frontier_path} (+ frontier.csv por mes)")

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

    summary_p = sub.add_parser(
        "summary",
        help="Grafica resumen (todas las tecnologias): mejor valor objetivo "
             "por mes en €/MW, a partir de los ranking_by_month.csv ya escritos",
    )
    summary_p.add_argument("--run", required=True, metavar="YAML",
                           help="Fichero YAML de ejecucion (determina results/<season>/)")

    beta_p = sub.add_parser(
        "beta-ranking",
        help="Barrido de beta: ranking anual por tipo de orden en funcion de la "
             "aversion al riesgo (re-optimiza theta* para cada beta)",
    )
    beta_p.add_argument("--run", required=True, metavar="YAML",
                        help="Fichero YAML de ejecucion")
    beta_p.add_argument("--tech", default="all", metavar="YAML|all",
                        help="Fichero YAML de tecnologia, o 'all' (por defecto)")
    beta_p.add_argument("--yaml-dir", default="yaml",
                        help="Directorio con los YAML de tecnologia para --tech all")
    beta_p.add_argument("--betas", default=None,
                        help="Lista de betas separadas por comas (por defecto: 0.0,0.1,...,1.0)")

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
        if args.tech == "all":
            from reporting.summary import run_summary
            for run_path in run_paths:
                run_summary(str(run_path))
    elif args.command == "family":
        from reporting.family import run_family, run_family_seasons
        if args.run:
            run_family(args.family_num, args.run, yaml_dir=args.yaml_dir)
        else:
            if not args.run_invierno:
                parser.error("--run-invierno es requerido junto a --run-verano")
            run_family_seasons(
                args.family_num, args.run_verano, args.run_invierno, yaml_dir=args.yaml_dir
            )
    elif args.command == "validate-oos":
        from reporting.validation import run_validate_oos
        run_validate_oos(
            args.tech, args.run,
            train_fraction=args.train_fraction,
            sco_model_override=args.sco_model,
        )
    elif args.command == "summary":
        from reporting.summary import run_summary
        _ensure_utf8_stdout()
        run_summary(args.run)
    elif args.command == "beta-ranking":
        from reporting.beta_ranking import run_beta_ranking
        _ensure_utf8_stdout()
        betas = [float(b) for b in args.betas.split(",")] if args.betas else None
        run_beta_ranking(args.run, tech=args.tech, yaml_dir=args.yaml_dir, betas=betas)
    elif args.command == "compare-seasons":
        from reporting.seasons import run_compare_seasons
        run_compare_seasons(args.tech, args.run_verano, args.run_invierno)
