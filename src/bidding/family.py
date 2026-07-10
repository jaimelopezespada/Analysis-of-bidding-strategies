"""Group technologies by family and produce an aggregate comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml as pyyaml

from .cli import _ensure_utf8_stdout, evaluate_technology
from .config import RunConfig, TechnologyConfig
from .plots import plot_family_comparison, plot_family_season_comparison
from .seasons import _season_label


def discover_family_techs(family_num: int, yaml_dir: str = "yaml") -> list[Path]:
    """Scan yaml_dir for technology YAMLs whose `family:` field matches, skipping
    run/execution configs (which have no `family` key) by construction."""
    matches = []
    for path in sorted(Path(yaml_dir).glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            data = pyyaml.safe_load(f)
        if isinstance(data, dict) and data.get("family") == family_num:
            matches.append(path)
    return matches


def run_family(family_num: str, run_path: str, yaml_dir: str = "yaml") -> None:
    _ensure_utf8_stdout()
    cfg = RunConfig.from_yaml(run_path)
    np.random.seed(cfg.seed)

    if family_num == "all":
        family_nums = [1, 2, 3, 4]
    else:
        family_nums = [int(family_num)]

    for fam in family_nums:
        tech_paths = discover_family_techs(fam, yaml_dir)
        if not tech_paths:
            print(f"[WARN] No technology YAMLs found for family {fam} in {yaml_dir}/", file=sys.stderr)
            continue

        sep = "=" * 62
        print(f"\n{sep}\n  Familia {fam} — {len(tech_paths)} tecnologia(s)\n{sep}")

        per_tech_rankings: dict[str, pd.DataFrame] = {}
        combined_rows = []
        for tech_path in tech_paths:
            tech = TechnologyConfig.from_yaml(tech_path)
            print(f"\n  Evaluando {tech.name}...")
            results, ranking, _ = evaluate_technology(tech, cfg)
            if ranking is None:
                print(f"  [WARN] Sin resultados para {tech.name} — omitida.", file=sys.stderr)
                continue
            per_tech_rankings[tech.name] = ranking
            tagged = ranking.copy()
            tagged.insert(0, "technology", tech.name)
            combined_rows.append(tagged)

        if not combined_rows:
            continue

        comparison = pd.concat(combined_rows, ignore_index=True)
        output_dir = Path(cfg.output_dir) / f"family_{fam}"
        output_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = output_dir / "comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"\n  Comparativa guardada en : {comparison_path}")

        try:
            plot_family_comparison(per_tech_rankings, f"Familia {fam}", output_dir)
            print(f"  Grafica en             : {output_dir / 'figs' / 'family_comparison.png'}")
        except Exception as exc:
            print(f"  [WARN] Grafica de familia no generada: {exc}", file=sys.stderr)


def run_family_seasons(
    family_num: str, run_verano_path: str, run_invierno_path: str, yaml_dir: str = "yaml"
) -> None:
    """Like run_family, but evaluates every technology of the family against
    both a verano and an invierno run config, tagging results with a
    `season` column so the two can be compared directly."""
    _ensure_utf8_stdout()

    if family_num == "all":
        family_nums = [1, 2, 3, 4]
    else:
        family_nums = [int(family_num)]

    for fam in family_nums:
        tech_paths = discover_family_techs(fam, yaml_dir)
        if not tech_paths:
            print(f"[WARN] No technology YAMLs found for family {fam} in {yaml_dir}/", file=sys.stderr)
            continue

        sep = "=" * 62
        print(f"\n{sep}\n  Familia {fam} (verano vs invierno) — {len(tech_paths)} tecnologia(s)\n{sep}")

        combined_rows = []
        for tech_path in tech_paths:
            tech = TechnologyConfig.from_yaml(tech_path)
            print(f"\n  Evaluando {tech.name}...")
            for run_path in (run_verano_path, run_invierno_path):
                cfg = RunConfig.from_yaml(run_path)
                np.random.seed(cfg.seed)
                _, ranking, _ = evaluate_technology(tech, cfg)
                if ranking is None:
                    print(f"  [WARN] Sin resultados para {tech.name} ({run_path}) — omitida.", file=sys.stderr)
                    continue
                tagged = ranking.copy()
                tagged.insert(0, "season", _season_label(cfg, run_path))
                tagged.insert(0, "technology", tech.name)
                combined_rows.append(tagged)

        if not combined_rows:
            continue

        comparison = pd.concat(combined_rows, ignore_index=True)
        output_dir = Path("results") / f"family_{fam}_estacional"
        output_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = output_dir / "comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"\n  Comparativa guardada en : {comparison_path}")

        try:
            plot_family_season_comparison(comparison, f"Familia {fam}", output_dir)
            print(f"  Grafica en             : {output_dir / 'figs' / 'family_season_comparison.png'}")
        except Exception as exc:
            print(f"  [WARN] Grafica de familia no generada: {exc}", file=sys.stderr)
