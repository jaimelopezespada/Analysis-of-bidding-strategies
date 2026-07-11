"""Out-of-sample (train/test) validation of the SCO strategy.

The grid search picks theta* using the very scenarios it is later scored on,
so the reported E[Pi]/CVaR carry an in-sample selection bias. This module
quantifies it: theta* is optimised on the chronologically earliest
``train_fraction`` of scenario days and then applied MECHANICALLY (same
declared acceptance rule, no re-optimisation) to the held-out later days —
what would actually happen if the agent submitted that fixed SCO in a real
deployment.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RunConfig, TechnologyConfig
from .metrics import compute_metrics
from .orders.sco import SCOStrategy, evaluate_sco_theta


def split_scenarios_chronological(
    S: int, train_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays (train, test): the first ceil(train_fraction*S) days are
    train, the remaining (later) days are test. Chronological rather than
    random — the realistic setting is optimising on past days and deploying
    on future ones. Requires at least one day on each side."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}.")
    n_train = math.ceil(S * train_fraction)
    n_train = min(n_train, S - 1)
    if n_train < 1:
        raise ValueError(f"Not enough scenarios (S={S}) to split train/test.")
    idx = np.arange(S)
    return idx[:n_train], idx[n_train:]


def validate_sco_oos(
    tech: TechnologyConfig,
    cfg: RunConfig,
    train_fraction: float = 0.7,
) -> dict:
    """Optimise theta* on the train days, re-evaluate it on the test days.

    Returns a dict with theta*, the scenario-day ranges, and per-split
    metrics (in-sample train vs out-of-sample test).
    """
    from .cli import load_scenarios  # deferred: cli imports this module's caller chain

    lambda_matrix, avail_matrix, probs_all, labels = load_scenarios(tech, cfg)
    S = lambda_matrix.shape[0]
    train_idx, test_idx = split_scenarios_chronological(S, train_fraction)

    splits: dict[str, np.ndarray] = {"train": train_idx, "test": test_idx}
    grid = tech.effective_grid(cfg.candidate_grid)
    strategy = SCOStrategy()

    # 1) Optimise theta* on train only (equiprobable days within the split,
    #    consistent with load_scenarios' uniform probabilities).
    probs_train = np.ones(len(train_idx)) / len(train_idx)
    train_result = strategy.evaluate(
        tech=tech,
        lambda_matrix=lambda_matrix[train_idx],
        avail_matrix=avail_matrix[train_idx],
        probs=probs_train,
        grid=grid,
        objective=cfg.objective,
        cvar_alpha=cfg.cvar_alpha,
        sco_model=cfg.sco_model,
    )
    theta = {
        k: train_result["optimal_params"][k]
        for k in ("price_variable", "fixed_term", "mav_fraction")
    }

    # 2) Apply theta* mechanically to each split (test = out-of-sample).
    rows = []
    for split_name, idx in splits.items():
        probs = np.ones(len(idx)) / len(idx)
        q, u, profits_s = evaluate_sco_theta(
            tech, lambda_matrix[idx], avail_matrix[idx], probs, theta,
            cfg.cvar_alpha, cfg.objective, sco_model=cfg.sco_model,
        )
        m = compute_metrics(profits_s, probs, u, q, cfg.cvar_alpha, tech.installed_capacity_mw)
        rows.append({
            "split": split_name,
            "n_days": len(idx),
            "date_from": labels[idx[0]],
            "date_to": labels[idx[-1]],
            **m,
        })

    return {
        "theta": theta,
        "sco_model": cfg.sco_model,
        "train_fraction": train_fraction,
        "metrics": pd.DataFrame(rows),
    }


def run_validate_oos(
    tech_path: str,
    run_path: str,
    train_fraction: float = 0.7,
    sco_model_override: str | None = None,
) -> None:
    """CLI entry point: print the train/test comparison and save it as CSV."""
    from .cli import _ensure_utf8_stdout, tech_output_dir

    _ensure_utf8_stdout()
    tech = TechnologyConfig.from_yaml(tech_path)
    cfg = RunConfig.from_yaml(run_path)
    if sco_model_override:
        cfg = cfg.model_copy(update={"sco_model": sco_model_override})
    np.random.seed(cfg.seed)

    try:
        result = validate_sco_oos(tech, cfg, train_fraction=train_fraction)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Validación out-of-sample SCO — {tech.name}")
    print(f"  Modelo SCO : {cfg.sco_model}  |  train_fraction = {train_fraction}")
    print(f"  θ* (train) : {result['theta']}")
    print(sep)
    metrics: pd.DataFrame = result["metrics"]
    print(metrics.to_string(index=False))
    print()

    output_dir = tech_output_dir(cfg, tech.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"oos_sco_{cfg.sco_model}.csv"
    tagged = metrics.copy()
    for key, value in result["theta"].items():
        tagged[key] = value
    tagged["sco_model"] = cfg.sco_model
    tagged.to_csv(out_path, index=False)
    print(f"  Comparativa guardada en : {out_path}")
    print(f"{sep}\n")
