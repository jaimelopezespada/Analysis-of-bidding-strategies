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

from bidding.config import RunConfig, TechnologyConfig
from bidding.metrics import compute_metrics
from bidding.monthly import monthly_mean_price, resolve_tech_for_month, split_by_month
from bidding.orders.sco import SCOStrategy, evaluate_sco_theta


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
    """Optimise theta* per month on that month's train days, re-evaluate on
    its test days.

    Since the run partitions scenarios by calendar month (each month has its
    own theta*, and hydro its own water value), the train/test split is
    chronological WITHIN each month. Months with fewer than 2 days are
    skipped with a warning. Returns a dict with {month: theta*} and one
    metrics row per (month, split).
    """
    from bidding.cli import load_scenarios  # deferred: cli imports this module's caller chain

    lambda_matrix, avail_matrix, _, labels = load_scenarios(tech, cfg)
    strategy = SCOStrategy()

    thetas: dict[str, dict] = {}
    rows = []
    for month, idx in split_by_month(labels):
        if len(idx) < 2:
            print(
                f"[WARN] {month}: solo {len(idx)} día(s) — no hay división train/test posible, "
                "mes omitido.",
                file=sys.stderr,
            )
            continue

        month_mean = monthly_mean_price(lambda_matrix, idx)
        tech_m = resolve_tech_for_month(tech, month, month_mean)
        grid = tech_m.resolved_grid(cfg.candidate_grid)
        lam_m = lambda_matrix[idx]
        avail_m = avail_matrix[idx]
        labels_m = [labels[i] for i in idx]
        train_idx, test_idx = split_scenarios_chronological(len(idx), train_fraction)

        # 1) Optimise theta* on the month's train days only (equiprobable).
        probs_train = np.ones(len(train_idx)) / len(train_idx)
        train_result = strategy.evaluate(
            tech=tech_m,
            lambda_matrix=lam_m[train_idx],
            avail_matrix=avail_m[train_idx],
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
        thetas[month] = theta

        # 2) Apply theta* mechanically to each split (test = out-of-sample).
        for split_name, split_idx in (("train", train_idx), ("test", test_idx)):
            probs = np.ones(len(split_idx)) / len(split_idx)
            q, u, profits_s = evaluate_sco_theta(
                tech_m, lam_m[split_idx], avail_m[split_idx], probs, theta,
                cfg.cvar_alpha, cfg.objective, sco_model=cfg.sco_model,
            )
            m = compute_metrics(profits_s, probs, u, q, cfg.cvar_alpha,
                                tech_m.installed_capacity_mw)
            rows.append({
                "month": month,
                "split": split_name,
                "n_days": len(split_idx),
                "date_from": labels_m[split_idx[0]],
                "date_to": labels_m[split_idx[-1]],
                **theta,
                **m,
            })

    if not rows:
        raise ValueError("Ningún mes tiene suficientes días para la validación train/test.")

    return {
        "theta": thetas,
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
    from bidding.cli import _ensure_utf8_stdout, tech_output_dir

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
    print(f"  Validación out-of-sample SCO — {tech.name} (por mes)")
    print(f"  Modelo SCO : {cfg.sco_model}  |  train_fraction = {train_fraction}")
    for month, theta in result["theta"].items():
        print(f"  θ* {month} : {theta}")
    print(sep)
    metrics: pd.DataFrame = result["metrics"]
    print(metrics.to_string(index=False))
    print()

    output_dir = tech_output_dir(cfg, tech.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"oos_sco_{cfg.sco_model}.csv"
    tagged = metrics.copy()
    tagged["sco_model"] = cfg.sco_model
    tagged.to_csv(out_path, index=False)
    print(f"  Comparativa guardada en : {out_path}")
    print(f"{sep}\n")
