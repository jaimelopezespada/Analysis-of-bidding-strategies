"""Assemble per-order-type results into a ranked comparison table."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import RiskObjective
from .metrics import compute_metrics, objective_value


def build_ranking(results: list[dict]) -> pd.DataFrame:
    """
    Build the ranking DataFrame from a list of evaluate() result dicts.
    Sorted by objective_value descending (the criterion each strategy actually
    optimized); rank index starts at 1.
    """
    rows = []
    for r in results:
        obj = r.get("objective_value", r["expected_profit"])
        obj_per_mw = r.get("objective_value_per_mw")
        if obj_per_mw is None:
            # expected_profit_per_mw / expected_profit == 1 / capacity_mw
            ep = r["expected_profit"]
            obj_per_mw = obj * r["expected_profit_per_mw"] / ep if ep else 0.0
        rows.append(
            {
                "order_type": r["order_type"],
                "expected_profit": round(r["expected_profit"], 2),
                "cvar": round(r["cvar"], 2),
                "match_probability": round(r["match_probability"], 4),
                "expected_matched_energy": round(r["expected_matched_energy"], 2),
                "expected_matched_periods": round(r["expected_matched_periods"], 2),
                "expected_profit_per_mw": round(r["expected_profit_per_mw"], 2),
                "objective_value": round(obj, 2),
                "objective_value_per_mw": round(obj_per_mw, 2),
                "optimal_params": json.dumps(r["optimal_params"], ensure_ascii=False),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("objective_value", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # rank 1, 2, 3
    df.index.name = "rank"
    return df


def _mean_params(params_by_month: dict[str, dict]) -> dict:
    """Month-averaged optimal params, for the aggregate offer-vs-price plot.

    Numeric scalars and numeric lists (e.g. the simple order's price_profile)
    are averaged element-wise across months; anything non-numeric (block ids,
    group ids) keeps the first month's value.
    """
    all_params = list(params_by_month.values())
    out = {}
    for key, val in all_params[0].items():
        vals = [p[key] for p in all_params if key in p]
        if isinstance(val, bool):
            out[key] = val
        elif isinstance(val, (int, float)):
            out[key] = float(np.mean(vals))
        elif isinstance(val, list) and val and isinstance(val[0], (int, float)):
            out[key] = np.mean(np.asarray(vals, dtype=float), axis=0).tolist()
        else:
            out[key] = val
    return out


def aggregate_results(
    monthly: list[dict],
    cvar_alpha: float,
    objective: RiskObjective,
    capacity_mw: float,
) -> list[dict]:
    """Aggregate the per-month evaluations into one result dict per order type.

    Each month was optimised separately (its own theta*, and for hydro its own
    water value), so the aggregate measures the full-period performance of the
    monthly-adaptive strategy: per order type, the per-scenario profit /
    matched / dispatch vectors of every month are concatenated and re-scored
    with day-count weights ``probs_m · (n_days_m / N_total)`` — uniform over
    all days in stochastic mode, and the correct day weighting of each month's
    single mean row in deterministic mode. ``optimal_params`` becomes a
    {month: params} map; ``mean_params`` averages them across months for the
    aggregate figures.

    ``monthly`` items need keys: month, n_days, results (list of evaluate()
    result dicts).
    """
    n_total = sum(m["n_days"] for m in monthly)
    order_types = [r["order_type"] for r in monthly[0]["results"]]

    results = []
    for order_type in order_types:
        profits, matched, dispatch, weights = [], [], [], []
        params_by_month = {}
        for m in monthly:
            r = next(r for r in m["results"] if r["order_type"] == order_type)
            s_m = len(r["profits"])
            profits.append(np.asarray(r["profits"], dtype=float))
            matched.append(np.asarray(r["matched"], dtype=float))
            dispatch.append(np.asarray(r["dispatch"], dtype=float))
            weights.append(np.full(s_m, (1.0 / s_m) * (m["n_days"] / n_total)))
            params_by_month[m["month"]] = r["optimal_params"]

        profits = np.concatenate(profits)
        matched = np.concatenate(matched)
        dispatch = np.concatenate(dispatch)
        weights = np.concatenate(weights)

        metrics = compute_metrics(profits, weights, matched, dispatch, cvar_alpha, capacity_mw)
        obj = objective_value(profits, weights, cvar_alpha, objective)
        results.append(
            {
                "order_type": order_type,
                "optimal_params": params_by_month,
                "mean_params": _mean_params(params_by_month),
                "profits": profits,
                "matched": matched,
                "dispatch": dispatch,
                "weights": weights,
                "objective_value": obj,
                "objective_value_per_mw": obj / capacity_mw,
                **metrics,
            }
        )

    return results


def build_aggregate_ranking(
    monthly: list[dict],
    cvar_alpha: float,
    objective: RiskObjective,
    capacity_mw: float,
) -> pd.DataFrame:
    """Full-period ranking over the aggregate_results() dicts."""
    return build_ranking(aggregate_results(monthly, cvar_alpha, objective, capacity_mw))
