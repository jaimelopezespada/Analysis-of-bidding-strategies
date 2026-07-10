"""Assemble per-order-type results into a ranked comparison table."""

from __future__ import annotations

import json

import pandas as pd


def build_ranking(results: list[dict]) -> pd.DataFrame:
    """
    Build the ranking DataFrame from a list of evaluate() result dicts.
    Sorted by objective_value descending (the criterion each strategy actually
    optimized); rank index starts at 1.
    """
    rows = []
    for r in results:
        rows.append(
            {
                "order_type": r["order_type"],
                "expected_profit": round(r["expected_profit"], 2),
                "cvar": round(r["cvar"], 2),
                "match_probability": round(r["match_probability"], 4),
                "expected_matched_energy": round(r["expected_matched_energy"], 2),
                "expected_profit_per_mw": round(r["expected_profit_per_mw"], 2),
                "objective_value": round(r.get("objective_value", r["expected_profit"]), 2),
                "optimal_params": json.dumps(r["optimal_params"], ensure_ascii=False),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("objective_value", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # rank 1, 2, 3
    df.index.name = "rank"
    return df
