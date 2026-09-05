"""Partition scenarios by calendar month and resolve month-dependent costs.

The optimizer picks ONE bid curve per run, so a cost that changes with the
month (the hydro water value: variable_cost_source="monthly_price_mean")
requires one optimization per calendar month. The CLI partitions the scenario
days with split_by_month and, for each month, resolves the technology into an
ordinary static-cost copy via resolve_tech_for_month before any strategy runs
— strategies never see an unresolved dynamic cost.

The monthly water value is the mean of ALL hourly prices of that month's
scenario days. Computing it from the run's own data introduces intra-month
perfect foresight — a documented limitation (hydro producers forecast the
monthly price level well); the OMIE 2025 monthly-mean table in the thesis
serves as a cross-check of the resolved values.
"""

from __future__ import annotations

import numpy as np

from .config import TechnologyConfig


def split_by_month(scenario_labels: list[str]) -> list[tuple[str, np.ndarray]]:
    """Ordered (month "YYYY-MM", scenario index array) pairs.

    Labels are "YYYY-MM-DD" strings (prices.py), so the month key is the
    first 7 characters. Order follows first appearance, which is
    chronological because the price matrix rows are date-sorted.
    """
    months: dict[str, list[int]] = {}
    for i, label in enumerate(scenario_labels):
        months.setdefault(label[:7], []).append(i)
    return [(month, np.asarray(idx, dtype=int)) for month, idx in months.items()]


def monthly_mean_price(lambda_matrix: np.ndarray, idx: np.ndarray) -> float:
    """Mean over all hourly prices of the month's scenario days."""
    return float(lambda_matrix[idx].mean())


def resolve_tech_for_month(
    tech: TechnologyConfig, month: str, month_mean: float
) -> TechnologyConfig:
    """Return a technology with any month-dependent cost resolved to a scalar.

    Static-cost technologies pass through unchanged. A
    "monthly_price_mean" technology becomes a static copy whose
    variable_cost — and therefore price_reference_value, so every
    price_levels_pct candidate — is the month's mean market price.
    """
    if tech.variable_cost_source != "monthly_price_mean":
        return tech
    if month_mean <= 0:
        raise ValueError(
            f"{tech.name}: mean market price for {month} is {month_mean:.2f} €/MWh "
            "(<= 0) — cannot be used as the water-value proxy: price_levels_pct "
            "would collapse or flip sign."
        )
    return tech.model_copy(
        update={"variable_cost": month_mean, "variable_cost_source": "static"}
    )
