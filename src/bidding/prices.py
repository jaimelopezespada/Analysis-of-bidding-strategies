"""Load and validate OMIE-style CSVs (price, renewable generation) into scenario matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PricesConfig


def _load_daily_matrix(
    csv_path: str,
    value_col: str,
    resolution: int,
    date_range: tuple[str, str] | None,
) -> tuple[np.ndarray, list[str]]:
    """
    Parse a CSV with columns (date, period, <value_col>, ...extra columns ok)
    into a (S, T) matrix, one row per calendar day, sorted by date.

    Returns
    -------
    matrix : np.ndarray, shape (S, T)
    scenario_labels : list[str]
        Date string ("YYYY-MM-DD") for each scenario row, length S.
    """
    df = pd.read_csv(csv_path)

    missing = {"date", "period", value_col} - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path!r} is missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["period"] = df["period"].astype(int)
    df[value_col] = df[value_col].astype(float)

    if date_range is not None:
        start = pd.to_datetime(date_range[0])
        end = pd.to_datetime(date_range[1])
        df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
        if df.empty:
            raise ValueError(
                f"No rows in CSV {csv_path!r} after filtering date_range {date_range}."
            )

    # Drop exact duplicate (date, period) rows — a known artefact of chunked
    # monthly downloads at chunk boundaries (see data/generacion_renovable.py).
    # Only silently dedupe when the duplicated rows agree on value_col; genuine
    # conflicting duplicates are a real data-quality error and must fail loudly.
    dup_mask = df.duplicated(subset=["date", "period"], keep=False)
    if dup_mask.any():
        dup_groups = df[dup_mask].groupby(["date", "period"])[value_col].nunique()
        conflicting = dup_groups[dup_groups > 1]
        if len(conflicting):
            raise ValueError(
                f"{len(conflicting)} (date, period) pair(s) in {csv_path!r} have "
                f"duplicated rows with conflicting {value_col!r} values: "
                f"{conflicting.index[:3].tolist()}"
            )
        df = df.drop_duplicates(subset=["date", "period"], keep="first")

    # Validate periods per day
    counts = df.groupby("date")["period"].count()
    bad_days = counts[counts != resolution]
    if len(bad_days):
        raise ValueError(
            f"{len(bad_days)} day(s) in {csv_path!r} have wrong period count "
            f"(expected {resolution}). First offenders: {bad_days.index[:3].tolist()}"
        )

    pivot = df.pivot(index="date", columns="period", values=value_col).sort_index()
    matrix = pivot.values.astype(float)  # (S, T)
    scenario_labels = [str(d.date()) for d in pivot.index]

    return matrix, scenario_labels


def load_price_matrix(
    cfg: PricesConfig, resolution: int
) -> tuple[np.ndarray, list[str]]:
    """
    Parse an OMIE-style CSV (columns: date, period, price) into a scenario matrix.

    Returns
    -------
    lambda_matrix : np.ndarray, shape (S, T)
        Market clearing prices λ_t^s in €/MWh.
    scenario_labels : list[str]
        Date string for each scenario row, length S.
    """
    return _load_daily_matrix(cfg.csv_path, "price", resolution, cfg.date_range)
