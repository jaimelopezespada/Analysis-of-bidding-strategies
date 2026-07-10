"""Scalar metrics: E[Π], CVaR, match probability, expected matched energy."""

from __future__ import annotations

import numpy as np

from .config import RiskObjective


def expected_profit(profits: np.ndarray, probs: np.ndarray) -> float:
    """E[Π] = Σ_s π_s · Π^s"""
    return float(probs @ profits)


def cvar(profits: np.ndarray, probs: np.ndarray, alpha: float) -> float:
    """
    CVaR_α: expected profit in the worst (1-alpha) fraction of scenarios.

    Uses the Rockafellar-Uryasev formula (maximisation form):
        CVaR_α = η − (1/(1−α)) · Σ_s π_s · max(η − Π^s, 0)
    where η = VaR_α is the (1-alpha)-quantile of the profit distribution.
    """
    if alpha >= 1.0:
        return float(np.min(profits))

    idx = np.argsort(profits)
    sorted_p = profits[idx]
    sorted_w = probs[idx]
    cumw = np.cumsum(sorted_w)

    # VaR_α: smallest profit such that P(Π ≤ η) ≥ 1−α
    var_idx = int(np.searchsorted(cumw, 1.0 - alpha))
    var_idx = min(var_idx, len(sorted_p) - 1)
    eta = sorted_p[var_idx]

    z = np.maximum(eta - profits, 0.0)
    return float(eta - (1.0 / (1.0 - alpha)) * float(probs @ z))


def match_probability(matched: np.ndarray, probs: np.ndarray) -> float:
    """P(order matched) = Σ_s π_s · 1[order accepted in s]"""
    return float(probs @ matched)


def expected_matched_energy(dispatch: np.ndarray, probs: np.ndarray) -> float:
    """Expected total energy matched per day (MWh/day)."""
    return float(probs @ dispatch.sum(axis=1))


def profit_per_mw(expected_profit_value: float, capacity_mw: float) -> float:
    """E[Π] normalised by installed capacity (€/MW/day).

    Lets technologies of very different sizes (e.g. a 400 MW CCGT vs a
    50 MW battery) be compared on equal footing rather than by absolute
    €/day, which mechanically favours larger plants.
    """
    if capacity_mw <= 0:
        return float("nan")
    return expected_profit_value / capacity_mw


def compute_metrics(
    profits: np.ndarray,
    probs: np.ndarray,
    matched: np.ndarray,
    dispatch: np.ndarray,
    alpha: float,
    capacity_mw: float,
) -> dict:
    ep = expected_profit(profits, probs)
    return {
        "expected_profit": ep,
        "cvar": cvar(profits, probs, alpha),
        "match_probability": match_probability(matched, probs),
        "expected_matched_energy": expected_matched_energy(dispatch, probs),
        "expected_profit_per_mw": profit_per_mw(ep, capacity_mw),
    }


def objective_value(
    profits: np.ndarray,
    probs: np.ndarray,
    alpha: float,
    objective: RiskObjective,
) -> float:
    """max (1-beta)*E[Pi] + beta*CVaR_alpha[Pi].

    Match probability is a reported metric only (see compute_metrics), not
    part of the objective — mixing euros with a dimensionless probability
    would conflate magnitudes of different nature.
    """
    ep = expected_profit(profits, probs)
    cv = cvar(profits, probs, alpha)
    beta = objective.beta
    return (1.0 - beta) * ep + beta * cv
