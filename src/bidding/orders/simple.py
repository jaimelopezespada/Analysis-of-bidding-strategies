"""Simple order strategy: per-period independent price optimisation."""

from __future__ import annotations

import numpy as np

from ..config import CandidateGrid, ObjectiveWeights, TechnologyConfig
from ..metrics import compute_metrics, objective_value
from .base import OrderStrategy


class SimpleOrderStrategy(OrderStrategy):
    """
    Each period t is matched independently: q_t^s = Q̂_t if P^V_t ≤ λ_t^s, else 0.

    Optimisation: for each period we enumerate K candidate prices and pick the one
    that maximises the weighted objective.  Periods are separable for E[Π] (default
    weights); with CVaR weight > 0 the solution is a tractable approximation.
    """

    order_type = "simple"

    def evaluate(
        self,
        tech: TechnologyConfig,
        lambda_matrix: np.ndarray,
        probs: np.ndarray,
        grid: CandidateGrid,
        weights: ObjectiveWeights,
        cvar_alpha: float,
    ) -> dict:
        S, T = lambda_matrix.shape
        avail = np.asarray(tech.availability.values, dtype=float)  # (T,)
        cost = np.asarray(tech.cost_array(T), dtype=float)         # (T,)
        price_levels = np.asarray(grid.price_levels, dtype=float)  # (K,)

        # accept[k, s, t] = 1 if price_levels[k] <= lambda_matrix[s, t]
        # Broadcasting: (K, 1, 1) <= (1, S, T)
        accept = price_levels[:, None, None] <= lambda_matrix[None, :, :]  # (K, S, T)

        # margin[s, t] = λ_t^s − C_t  (net margin per MWh)
        margin = lambda_matrix - cost[None, :]  # (S, T)

        # per-period expected profit for each price level
        # profit[k, s, t] = margin[s, t] * avail[t] * accept[k, s, t]
        profit_kst = margin[None, :, :] * avail[None, None, :] * accept  # (K, S, T)
        exp_profit_kt = np.einsum("s,kst->kt", probs, profit_kst)         # (K, T)

        # Best price level per period (argmax over E[Π_t])
        best_k = np.argmax(exp_profit_kt, axis=0)   # (T,)
        optimal_prices = price_levels[best_k]        # (T,)

        # Dispatch with optimal per-period prices
        q = np.where(optimal_prices[None, :] <= lambda_matrix, avail[None, :], 0.0)  # (S, T)

        # Battery energy cap: concentrate discharge in highest-price periods
        if tech.energy_capacity is not None:
            q = _apply_energy_cap(q, lambda_matrix, avail, tech.energy_capacity)

        # Per-scenario profit
        profits_s = np.sum(margin * q, axis=1)  # (S,)

        # Startup cost when any energy is dispatched
        if tech.startup_cost > 0.0:
            started = (q.sum(axis=1) > 0).astype(float)
            profits_s -= tech.startup_cost * started

        matched_s = (q.sum(axis=1) > 0).astype(float)

        m = compute_metrics(profits_s, probs, matched_s, q, cvar_alpha)
        return {
            "order_type": self.order_type,
            "optimal_params": {"price_profile": optimal_prices.tolist()},
            "dispatch": q,
            "profits": profits_s,
            "matched": matched_s,
            **m,
        }


def _apply_energy_cap(
    q: np.ndarray,
    lambda_matrix: np.ndarray,
    avail: np.ndarray,
    e_max: float,
) -> np.ndarray:
    """
    For each scenario cap total daily dispatch to e_max MWh.
    Greedy: fill highest-price accepted periods first.
    """
    q_out = np.zeros_like(q)
    S = q.shape[0]
    for s in range(S):
        active = np.where(q[s] > 0)[0]
        if active.size == 0:
            continue
        order = active[np.argsort(lambda_matrix[s, active])[::-1]]
        energy = 0.0
        for t in order:
            remaining = e_max - energy
            if remaining <= 0:
                break
            dispatch = min(avail[t], remaining)
            q_out[s, t] = dispatch
            energy += dispatch
    return q_out
