"""Scalable Block Order (SBO) strategy."""

from __future__ import annotations

import itertools

import numpy as np

from ..config import CandidateGrid, ObjectiveWeights, TechnologyConfig
from ..metrics import compute_metrics, objective_value
from .base import OrderStrategy
from .simple import _apply_energy_cap


class SBOStrategy(OrderStrategy):
    """
    SBO: the entire daily profile Q̂_t is offered at a single block price P^B.
    The block is accepted if its welfare is non-negative:
        welfare^s = Σ_t (λ_t^s − P^B) · Q̂_t ≥ 0  →  u^s = 1
        q_t^s = u^s · Q̂_t

    MAR (Minimum Acceptance Ratio) is declared and stored in optimal_params.
    In the price-taker model the block always dispatches at x^s = 1 when
    accepted, so MAR only acts as a market-side minimum guarantee.

    The grid (P^B × MAR) is enumerated; the combination maximising the
    weighted objective is returned.
    """

    order_type = "sbo"

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
        avail = np.asarray(tech.availability.values, dtype=float)
        cost = np.asarray(tech.cost_array(T), dtype=float)
        margin = lambda_matrix - cost[None, :]  # (S, T)

        # Pre-compute Σ_t λ_t^s · Q̂_t for all scenarios (reused in welfare)
        revenue_s = lambda_matrix @ avail  # (S,)  = Σ_t λ_t^s · Q̂_t

        price_levels = grid.price_levels
        mar_levels = grid.mar_levels

        best_obj = -np.inf
        best_result: dict | None = None

        for pb, mar in itertools.product(price_levels, mar_levels):
            pb_f = float(pb)

            # welfare^s = Σ_t (λ_t^s − P^B) · Q̂_t
            welfare = revenue_s - pb_f * avail.sum()  # (S,)

            u = (welfare >= 0.0).astype(float)        # (S,)
            q = u[:, None] * avail[None, :]           # (S, T)

            if tech.energy_capacity is not None:
                q = _apply_energy_cap(q, lambda_matrix, avail, tech.energy_capacity)

            profits_s = np.sum(margin * q, axis=1)
            if tech.startup_cost > 0.0:
                profits_s = profits_s - tech.startup_cost * u

            matched_s = u

            obj = objective_value(profits_s, probs, matched_s, cvar_alpha, weights)
            if obj > best_obj:
                best_obj = obj
                m = compute_metrics(profits_s, probs, matched_s, q, cvar_alpha)
                best_result = {
                    "order_type": self.order_type,
                    "optimal_params": {
                        "block_price": pb_f,
                        "mar": float(mar),
                    },
                    "dispatch": q,
                    "profits": profits_s,
                    "matched": matched_s,
                    **m,
                }

        assert best_result is not None, "SBO grid is empty — check candidate_grid config."
        return best_result
