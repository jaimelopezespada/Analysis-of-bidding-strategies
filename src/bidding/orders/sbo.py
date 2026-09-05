"""Scalable Block Order (SBO) strategy."""

from __future__ import annotations

import itertools

import numpy as np

from reporting.optimizer import validate_block_profile

from ..config import ResolvedGrid, RiskObjective, TechnologyConfig
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

    The block's Q̂_t profile is declared data, not a per-scenario decision
    (fill-or-kill on the whole declared profile) — so unlike SCO, there is no
    intra-day dispatch freedom to solve for, and technical_min/ramp reduce to
    a static validity check on the declared profile itself (see
    optimizer.validate_block_profile), not a MILP.

    The grid (P^B × MAR) is enumerated; the combination maximising the
    weighted objective is returned.
    """

    order_type = "sbo"

    def evaluate(
        self,
        tech: TechnologyConfig,
        lambda_matrix: np.ndarray,
        avail_matrix: np.ndarray,
        probs: np.ndarray,
        grid: ResolvedGrid,
        objective: RiskObjective,
        cvar_alpha: float,
        startup_per_transition: bool = False,  # unused: SBO dispatches one contiguous block
        sco_model: str = "aware",  # unused: only SCO has a solver-based clearing model
    ) -> dict:
        S, T = lambda_matrix.shape
        avail = avail_matrix  # (S, T)
        validate_block_profile(tech, avail)
        cost = np.asarray(tech.cost_array(T), dtype=float)
        margin = lambda_matrix - cost[None, :]  # (S, T)

        # Pre-compute Σ_t λ_t^s · Q̂_t^s and Σ_t Q̂_t^s per scenario (reused in welfare)
        revenue_s = np.sum(lambda_matrix * avail, axis=1)  # (S,) = Σ_t λ_t^s · Q̂_t^s
        avail_sum_s = avail.sum(axis=1)                    # (S,) = Σ_t Q̂_t^s

        price_levels = grid.price_levels
        mar_levels = grid.mar_levels

        best_obj = -np.inf
        best_result: dict | None = None

        for pb, mar in itertools.product(price_levels, mar_levels):
            pb_f = float(pb)

            # welfare^s = Σ_t (λ_t^s − P^B) · Q̂_t^s
            welfare = revenue_s - pb_f * avail_sum_s  # (S,)

            u = (welfare >= 0.0).astype(float)        # (S,)
            q = u[:, None] * avail                    # (S, T)

            if tech.energy_capacity is not None:
                q = _apply_energy_cap(q, lambda_matrix, avail, tech.energy_capacity)

            profits_s = np.sum(margin * q, axis=1)
            if tech.startup_cost > 0.0:
                profits_s = profits_s - tech.startup_cost * u

            matched_s = u

            obj = objective_value(profits_s, probs, cvar_alpha, objective)
            if obj > best_obj:
                best_obj = obj
                m = compute_metrics(profits_s, probs, matched_s, q, cvar_alpha, tech.installed_capacity_mw)
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
