"""Scalable Complex Order (SCO) strategy with Minimum Income Condition."""

from __future__ import annotations

import itertools

import numpy as np

from ..config import CandidateGrid, ObjectiveWeights, TechnologyConfig
from ..metrics import compute_metrics, objective_value
from .base import OrderStrategy
from .simple import _apply_energy_cap


class SCOStrategy(OrderStrategy):
    """
    SCO with a single day-level variable price P^V, fixed term TF, and
    per-period minimum acceptance volume MAV_t = mav_frac · Q̂_t.

    Acceptance rule (per scenario s):
      δ_t^s = 1[P^V ≤ λ_t^s]                          (period active?)
      MIC^s = Σ_t (λ_t^s − P^V) · Q̂_t · δ_t^s ≥ TF  (minimum income)
      u^s   = 1[MIC^s ≥ 0]                             (full-day accept/reject)
      q_t^s = Q̂_t · δ_t^s · u^s

    The grid (P^V × TF × mav_frac) is enumerated; the combination maximising
    the weighted objective is returned.  mav_frac is stored in optimal_params
    but does not change dispatch in the price-taker model (Q̂_t ≥ MAV_t always).
    """

    order_type = "sco"

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

        price_levels = grid.price_levels
        tf_levels = grid.tf_levels
        mav_fracs = grid.mav_fraction_levels

        best_obj = -np.inf
        best_result: dict | None = None

        for pv, tf, mav_frac in itertools.product(price_levels, tf_levels, mav_fracs):
            pv_f = float(pv)
            tf_f = float(tf)

            # δ_t^s: period t active in scenario s
            delta = (pv_f <= lambda_matrix).astype(float)          # (S, T)

            # Revenue surplus over declared variable cost, per scenario
            mic_surplus = np.sum((lambda_matrix - pv_f) * avail[None, :] * delta, axis=1) - tf_f  # (S,)

            # Full-day accept/reject binary
            u = (mic_surplus >= 0.0).astype(float)                 # (S,)

            q = avail[None, :] * delta * u[:, None]                # (S, T)

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
                        "price_variable": pv_f,
                        "fixed_term": tf_f,
                        "mav_fraction": float(mav_frac),
                        "mav_profile": (float(mav_frac) * avail).tolist(),
                    },
                    "dispatch": q,
                    "profits": profits_s,
                    "matched": matched_s,
                    **m,
                }

        assert best_result is not None, "SCO grid is empty — check candidate_grid config."
        return best_result
