"""Scalable Complex Order (SCO) strategy with Minimum Income Condition."""

from __future__ import annotations

import itertools

import numpy as np

from ..config import CandidateGrid, RiskObjective, TechnologyConfig
from ..metrics import compute_metrics, objective_value
from ..optimizer import build_sco_model, extract_sco_results, solve_model
from .base import OrderStrategy
from .simple import _apply_energy_cap


class SCOStrategy(OrderStrategy):
    """
    SCO with a single day-level variable price P^V, fixed term TF, and
    per-period minimum acceptance volume MAV_t = mav_frac · Q̂_t.

    Acceptance rule (per scenario s, eq. 2-4):
      w_t^s ∈ {0,1}                                          (period committed — a decision,
                                                                for in-the-money AND
                                                                out-of-the-money periods alike)
      MAV_t^s ≤ q_t^s ≤ Q̂_t^s   when w_t^s = 1                (dispatch band once committed)
      MIC^s = Σ_t (λ_t^s - P^V) · q_t^s ≥ TF                (minimum income, gates u^s)
      u^s   = 1[MIC^s achievable ≥ 0]                        (full-day accept/reject)

    A period being out-of-the-money (P^V > λ_t^s) does not by itself prevent
    it from committing: it can still clear (subject to the same MAV_t floor)
    if doing so still leaves the aggregate MIC^s non-negative once its
    negative contribution to Σ_t (λ_t^s - P^V) · q_t^s is included.

    Each candidate theta = (P^V, TF, mav_frac) in the grid is solved as a single
    MILP spanning all scenarios (optimizer.build_sco_model), which is the only
    way to jointly resolve the day-level accept/reject with the per-period
    dispatch band and (when enforced) technical_min/ramp — see optimizer.py's
    module docstring for why Simple/SBO do not need a solver.
    """

    order_type = "sco"

    def evaluate(
        self,
        tech: TechnologyConfig,
        lambda_matrix: np.ndarray,
        avail_matrix: np.ndarray,
        probs: np.ndarray,
        grid: CandidateGrid,
        objective: RiskObjective,
        cvar_alpha: float,
        startup_per_transition: bool = False,  # unused: SCO acceptance is day-level
    ) -> dict:
        S, T = lambda_matrix.shape
        avail = avail_matrix  # (S, T)
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
            theta = {"price_variable": pv_f, "fixed_term": tf_f, "mav_fraction": float(mav_frac)}

            model = build_sco_model(tech, lambda_matrix, avail, probs, cvar_alpha, objective, theta)
            solve_model(model)
            q, u = extract_sco_results(model, S, T)

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
                        "price_variable": pv_f,
                        "fixed_term": tf_f,
                        "mav_fraction": float(mav_frac),
                        "mav_profile": (float(mav_frac) * avail.mean(axis=0)).tolist(),
                    },
                    "dispatch": q,
                    "profits": profits_s,
                    "matched": matched_s,
                    **m,
                }

        assert best_result is not None, "SCO grid is empty — check candidate_grid config."
        return best_result
