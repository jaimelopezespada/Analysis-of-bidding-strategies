"""Scalable Complex Order (SCO) strategy with Minimum Income Condition."""

from __future__ import annotations

import itertools

import numpy as np

from ..config import CandidateGrid, RiskObjective, TechnologyConfig
from ..metrics import compute_metrics, objective_value
from ..optimizer import build_sco_model_naive, extract_sco_results, solve_model
from .base import OrderStrategy
from .simple import _apply_energy_cap


def _clear_sco_aware(
    tech: TechnologyConfig,
    lambda_matrix: np.ndarray,
    avail_matrix: np.ndarray,
    theta: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form solution of the revised day-level aware SCO clearing model.

    Under the revised aware semantics, the SCO is treated as a daily order.
    Once the day is accepted by the day-level MIC, the declared daily profile is
    dispatched in all periods. The acceptance rule is therefore:

    1. start from q_t = Q̄_t for every period t in the day
    2. if technical constraints are enforced, discard periods whose available
       quantity is below the plant's technical minimum
    3. compute the day-level declared surplus Σ_t (λ_t - P^V) · q_t
    4. accept the order (u=1) iff that surplus covers TF

    This means the order is no longer screened period-by-period by
    moneyness; the day-level MIC is the only gate on whether the SCO is
    accepted. The closed form remains fast because it just needs the daily
    surplus test.
    """
    pv = float(theta["price_variable"])
    tf = float(theta["fixed_term"])

    q = avail_matrix.copy()  # (S, T)
    if tech.enforce_technical_constraints and tech.technical_min > 0:
        q = np.where(avail_matrix >= tech.technical_min, avail_matrix, 0.0)

    surplus = ((lambda_matrix - pv) * q).sum(axis=1)  # (S,)
    u = (surplus >= tf - 1e-6).astype(float)          # MIC, eq. 4
    q = q * u[:, None]
    # An order that dispatches nothing is not matched (guards the TF<=0,
    # zero-surplus degeneracy from charging a phantom startup).
    u = (q.sum(axis=1) > 1e-9).astype(float)
    return q, u


def evaluate_sco_theta(
    tech: TechnologyConfig,
    lambda_matrix: np.ndarray,
    avail_matrix: np.ndarray,
    probs: np.ndarray,
    theta: dict,
    cvar_alpha: float,
    objective: RiskObjective,
    sco_model: str = "aware",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clear one fixed SCO candidate theta and price the result at real cost.

    Clears theta = {price_variable, fixed_term, mav_fraction} — in closed
    form under sco_model="aware", via the legacy MILP under "naive" — then
    computes the realised per-scenario profit on the resulting dispatch using
    the generator's private costs (C, C^SU), which the "aware" clearing step
    never sees.

    Returns (q, u, profits_s): dispatch (S, T), day-level acceptance (S,)
    and realised profit per scenario (S,). Shared by SCOStrategy.evaluate
    (grid search) and validation.py (out-of-sample evaluation of a fixed
    theta*).
    """
    S, T = lambda_matrix.shape
    cost = np.asarray(tech.cost_array(T), dtype=float)
    margin = lambda_matrix - cost[None, :]  # (S, T)

    if sco_model == "aware":
        q, u = _clear_sco_aware(tech, lambda_matrix, avail_matrix, theta)
    elif sco_model == "naive":
        model = build_sco_model_naive(
            tech, lambda_matrix, avail_matrix, probs, cvar_alpha, objective, theta
        )
        solve_model(model)
        q, u = extract_sco_results(model, S, T)
    else:
        raise ValueError(f"Unknown sco_model {sco_model!r} — expected 'naive' or 'aware'.")

    if tech.energy_capacity is not None:
        q = _apply_energy_cap(q, lambda_matrix, avail_matrix, tech.energy_capacity)

    profits_s = np.sum(margin * q, axis=1)
    if tech.startup_cost > 0.0:
        profits_s = profits_s - tech.startup_cost * u

    return q, u, profits_s


class SCOStrategy(OrderStrategy):
    """
    SCO with a single day-level variable price P^V, fixed term TF, and
    per-period minimum acceptance volume MAV_t = mav_frac · Q̂_t.

    Acceptance rule (per scenario s, eq. 2-4):
      w_t^s ∈ {0,1}                                          (period committed)
      MAV_t^s ≤ q_t^s ≤ Q̂_t^s   when w_t^s = 1                (dispatch band once committed)
      MIC^s = Σ_t (λ_t^s - P^V) · q_t^s ≥ TF                (minimum income, gates u^s)
      u^s   = 1[MIC^s achievable ≥ 0]                        (full-day accept/reject)

    How (q, u, w) are decided depends on the clearing model (``sco_model``,
    see optimizer.py):
    - "aware" (default): clearing maximises the declared surplus
      Σ_t (λ_t^s − P^V)·q_t^s — the only information EUPHEMIA has. The real
      costs (C, C^SU) enter only below, when profits_s is computed on the
      fixed dispatch, exactly like SBO/Simple. Under the revised day-level
      aware semantics, an out-of-the-money period (P^V > λ_t^s) is not ruled
      out ex ante; its declared contribution only enters the day-level MIC.
      If the daily declared surplus covers TF, the accepted day dispatches the
      full daily profile, so the solution remains closed form under
      _clear_sco_aware.
    - "naive": legacy benchmark where a MILP spanning all scenarios
      (optimizer.build_sco_model_naive) maximises the real risk-adjusted
      profit, i.e. clearing sees the private costs — an upper bound the
      market cannot replicate. Here an OTM period can still clear (subject
      to MAV/MIC) whenever it is profitable at real cost, a genuine coupled
      decision that does require the solver.
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
        sco_model: str = "aware",
    ) -> dict:
        avail = avail_matrix  # (S, T)

        price_levels = grid.price_levels
        tf_levels = grid.tf_levels
        mav_fracs = grid.mav_fraction_levels

        best_obj = -np.inf
        best_result: dict | None = None

        for pv, tf, mav_frac in itertools.product(price_levels, tf_levels, mav_fracs):
            pv_f = float(pv)
            tf_f = float(tf)
            theta = {"price_variable": pv_f, "fixed_term": tf_f, "mav_fraction": float(mav_frac)}

            q, u, profits_s = evaluate_sco_theta(
                tech, lambda_matrix, avail, probs, theta, cvar_alpha, objective,
                sco_model=sco_model,
            )

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
                        "sco_model": sco_model,
                    },
                    "dispatch": q,
                    "profits": profits_s,
                    "matched": matched_s,
                    **m,
                }

        assert best_result is not None, "SCO grid is empty — check candidate_grid config."
        return best_result
