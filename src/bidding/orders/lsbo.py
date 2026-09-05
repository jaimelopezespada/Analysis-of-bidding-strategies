"""Linked Block Orders (LSBO) strategy — parent/child block hierarchy."""

from __future__ import annotations

import itertools

import numpy as np

from ..config import ResolvedGrid, RiskObjective, TechnologyConfig
from ..metrics import compute_metrics, objective_value
from .base import OrderStrategy, block_avail_matrix


class LSBOStrategy(OrderStrategy):
    """
    LSBO: a parent block plus one or more child blocks (eq. 8-9).

        x_h^s ≤ x_p^s                                  for every child h
        family accepted iff  Σ_{b∈family} W_b^s ≥ 0     (joint welfare)
        a child is NEVER accepted with W_h^s < 0         (individual floor)

    Given each block's own (fixed) candidate price, this decomposes in closed
    form per scenario: since only children with non-negative own welfare are
    ever eligible, and including any eligible child can only raise (never
    lower) the family's joint welfare, the profit-maximising choice given the
    parent is accepted is always to include every eligible child. So per
    scenario: accept the parent (and all children with W_h^s ≥ 0) iff
    W_p^s + Σ_{h eligible} W_h^s ≥ 0; otherwise nothing in the family matches.
    No MILP is needed — the same reasoning that makes SBO/EXBO closed-form
    (see optimizer.py).
    """

    order_type = "lsbo"

    def evaluate(
        self,
        tech: TechnologyConfig,
        lambda_matrix: np.ndarray,
        avail_matrix: np.ndarray,
        probs: np.ndarray,
        grid: ResolvedGrid,
        objective: RiskObjective,
        cvar_alpha: float,
        startup_per_transition: bool = False,  # unused: family accept/reject is day-level
        sco_model: str = "aware",  # unused: only SCO has a solver-based clearing model
    ) -> dict:
        if not tech.lsbo_families:
            raise ValueError(
                f"{tech.name}: order_type 'lsbo' requested but no lsbo_families "
                "are configured for this technology."
            )
        S, T = lambda_matrix.shape
        cost = np.asarray(tech.cost_array(T), dtype=float)
        margin = lambda_matrix - cost[None, :]  # (S, T)

        best_obj = -np.inf
        best_result: dict | None = None

        ref = tech.price_reference_value
        for family in tech.lsbo_families:
            parent = family.parent
            children = family.children
            parent_price_grid = parent.resolved_price_levels(ref, grid.price_levels)
            child_price_grids = [
                c.resolved_price_levels(ref, grid.price_levels) for c in children
            ]

            for parent_price, *child_prices in itertools.product(parent_price_grid, *child_price_grids):
                parent_avail = block_avail_matrix(parent, avail_matrix)
                w_parent = (
                    np.sum(lambda_matrix * parent_avail, axis=1)
                    - float(parent_price) * parent_avail.sum(axis=1)
                )  # (S,)

                child_avail = np.zeros((len(children), S, T))
                w_children = np.zeros((len(children), S))
                for i, (child, price) in enumerate(zip(children, child_prices)):
                    avail_h = block_avail_matrix(child, avail_matrix)
                    child_avail[i] = avail_h
                    w_children[i] = (
                        np.sum(lambda_matrix * avail_h, axis=1) - float(price) * avail_h.sum(axis=1)
                    )

                eligible = w_children >= 0.0  # (n_children, S) — individual welfare floor
                family_welfare = w_parent + np.sum(
                    np.where(eligible, w_children, 0.0), axis=0
                )  # (S,)
                accepted = family_welfare >= 0.0  # (S,)

                q = np.zeros((S, T))
                q[accepted] = parent_avail[accepted]
                for i in range(len(children)):
                    child_accepted = accepted & eligible[i]
                    q[child_accepted] += child_avail[i][child_accepted]

                matched_s = accepted.astype(float)
                profits_s = np.sum(margin * q, axis=1)
                if tech.startup_cost > 0.0:
                    profits_s = profits_s - tech.startup_cost * matched_s

                obj = objective_value(profits_s, probs, cvar_alpha, objective)
                if obj > best_obj:
                    best_obj = obj
                    m = compute_metrics(profits_s, probs, matched_s, q, cvar_alpha, tech.installed_capacity_mw)
                    best_result = {
                        "order_type": self.order_type,
                        "optimal_params": {
                            "family_id": family.family_id,
                            "parent_price": float(parent_price),
                            "child_prices": {
                                c.id: float(p) for c, p in zip(children, child_prices)
                            },
                        },
                        "dispatch": q,
                        "profits": profits_s,
                        "matched": matched_s,
                        **m,
                    }

        assert best_result is not None, "LSBO grid is empty — check lsbo_families config."
        return best_result
