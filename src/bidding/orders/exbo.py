"""Exclusive Group of Block Orders (EXBO) strategy."""

from __future__ import annotations

import itertools

import numpy as np

from ..config import CandidateGrid, RiskObjective, TechnologyConfig
from ..metrics import compute_metrics, objective_value
from .base import OrderStrategy


class EXBOStrategy(OrderStrategy):
    """
    EXBO: a group of mutually-exclusive blocks (eq. 7), Σ_{b∈G} x_b^s ≤ 1.

    Each block in the group is itself an SBO-style fill-or-kill block with its
    own declared price and availability profile. Because each block's welfare
    only depends on its own (fixed) candidate price, the ≤1 exclusivity
    constraint decomposes in closed form per scenario: among the blocks with
    non-negative welfare, the one with the highest welfare is selected (if
    none qualify, the group matches nothing that scenario) — no MILP needed,
    the same reasoning that makes SBO closed-form (see optimizer.py).

    The grid enumerated is the cartesian product of each block's own
    price_levels (falling back to the run/tech grid when a block does not
    declare its own).
    """

    order_type = "exbo"

    def evaluate(
        self,
        tech: TechnologyConfig,
        lambda_matrix: np.ndarray,
        avail_matrix: np.ndarray,
        probs: np.ndarray,
        grid: CandidateGrid,
        objective: RiskObjective,
        cvar_alpha: float,
        startup_per_transition: bool = False,  # unused: at most one block matches per day
        sco_model: str = "aware",  # unused: only SCO has a solver-based clearing model
    ) -> dict:
        if not tech.exbo_groups:
            raise ValueError(
                f"{tech.name}: order_type 'exbo' requested but no exbo_groups "
                "are configured for this technology."
            )
        S, T = lambda_matrix.shape
        cost = np.asarray(tech.cost_array(T), dtype=float)
        margin = lambda_matrix - cost[None, :]  # (S, T)

        best_obj = -np.inf
        best_result: dict | None = None

        for group in tech.exbo_groups:
            blocks = group.blocks
            block_price_grids = [
                b.price_levels if b.price_levels is not None else grid.price_levels
                for b in blocks
            ]

            for price_combo in itertools.product(*block_price_grids):
                # welfare_b^s = Σ_t (λ_t^s − P_b) · Q̂_{b,t}^s, for each block
                welfare = np.zeros((len(blocks), S))
                block_avail = np.zeros((len(blocks), S, T))
                for i, (block, price) in enumerate(zip(blocks, price_combo)):
                    avail_b = np.tile(np.asarray(block.availability.values, dtype=float), (S, 1))
                    block_avail[i] = avail_b
                    revenue_b = np.sum(lambda_matrix * avail_b, axis=1)
                    welfare[i] = revenue_b - float(price) * avail_b.sum(axis=1)

                # Best qualifying block per scenario (highest welfare among those >= 0)
                qualifies = welfare >= 0.0                      # (n_blocks, S)
                masked_welfare = np.where(qualifies, welfare, -np.inf)
                any_qualifies = qualifies.any(axis=0)            # (S,)
                best_block_idx = np.argmax(masked_welfare, axis=0)  # (S,)

                q = np.zeros((S, T))
                matched_s = np.zeros(S)
                for s in range(S):
                    if any_qualifies[s]:
                        b = best_block_idx[s]
                        q[s] = block_avail[b, s]
                        matched_s[s] = 1.0

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
                            "group_id": group.group_id,
                            "block_prices": {
                                b.id: float(p) for b, p in zip(blocks, price_combo)
                            },
                        },
                        "dispatch": q,
                        "profits": profits_s,
                        "matched": matched_s,
                        **m,
                    }

        assert best_result is not None, "EXBO grid is empty — check exbo_groups config."
        return best_result
