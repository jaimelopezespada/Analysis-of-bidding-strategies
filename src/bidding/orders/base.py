"""Abstract base for order-type strategies (Strategy pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import BlockConfig, ResolvedGrid, RiskObjective, TechnologyConfig


def block_avail_matrix(block: BlockConfig, avail_matrix: np.ndarray) -> np.ndarray:
    """(S, T) declared availability of an EXBO/LSBO block.

    source="static": the declared per-period profile, identical in every
    scenario (legacy behaviour — fine for dispatchable plants whose
    availability is constant anyway).
    source="tech_fraction": per-period fractions of the technology's real
    scenario-varying availability, so the block's declared quantity tracks
    the actual daily resource (solar/wind) instead of promising MW that do
    not exist on low-resource days.
    """
    S = avail_matrix.shape[0]
    values = np.asarray(block.availability.values, dtype=float)
    if block.availability.source == "tech_fraction":
        return values[None, :] * avail_matrix
    return np.tile(values, (S, 1))


class OrderStrategy(ABC):
    """
    Each concrete subclass models one EUPHEMIA product (Simple, SCO, SBO, EXBO, LSBO).
    """

    @property
    @abstractmethod
    def order_type(self) -> str:
        """Short identifier used in output tables: 'simple', 'sco', 'sbo', …"""
        ...

    @abstractmethod
    def evaluate(
        self,
        tech: TechnologyConfig,
        lambda_matrix: np.ndarray,  # shape (S, T)
        avail_matrix: np.ndarray,   # shape (S, T) — Q_bar_t^s, scenario-varying
        probs: np.ndarray,          # shape (S,)
        grid: ResolvedGrid,
        objective: RiskObjective,
        cvar_alpha: float,
        startup_per_transition: bool = False,
        sco_model: str = "aware",
    ) -> dict:
        """
        Enumerate the candidate grid and return the best combination.

        ``startup_per_transition``: charge one startup per zero→production
        transition instead of a single daily startup. Only simple orders can
        restart within the day, so the other strategies accept and ignore it.

        ``sco_model``: which SCO clearing model to use ("aware" = declared
        surplus, "naive" = legacy real-profit benchmark; see optimizer.py).
        Only the SCO strategy uses it; the others accept and ignore it.

        Returns a dict containing:
            order_type          str
            optimal_params      dict  – declared bid parameters θ*
            dispatch            ndarray (S, T)  – energy matched per scenario/period
            profits             ndarray (S,)    – net profit per scenario
            matched             ndarray (S,)    – 1.0 if order accepted in scenario s
            expected_profit     float
            cvar                float
            match_probability   float
            expected_matched_energy float
            expected_matched_periods float – N̄ = Σ_s ρ_s · Σ_t 1[q_t^s > 0]
            expected_profit_per_mw float  – E[Π] / installed capacity (€/MW)
        """
        ...
