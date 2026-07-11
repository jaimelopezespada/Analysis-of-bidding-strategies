"""Abstract base for order-type strategies (Strategy pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import CandidateGrid, RiskObjective, TechnologyConfig


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
        grid: CandidateGrid,
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
            expected_profit_per_mw float  – E[Π] / installed capacity (€/MW)
        """
        ...
