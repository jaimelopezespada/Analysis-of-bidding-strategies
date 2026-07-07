"""Abstract base for order-type strategies (Strategy pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import CandidateGrid, ObjectiveWeights, TechnologyConfig


class OrderStrategy(ABC):
    """
    Each concrete subclass models one EUPHEMIA product (Simple, SCO, SBO, …).
    Adding EXBO/LSBO in v2 means adding new subclasses here.
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
        probs: np.ndarray,          # shape (S,)
        grid: CandidateGrid,
        weights: ObjectiveWeights,
        cvar_alpha: float,
    ) -> dict:
        """
        Enumerate the candidate grid and return the best combination.

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
        """
        ...
