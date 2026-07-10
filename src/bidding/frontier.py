"""Risk-return frontier: sweep beta and re-solve for each value (eq. 15, H4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CandidateGrid, RiskObjective, TechnologyConfig
from .metrics import objective_value
from .orders import STRATEGIES


def sweep_beta(
    tech: TechnologyConfig,
    lambda_matrix: np.ndarray,
    avail_matrix: np.ndarray,
    probs: np.ndarray,
    grid: CandidateGrid,
    cvar_alpha: float,
    order_type: str,
    betas: list[float],
    startup_per_transition: bool = False,
) -> pd.DataFrame:
    """
    Re-evaluate ``order_type`` once per beta in ``betas``. The optimal theta
    can legitimately shift with beta, so each point re-runs the full grid
    search under that beta rather than re-scoring a fixed theta — this is the
    only correct way to trace eq. 15's beneficio-riesgo frontier.
    """
    strategy = STRATEGIES[order_type]()
    rows = []
    for beta in betas:
        objective = RiskObjective(beta=beta)
        result = strategy.evaluate(
            tech, lambda_matrix, avail_matrix, probs, grid, objective, cvar_alpha,
            startup_per_transition=startup_per_transition,
        )
        obj_value = objective_value(result["profits"], probs, cvar_alpha, objective)
        rows.append(
            {
                "order_type": order_type,
                "beta": beta,
                "expected_profit": result["expected_profit"],
                "cvar": result["cvar"],
                "objective_value": obj_value,
            }
        )
    return pd.DataFrame(rows)
