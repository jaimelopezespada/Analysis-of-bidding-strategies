"""Shared test helpers for order-strategy unit tests."""

from __future__ import annotations

import numpy as np

from bidding.availability import static_availability_matrix
from bidding.config import ResolvedGrid, RiskObjective, TechnologyConfig


def make_tech(
    avail: list[float],
    cost: float = 0.0,
    startup: float = 0.0,
    energy_cap: float | None = None,
    family: int = 1,
    technical_min: float = 0.0,
    ramp_limit: float | None = None,
    enforce_technical_constraints: bool = False,
    price_reference: float | None = None,
) -> TechnologyConfig:
    # price_reference=1.0 by default when cost is 0 (required by the config
    # validator) so pct block levels in tests read as absolute €/MWh.
    if price_reference is None and cost == 0.0:
        price_reference = 1.0
    return TechnologyConfig(
        name="Test",
        family=family,
        side="sell",
        variable_cost=cost,
        price_reference=price_reference,
        startup_cost=startup,
        technical_min=technical_min,
        ramp_limit=ramp_limit,
        enforce_technical_constraints=enforce_technical_constraints,
        energy_capacity=energy_cap,
        availability={"values": avail},
    )


def make_grid(**kwargs) -> ResolvedGrid:
    """Strategy-facing grid with ABSOLUTE price levels (ResolvedGrid)."""
    defaults = dict(
        price_levels=[0, 10, 20, 40, 60],
        mar_levels=[0.5],
        mav_fraction_levels=[0.5],
        tf_levels=[0],
    )
    defaults.update(kwargs)
    return ResolvedGrid(**defaults)


def avail_matrix(tech: TechnologyConfig, S: int) -> np.ndarray:
    """Tile the technology's static availability profile across S scenarios."""
    return static_availability_matrix(tech, S)


W = RiskObjective()  # beta=0.0 -> plain expected-profit maximisation
