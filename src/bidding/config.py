"""Pydantic models for technology and run configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class AvailabilityConfig(BaseModel):
    """Q_bar_t: capacity/forecast available per period.

    source="static": a single T-length profile, repeated identically for
    every scenario (legacy behaviour; still used as the deterministic-mode
    fallback even when source="renewable_csv").
    source="renewable_csv": a scenario-varying (S,T) profile is built at
    run time from the real ESIOS generation CSV (see availability.py),
    scaled proportionally from national aggregate generation to this
    technology's nameplate capacity.
    """

    values: Optional[list[float]] = None
    source: Literal["static", "renewable_csv"] = "static"
    resource: Optional[Literal["solar", "wind"]] = None
    nameplate_capacity_mw: Optional[float] = None

    @model_validator(mode="after")
    def _check_source_fields(self) -> "AvailabilityConfig":
        if self.source == "static":
            if self.values is None:
                raise ValueError("availability.values is required when source='static'.")
        else:
            if self.resource is None or self.nameplate_capacity_mw is None:
                raise ValueError(
                    "availability.resource and availability.nameplate_capacity_mw "
                    "are required when source='renewable_csv'."
                )
        return self


class BlockConfig(BaseModel):
    """A single block used by SBO, or one member of an EXBO group / LSBO family."""

    id: str
    price_levels: Optional[list[float]] = None
    mar_levels: Optional[list[float]] = None
    availability: AvailabilityConfig


class ExboGroupConfig(BaseModel):
    group_id: str
    blocks: list[BlockConfig]

    @model_validator(mode="after")
    def _check_min_blocks(self) -> "ExboGroupConfig":
        if len(self.blocks) < 2:
            raise ValueError(f"EXBO group '{self.group_id}' needs at least 2 blocks.")
        return self


class LsboFamilyConfig(BaseModel):
    family_id: str
    parent: BlockConfig
    children: list[BlockConfig]

    @model_validator(mode="after")
    def _check_min_children(self) -> "LsboFamilyConfig":
        if len(self.children) < 1:
            raise ValueError(f"LSBO family '{self.family_id}' needs at least 1 child block.")
        return self


class CandidateGrid(BaseModel):
    """Discrete candidate values enumerated during optimisation.

    Set these per-technology in the tech YAML (recommended) so the grid is
    centred around each technology's actual variable cost and startup cost.
    The RunConfig holds a generic fallback grid used when the tech YAML does
    not define its own.
    """

    price_levels: list[float] = [0, 1, 5, 10, 20, 40, 60, 80]
    mar_levels: list[float] = [0.0, 0.3, 0.5, 0.8, 1.0]
    mav_fraction_levels: list[float] = [0.0, 0.5, 0.8, 1.0]
    tf_levels: list[float] = [0, 50, 100, 200, 500]


class TechnologyConfig(BaseModel):
    name: str
    family: int = Field(ge=1, le=4)
    side: Literal["sell"]
    variable_cost: float | list[float] = 0.0
    startup_cost: float = 0.0
    technical_min: float = 0.0
    ramp_limit: Optional[float] = None
    # Opt-in: enforce technical_min/ramp_limit as real MILP constraints.
    # Off by default so Family-1/battery behaviour is unaffected.
    enforce_technical_constraints: bool = False
    # MWh/day energy budget (batteries, reservoir hydro). None = unlimited.
    energy_capacity: Optional[float] = None
    availability: AvailabilityConfig
    # Technology-specific grid; overrides RunConfig.candidate_grid when present.
    candidate_grid: Optional[CandidateGrid] = None
    exbo_groups: Optional[list[ExboGroupConfig]] = None
    lsbo_families: Optional[list[LsboFamilyConfig]] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TechnologyConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def cost_array(self, T: int) -> list[float]:
        """Variable cost as a list of length T."""
        if isinstance(self.variable_cost, list):
            if len(self.variable_cost) != T:
                raise ValueError(
                    f"variable_cost list length {len(self.variable_cost)} != resolution {T}"
                )
            return self.variable_cost
        return [self.variable_cost] * T

    def effective_grid(self, fallback: CandidateGrid) -> CandidateGrid:
        """Return the technology grid if defined, otherwise the run-level fallback.

        Raises if any MAR/MAV level is below technical_min/installed_capacity_mw:
        the plant can only clear at 0 or >= technical_min (never in between), so
        a level in [0, technical_min) would declare an acceptance/dispatch
        minimum it cannot physically honour.
        """
        grid = self.candidate_grid if self.candidate_grid is not None else fallback
        if self.technical_min > 0:
            floor = self.technical_min / self.installed_capacity_mw
            bad = sorted({
                lvl for lvl in (*grid.mar_levels, *grid.mav_fraction_levels) if lvl < floor - 1e-9
            })
            if bad:
                raise ValueError(
                    f"{self.name}: candidate_grid tiene nivel(es) de MAR/MAV {bad} por debajo de "
                    f"technical_min/installed_capacity ({floor:.3f}) — la planta solo puede despachar "
                    f"0 o >= technical_min ({self.technical_min}), nunca un valor intermedio."
                )
        return grid

    @property
    def installed_capacity_mw(self) -> float:
        """Nameplate/installed capacity in MW.

        Used to normalise profit metrics across technologies of different
        size (€/MW). For source="renewable_csv" this is the declared
        nameplate; for source="static" (dispatchable plants, batteries) the
        availability profile IS the plant's power capacity per period, so
        its peak is the installed capacity.
        """
        if self.availability.source == "renewable_csv":
            return float(self.availability.nameplate_capacity_mw)
        return float(max(self.availability.values))


class RiskObjective(BaseModel):
    """max (1-beta)*E[Pi] + beta*CVaR_alpha[Pi]. beta=0 recovers plain expected profit."""

    beta: float = Field(default=0.0, ge=0.0, le=1.0)


class PricesConfig(BaseModel):
    csv_path: str
    scenario_mode: Literal["per_day"] = "per_day"
    date_range: Optional[tuple[str, str]] = None
    renewable_csv_path: Optional[str] = None


class RunConfig(BaseModel):
    mode: Literal["deterministic", "stochastic"] = "stochastic"
    resolution: Literal[24, 96] = 24
    order_types: list[Literal["simple", "sco", "sbo", "exbo", "lsbo"]] = ["simple", "sco", "sbo"]
    prices: PricesConfig
    objective: RiskObjective = Field(default_factory=RiskObjective)
    cvar_alpha: float = Field(default=0.95, ge=0.0, le=1.0)
    candidate_grid: CandidateGrid = Field(default_factory=CandidateGrid)
    output_dir: str = "results"
    # Optional label (e.g. "verano", "invierno") namespacing results/ so two
    # run configs sharing the same output_dir don't overwrite each other's
    # ranking.csv/figs when run against the same technology.
    season: Optional[str] = None
    seed: int = 42
    beta_sweep: Optional[list[float]] = None
    # Opt-in: charge one startup per zero→production transition instead of a
    # single daily startup. Only affects simple orders — block products
    # (SBO/SCO/EXBO/LSBO) dispatch contiguous blocks with at most one startup.
    startup_per_transition: bool = False

    @model_validator(mode="after")
    def check_order_types_nonempty(self) -> "RunConfig":
        if not self.order_types:
            raise ValueError("order_types must contain at least one entry.")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
