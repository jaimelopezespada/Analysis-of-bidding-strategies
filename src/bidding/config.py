"""Pydantic models for technology and run configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AvailabilityConfig(BaseModel):
    """Q_bar_t: capacity/forecast available per period.

    source="static": a single T-length profile, repeated identically for
    every scenario (legacy behaviour; still used as the deterministic-mode
    fallback even when source="renewable_csv").
    source="renewable_csv": a scenario-varying (S,T) profile is built at
    run time from the real ESIOS generation CSV (see availability.py),
    scaled proportionally from national aggregate generation to this
    technology's nameplate capacity.
    source="tech_fraction": only valid inside EXBO/LSBO blocks — ``values``
    are per-period fractions in [0, 1] of the TECHNOLOGY's scenario-varying
    availability matrix, so the block's declared quantity tracks the real
    per-day resource instead of a static profile (block matrix =
    values × avail_matrix, see orders/base.py::block_avail_matrix).
    """

    values: Optional[list[float]] = None
    source: Literal["static", "renewable_csv", "tech_fraction"] = "static"
    resource: Optional[Literal["solar", "wind"]] = None
    nameplate_capacity_mw: Optional[float] = None
    # Fracción del máximo diario por debajo de la cual la generación del CSV
    # se trunca a 0 (limpia la generación residual nocturna del agregado
    # nacional ESIOS, que infla las horas casadas esperadas). 0.0 = sin filtro.
    daily_min_generation_pct: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check_source_fields(self) -> "AvailabilityConfig":
        if self.daily_min_generation_pct > 0 and self.source != "renewable_csv":
            raise ValueError(
                "availability.daily_min_generation_pct only applies when "
                "source='renewable_csv' (it filters the ESIOS generation CSV)."
            )
        if self.source == "static":
            if self.values is None:
                raise ValueError("availability.values is required when source='static'.")
        elif self.source == "tech_fraction":
            if self.values is None:
                raise ValueError("availability.values is required when source='tech_fraction'.")
            bad = [v for v in self.values if not 0.0 <= v <= 1.0]
            if bad:
                raise ValueError(
                    f"availability.values must be fractions in [0, 1] when "
                    f"source='tech_fraction' — got {bad[:3]}."
                )
        else:
            if self.resource is None or self.nameplate_capacity_mw is None:
                raise ValueError(
                    "availability.resource and availability.nameplate_capacity_mw "
                    "are required when source='renewable_csv'."
                )
        return self


class BlockConfig(BaseModel):
    """A single block used by SBO, or one member of an EXBO group / LSBO family."""

    model_config = ConfigDict(extra="forbid")

    id: str
    # Fractions of the technology's price reference (see
    # TechnologyConfig.price_reference_value). None → the block inherits the
    # already-resolved absolute levels of the candidate grid.
    price_levels_pct: Optional[list[float]] = None
    mar_levels: Optional[list[float]] = None
    availability: AvailabilityConfig

    def resolved_price_levels(self, reference: float, fallback: list[float]) -> list[float]:
        """Absolute €/MWh price candidates for this block.

        ``fallback`` must already be absolute (a ResolvedGrid's price_levels).
        """
        if self.price_levels_pct is None:
            return fallback
        return [p * reference for p in self.price_levels_pct]


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
    """Discrete candidate values enumerated during optimisation (YAML-facing).

    ``price_levels_pct`` are FRACTIONS of the technology's price reference
    (by default its variable cost; see TechnologyConfig.price_reference_value):
    1.0 = bid exactly at the reference. They are resolved to absolute €/MWh
    via TechnologyConfig.resolved_grid(), which returns a ResolvedGrid — the
    only grid type strategies ever see. Expressing levels relative to the
    reference means a change in a technology's cost (including the hydro
    water value, resolved per month) rescales every candidate automatically.

    mar_levels / mav_fraction_levels are capacity fractions and tf_levels are
    absolute € (startup-cost related): they stay absolute.

    extra="forbid" so a stale YAML still declaring absolute ``price_levels``
    fails loudly instead of silently using the default pct grid.
    """

    model_config = ConfigDict(extra="forbid")

    price_levels_pct: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    mar_levels: list[float] = [0.0, 0.3, 0.5, 0.8, 1.0]
    mav_fraction_levels: list[float] = [0.0, 0.5, 0.8, 1.0]
    tf_levels: list[float] = [0, 50, 100, 200, 500]


class ResolvedGrid(BaseModel):
    """A CandidateGrid with price levels resolved to absolute €/MWh.

    Built by TechnologyConfig.resolved_grid(); this is what the order
    strategies consume (they never see pct levels).
    """

    price_levels: list[float]
    mar_levels: list[float] = [0.0, 0.3, 0.5, 0.8, 1.0]
    mav_fraction_levels: list[float] = [0.0, 0.5, 0.8, 1.0]
    tf_levels: list[float] = [0, 50, 100, 200, 500]


class TechnologyConfig(BaseModel):
    name: str
    family: int = Field(ge=1, le=4)
    side: Literal["sell"]
    variable_cost: float | list[float] = 0.0
    # "static": variable_cost is the YAML value (legacy behaviour).
    # "monthly_price_mean": the cost is a proxy for a monthly opportunity cost
    # (hydro water value) and is resolved at run time to the mean hourly
    # market price of each calendar month's scenarios — see
    # monthly.resolve_tech_for_month. Until resolved, cost_array /
    # price_reference_value raise.
    variable_cost_source: Literal["static", "monthly_price_mean"] = "static"
    # Reference (€/MWh) that price_levels_pct are fractions of. Defaults to
    # the (scalar) variable_cost; REQUIRED explicitly when variable_cost is 0
    # (solar/wind: pct of 0 collapses every level to 0) or a per-period list.
    # Forbidden with variable_cost_source="monthly_price_mean", where the
    # reference must follow the dynamically resolved monthly cost.
    price_reference: Optional[float] = None
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

    @model_validator(mode="after")
    def _check_price_reference(self) -> "TechnologyConfig":
        if self.availability.source == "tech_fraction":
            raise ValueError(
                f"{self.name}: availability.source='tech_fraction' is only valid "
                "inside EXBO/LSBO blocks, not at technology level."
            )
        if self.variable_cost_source == "monthly_price_mean":
            if self.price_reference is not None:
                raise ValueError(
                    f"{self.name}: price_reference is not allowed with "
                    "variable_cost_source='monthly_price_mean' — the reference "
                    "follows the monthly resolved cost."
                )
            return self
        if self.price_reference is not None:
            if self.price_reference <= 0:
                raise ValueError(f"{self.name}: price_reference must be > 0.")
        elif isinstance(self.variable_cost, list) or self.variable_cost == 0.0:
            raise ValueError(
                f"{self.name}: price_reference (> 0) is required when variable_cost "
                "is 0 or a per-period list — price_levels_pct cannot be scaled by a "
                "zero/ambiguous cost."
            )
        return self

    def _require_resolved(self) -> None:
        if self.variable_cost_source == "monthly_price_mean":
            raise RuntimeError(
                f"{self.name}: variable_cost is dynamic (monthly_price_mean) and has "
                "not been resolved — call monthly.resolve_tech_for_month first."
            )

    def cost_array(self, T: int) -> list[float]:
        """Variable cost as a list of length T."""
        self._require_resolved()
        if isinstance(self.variable_cost, list):
            if len(self.variable_cost) != T:
                raise ValueError(
                    f"variable_cost list length {len(self.variable_cost)} != resolution {T}"
                )
            return self.variable_cost
        return [self.variable_cost] * T

    @property
    def price_reference_value(self) -> float:
        """Absolute €/MWh reference that price_levels_pct are fractions of."""
        self._require_resolved()
        if self.price_reference is not None:
            return float(self.price_reference)
        # The validator guarantees a nonzero scalar here.
        return float(self.variable_cost)

    def resolved_grid(self, fallback: CandidateGrid) -> ResolvedGrid:
        """Resolve the tech grid (or the run-level fallback) to absolute prices.

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
        ref = self.price_reference_value
        return ResolvedGrid(
            price_levels=[p * ref for p in grid.price_levels_pct],
            mar_levels=grid.mar_levels,
            mav_fraction_levels=grid.mav_fraction_levels,
            tf_levels=grid.tf_levels,
        )

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
    # SCO clearing model:
    #   "aware" (default) — the MILP that decides acceptance/dispatch maximises
    #   the *declared* surplus sum_t (lambda_t - P^V)*q_t, exactly what
    #   EUPHEMIA sees (P^V, TF, MAV); the generator's private costs (C, C^SU)
    #   only enter afterwards when computing realised profit per scenario.
    #   "naive" — legacy model: the MILP maximises the real profit (with C and
    #   C^SU), i.e. the market is assumed to clear with perfect knowledge of
    #   private costs. Kept as an upper-bound benchmark.
    sco_model: Literal["naive", "aware"] = "aware"
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
