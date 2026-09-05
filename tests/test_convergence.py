"""Unit tests for the theta grid-convergence test (reporting.convergence)."""

import numpy as np
import pandas as pd

from bidding.config import CandidateGrid, ResolvedGrid, RiskObjective
from bidding.metrics import objective_value
from bidding.orders.sbo import SBOStrategy
from bidding.orders.sco import SCOStrategy
from reporting.convergence import (
    n_combos,
    refine_grid,
    refine_levels,
    summarize_convergence,
)

from .conftest import avail_matrix, make_tech


class TestRefineLevels:
    def test_factor_one_is_identity_sorted_dedup(self):
        assert refine_levels([1.0, 0.5, 1.0], 1) == [0.5, 1.0]

    def test_endpoints_preserved_and_length(self):
        base = [0.7, 1.0, 1.06]
        for factor in (2, 4, 8):
            fine = refine_levels(base, factor)
            assert len(fine) == factor * (len(base) - 1) + 1
            assert fine[0] == base[0] and fine[-1] == base[-1]
            assert fine == sorted(fine)

    def test_nesting_when_factors_divide(self):
        base = [0.7059, 0.8235, 0.9765, 1.0, 1.0235, 1.0588]
        f1 = refine_levels(base, 1)
        f2 = refine_levels(base, 2)
        f4 = refine_levels(base, 4)
        f8 = refine_levels(base, 8)
        # Rounded to 9 decimals inside refine_levels, so set inclusion is exact.
        assert set(f1) <= set(f2) <= set(f4) <= set(f8)

    def test_nesting_with_large_absolute_levels(self):
        tf = [0, 10000, 20000, 55000, 60000, 65000, 70000, 80000, 90000]
        assert set(refine_levels(tf, 2)) <= set(refine_levels(tf, 8))


class TestRefineGrid:
    BASE = CandidateGrid(
        price_levels_pct=[0.7, 1.0, 1.06],
        mar_levels=[0.3, 0.5, 1.0],
        mav_fraction_levels=[0.3, 1.0],
        tf_levels=[0, 60000, 90000],
    )

    def test_applies_to_all_four_lists(self):
        g = refine_grid(self.BASE, 2)
        assert len(g.price_levels_pct) == 5
        assert len(g.mar_levels) == 5
        assert len(g.mav_fraction_levels) == 3
        assert len(g.tf_levels) == 5

    def test_no_refine_mav_keeps_mav_coarse(self):
        g = refine_grid(self.BASE, 4, refine_mav=False)
        assert g.mav_fraction_levels == [0.3, 1.0]
        assert len(g.tf_levels) == 9

    def test_n_combos(self):
        g = refine_grid(self.BASE, 2)
        assert n_combos(g, "simple") == 5
        assert n_combos(g, "sbo") == 5 * 5
        assert n_combos(g, "sco") == 5 * 5 * 3

    def test_refined_grid_respects_technical_min_floor(self):
        """MAR/MAV levels >= technical_min/capacity stay above the floor after
        bisection, so resolved_grid() must not raise."""
        tech = make_tech(
            [400.0] * 24,
            cost=85.0,
            startup=60000.0,
            technical_min=120.0,
            enforce_technical_constraints=True,
        )
        tech_f = tech.model_copy(update={"candidate_grid": refine_grid(self.BASE, 4)})
        resolved = tech_f.resolved_grid(CandidateGrid())
        assert min(resolved.mar_levels) >= 120.0 / 400.0 - 1e-9
        assert min(resolved.mav_fraction_levels) >= 120.0 / 400.0 - 1e-9


class TestObjectiveMonotoneWithNestedGrid:
    """With nested grids the exhaustive argmax sees a superset of candidates,
    so the per-month objective is monotone non-decreasing in the factor."""

    def _setup(self):
        tech = make_tech([100.0] * 24, cost=40.0, startup=2000.0)
        rng = np.random.default_rng(7)
        lam = rng.uniform(0.0, 120.0, size=(5, 24))
        probs = np.ones(5) / 5
        return tech, lam, avail_matrix(tech, 5), probs

    def _objective(self, strategy, tech, lam, avail, probs, grid, objective):
        result = strategy.evaluate(
            tech=tech,
            lambda_matrix=lam,
            avail_matrix=avail,
            probs=probs,
            grid=grid,
            objective=objective,
            cvar_alpha=0.95,
        )
        return objective_value(result["profits"], probs, 0.95, objective)

    def test_sbo_and_sco_objectives_non_decreasing(self):
        tech, lam, avail, probs = self._setup()
        objective = RiskObjective(beta=0.5)
        price_base = [0.0, 40.0, 80.0, 120.0]
        tf_base = [0.0, 500.0, 2000.0]
        mar_base = [0.0, 0.5, 1.0]
        mav_base = [0.0, 1.0]

        for strategy in (SBOStrategy(), SCOStrategy()):
            objs = []
            for factor in (1, 2, 4):
                grid = ResolvedGrid(
                    price_levels=refine_levels(price_base, factor),
                    mar_levels=refine_levels(mar_base, factor),
                    mav_fraction_levels=refine_levels(mav_base, factor),
                    tf_levels=refine_levels(tf_base, factor),
                )
                objs.append(self._objective(strategy, tech, lam, avail, probs, grid, objective))
            for earlier, later in zip(objs, objs[1:]):
                assert later >= earlier - 1e-9, f"{strategy.order_type}: {objs}"


class TestSummarizeConvergence:
    @staticmethod
    def _summary(rows):
        return pd.DataFrame(rows)

    def test_detects_winner_change(self):
        df = self._summary([
            {"factor": 1, "order_type": "sco", "rank": 1, "objective_value": 100.0,
             "delta_obj_rel": np.nan, "theta_stable": None},
            {"factor": 1, "order_type": "sbo", "rank": 2, "objective_value": 90.0,
             "delta_obj_rel": np.nan, "theta_stable": None},
            {"factor": 2, "order_type": "sbo", "rank": 1, "objective_value": 105.0,
             "delta_obj_rel": 0.16, "theta_stable": True},
            {"factor": 2, "order_type": "sco", "rank": 2, "objective_value": 100.0,
             "delta_obj_rel": 0.0, "theta_stable": True},
        ])
        verdict = summarize_convergence(df, tol=1e-3)
        assert verdict["winner_stable"] is False
        assert verdict["converged_at_factor"] is None
        assert verdict["winner_by_factor"] == {1: "sco", 2: "sbo"}

    def test_converges_when_winner_obj_and_theta_stable(self):
        df = self._summary([
            {"factor": 1, "order_type": "sco", "rank": 1, "objective_value": 100.0,
             "delta_obj_rel": np.nan, "theta_stable": None},
            {"factor": 2, "order_type": "sco", "rank": 1, "objective_value": 100.5,
             "delta_obj_rel": 5e-3, "theta_stable": False},
            {"factor": 4, "order_type": "sco", "rank": 1, "objective_value": 100.5,
             "delta_obj_rel": 0.0, "theta_stable": True},
        ])
        verdict = summarize_convergence(df, tol=1e-3)
        assert verdict["winner_stable"] is True
        assert verdict["converged_at_factor"] == 4
        assert verdict["obj_converged"] is True
        assert verdict["theta_stable"] is True
