"""Unit tests for the beta risk-return frontier sweep."""

import numpy as np

from bidding.config import ResolvedGrid
from reporting.frontier import sweep_beta

from .conftest import avail_matrix, make_tech


class TestSweepBetaMonotonicity:
    def test_cvar_non_decreasing_as_beta_increases(self):
        """As beta grows, the weighted objective favours strategies with a
        higher achieved CVaR (weakly), trading off expected profit — the
        standard weighted-sum efficient-frontier property."""
        tech = make_tech([100.0] * 24, cost=0.0)
        # A mix of very good, mediocre, and disastrous price scenarios so a
        # higher block price captures more upside but risks a large loss.
        lam = np.array([
            [80.0] * 24,
            [80.0] * 24,
            [80.0] * 24,
            [5.0] * 24,   # tail scenario: a high block price would reject here anyway
        ])
        probs = np.ones(4) / 4
        grid = ResolvedGrid(
            price_levels=[0, 10, 30, 50, 70],
            mar_levels=[0.5],
            mav_fraction_levels=[0.5],
            tf_levels=[0],
        )
        betas = [0.0, 0.25, 0.5, 0.75, 1.0]
        df = sweep_beta(tech, lam, avail_matrix(tech, 4), probs, grid, 0.95, "sbo", betas)

        cvars = df.sort_values("beta")["cvar"].tolist()
        for earlier, later in zip(cvars, cvars[1:]):
            assert later >= earlier - 1e-6

    def test_returns_one_row_per_beta(self):
        tech = make_tech([100.0] * 24, cost=0.0)
        lam = np.full((3, 24), 50.0)
        probs = np.ones(3) / 3
        grid = ResolvedGrid(price_levels=[0, 40], mar_levels=[0.5], mav_fraction_levels=[0.5], tf_levels=[0])
        betas = [0.0, 0.5, 1.0]
        df = sweep_beta(tech, lam, avail_matrix(tech, 3), probs, grid, 0.95, "sbo", betas)

        assert df["beta"].tolist() == betas
        assert set(df.columns) == {"order_type", "beta", "expected_profit", "cvar", "objective_value"}
