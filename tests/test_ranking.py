"""Unit tests for ranking assembly."""

import pandas as pd

from bidding.ranking import build_ranking


def _result(order_type, expected_profit, objective_value=None):
    return {
        "order_type": order_type,
        "expected_profit": expected_profit,
        "cvar": expected_profit * 0.5,
        "match_probability": 0.8,
        "expected_matched_energy": 100.0,
        "expected_matched_periods": 12.0,
        "expected_profit_per_mw": expected_profit / 100.0,
        "objective_value": objective_value if objective_value is not None else expected_profit,
        "optimal_params": {"foo": "bar"},
    }


class TestBuildRanking:
    def test_sorted_by_expected_profit_descending(self):
        results = [_result("sco", 100.0), _result("simple", 300.0), _result("sbo", 200.0)]
        ranking = build_ranking(results)
        assert ranking["order_type"].tolist() == ["simple", "sbo", "sco"]

    def test_sorted_by_objective_value_when_it_disagrees_with_expected_profit(self):
        """Ranking criterion is objective_value (what each strategy optimized), not raw E[Pi]."""
        results = [
            _result("sco", 100.0, objective_value=500.0),
            _result("simple", 300.0, objective_value=10.0),
        ]
        ranking = build_ranking(results)
        assert ranking["order_type"].tolist() == ["sco", "simple"]

    def test_rank_index_starts_at_one(self):
        results = [_result("simple", 100.0)]
        ranking = build_ranking(results)
        assert ranking.index.tolist() == [1]
        assert ranking.index.name == "rank"

    def test_expected_columns_present(self):
        results = [_result("simple", 100.0)]
        ranking = build_ranking(results)
        expected_cols = {
            "order_type", "expected_profit", "cvar", "match_probability",
            "expected_matched_energy", "expected_matched_periods",
            "objective_value", "optimal_params",
        }
        assert expected_cols.issubset(set(ranking.columns))
