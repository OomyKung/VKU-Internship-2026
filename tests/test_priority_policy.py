"""Tests for shared professor-priority ranking and gates."""

from __future__ import annotations

import unittest

import pandas as pd

from fim_hybrid.priority_policy import (
    ProfessorPriorityConfig,
    fairness_gate_failures,
    normalize_ranking_policy,
    professor_priority_warning,
    rank_frame_professor_priority,
)


class ProfessorPriorityPolicyTestCase(unittest.TestCase):
    def test_runtime_cannot_beat_clear_fscore_gap(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "stack_name": "slow_better_fairness",
                    "status": "ok",
                    "f_score": 0.200,
                    "mf": 0.20,
                    "dcv": 0.05,
                    "total_spread": 20.0,
                    "extra_spread": 5.0,
                    "runtime_seconds": 100.0,
                    "scalability_pass": True,
                },
                {
                    "stack_name": "fast_worse_fairness",
                    "status": "ok",
                    "f_score": 0.190,
                    "mf": 0.21,
                    "dcv": 0.04,
                    "total_spread": 25.0,
                    "extra_spread": 10.0,
                    "runtime_seconds": 1.0,
                    "scalability_pass": True,
                },
            ]
        )

        ranked = rank_frame_professor_priority(frame, ProfessorPriorityConfig(fairness_close_threshold=0.003))

        self.assertEqual(str(ranked.iloc[0]["stack_name"]), "slow_better_fairness")

    def test_fscore_is_strict_before_mf_dcv_and_runtime(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "stack_name": "fast_lower_mf",
                    "status": "ok",
                    "f_score": 0.2000,
                    "mf": 0.100,
                    "dcv": 0.05,
                    "total_spread": 30.0,
                    "runtime_seconds": 1.0,
                    "scalability_pass": True,
                },
                {
                    "stack_name": "slow_higher_mf",
                    "status": "ok",
                    "f_score": 0.1990,
                    "mf": 0.110,
                    "dcv": 0.06,
                    "total_spread": 20.0,
                    "runtime_seconds": 100.0,
                    "scalability_pass": True,
                },
            ]
        )

        ranked = rank_frame_professor_priority(frame, ProfessorPriorityConfig(fairness_close_threshold=0.003))

        self.assertEqual(str(ranked.iloc[0]["stack_name"]), "fast_lower_mf")

    def test_better_fscore_mf_and_dcv_always_wins_over_runtime(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "stack_name": "slow_better_quality",
                    "status": "ok",
                    "f_score": 0.201,
                    "mf": 0.120,
                    "dcv": 0.030,
                    "total_spread": 20.0,
                    "extra_spread": 2.0,
                    "runtime_seconds": 100.0,
                    "scalability_pass": True,
                },
                {
                    "stack_name": "fast_worse_quality",
                    "status": "ok",
                    "f_score": 0.200,
                    "mf": 0.110,
                    "dcv": 0.040,
                    "total_spread": 30.0,
                    "extra_spread": 12.0,
                    "runtime_seconds": 1.0,
                    "scalability_pass": True,
                },
            ]
        )

        ranked = rank_frame_professor_priority(frame, ProfessorPriorityConfig(fairness_close_threshold=0.003))

        self.assertEqual(str(ranked.iloc[0]["stack_name"]), "slow_better_quality")

    def test_all_failed_gates_still_rank_least_bad_with_warning(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "stack_name": "less_bad",
                    "status": "ok",
                    "f_score": -0.01,
                    "mf": 0.0,
                    "dcv": 0.40,
                    "runtime_seconds": 10.0,
                },
                {
                    "stack_name": "worse",
                    "status": "ok",
                    "f_score": -0.20,
                    "mf": 0.0,
                    "dcv": 0.60,
                    "runtime_seconds": 1.0,
                },
            ]
        )
        config = ProfessorPriorityConfig()

        ranked = rank_frame_professor_priority(frame, config)

        self.assertEqual(str(ranked.iloc[0]["stack_name"]), "less_bad")
        self.assertIsNotNone(professor_priority_warning(frame, config))
        self.assertIn("min_f_score", fairness_gate_failures(frame.iloc[0], config))

    def test_fairness_first_priority_alias_normalizes_to_professor_priority(self) -> None:
        self.assertEqual(normalize_ranking_policy("fairness_first_priority"), "professor_priority")


if __name__ == "__main__":
    unittest.main()
