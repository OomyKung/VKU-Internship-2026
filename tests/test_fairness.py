"""Unit tests for fairness score variants."""

from __future__ import annotations

import unittest

from fim_hybrid.fairness import evaluate_fairness


class FairnessScoreModeTestCase(unittest.TestCase):
    """Check that fairness score modes behave as expected."""

    def test_raw_score_mode_preserves_existing_formula(self) -> None:
        metrics = evaluate_fairness(
            group_spread={"1": 2.3, "0": 1.1, "2": 1.0},
            group_sizes={"1": 300, "0": 125, "2": 75},
            lambda_weight=0.5,
            total_spread=4.4,
            score_mode="raw_mf",
        )

        self.assertEqual(metrics.score_mode, "raw_mf")
        self.assertAlmostEqual(metrics.mf_component, metrics.mf)
        self.assertAlmostEqual(metrics.combined_score, -0.017631313131313147)

    def test_normalized_score_mode_uses_mf_to_ideal_ratio(self) -> None:
        metrics = evaluate_fairness(
            group_spread={"1": 2.3, "0": 1.1, "2": 1.0},
            group_sizes={"1": 300, "0": 125, "2": 75},
            lambda_weight=0.5,
            total_spread=4.4,
            score_mode="normalized_mf",
        )

        self.assertEqual(metrics.score_mode, "normalized_mf")
        self.assertAlmostEqual(metrics.mf_component, metrics.mf_to_ideal_ratio)
        self.assertGreater(metrics.combined_score, 0.0)


if __name__ == "__main__":
    unittest.main()
