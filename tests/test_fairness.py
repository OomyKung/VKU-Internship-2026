"""Unit tests for Phase 2 fairness metrics."""

from __future__ import annotations

import unittest

from fim_hybrid.fairness import (
    compute_dcv,
    compute_normalized_group_spread,
    compute_soft_mf,
    compute_strict_mf,
    evaluate_fairness,
)


class FairnessTestCase(unittest.TestCase):
    """Check strict MF, diagnostic soft MF, and DCV behavior."""

    def test_normalized_group_spread_divides_by_group_size(self) -> None:
        normalized = compute_normalized_group_spread(
            group_spread={"A": 2.0, "B": 3.0},
            group_sizes={"A": 4, "B": 6},
        )

        self.assertEqual(normalized, {"A": 0.5, "B": 0.5})

    def test_strict_mf_is_zero_when_any_group_has_zero_coverage(self) -> None:
        mf = compute_strict_mf({"A": 0.2, "B": 0.0, "C": 0.3})
        self.assertEqual(mf, 0.0)

    def test_soft_mf_is_mean_of_normalized_group_spread(self) -> None:
        soft_mf = compute_soft_mf({"A": 0.2, "B": 0.0, "C": 0.4})
        self.assertAlmostEqual(soft_mf, 0.2)

    def test_dcv_is_zero_for_exact_population_proportional_targets(self) -> None:
        dcv, targets = compute_dcv(
            group_spread={"A": 1.0, "B": 1.0},
            group_sizes={"A": 2, "B": 2},
            total_spread=2.0,
        )

        self.assertEqual(targets, {"A": 1.0, "B": 1.0})
        self.assertEqual(dcv, 0.0)

    def test_dcv_is_positive_when_groups_fall_short(self) -> None:
        dcv, _ = compute_dcv(
            group_spread={"A": 2.0, "B": 0.0},
            group_sizes={"A": 2, "B": 2},
            total_spread=2.0,
        )

        self.assertGreater(dcv, 0.0)

    def test_evaluate_fairness_raises_on_unknown_group_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown groups"):
            evaluate_fairness(
                group_spread={"A": 1.0, "C": 1.0},
                group_sizes={"A": 2, "B": 2},
            )


if __name__ == "__main__":
    unittest.main()
