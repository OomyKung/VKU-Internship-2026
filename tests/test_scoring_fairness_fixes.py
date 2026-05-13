"""Unit tests for F-score formula, group bonus/penalty scoring, and feasible ideal influences.

Covers:
  - compute_f_score: shortfall-only formula, boundary values, error handling
  - _combined_guidance_score_frame over-served penalty: positive penalty for high-mean-score groups
  - _combined_guidance_score_frame ideal-influence bonus: monotonicity with respect to ideal share
  - compute_ideal_influences_feasible: non-negativity, ceiling_factor=0 collapses to zero,
    feasible ≤ proportional, all groups present
"""

from __future__ import annotations

import unittest

import networkx as nx
import pandas as pd

from fim_hybrid.data_loader import LoadedDataset, ProtectedGroupReport, verify_protected_groups
from fim_hybrid.evaluation import (
    compute_f_score,
    compute_ideal_influences_feasible,
    compute_ideal_influences_proportional,
)
from fim_hybrid.permutations import (
    _combined_guidance_score_frame,
    get_fim_permutation_spec,
)


# ---------------------------------------------------------------------------
# Shared toy fixtures
# ---------------------------------------------------------------------------

def _two_group_dataset() -> tuple[LoadedDataset, ProtectedGroupReport]:
    """Toy graph: 'large' (4 nodes, dense clique) and 'small' (2 nodes, single edge)."""
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])  # clique
    graph.add_edge(5, 6)
    groups = {1: "large", 2: "large", 3: "large", 4: "large", 5: "small", 6: "small"}
    for node_id, group in groups.items():
        graph.nodes[node_id]["group"] = group
    attrs = pd.DataFrame(
        [{"node_id": nid, "group": g} for nid, g in sorted(groups.items())]
    ).set_index("node_id", drop=False)
    dataset = LoadedDataset(name="toy_two_group", graph=graph, node_attributes=attrs)
    return dataset, verify_protected_groups(dataset, "group")


def _score_frame(ranking_scores, report, **kwargs):
    spec = get_fim_permutation_spec("community_fair_greedy_baseline")
    return _combined_guidance_score_frame(
        spec,
        ranking_scores,
        None,
        None,
        protected_group_report=report,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. F-score formula
# ---------------------------------------------------------------------------

class FScoreFormulaTestCase(unittest.TestCase):
    """Verify compute_f_score uses MF - DCV."""

    def test_shortfall_formula(self) -> None:
        result = compute_f_score(mf=0.8, dcv=0.2, lambda_weight=0.5)
        self.assertAlmostEqual(result, 0.8 - 0.2)

    def test_lambda_zero_keeps_shortfall_formula(self) -> None:
        result = compute_f_score(mf=1.0, dcv=0.4, lambda_weight=0.0)
        self.assertAlmostEqual(result, 0.6)

    def test_lambda_one_keeps_shortfall_formula(self) -> None:
        result = compute_f_score(mf=0.7, dcv=0.9, lambda_weight=1.0)
        self.assertAlmostEqual(result, -0.2)

    def test_zero_dcv_returns_mf(self) -> None:
        result = compute_f_score(mf=0.6, dcv=0.0, lambda_weight=0.5)
        self.assertAlmostEqual(result, 0.6)

    def test_explicit_dcv_shortfall_is_primary_dcv(self) -> None:
        result = compute_f_score(
            mf=0.8,
            dcv=0.2,
            lambda_weight=0.5,
            dcv_shortfall=0.1,
            shortfall_dcv_weight=1.0,
        )
        self.assertAlmostEqual(result, 0.8 - 0.1)

    def test_zero_shortfall_returns_mf(self) -> None:
        result = compute_f_score(
            mf=0.6,
            dcv=0.4,
            lambda_weight=0.5,
            dcv_shortfall=0.0,
            shortfall_dcv_weight=1.0,
        )
        self.assertAlmostEqual(result, 0.6)

    def test_none_dcv_shortfall_falls_back_to_dcv(self) -> None:
        result = compute_f_score(mf=0.6, dcv=0.2, lambda_weight=0.5, dcv_shortfall=None)
        self.assertAlmostEqual(result, 0.4)

    def test_invalid_lambda_above_one_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_f_score(mf=0.5, dcv=0.1, lambda_weight=1.1)

    def test_invalid_lambda_below_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_f_score(mf=0.5, dcv=0.1, lambda_weight=-0.01)

    def test_shortfall_weight_zero_returns_mf(self) -> None:
        result = compute_f_score(
            mf=0.7,
            dcv=0.3,
            lambda_weight=0.5,
            dcv_shortfall=0.5,
            shortfall_dcv_weight=0.0,
        )
        self.assertAlmostEqual(result, 0.7)


# ---------------------------------------------------------------------------
# 2. Over-served group penalty
# ---------------------------------------------------------------------------

class OverServedPenaltyTestCase(unittest.TestCase):
    """Verify over_served_group_penalty is positive for high-mean-score groups."""

    def test_over_served_group_gets_positive_penalty(self) -> None:
        _, report = _two_group_dataset()
        # large group has uniformly high ML scores → over-served
        ranking_scores = {1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.1, 6: 0.1}
        frame = _score_frame(
            ranking_scores,
            report,
            use_over_served_group_penalty=True,
            over_served_penalty_weight=2.0,
            under_served_bonus_weight=1.5,
            parity_tolerance=0.0,
        )
        large_penalty = frame.loc[frame["candidate_group"] == "large", "over_served_group_penalty"].mean()
        self.assertGreater(large_penalty, 0.0, "over-served group should have positive penalty")

    def test_under_served_group_has_zero_penalty(self) -> None:
        _, report = _two_group_dataset()
        ranking_scores = {1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.1, 6: 0.1}
        frame = _score_frame(
            ranking_scores,
            report,
            use_over_served_group_penalty=True,
            over_served_penalty_weight=2.0,
            under_served_bonus_weight=1.5,
            parity_tolerance=0.0,
        )
        small_penalty = frame.loc[frame["candidate_group"] == "small", "over_served_group_penalty"].mean()
        self.assertEqual(small_penalty, 0.0, "under-served group should have zero penalty")

    def test_under_served_group_gets_positive_bonus(self) -> None:
        _, report = _two_group_dataset()
        ranking_scores = {1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.1, 6: 0.1}
        frame = _score_frame(
            ranking_scores,
            report,
            use_over_served_group_penalty=True,
            over_served_penalty_weight=2.0,
            under_served_bonus_weight=1.5,
            parity_tolerance=0.0,
        )
        small_bonus = frame.loc[frame["candidate_group"] == "small", "under_served_group_bonus"].mean()
        self.assertGreater(small_bonus, 0.0, "under-served group should have positive bonus")

    def test_over_served_group_has_zero_bonus(self) -> None:
        _, report = _two_group_dataset()
        ranking_scores = {1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.1, 6: 0.1}
        frame = _score_frame(
            ranking_scores,
            report,
            use_over_served_group_penalty=True,
            over_served_penalty_weight=2.0,
            under_served_bonus_weight=1.5,
            parity_tolerance=0.0,
        )
        large_bonus = frame.loc[frame["candidate_group"] == "large", "under_served_group_bonus"].mean()
        self.assertEqual(large_bonus, 0.0, "over-served group should have zero bonus")

    def test_no_penalty_when_flag_disabled(self) -> None:
        _, report = _two_group_dataset()
        ranking_scores = {1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.1, 6: 0.1}
        frame = _score_frame(ranking_scores, report, use_over_served_group_penalty=False)
        self.assertTrue((frame["over_served_group_penalty"] == 0.0).all())
        self.assertTrue((frame["under_served_group_bonus"] == 0.0).all())

    def test_uniform_scores_no_penalty_no_bonus(self) -> None:
        # All groups same mean → neither penalty nor bonus
        _, report = _two_group_dataset()
        ranking_scores = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.5}
        frame = _score_frame(
            ranking_scores,
            report,
            use_over_served_group_penalty=True,
            over_served_penalty_weight=2.0,
            under_served_bonus_weight=1.5,
            parity_tolerance=0.0,
        )
        self.assertTrue((frame["over_served_group_penalty"] == 0.0).all())
        self.assertTrue((frame["under_served_group_bonus"] == 0.0).all())


# ---------------------------------------------------------------------------
# 3. Ideal-influence group bonus monotonicity
# ---------------------------------------------------------------------------

class IdealInfluenceGroupBonusTestCase(unittest.TestCase):
    """Verify use_ideal_influence_group_bonus assigns bonuses proportional to ideal share."""

    def _frame_with_ideals(self, ideal_influences, under_served_bonus_weight=1.5):
        _, report = _two_group_dataset()
        ranking_scores = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.5}
        return _score_frame(
            ranking_scores,
            report,
            ideal_influences=ideal_influences,
            use_ideal_influence_group_bonus=True,
            under_served_bonus_weight=under_served_bonus_weight,
        )

    def test_higher_ideal_gives_higher_bonus(self) -> None:
        frame = self._frame_with_ideals({"large": 10.0, "small": 2.0})
        large_bonus = frame.loc[frame["candidate_group"] == "large", "under_served_group_bonus"].mean()
        small_bonus = frame.loc[frame["candidate_group"] == "small", "under_served_group_bonus"].mean()
        self.assertGreater(large_bonus, small_bonus)

    def test_bonus_proportional_to_ideal_share(self) -> None:
        # bonus = weight * (ideal_g / total_ideal); total = 8+2 = 10
        frame = self._frame_with_ideals({"large": 8.0, "small": 2.0}, under_served_bonus_weight=1.0)
        large_bonus = frame.loc[frame["candidate_group"] == "large", "under_served_group_bonus"].iloc[0]
        small_bonus = frame.loc[frame["candidate_group"] == "small", "under_served_group_bonus"].iloc[0]
        self.assertAlmostEqual(large_bonus, 8.0 / 10.0, places=6)
        self.assertAlmostEqual(small_bonus, 2.0 / 10.0, places=6)

    def test_bonus_scales_with_weight(self) -> None:
        frame_w1 = self._frame_with_ideals({"large": 6.0, "small": 4.0}, under_served_bonus_weight=1.0)
        frame_w2 = self._frame_with_ideals({"large": 6.0, "small": 4.0}, under_served_bonus_weight=2.0)
        b1 = frame_w1.loc[frame_w1["candidate_group"] == "large", "under_served_group_bonus"].iloc[0]
        b2 = frame_w2.loc[frame_w2["candidate_group"] == "large", "under_served_group_bonus"].iloc[0]
        self.assertAlmostEqual(b2, 2.0 * b1, places=6)

    def test_equal_ideals_give_equal_bonuses(self) -> None:
        frame = self._frame_with_ideals({"large": 5.0, "small": 5.0})
        large_bonus = frame.loc[frame["candidate_group"] == "large", "under_served_group_bonus"].mean()
        small_bonus = frame.loc[frame["candidate_group"] == "small", "under_served_group_bonus"].mean()
        self.assertAlmostEqual(large_bonus, small_bonus, places=6)

    def test_no_over_served_penalty_in_ideal_mode(self) -> None:
        # ideal-influence mode always sets over_served_group_penalty = 0
        frame = self._frame_with_ideals({"large": 10.0, "small": 2.0})
        self.assertTrue((frame["over_served_group_penalty"] == 0.0).all())

    def test_monotonicity_with_three_levels(self) -> None:
        # Build a three-group graph to check strict ordering
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (3, 4), (5, 6)])
        node_groups = {1: "high", 2: "high", 3: "mid", 4: "mid", 5: "low", 6: "low"}
        for nid, g in node_groups.items():
            graph.nodes[nid]["group"] = g
        attrs = pd.DataFrame(
            [{"node_id": nid, "group": g} for nid, g in sorted(node_groups.items())]
        ).set_index("node_id", drop=False)
        dataset = LoadedDataset(name="three_group", graph=graph, node_attributes=attrs)
        report = verify_protected_groups(dataset, "group")

        spec = get_fim_permutation_spec("community_fair_greedy_baseline")
        ranking_scores = {nid: 0.5 for nid in graph.nodes()}
        ideal_influences = {"high": 9.0, "mid": 3.0, "low": 1.0}
        frame = _combined_guidance_score_frame(
            spec, ranking_scores, None, None,
            protected_group_report=report,
            ideal_influences=ideal_influences,
            use_ideal_influence_group_bonus=True,
            under_served_bonus_weight=1.0,
        )
        bonus_high = frame.loc[frame["candidate_group"] == "high", "under_served_group_bonus"].iloc[0]
        bonus_mid = frame.loc[frame["candidate_group"] == "mid", "under_served_group_bonus"].iloc[0]
        bonus_low = frame.loc[frame["candidate_group"] == "low", "under_served_group_bonus"].iloc[0]
        self.assertGreater(bonus_high, bonus_mid)
        self.assertGreater(bonus_mid, bonus_low)


# ---------------------------------------------------------------------------
# 4. Feasible ideal influences
# ---------------------------------------------------------------------------

class FeasibleIdealInfluencesTestCase(unittest.TestCase):
    """Verify compute_ideal_influences_feasible satisfies structural constraints."""

    def _feasible(self, dataset, report, **kwargs):
        return compute_ideal_influences_feasible(
            dataset=dataset,
            protected_group_report=report,
            budget=3,
            propagation_probability=0.0,  # deterministic: spread = seed count
            mc_runs=5,
            random_seed=42,
            **kwargs,
        )

    def test_all_values_nonnegative(self) -> None:
        dataset, report = _two_group_dataset()
        ideals = self._feasible(dataset, report)
        for group, value in ideals.items():
            self.assertGreaterEqual(value, 0.0, f"ideal for '{group}' is negative")

    def test_all_groups_have_entry(self) -> None:
        dataset, report = _two_group_dataset()
        ideals = self._feasible(dataset, report)
        for group in report.group_sizes:
            self.assertIn(group, ideals)

    def test_ceiling_factor_zero_gives_zero_ideals(self) -> None:
        # min(proportional, ceiling * 0.0) = 0 for every group
        dataset, report = _two_group_dataset()
        ideals = self._feasible(dataset, report, ceiling_factor=0.0)
        for group, value in ideals.items():
            self.assertAlmostEqual(value, 0.0, msg=f"ceiling_factor=0 should yield 0 for '{group}'")

    def test_feasible_le_proportional(self) -> None:
        # Feasible mode takes min(), so it never exceeds the proportional ideal.
        # With p=0.0 both pass1 and pass2 are deterministic seed counts.
        dataset, report = _two_group_dataset()
        proportional = compute_ideal_influences_proportional(
            dataset=dataset,
            protected_group_report=report,
            budget=3,
            propagation_probability=0.0,
            mc_runs=5,
            random_seed=42,
        )
        feasible = self._feasible(dataset, report, ceiling_factor=0.95)
        for group in report.group_sizes:
            self.assertLessEqual(
                feasible[group],
                proportional[group] + 1e-9,
                f"feasible ideal for '{group}' exceeds proportional ideal",
            )

    def test_feasible_le_group_size(self) -> None:
        # No ideal can exceed the group's node count
        dataset, report = _two_group_dataset()
        ideals = self._feasible(dataset, report, ceiling_factor=1.0)
        for group, value in ideals.items():
            self.assertLessEqual(value, float(report.group_sizes[group]) + 1e-9)

    def test_ceiling_factor_one_is_at_least_ceiling_factor_half(self) -> None:
        # Higher ceiling_factor → ideal ≥ ideal with lower ceiling_factor
        dataset, report = _two_group_dataset()
        ideals_half = self._feasible(dataset, report, ceiling_factor=0.5)
        ideals_one = self._feasible(dataset, report, ceiling_factor=1.0)
        for group in report.group_sizes:
            self.assertGreaterEqual(
                ideals_one[group],
                ideals_half[group] - 1e-9,
                f"higher ceiling_factor should give ≥ ideal for '{group}'",
            )

    def test_isolated_group_deterministic(self) -> None:
        # With p=0.0, spread = number of seeds; feasible = min(k_g, k_ceil) * cf
        dataset, report = _two_group_dataset()
        # large: k_g = ceil(3*4/6) = 2, k_ceil = min(3,4) = 3, cf=0.95
        # feasible = min(2.0, 3.0*0.95) = min(2.0, 2.85) = 2.0
        # small: k_g = ceil(3*2/6) = 1, k_ceil = min(3,2) = 2, cf=0.95
        # feasible = min(1.0, 2.0*0.95) = min(1.0, 1.9) = 1.0
        ideals = self._feasible(dataset, report, ceiling_factor=0.95)
        self.assertAlmostEqual(ideals["large"], 2.0, places=6)
        self.assertAlmostEqual(ideals["small"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
