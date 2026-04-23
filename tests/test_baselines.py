"""Unit tests for Phase 3 baseline methods."""

from __future__ import annotations

import unittest

import networkx as nx
import pandas as pd

from fim_hybrid.baselines import (
    available_baseline_methods,
    get_baseline_method_spec,
    run_baseline,
    select_baseline_seed_set,
)
from fim_hybrid.data_loader import LoadedDataset, ProtectedGroupReport, verify_protected_groups


def _toy_baseline_dataset() -> tuple[LoadedDataset, ProtectedGroupReport]:
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (1, 2),
            (1, 3),
            (1, 4),
            (5, 6),
            (5, 7),
            (5, 8),
        ]
    )
    for node_id, group_name in {
        1: "A",
        2: "A",
        3: "B",
        4: "B",
        5: "C",
        6: "C",
        7: "D",
        8: "D",
    }.items():
        graph.nodes[node_id]["group"] = group_name

    node_attributes = pd.DataFrame(
        [{"node_id": node_id, "group": graph.nodes[node_id]["group"]} for node_id in sorted(graph.nodes())]
    ).set_index("node_id", drop=False)
    dataset = LoadedDataset(name="toy_baselines", graph=graph, node_attributes=node_attributes)
    report = verify_protected_groups(dataset, "group")
    return dataset, report


class BaselineTestCase(unittest.TestCase):
    """Check Phase 3 baseline selection and end-to-end evaluation."""

    def test_select_baseline_seed_set_returns_deterministic_selection_only(self) -> None:
        dataset, _ = _toy_baseline_dataset()

        seed_set = select_baseline_seed_set(
            dataset=dataset,
            method="degree",
            budget=2,
            random_seed=7,
        )

        self.assertEqual(seed_set, (1, 5))

    def test_baseline_registry_lists_new_greedy_methods(self) -> None:
        self.assertEqual(
            available_baseline_methods(),
            (
                "random",
                "degree",
                "pagerank",
                "greedy",
                "fairness_weighted_greedy",
                "maximin_greedy",
                "community_round_robin",
            ),
        )
        self.assertTrue(get_baseline_method_spec("greedy").requires_shared_evaluation)
        self.assertTrue(get_baseline_method_spec("community_round_robin").uses_community_assignments)

    def test_random_baseline_is_deterministic_for_same_seed(self) -> None:
        dataset, report = _toy_baseline_dataset()

        first = run_baseline(
            dataset=dataset,
            protected_group_report=report,
            method="random",
            budget=3,
            propagation_probability=0.0,
            mc_runs=5,
            random_seed=11,
            diffusion_model="ic",
        )
        second = run_baseline(
            dataset=dataset,
            protected_group_report=report,
            method="random",
            budget=3,
            propagation_probability=0.0,
            mc_runs=5,
            random_seed=11,
            diffusion_model="ic",
        )

        self.assertEqual(first.seed_set, second.seed_set)
        self.assertEqual(len(first.seed_set), 3)
        self.assertEqual(len(set(first.seed_set)), 3)
        self.assertEqual(first.total_spread_mean, 3.0)

    def test_degree_baseline_selects_highest_degree_nodes(self) -> None:
        dataset, report = _toy_baseline_dataset()

        result = run_baseline(
            dataset=dataset,
            protected_group_report=report,
            method="degree",
            budget=2,
            propagation_probability=0.0,
            mc_runs=5,
            random_seed=7,
        )

        self.assertEqual(result.seed_set, (1, 5))
        self.assertEqual(result.total_spread_mean, 2.0)

    def test_pagerank_baseline_selects_highest_pagerank_nodes(self) -> None:
        dataset, report = _toy_baseline_dataset()

        result = run_baseline(
            dataset=dataset,
            protected_group_report=report,
            method="pagerank",
            budget=2,
            propagation_probability=0.0,
            mc_runs=5,
            random_seed=7,
        )

        self.assertEqual(result.seed_set, (1, 5))
        self.assertEqual(result.total_spread_mean, 2.0)

    def test_community_round_robin_covers_multiple_communities(self) -> None:
        dataset, report = _toy_baseline_dataset()

        result = run_baseline(
            dataset=dataset,
            protected_group_report=report,
            method="community_round_robin",
            budget=2,
            propagation_probability=0.0,
            mc_runs=5,
            random_seed=7,
        )

        self.assertEqual(result.seed_set, (1, 5))
        self.assertEqual(result.total_spread_mean, 2.0)
        self.assertEqual(result.mf, 0.0)
        self.assertEqual(result.f_score, -0.25)

    def test_greedy_baseline_runs_through_shared_evaluation(self) -> None:
        dataset, report = _toy_baseline_dataset()

        result = run_baseline(
            dataset=dataset,
            protected_group_report=report,
            method="greedy",
            budget=2,
            propagation_probability=1.0,
            mc_runs=3,
            random_seed=7,
        )

        self.assertEqual(result.seed_set, (1, 5))
        self.assertGreaterEqual(result.total_spread_mean, 8.0)

    def test_fairness_weighted_greedy_baseline_is_supported(self) -> None:
        dataset, report = _toy_baseline_dataset()

        result = run_baseline(
            dataset=dataset,
            protected_group_report=report,
            method="fairness_weighted_greedy",
            budget=2,
            propagation_probability=1.0,
            mc_runs=3,
            random_seed=7,
        )

        self.assertEqual(len(result.seed_set), 2)
        self.assertGreaterEqual(result.f_score, 0.0)

    def test_maximin_greedy_baseline_is_supported(self) -> None:
        dataset, report = _toy_baseline_dataset()

        result = run_baseline(
            dataset=dataset,
            protected_group_report=report,
            method="maximin_greedy",
            budget=2,
            propagation_probability=1.0,
            mc_runs=3,
            random_seed=7,
        )

        self.assertEqual(len(result.seed_set), 2)
        self.assertGreaterEqual(result.mf, 0.0)

    def test_unknown_baseline_method_raises(self) -> None:
        dataset, report = _toy_baseline_dataset()

        with self.assertRaisesRegex(ValueError, "Unsupported baseline method"):
            run_baseline(
                dataset=dataset,
                protected_group_report=report,
                method="not_a_method",
                budget=2,
            )

    def test_baseline_supports_linear_threshold(self) -> None:
        dataset, report = _toy_baseline_dataset()

        result = run_baseline(
            dataset=dataset,
            protected_group_report=report,
            method="degree",
            budget=2,
            propagation_probability=1.0,
            mc_runs=3,
            diffusion_model="lt",
        )

        self.assertEqual(result.method, "degree")
        self.assertGreaterEqual(result.total_spread_mean, 2.0)

    def test_baseline_rejects_unknown_diffusion_model(self) -> None:
        dataset, report = _toy_baseline_dataset()

        with self.assertRaisesRegex(ValueError, "Unsupported diffusion_model"):
            run_baseline(
                dataset=dataset,
                protected_group_report=report,
                method="degree",
                budget=2,
                diffusion_model="not_a_model",
            )


if __name__ == "__main__":
    unittest.main()
