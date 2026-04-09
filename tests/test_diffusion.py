"""Unit tests for Phase 2 Independent Cascade simulation."""

from __future__ import annotations

import unittest

import networkx as nx
import numpy as np
import pandas as pd

from fim_hybrid.data_loader import LoadedDataset, ProtectedGroupReport, verify_protected_groups
from fim_hybrid.diffusion import simulate_independent_cascade, simulate_independent_cascade_once


def _toy_dataset() -> tuple[LoadedDataset, ProtectedGroupReport]:
    graph = nx.DiGraph()
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_node(4)
    graph.nodes[1]["group"] = "A"
    graph.nodes[2]["group"] = "A"
    graph.nodes[3]["group"] = "B"
    graph.nodes[4]["group"] = "C"

    node_attributes = pd.DataFrame(
        [
            {"node_id": 1, "group": "A"},
            {"node_id": 2, "group": "A"},
            {"node_id": 3, "group": "B"},
            {"node_id": 4, "group": "C"},
        ]
    ).set_index("node_id", drop=False)
    dataset = LoadedDataset(name="toy", graph=graph, node_attributes=node_attributes)
    report = verify_protected_groups(dataset, "group")
    return dataset, report


class DiffusionTestCase(unittest.TestCase):
    """Check deterministic and validated Phase 2 diffusion behavior."""

    def test_simulate_once_with_probability_one_activates_reachable_chain(self) -> None:
        dataset, _ = _toy_dataset()
        rng = np.random.default_rng(42)

        active_nodes = simulate_independent_cascade_once(
            graph=dataset.graph,
            seed_set=[1],
            propagation_probability=1.0,
            rng=rng,
        )

        self.assertEqual(active_nodes, {1, 2, 3})

    def test_simulate_once_with_probability_zero_only_keeps_seed(self) -> None:
        dataset, _ = _toy_dataset()
        rng = np.random.default_rng(42)

        active_nodes = simulate_independent_cascade_once(
            graph=dataset.graph,
            seed_set=[1],
            propagation_probability=0.0,
            rng=rng,
        )

        self.assertEqual(active_nodes, {1})

    def test_monte_carlo_result_includes_zero_covered_groups(self) -> None:
        dataset, report = _toy_dataset()

        result = simulate_independent_cascade(
            dataset=dataset,
            protected_group_report=report,
            seed_set=[1],
            propagation_probability=1.0,
            mc_runs=3,
            random_seed=42,
        )

        self.assertEqual(result.group_spread_mean["A"], 2.0)
        self.assertEqual(result.group_spread_mean["B"], 1.0)
        self.assertEqual(result.group_spread_mean["C"], 0.0)
        self.assertEqual(result.group_spread_std["C"], 0.0)

    def test_duplicate_seeds_raise(self) -> None:
        dataset, report = _toy_dataset()

        with self.assertRaisesRegex(ValueError, "must not contain duplicate"):
            simulate_independent_cascade(
                dataset=dataset,
                protected_group_report=report,
                seed_set=[1, 1],
            )

    def test_missing_seed_nodes_raise(self) -> None:
        dataset, report = _toy_dataset()

        with self.assertRaisesRegex(ValueError, "not present in the graph"):
            simulate_independent_cascade(
                dataset=dataset,
                protected_group_report=report,
                seed_set=[99],
            )

    def test_repeated_calls_with_same_random_seed_match(self) -> None:
        dataset, report = _toy_dataset()
        dataset.graph[1][2]["p"] = 0.5
        dataset.graph[2][3]["p"] = 0.5

        first = simulate_independent_cascade(
            dataset=dataset,
            protected_group_report=report,
            seed_set=[1],
            propagation_probability=0.5,
            mc_runs=25,
            random_seed=7,
        )
        second = simulate_independent_cascade(
            dataset=dataset,
            protected_group_report=report,
            seed_set=[1],
            propagation_probability=0.5,
            mc_runs=25,
            random_seed=7,
        )

        self.assertEqual(first.seed_set, second.seed_set)
        self.assertEqual(first.total_spread_mean, second.total_spread_mean)
        self.assertEqual(first.total_spread_std, second.total_spread_std)
        self.assertEqual(first.group_spread_mean, second.group_spread_mean)
        self.assertEqual(first.group_spread_std, second.group_spread_std)


if __name__ == "__main__":
    unittest.main()
