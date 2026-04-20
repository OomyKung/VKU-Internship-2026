"""Unit tests for Reverse Influence Sampling guidance helpers."""

from __future__ import annotations

import unittest

import networkx as nx
import pandas as pd

from fim_hybrid.data_loader import LoadedDataset, verify_protected_groups
from fim_hybrid.ris_guidance import RISConfig, generate_ris_guidance


def _toy_ris_fixture() -> tuple[LoadedDataset, object]:
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 4),
            (4, 5),
            (5, 6),
            (3, 7),
            (7, 8),
        ]
    )
    node_groups = {
        1: "A",
        2: "A",
        3: "B",
        4: "B",
        5: "C",
        6: "C",
        7: "D",
        8: "D",
    }
    for node_id, group_name in node_groups.items():
        graph.nodes[node_id]["group"] = group_name

    node_attributes = pd.DataFrame(
        [{"node_id": node_id, "group": group_name} for node_id, group_name in sorted(node_groups.items())]
    ).set_index("node_id", drop=False)
    dataset = LoadedDataset(name="toy_ris", graph=graph, node_attributes=node_attributes)
    protected_group_report = verify_protected_groups(dataset, "group")
    return dataset, protected_group_report


class RISGuidanceTestCase(unittest.TestCase):
    """Check deterministic RR-set generation and node-level RIS scoring."""

    def test_generate_ris_guidance_runs_end_to_end(self) -> None:
        dataset, protected_group_report = _toy_ris_fixture()

        ris_result = generate_ris_guidance(
            dataset=dataset,
            protected_group_report=protected_group_report,
            propagation_probability=0.5,
            config=RISConfig(num_rr_sets=32, random_seed=7, mode="global"),
        )

        self.assertEqual(len(ris_result.rr_sets), 32)
        self.assertEqual(len(ris_result.rr_root_nodes), 32)
        self.assertEqual(len(ris_result.rr_root_groups), 32)
        self.assertEqual(set(ris_result.global_node_scores), set(dataset.graph.nodes()))
        self.assertEqual(set(ris_result.node_rr_counts), set(dataset.graph.nodes()))
        self.assertTrue(all(len(rr_set) >= 1 for rr_set in ris_result.rr_sets))
        self.assertTrue(all(0.0 <= score <= 1.0 for score in ris_result.global_node_scores.values()))

    def test_generate_ris_guidance_is_reproducible(self) -> None:
        dataset, protected_group_report = _toy_ris_fixture()
        config = RISConfig(num_rr_sets=32, random_seed=7, mode="global")

        first = generate_ris_guidance(
            dataset=dataset,
            protected_group_report=protected_group_report,
            propagation_probability=0.5,
            config=config,
        )
        second = generate_ris_guidance(
            dataset=dataset,
            protected_group_report=protected_group_report,
            propagation_probability=0.5,
            config=config,
        )

        self.assertEqual(first.rr_sets, second.rr_sets)
        self.assertEqual(first.rr_root_nodes, second.rr_root_nodes)
        self.assertEqual(first.rr_root_groups, second.rr_root_groups)
        self.assertEqual(first.global_node_scores, second.global_node_scores)

    def test_weighted_node_scores_cover_all_nodes(self) -> None:
        dataset, protected_group_report = _toy_ris_fixture()
        ris_result = generate_ris_guidance(
            dataset=dataset,
            protected_group_report=protected_group_report,
            propagation_probability=0.5,
            config=RISConfig(num_rr_sets=48, random_seed=11, mode="weak_group_weighted"),
        )

        weighted_scores = ris_result.weighted_node_scores({"A": 1.0, "B": 1.5, "C": 1.0, "D": 2.0})

        self.assertEqual(set(weighted_scores), set(dataset.graph.nodes()))
        self.assertTrue(all(0.0 <= score <= 1.0 for score in weighted_scores.values()))
        self.assertEqual(set(ris_result.rr_set_counts_by_group), set(protected_group_report.group_sizes))

    def test_weighted_node_scores_change_when_group_weights_shift(self) -> None:
        dataset, protected_group_report = _toy_ris_fixture()
        ris_result = generate_ris_guidance(
            dataset=dataset,
            protected_group_report=protected_group_report,
            propagation_probability=0.5,
            config=RISConfig(num_rr_sets=48, random_seed=11, mode="weak_group_weighted"),
        )

        weighted_scores = ris_result.weighted_node_scores({"A": 1.0, "B": 1.0, "C": 1.0, "D": 3.0})

        self.assertEqual(set(weighted_scores), set(dataset.graph.nodes()))
        self.assertNotEqual(weighted_scores, ris_result.global_node_scores)


if __name__ == "__main__":
    unittest.main()
