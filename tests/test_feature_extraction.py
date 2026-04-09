"""Unit tests for structural candidate feature extraction."""

from __future__ import annotations

import unittest

import networkx as nx
import pandas as pd

from fim_hybrid.community_detection import detect_communities
from fim_hybrid.data_loader import LoadedDataset, verify_protected_groups
from fim_hybrid.feature_extraction import compute_node_features, compute_structural_node_scores


def _toy_feature_dataset() -> tuple[LoadedDataset, object, object]:
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 4),
            (4, 5),
            (4, 6),
            (5, 6),
        ]
    )
    for node_id, group_name in {
        1: "A",
        2: "A",
        3: "B",
        4: "B",
        5: "C",
        6: "C",
    }.items():
        graph.nodes[node_id]["group"] = group_name

    node_attributes = pd.DataFrame(
        [{"node_id": node_id, "group": graph.nodes[node_id]["group"]} for node_id in sorted(graph.nodes())]
    ).set_index("node_id", drop=False)
    dataset = LoadedDataset(name="toy_features", graph=graph, node_attributes=node_attributes)
    report = verify_protected_groups(dataset, "group")
    community_result = detect_communities(graph, method="louvain", seed=7)
    return dataset, report, community_result


class FeatureExtractionTestCase(unittest.TestCase):
    """Check deterministic structural candidate features."""

    def test_compute_node_features_includes_expected_columns(self) -> None:
        dataset, report, community_result = _toy_feature_dataset()

        feature_frame = compute_node_features(dataset, report, community_result)

        self.assertEqual(set(feature_frame["node_id"]), set(dataset.graph.nodes()))
        self.assertIn("structural_score", feature_frame.columns)
        self.assertIn("cross_community_degree", feature_frame.columns)
        self.assertIn("protected_group", feature_frame.columns)

    def test_compute_structural_node_scores_returns_one_score_per_node(self) -> None:
        dataset, report, community_result = _toy_feature_dataset()
        feature_frame = compute_node_features(dataset, report, community_result)

        scores = compute_structural_node_scores(feature_frame)

        self.assertEqual(set(scores), set(dataset.graph.nodes()))
        self.assertTrue(all(score >= 0.0 for score in scores.values()))


if __name__ == "__main__":
    unittest.main()
