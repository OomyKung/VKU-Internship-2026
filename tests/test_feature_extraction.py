"""Unit tests for structural candidate feature extraction."""

from __future__ import annotations

import unittest

import networkx as nx
import pandas as pd

from fim_hybrid.community_detection import detect_communities
from fim_hybrid.data_loader import LoadedDataset, verify_protected_groups
from fim_hybrid.feature_extraction import compute_node_features, compute_structural_node_scores
from fim_hybrid.node2vec_embeddings import Node2VecConfig


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
        self.assertIn("normalized_degree", feature_frame.columns)
        self.assertIn("betweenness", feature_frame.columns)
        self.assertIn("clustering_coefficient", feature_frame.columns)
        self.assertIn("community_size", feature_frame.columns)
        self.assertIn("within_community_degree", feature_frame.columns)
        self.assertIn("protected_group_frequency", feature_frame.columns)
        self.assertIn("minority_group_indicator", feature_frame.columns)
        self.assertIn("neighborhood_group_entropy", feature_frame.columns)
        self.assertIn("fraction_neighbors_in_undercovered_groups", feature_frame.columns)
        self.assertFalse(feature_frame.isna().any().any())

    def test_community_feature_modes_control_explicit_ml_columns(self) -> None:
        dataset, report, community_result = _toy_feature_dataset()

        no_community = compute_node_features(
            dataset,
            report,
            community_result,
            use_community_features_for_ml=False,
        )
        basic = compute_node_features(
            dataset,
            report,
            community_result,
            community_feature_mode="basic",
        )
        full = compute_node_features(
            dataset,
            report,
            community_result,
            community_feature_mode="full",
        )

        self.assertNotIn("community_id", no_community.columns)
        self.assertNotIn("cross_community_degree", no_community.columns)
        self.assertIn("structural_score", no_community.columns)
        self.assertIn("community_id", basic.columns)
        self.assertIn("community_group_fraction_a", full.columns)
        self.assertIn("community_protected_group_entropy", full.columns)

    def test_invalid_community_feature_mode_fails_clearly(self) -> None:
        dataset, report, community_result = _toy_feature_dataset()

        with self.assertRaisesRegex(ValueError, "community_feature_mode"):
            compute_node_features(dataset, report, community_result, community_feature_mode="everything")

    def test_compute_node_features_is_deterministic(self) -> None:
        dataset, report, community_result = _toy_feature_dataset()

        first = compute_node_features(dataset, report, community_result)
        second = compute_node_features(dataset, report, community_result)

        self.assertTrue(first.equals(second))

    def test_compute_node_features_can_include_node2vec_embeddings(self) -> None:
        dataset, report, community_result = _toy_feature_dataset()

        plain_frame = compute_node_features(dataset, report, community_result)
        node2vec_frame = compute_node_features(
            dataset,
            report,
            community_result,
            node2vec_config=Node2VecConfig(
                dimensions=4,
                walk_length=6,
                num_walks=4,
                window=2,
                random_seed=7,
            ),
        )

        self.assertEqual(len(node2vec_frame), len(plain_frame))
        self.assertTrue(set(plain_frame["node_id"]) == set(node2vec_frame["node_id"]))
        self.assertEqual(len(node2vec_frame.filter(like="node2vec_").columns), 4)
        self.assertFalse(node2vec_frame.filter(like="node2vec_").isna().any().any())

    def test_feature_values_respect_basic_bounds(self) -> None:
        dataset, report, community_result = _toy_feature_dataset()

        feature_frame = compute_node_features(dataset, report, community_result)

        self.assertTrue(((feature_frame["normalized_degree"] >= 0.0) & (feature_frame["normalized_degree"] <= 1.0)).all())
        self.assertTrue(
            ((feature_frame["clustering_coefficient"] >= 0.0) & (feature_frame["clustering_coefficient"] <= 1.0)).all()
        )
        self.assertTrue(
            (
                (feature_frame["fraction_neighbors_in_undercovered_groups"] >= 0.0)
                & (feature_frame["fraction_neighbors_in_undercovered_groups"] <= 1.0)
            ).all()
        )
        self.assertTrue(
            ((feature_frame["neighborhood_group_entropy"] >= 0.0) & (feature_frame["neighborhood_group_entropy"] <= 1.0)).all()
        )

    def test_compute_structural_node_scores_returns_one_score_per_node(self) -> None:
        dataset, report, community_result = _toy_feature_dataset()
        feature_frame = compute_node_features(dataset, report, community_result)

        scores = compute_structural_node_scores(feature_frame)

        self.assertEqual(set(scores), set(dataset.graph.nodes()))
        self.assertTrue(all(score >= 0.0 for score in scores.values()))


if __name__ == "__main__":
    unittest.main()
