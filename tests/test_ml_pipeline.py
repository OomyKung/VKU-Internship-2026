"""Unit tests for ML-guided candidate selection helpers."""

from __future__ import annotations

import unittest

import networkx as nx
import pandas as pd

from fim_hybrid.community_detection import detect_communities
from fim_hybrid.data_loader import LoadedDataset, verify_protected_groups
from fim_hybrid.feature_extraction import compute_node_features
from fim_hybrid.label_generation import generate_singleton_node_utility_labels
from fim_hybrid.ml_training import select_ml_candidate_nodes, train_node_utility_model
from fim_hybrid.node2vec_embeddings import Node2VecConfig


def _toy_ml_fixture() -> tuple[LoadedDataset, object, object]:
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 4),
            (4, 5),
            (5, 6),
            (2, 7),
            (7, 8),
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
    dataset = LoadedDataset(name="toy_ml", graph=graph, node_attributes=node_attributes)
    protected_group_report = verify_protected_groups(dataset, "group")
    community_result = detect_communities(graph, method="louvain", seed=42)
    return dataset, protected_group_report, community_result


class MLLabelGenerationTestCase(unittest.TestCase):
    """Check singleton label generation on a toy graph."""

    def test_feature_table_builds_for_ml_fixture(self) -> None:
        dataset, protected_group_report, community_result = _toy_ml_fixture()

        feature_frame = compute_node_features(dataset, protected_group_report, community_result)

        self.assertEqual(len(feature_frame), dataset.graph.number_of_nodes())
        self.assertIn("betweenness", feature_frame.columns)
        self.assertIn("fraction_neighbors_in_undercovered_groups", feature_frame.columns)

    def test_singleton_labels_include_expected_columns(self) -> None:
        dataset, protected_group_report, _ = _toy_ml_fixture()

        label_result = generate_singleton_node_utility_labels(
            dataset=dataset,
            protected_group_report=protected_group_report,
            propagation_probability=1.0,
            mc_runs=3,
            lambda_weight=0.5,
            random_seed=7,
        )

        self.assertEqual(set(label_result.label_frame["node_id"]), set(dataset.graph.nodes()))
        self.assertIn("singleton_total_spread", label_result.label_frame.columns)
        self.assertIn("singleton_soft_fair_score", label_result.label_frame.columns)
        self.assertIn("label_score", label_result.label_frame.columns)
        self.assertGreater(label_result.label_variance, 0.0)

    def test_singleton_labels_raise_for_degenerate_targets(self) -> None:
        dataset, protected_group_report, _ = _toy_ml_fixture()

        with self.assertRaisesRegex(ValueError, "degenerate singleton labels"):
            generate_singleton_node_utility_labels(
                dataset=dataset,
                protected_group_report=protected_group_report,
                propagation_probability=0.0,
                mc_runs=3,
                lambda_weight=0.5,
                random_seed=7,
            )


class MLTrainingTestCase(unittest.TestCase):
    """Check model fitting, ranking, and candidate filtering behavior."""

    def test_train_node_utility_model_runs_end_to_end(self) -> None:
        dataset, protected_group_report, community_result = _toy_ml_fixture()
        feature_frame = compute_node_features(dataset, protected_group_report, community_result)
        label_result = generate_singleton_node_utility_labels(
            dataset=dataset,
            protected_group_report=protected_group_report,
            propagation_probability=1.0,
            mc_runs=3,
            lambda_weight=0.5,
            random_seed=7,
        )

        training_result = train_node_utility_model(
            feature_frame=feature_frame,
            label_frame=label_result.label_frame,
            budget=3,
            top_fraction=0.5,
            random_seed=7,
        )

        self.assertEqual(set(training_result.predicted_scores), set(dataset.graph.nodes()))
        self.assertEqual(len(training_result.ranked_nodes), dataset.graph.number_of_nodes())
        self.assertEqual(len(training_result.candidate_nodes), 4)
        self.assertTrue(training_result.validation_spearman <= 1.0)
        self.assertTrue(training_result.validation_spearman >= -1.0)
        self.assertTrue(training_result.validation_precision_at_budget >= 0.0)
        self.assertTrue(training_result.validation_precision_at_budget <= 1.0)
        self.assertEqual(training_result.model_type, "random_forest")

    def test_train_node_utility_model_runs_with_node2vec_features(self) -> None:
        dataset, protected_group_report, community_result = _toy_ml_fixture()
        feature_frame = compute_node_features(
            dataset,
            protected_group_report,
            community_result,
            node2vec_config=Node2VecConfig(
                dimensions=4,
                walk_length=6,
                num_walks=4,
                window=2,
                random_seed=7,
            ),
        )
        label_result = generate_singleton_node_utility_labels(
            dataset=dataset,
            protected_group_report=protected_group_report,
            propagation_probability=1.0,
            mc_runs=3,
            lambda_weight=0.5,
            random_seed=7,
        )

        training_result = train_node_utility_model(
            feature_frame=feature_frame,
            label_frame=label_result.label_frame,
            budget=3,
            top_fraction=0.5,
            random_seed=7,
        )

        self.assertEqual(len(feature_frame.filter(like="node2vec_").columns), 4)
        self.assertEqual(set(training_result.predicted_scores), set(dataset.graph.nodes()))
        self.assertEqual(len(training_result.ranked_nodes), dataset.graph.number_of_nodes())

    def test_train_node_utility_model_can_use_xgboost_when_available(self) -> None:
        import importlib.util  # noqa: PLC0415

        if importlib.util.find_spec("xgboost") is None:
            self.skipTest("xgboost is not installed")

        dataset, protected_group_report, community_result = _toy_ml_fixture()
        feature_frame = compute_node_features(dataset, protected_group_report, community_result)
        label_result = generate_singleton_node_utility_labels(
            dataset=dataset,
            protected_group_report=protected_group_report,
            propagation_probability=1.0,
            mc_runs=3,
            lambda_weight=0.5,
            random_seed=7,
        )

        training_result = train_node_utility_model(
            feature_frame=feature_frame,
            label_frame=label_result.label_frame,
            budget=3,
            model_type="xgboost",
            top_fraction=0.5,
            random_seed=7,
        )

        self.assertEqual(training_result.model_type, "xgboost")
        self.assertEqual(set(training_result.predicted_scores), set(dataset.graph.nodes()))

    def test_select_ml_candidate_nodes_obeys_precedence_and_budget_floor(self) -> None:
        ranked_nodes = tuple(range(1, 11))

        selected_by_fraction = select_ml_candidate_nodes(
            ranked_nodes=ranked_nodes,
            budget=3,
            top_fraction=0.2,
        )
        selected_by_n = select_ml_candidate_nodes(
            ranked_nodes=ranked_nodes,
            budget=3,
            top_fraction=0.2,
            top_n=5,
        )
        selected_with_cap = select_ml_candidate_nodes(
            ranked_nodes=ranked_nodes,
            budget=3,
            top_n=6,
            max_nodes=4,
        )

        self.assertEqual(selected_by_fraction, (1, 2, 3))
        self.assertEqual(selected_by_n, (1, 2, 3, 4, 5))
        self.assertEqual(selected_with_cap, (1, 2, 3, 4))


if __name__ == "__main__":
    unittest.main()
