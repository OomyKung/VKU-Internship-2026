"""Unit tests for ML-guided candidate selection helpers."""

from __future__ import annotations

from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

import networkx as nx
import pandas as pd
from sklearn.neural_network import MLPRegressor as SklearnMLPRegressor

from fim_hybrid.community_detection import detect_communities
from fim_hybrid.data_loader import LoadedDataset, verify_protected_groups
from fim_hybrid.experiment_runner import ExperimentSettings, _build_gnn_label_frame
from fim_hybrid.feature_extraction import compute_node_features
from fim_hybrid.gnn_training import gnn_dependencies_available, train_gnn_node_utility_model
from fim_hybrid.label_generation import generate_singleton_node_utility_labels
from fim_hybrid.ml_training import (
    available_ranking_models,
    get_ranking_model_spec,
    select_ml_candidate_nodes,
    train_ranking_model as train_dispatch_ranking_model,
    train_node_utility_model,
)
from fim_hybrid.node2vec_embeddings import Node2VecConfig
from fim_hybrid.stack_pipeline import build_ranking_feature_frame, prepare_optional_clustering


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = REPO_ROOT / ".test-artifacts"
TEST_TMP_ROOT.mkdir(exist_ok=True)


class _WorkspaceScratchDir:
    def __init__(self) -> None:
        self.path = TEST_TMP_ROOT / f"scratch_{uuid4().hex}"

    def __enter__(self) -> str:
        self.path.mkdir(parents=True, exist_ok=False)
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def _workspace_tempdir() -> _WorkspaceScratchDir:
    return _WorkspaceScratchDir()


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
        self.assertIn("singleton_weak_group_gain", label_result.label_frame.columns)
        self.assertIn("singleton_mean_target_attainment", label_result.label_frame.columns)
        self.assertIn("weak_group_gain_norm", label_result.label_frame.columns)
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

    def test_build_ranking_feature_frame_merges_embedding_clustering_and_ris_inputs(self) -> None:
        dataset, protected_group_report, community_result = _toy_ml_fixture()
        embedding_frame = pd.DataFrame(
            {
                "node_id": list(sorted(dataset.graph.nodes())),
                "embedding_0": [float(index) for index, _ in enumerate(sorted(dataset.graph.nodes()), start=1)],
                "embedding_1": [float(index % 2) for index, _ in enumerate(sorted(dataset.graph.nodes()), start=1)],
            }
        )
        clustering_artifact = prepare_optional_clustering(
            dataset=dataset,
            method_name="kmeans",
            input_mode="embedding",
            embeddings=embedding_frame,
            config={"n_clusters": 2},
            random_seed=7,
        )
        merged = build_ranking_feature_frame(
            dataset=dataset,
            protected_group_report=protected_group_report,
            community_result=community_result,
            embedding_frame=embedding_frame,
            clustering_result=clustering_artifact.clustering_result,
            ris_scores={node_id: float(node_id) for node_id in dataset.graph.nodes()},
            fair_ris_scores={node_id: float(node_id) / 10.0 for node_id in dataset.graph.nodes()},
        )

        self.assertEqual(len(merged), dataset.graph.number_of_nodes())
        self.assertIn("embedding_0", merged.columns)
        self.assertIn("clustering_cluster_id", merged.columns)
        self.assertIn("clustering_cluster_size", merged.columns)
        self.assertIn("ris_score", merged.columns)
        self.assertIn("fair_ris_score", merged.columns)

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
        self.assertEqual(training_result.target_type, "regression")

    def test_protected_group_features_are_excluded_from_ml_by_default(self) -> None:
        dataset, protected_group_report, community_result = _toy_ml_fixture()
        feature_frame = compute_node_features(dataset, protected_group_report, community_result, community_feature_mode="full")
        label_result = generate_singleton_node_utility_labels(
            dataset=dataset,
            protected_group_report=protected_group_report,
            propagation_probability=1.0,
            mc_runs=3,
            lambda_weight=0.5,
            random_seed=7,
        )

        default_result = train_node_utility_model(
            feature_frame=feature_frame,
            label_frame=label_result.label_frame,
            budget=3,
            top_fraction=0.5,
            random_seed=7,
        )
        allowed_result = train_node_utility_model(
            feature_frame=feature_frame,
            label_frame=label_result.label_frame,
            budget=3,
            top_fraction=0.5,
            random_seed=7,
            allow_protected_features_in_ml=True,
        )

        self.assertNotIn("protected_group", default_result.metadata["feature_columns"])
        self.assertIn("protected_group", default_result.metadata["excluded_protected_feature_columns"])
        self.assertIn("protected_group", allowed_result.metadata["feature_columns"])

    def test_ranking_model_registry_lists_available_models(self) -> None:
        self.assertEqual(
            available_ranking_models(),
            (
                "random_forest",
                "xgboost",
                "mlp",
                "logistic_regression",
                "graphsage",
                "gcn",
                "ris_guidance",
            ),
        )
        self.assertEqual(get_ranking_model_spec("mlp").objective_type, "regression")
        self.assertEqual(
            get_ranking_model_spec("logistic_regression").objective_type,
            "binary_top_budget_classification",
        )

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

    def test_train_node_utility_model_can_use_mlp(self) -> None:
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
            model_type="mlp",
            top_fraction=0.5,
            random_seed=7,
        )

        self.assertEqual(training_result.model_type, "mlp")
        self.assertEqual(training_result.target_type, "regression")
        self.assertEqual(set(training_result.predicted_scores), set(dataset.graph.nodes()))

    def test_train_node_utility_model_can_use_logistic_regression_ranking(self) -> None:
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
            model_type="logistic_regression",
            top_fraction=0.5,
            random_seed=7,
        )

        self.assertEqual(training_result.model_type, "logistic_regression")
        self.assertEqual(training_result.target_type, "binary_top_budget_classification")
        self.assertEqual(set(training_result.predicted_scores), set(dataset.graph.nodes()))

    def test_logistic_regression_ranking_falls_back_when_stratified_split_is_too_sparse(self) -> None:
        feature_frame = pd.DataFrame(
            {
                "node_id": [1, 2, 3, 4, 5, 6],
                "degree": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                "protected_group": ["A", "A", "B", "B", "C", "C"],
            }
        )
        label_frame = pd.DataFrame(
            {
                "node_id": [1, 2, 3, 4, 5, 6],
                "label_score": [1.0, 0.8, 0.4, 0.3, 0.2, 0.1],
            }
        )

        training_result = train_node_utility_model(
            feature_frame=feature_frame,
            label_frame=label_frame,
            budget=1,
            model_type="logistic_regression",
            top_fraction=0.5,
            random_seed=7,
        )

        self.assertEqual(training_result.model_type, "logistic_regression")
        self.assertEqual(set(training_result.predicted_scores), set(feature_frame["node_id"]))
        self.assertEqual(len(training_result.candidate_nodes), 3)

    def test_train_ranking_model_dispatches_tabular_backend(self) -> None:
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

        training_result = train_dispatch_ranking_model(
            dataset=dataset,
            feature_frame=feature_frame,
            label_frame=label_result.label_frame,
            budget=3,
            model_type="logistic_regression",
            target_column="label_score",
            top_fraction=0.5,
            random_seed=7,
        )

        self.assertEqual(training_result.backend, "tabular")
        self.assertEqual(training_result.model_type, "logistic_regression")
        self.assertEqual(training_result.target_column, "label_score")
        self.assertEqual(training_result.debias_mode, "none")
        self.assertEqual(set(training_result.predicted_scores), set(dataset.graph.nodes()))

    def test_mlp_convergence_warnings_are_captured_in_metadata(self) -> None:
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

        def quick_mlp(*args, **kwargs):
            kwargs["max_iter"] = 1
            return SklearnMLPRegressor(*args, **kwargs)

        with patch("fim_hybrid.ml_training.MLPRegressor", side_effect=quick_mlp):
            training_result = train_dispatch_ranking_model(
                dataset=dataset,
                feature_frame=feature_frame,
                label_frame=label_result.label_frame,
                budget=3,
                model_type="mlp",
                target_column="label_score",
                top_fraction=0.5,
                random_seed=7,
            )

        self.assertEqual(training_result.model_type, "mlp")
        self.assertTrue(training_result.metadata["fit_warnings"])
        self.assertTrue(any("Maximum iterations" in warning for warning in training_result.metadata["fit_warnings"]))

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


class GNNTrainingTestCase(unittest.TestCase):
    """Check optional GNN model fitting, caching, and dependency guards."""

    def test_build_gnn_label_frame_adds_proxy_enhanced_target(self) -> None:
        dataset, protected_group_report, community_result = _toy_ml_fixture()
        label_result = generate_singleton_node_utility_labels(
            dataset=dataset,
            protected_group_report=protected_group_report,
            propagation_probability=1.0,
            mc_runs=3,
            lambda_weight=0.5,
            random_seed=7,
        )

        label_frame, runtime_seconds = _build_gnn_label_frame(
            dataset=dataset,
            protected_group_report=protected_group_report,
            community_result=community_result,
            settings=ExperimentSettings(
                protected_attribute="group",
                budget=3,
                propagation_probability=1.0,
                mc_runs_search=3,
                population_size=5,
                generations=3,
                random_seed=7,
            ),
            label_result=label_result,
        )

        self.assertGreaterEqual(runtime_seconds, 0.0)
        self.assertEqual(set(label_frame["node_id"]), set(dataset.graph.nodes()))
        self.assertIn("marginal_proxy_score", label_frame.columns)
        self.assertIn("marginal_proxy_norm", label_frame.columns)
        self.assertIn("weak_group_gain_norm", label_frame.columns)
        self.assertIn("gnn_label_score", label_frame.columns)
        self.assertTrue(label_frame["marginal_proxy_norm"].between(0.0, 1.0).all())
        self.assertGreater(float(label_frame["gnn_label_score"].var(ddof=0)), 0.0)

    def test_train_gnn_node_utility_model_errors_without_optional_dependencies(self) -> None:
        if gnn_dependencies_available():
            self.skipTest("torch and torch_geometric are installed")

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

        with self.assertRaisesRegex(ValueError, "optional dependencies are unavailable"):
            train_gnn_node_utility_model(
                dataset=dataset,
                feature_frame=feature_frame,
                label_frame=label_result.label_frame,
                budget=3,
                top_fraction=0.5,
                random_seed=7,
            )

    def test_train_gnn_node_utility_model_runs_graphsage_when_available(self) -> None:
        if not gnn_dependencies_available():
            self.skipTest("torch and torch_geometric are not installed")

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
        gnn_label_frame, _ = _build_gnn_label_frame(
            dataset=dataset,
            protected_group_report=protected_group_report,
            community_result=community_result,
            settings=ExperimentSettings(
                protected_attribute="group",
                budget=3,
                propagation_probability=1.0,
                mc_runs_search=3,
                population_size=5,
                generations=3,
                random_seed=7,
            ),
            label_result=label_result,
        )

        training_result = train_gnn_node_utility_model(
            dataset=dataset,
            feature_frame=feature_frame,
            label_frame=gnn_label_frame,
            budget=3,
            top_fraction=0.5,
            target_column="gnn_label_score",
            random_seed=7,
            epochs=20,
        )

        self.assertEqual(training_result.model_type, "graphsage")
        self.assertFalse(training_result.loaded_from_cache)
        self.assertEqual(training_result.feature_matrix_shape[0], dataset.graph.number_of_nodes())
        self.assertGreater(training_result.feature_matrix_shape[1], 0)
        self.assertEqual(training_result.edge_index_shape[0], 2)
        self.assertGreater(training_result.edge_index_shape[1], 0)
        self.assertEqual(set(training_result.predicted_scores), set(dataset.graph.nodes()))
        self.assertEqual(len(training_result.ranked_nodes), dataset.graph.number_of_nodes())
        self.assertEqual(len(training_result.candidate_nodes), 4)
        self.assertGreaterEqual(training_result.validation_spearman, -1.0)
        self.assertLessEqual(training_result.validation_spearman, 1.0)
        self.assertGreaterEqual(training_result.validation_precision_at_budget, 0.0)
        self.assertLessEqual(training_result.validation_precision_at_budget, 1.0)

    def test_train_gnn_node_utility_model_supports_gcn_when_available(self) -> None:
        if not gnn_dependencies_available():
            self.skipTest("torch and torch_geometric are not installed")

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
        gnn_label_frame, _ = _build_gnn_label_frame(
            dataset=dataset,
            protected_group_report=protected_group_report,
            community_result=community_result,
            settings=ExperimentSettings(
                protected_attribute="group",
                budget=3,
                propagation_probability=1.0,
                mc_runs_search=3,
                population_size=5,
                generations=3,
                random_seed=7,
            ),
            label_result=label_result,
        )

        training_result = train_gnn_node_utility_model(
            dataset=dataset,
            feature_frame=feature_frame,
            label_frame=gnn_label_frame,
            budget=3,
            model_type="gcn",
            top_fraction=0.5,
            target_column="gnn_label_score",
            random_seed=7,
            epochs=20,
        )

        self.assertEqual(training_result.model_type, "gcn")
        self.assertEqual(set(training_result.predicted_scores), set(dataset.graph.nodes()))

    def test_train_ranking_model_supports_gnn_debias_mode_when_available(self) -> None:
        if not gnn_dependencies_available():
            self.skipTest("torch and torch_geometric are not installed")

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

        training_result = train_dispatch_ranking_model(
            dataset=dataset,
            feature_frame=feature_frame,
            label_frame=label_result.label_frame,
            budget=3,
            model_type="graphsage",
            target_column="label_score",
            top_fraction=0.5,
            random_seed=7,
            epochs=10,
            debias_mode="worst_group_boost",
        )

        self.assertEqual(training_result.backend, "gnn")
        self.assertEqual(training_result.model_type, "graphsage")
        self.assertEqual(training_result.debias_mode, "worst_group_boost")
        self.assertEqual(training_result.target_column, "label_score")

    def test_train_gnn_node_utility_model_supports_node2vec_concat_when_available(self) -> None:
        if not gnn_dependencies_available():
            self.skipTest("torch and torch_geometric are not installed")

        dataset, protected_group_report, community_result = _toy_ml_fixture()
        plain_feature_frame = compute_node_features(dataset, protected_group_report, community_result)
        node2vec_config = Node2VecConfig(
            dimensions=4,
            walk_length=6,
            num_walks=4,
            window=2,
            random_seed=7,
        )
        node2vec_feature_frame = compute_node_features(
            dataset,
            protected_group_report,
            community_result,
            node2vec_config=node2vec_config,
        )
        label_result = generate_singleton_node_utility_labels(
            dataset=dataset,
            protected_group_report=protected_group_report,
            propagation_probability=1.0,
            mc_runs=3,
            lambda_weight=0.5,
            random_seed=7,
        )
        gnn_label_frame, _ = _build_gnn_label_frame(
            dataset=dataset,
            protected_group_report=protected_group_report,
            community_result=community_result,
            settings=ExperimentSettings(
                protected_attribute="group",
                budget=3,
                propagation_probability=1.0,
                mc_runs_search=3,
                population_size=5,
                generations=3,
                random_seed=7,
            ),
            label_result=label_result,
        )

        plain_result = train_gnn_node_utility_model(
            dataset=dataset,
            feature_frame=plain_feature_frame,
            label_frame=gnn_label_frame,
            budget=3,
            top_fraction=0.5,
            target_column="gnn_label_score",
            random_seed=7,
            epochs=20,
        )
        node2vec_result = train_gnn_node_utility_model(
            dataset=dataset,
            feature_frame=node2vec_feature_frame,
            label_frame=gnn_label_frame,
            budget=3,
            top_fraction=0.5,
            target_column="gnn_label_score",
            random_seed=7,
            epochs=20,
            node2vec_mode="input_concat",
            node2vec_config={
                "dimensions": 4,
                "walk_length": 6,
                "num_walks": 4,
                "window": 2,
                "p": 1.0,
                "q": 1.0,
                "scale_embeddings": False,
                "pca_components": None,
                "random_seed": 7,
            },
        )

        self.assertEqual(len(node2vec_feature_frame.filter(like="node2vec_").columns), 4)
        self.assertGreater(node2vec_result.feature_matrix_shape[1], plain_result.feature_matrix_shape[1])
        self.assertEqual(node2vec_result.feature_matrix_shape[0], dataset.graph.number_of_nodes())
        self.assertEqual(set(node2vec_result.predicted_scores), set(dataset.graph.nodes()))

    def test_train_gnn_node_utility_model_reuses_cache_when_available(self) -> None:
        if not gnn_dependencies_available():
            self.skipTest("torch and torch_geometric are not installed")

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
        gnn_label_frame, _ = _build_gnn_label_frame(
            dataset=dataset,
            protected_group_report=protected_group_report,
            community_result=community_result,
            settings=ExperimentSettings(
                protected_attribute="group",
                budget=3,
                propagation_probability=1.0,
                mc_runs_search=3,
                population_size=5,
                generations=3,
                random_seed=7,
            ),
            label_result=label_result,
        )

        with _workspace_tempdir() as temp_dir:
            cache_path = Path(temp_dir) / "toy_gnn_scores.pkl"
            first = train_gnn_node_utility_model(
                dataset=dataset,
                feature_frame=feature_frame,
                label_frame=gnn_label_frame,
                budget=3,
                top_fraction=0.5,
                target_column="gnn_label_score",
                random_seed=7,
                epochs=20,
                cache_path=cache_path,
            )
            second = train_gnn_node_utility_model(
                dataset=dataset,
                feature_frame=feature_frame,
                label_frame=gnn_label_frame,
                budget=3,
                top_fraction=0.5,
                target_column="gnn_label_score",
                random_seed=7,
                epochs=20,
                cache_path=cache_path,
            )

        self.assertFalse(first.loaded_from_cache)
        self.assertTrue(second.loaded_from_cache)
        self.assertEqual(first.predicted_scores, second.predicted_scores)

    def test_train_gnn_node_utility_model_separates_plain_and_node2vec_cache_fingerprints(self) -> None:
        if not gnn_dependencies_available():
            self.skipTest("torch and torch_geometric are not installed")

        dataset, protected_group_report, community_result = _toy_ml_fixture()
        plain_feature_frame = compute_node_features(dataset, protected_group_report, community_result)
        node2vec_feature_frame = compute_node_features(
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
        gnn_label_frame, _ = _build_gnn_label_frame(
            dataset=dataset,
            protected_group_report=protected_group_report,
            community_result=community_result,
            settings=ExperimentSettings(
                protected_attribute="group",
                budget=3,
                propagation_probability=1.0,
                mc_runs_search=3,
                population_size=5,
                generations=3,
                random_seed=7,
            ),
            label_result=label_result,
        )

        with _workspace_tempdir() as temp_dir:
            cache_path = Path(temp_dir) / "toy_gnn_scores.pkl"
            first_plain = train_gnn_node_utility_model(
                dataset=dataset,
                feature_frame=plain_feature_frame,
                label_frame=gnn_label_frame,
                budget=3,
                top_fraction=0.5,
                target_column="gnn_label_score",
                random_seed=7,
                epochs=20,
                cache_path=cache_path,
                node2vec_mode="off",
            )
            first_node2vec = train_gnn_node_utility_model(
                dataset=dataset,
                feature_frame=node2vec_feature_frame,
                label_frame=gnn_label_frame,
                budget=3,
                top_fraction=0.5,
                target_column="gnn_label_score",
                random_seed=7,
                epochs=20,
                cache_path=cache_path,
                node2vec_mode="input_concat",
                node2vec_config={
                    "dimensions": 4,
                    "walk_length": 6,
                    "num_walks": 4,
                    "window": 2,
                    "p": 1.0,
                    "q": 1.0,
                    "scale_embeddings": False,
                    "pca_components": None,
                    "random_seed": 7,
                },
            )
            second_node2vec = train_gnn_node_utility_model(
                dataset=dataset,
                feature_frame=node2vec_feature_frame,
                label_frame=gnn_label_frame,
                budget=3,
                top_fraction=0.5,
                target_column="gnn_label_score",
                random_seed=7,
                epochs=20,
                cache_path=cache_path,
                node2vec_mode="input_concat",
                node2vec_config={
                    "dimensions": 4,
                    "walk_length": 6,
                    "num_walks": 4,
                    "window": 2,
                    "p": 1.0,
                    "q": 1.0,
                    "scale_embeddings": False,
                    "pca_components": None,
                    "random_seed": 7,
                },
            )

        self.assertFalse(first_plain.loaded_from_cache)
        self.assertFalse(first_node2vec.loaded_from_cache)
        self.assertTrue(second_node2vec.loaded_from_cache)


if __name__ == "__main__":
    unittest.main()
