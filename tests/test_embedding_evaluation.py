"""Tests for downstream embedding evaluation helpers."""

from __future__ import annotations

import importlib.util
import json
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from fim_hybrid.data_loader import LoadedDataset
from fim_hybrid.embeddings.base import EmbeddingResult
from fim_hybrid.embeddings.benchmark import BenchmarkRunResult
from fim_hybrid.embeddings.evaluation import evaluate_embedding_benchmark_repeated, evaluate_embedding_result
from fim_hybrid.embeddings.evaluation_splits import (
    build_link_prediction_split,
    build_node_classification_split,
    build_node_train_validation_split,
)
from fim_hybrid.embeddings.node_classification_models import summarize_group_classification_metrics
from fim_hybrid.embeddings.node_classification_models import (
    GraphSAGENodeClassificationConfig,
    _group_robust_sample_weights,
    _resolve_validation_probe_subset,
    validate_graphsage_node_classification_config,
)


def _toy_dataset() -> LoadedDataset:
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
        ]
    )
    labels = {
        1: "A",
        2: "A",
        3: "B",
        4: "B",
        5: "C",
        6: "C",
        7: "D",
        8: "D",
    }
    regions = {
        1: "north",
        2: "north",
        3: "east",
        4: "east",
        5: "south",
        6: "south",
        7: "west",
        8: "west",
    }
    node_attributes = pd.DataFrame(
        [{"node_id": node_id, "group": labels[node_id], "region": regions[node_id]} for node_id in sorted(graph.nodes())]
    ).set_index("node_id", drop=False)
    return LoadedDataset(name="toy_embedding_eval", graph=graph, node_attributes=node_attributes)


def _manual_embedding_frame() -> pd.DataFrame:
    vectors = {
        1: [2.0, 0.0],
        2: [2.2, 0.1],
        3: [0.0, 2.0],
        4: [0.1, 2.2],
        5: [-2.0, 0.0],
        6: [-2.2, 0.1],
        7: [0.0, -2.0],
        8: [0.1, -2.2],
    }
    rows = [{"node_id": node_id, "embedding_0": values[0], "embedding_1": values[1]} for node_id, values in vectors.items()]
    return pd.DataFrame(list(reversed(rows)))


def _imbalanced_toy_dataset() -> LoadedDataset:
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (2, 5),
            (3, 6),
            (7, 9),
            (10, 12),
        ]
    )
    labels = {
        1: "A",
        2: "A",
        3: "A",
        4: "A",
        5: "A",
        6: "A",
        7: "B",
        8: "B",
        9: "B",
        10: "C",
        11: "C",
        12: "C",
    }
    regions = {
        1: "north",
        2: "north",
        3: "north",
        4: "north",
        5: "central",
        6: "central",
        7: "east",
        8: "east",
        9: "east",
        10: "south",
        11: "south",
        12: "south",
    }
    node_attributes = pd.DataFrame(
        [{"node_id": node_id, "group": labels[node_id], "region": regions[node_id]} for node_id in sorted(graph.nodes())]
    ).set_index("node_id", drop=False)
    return LoadedDataset(name="imbalanced_toy_embedding_eval", graph=graph, node_attributes=node_attributes)


def _imbalanced_embedding_frame() -> pd.DataFrame:
    vectors = {
        1: [2.2, 0.0],
        2: [2.1, 0.1],
        3: [2.0, -0.1],
        4: [1.9, 0.1],
        5: [1.8, -0.1],
        6: [1.7, 0.0],
        7: [0.1, 2.0],
        8: [0.0, 2.1],
        9: [0.2, 1.9],
        10: [-2.0, 0.1],
        11: [-1.9, -0.1],
        12: [-2.1, 0.0],
    }
    rows = [{"node_id": node_id, "embedding_0": values[0], "embedding_1": values[1]} for node_id, values in vectors.items()]
    return pd.DataFrame(rows)


class EvaluationSplitTestCase(unittest.TestCase):
    """Verify deterministic split construction."""

    def test_node_classification_split_is_reproducible(self) -> None:
        split_a = build_node_classification_split(
            [1, 2, 3, 4, 5, 6, 7, 8],
            ["A", "A", "B", "B", "C", "C", "D", "D"],
            test_fraction=0.25,
            random_seed=7,
        )
        split_b = build_node_classification_split(
            [1, 2, 3, 4, 5, 6, 7, 8],
            ["A", "A", "B", "B", "C", "C", "D", "D"],
            test_fraction=0.25,
            random_seed=7,
        )

        self.assertEqual(split_a, split_b)

    def test_link_prediction_split_is_reproducible(self) -> None:
        dataset = _toy_dataset()
        split_a = build_link_prediction_split(
            dataset.graph.to_undirected(),
            test_fraction=0.25,
            negative_ratio=1.0,
            random_seed=7,
        )
        split_b = build_link_prediction_split(
            dataset.graph.to_undirected(),
            test_fraction=0.25,
            negative_ratio=1.0,
            random_seed=7,
        )

        self.assertEqual(split_a, split_b)

    def test_node_train_validation_split_is_reproducible(self) -> None:
        split_a = build_node_train_validation_split(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            ["A", "A", "A", "A", "B", "B", "C", "C", "C"],
            validation_fraction=0.25,
            random_seed=7,
        )
        split_b = build_node_train_validation_split(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            ["A", "A", "A", "A", "B", "B", "C", "C", "C"],
            validation_fraction=0.25,
            random_seed=7,
        )

        self.assertEqual(split_a, split_b)


class EvaluationTaskTestCase(unittest.TestCase):
    """Validate task-level evaluation behavior and skip handling."""

    def test_evaluate_embedding_result_runs_all_tasks(self) -> None:
        dataset = _toy_dataset()
        frame = _manual_embedding_frame()

        result = evaluate_embedding_result(
            dataset,
            frame,
            method_name="manual",
            tasks=["node_classification", "link_prediction", "node_clustering"],
            label_column="group",
            random_seed=7,
        )

        self.assertEqual(set(result["task"]), {"node_classification", "link_prediction", "node_clustering"})
        self.assertEqual(set(result["status"]), {"ok"})
        self.assertTrue((result["method"] == "manual").all())
        classification_row = result.loc[result["task"] == "node_classification"].iloc[0]
        clustering_row = result.loc[result["task"] == "node_clustering"].iloc[0]
        self.assertGreaterEqual(float(classification_row["accuracy"]), 0.0)
        self.assertGreaterEqual(float(classification_row["macro_f1"]), 0.0)
        self.assertGreaterEqual(float(classification_row["weighted_f1"]), 0.0)
        self.assertGreaterEqual(float(classification_row["balanced_accuracy"]), 0.0)
        self.assertGreaterEqual(float(classification_row["macro_precision"]), 0.0)
        self.assertGreaterEqual(float(classification_row["macro_recall"]), 0.0)
        self.assertFalse(pd.isna(classification_row["per_class_metrics_json"]))
        self.assertFalse(pd.isna(classification_row["confusion_matrix_json"]))
        self.assertFalse(pd.isna(classification_row["class_distribution_json"]))
        self.assertFalse(pd.isna(classification_row["split_diagnostics_json"]))
        per_class_metrics = json.loads(str(classification_row["per_class_metrics_json"]))
        confusion_summary = json.loads(str(classification_row["confusion_matrix_json"]))
        class_distribution = json.loads(str(classification_row["class_distribution_json"]))
        split_diagnostics = json.loads(str(classification_row["split_diagnostics_json"]))
        self.assertEqual(len(per_class_metrics), 4)
        self.assertEqual(sorted(item["label"] for item in per_class_metrics), ["A", "B", "C", "D"])
        self.assertEqual(confusion_summary["labels"], ["A", "B", "C", "D"])
        self.assertEqual(len(confusion_summary["matrix"]), 4)
        self.assertEqual(class_distribution["labels"], ["A", "B", "C", "D"])
        self.assertEqual(sum(class_distribution["true_counts"].values()), int(classification_row["test_count"]))
        self.assertEqual(sorted(split_diagnostics.keys()), ["test", "train"])
        self.assertEqual(int(split_diagnostics["train"]["count"]), int(classification_row["train_count"]))
        self.assertEqual(int(split_diagnostics["test"]["count"]), int(classification_row["test_count"]))
        self.assertGreaterEqual(float(clustering_row["nmi"]), 0.0)
        self.assertGreaterEqual(float(clustering_row["ari"]), 0.0)

    def test_evaluate_embedding_result_is_reproducible(self) -> None:
        dataset = _toy_dataset()
        frame = _manual_embedding_frame()

        first = evaluate_embedding_result(
            dataset,
            frame,
            method_name="manual",
            tasks=["node_classification", "link_prediction"],
            label_column="group",
            random_seed=11,
        )
        second = evaluate_embedding_result(
            dataset,
            frame,
            method_name="manual",
            tasks=["node_classification", "link_prediction"],
            label_column="group",
            random_seed=11,
        )

        comparable_columns = [column for column in first.columns if column != "runtime_seconds"]
        pd.testing.assert_frame_equal(first[comparable_columns], second[comparable_columns])

    def test_missing_label_column_skips_cleanly(self) -> None:
        dataset = _toy_dataset()
        frame = _manual_embedding_frame()

        result = evaluate_embedding_result(
            dataset,
            frame,
            method_name="manual",
            tasks=["node_classification", "node_clustering"],
            label_column="missing_group",
            random_seed=7,
        )

        self.assertEqual(set(result["status"]), {"skipped"})
        self.assertTrue(result["skipped_reason"].str.contains("not present", regex=False).all())

    def test_protected_group_metrics_are_computed_when_requested(self) -> None:
        dataset = _toy_dataset()
        frame = _manual_embedding_frame()

        result = evaluate_embedding_result(
            dataset,
            frame,
            method_name="manual",
            tasks=["node_classification"],
            label_column="group",
            protected_attribute_column="region",
            random_seed=7,
        )

        row = result.loc[result["task"] == "node_classification"].iloc[0]
        self.assertEqual(row["protected_attribute_column"], "region")
        self.assertGreaterEqual(float(row["best_group_accuracy"]), 0.0)
        self.assertGreaterEqual(float(row["worst_group_accuracy"]), 0.0)
        self.assertGreaterEqual(float(row["group_accuracy_gap"]), 0.0)
        self.assertGreaterEqual(float(row["accuracy_gap"]), 0.0)
        self.assertGreaterEqual(float(row["best_group_macro_f1"]), 0.0)
        self.assertGreaterEqual(float(row["worst_group_macro_f1"]), 0.0)
        self.assertGreaterEqual(float(row["group_macro_f1_gap"]), 0.0)
        self.assertGreaterEqual(float(row["best_group_f1"]), 0.0)
        self.assertGreaterEqual(float(row["worst_group_f1"]), 0.0)
        self.assertGreaterEqual(float(row["macro_f1_gap"]), 0.0)
        self.assertFalse(pd.isna(row["per_group_metrics_json"]))
        self.assertFalse(pd.isna(row["per_group_confusion_json"]))
        self.assertFalse(pd.isna(row["group_error_diagnostics_json"]))
        self.assertFalse(pd.isna(row["split_diagnostics_json"]))
        per_group_metrics = json.loads(str(row["per_group_metrics_json"]))
        per_group_confusions = json.loads(str(row["per_group_confusion_json"]))
        group_error_diagnostics = json.loads(str(row["group_error_diagnostics_json"]))
        split_diagnostics = json.loads(str(row["split_diagnostics_json"]))
        self.assertTrue(per_group_metrics)
        for metric_row in per_group_metrics:
            self.assertEqual(
                sorted(metric_row.keys()),
                [
                    "accuracy",
                    "balanced_accuracy",
                    "below_support_threshold",
                    "count",
                    "group",
                    "macro_f1",
                    "macro_precision",
                    "macro_recall",
                    "supported",
                    "weighted_f1",
                ],
            )
        self.assertEqual(len(per_group_confusions), len(per_group_metrics))
        self.assertEqual(len(group_error_diagnostics), len(per_group_metrics))
        self.assertEqual(sorted(split_diagnostics.keys()), ["test", "train"])
        self.assertIn("protected_counts", split_diagnostics["train"])
        self.assertIn("label_by_protected_counts", split_diagnostics["test"])
        self.assertIn("prediction_collapse", group_error_diagnostics[0])
        self.assertIn("top_confusions", group_error_diagnostics[0])
        self.assertEqual(str(row["unavailable_reason"]), "")

    def test_group_summary_reports_raw_and_supported_fairness_separately(self) -> None:
        summary, per_group_json, _, _ = summarize_group_classification_metrics(
            y_true=np.asarray(["A", "A", "A", "B"], dtype=object),
            y_pred=np.asarray(["A", "A", "A", "A"], dtype=object),
            group_values=np.asarray(["large", "large", "large", "tiny"], dtype=object),
            min_group_support_threshold=2,
            report_small_group_metrics=True,
        )

        self.assertAlmostEqual(float(summary["worst_group_f1_raw"]), 0.0, places=6)
        self.assertAlmostEqual(float(summary["worst_group_f1_supported"]), 1.0, places=6)
        self.assertEqual(int(summary["small_group_count"]), 1)
        self.assertEqual(int(summary["supported_group_count"]), 1)
        self.assertIn('"group":"tiny"', str(summary["groups_below_support_threshold_json"]))
        self.assertIn('"group":"large"', str(summary["group_support_counts_json"]))
        per_group_metrics = json.loads(str(per_group_json))
        self.assertEqual(len(per_group_metrics), 2)
        tiny_row = next(item for item in per_group_metrics if item["group"] == "tiny")
        self.assertTrue(bool(tiny_row["below_support_threshold"]))
        self.assertFalse(bool(tiny_row["supported"]))

    def test_fixed_training_mode_bundles_resolve_to_expected_settings(self) -> None:
        support_aware = validate_graphsage_node_classification_config(
            GraphSAGENodeClassificationConfig(training_mode="support_aware_group_robust")
        )
        self.assertEqual(support_aware.training_mode, "support_aware_group_robust")
        self.assertEqual(support_aware.imbalance_mode, "none")
        self.assertEqual(support_aware.debias_mode, "none")
        self.assertEqual(support_aware.group_weight_mode, "inverse_frequency")
        self.assertAlmostEqual(float(support_aware.group_robust_weight), 0.25, places=6)
        self.assertFalse(bool(support_aware.rebalance_batches_by_group))
        self.assertEqual(support_aware.early_stop_metric, "worst_group_f1_supported")

        anti_collapse = validate_graphsage_node_classification_config(
            GraphSAGENodeClassificationConfig(training_mode="anti_collapse_group_robust")
        )
        self.assertEqual(anti_collapse.training_mode, "anti_collapse_group_robust")
        self.assertEqual(anti_collapse.group_weight_mode, "min_support_boost")
        self.assertAlmostEqual(float(anti_collapse.group_robust_weight), 0.25, places=6)
        self.assertAlmostEqual(float(anti_collapse.min_support_boost_factor), 2.0, places=6)
        self.assertTrue(bool(anti_collapse.rebalance_batches_by_group))
        self.assertEqual(anti_collapse.early_stop_metric, "worst_group_f1_raw")

        mild_adversarial = validate_graphsage_node_classification_config(
            GraphSAGENodeClassificationConfig(
                training_mode="anti_collapse_group_robust_with_mild_adversarial"
            )
        )
        self.assertEqual(
            mild_adversarial.training_mode,
            "anti_collapse_group_robust_with_mild_adversarial",
        )
        self.assertEqual(mild_adversarial.debias_mode, "adversarial")
        self.assertEqual(mild_adversarial.group_weight_mode, "min_support_boost")
        self.assertAlmostEqual(float(mild_adversarial.adversary_loss_weight), 0.02, places=6)
        self.assertAlmostEqual(float(mild_adversarial.gradient_reversal_lambda), 0.02, places=6)
        self.assertEqual(int(mild_adversarial.adversary_hidden_dim), 64)
        self.assertEqual(int(mild_adversarial.adversary_num_layers), 1)
        self.assertAlmostEqual(float(mild_adversarial.adversary_dropout), 0.1, places=6)
        self.assertEqual(mild_adversarial.early_stop_metric, "fairness_score")

    def test_min_support_boost_weights_small_groups_more_heavily_when_available(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed")

        import torch

        group_targets = torch.tensor([0, 0, 0, 0, 1], dtype=torch.long)
        loss_vector = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        predicted_targets = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
        result = _group_robust_sample_weights(
            group_targets,
            loss_vector=loss_vector,
            predicted_targets=predicted_targets,
            group_count=2,
            group_weight_mode="min_support_boost",
            group_robust_weight=1.0,
            worst_group_boost_factor=2.0,
            min_support_boost_factor=2.0,
            min_group_support_threshold=2,
            group_state=None,
            torch=torch,
        )

        self.assertIsNotNone(result)
        sample_weights = result["sample_weights"].detach().cpu().numpy()
        self.assertGreater(float(sample_weights[-1]), float(sample_weights[0]))
        self.assertEqual(result["boosted_group_indices"], [1])

    def test_validation_probe_subset_falls_back_to_supported_shared_groups(self) -> None:
        resolution = _resolve_validation_probe_subset(
            ["lancaster", "lancaster", "palmdale", "palmdale", "acton"],
            ["lancaster", "lancaster", "palmdale", "palmdale", "leona_valley"],
            min_group_support_threshold=2,
            prefer_support_aware_subset=True,
        )

        self.assertEqual(resolution["reason"], "")
        self.assertEqual(resolution["scope"], "supported_shared")
        self.assertEqual(resolution["train_mask"].tolist(), [True, True, True, True, False])
        self.assertEqual(resolution["test_mask"].tolist(), [True, True, True, True, False])

    def test_repeated_evaluation_aggregates_macro_and_fairness_metrics(self) -> None:
        dataset = _toy_dataset()
        frame = _manual_embedding_frame()
        benchmark_result = BenchmarkRunResult(
            dataset_name=dataset.name,
            summary_frame=pd.DataFrame(
                [
                    {
                        "method": "manual",
                        "status": "ok",
                        "runtime_seconds": 0.0,
                        "embedding_dim": 2,
                        "node_count": dataset.graph.number_of_nodes(),
                    }
                ]
            ),
            results_by_method={
                "manual": EmbeddingResult(
                    method_name="manual",
                    embedding_frame=frame,
                    runtime_seconds=0.0,
                    node_count=dataset.graph.number_of_nodes(),
                    embedding_dim=2,
                    config={},
                )
            },
            output_dir=None,
        )

        per_run_frame, aggregate_frame = evaluate_embedding_benchmark_repeated(
            dataset,
            benchmark_result,
            evaluation_random_seeds=[7, 11],
            repeated_split_count=2,
            tasks=["node_classification"],
            random_seed=7,
            label_column="group",
            protected_attribute_column="region",
            min_group_support_eval=2,
        )

        self.assertEqual(len(per_run_frame), 4)
        self.assertEqual(sorted(per_run_frame["effective_random_seed"].astype(int).tolist()), [7, 11, 1007, 1011])
        self.assertEqual(len(aggregate_frame), 1)
        aggregate_row = aggregate_frame.iloc[0]
        self.assertEqual(int(aggregate_row["run_count"]), 4)
        self.assertIn("mean_macro_f1", aggregate_frame.columns)
        self.assertIn("std_macro_f1", aggregate_frame.columns)
        self.assertIn("mean_worst_group_f1_raw", aggregate_frame.columns)
        self.assertIn("std_worst_group_f1_supported", aggregate_frame.columns)
        self.assertIn("mean_protected_probe_macro_f1", aggregate_frame.columns)

    def test_split_diagnostics_track_underrepresented_groups_when_requested(self) -> None:
        dataset = _toy_dataset()
        frame = _manual_embedding_frame()

        result = evaluate_embedding_result(
            dataset,
            frame,
            method_name="manual",
            tasks=["node_classification"],
            label_column="group",
            protected_attribute_column="region",
            random_seed=7,
        )

        row = result.loc[result["task"] == "node_classification"].iloc[0]
        split_diagnostics = json.loads(str(row["split_diagnostics_json"]))
        self.assertEqual(int(split_diagnostics["train"]["count"]) + int(split_diagnostics["test"]["count"]), 8)
        self.assertIn("underrepresented_protected_groups", split_diagnostics["train"])
        self.assertIn("underrepresented_intersections", split_diagnostics["test"])
        self.assertTrue(split_diagnostics["train"]["underrepresented_protected_groups"])
        self.assertTrue(split_diagnostics["test"]["underrepresented_intersections"])

    def test_protected_group_metrics_record_unavailable_reason_when_missing(self) -> None:
        dataset = _toy_dataset()
        frame = _manual_embedding_frame()

        result = evaluate_embedding_result(
            dataset,
            frame,
            method_name="manual",
            tasks=["node_classification"],
            label_column="group",
            protected_attribute_column="missing_region",
            random_seed=7,
        )

        row = result.loc[result["task"] == "node_classification"].iloc[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["protected_attribute_column"], "missing_region")
        self.assertTrue(pd.isna(row["worst_group_accuracy"]))
        self.assertIn("not present", str(row["unavailable_reason"]))

    def test_protected_group_metrics_record_unavailable_reason_when_test_groups_missing(self) -> None:
        dataset = _toy_dataset()
        dataset.node_attributes.loc[[3, 4, 7, 8], "region"] = pd.NA
        frame = _manual_embedding_frame()

        result = evaluate_embedding_result(
            dataset,
            frame,
            method_name="manual",
            tasks=["node_classification"],
            label_column="group",
            protected_attribute_column="region",
            random_seed=7,
        )

        row = result.loc[result["task"] == "node_classification"].iloc[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["protected_attribute_column"], "region")
        self.assertTrue(pd.isna(row["per_group_metrics_json"]))
        self.assertIn("contains missing values", str(row["unavailable_reason"]))

    def test_logistic_node_classification_rejects_graphsage_only_options(self) -> None:
        dataset = _toy_dataset()
        frame = _manual_embedding_frame()

        with self.assertRaisesRegex(ValueError, "does not support imbalance"):
            evaluate_embedding_result(
                dataset,
                frame,
                method_name="manual",
                tasks=["node_classification"],
                label_column="group",
                random_seed=7,
                node_classification_model="logistic_regression",
                imbalance_mode="focal_loss",
            )

    def test_logistic_node_classification_rejects_adversarial_options(self) -> None:
        dataset = _toy_dataset()
        frame = _manual_embedding_frame()

        with self.assertRaisesRegex(ValueError, "does not support imbalance, debiasing"):
            evaluate_embedding_result(
                dataset,
                frame,
                method_name="manual",
                tasks=["node_classification"],
                label_column="group",
                protected_attribute_column="region",
                random_seed=7,
                node_classification_model="logistic_regression",
                debias_mode="adversarial",
            )

    def test_adversarial_graphsage_requires_protected_attribute_column(self) -> None:
        dataset = _imbalanced_toy_dataset()
        frame = _imbalanced_embedding_frame()

        with self.assertRaisesRegex(ValueError, "requires protected_attribute_column"):
            evaluate_embedding_result(
                dataset,
                frame,
                method_name="manual",
                tasks=["node_classification"],
                label_column="group",
                random_seed=11,
                node_classification_model="graphsage",
                debias_mode="adversarial",
            )

    def test_graphsage_node_classification_supports_imbalance_modes_when_available(self) -> None:
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_geometric") is None:
            self.skipTest("torch and torch_geometric are not installed")

        dataset = _imbalanced_toy_dataset()
        frame = _imbalanced_embedding_frame()
        for imbalance_mode in ("class_weighted", "focal_loss", "weighted_sampler"):
            with self.subTest(imbalance_mode=imbalance_mode):
                result = evaluate_embedding_result(
                    dataset,
                    frame,
                    method_name="manual",
                    tasks=["node_classification"],
                    label_column="group",
                    random_seed=11,
                    node_classification_model="graphsage",
                    imbalance_mode=imbalance_mode,
                    early_stop_metric="macro_f1",
                    early_stop_patience=5,
                    node_validation_fraction=0.25,
                )

                row = result.loc[result["task"] == "node_classification"].iloc[0]
                self.assertEqual(row["classifier"], "graphsage")
                self.assertEqual(row["status"], "ok")
                self.assertEqual(row["training_mode"], "imbalance_only")
                self.assertEqual(row["debias_mode"], "none")
                self.assertEqual(row["imbalance_mode"], imbalance_mode)
                self.assertEqual(row["early_stop_metric"], "macro_f1")
                self.assertGreaterEqual(float(row["accuracy"]), 0.0)
                self.assertGreaterEqual(float(row["macro_f1"]), 0.0)
                self.assertGreaterEqual(float(row["micro_f1"]), 0.0)
                self.assertGreaterEqual(float(row["weighted_f1"]), 0.0)
                self.assertGreaterEqual(float(row["balanced_accuracy"]), 0.0)
                self.assertGreaterEqual(float(row["macro_precision"]), 0.0)
                self.assertGreaterEqual(float(row["macro_recall"]), 0.0)
                self.assertFalse(pd.isna(row["per_class_metrics_json"]))
                self.assertFalse(pd.isna(row["confusion_matrix_json"]))
                self.assertFalse(pd.isna(row["class_distribution_json"]))
                self.assertEqual(str(row["group_weight_mode"]), "none")
                self.assertAlmostEqual(float(row["group_robust_weight"]), 0.0, places=6)
                self.assertIn(f"imbalance_mode={imbalance_mode}", str(row["notes"]))
                self.assertIn("early_stop_metric=macro_f1", str(row["notes"]))

    def test_graphsage_node_classification_supports_adversarial_debiasing_when_available(self) -> None:
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_geometric") is None:
            self.skipTest("torch and torch_geometric are not installed")

        dataset = _imbalanced_toy_dataset()
        frame = _imbalanced_embedding_frame()
        result = evaluate_embedding_result(
            dataset,
            frame,
            method_name="manual",
            tasks=["node_classification"],
            label_column="group",
            protected_attribute_column="region",
            random_seed=17,
            node_classification_model="graphsage",
            debias_mode="adversarial",
            imbalance_mode="class_weighted",
            early_stop_metric="worst_group_f1",
            early_stop_patience=5,
            node_validation_fraction=0.25,
            adversary_loss_weight=1.0,
            gradient_reversal_lambda=1.0,
            adversary_hidden_dim=32,
            adversary_num_layers=2,
            adversary_dropout=0.1,
        )

        row = result.loc[result["task"] == "node_classification"].iloc[0]
        self.assertEqual(row["classifier"], "graphsage")
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["training_mode"], "adversarial")
        self.assertEqual(row["comparison_mode"], "adversarial")
        self.assertEqual(row["debias_mode"], "adversarial")
        self.assertEqual(row["imbalance_mode"], "class_weighted")
        self.assertEqual(row["early_stop_metric"], "worst_group_f1_raw")
        self.assertEqual(float(row["adversary_loss_weight"]), 1.0)
        self.assertEqual(float(row["gradient_reversal_lambda"]), 1.0)
        self.assertEqual(int(row["adversary_hidden_dim"]), 32)
        self.assertEqual(int(row["adversary_num_layers"]), 2)
        self.assertAlmostEqual(float(row["adversary_dropout"]), 0.1, places=6)
        self.assertGreaterEqual(float(row["worst_group_f1"]), 0.0)
        self.assertIn(str(row["protected_probe_status"]), {"ok", "skipped"})
        self.assertIn("debias_mode=adversarial", str(row["notes"]))
        self.assertIn("early_stop_metric=worst_group_f1_raw", str(row["notes"]))

    def test_graphsage_node_classification_supports_group_robust_only_mode_when_available(self) -> None:
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_geometric") is None:
            self.skipTest("torch and torch_geometric are not installed")

        dataset = _imbalanced_toy_dataset()
        frame = _imbalanced_embedding_frame()
        result = evaluate_embedding_result(
            dataset,
            frame,
            method_name="manual",
            tasks=["node_classification"],
            label_column="group",
            protected_attribute_column="region",
            random_seed=19,
            node_classification_model="graphsage",
            training_mode="group_robust",
            group_weight_mode="group_dro",
            group_robust_weight=0.5,
            min_group_support_threshold=2,
            early_stop_metric="worst_group_f1",
            early_stop_patience=3,
            node_validation_fraction=0.25,
        )

        row = result.loc[result["task"] == "node_classification"].iloc[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["training_mode"], "group_robust")
        self.assertEqual(row["comparison_mode"], "group_robust")
        self.assertEqual(row["debias_mode"], "none")
        self.assertEqual(row["group_weight_mode"], "group_dro")
        self.assertAlmostEqual(float(row["group_robust_weight"]), 0.5, places=6)
        self.assertEqual(int(row["min_group_support_threshold"]), 2)
        self.assertGreaterEqual(int(row["best_epoch"]), 1)
        history_payload = json.loads(str(row["training_history_json"]))
        self.assertTrue(history_payload)
        self.assertIn("epoch", history_payload[0])
        self.assertIn("worst_group_f1", history_payload[0])
        self.assertIn("collapsed_groups", history_payload[0])
        self.assertIn("small_group_count", row.index)
        self.assertIn("collapsed_groups", row.index)
        self.assertIn("max_group_error", row.index)
        self.assertIn("comparison_mode=group_robust", str(row["notes"]))

    def test_graphsage_node_classification_supports_anti_collapse_mode_alias_when_available(self) -> None:
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_geometric") is None:
            self.skipTest("torch and torch_geometric are not installed")

        dataset = _imbalanced_toy_dataset()
        frame = _imbalanced_embedding_frame()
        result = evaluate_embedding_result(
            dataset,
            frame,
            method_name="manual",
            tasks=["node_classification"],
            label_column="group",
            protected_attribute_column="region",
            random_seed=23,
            node_classification_model="graphsage",
            training_mode="anti_collapse_group_robust",
            node_validation_fraction=0.25,
            early_stop_patience=3,
        )

        row = result.loc[result["task"] == "node_classification"].iloc[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["training_mode"], "anti_collapse_group_robust")
        self.assertEqual(row["comparison_mode"], "anti_collapse_group_robust")
        self.assertEqual(row["debias_mode"], "none")
        self.assertEqual(row["group_weight_mode"], "min_support_boost")
        self.assertAlmostEqual(float(row["group_robust_weight"]), 0.25, places=6)
        self.assertAlmostEqual(float(row["min_support_boost_factor"]), 2.0, places=6)
        self.assertTrue(bool(row["rebalance_batches_by_group"]))
        self.assertEqual(row["early_stop_metric"], "worst_group_f1_raw")
        self.assertIn("min_support_boost_factor=2.0", str(row["notes"]))

    def test_graphsage_node_classification_supports_group_robust_fairness_score_when_available(self) -> None:
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_geometric") is None:
            self.skipTest("torch and torch_geometric are not installed")

        dataset = _imbalanced_toy_dataset()
        frame = _imbalanced_embedding_frame()
        result = evaluate_embedding_result(
            dataset,
            frame,
            method_name="manual",
            tasks=["node_classification"],
            label_column="group",
            protected_attribute_column="region",
            random_seed=19,
            node_classification_model="graphsage",
            training_mode="adversarial_group_robust",
            debias_mode="adversarial",
            imbalance_mode="class_weighted",
            group_weight_mode="inverse_frequency",
            group_robust_weight=1.0,
            min_group_support_threshold=1,
            early_stop_metric="fairness_score",
            early_stop_patience=3,
            fairness_score_alpha=0.25,
            fairness_score_beta=0.5,
            probe_model_type="nonlinear",
            node_validation_fraction=0.25,
            adversary_loss_weight=1.0,
            gradient_reversal_lambda=1.0,
            adversary_hidden_dim=32,
            adversary_num_layers=2,
            adversary_dropout=0.1,
        )

        row = result.loc[result["task"] == "node_classification"].iloc[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["training_mode"], "adversarial_group_robust")
        self.assertEqual(row["comparison_mode"], "adversarial_group_robust")
        self.assertEqual(row["debias_mode"], "adversarial")
        self.assertEqual(row["group_weight_mode"], "inverse_frequency")
        self.assertAlmostEqual(float(row["group_robust_weight"]), 1.0, places=6)
        self.assertEqual(row["early_stop_metric"], "fairness_score")
        self.assertAlmostEqual(float(row["worst_group_boost_factor"]), 2.0, places=6)
        self.assertEqual(int(row["min_group_support_threshold"]), 1)
        self.assertEqual(int(row["min_group_support_train"]), 1)
        self.assertEqual(int(row["min_group_support_eval"]), 1)
        self.assertAlmostEqual(float(row["fairness_score_alpha"]), 0.25, places=6)
        self.assertAlmostEqual(float(row["fairness_score_beta"]), 0.5, places=6)
        self.assertEqual(row["protected_probe_model_type"], "nonlinear")
        self.assertGreaterEqual(int(row["best_epoch"]), 1)
        history_payload = json.loads(str(row["training_history_json"]))
        self.assertTrue(history_payload)
        self.assertIn("protected_probe_macro_f1", history_payload[-1])
        self.assertIn("training_mode=adversarial_group_robust", str(row["notes"]))
        self.assertIn("comparison_mode=adversarial_group_robust", str(row["notes"]))
        self.assertIn("group_weight_mode=inverse_frequency", str(row["notes"]))
        self.assertIn("fairness_score_alpha=0.25", str(row["notes"]))
        self.assertIn(str(row["protected_probe_status"]), {"ok", "skipped"})

    def test_graphsage_node_classification_is_reproducible_when_available(self) -> None:
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_geometric") is None:
            self.skipTest("torch and torch_geometric are not installed")

        dataset = _imbalanced_toy_dataset()
        frame = _imbalanced_embedding_frame()
        first = evaluate_embedding_result(
            dataset,
            frame,
            method_name="manual",
            tasks=["node_classification"],
            label_column="group",
            random_seed=13,
            node_classification_model="graphsage",
            imbalance_mode="class_weighted",
            early_stop_metric="macro_f1",
            early_stop_patience=5,
            node_validation_fraction=0.25,
        )
        second = evaluate_embedding_result(
            dataset,
            frame,
            method_name="manual",
            tasks=["node_classification"],
            label_column="group",
            random_seed=13,
            node_classification_model="graphsage",
            imbalance_mode="class_weighted",
            early_stop_metric="macro_f1",
            early_stop_patience=5,
            node_validation_fraction=0.25,
        )

        comparable_columns = [column for column in first.columns if column != "runtime_seconds"]
        pd.testing.assert_frame_equal(first[comparable_columns], second[comparable_columns])


if __name__ == "__main__":
    unittest.main()
