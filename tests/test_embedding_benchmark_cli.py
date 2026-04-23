"""Tests for the embedding benchmark CLI helpers."""

from __future__ import annotations

from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from uuid import uuid4

import pandas as pd

from scripts.run_embedding_benchmark import (
    _all_attributes_requested,
    _graphsage_comparison_variants,
    _sorted_graphsage_comparison_frame,
    _sorted_graphsage_repeated_comparison_frame,
    benchmark_report_path,
    build_dataset_config,
    build_evaluation_kwargs,
    build_method_configs,
    attribute_report_path,
    format_benchmark_report,
    format_all_attributes_benchmark_report,
    print_saved_benchmark_summary,
    render_saved_benchmark_summary_text,
    save_benchmark_report,
    save_all_attributes_benchmark_report,
    save_attribute_benchmark_report,
)


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


class EmbeddingBenchmarkCliTestCase(unittest.TestCase):
    """Check CLI config building and report rendering."""

    def test_build_dataset_config_supports_custom_graph_path(self) -> None:
        with _workspace_tempdir() as temp_dir:
            edge_path = Path(temp_dir) / "custom_edges.txt"
            edge_path.write_text("1 2\n2 3\n", encoding="utf-8")
            args = Namespace(
                dataset="graph_spa_500_0",
                graph_path=str(edge_path),
                attributes_path=None,
                dataset_format="auto",
                dataset_config=None,
                directed=False,
                source_col=None,
                target_col=None,
                node_id_col=None,
            )

            config = build_dataset_config(args)

            self.assertEqual(config.edge_path, edge_path)
            self.assertFalse(config.directed)
            self.assertEqual(config.name, "custom_edges")

    def test_all_attributes_requested_detects_label_column_all(self) -> None:
        args = Namespace(
            label_column="all",
            classification_label_column=None,
            clustering_label_column=None,
        )

        self.assertTrue(_all_attributes_requested(args))

    def test_build_method_configs_populates_method_specific_fields(self) -> None:
        args = Namespace(
            embedding_dim=8,
            random_seed=7,
            walk_length=10,
            num_walks=5,
            window_size=3,
            node2vec_p=0.5,
            node2vec_q=2.0,
            line_order="both",
            hidden_dim=16,
            num_layers=2,
            dropout=0.2,
            learning_rate=1e-3,
            weight_decay=5e-4,
            epochs=25,
            projection_dim=8,
            temperature=0.5,
            edge_dropout_probability=0.1,
            feature_mask_probability=0.2,
        )

        configs = build_method_configs(args, ["deepwalk", "node2vec", "line", "graphcl"])

        self.assertEqual(configs["deepwalk"]["walk_length"], 10)
        self.assertEqual(configs["node2vec"]["p"], 0.5)
        self.assertEqual(configs["node2vec"]["q"], 2.0)
        self.assertEqual(configs["line"]["order"], "both")
        self.assertEqual(configs["graphcl"]["projection_dim"], 8)
        self.assertEqual(configs["graphcl"]["epochs"], 25)

    def test_build_evaluation_kwargs_includes_node_classification_options(self) -> None:
        args = Namespace(
            node_test_fraction=0.3,
            link_test_fraction=0.2,
            link_negative_ratio=1.5,
            link_prediction_edge_feature="concat",
            node_classification_model="graphsage",
            training_mode="adversarial_group_robust",
            imbalance_mode="focal_loss",
            debias_mode="adversarial",
            focal_gamma=1.5,
            use_stratified_split=False,
            early_stop_metric="fairness_score",
            early_stop_patience=8,
            class_weight_smoothing=0.25,
            node_validation_fraction=0.3,
            group_robust_weight=1.0,
            group_weight_mode="inverse_frequency",
            worst_group_boost_factor=3.0,
            min_group_support_threshold=4,
            min_group_support_train=6,
            min_group_support_eval=8,
            report_small_group_metrics=True,
            rebalance_batches_by_group=True,
            fairness_score_alpha=0.25,
            fairness_score_beta=0.5,
            probe_model_type="nonlinear",
            adversary_loss_weight=1.2,
            gradient_reversal_lambda=0.8,
            adversary_warmup_epochs=4,
            group_robust_warmup_epochs=6,
            adversary_hidden_dim=32,
            adversary_num_layers=2,
            adversary_dropout=0.1,
            min_support_boost_factor=2.5,
            protected_attribute_column="region",
        )

        kwargs = build_evaluation_kwargs(args)

        self.assertEqual(kwargs["node_classification_model"], "graphsage")
        self.assertEqual(kwargs["training_mode"], "adversarial_group_robust")
        self.assertEqual(kwargs["imbalance_mode"], "focal_loss")
        self.assertEqual(kwargs["debias_mode"], "adversarial")
        self.assertEqual(kwargs["focal_gamma"], 1.5)
        self.assertFalse(bool(kwargs["use_stratified_split"]))
        self.assertEqual(kwargs["early_stop_metric"], "fairness_score")
        self.assertEqual(kwargs["early_stop_patience"], 8)
        self.assertEqual(kwargs["class_weight_smoothing"], 0.25)
        self.assertEqual(kwargs["node_validation_fraction"], 0.3)
        self.assertEqual(kwargs["group_robust_weight"], 1.0)
        self.assertEqual(kwargs["group_weight_mode"], "inverse_frequency")
        self.assertEqual(kwargs["worst_group_boost_factor"], 3.0)
        self.assertEqual(kwargs["min_group_support_threshold"], 4)
        self.assertEqual(kwargs["min_group_support_train"], 6)
        self.assertEqual(kwargs["min_group_support_eval"], 8)
        self.assertTrue(bool(kwargs["report_small_group_metrics"]))
        self.assertTrue(bool(kwargs["rebalance_batches_by_group"]))
        self.assertEqual(kwargs["fairness_score_alpha"], 0.25)
        self.assertEqual(kwargs["fairness_score_beta"], 0.5)
        self.assertEqual(kwargs["probe_model_type"], "nonlinear")
        self.assertEqual(kwargs["adversary_loss_weight"], 1.2)
        self.assertEqual(kwargs["gradient_reversal_lambda"], 0.8)
        self.assertEqual(kwargs["adversary_warmup_epochs"], 4)
        self.assertEqual(kwargs["group_robust_warmup_epochs"], 6)
        self.assertEqual(kwargs["adversary_hidden_dim"], 32)
        self.assertEqual(kwargs["adversary_num_layers"], 2)
        self.assertEqual(kwargs["adversary_dropout"], 0.1)
        self.assertEqual(kwargs["min_support_boost_factor"], 2.5)
        self.assertEqual(kwargs["protected_attribute_column"], "region")

    def test_graphsage_comparison_variants_emit_requested_four_modes(self) -> None:
        variants = _graphsage_comparison_variants(
            {
                "node_classification_model": "graphsage",
                "protected_attribute_column": "region",
            }
        )

        self.assertEqual(
            [label for label, _ in variants],
            [
                "baseline",
                "support_aware_group_robust",
                "anti_collapse_group_robust",
                "anti_collapse_group_robust_with_mild_adversarial",
            ],
        )
        baseline_kwargs = variants[0][1]
        self.assertEqual(baseline_kwargs["training_mode"], "baseline")
        self.assertEqual(baseline_kwargs["group_weight_mode"], "none")
        self.assertEqual(baseline_kwargs["group_robust_weight"], 0.0)
        self.assertFalse(bool(baseline_kwargs["rebalance_batches_by_group"]))

    def test_raw_first_single_run_comparison_sort_prefers_higher_raw_worst_group_f1(self) -> None:
        comparison_frame = pd.DataFrame(
            [
                {
                    "method": "graphsage",
                    "comparison_mode": "baseline",
                    "status": "ok",
                    "training_mode": "baseline",
                    "debias_mode": "none",
                    "group_weight_mode": "none",
                    "group_robust_weight": 0.0,
                    "macro_f1": 0.93,
                    "worst_group_f1_raw": 0.24,
                    "worst_group_f1_supported": 0.92,
                    "macro_f1_gap_raw": 0.76,
                    "macro_f1_gap_supported": 0.05,
                    "protected_probe_macro_f1": 0.74,
                },
                {
                    "method": "graphsage",
                    "comparison_mode": "support_aware_group_robust",
                    "status": "ok",
                    "training_mode": "support_aware_group_robust",
                    "debias_mode": "none",
                    "group_weight_mode": "inverse_frequency",
                    "group_robust_weight": 0.25,
                    "macro_f1": 0.92,
                    "worst_group_f1_raw": 0.10,
                    "worst_group_f1_supported": 0.95,
                    "macro_f1_gap_raw": 0.90,
                    "macro_f1_gap_supported": 0.02,
                    "protected_probe_macro_f1": 0.70,
                },
            ]
        )

        ordered = _sorted_graphsage_comparison_frame(comparison_frame)

        self.assertEqual(ordered.iloc[0]["comparison_mode"], "baseline")
        self.assertAlmostEqual(float(ordered.iloc[0]["macro_f1_drop_vs_baseline"]), 0.0, places=6)
        self.assertAlmostEqual(float(ordered.iloc[1]["macro_f1_drop_vs_baseline"]), 0.01, places=6)

    def test_raw_first_repeated_comparison_sort_prefers_higher_raw_worst_group_f1(self) -> None:
        comparison_frame = pd.DataFrame(
            [
                {
                    "method": "graphsage",
                    "comparison_mode": "baseline",
                    "status": "ok",
                    "run_count": 9,
                    "mean_macro_f1": 0.925,
                    "std_macro_f1": 0.02,
                    "mean_worst_group_f1_raw": 0.18,
                    "std_worst_group_f1_raw": 0.14,
                    "mean_worst_group_f1_supported": 0.93,
                    "std_worst_group_f1_supported": 0.03,
                    "mean_macro_f1_gap_raw": 0.84,
                    "mean_macro_f1_gap_supported": 0.05,
                    "mean_protected_probe_macro_f1": 0.77,
                    "std_protected_probe_macro_f1": 0.12,
                },
                {
                    "method": "graphsage",
                    "comparison_mode": "support_aware_group_robust",
                    "status": "ok",
                    "run_count": 9,
                    "mean_macro_f1": 0.921,
                    "std_macro_f1": 0.02,
                    "mean_worst_group_f1_raw": 0.11,
                    "std_worst_group_f1_raw": 0.16,
                    "mean_worst_group_f1_supported": 0.94,
                    "std_worst_group_f1_supported": 0.03,
                    "mean_macro_f1_gap_raw": 0.89,
                    "mean_macro_f1_gap_supported": 0.04,
                    "mean_protected_probe_macro_f1": 0.75,
                    "std_protected_probe_macro_f1": 0.10,
                },
            ]
        )

        ordered = _sorted_graphsage_repeated_comparison_frame(comparison_frame)

        self.assertEqual(ordered.iloc[0]["comparison_mode"], "baseline")
        self.assertAlmostEqual(float(ordered.iloc[0]["macro_f1_drop_vs_baseline"]), 0.0, places=6)
        self.assertAlmostEqual(float(ordered.iloc[1]["macro_f1_drop_vs_baseline"]), 0.004, places=6)

    def test_format_benchmark_report_includes_statuses_and_paths(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "method": "deepwalk",
                    "status": "ok",
                    "runtime_seconds": 1.25,
                    "embedding_dim": 8,
                    "node_count": 500,
                    "ml_ready": True,
                    "all_nodes_embedded": True,
                    "all_finite": True,
                    "mean_pairwise_cosine": 0.25,
                    "label_probe_status": "ok",
                    "label_probe_accuracy": 0.75,
                    "label_probe_macro_f1": 0.72,
                    "label_probe_model_type": "linear",
                    "protected_probe_status": "ok",
                    "protected_probe_column": "region",
                    "protected_probe_accuracy": 0.55,
                    "protected_probe_macro_f1": 0.53,
                    "protected_probe_model_type": "linear",
                    "protected_probe_reason": "",
                    "csv_path": "results/toy/deepwalk.csv",
                    "pickle_path": "results/toy/deepwalk.pkl",
                    "npy_path": "results/toy/deepwalk.npy",
                    "skip_reason": "",
                    "error_message": "",
                },
                {
                    "method": "metapath2vec",
                    "status": "skipped",
                    "runtime_seconds": 0.0,
                    "embedding_dim": pd.NA,
                    "node_count": 500,
                    "ml_ready": False,
                    "all_nodes_embedded": False,
                    "all_finite": False,
                    "mean_pairwise_cosine": pd.NA,
                    "label_probe_status": "not_run",
                    "label_probe_accuracy": pd.NA,
                    "label_probe_macro_f1": pd.NA,
                    "label_probe_model_type": pd.NA,
                    "protected_probe_status": "not_requested",
                    "protected_probe_column": pd.NA,
                    "protected_probe_accuracy": pd.NA,
                    "protected_probe_macro_f1": pd.NA,
                    "protected_probe_model_type": pd.NA,
                    "protected_probe_reason": "",
                    "csv_path": pd.NA,
                    "pickle_path": pd.NA,
                    "npy_path": pd.NA,
                    "skip_reason": "requires a heterogeneous graph",
                    "error_message": "",
                },
            ]
        )
        evaluation_frame = pd.DataFrame(
            [
                {
                    "method": "deepwalk",
                    "task": "node_classification",
                    "status": "ok",
                    "runtime_seconds": 0.25,
                    "embedding_runtime_seconds": 1.25,
                    "embedding_dim": 8,
                    "evaluated_count": 100,
                    "train_count": 300,
                    "test_count": 100,
                    "classifier": "logistic_regression",
                    "label_column": "group",
                    "edge_feature": pd.NA,
                    "comparison_mode": "baseline",
                    "training_mode": "baseline",
                    "imbalance_mode": "none",
                    "early_stop_metric": "accuracy",
                    "debias_mode": "none",
                    "group_robust_weight": 0.0,
                    "group_weight_mode": "none",
                    "worst_group_boost_factor": 2.0,
                    "min_group_support_threshold": 5,
                    "min_group_support_train": 5,
                    "min_group_support_eval": 5,
                    "report_small_group_metrics": False,
                    "rebalance_batches_by_group": False,
                    "fairness_score_alpha": 0.25,
                    "fairness_score_beta": 0.25,
                    "adversary_loss_weight": pd.NA,
                    "gradient_reversal_lambda": pd.NA,
                    "adversary_hidden_dim": pd.NA,
                    "adversary_num_layers": pd.NA,
                    "adversary_dropout": pd.NA,
                    "accuracy": 0.8,
                    "macro_f1": 0.79,
                    "micro_f1": 0.8,
                    "weighted_f1": 0.78,
                    "balanced_accuracy": 0.77,
                    "macro_precision": 0.76,
                    "macro_recall": 0.75,
                    "roc_auc": pd.NA,
                    "average_precision": pd.NA,
                    "nmi": pd.NA,
                    "ari": pd.NA,
                    "protected_attribute_column": "region",
                    "worst_group_accuracy": 0.7,
                    "worst_group_accuracy_raw": 0.7,
                    "worst_group_accuracy_supported": 0.7,
                    "accuracy_gap": 0.2,
                    "accuracy_gap_raw": 0.2,
                    "accuracy_gap_supported": 0.2,
                    "worst_group_accuracy_name": "tiny",
                    "worst_group_accuracy_count": 2,
                    "worst_group_f1": 0.68,
                    "worst_group_f1_raw": 0.68,
                    "worst_group_f1_supported": 0.68,
                    "macro_f1_gap": 0.18,
                    "macro_f1_gap_raw": 0.18,
                    "macro_f1_gap_supported": 0.18,
                    "worst_group_f1_name": "tiny",
                    "worst_group_f1_count": 2,
                    "support_threshold_used": 5,
                    "supported_group_count": 1,
                    "group_support_counts_json": "[{\"group\":\"tiny\",\"count\":2,\"supported\":true}]",
                    "groups_below_support_threshold_json": "[]",
                    "protected_probe_status": "ok",
                    "protected_probe_accuracy": 0.61,
                    "protected_probe_macro_f1": 0.6,
                    "protected_probe_model_type": "linear",
                    "protected_probe_reason": "",
                    "split_diagnostics_json": "{\"train\":{\"count\":300,\"underrepresented_protected_groups\":[]},\"test\":{\"count\":100,\"underrepresented_protected_groups\":[{\"group\":\"tiny\",\"count\":2}],\"protected_missing_count\":0}}",
                    "collapsed_groups": "tiny",
                    "small_group_count": 1,
                    "max_group_error": 0.5,
                    "group_error_diagnostics_json": "[{\"group\":\"tiny\",\"prediction_collapse\":true,\"misclassification_rate\":0.5}]",
                    "best_epoch": 4,
                    "training_history_json": "[{\"epoch\":1,\"macro_f1\":0.7,\"collapsed_groups\":[]},{\"epoch\":4,\"macro_f1\":0.79,\"worst_group_f1_raw\":0.68,\"worst_group_f1_supported\":0.68,\"protected_probe_macro_f1\":0.6,\"collapsed_groups\":[\"tiny\"]}]",
                    "notes": "classes=3",
                    "unavailable_reason": "",
                    "skipped_reason": "",
                }
            ]
        )
        comparison_frame = pd.DataFrame(
            [
                {
                    "comparison_rank": 2,
                    "method": "deepwalk",
                    "comparison_mode": "baseline",
                    "status": "ok",
                    "training_mode": "baseline",
                    "debias_mode": "none",
                    "group_weight_mode": "none",
                    "group_robust_weight": 0.0,
                    "macro_f1": 0.82,
                    "worst_group_f1": 0.68,
                    "worst_group_f1_raw": 0.68,
                    "worst_group_f1_supported": 0.71,
                    "worst_group_accuracy": 0.7,
                    "macro_f1_gap": 0.16,
                    "macro_f1_gap_raw": 0.16,
                    "macro_f1_gap_supported": 0.14,
                    "protected_probe_macro_f1": 0.57,
                    "accuracy_gap": 0.19,
                    "macro_f1_drop_vs_baseline": 0.0,
                    "rebalance_batches_by_group": False,
                    "support_threshold_used": 5,
                    "small_group_count": 1,
                    "collapsed_groups": "tiny",
                },
                {
                    "comparison_rank": 1,
                    "method": "deepwalk",
                    "comparison_mode": "anti_collapse_group_robust",
                    "status": "ok",
                    "training_mode": "anti_collapse_group_robust",
                    "debias_mode": "none",
                    "group_weight_mode": "min_support_boost",
                    "group_robust_weight": 0.25,
                    "macro_f1": 0.81,
                    "worst_group_f1": 0.7,
                    "worst_group_f1_raw": 0.7,
                    "worst_group_f1_supported": 0.74,
                    "worst_group_accuracy": 0.72,
                    "macro_f1_gap": 0.15,
                    "macro_f1_gap_raw": 0.15,
                    "macro_f1_gap_supported": 0.12,
                    "protected_probe_macro_f1": 0.58,
                    "accuracy_gap": 0.18,
                    "macro_f1_drop_vs_baseline": 0.01,
                    "rebalance_batches_by_group": True,
                    "support_threshold_used": 5,
                    "small_group_count": 1,
                    "collapsed_groups": "tiny",
                }
            ]
        )
        tuning_frame = pd.DataFrame(
            [
                {
                    "tuning_rank": 1,
                    "method": "deepwalk",
                    "tuning_label": "adversarial:a=0.020,grl=0.020",
                    "comparison_mode": "adversarial",
                    "status": "ok",
                    "training_mode": "adversarial",
                    "debias_mode": "adversarial",
                    "group_weight_mode": "none",
                    "group_robust_weight": 0.0,
                    "adversary_loss_weight": 0.02,
                    "gradient_reversal_lambda": 0.02,
                    "macro_f1": 0.8,
                    "macro_f1_drop": 0.01,
                    "within_macro_f1_budget": True,
                    "worst_group_f1": 0.72,
                    "worst_group_accuracy": 0.74,
                    "macro_f1_gap": 0.14,
                    "protected_probe_macro_f1": 0.5,
                    "accuracy_gap": 0.16,
                }
            ]
        )

        report = format_benchmark_report(
            frame,
            evaluation_frame=evaluation_frame,
            comparison_frame=comparison_frame,
            tuning_frame=tuning_frame,
            dataset_name="toy_graph",
            output_dir=Path("results"),
        )

        self.assertIn("Graph Embedding Benchmark Summary", report)
        self.assertIn("Dataset: toy_graph", report)
        self.assertIn("deepwalk [ok]", report)
        self.assertIn("metapath2vec [skipped]", report)
        self.assertIn("exports: csv=results/toy/deepwalk.csv", report)
        self.assertIn("skip_reason=requires a heterogeneous graph", report)
        self.assertIn("Graph Embedding Evaluation Summary", report)
        self.assertIn("deepwalk | node_classification [ok]", report)
        self.assertIn("metrics: accuracy=0.8000", report)
        self.assertIn("weighted_f1=0.7800", report)
        self.assertIn("balanced_accuracy=0.7700", report)
        self.assertIn("macro_precision=0.7600", report)
        self.assertIn("macro_recall=0.7500", report)
        self.assertIn("fairness_raw: protected_attribute=region", report)
        self.assertIn("fairness_supported: support_threshold=5 | supported_groups=1", report)
        self.assertIn("worst_groups: accuracy=tiny(n=2) | f1=tiny(n=2)", report)
        self.assertIn("training: comparison_mode=baseline | mode=baseline | debias_mode=none | imbalance_mode=none | group_weight_mode=none | group_robust_weight=0.0000 | rebalance_batches_by_group=False | early_stop_metric=accuracy", report)
        self.assertIn("leakage: protected_probe=ok | accuracy=0.6100 | macro_f1=0.6000 | type=linear", report)
        self.assertIn("diagnostics: train:n=300,groups=-,small_groups=0,missing=0,threshold=- | test:n=100,groups=-,small_groups=1,missing=0,threshold=-", report)
        self.assertIn("diagnostics_summary: collapsed_groups=tiny | small_group_count=1 | max_group_error=0.5000", report)
        self.assertIn("support_counts: tiny:2 | below_threshold=-", report)
        self.assertIn("group_errors: collapse_groups=tiny | max_group_error=0.5000", report)
        self.assertIn("history: best_epoch=4 | epochs=2 | collapse_epochs=1 | last_worst_group_f1_raw=0.6800 | last_worst_group_f1_supported=0.6800 | last_probe_macro_f1=0.6000", report)
        self.assertIn("protected_probe=ok | column=region | accuracy=0.5500 | macro_f1=0.5300 | type=linear", report)
        self.assertIn("GraphSAGE Fairness Comparison", report)
        self.assertIn("1. deepwalk | anti_collapse_group_robust [ok]", report)
        self.assertIn("Ranked by worst_group_f1_raw desc, worst_group_f1_supported desc", report)
        self.assertIn("worst_group_f1_supported=0.7400", report)
        self.assertIn("training: mode=anti_collapse_group_robust | debias_mode=none | group_weight_mode=min_support_boost | group_robust_weight=0.2500 | macro_f1_drop_vs_baseline=0.0100 | rebalance=True", report)
        self.assertIn("GraphSAGE Mild Tuning Sweep", report)
        self.assertIn("1. deepwalk | adversarial:a=0.020,grl=0.020 [ok]", report)
        self.assertIn("within_budget=true", report)

    def test_format_benchmark_report_sorts_evaluation_rows_by_best_metric(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "method": "alpha",
                    "status": "ok",
                    "runtime_seconds": 1.0,
                    "embedding_dim": 8,
                    "node_count": 100,
                    "ml_ready": True,
                    "all_nodes_embedded": True,
                    "all_finite": True,
                    "mean_pairwise_cosine": 0.1,
                    "label_probe_status": "not_requested",
                    "label_probe_accuracy": pd.NA,
                    "label_probe_macro_f1": pd.NA,
                    "protected_probe_status": "not_requested",
                    "protected_probe_column": pd.NA,
                    "protected_probe_accuracy": pd.NA,
                    "protected_probe_macro_f1": pd.NA,
                    "protected_probe_reason": "",
                    "csv_path": pd.NA,
                    "pickle_path": pd.NA,
                    "npy_path": pd.NA,
                    "skip_reason": "",
                    "error_message": "",
                },
                {
                    "method": "beta",
                    "status": "ok",
                    "runtime_seconds": 1.0,
                    "embedding_dim": 8,
                    "node_count": 100,
                    "ml_ready": True,
                    "all_nodes_embedded": True,
                    "all_finite": True,
                    "mean_pairwise_cosine": 0.1,
                    "label_probe_status": "not_requested",
                    "label_probe_accuracy": pd.NA,
                    "label_probe_macro_f1": pd.NA,
                    "protected_probe_status": "not_requested",
                    "protected_probe_column": pd.NA,
                    "protected_probe_accuracy": pd.NA,
                    "protected_probe_macro_f1": pd.NA,
                    "protected_probe_reason": "",
                    "csv_path": pd.NA,
                    "pickle_path": pd.NA,
                    "npy_path": pd.NA,
                    "skip_reason": "",
                    "error_message": "",
                },
                {
                    "method": "gamma",
                    "status": "skipped",
                    "runtime_seconds": 0.0,
                    "embedding_dim": pd.NA,
                    "node_count": 100,
                    "ml_ready": False,
                    "all_nodes_embedded": False,
                    "all_finite": False,
                    "mean_pairwise_cosine": pd.NA,
                    "label_probe_status": "not_run",
                    "label_probe_accuracy": pd.NA,
                    "label_probe_macro_f1": pd.NA,
                    "protected_probe_status": "not_requested",
                    "protected_probe_column": pd.NA,
                    "protected_probe_accuracy": pd.NA,
                    "protected_probe_macro_f1": pd.NA,
                    "protected_probe_reason": "",
                    "csv_path": pd.NA,
                    "pickle_path": pd.NA,
                    "npy_path": pd.NA,
                    "skip_reason": "unsupported",
                    "error_message": "",
                },
            ]
        )
        evaluation_frame = pd.DataFrame(
            [
                {
                    "method": "alpha",
                    "task": "node_classification",
                    "status": "ok",
                    "runtime_seconds": 0.2,
                    "embedding_runtime_seconds": 1.0,
                    "embedding_dim": 8,
                    "node_count": 100,
                    "evaluated_count": 20,
                    "train_count": 80,
                    "test_count": 20,
                    "label_column": "group",
                    "classifier": "logistic_regression",
                    "edge_feature": pd.NA,
                    "training_mode": "baseline",
                    "imbalance_mode": "none",
                    "early_stop_metric": "accuracy",
                    "debias_mode": "none",
                    "adversary_loss_weight": pd.NA,
                    "gradient_reversal_lambda": pd.NA,
                    "adversary_hidden_dim": pd.NA,
                    "adversary_num_layers": pd.NA,
                    "adversary_dropout": pd.NA,
                    "accuracy": 0.75,
                    "macro_f1": 0.74,
                    "micro_f1": 0.75,
                    "weighted_f1": 0.73,
                    "balanced_accuracy": 0.72,
                    "macro_precision": 0.71,
                    "macro_recall": 0.70,
                    "roc_auc": pd.NA,
                    "average_precision": pd.NA,
                    "nmi": pd.NA,
                    "ari": pd.NA,
                    "notes": "",
                    "unavailable_reason": "",
                    "skipped_reason": "",
                },
                {
                    "method": "beta",
                    "task": "node_classification",
                    "status": "ok",
                    "runtime_seconds": 0.2,
                    "embedding_runtime_seconds": 1.0,
                    "embedding_dim": 8,
                    "node_count": 100,
                    "evaluated_count": 20,
                    "train_count": 80,
                    "test_count": 20,
                    "label_column": "group",
                    "classifier": "logistic_regression",
                    "edge_feature": pd.NA,
                    "training_mode": "baseline",
                    "imbalance_mode": "none",
                    "early_stop_metric": "accuracy",
                    "debias_mode": "none",
                    "adversary_loss_weight": pd.NA,
                    "gradient_reversal_lambda": pd.NA,
                    "adversary_hidden_dim": pd.NA,
                    "adversary_num_layers": pd.NA,
                    "adversary_dropout": pd.NA,
                    "accuracy": 0.91,
                    "macro_f1": 0.90,
                    "micro_f1": 0.91,
                    "weighted_f1": 0.90,
                    "balanced_accuracy": 0.89,
                    "macro_precision": 0.88,
                    "macro_recall": 0.87,
                    "roc_auc": pd.NA,
                    "average_precision": pd.NA,
                    "nmi": pd.NA,
                    "ari": pd.NA,
                    "notes": "",
                    "unavailable_reason": "",
                    "skipped_reason": "",
                },
                {
                    "method": "gamma",
                    "task": "node_classification",
                    "status": "skipped",
                    "runtime_seconds": 0.0,
                    "embedding_runtime_seconds": 0.0,
                    "embedding_dim": 0,
                    "node_count": 100,
                    "evaluated_count": pd.NA,
                    "train_count": pd.NA,
                    "test_count": pd.NA,
                    "label_column": "group",
                    "classifier": "logistic_regression",
                    "edge_feature": pd.NA,
                    "training_mode": "baseline",
                    "imbalance_mode": "none",
                    "early_stop_metric": "accuracy",
                    "debias_mode": "none",
                    "adversary_loss_weight": pd.NA,
                    "gradient_reversal_lambda": pd.NA,
                    "adversary_hidden_dim": pd.NA,
                    "adversary_num_layers": pd.NA,
                    "adversary_dropout": pd.NA,
                    "accuracy": pd.NA,
                    "macro_f1": pd.NA,
                    "micro_f1": pd.NA,
                    "weighted_f1": pd.NA,
                    "balanced_accuracy": pd.NA,
                    "macro_precision": pd.NA,
                    "macro_recall": pd.NA,
                    "roc_auc": pd.NA,
                    "average_precision": pd.NA,
                    "nmi": pd.NA,
                    "ari": pd.NA,
                    "notes": "",
                    "unavailable_reason": "",
                    "skipped_reason": "unsupported",
                },
            ]
        )

        report = format_benchmark_report(
            frame,
            evaluation_frame=evaluation_frame,
            dataset_name="toy_graph",
            output_dir=Path("results"),
        )

        self.assertLess(report.index("beta | node_classification [ok]"), report.index("alpha | node_classification [ok]"))
        self.assertLess(report.index("alpha | node_classification [ok]"), report.index("gamma | node_classification [skipped]"))
        self.assertIn("primary_metric=accuracy (0.9100)", report)

    def test_print_saved_benchmark_summary_includes_evaluation_metrics(self) -> None:
        result = SimpleNamespace(
            summary_frame=pd.DataFrame(
                [
                    {
                        "method": "dgi",
                        "status": "ok",
                        "runtime_seconds": 2.0,
                        "embedding_dim": 64,
                        "node_count": 100,
                        "ml_ready": True,
                        "all_nodes_embedded": True,
                        "all_finite": True,
                        "mean_pairwise_cosine": 0.2,
                        "label_probe_status": "ok",
                        "label_probe_accuracy": 0.65,
                        "label_probe_macro_f1": 0.63,
                        "label_probe_model_type": "linear",
                        "protected_probe_status": "ok",
                        "protected_probe_column": "region",
                        "protected_probe_accuracy": 0.55,
                        "protected_probe_macro_f1": 0.52,
                        "protected_probe_model_type": "linear",
                        "protected_probe_reason": "",
                        "csv_path": pd.NA,
                        "pickle_path": pd.NA,
                        "npy_path": pd.NA,
                        "skip_reason": "",
                        "error_message": "",
                    }
                ]
            ),
            evaluation_frame=pd.DataFrame(
                [
                    {
                        "method": "dgi",
                        "task": "node_classification",
                        "status": "ok",
                        "runtime_seconds": 0.2,
                        "embedding_runtime_seconds": 2.0,
                        "embedding_dim": 64,
                        "evaluated_count": 25,
                        "train_count": 75,
                        "test_count": 25,
                        "classifier": "graphsage",
                        "label_column": "group",
                        "edge_feature": pd.NA,
                        "training_mode": "adversarial",
                        "imbalance_mode": "focal_loss",
                        "early_stop_metric": "worst_group_f1_raw",
                        "debias_mode": "adversarial",
                        "group_robust_weight": 0.0,
                        "group_weight_mode": "none",
                        "min_group_support_train": 10,
                        "min_group_support_eval": 10,
                        "report_small_group_metrics": False,
                        "rebalance_batches_by_group": False,
                        "fairness_score_alpha": 0.25,
                        "fairness_score_beta": 0.25,
                        "adversary_loss_weight": 1.0,
                        "gradient_reversal_lambda": 0.8,
                        "adversary_hidden_dim": 32,
                        "adversary_num_layers": 2,
                        "adversary_dropout": 0.1,
                        "accuracy": 0.8,
                        "macro_f1": 0.77,
                        "micro_f1": 0.8,
                        "weighted_f1": 0.79,
                        "balanced_accuracy": 0.78,
                        "macro_precision": 0.76,
                        "macro_recall": 0.75,
                        "roc_auc": pd.NA,
                        "average_precision": pd.NA,
                        "nmi": pd.NA,
                        "ari": pd.NA,
                        "protected_attribute_column": "region",
                        "worst_group_accuracy": 0.6,
                        "worst_group_accuracy_raw": 0.6,
                        "accuracy_gap": 0.25,
                        "accuracy_gap_raw": 0.25,
                        "worst_group_f1": 0.58,
                        "worst_group_f1_raw": 0.58,
                        "macro_f1_gap": 0.2,
                        "macro_f1_gap_raw": 0.2,
                        "protected_probe_status": "ok",
                        "protected_probe_accuracy": 0.49,
                        "protected_probe_macro_f1": 0.48,
                        "protected_probe_model_type": "linear",
                        "protected_probe_reason": "",
                        "notes": "classes=3; stratified=True; debias_mode=adversarial; imbalance_mode=focal_loss",
                        "unavailable_reason": "",
                        "skipped_reason": "",
                    }
                ]
            ),
        )
        buffer = StringIO()
        with redirect_stdout(buffer):
            print_saved_benchmark_summary(result, report_path=Path("results/toy_graph/report.txt"))

        output = buffer.getvalue()
        self.assertIn("Saved benchmark report to results\\toy_graph\\report.txt", output)
        self.assertIn("Method status counts: {'ok': 1}", output)
        self.assertIn("Evaluation status counts: {'ok': 1}", output)
        self.assertIn("dgi | node_classification [ok]", output)
        self.assertIn("macro_f1=0.7700", output)
        self.assertIn("micro_f1=0.8000", output)
        self.assertIn("weighted_f1=0.7900", output)
        self.assertIn("balanced_accuracy=0.7800", output)
        self.assertIn("protected_probe=ok | column=region | accuracy=0.5500 | macro_f1=0.5200 | type=linear", output)
        self.assertIn("fairness_raw: protected_attribute=region", output)
        self.assertIn("training: mode=adversarial | debias_mode=adversarial | imbalance_mode=focal_loss | group_weight_mode=none | group_robust_weight=0.0000 | rebalance_batches_by_group=False | early_stop_metric=worst_group_f1_raw", output)
        self.assertIn("leakage: protected_probe=ok | accuracy=0.4900 | macro_f1=0.4800 | type=linear", output)

    def test_render_saved_benchmark_summary_text_can_be_saved_to_report_file(self) -> None:
        result = SimpleNamespace(
            summary_frame=pd.DataFrame(
                [
                    {
                        "method": "graphsage",
                        "status": "ok",
                        "runtime_seconds": 1.0,
                        "embedding_dim": 64,
                        "node_count": 100,
                        "ml_ready": True,
                        "all_nodes_embedded": True,
                        "all_finite": True,
                        "mean_pairwise_cosine": 0.2,
                        "label_probe_status": "ok",
                        "label_probe_accuracy": 0.7,
                        "label_probe_macro_f1": 0.69,
                        "protected_probe_status": "not_requested",
                        "protected_probe_column": pd.NA,
                        "protected_probe_accuracy": pd.NA,
                        "protected_probe_macro_f1": pd.NA,
                        "protected_probe_reason": "",
                        "csv_path": pd.NA,
                        "pickle_path": pd.NA,
                        "npy_path": pd.NA,
                        "skip_reason": "",
                        "error_message": "",
                    }
                ]
            ),
            evaluation_frame=pd.DataFrame(
                [
                    {
                        "method": "graphsage",
                        "task": "node_classification",
                        "status": "ok",
                        "runtime_seconds": 0.1,
                        "embedding_runtime_seconds": 1.0,
                        "embedding_dim": 64,
                        "evaluated_count": 25,
                        "train_count": 75,
                        "test_count": 25,
                        "classifier": "graphsage",
                        "label_column": "group",
                        "edge_feature": pd.NA,
                        "accuracy": 0.84,
                        "macro_f1": 0.83,
                        "micro_f1": 0.84,
                        "weighted_f1": 0.84,
                        "balanced_accuracy": 0.82,
                        "macro_precision": 0.81,
                        "macro_recall": 0.82,
                        "roc_auc": pd.NA,
                        "average_precision": pd.NA,
                        "nmi": pd.NA,
                        "ari": pd.NA,
                        "notes": "classes=2",
                        "unavailable_reason": "",
                        "skipped_reason": "",
                    }
                ]
            ),
        )
        with _workspace_tempdir() as temp_dir:
            output_dir = Path(temp_dir)
            report_path = benchmark_report_path(output_dir, "toy_graph")
            self.assertIsNotNone(report_path)
            report_text = render_saved_benchmark_summary_text(result, report_path=report_path)
            saved_path = save_benchmark_report(
                report_text,
                output_dir=output_dir,
                dataset_name="toy_graph",
            )

            self.assertEqual(saved_path, report_path)
            saved_text = report_path.read_text(encoding="utf-8")
            self.assertIn(f"Saved benchmark report to {report_path}", saved_text)
            self.assertIn("Method status counts: {'ok': 1}", saved_text)
            self.assertIn("Evaluation status counts: {'ok': 1}", saved_text)
            self.assertIn("graphsage | node_classification [ok]", saved_text)

    def test_format_benchmark_report_shows_unavailable_reason_when_fairness_is_skipped(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "method": "deepwalk",
                    "status": "ok",
                    "runtime_seconds": 1.0,
                    "embedding_dim": 8,
                    "node_count": 100,
                    "ml_ready": True,
                    "all_nodes_embedded": True,
                    "all_finite": True,
                    "mean_pairwise_cosine": 0.1,
                    "label_probe_status": "not_requested",
                    "label_probe_accuracy": pd.NA,
                    "label_probe_macro_f1": pd.NA,
                    "protected_probe_status": "not_requested",
                    "protected_probe_column": pd.NA,
                    "protected_probe_accuracy": pd.NA,
                    "protected_probe_macro_f1": pd.NA,
                    "protected_probe_reason": "",
                    "csv_path": pd.NA,
                    "pickle_path": pd.NA,
                    "npy_path": pd.NA,
                    "skip_reason": "",
                    "error_message": "",
                }
            ]
        )
        evaluation_frame = pd.DataFrame(
            [
                {
                    "method": "deepwalk",
                    "task": "node_classification",
                    "status": "ok",
                    "runtime_seconds": 0.1,
                    "embedding_runtime_seconds": 1.0,
                    "embedding_dim": 8,
                    "node_count": 100,
                    "evaluated_count": 25,
                    "train_count": 75,
                    "test_count": 25,
                    "label_column": "group",
                    "classifier": "logistic_regression",
                    "edge_feature": pd.NA,
                    "training_mode": "baseline",
                    "imbalance_mode": "none",
                    "early_stop_metric": "accuracy",
                    "debias_mode": "none",
                    "adversary_loss_weight": pd.NA,
                    "gradient_reversal_lambda": pd.NA,
                    "adversary_hidden_dim": pd.NA,
                    "adversary_num_layers": pd.NA,
                    "adversary_dropout": pd.NA,
                    "accuracy": 0.8,
                    "macro_f1": 0.79,
                    "micro_f1": 0.8,
                    "weighted_f1": 0.78,
                    "balanced_accuracy": 0.77,
                    "macro_precision": 0.76,
                    "macro_recall": 0.75,
                    "roc_auc": pd.NA,
                    "average_precision": pd.NA,
                    "nmi": pd.NA,
                    "ari": pd.NA,
                    "protected_attribute_column": "missing_region",
                    "worst_group_accuracy": pd.NA,
                    "group_accuracy_gap": pd.NA,
                    "worst_group_macro_f1": pd.NA,
                    "group_macro_f1_gap": pd.NA,
                    "notes": "classes=3",
                    "unavailable_reason": "Protected attribute column 'missing_region' is not present in dataset.node_attributes.",
                    "skipped_reason": "",
                }
            ]
        )

        report = format_benchmark_report(
            frame,
            evaluation_frame=evaluation_frame,
            dataset_name="toy_graph",
            output_dir=Path("results"),
        )

        self.assertIn("unavailable_reason=Protected attribute column 'missing_region' is not present", report)

    def test_save_benchmark_report_writes_text_file(self) -> None:
        with _workspace_tempdir() as temp_dir:
            output_dir = Path(temp_dir)
            report_path = save_benchmark_report(
                "benchmark report body",
                output_dir=output_dir,
                dataset_name="toy_graph",
            )

            self.assertEqual(report_path, benchmark_report_path(output_dir, "toy_graph"))
            self.assertIsNotNone(report_path)
            self.assertTrue(report_path.is_file())
            self.assertEqual(report_path.read_text(encoding="utf-8"), "benchmark report body")

    def test_save_attribute_benchmark_report_writes_text_file(self) -> None:
        with _workspace_tempdir() as temp_dir:
            output_dir = Path(temp_dir)
            report_path = save_attribute_benchmark_report(
                "attribute report body",
                output_dir=output_dir,
                dataset_name="graph_spa_500_0",
                attribute_name="gender",
            )

            self.assertEqual(report_path, attribute_report_path(output_dir, "graph_spa_500_0", "gender"))
            self.assertIsNotNone(report_path)
            self.assertTrue(report_path.is_file())
            self.assertEqual(report_path.read_text(encoding="utf-8"), "attribute report body")

    def test_save_all_attributes_benchmark_report_writes_text_file(self) -> None:
        with _workspace_tempdir() as temp_dir:
            output_dir = Path(temp_dir)
            report_text = format_all_attributes_benchmark_report(
                "graph_spa_500_0",
                base_report_text="base benchmark body",
                attribute_reports=[
                    {
                        "attribute_name": "gender",
                        "report_text": "gender body",
                        "report_path": Path("results/graph_spa_500_0/reports/graph_spa_500_0_embedding_benchmark_report_gender.txt"),
                    },
                    {
                        "attribute_name": "ethnicity",
                        "report_text": "ethnicity body",
                        "report_path": Path("results/graph_spa_500_0/reports/graph_spa_500_0_embedding_benchmark_report_ethnicity.txt"),
                    },
                ],
                output_dir=output_dir,
            )
            report_path = save_all_attributes_benchmark_report(
                report_text,
                output_dir=output_dir,
                dataset_name="graph_spa_500_0",
            )

            self.assertIsNotNone(report_path)
            self.assertTrue(report_path.is_file())
            self.assertIn("All-Attributes Graph Embedding Benchmark Summary", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
