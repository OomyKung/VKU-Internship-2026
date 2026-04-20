"""Tests for the embedding benchmark CLI helpers."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import shutil
import unittest
from uuid import uuid4

import pandas as pd

from scripts.run_embedding_benchmark import (
    _all_attributes_requested,
    benchmark_report_path,
    build_dataset_config,
    build_method_configs,
    attribute_report_path,
    format_benchmark_report,
    format_all_attributes_benchmark_report,
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
                    "accuracy": 0.8,
                    "macro_f1": 0.79,
                    "micro_f1": 0.8,
                    "roc_auc": pd.NA,
                    "average_precision": pd.NA,
                    "nmi": pd.NA,
                    "ari": pd.NA,
                    "notes": "classes=3",
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

        self.assertIn("Graph Embedding Benchmark Summary", report)
        self.assertIn("Dataset: toy_graph", report)
        self.assertIn("deepwalk [ok]", report)
        self.assertIn("metapath2vec [skipped]", report)
        self.assertIn("exports: csv=results/toy/deepwalk.csv", report)
        self.assertIn("skip_reason=requires a heterogeneous graph", report)
        self.assertIn("Graph Embedding Evaluation Summary", report)
        self.assertIn("deepwalk | node_classification [ok]", report)
        self.assertIn("metrics: accuracy=0.8000", report)

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
                    "accuracy": 0.75,
                    "macro_f1": 0.74,
                    "micro_f1": 0.75,
                    "roc_auc": pd.NA,
                    "average_precision": pd.NA,
                    "nmi": pd.NA,
                    "ari": pd.NA,
                    "notes": "",
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
                    "accuracy": 0.91,
                    "macro_f1": 0.90,
                    "micro_f1": 0.91,
                    "roc_auc": pd.NA,
                    "average_precision": pd.NA,
                    "nmi": pd.NA,
                    "ari": pd.NA,
                    "notes": "",
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
                    "accuracy": pd.NA,
                    "macro_f1": pd.NA,
                    "micro_f1": pd.NA,
                    "roc_auc": pd.NA,
                    "average_precision": pd.NA,
                    "nmi": pd.NA,
                    "ari": pd.NA,
                    "notes": "",
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
