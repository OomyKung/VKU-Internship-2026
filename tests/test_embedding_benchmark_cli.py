"""Tests for the embedding benchmark CLI helpers."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import shutil
import unittest
from uuid import uuid4

import pandas as pd

from scripts.run_embedding_benchmark import build_dataset_config, build_method_configs, format_benchmark_report


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


if __name__ == "__main__":
    unittest.main()
