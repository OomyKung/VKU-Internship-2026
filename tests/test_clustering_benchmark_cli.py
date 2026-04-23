"""Tests for the clustering benchmark CLI helpers."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import shutil
import unittest
from uuid import uuid4

import pandas as pd

from scripts.run_clustering_benchmark import (
    build_dataset_config,
    clustering_report_path,
    format_clustering_report,
    save_clustering_report,
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


class ClusteringBenchmarkCliTestCase(unittest.TestCase):
    """Check the clustering benchmark CLI helper surfaces."""

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

    def test_format_clustering_report_includes_metrics_and_skip_reason(self) -> None:
        summary_frame = pd.DataFrame(
            [
                {
                    "method": "louvain",
                    "category": "graph_native",
                    "status": "ok",
                    "requested_input_mode": "graph",
                    "resolved_input_mode": "graph",
                    "runtime_seconds": 0.12,
                    "num_clusters": 4,
                    "largest_cluster_size": 20,
                    "smallest_cluster_size": 4,
                    "average_cluster_size": 8.0,
                    "cluster_size_std": 2.1,
                    "cluster_size_summary": "0:20,1:8,2:6,3:4",
                    "modularity": 0.41,
                    "mean_conductance": 0.18,
                    "silhouette_score": pd.NA,
                    "davies_bouldin_score": pd.NA,
                    "calinski_harabasz_score": pd.NA,
                    "nmi": 0.52,
                    "ari": 0.47,
                    "assignments_csv_path": "results/toy/louvain_assignments.csv",
                    "cluster_sizes_csv_path": "results/toy/louvain_sizes.csv",
                    "skip_reason": "",
                },
                {
                    "method": "gmm",
                    "category": "embedding_space",
                    "status": "skipped",
                    "requested_input_mode": "embedding",
                    "resolved_input_mode": pd.NA,
                    "runtime_seconds": 0.0,
                    "num_clusters": pd.NA,
                    "largest_cluster_size": pd.NA,
                    "smallest_cluster_size": pd.NA,
                    "average_cluster_size": pd.NA,
                    "cluster_size_std": pd.NA,
                    "cluster_size_summary": "",
                    "modularity": pd.NA,
                    "mean_conductance": pd.NA,
                    "silhouette_score": pd.NA,
                    "davies_bouldin_score": pd.NA,
                    "calinski_harabasz_score": pd.NA,
                    "nmi": pd.NA,
                    "ari": pd.NA,
                    "assignments_csv_path": pd.NA,
                    "cluster_sizes_csv_path": pd.NA,
                    "skip_reason": "Embedding-space clustering requires embeddings or numeric feature vectors.",
                },
            ]
        )

        report = format_clustering_report(summary_frame, dataset_name="toy_graph")

        self.assertIn("Clustering Benchmark Summary", report)
        self.assertIn("1. louvain | graph_native [ok]", report)
        self.assertIn("metrics: modularity=0.4100 | conductance=0.1800", report)
        self.assertIn("outputs: assignments=results/toy/louvain_assignments.csv", report)
        self.assertIn("2. gmm | embedding_space [skipped]", report)
        self.assertIn("skip_reason=Embedding-space clustering requires embeddings or numeric feature vectors.", report)

    def test_save_clustering_report_writes_text_file(self) -> None:
        with _workspace_tempdir() as temp_dir:
            output_dir = Path(temp_dir)
            report_path = save_clustering_report(
                "clustering report body",
                output_dir=output_dir,
                dataset_name="toy_graph",
            )

            self.assertEqual(report_path, clustering_report_path(output_dir, "toy_graph"))
            self.assertIsNotNone(report_path)
            self.assertTrue(report_path.is_file())
            self.assertEqual(report_path.read_text(encoding="utf-8"), "clustering report body")


if __name__ == "__main__":
    unittest.main()
