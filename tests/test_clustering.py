"""Unit tests for the shared clustering framework."""

from __future__ import annotations

from pathlib import Path
import shutil
import unittest
from uuid import uuid4

import networkx as nx
import numpy as np
import pandas as pd

try:
    import igraph as ig
except ImportError:
    ig = None

from fim_hybrid.clustering import (
    ClusteringInputError,
    available_clustering_methods,
    cluster_nodes,
    run_clustering_benchmark,
)
from fim_hybrid.data_loader import LoadedDataset


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


def _toy_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (1, 2),
            (1, 3),
            (2, 3),
            (4, 5),
            (4, 6),
            (5, 6),
            (3, 4),
        ]
    )
    return graph


def _toy_dataset() -> LoadedDataset:
    graph = _toy_graph()
    node_attributes = pd.DataFrame(
        [
            {"node_id": 1, "group": "A"},
            {"node_id": 2, "group": "A"},
            {"node_id": 3, "group": "A"},
            {"node_id": 4, "group": "B"},
            {"node_id": 5, "group": "B"},
            {"node_id": 6, "group": "B"},
        ]
    ).set_index("node_id", drop=False)
    return LoadedDataset(name="toy_cluster", graph=graph, node_attributes=node_attributes)


def _toy_embeddings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "node_id": [1, 2, 3, 4, 5, 6],
            "embedding_0": [0.0, 0.1, 0.2, 10.0, 10.1, 10.2],
            "embedding_1": [0.0, 0.2, 0.1, 10.0, 10.2, 10.1],
        }
    )


class ClusteringFrameworkTestCase(unittest.TestCase):
    """Check the shared clustering wrappers and benchmark runner."""

    def test_available_methods_include_graph_and_embedding_variants(self) -> None:
        self.assertEqual(
            available_clustering_methods(),
            (
                "louvain",
                "leiden",
                "multilevel",
                "infomap",
                "label_propagation",
                "walktrap",
                "kmeans",
                "spectral",
                "agglomerative",
                "dbscan_or_hdbscan",
                "gmm",
            ),
        )

    def test_graph_native_louvain_assigns_every_node(self) -> None:
        graph = _toy_graph()
        result = cluster_nodes(graph, "louvain", random_seed=7, input_mode="graph")

        self.assertEqual(set(result.assignment_frame["node_id"]), set(graph.nodes()))
        self.assertEqual(set(result.cluster_id_by_node), set(graph.nodes()))
        self.assertEqual(sum(result.cluster_sizes.values()), graph.number_of_nodes())
        self.assertEqual(result.category, "graph_native")
        self.assertEqual(result.resolved_input_mode, "graph")

    @unittest.skipUnless(ig is not None, "igraph is unavailable")
    def test_graph_native_igraph_methods_assign_every_node(self) -> None:
        graph = _toy_graph()
        for method_name in ["leiden", "multilevel", "infomap", "walktrap"]:
            result = cluster_nodes(graph, method_name, random_seed=7, input_mode="graph")
            self.assertEqual(set(result.cluster_id_by_node), set(graph.nodes()))
            self.assertGreaterEqual(result.num_clusters, 1)

    def test_label_propagation_assigns_every_node(self) -> None:
        graph = _toy_graph()
        result = cluster_nodes(graph, "label_propagation", random_seed=7, input_mode="graph")

        self.assertEqual(set(result.cluster_id_by_node), set(graph.nodes()))
        self.assertGreaterEqual(result.num_clusters, 1)

    def test_embedding_space_kmeans_uses_embedding_frame(self) -> None:
        graph = _toy_graph()
        embeddings = _toy_embeddings()
        labels = pd.Series(["A", "A", "A", "B", "B", "B"], index=[1, 2, 3, 4, 5, 6])

        result = cluster_nodes(
            graph,
            "kmeans",
            embeddings=embeddings,
            labels=labels,
            config={"n_clusters": 2},
            random_seed=7,
            input_mode="embedding",
        )

        self.assertEqual(result.category, "embedding_space")
        self.assertEqual(result.resolved_input_mode, "embedding")
        self.assertEqual(result.num_clusters, 2)
        self.assertIsNotNone(result.silhouette_score)
        self.assertIsNotNone(result.nmi)
        self.assertIsNotNone(result.ari)

    def test_embedding_space_methods_can_reuse_label_cardinality(self) -> None:
        graph = _toy_graph()
        embeddings = _toy_embeddings()
        labels = pd.Series(["A", "A", "A", "B", "B", "B"], index=[1, 2, 3, 4, 5, 6])

        for method_name in ["kmeans", "spectral", "agglomerative", "gmm"]:
            result = cluster_nodes(
                graph,
                method_name,
                embeddings=embeddings,
                labels=labels,
                random_seed=7,
                input_mode="embedding",
            )
            self.assertEqual(set(result.cluster_id_by_node), set(graph.nodes()))
            self.assertGreaterEqual(result.num_clusters, 1)

    def test_dbscan_or_hdbscan_assigns_every_node(self) -> None:
        graph = _toy_graph()
        embeddings = _toy_embeddings()
        result = cluster_nodes(
            graph,
            "dbscan_or_hdbscan",
            embeddings=embeddings,
            config={"eps": 0.6, "min_samples": 2},
            random_seed=7,
            input_mode="embedding",
        )

        self.assertEqual(set(result.cluster_id_by_node), set(graph.nodes()))
        self.assertGreaterEqual(result.num_clusters, 1)

    def test_embedding_space_clustering_can_fallback_to_prepared_features(self) -> None:
        dataset = _toy_dataset()
        result = cluster_nodes(
            dataset.graph,
            "kmeans",
            dataset=dataset,
            config={"n_clusters": 2},
            random_seed=7,
            input_mode="auto",
        )

        self.assertEqual(result.resolved_input_mode, "feature")
        self.assertEqual(set(result.cluster_id_by_node), set(dataset.graph.nodes()))

    def test_embedding_space_methods_fail_clearly_without_input(self) -> None:
        graph = _toy_graph()

        with self.assertRaisesRegex(ClusteringInputError, "requires embeddings or numeric feature vectors"):
            cluster_nodes(graph, "kmeans", input_mode="embedding", random_seed=7)

    def test_benchmark_runner_saves_assignments_and_sizes(self) -> None:
        dataset = _toy_dataset()
        embeddings = _toy_embeddings()
        with _workspace_tempdir() as temp_dir:
            output_dir = Path(temp_dir)
            result = run_clustering_benchmark(
                dataset,
                methods=["louvain", "kmeans"],
                embeddings=embeddings,
                labels=dataset.node_attributes["group"],
                output_dir=output_dir,
                input_mode="auto",
                method_configs={"kmeans": {"n_clusters": 2}},
                random_seed=7,
            )

            self.assertEqual(set(result.summary_frame["status"]), {"ok"})
            benchmark_dir = output_dir / dataset.name / "clustering"
            self.assertTrue((benchmark_dir / f"{dataset.name}_clustering_summary.csv").is_file())
            self.assertTrue((benchmark_dir / f"{dataset.name}_louvain_cluster_assignments.csv").is_file())
            self.assertTrue((benchmark_dir / f"{dataset.name}_louvain_cluster_sizes.csv").is_file())
            self.assertTrue((benchmark_dir / f"{dataset.name}_kmeans_cluster_assignments.csv").is_file())
            self.assertTrue((benchmark_dir / f"{dataset.name}_kmeans_cluster_sizes.csv").is_file())


if __name__ == "__main__":
    unittest.main()
