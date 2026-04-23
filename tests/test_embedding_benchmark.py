"""Tests for the modular graph embedding benchmark framework."""

from __future__ import annotations

from importlib.util import find_spec as real_find_spec
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

import networkx as nx
import numpy as np
import pandas as pd

from fim_hybrid.data_loader import LoadedDataset
from fim_hybrid.embeddings._pyg_common import pyg_dependencies_available
from fim_hybrid.embeddings.benchmark import run_embedding_benchmark, run_embedding_method


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
    for node_id in graph.nodes():
        graph.nodes[node_id]["group"] = labels[node_id]
        graph.nodes[node_id]["region"] = regions[node_id]

    node_attributes = pd.DataFrame(
        [
            {
                "node_id": node_id,
                "group": labels[node_id],
                "region": regions[node_id],
            }
            for node_id in sorted(graph.nodes())
        ]
    ).set_index("node_id", drop=False)
    return LoadedDataset(name="toy_embedding", graph=graph, node_attributes=node_attributes)


class EmbeddingMethodTestCase(unittest.TestCase):
    """Validate per-method output guarantees."""

    def test_shallow_methods_cover_all_nodes_with_expected_dimensions(self) -> None:
        dataset = _toy_dataset()
        for method_name in ["deepwalk", "node2vec", "line"]:
            config = {
                "embedding_dim": 4,
                "walk_length": 6,
                "num_walks": 4,
                "window_size": 2,
                "order": "both",
                "random_seed": 7,
            }
            result = run_embedding_method(
                dataset,
                method_name,
                config=config,
                output_dir=None,
                export_formats=(),
            )

            self.assertEqual(set(result.embedding_frame["node_id"]), set(dataset.graph.nodes()))
            self.assertEqual(len(result.vector_columns), 4)
            self.assertEqual(result.node_count, dataset.graph.number_of_nodes())

    def test_gnn_autoencoder_methods_cover_all_nodes_with_expected_dimensions(self) -> None:
        if not pyg_dependencies_available():
            self.skipTest("torch and torch_geometric are not installed")

        dataset = _toy_dataset()
        for method_name in ["gcn", "graphsage", "vgae"]:
            result = run_embedding_method(
                dataset,
                method_name,
                config={
                    "embedding_dim": 4,
                    "hidden_dim": 8,
                    "num_layers": 2,
                    "epochs": 10,
                    "random_seed": 7,
                },
                output_dir=None,
                export_formats=(),
            )

            self.assertEqual(set(result.embedding_frame["node_id"]), set(dataset.graph.nodes()))
            self.assertEqual(len(result.vector_columns), 4)
            self.assertTrue(np.isfinite(result.embedding_frame[result.vector_columns].to_numpy(dtype=float)).all())

    def test_self_supervised_methods_cover_all_nodes_with_expected_dimensions(self) -> None:
        if not pyg_dependencies_available():
            self.skipTest("torch and torch_geometric are not installed")

        dataset = _toy_dataset()
        for method_name in ["dgi", "graphcl"]:
            result = run_embedding_method(
                dataset,
                method_name,
                config={
                    "embedding_dim": 4,
                    "hidden_dim": 8,
                    "projection_dim": 4,
                    "num_layers": 2,
                    "epochs": 10,
                    "random_seed": 7,
                },
                output_dir=None,
                export_formats=(),
            )

            self.assertEqual(set(result.embedding_frame["node_id"]), set(dataset.graph.nodes()))
            self.assertEqual(len(result.vector_columns), 4)
            self.assertTrue(np.isfinite(result.embedding_frame[result.vector_columns].to_numpy(dtype=float)).all())


class BenchmarkRunnerTestCase(unittest.TestCase):
    """Validate multi-method benchmark behavior, exports, and skips."""

    def test_benchmark_exports_embeddings_and_runs_label_probe(self) -> None:
        dataset = _toy_dataset()
        with _workspace_tempdir() as temp_dir:
            output_dir = Path(temp_dir)
            result = run_embedding_benchmark(
                dataset,
                methods=["deepwalk", "line"],
                method_configs={
                    "deepwalk": {"embedding_dim": 4, "walk_length": 6, "num_walks": 4, "window_size": 2, "random_seed": 7},
                    "line": {"embedding_dim": 4, "order": "both", "random_seed": 7},
                },
                output_dir=output_dir,
                export_formats=["csv", "pickle", "npy"],
                label_column="group",
                evaluation_tasks=["node_classification", "link_prediction", "node_clustering"],
                max_workers=2,
                continue_on_error=False,
            )

            self.assertEqual(set(result.summary_frame["status"]), {"ok"})
            self.assertEqual(result.summary_frame["method"].tolist(), ["deepwalk", "line"])
            self.assertTrue((result.summary_frame["all_nodes_embedded"]).all())
            self.assertTrue((result.summary_frame["all_finite"]).all())
            self.assertTrue((result.summary_frame["label_probe_status"] == "ok").all())
            self.assertTrue((result.summary_frame["label_probe_macro_f1"].notna()).all())
            self.assertEqual(
                set(result.evaluation_frame["task"]),
                {"node_classification", "link_prediction", "node_clustering"},
            )
            self.assertEqual(set(result.evaluation_frame["status"]), {"ok"})

            benchmark_dir = output_dir / dataset.name / "embeddings"
            self.assertTrue((benchmark_dir / f"{dataset.name}_embedding_benchmark_summary.csv").is_file())
            self.assertTrue((benchmark_dir / f"{dataset.name}_embedding_evaluation_summary.csv").is_file())
            for method_name in ["deepwalk", "line"]:
                self.assertTrue((benchmark_dir / f"{dataset.name}_{method_name}_embeddings.csv").is_file())
                self.assertTrue((benchmark_dir / f"{dataset.name}_{method_name}_embeddings.pkl").is_file())
                self.assertTrue((benchmark_dir / f"{dataset.name}_{method_name}_embeddings.npy").is_file())
                self.assertTrue((benchmark_dir / f"{dataset.name}_{method_name}_embeddings_node_order.json").is_file())

    def test_benchmark_runs_protected_attribute_probe_when_requested(self) -> None:
        dataset = _toy_dataset()
        result = run_embedding_benchmark(
            dataset,
            methods=["deepwalk"],
            method_configs={
                "deepwalk": {"embedding_dim": 4, "walk_length": 6, "num_walks": 4, "window_size": 2, "random_seed": 7},
            },
            output_dir=None,
            export_formats=(),
            label_column="group",
            protected_attribute_column="region",
            run_protected_attribute_probe=True,
            continue_on_error=False,
        )

        self.assertEqual(len(result.summary_frame), 1)
        row = result.summary_frame.iloc[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["protected_probe_status"], "ok")
        self.assertEqual(row["protected_probe_column"], "region")
        self.assertGreaterEqual(float(row["protected_probe_accuracy"]), 0.0)
        self.assertGreaterEqual(float(row["protected_probe_macro_f1"]), 0.0)

    def test_benchmark_supports_adversarial_graphsage_evaluation_when_available(self) -> None:
        if not pyg_dependencies_available():
            self.skipTest("torch and torch_geometric are not installed")

        dataset = _toy_dataset()
        result = run_embedding_benchmark(
            dataset,
            methods=["deepwalk"],
            method_configs={
                "deepwalk": {"embedding_dim": 4, "walk_length": 6, "num_walks": 4, "window_size": 2, "random_seed": 7},
            },
            output_dir=None,
            export_formats=(),
            label_column="group",
            protected_attribute_column="region",
            evaluation_tasks=["node_classification"],
            node_classification_model="graphsage",
            debias_mode="adversarial",
            early_stop_metric="worst_group_f1",
            early_stop_patience=3,
            adversary_loss_weight=1.0,
            gradient_reversal_lambda=1.0,
            continue_on_error=False,
        )

        row = result.evaluation_frame.iloc[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["training_mode"], "adversarial")
        self.assertEqual(row["debias_mode"], "adversarial")
        self.assertEqual(row["early_stop_metric"], "worst_group_f1_raw")
        self.assertIn(str(row["protected_probe_status"]), {"ok", "skipped"})

    def test_benchmark_supports_group_dro_graphsage_evaluation_when_available(self) -> None:
        if not pyg_dependencies_available():
            self.skipTest("torch and torch_geometric are not installed")

        dataset = _toy_dataset()
        result = run_embedding_benchmark(
            dataset,
            methods=["deepwalk"],
            method_configs={
                "deepwalk": {"embedding_dim": 4, "walk_length": 6, "num_walks": 4, "window_size": 2, "random_seed": 7},
            },
            output_dir=None,
            export_formats=(),
            label_column="group",
            protected_attribute_column="region",
            evaluation_tasks=["node_classification"],
            node_classification_model="graphsage",
            training_mode="group_robust",
            group_weight_mode="group_dro",
            group_robust_weight=0.5,
            min_group_support_threshold=2,
            early_stop_metric="worst_group_f1",
            early_stop_patience=3,
            continue_on_error=False,
        )

        row = result.evaluation_frame.iloc[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["training_mode"], "group_robust")
        self.assertEqual(row["comparison_mode"], "group_robust")
        self.assertEqual(row["group_weight_mode"], "group_dro")
        self.assertAlmostEqual(float(row["group_robust_weight"]), 0.5, places=6)
        self.assertEqual(int(row["min_group_support_threshold"]), 2)
        self.assertGreaterEqual(int(row["best_epoch"]), 1)
        self.assertTrue(json.loads(str(row["training_history_json"])))

    def test_benchmark_supports_anti_collapse_graphsage_mode_alias_when_available(self) -> None:
        if not pyg_dependencies_available():
            self.skipTest("torch and torch_geometric are not installed")

        dataset = _toy_dataset()
        result = run_embedding_benchmark(
            dataset,
            methods=["deepwalk"],
            method_configs={
                "deepwalk": {"embedding_dim": 4, "walk_length": 6, "num_walks": 4, "window_size": 2, "random_seed": 7},
            },
            output_dir=None,
            export_formats=(),
            label_column="group",
            protected_attribute_column="region",
            evaluation_tasks=["node_classification"],
            node_classification_model="graphsage",
            training_mode="anti_collapse_group_robust",
            min_group_support_threshold=2,
            min_group_support_train=2,
            min_group_support_eval=2,
            early_stop_patience=3,
            continue_on_error=False,
        )

        row = result.evaluation_frame.iloc[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["training_mode"], "anti_collapse_group_robust")
        self.assertEqual(row["comparison_mode"], "anti_collapse_group_robust")
        self.assertEqual(row["group_weight_mode"], "min_support_boost")
        self.assertAlmostEqual(float(row["group_robust_weight"]), 0.25, places=6)
        self.assertAlmostEqual(float(row["min_support_boost_factor"]), 2.0, places=6)
        self.assertTrue(bool(row["rebalance_batches_by_group"]))
        self.assertEqual(row["early_stop_metric"], "worst_group_f1_raw")
        self.assertTrue(json.loads(str(row["training_history_json"])))

    def test_benchmark_skips_metapath2vec_on_homogeneous_graph(self) -> None:
        dataset = _toy_dataset()
        result = run_embedding_benchmark(
            dataset,
            methods=["metapath2vec"],
            output_dir=None,
            export_formats=(),
            evaluation_tasks=["link_prediction"],
            continue_on_error=True,
        )

        self.assertEqual(len(result.summary_frame), 1)
        row = result.summary_frame.iloc[0]
        self.assertEqual(row["status"], "skipped")
        self.assertIn("heterogeneous graph", row["skip_reason"])
        self.assertEqual(len(result.evaluation_frame), 1)
        evaluation_row = result.evaluation_frame.iloc[0]
        self.assertEqual(evaluation_row["status"], "skipped")
        self.assertIn("heterogeneous graph", evaluation_row["skipped_reason"])

    def test_benchmark_skips_missing_optional_dependencies_per_method(self) -> None:
        dataset = _toy_dataset()

        def _fake_find_spec(name: str):
            if name in {"torch", "torch_geometric"}:
                return None
            return real_find_spec(name)

        with patch("fim_hybrid.embeddings.registry.importlib.util.find_spec", side_effect=_fake_find_spec):
            result = run_embedding_benchmark(
                dataset,
                methods=["gcn"],
                output_dir=None,
                export_formats=(),
                continue_on_error=True,
            )

        self.assertEqual(len(result.summary_frame), 1)
        row = result.summary_frame.iloc[0]
        self.assertEqual(row["status"], "skipped")
        self.assertIn("torch", row["skip_reason"])


if __name__ == "__main__":
    unittest.main()
