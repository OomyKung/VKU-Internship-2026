"""Tests for downstream embedding evaluation helpers."""

from __future__ import annotations

import unittest

import networkx as nx
import pandas as pd

from fim_hybrid.data_loader import LoadedDataset
from fim_hybrid.embeddings.evaluation import evaluate_embedding_result
from fim_hybrid.embeddings.evaluation_splits import build_link_prediction_split, build_node_classification_split


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
    node_attributes = pd.DataFrame(
        [{"node_id": node_id, "group": labels[node_id]} for node_id in sorted(graph.nodes())]
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


if __name__ == "__main__":
    unittest.main()
