"""Focused tests for the Phase 1 data and protected-group layer."""

from __future__ import annotations

import pickle
from pathlib import Path
import tempfile
import unittest

import networkx as nx

from fim_hybrid.config import DatasetConfig
from fim_hybrid.data_loader import load_dataset, resolve_builtin_dataset, verify_protected_groups


REPO_ROOT = Path(__file__).resolve().parents[1]


class Phase1DataLoaderTestCase(unittest.TestCase):
    """Check strict loading and protected-group verification behavior."""

    def test_builtin_graph_spa_ethnicity_verification(self) -> None:
        config = resolve_builtin_dataset("graph_spa_500_0", REPO_ROOT)
        dataset = load_dataset(config)
        report = verify_protected_groups(dataset, "ethnicity")

        self.assertEqual(report.node_count, 500)
        self.assertEqual(report.edge_count, 1689)
        self.assertEqual(sum(report.group_sizes.values()), 500)
        self.assertEqual(set(report.group_sizes), {"asian", "black", "latino", "other", "white"})

    def test_missing_protected_attribute_raises(self) -> None:
        config = resolve_builtin_dataset("graph_spa_500_0", REPO_ROOT)
        dataset = load_dataset(config)

        with self.assertRaisesRegex(ValueError, "missing from dataset"):
            verify_protected_groups(dataset, "color")

    def test_all_null_protected_attribute_raises(self) -> None:
        graph = nx.DiGraph()
        graph.add_edge(1, 2)
        graph.nodes[1]["group"] = None
        graph.nodes[2]["group"] = None

        with tempfile.TemporaryDirectory() as temp_dir:
            pickle_path = Path(temp_dir) / "graph.pickle"
            with pickle_path.open("wb") as handle:
                pickle.dump(graph, handle)

            dataset = load_dataset(DatasetConfig(name="temp", pickle_path=pickle_path))
            with self.assertRaisesRegex(ValueError, "all-null"):
                verify_protected_groups(dataset, "group")

    def test_inconsistent_attribute_node_ids_raise(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            edge_path = temp_path / "edges.txt"
            attribute_path = temp_path / "attributes.csv"

            edge_path.write_text("1 2\n2 3\n", encoding="utf-8")
            attribute_path.write_text("node_id,group\nA,x\n2,y\n3,z\n", encoding="utf-8")

            config = DatasetConfig(
                name="temp",
                edge_path=edge_path,
                attribute_path=attribute_path,
            )

            with self.assertRaisesRegex(ValueError, "Inconsistent node ID types"):
                load_dataset(config)


if __name__ == "__main__":
    unittest.main()
