"""Focused tests for the Phase 1 data and protected-group layer."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx

from fim_hybrid.config import DatasetConfig
from fim_hybrid.data_loader import (
    load_dataset,
    load_dataset_config_file,
    resolve_builtin_dataset,
    resolve_dataset_config,
    verify_protected_groups,
)


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

    def test_builtin_email_eu_core_department_verification(self) -> None:
        config = resolve_builtin_dataset("email_Eu_core", REPO_ROOT)
        dataset = load_dataset(config)
        report = verify_protected_groups(dataset, "department")

        self.assertEqual(report.node_count, 1005)
        self.assertEqual(report.edge_count, 25571)
        self.assertIn("department", dataset.node_attributes.columns)
        self.assertGreaterEqual(len(report.group_sizes), 2)
        self.assertEqual(sum(report.group_sizes.values()), 1005)

    def test_builtin_facebook_combined_loads_graph_without_circle_attributes(self) -> None:
        config = resolve_builtin_dataset("facebook_combined", REPO_ROOT)
        dataset = load_dataset(config)

        self.assertEqual(dataset.graph.number_of_nodes(), 4039)
        self.assertEqual(dataset.graph.number_of_edges(), 88234)
        self.assertEqual(dataset.node_attributes.columns.tolist(), ["node_id"])

    def test_facebook_combined_circles_raise_clear_error(self) -> None:
        config = resolve_builtin_dataset("facebook_combined", REPO_ROOT)
        dataset = load_dataset(config)

        with self.assertRaisesRegex(
            ValueError,
            "contains only the combined edge list and no circle labels",
        ):
            verify_protected_groups(dataset, "circles")

    def test_facebook_combined_can_derive_community_id_with_louvain(self) -> None:
        config = resolve_builtin_dataset("facebook_combined", REPO_ROOT)
        dataset = load_dataset(config)

        report = verify_protected_groups(
            dataset,
            "community_id",
            derive_protected_groups=True,
            derived_group_method="louvain",
            random_seed=42,
        )

        self.assertIn("community_id", dataset.node_attributes.columns)
        self.assertEqual(int(dataset.node_attributes["community_id"].notna().sum()), dataset.graph.number_of_nodes())
        self.assertEqual(sum(report.group_sizes.values()), dataset.graph.number_of_nodes())
        self.assertGreaterEqual(len(report.group_sizes), 2)

    def test_verify_protected_groups_does_not_overwrite_existing_community_id(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3)])
        for node_id, community_id in {1: 10, 2: 10, 3: 11}.items():
            graph.nodes[node_id]["community_id"] = community_id
        dataset = load_dataset(DatasetConfig(name="temp", pickle_path=self._write_pickle(graph)))

        with patch("fim_hybrid.data_loader.detect_communities") as mocked_detect:
            report = verify_protected_groups(
                dataset,
                "community_id",
                derive_protected_groups=True,
                derived_group_method="louvain",
                random_seed=7,
            )

        mocked_detect.assert_not_called()
        self.assertEqual(report.group_sizes, {"10": 2, "11": 1})

    def test_derive_community_id_with_leiden_unavailable_raises_clear_error(self) -> None:
        config = resolve_builtin_dataset("facebook_combined", REPO_ROOT)
        dataset = load_dataset(config)

        with patch("fim_hybrid.community_detection.ig", None):
            with self.assertRaisesRegex(
                ValueError,
                "Unable to derive protected attribute 'community_id' using method 'leiden'",
            ):
                verify_protected_groups(
                    dataset,
                    "community_id",
                    derive_protected_groups=True,
                    derived_group_method="leiden",
                    random_seed=42,
                )

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

    def _write_pickle(self, graph: nx.Graph) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        pickle_path = Path(temp_dir.name) / "graph.pickle"
        with pickle_path.open("wb") as handle:
            pickle.dump(graph, handle)
        return pickle_path

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

    def test_external_pickle_graph_loads_correctly(self) -> None:
        graph = nx.DiGraph()
        graph.add_edge(1, 2)
        graph.nodes[1]["group"] = "A"
        graph.nodes[2]["group"] = "B"

        with tempfile.TemporaryDirectory() as temp_dir:
            pickle_path = Path(temp_dir) / "graph.pkl"
            with pickle_path.open("wb") as handle:
                pickle.dump(graph, handle)

            dataset = load_dataset(DatasetConfig(name="custom_pickle", pickle_path=pickle_path, dataset_format="pickle"))

            self.assertEqual(dataset.graph.number_of_nodes(), 2)
            self.assertEqual(dataset.graph.number_of_edges(), 1)
            self.assertIn("group", dataset.node_attributes.columns)

    def test_external_txt_edge_list_loads_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            edge_path = Path(temp_dir) / "edges.txt"
            edge_path.write_text("1 2\n2 3\n", encoding="utf-8")

            dataset = load_dataset(DatasetConfig(name="custom_txt", edge_path=edge_path, dataset_format="txt"))

            self.assertEqual(dataset.graph.number_of_nodes(), 3)
            self.assertEqual(dataset.graph.number_of_edges(), 2)
            self.assertTrue(dataset.graph.is_directed())

    def test_external_txt_gz_edge_list_loads_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            edge_path = Path(temp_dir) / "edges.txt.gz"
            with gzip.open(edge_path, "wt", encoding="utf-8") as handle:
                handle.write("1 2\n2 3\n")

            dataset = load_dataset(DatasetConfig(name="custom_txt_gz", edge_path=edge_path))

            self.assertEqual(dataset.graph.number_of_nodes(), 3)
            self.assertEqual(dataset.graph.number_of_edges(), 2)

    def test_external_csv_edge_list_loads_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            edge_path = Path(temp_dir) / "edges.csv"
            edge_path.write_text("src,dst\n1,2\n2,3\n", encoding="utf-8")

            dataset = load_dataset(
                DatasetConfig(
                    name="custom_csv",
                    edge_path=edge_path,
                    dataset_format="csv",
                    source_column="src",
                    target_column="dst",
                    directed=False,
                )
            )

            self.assertEqual(dataset.graph.number_of_nodes(), 3)
            self.assertEqual(dataset.graph.number_of_edges(), 2)
            self.assertFalse(dataset.graph.is_directed())

    def test_csv_edge_list_missing_columns_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            edge_path = Path(temp_dir) / "edges.csv"
            edge_path.write_text("u,v\n1,2\n2,3\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must contain source column 'source' and target column 'target'"):
                load_dataset(DatasetConfig(name="custom_csv", edge_path=edge_path, dataset_format="csv"))

    def test_separate_attribute_file_can_be_attached(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            edge_path = temp_path / "edges.txt"
            attribute_path = temp_path / "attributes.csv"
            edge_path.write_text("1 2\n2 3\n", encoding="utf-8")
            attribute_path.write_text("node_id,group\n1,A\n2,B\n3,C\n", encoding="utf-8")

            dataset = load_dataset(
                DatasetConfig(
                    name="custom_with_attrs",
                    edge_path=edge_path,
                    attribute_path=attribute_path,
                )
            )

            self.assertEqual(dataset.node_attributes.loc[1, "group"], "A")
            self.assertEqual(dataset.graph.nodes[3]["group"], "C")

    def test_attribute_file_with_missing_graph_nodes_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            edge_path = temp_path / "edges.txt"
            attribute_path = temp_path / "attributes.csv"
            edge_path.write_text("1 2\n2 3\n", encoding="utf-8")
            attribute_path.write_text("node_id,group\n1,A\n2,B\n4,C\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not present in the graph"):
                load_dataset(
                    DatasetConfig(
                        name="custom_with_bad_attrs",
                        edge_path=edge_path,
                        attribute_path=attribute_path,
                    )
                )

    def test_load_dataset_config_file_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            edge_path = temp_path / "edges.csv"
            config_path = temp_path / "dataset.json"
            edge_path.write_text("src,dst\n1,2\n2,3\n", encoding="utf-8")
            config_path.write_text(
                json.dumps(
                    {
                        "name": "json_dataset",
                        "graph_path": "edges.csv",
                        "dataset_format": "csv",
                        "source_col": "src",
                        "target_col": "dst",
                        "directed": False,
                    }
                ),
                encoding="utf-8",
            )

            config = load_dataset_config_file(config_path)
            dataset = load_dataset(config)

            self.assertEqual(config.name, "json_dataset")
            self.assertEqual(config.edge_path, edge_path)
            self.assertFalse(config.directed)
            self.assertEqual(dataset.graph.number_of_edges(), 2)

    def test_resolve_dataset_config_supports_builtin_and_custom_stem_lookup(self) -> None:
        builtin = resolve_dataset_config("graph_spa_500_0", base_dir=REPO_ROOT)
        custom = resolve_dataset_config("email_Eu_core", base_dir=REPO_ROOT)

        self.assertEqual(builtin.name, "graph_spa_500_0")
        self.assertTrue(Path(custom.pickle_path or custom.edge_path).exists())
        self.assertEqual(custom.name, "email_Eu_core")


if __name__ == "__main__":
    unittest.main()
