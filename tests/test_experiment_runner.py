"""Unit tests for the experiment comparison runner."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import networkx as nx
import pandas as pd

from fim_hybrid.data_loader import LoadedDataset, verify_protected_groups
from fim_hybrid.experiment_runner import ExperimentSettings, run_loaded_experiment


def _toy_experiment_fixture() -> tuple[LoadedDataset, object]:
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 4),
            (4, 5),
            (4, 6),
            (5, 6),
            (6, 7),
            (7, 8),
        ]
    )
    for node_id, group_name in {
        1: "A",
        2: "A",
        3: "B",
        4: "B",
        5: "C",
        6: "C",
        7: "D",
        8: "D",
    }.items():
        graph.nodes[node_id]["group"] = group_name

    node_attributes = pd.DataFrame(
        [{"node_id": node_id, "group": graph.nodes[node_id]["group"]} for node_id in sorted(graph.nodes())]
    ).set_index("node_id", drop=False)
    dataset = LoadedDataset(name="toy_experiment", graph=graph, node_attributes=node_attributes)
    protected_group_report = verify_protected_groups(dataset, "group")
    return dataset, protected_group_report


class ExperimentRunnerTestCase(unittest.TestCase):
    """Check that experiment comparisons are reproducible and complete."""

    def test_run_loaded_experiment_returns_comparable_results(self) -> None:
        dataset, protected_group_report = _toy_experiment_fixture()
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=3,
            community_method="louvain",
            propagation_probability=0.0,
            mc_runs=3,
            population_size=5,
            generations=3,
            random_seed=9,
        )

        result_frame = run_loaded_experiment(
            dataset=dataset,
            protected_group_report=protected_group_report,
            settings=settings,
            community_methods=["louvain"],
            baseline_methods=["degree", "random"],
            include_ablations=False,
        )

        self.assertEqual(set(result_frame["method"]), {"degree", "random", "cea_fim", "hybrid_siea"})
        self.assertIn("f_score", result_frame.columns)
        self.assertIn("community_modularity", result_frame.columns)
        self.assertTrue((result_frame["community_method"] == "louvain").all())

    def test_run_loaded_experiment_is_reproducible(self) -> None:
        dataset, protected_group_report = _toy_experiment_fixture()
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=3,
            community_method="louvain",
            propagation_probability=0.0,
            mc_runs=3,
            population_size=5,
            generations=3,
            random_seed=9,
        )

        first = run_loaded_experiment(
            dataset=dataset,
            protected_group_report=protected_group_report,
            settings=settings,
            community_methods=["louvain"],
            baseline_methods=["degree", "random"],
            include_ablations=True,
        ).sort_values(["method"]).reset_index(drop=True)
        second = run_loaded_experiment(
            dataset=dataset,
            protected_group_report=protected_group_report,
            settings=settings,
            community_methods=["louvain"],
            baseline_methods=["degree", "random"],
            include_ablations=True,
        ).sort_values(["method"]).reset_index(drop=True)

        comparable_columns = [
            "method",
            "variant_type",
            "total_spread",
            "mf",
            "dcv",
            "f_score",
            "seed_set",
            "community_modularity",
        ]
        self.assertTrue(first[comparable_columns].equals(second[comparable_columns]))

    def test_node2vec_flag_fails_clearly(self) -> None:
        dataset, protected_group_report = _toy_experiment_fixture()
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=3,
            use_node2vec=True,
        )

        with self.assertRaisesRegex(NotImplementedError, "Node2Vec"):
            run_loaded_experiment(
                dataset=dataset,
                protected_group_report=protected_group_report,
                settings=settings,
                include_ablations=False,
            )

    def test_results_are_saved_under_dataset_directory(self) -> None:
        dataset, protected_group_report = _toy_experiment_fixture()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            settings = ExperimentSettings(
                protected_attribute="group",
                budget=3,
                community_method="louvain",
                propagation_probability=0.0,
                mc_runs=3,
                population_size=5,
                generations=3,
                random_seed=9,
                output_dir=output_dir,
            )

            run_loaded_experiment(
                dataset=dataset,
                protected_group_report=protected_group_report,
                settings=settings,
                community_methods=["louvain"],
                baseline_methods=["degree", "random"],
                include_ablations=False,
            )

            dataset_dir = output_dir / dataset.name
            self.assertTrue(dataset_dir.is_dir())
            self.assertTrue((dataset_dir / f"{dataset.name}_budget3_results.csv").is_file())
            self.assertTrue((dataset_dir / f"{dataset.name}_budget3_louvain_cea_fim_history.csv").is_file())
            self.assertTrue((dataset_dir / f"{dataset.name}_budget3_louvain_hybrid_siea_history.csv").is_file())


if __name__ == "__main__":
    unittest.main()
