"""Unit tests for the cleaned experiment comparison runner."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import pandas as pd

from fim_hybrid.data_loader import LoadedDataset, verify_protected_groups
from fim_hybrid.experiment_runner import ExperimentSettings, run_loaded_experiment


KEPT_ML_LABEL = "hybrid_siea_ml_two_tier_tuned_swap_local_search"


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
    """Check that the cleaned runner only exposes the supported ML path."""

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
        self.assertIn("diffusion_model", result_frame.columns)
        self.assertIn("optimization_mode", result_frame.columns)
        self.assertIn("community_modularity", result_frame.columns)
        self.assertIn("ml_validation_spearman", result_frame.columns)
        self.assertIn("ml_validation_precision_at_budget", result_frame.columns)
        self.assertIn("delta_f_score", result_frame.columns)
        self.assertIn("zero_covered_groups_count", result_frame.columns)
        self.assertIn("bottom_3_avg_group_spread", result_frame.columns)
        self.assertIn("fraction_groups_covered", result_frame.columns)
        self.assertIn("weakest_groups_note", result_frame.columns)
        self.assertTrue((result_frame["community_method"] == "louvain").all())
        self.assertTrue((result_frame["diffusion_model"] == "ic").all())

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

    def test_run_loaded_experiment_rejects_unsupported_diffusion_model(self) -> None:
        dataset, protected_group_report = _toy_experiment_fixture()
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=3,
            diffusion_model="lt",
        )

        with self.assertRaisesRegex(ValueError, "Unsupported diffusion_model"):
            run_loaded_experiment(
                dataset=dataset,
                protected_group_report=protected_group_report,
                settings=settings,
                include_ablations=False,
            )

    def test_run_loaded_experiment_with_ml_adds_only_surviving_ml_row(self) -> None:
        dataset, protected_group_report = _toy_experiment_fixture()
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=3,
            community_method="louvain",
            propagation_probability=0.5,
            mc_runs=3,
            population_size=5,
            generations=3,
            random_seed=9,
            use_ml=True,
            ml_guidance_mode="two_tier",
            ml_top_fraction=0.5,
            ml_singleton_runs=3,
        )

        result_frame = run_loaded_experiment(
            dataset=dataset,
            protected_group_report=protected_group_report,
            settings=settings,
            community_methods=["louvain"],
            baseline_methods=["degree", "random"],
            include_ablations=False,
        )

        self.assertEqual(
            set(result_frame["method"]),
            {"degree", "random", "cea_fim", "hybrid_siea", KEPT_ML_LABEL},
        )
        ml_row = result_frame[result_frame["method"] == KEPT_ML_LABEL].iloc[0]
        self.assertFalse(pd.isna(ml_row["ml_validation_spearman"]))
        self.assertFalse(pd.isna(ml_row["ml_validation_precision_at_budget"]))
        self.assertEqual(int(ml_row["candidate_pool_size"]), dataset.graph.number_of_nodes())
        self.assertEqual(ml_row["ml_guidance_mode"], "two_tier")

    def test_run_loaded_experiment_treats_ml_off_as_single_supported_ml_path(self) -> None:
        dataset, protected_group_report = _toy_experiment_fixture()
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=3,
            community_method="louvain",
            propagation_probability=0.5,
            mc_runs=3,
            population_size=5,
            generations=3,
            random_seed=9,
            use_ml=True,
            ml_guidance_mode="off",
            ml_top_fraction=0.5,
            ml_singleton_runs=3,
        )

        result_frame = run_loaded_experiment(
            dataset=dataset,
            protected_group_report=protected_group_report,
            settings=settings,
            community_methods=["louvain"],
            baseline_methods=["degree", "random"],
            include_ablations=False,
        )

        ml_methods = sorted(method for method in result_frame["method"] if str(method).startswith("hybrid_siea_ml_"))
        self.assertEqual(ml_methods, [KEPT_ML_LABEL])

    def test_run_loaded_experiment_trains_ml_model_once_for_supported_variant(self) -> None:
        import fim_hybrid.experiment_runner as experiment_runner_module  # noqa: PLC0415

        dataset, protected_group_report = _toy_experiment_fixture()
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=3,
            community_method="louvain",
            propagation_probability=0.5,
            mc_runs=3,
            population_size=5,
            generations=3,
            random_seed=9,
            use_ml=True,
            ml_guidance_mode="two_tier",
            ml_top_fraction=0.5,
            ml_singleton_runs=3,
        )
        original_train = experiment_runner_module.train_node_utility_model

        with patch("fim_hybrid.experiment_runner.train_node_utility_model") as mocked_train:
            mocked_train.side_effect = original_train
            run_loaded_experiment(
                dataset=dataset,
                protected_group_report=protected_group_report,
                settings=settings,
                community_methods=["louvain"],
                baseline_methods=["degree", "random"],
                include_ablations=False,
            )

        self.assertEqual(mocked_train.call_count, 1)

    def test_removed_ml_guidance_mode_errors(self) -> None:
        dataset, protected_group_report = _toy_experiment_fixture()
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=3,
            use_ml=True,
            ml_guidance_mode="hard_filter",
        )

        with self.assertRaisesRegex(ValueError, "Only ml_guidance_mode='two_tier' is supported"):
            run_loaded_experiment(
                dataset=dataset,
                protected_group_report=protected_group_report,
                settings=settings,
                include_ablations=False,
            )

    def test_removed_compare_flags_error(self) -> None:
        dataset, protected_group_report = _toy_experiment_fixture()
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=3,
            use_ml=True,
            compare_refinement_variants=True,
        )

        with self.assertRaisesRegex(ValueError, "Obsolete ML comparison families were removed"):
            run_loaded_experiment(
                dataset=dataset,
                protected_group_report=protected_group_report,
                settings=settings,
                include_ablations=False,
            )

    def test_removed_node2vec_error(self) -> None:
        dataset, protected_group_report = _toy_experiment_fixture()
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=3,
            use_ml=True,
            use_node2vec=True,
        )

        with self.assertRaisesRegex(ValueError, "Node2Vec ML variants were removed"):
            run_loaded_experiment(
                dataset=dataset,
                protected_group_report=protected_group_report,
                settings=settings,
                include_ablations=False,
            )

    def test_removed_selected_ml_variant_errors(self) -> None:
        dataset, protected_group_report = _toy_experiment_fixture()
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=3,
            use_ml=True,
            ml_guidance_mode="two_tier",
        )

        with self.assertRaisesRegex(ValueError, "Removed ML variants are no longer supported"):
            run_loaded_experiment(
                dataset=dataset,
                protected_group_report=protected_group_report,
                settings=settings,
                selected_methods=["hybrid_siea_ml_soft_bias"],
                include_ablations=False,
            )

    def test_run_loaded_experiment_can_filter_to_cea_fim_and_swap_local_search(self) -> None:
        dataset, protected_group_report = _toy_experiment_fixture()
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=3,
            community_method="louvain",
            propagation_probability=0.5,
            mc_runs=3,
            population_size=5,
            generations=3,
            random_seed=9,
            use_ml=True,
            ml_guidance_mode="two_tier",
            ml_top_fraction=0.5,
            ml_singleton_runs=3,
        )

        result_frame = run_loaded_experiment(
            dataset=dataset,
            protected_group_report=protected_group_report,
            settings=settings,
            community_methods=["louvain"],
            baseline_methods=["degree", "random"],
            selected_methods=["cea_fim", KEPT_ML_LABEL],
            include_ablations=False,
        )

        self.assertEqual(set(result_frame["method"]), {"cea_fim", KEPT_ML_LABEL})

    def test_results_are_saved_under_protected_attribute_directory(self) -> None:
        dataset, protected_group_report = _toy_experiment_fixture()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            settings = ExperimentSettings(
                protected_attribute="group",
                budget=3,
                community_method="louvain",
                propagation_probability=0.5,
                mc_runs=3,
                population_size=5,
                generations=3,
                random_seed=9,
                output_dir=output_dir,
                use_ml=True,
                ml_guidance_mode="two_tier",
                ml_singleton_runs=3,
            )

            run_loaded_experiment(
                dataset=dataset,
                protected_group_report=protected_group_report,
                settings=settings,
                community_methods=["louvain"],
                baseline_methods=["degree", "random"],
                include_ablations=False,
            )

            attribute_dir = output_dir / dataset.name / "group"
            self.assertTrue(attribute_dir.is_dir())
            self.assertTrue((attribute_dir / f"{dataset.name}_budget3_results.csv").is_file())
            self.assertTrue((attribute_dir / f"{dataset.name}_budget3_louvain_cea_fim_history.csv").is_file())
            self.assertTrue((attribute_dir / f"{dataset.name}_budget3_louvain_hybrid_siea_history.csv").is_file())
            self.assertTrue((attribute_dir / f"{dataset.name}_budget3_louvain_{KEPT_ML_LABEL}_history.csv").is_file())

    def test_results_sanitize_protected_attribute_directory_name(self) -> None:
        dataset, _ = _toy_experiment_fixture()
        protected_attribute = "group/2026"
        dataset.node_attributes[protected_attribute] = dataset.node_attributes["group"]
        for node_id in dataset.graph.nodes():
            dataset.graph.nodes[node_id][protected_attribute] = dataset.graph.nodes[node_id]["group"]
        protected_group_report = verify_protected_groups(dataset, protected_attribute)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            settings = ExperimentSettings(
                protected_attribute=protected_attribute,
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

            attribute_dir = output_dir / dataset.name / "group_2026"
            self.assertTrue(attribute_dir.is_dir())
            self.assertTrue((attribute_dir / f"{dataset.name}_budget3_results.csv").is_file())


if __name__ == "__main__":
    unittest.main()
