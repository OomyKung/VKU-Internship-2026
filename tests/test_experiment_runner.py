"""Unit tests for the cleaned experiment comparison runner."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import pandas as pd

from fim_hybrid.data_loader import LoadedDataset, verify_protected_groups
from fim_hybrid.gnn_training import gnn_dependencies_available
from fim_hybrid.experiment_runner import ExperimentSettings, run_loaded_experiment


KEPT_ML_LABEL = "hybrid_siea_ml_two_tier_tuned_swap_local_search"
KEPT_GNN_ML_LABEL = "hybrid_siea_ml_gnn_two_tier_tuned_swap_local_search"
KEPT_GNN_NODE2VEC_ML_LABEL = "hybrid_siea_ml_gnn_node2vec_two_tier_tuned_swap_local_search"
KEPT_RIS_ML_LABEL = "hybrid_siea_ml_ris_two_tier_tuned_swap_local_search"
KEPT_GNN_RIS_ML_LABEL = "hybrid_siea_ml_gnn_ris_two_tier_tuned_swap_local_search"
KEPT_GNN_RIS_NODE2VEC_ML_LABEL = "hybrid_siea_ml_gnn_ris_node2vec_two_tier_tuned_swap_local_search"


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
            mc_runs_search=3,
            mc_runs_eval=5,
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
        self.assertIn("mc_runs_search", result_frame.columns)
        self.assertIn("mc_runs_eval", result_frame.columns)
        self.assertIn("mc_runs_search_used", result_frame.columns)
        self.assertIn("mc_runs_eval_used", result_frame.columns)
        self.assertIn("search_runtime_seconds", result_frame.columns)
        self.assertIn("final_eval_runtime_seconds", result_frame.columns)
        self.assertIn("extra_spread", result_frame.columns)
        self.assertIn("spread_includes_seed_nodes", result_frame.columns)
        self.assertIn("final_recheck_applied", result_frame.columns)
        self.assertIn("final_recheck_mc_runs_used", result_frame.columns)
        self.assertIn("zero_covered_groups_count", result_frame.columns)
        self.assertIn("bottom_3_avg_group_spread", result_frame.columns)
        self.assertIn("fraction_groups_covered", result_frame.columns)
        self.assertIn("weakest_groups_note", result_frame.columns)
        self.assertTrue((result_frame["community_method"] == "louvain").all())
        self.assertTrue((result_frame["diffusion_model"] == "ic").all())
        self.assertTrue((result_frame["mc_runs_search"] == 3).all())
        self.assertTrue((result_frame["mc_runs_eval"] == 5).all())
        self.assertTrue((result_frame["mc_runs_search_used"] == 3).all())
        self.assertTrue((result_frame["mc_runs_eval_used"] == 5).all())
        self.assertTrue((result_frame["spread_includes_seed_nodes"]).all())
        for row in result_frame.itertuples(index=False):
            self.assertAlmostEqual(
                float(row.runtime_seconds),
                float(row.search_runtime_seconds) + float(row.final_eval_runtime_seconds),
                places=9,
            )
            self.assertAlmostEqual(float(row.extra_spread), float(row.total_spread) - settings.budget, places=9)
            self.assertGreaterEqual(float(row.total_spread), float(settings.budget))

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
        self.assertEqual(ml_row["ml_backend"], "tabular")
        self.assertTrue(pd.isna(ml_row["gnn_model_type"]))

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

    def test_run_loaded_experiment_errors_for_gnn_backend_without_optional_dependencies(self) -> None:
        if gnn_dependencies_available():
            self.skipTest("torch and torch_geometric are installed")

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
            ml_backend="gnn",
            ml_guidance_mode="two_tier",
            ml_top_fraction=0.5,
            ml_singleton_runs=3,
        )

        with self.assertRaisesRegex(ValueError, "optional dependencies are unavailable"):
            run_loaded_experiment(
                dataset=dataset,
                protected_group_report=protected_group_report,
                settings=settings,
                community_methods=["louvain"],
                baseline_methods=["degree", "random"],
                include_ablations=False,
            )

    def test_run_loaded_experiment_errors_for_both_backend_without_optional_dependencies(self) -> None:
        if gnn_dependencies_available():
            self.skipTest("torch and torch_geometric are installed")

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
            ml_backend="both",
            ml_guidance_mode="two_tier",
            ml_top_fraction=0.5,
            ml_singleton_runs=3,
        )

        with self.assertRaisesRegex(ValueError, "optional dependencies are unavailable"):
            run_loaded_experiment(
                dataset=dataset,
                protected_group_report=protected_group_report,
                settings=settings,
                community_methods=["louvain"],
                baseline_methods=["degree", "random"],
                include_ablations=False,
            )

    def test_run_loaded_experiment_errors_for_gnn_ris_backend_without_optional_dependencies(self) -> None:
        if gnn_dependencies_available():
            self.skipTest("torch and torch_geometric are installed")

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
            ml_backend="gnn_ris",
            ml_guidance_mode="two_tier",
            ml_top_fraction=0.5,
            ml_singleton_runs=3,
        )

        with self.assertRaisesRegex(ValueError, "optional dependencies are unavailable"):
            run_loaded_experiment(
                dataset=dataset,
                protected_group_report=protected_group_report,
                settings=settings,
                community_methods=["louvain"],
                baseline_methods=["degree", "random"],
                include_ablations=False,
            )

    def test_run_loaded_experiment_with_both_backends_adds_two_ml_rows_when_available(self) -> None:
        if not gnn_dependencies_available():
            self.skipTest("torch and torch_geometric are not installed")

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
            ml_backend="both",
            ml_guidance_mode="two_tier",
            ml_top_fraction=0.5,
            ml_singleton_runs=3,
            gnn_epochs=20,
        )

        result_frame = run_loaded_experiment(
            dataset=dataset,
            protected_group_report=protected_group_report,
            settings=settings,
            community_methods=["louvain"],
            baseline_methods=["degree", "random"],
            include_ablations=False,
        )

        self.assertIn(KEPT_ML_LABEL, set(result_frame["method"]))
        self.assertIn(KEPT_GNN_ML_LABEL, set(result_frame["method"]))
        gnn_row = result_frame[result_frame["method"] == KEPT_GNN_ML_LABEL].iloc[0]
        self.assertEqual(gnn_row["ml_backend"], "gnn")
        self.assertEqual(gnn_row["gnn_model_type"], "graphsage")
        self.assertFalse(bool(gnn_row["node2vec_enabled"]))
        self.assertEqual(gnn_row["node2vec_mode"], "off")

    def test_run_loaded_experiment_with_gnn_node2vec_concat_adds_node2vec_row_when_available(self) -> None:
        if not gnn_dependencies_available():
            self.skipTest("torch and torch_geometric are not installed")

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
            ml_backend="gnn",
            gnn_node2vec_mode="input_concat",
            ml_guidance_mode="two_tier",
            ml_top_fraction=0.5,
            ml_singleton_runs=3,
            gnn_epochs=10,
            node2vec_dimensions=4,
            node2vec_walk_length=6,
            node2vec_num_walks=4,
            node2vec_window=2,
        )

        result_frame = run_loaded_experiment(
            dataset=dataset,
            protected_group_report=protected_group_report,
            settings=settings,
            community_methods=["louvain"],
            baseline_methods=["degree", "random"],
            include_ablations=False,
        )

        self.assertIn(KEPT_GNN_NODE2VEC_ML_LABEL, set(result_frame["method"]))
        self.assertNotIn(KEPT_GNN_ML_LABEL, set(result_frame["method"]))
        gnn_row = result_frame[result_frame["method"] == KEPT_GNN_NODE2VEC_ML_LABEL].iloc[0]
        self.assertEqual(gnn_row["ml_backend"], "gnn")
        self.assertEqual(gnn_row["gnn_model_type"], "graphsage")
        self.assertTrue(bool(gnn_row["node2vec_enabled"]))
        self.assertEqual(gnn_row["node2vec_mode"], "input_concat")

    def test_run_loaded_experiment_with_gnn_compare_adds_plain_and_node2vec_rows_when_available(self) -> None:
        if not gnn_dependencies_available():
            self.skipTest("torch and torch_geometric are not installed")

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
            ml_backend="gnn",
            gnn_node2vec_mode="compare",
            ml_guidance_mode="two_tier",
            ml_top_fraction=0.5,
            ml_singleton_runs=3,
            gnn_epochs=10,
            node2vec_dimensions=4,
            node2vec_walk_length=6,
            node2vec_num_walks=4,
            node2vec_window=2,
        )

        result_frame = run_loaded_experiment(
            dataset=dataset,
            protected_group_report=protected_group_report,
            settings=settings,
            community_methods=["louvain"],
            baseline_methods=["degree", "random"],
            include_ablations=False,
        )

        self.assertIn(KEPT_GNN_ML_LABEL, set(result_frame["method"]))
        self.assertIn(KEPT_GNN_NODE2VEC_ML_LABEL, set(result_frame["method"]))
        plain_row = result_frame[result_frame["method"] == KEPT_GNN_ML_LABEL].iloc[0]
        node2vec_row = result_frame[result_frame["method"] == KEPT_GNN_NODE2VEC_ML_LABEL].iloc[0]
        self.assertEqual(plain_row["node2vec_mode"], "off")
        self.assertFalse(bool(plain_row["node2vec_enabled"]))
        self.assertEqual(node2vec_row["node2vec_mode"], "input_concat")
        self.assertTrue(bool(node2vec_row["node2vec_enabled"]))

    def test_run_loaded_experiment_with_ris_backend_adds_ris_row(self) -> None:
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
            ml_backend="ris",
            ml_guidance_mode="two_tier",
            ris_num_rr_sets=32,
            ris_mode="weak_group_weighted",
            fairness_urgency_weight=0.25,
            diversity_weight=0.25,
        )

        result_frame = run_loaded_experiment(
            dataset=dataset,
            protected_group_report=protected_group_report,
            settings=settings,
            community_methods=["louvain"],
            baseline_methods=["degree", "random"],
            include_ablations=False,
        )

        self.assertIn(KEPT_RIS_ML_LABEL, set(result_frame["method"]))
        ris_row = result_frame[result_frame["method"] == KEPT_RIS_ML_LABEL].iloc[0]
        self.assertEqual(ris_row["ml_backend"], "ris")
        self.assertTrue(bool(ris_row["ris_enabled"]))
        self.assertEqual(ris_row["ris_mode"], "weak_group_weighted")
        self.assertTrue(pd.isna(ris_row["gnn_model_type"]))

    def test_run_loaded_experiment_with_gnn_ris_backend_adds_combined_row_when_available(self) -> None:
        if not gnn_dependencies_available():
            self.skipTest("torch and torch_geometric are not installed")

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
            ml_backend="gnn_ris",
            ml_guidance_mode="two_tier",
            ml_top_fraction=0.5,
            ml_singleton_runs=3,
            gnn_epochs=10,
            ris_num_rr_sets=32,
        )

        result_frame = run_loaded_experiment(
            dataset=dataset,
            protected_group_report=protected_group_report,
            settings=settings,
            community_methods=["louvain"],
            baseline_methods=["degree", "random"],
            include_ablations=False,
        )

        self.assertIn(KEPT_GNN_RIS_ML_LABEL, set(result_frame["method"]))
        combined_row = result_frame[result_frame["method"] == KEPT_GNN_RIS_ML_LABEL].iloc[0]
        self.assertEqual(combined_row["ml_backend"], "gnn_ris")
        self.assertEqual(combined_row["gnn_model_type"], "graphsage")
        self.assertTrue(bool(combined_row["ris_enabled"]))
        self.assertEqual(combined_row["ris_mode"], "global")

    def test_run_loaded_experiment_with_gnn_ris_node2vec_concat_adds_combined_node2vec_row_when_available(self) -> None:
        if not gnn_dependencies_available():
            self.skipTest("torch and torch_geometric are not installed")

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
            ml_backend="gnn_ris",
            gnn_node2vec_mode="input_concat",
            ml_guidance_mode="two_tier",
            ml_top_fraction=0.5,
            ml_singleton_runs=3,
            gnn_epochs=10,
            ris_num_rr_sets=32,
            node2vec_dimensions=4,
            node2vec_walk_length=6,
            node2vec_num_walks=4,
            node2vec_window=2,
        )

        result_frame = run_loaded_experiment(
            dataset=dataset,
            protected_group_report=protected_group_report,
            settings=settings,
            community_methods=["louvain"],
            baseline_methods=["degree", "random"],
            include_ablations=False,
        )

        self.assertIn(KEPT_GNN_RIS_NODE2VEC_ML_LABEL, set(result_frame["method"]))
        combined_row = result_frame[result_frame["method"] == KEPT_GNN_RIS_NODE2VEC_ML_LABEL].iloc[0]
        self.assertEqual(combined_row["ml_backend"], "gnn_ris")
        self.assertEqual(combined_row["node2vec_mode"], "input_concat")
        self.assertTrue(bool(combined_row["node2vec_enabled"]))
        self.assertTrue(bool(combined_row["ris_enabled"]))

    def test_run_loaded_experiment_uses_eval_budget_and_seed_offset_for_reported_rows(self) -> None:
        import fim_hybrid.experiment_runner as experiment_runner_module  # noqa: PLC0415

        dataset, protected_group_report = _toy_experiment_fixture()
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=3,
            community_method="louvain",
            propagation_probability=0.5,
            mc_runs_search=3,
            mc_runs_eval=7,
            population_size=5,
            generations=3,
            random_seed=9,
        )
        original_evaluate = experiment_runner_module.evaluate_seed_set
        captured_calls: list[tuple[int, int]] = []

        def _recording_evaluate(*args, **kwargs):
            captured_calls.append((int(kwargs["mc_runs"]), int(kwargs["random_seed"])))
            return original_evaluate(*args, **kwargs)

        with patch("fim_hybrid.experiment_runner.evaluate_seed_set", side_effect=_recording_evaluate):
            result_frame = run_loaded_experiment(
                dataset=dataset,
                protected_group_report=protected_group_report,
                settings=settings,
                community_methods=["louvain"],
                baseline_methods=["degree"],
                selected_methods=["degree", "hybrid_siea"],
                include_ablations=False,
            )

        self.assertEqual(set(result_frame["method"]), {"degree", "hybrid_siea"})
        self.assertEqual(captured_calls, [(7, 1_000_009), (7, 1_000_009)])

    def test_run_loaded_experiment_can_apply_final_recheck_with_separate_budget(self) -> None:
        import fim_hybrid.experiment_runner as experiment_runner_module  # noqa: PLC0415

        dataset, protected_group_report = _toy_experiment_fixture()
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=3,
            community_method="louvain",
            propagation_probability=0.5,
            mc_runs_search=3,
            mc_runs_eval=7,
            enable_final_recheck=True,
            final_recheck_mc_runs=11,
            final_recheck_top_k=0,
            population_size=5,
            generations=3,
            random_seed=9,
        )
        original_evaluate = experiment_runner_module.evaluate_seed_set
        captured_calls: list[tuple[int, int]] = []

        def _recording_evaluate(*args, **kwargs):
            captured_calls.append((int(kwargs["mc_runs"]), int(kwargs["random_seed"])))
            return original_evaluate(*args, **kwargs)

        with patch("fim_hybrid.experiment_runner.evaluate_seed_set", side_effect=_recording_evaluate):
            result_frame = run_loaded_experiment(
                dataset=dataset,
                protected_group_report=protected_group_report,
                settings=settings,
                community_methods=["louvain"],
                baseline_methods=["degree"],
                selected_methods=["degree", "hybrid_siea"],
                include_ablations=False,
            )

        self.assertEqual(
            captured_calls,
            [
                (7, 1_000_009),
                (7, 1_000_009),
                (11, 2_000_009),
                (11, 2_000_009),
            ],
        )
        self.assertTrue(result_frame["final_recheck_applied"].all())
        self.assertTrue((result_frame["final_recheck_mc_runs_used"] == 11).all())
        self.assertTrue((result_frame["final_recheck_random_seed"] == 2_000_009).all())
        self.assertTrue(result_frame["final_recheck_f_score"].notna().all())
        self.assertTrue(result_frame["final_recheck_total_spread"].notna().all())
        self.assertTrue(result_frame["final_recheck_extra_spread"].notna().all())

    def test_run_loaded_experiment_can_limit_final_recheck_to_top_k(self) -> None:
        dataset, protected_group_report = _toy_experiment_fixture()
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=3,
            community_method="louvain",
            propagation_probability=0.5,
            mc_runs_search=3,
            mc_runs_eval=7,
            enable_final_recheck=True,
            final_recheck_mc_runs=11,
            final_recheck_top_k=1,
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
            selected_methods=["degree", "random", "hybrid_siea"],
            include_ablations=False,
        )

        rechecked_rows = result_frame[result_frame["final_recheck_applied"]]
        self.assertEqual(len(rechecked_rows), 1)
        self.assertEqual(int(rechecked_rows.iloc[0]["final_recheck_top_k_rank"]), 1)

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
