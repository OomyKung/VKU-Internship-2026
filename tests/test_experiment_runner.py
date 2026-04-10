"""Unit tests for the experiment comparison runner."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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

    def test_node2vec_requires_ml_mode(self) -> None:
        dataset, protected_group_report = _toy_experiment_fixture()
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=3,
            use_node2vec=True,
        )

        with self.assertRaisesRegex(ValueError, "use_node2vec requires use_ml"):
            run_loaded_experiment(
                dataset=dataset,
                protected_group_report=protected_group_report,
                settings=settings,
                include_ablations=False,
            )

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

    def test_run_loaded_experiment_with_ml_adds_ml_rows(self) -> None:
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

        self.assertEqual(
            set(result_frame["method"]),
            {
                "degree",
                "random",
                "cea_fim",
                "hybrid_siea",
                "ml_topk",
                "hybrid_siea_ml_hard_filter",
                "hybrid_siea_ml_soft_bias",
                "hybrid_siea_ml_two_tier",
                "hybrid_siea_ml_two_tier_tuned",
            },
        )
        ml_rows = result_frame[result_frame["method"].isin([
            "ml_topk",
            "hybrid_siea_ml_hard_filter",
            "hybrid_siea_ml_soft_bias",
            "hybrid_siea_ml_two_tier",
            "hybrid_siea_ml_two_tier_tuned",
        ])]
        self.assertTrue(ml_rows["ml_validation_spearman"].notna().all())
        self.assertTrue(ml_rows["ml_validation_precision_at_budget"].notna().all())
        hard_filter_row = result_frame[result_frame["method"] == "hybrid_siea_ml_hard_filter"].iloc[0]
        soft_bias_row = result_frame[result_frame["method"] == "hybrid_siea_ml_soft_bias"].iloc[0]
        two_tier_row = result_frame[result_frame["method"] == "hybrid_siea_ml_two_tier"].iloc[0]
        tuned_two_tier_row = result_frame[result_frame["method"] == "hybrid_siea_ml_two_tier_tuned"].iloc[0]
        self.assertLess(int(hard_filter_row["candidate_pool_size"]), dataset.graph.number_of_nodes())
        self.assertEqual(int(soft_bias_row["candidate_pool_size"]), dataset.graph.number_of_nodes())
        self.assertEqual(int(two_tier_row["candidate_pool_size"]), dataset.graph.number_of_nodes())
        self.assertEqual(int(tuned_two_tier_row["candidate_pool_size"]), dataset.graph.number_of_nodes())
        self.assertEqual(hard_filter_row["ml_guidance_mode"], "hard_filter")
        self.assertEqual(soft_bias_row["ml_guidance_mode"], "soft_bias")
        self.assertEqual(two_tier_row["ml_guidance_mode"], "two_tier")
        self.assertEqual(tuned_two_tier_row["ml_guidance_mode"], "two_tier")

    def test_run_loaded_experiment_trains_ml_model_once_for_all_modes(self) -> None:
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
            ml_guidance_mode="off",
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

    def test_run_loaded_experiment_with_fairness_compare_adds_variant_rows(self) -> None:
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
            compare_fairness_variants=True,
        )

        result_frame = run_loaded_experiment(
            dataset=dataset,
            protected_group_report=protected_group_report,
            settings=settings,
            community_methods=["louvain"],
            baseline_methods=["degree", "random"],
            include_ablations=False,
        )

        expected_methods = {
            "hybrid_siea_ml_two_tier_tuned",
            "hybrid_siea_ml_two_tier_tuned_fair_init",
            "hybrid_siea_ml_two_tier_tuned_weak_mutation",
            "hybrid_siea_ml_two_tier_tuned_fair_repair",
            "hybrid_siea_ml_two_tier_tuned_worst_group_local_search",
            "hybrid_siea_ml_two_tier_tuned_fairness_full",
        }
        self.assertTrue(expected_methods.issubset(set(result_frame["method"])))
        fairness_rows = result_frame[result_frame["variant_type"] == "fairness_variant"]
        self.assertEqual(set(fairness_rows["ml_guidance_mode"]), {"two_tier"})
        self.assertTrue(fairness_rows["ml_validation_spearman"].notna().all())

    def test_run_loaded_experiment_trains_ml_model_once_with_fairness_compare(self) -> None:
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
            compare_fairness_variants=True,
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

    def test_run_loaded_experiment_with_refinement_compare_adds_variant_rows(self) -> None:
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
            compare_refinement_variants=True,
        )

        result_frame = run_loaded_experiment(
            dataset=dataset,
            protected_group_report=protected_group_report,
            settings=settings,
            community_methods=["louvain"],
            baseline_methods=["degree", "random"],
            include_ablations=False,
        )

        expected_methods = {
            "hybrid_siea_ml_two_tier_tuned_fairness_full",
            "hybrid_siea_ml_two_tier_tuned_marginal_gain",
            "hybrid_siea_ml_two_tier_tuned_swap_local_search",
            "hybrid_siea_ml_two_tier_tuned_urgency_weighted",
            "hybrid_siea_ml_two_tier_tuned_overlap_penalty",
            "hybrid_siea_ml_two_tier_tuned_refinement_full",
        }
        self.assertTrue(expected_methods.issubset(set(result_frame["method"])))
        refinement_rows = result_frame[result_frame["variant_type"].isin(["refinement_baseline", "refinement_variant"])]
        self.assertEqual(set(refinement_rows["ml_guidance_mode"]), {"two_tier"})
        self.assertTrue(refinement_rows["delta_f_score"].notna().all())
        baseline_row = result_frame[result_frame["method"] == "hybrid_siea_ml_two_tier_tuned_fairness_full"].iloc[0]
        self.assertEqual(baseline_row["comparison_baseline_method"], "hybrid_siea_ml_two_tier_tuned_fairness_full")
        self.assertAlmostEqual(float(baseline_row["delta_f_score"]), 0.0)

    def test_run_loaded_experiment_trains_ml_model_once_with_refinement_compare(self) -> None:
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
            compare_refinement_variants=True,
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

    def test_run_loaded_experiment_with_runtime_compare_adds_runtime_rows(self) -> None:
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
            compare_runtime_variants=True,
        )

        result_frame = run_loaded_experiment(
            dataset=dataset,
            protected_group_report=protected_group_report,
            settings=settings,
            community_methods=["louvain"],
            baseline_methods=["degree", "random"],
            include_ablations=False,
        )

        expected_methods = {
            "hybrid_siea_ml_two_tier_tuned_marginal_gain",
            "hybrid_siea_ml_two_tier_tuned_marginal_gain_optimized",
            "hybrid_siea_ml_two_tier_tuned_marginal_gain_balanced",
            "hybrid_siea_ml_two_tier_tuned_marginal_gain_fast",
        }
        self.assertTrue(expected_methods.issubset(set(result_frame["method"])))
        runtime_rows = result_frame[result_frame["variant_type"].isin(["runtime_baseline", "runtime_variant"])]
        self.assertEqual(set(runtime_rows["comparison_baseline_method"]), {"hybrid_siea_ml_two_tier_tuned_marginal_gain"})
        baseline_row = result_frame[result_frame["method"] == "hybrid_siea_ml_two_tier_tuned_marginal_gain"].iloc[0]
        self.assertAlmostEqual(float(baseline_row["delta_f_score"]), 0.0)

    def test_run_loaded_experiment_trains_ml_model_once_with_runtime_compare(self) -> None:
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
            compare_runtime_variants=True,
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

    def test_run_loaded_experiment_with_swap_runtime_compare_adds_runtime_rows(self) -> None:
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
            compare_swap_runtime_variants=True,
        )

        result_frame = run_loaded_experiment(
            dataset=dataset,
            protected_group_report=protected_group_report,
            settings=settings,
            community_methods=["louvain"],
            baseline_methods=["degree", "random"],
            include_ablations=False,
        )

        expected_methods = {
            "hybrid_siea_ml_two_tier_tuned_swap_local_search",
            "hybrid_siea_ml_two_tier_tuned_swap_local_search_optimized",
            "hybrid_siea_ml_two_tier_tuned_swap_local_search_first_improvement",
            "hybrid_siea_ml_two_tier_tuned_swap_local_search_reduced_candidates",
        }
        self.assertTrue(expected_methods.issubset(set(result_frame["method"])))
        runtime_rows = result_frame[result_frame["variant_type"].isin(["swap_runtime_baseline", "swap_runtime_variant"])]
        self.assertEqual(set(runtime_rows["comparison_baseline_method"]), {"hybrid_siea_ml_two_tier_tuned_swap_local_search"})
        baseline_row = result_frame[result_frame["method"] == "hybrid_siea_ml_two_tier_tuned_swap_local_search"].iloc[0]
        self.assertAlmostEqual(float(baseline_row["delta_f_score"]), 0.0)

    def test_run_loaded_experiment_with_scalability_compare_adds_rows(self) -> None:
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
            compare_scalability_variants=True,
        )

        result_frame = run_loaded_experiment(
            dataset=dataset,
            protected_group_report=protected_group_report,
            settings=settings,
            community_methods=["louvain"],
            baseline_methods=["degree", "random"],
            include_ablations=False,
        )

        expected_methods = {
            "hybrid_siea_ml_two_tier_tuned_fairness_full",
            "hybrid_siea_ml_two_tier_tuned_fairness_full_balanced",
            "hybrid_siea_ml_two_tier_tuned_fairness_full_fast",
        }
        self.assertTrue(expected_methods.issubset(set(result_frame["method"])))
        scalability_rows = result_frame[result_frame["variant_type"].isin(["scalability_baseline", "scalability_variant"])]
        self.assertEqual(set(scalability_rows["comparison_baseline_method"]), {"hybrid_siea_ml_two_tier_tuned_fairness_full"})
        self.assertEqual(
            set(scalability_rows["optimization_mode"]),
            {"full", "balanced", "fast"},
        )

    def test_run_loaded_experiment_trains_ml_model_once_with_scalability_compare(self) -> None:
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
            compare_scalability_variants=True,
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

    def test_run_loaded_experiment_trains_ml_model_once_with_swap_runtime_compare(self) -> None:
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
            compare_swap_runtime_variants=True,
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

    def test_run_loaded_experiment_with_node2vec_adds_node2vec_comparison_rows(self) -> None:
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
            use_node2vec=True,
            ml_guidance_mode="two_tier",
            ml_top_fraction=0.5,
            ml_singleton_runs=3,
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

        self.assertIn("hybrid_siea", set(result_frame["method"]))
        self.assertIn("hybrid_siea_ml_two_tier_tuned", set(result_frame["method"]))
        self.assertIn("hybrid_siea_ml_two_tier_tuned_node2vec", set(result_frame["method"]))
        node2vec_row = result_frame[result_frame["method"] == "hybrid_siea_ml_two_tier_tuned_node2vec"].iloc[0]
        self.assertTrue(bool(node2vec_row["node2vec_enabled"]))
        self.assertEqual(node2vec_row["node2vec_mode"], "feature_concat")
        self.assertEqual(node2vec_row["ml_guidance_mode"], "two_tier")
        self.assertFalse(pd.isna(node2vec_row["ml_validation_spearman"]))
        self.assertFalse(pd.isna(node2vec_row["ml_validation_precision_at_budget"]))

    def test_run_loaded_experiment_with_node2vec_both_modes_adds_diversity_row(self) -> None:
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
            use_node2vec=True,
            ml_guidance_mode="two_tier",
            ml_top_fraction=0.5,
            ml_singleton_runs=3,
            node2vec_dimensions=4,
            node2vec_walk_length=6,
            node2vec_num_walks=4,
            node2vec_window=2,
            node2vec_integration_mode="both",
        )

        result_frame = run_loaded_experiment(
            dataset=dataset,
            protected_group_report=protected_group_report,
            settings=settings,
            community_methods=["louvain"],
            baseline_methods=["degree", "random"],
            include_ablations=False,
        )

        self.assertIn("hybrid_siea_ml_two_tier_tuned_node2vec_diversity", set(result_frame["method"]))
        diversity_row = result_frame[result_frame["method"] == "hybrid_siea_ml_two_tier_tuned_node2vec_diversity"].iloc[0]
        self.assertTrue(bool(diversity_row["node2vec_enabled"]))
        self.assertEqual(diversity_row["node2vec_mode"], "diversity_signal")
        self.assertEqual(diversity_row["ml_guidance_mode"], "two_tier")

    def test_run_loaded_experiment_can_select_one_ml_guidance_mode(self) -> None:
        dataset, protected_group_report = _toy_experiment_fixture()

        for guidance_mode in ["hard_filter", "soft_bias", "two_tier"]:
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
                ml_guidance_mode=guidance_mode,
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
            expected_method = (
                "hybrid_siea_ml_two_tier_tuned"
                if guidance_mode == "two_tier"
                else f"hybrid_siea_ml_{guidance_mode}"
            )
            self.assertEqual(ml_methods, [expected_method])

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
