"""Tests for named FIM algorithm-stack permutations."""

from __future__ import annotations

from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

import networkx as nx
import pandas as pd

from fim_hybrid.data_loader import LoadedDataset, verify_protected_groups
from fim_hybrid.permutations import (
    FIMPermutationRunConfig,
    _combine_guidance_scores,
    _hybrid_optimizer_config,
    available_fim_permutations,
    format_fim_permutation_report,
    get_fim_permutation_spec,
    run_fim_permutation_benchmark,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = REPO_ROOT / ".test-artifacts"
TEST_TMP_ROOT.mkdir(exist_ok=True)


class _WorkspaceScratchDir:
    def __init__(self) -> None:
        self.path = TEST_TMP_ROOT / f"scratch_{uuid4().hex}"

    def __enter__(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=False)
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def _toy_dataset() -> tuple[LoadedDataset, object]:
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
    groups = {
        1: "A",
        2: "A",
        3: "B",
        4: "B",
        5: "C",
        6: "C",
        7: "D",
        8: "D",
    }
    for node_id, group_name in groups.items():
        graph.nodes[node_id]["group"] = group_name
    attributes = pd.DataFrame(
        [{"node_id": node_id, "group": group_name} for node_id, group_name in sorted(groups.items())]
    ).set_index("node_id", drop=False)
    dataset = LoadedDataset(name="toy_permutation", graph=graph, node_attributes=attributes)
    return dataset, verify_protected_groups(dataset, "group")


class FIMPermutationTestCase(unittest.TestCase):
    def test_registry_lists_requested_permutations(self) -> None:
        self.assertTrue(
            {
                "community_aware_fair_greedy",
                "leiden_graphsage_fair_ris_hybrid",
                "leiden_gcn_fair_ris_hybrid",
                "leiden_line_logreg_greedy",
                "infomap_graphcl_logreg_maximin",
                "leiden_node2vec_xgboost_hybrid",
                "leiden_node2vec_kmeans_logreg_hybrid",
            }.issubset(set(available_fim_permutations()))
        )
        spec = get_fim_permutation_spec("graphsage_fair_ris_hybrid")
        self.assertEqual(spec.name, "leiden_graphsage_fair_ris_hybrid")
        self.assertEqual(spec.embedding_method, "graphsage")
        self.assertEqual(spec.spread_estimator_final, "monte_carlo")

    def test_community_aware_fair_greedy_runs_and_writes_outputs(self) -> None:
        dataset, report = _toy_dataset()
        with _WorkspaceScratchDir() as output_dir:
            config = FIMPermutationRunConfig(
                protected_attribute="group",
                budget=2,
                propagation_probability=0.0,
                mc_runs_search=2,
                mc_runs_eval=3,
                random_seed=7,
                output_dir=output_dir,
                swap_candidate_pool_size=4,
                local_search_steps=1,
            )
            result = run_fim_permutation_benchmark(
                dataset,
                report,
                config,
                permutations=["community_aware_fair_greedy"],
            )

            frame = result.summary_frame
            self.assertEqual(frame.loc[0, "status"], "ok")
            self.assertEqual(frame.loc[0, "stack_name"], "community_aware_fair_greedy")
            self.assertEqual(frame.loc[0, "final_spread_estimator"], "monte_carlo")
            self.assertIn("total_spread", frame.columns)
            self.assertIn("extra_spread", frame.columns)
            self.assertIn("mf", frame.columns)
            self.assertIn("dcv", frame.columns)
            self.assertIn("f_score", frame.columns)
            self.assertIn("ranking_model", frame.columns)
            self.assertIn("clustering_input_mode", frame.columns)
            self.assertTrue(result.comparison_csv_path is not None and result.comparison_csv_path.is_file())
            self.assertTrue(result.report_path is not None and result.report_path.is_file())

    def test_leiden_graphsage_fair_ris_hybrid_delegates_to_existing_experiment_runner(self) -> None:
        dataset, report = _toy_dataset()
        fake_frame = pd.DataFrame(
            [
                {
                    "dataset": dataset.name,
                    "protected_attribute": "group",
                    "community_method": "leiden",
                    "diffusion_model": "ic",
                    "method": "hybrid_siea_ml_gnn_ris_two_tier_tuned_swap_local_search",
                    "variant_type": "ml_guided",
                    "seed_set": "[1, 4]",
                    "total_spread": 2.0,
                    "extra_spread": 0.0,
                    "mf": 0.25,
                    "dcv": 0.0,
                    "f_score": 0.125,
                    "runtime_seconds": 0.5,
                    "search_runtime_seconds": 0.4,
                    "final_eval_runtime_seconds": 0.1,
                    "mc_runs_search": 2,
                    "mc_runs_eval": 3,
                    "note": "fake",
                }
            ]
        )
        config = FIMPermutationRunConfig(
            protected_attribute="group",
            budget=2,
            mc_runs_search=2,
            mc_runs_eval=3,
            random_seed=7,
        )

        with patch("fim_hybrid.permutations.run_loaded_experiment", return_value=fake_frame) as mocked:
            result = run_fim_permutation_benchmark(
                dataset,
                report,
                config,
                permutations=["leiden_graphsage_fair_ris_hybrid"],
            )

        self.assertTrue(mocked.called)
        row = result.summary_frame.iloc[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["stack_name"], "leiden_graphsage_fair_ris_hybrid")
        self.assertEqual(row["embedding_method"], "graphsage")
        self.assertEqual(row["ranking_model"], "graphsage")
        self.assertEqual(row["spread_estimator_search"], "fairness_aware_ris")
        self.assertEqual(row["optimizer_mode"], "hybrid_si_ea")
        self.assertEqual(row["debias_mode"], "worst_group_boost")

    def test_leiden_line_logreg_greedy_runs_end_to_end(self) -> None:
        dataset, report = _toy_dataset()
        config = FIMPermutationRunConfig(
            protected_attribute="group",
            budget=2,
            propagation_probability=0.2,
            mc_runs_search=2,
            mc_runs_eval=3,
            random_seed=7,
        )
        result = run_fim_permutation_benchmark(
            dataset,
            report,
            config,
            permutations=["leiden_line_logreg_greedy"],
        )

        row = result.summary_frame.iloc[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["embedding_method"], "line")
        self.assertEqual(row["ranking_model"], "logistic_regression")
        self.assertEqual(row["clustering_method"], "none")
        self.assertEqual(row["spread_estimator_final"], "monte_carlo")

    def test_clustering_enhanced_stack_runs_or_skips_clearly(self) -> None:
        dataset, report = _toy_dataset()
        config = FIMPermutationRunConfig(
            protected_attribute="group",
            budget=2,
            propagation_probability=0.2,
            mc_runs_search=1,
            mc_runs_eval=2,
            random_seed=7,
            population_size=4,
            generations=2,
            local_search_steps=1,
            ris_num_rr_sets=8,
        )
        result = run_fim_permutation_benchmark(
            dataset,
            report,
            config,
            permutations=["leiden_node2vec_kmeans_logreg_hybrid"],
        )

        row = result.summary_frame.iloc[0]
        self.assertIn(row["status"], {"ok", "skipped"})
        if row["status"] == "ok":
            self.assertEqual(row["clustering_method"], "kmeans")
            self.assertEqual(row["ranking_model"], "logistic_regression")
            self.assertEqual(row["spread_estimator_final"], "monte_carlo")
        else:
            self.assertTrue(str(row["skip_reason"]).strip())

    def test_failures_become_skipped_rows_when_configured(self) -> None:
        dataset, report = _toy_dataset()
        config = FIMPermutationRunConfig(
            protected_attribute="group",
            budget=2,
            mc_runs_search=2,
            mc_runs_eval=3,
            random_seed=7,
            continue_on_error=True,
        )
        with patch("fim_hybrid.permutations.detect_communities", side_effect=ImportError("missing optional package")):
            result = run_fim_permutation_benchmark(
                dataset,
                report,
                config,
                permutations=["infomap_graphcl_logreg_maximin"],
            )

        row = result.summary_frame.iloc[0]
        self.assertEqual(row["status"], "skipped")
        self.assertIn("missing optional package", str(row["skip_reason"]))

    def test_selected_pipeline_config_controls_hybrid_ml_score_stages(self) -> None:
        spec = get_fim_permutation_spec("leiden_node2vec_xgboost_hybrid")
        default_config = FIMPermutationRunConfig(protected_attribute="group", budget=2)
        disabled_config = FIMPermutationRunConfig(
            protected_attribute="group",
            budget=2,
            use_ml_scores_in_initialization=False,
            use_ml_scores_in_mutation=False,
            use_ml_scores_in_repair=False,
            use_ml_scores_in_local_search=False,
        )

        default_optimizer_config = _hybrid_optimizer_config(spec, default_config)
        disabled_optimizer_config = _hybrid_optimizer_config(spec, disabled_config)

        self.assertGreater(default_optimizer_config.ml_initialization_bias, 0.0)
        self.assertGreater(default_optimizer_config.ml_mutation_bias_weight, 0.0)
        self.assertEqual(disabled_optimizer_config.ml_initialization_bias, 0.0)
        self.assertEqual(disabled_optimizer_config.ml_mutation_bias_weight, 0.0)
        self.assertEqual(disabled_optimizer_config.ml_repair_bias_weight, 0.0)
        self.assertEqual(disabled_optimizer_config.ml_local_search_bias_weight, 0.0)

    def test_combined_guidance_scores_include_configurable_bonus_weights(self) -> None:
        spec = get_fim_permutation_spec("leiden_node2vec_xgboost_hybrid")

        scores = _combine_guidance_scores(
            spec,
            ranking_scores={1: 0.1, 2: 0.2},
            ris_scores={1: 0.0, 2: 0.0},
            fair_ris_scores={1: 0.0, 2: 0.0},
            ml_score_weight=1.0,
            fairness_bonus_scores={1: 1.0, 2: 0.0},
            fairness_bonus_weight=2.0,
        )

        self.assertGreater(scores[1], scores[2])

    def test_report_includes_side_by_side_metrics(self) -> None:
        dataset, report = _toy_dataset()
        config = FIMPermutationRunConfig(
            protected_attribute="group",
            budget=2,
            propagation_probability=0.0,
            mc_runs_search=2,
            mc_runs_eval=3,
            random_seed=7,
            swap_candidate_pool_size=4,
        )
        result = run_fim_permutation_benchmark(
            dataset,
            report,
            config,
            permutations=["community_aware_fair_greedy"],
        )
        text = format_fim_permutation_report(result.summary_frame, config)
        self.assertIn("FIM Algorithm-Stack Permutation Comparison", text)
        self.assertIn("community_aware_fair_greedy [ok]", text)
        self.assertIn("F-score=", text)
        self.assertIn("modules: diffusion=ic", text)
        self.assertIn("Recommendations", text)


if __name__ == "__main__":
    unittest.main()
