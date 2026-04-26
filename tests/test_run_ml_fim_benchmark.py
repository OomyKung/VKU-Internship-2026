"""Tests for the unified ML FIM benchmark runner."""

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
    FIMPermutationBenchmarkResult,
    FIMPermutationRunConfig,
    FIMPermutationSpec,
    run_fim_permutation_benchmark,
)
from scripts.run_ml_fim_benchmark import (
    build_ml_benchmark_insights,
    resolve_ml_benchmark_specs,
    run_ml_fim_benchmark,
)
from scripts.evaluate_fim_results import InsightThresholds


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
    dataset = LoadedDataset(name="toy_ml_benchmark", graph=graph, node_attributes=attributes)
    return dataset, verify_protected_groups(dataset, "group")


class RunMlFimBenchmarkTestCase(unittest.TestCase):
    def test_resolve_ml_benchmark_specs_includes_default_stack_set_and_baseline(self) -> None:
        specs = resolve_ml_benchmark_specs(
            ml_stacks=["strong_ml"],
            include_baseline=True,
            embedding_methods=None,
            ranking_models=None,
            community_method=None,
            clustering_method=None,
            spread_estimator_search=None,
            spread_estimator_final="monte_carlo",
            optimizer_mode=None,
        )
        stack_names = [spec.name for spec in specs]
        self.assertIn("community_aware_fair_greedy", stack_names)
        self.assertIn("graphsage_community_siea", stack_names)
        self.assertIn("node2vec_xgboost_community_siea", stack_names)
        self.assertIn("gcn_community_siea", stack_names)
        self.assertNotIn("deepwalk_mlp", stack_names)
        self.assertNotIn("line_fast_ml", stack_names)

    def test_resolve_ml_benchmark_specs_includes_weak_baselines_only_when_requested(self) -> None:
        specs = resolve_ml_benchmark_specs(
            ml_stacks=["strong_ml"],
            include_baseline=True,
            include_weak_ml_baselines=True,
            embedding_methods=None,
            ranking_models=None,
            community_method=None,
            clustering_method=None,
            spread_estimator_search=None,
            spread_estimator_final="monte_carlo",
            optimizer_mode=None,
        )
        stack_names = [spec.name for spec in specs]
        self.assertIn("line_fast_ml", stack_names)
        self.assertIn("deepwalk_mlp", stack_names)

    def test_resolve_ml_benchmark_specs_generates_custom_stack_when_filters_eliminate_named_ones(self) -> None:
        specs = resolve_ml_benchmark_specs(
            ml_stacks=["strong_ml"],
            include_baseline=False,
            embedding_methods=["node2vec"],
            ranking_models=["random_forest"],
            community_method="leiden",
            clustering_method="none",
            spread_estimator_search="monte_carlo",
            spread_estimator_final="monte_carlo",
            optimizer_mode="greedy",
        )

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].embedding_method, "node2vec")
        self.assertEqual(specs[0].ranking_model, "random_forest")
        self.assertEqual(specs[0].runner_kind, "ranked_greedy")

    def test_run_fim_permutation_benchmark_accepts_custom_stack_specs(self) -> None:
        dataset, report = _toy_dataset()
        custom_spec = FIMPermutationSpec(
            name="custom_line_logreg",
            description="Toy custom stack.",
            runner_kind="ranked_greedy",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="monte_carlo",
            spread_estimator_final="monte_carlo",
            embedding_method="line",
            ranking_model="logistic_regression",
            optimizer_mode="greedy",
            fairness_objective="f_score",
            variant_family="ml",
            candidate_top_fraction=0.5,
        )
        config = FIMPermutationRunConfig(
            protected_attribute="group",
            budget=2,
            propagation_probability=0.1,
            mc_runs_search=1,
            mc_runs_eval=2,
            random_seed=7,
            local_search_steps=1,
        )

        result = run_fim_permutation_benchmark(
            dataset,
            report,
            config,
            permutations=[custom_spec],
        )

        row = result.summary_frame.iloc[0]
        self.assertEqual(row["stack_name"], "custom_line_logreg")
        self.assertIn(row["status"], {"ok", "skipped"})

    def test_run_ml_fim_benchmark_saves_outputs_and_builds_ml_insights(self) -> None:
        fake_frame = pd.DataFrame(
            [
                {
                    "stack_name": "community_aware_fair_greedy",
                    "status": "ok",
                    "dataset": "toy_graph",
                    "protected_attribute": "group",
                    "budget": 4,
                    "diffusion_model": "ic",
                    "community_method": "leiden",
                    "embedding_method": "none",
                    "clustering_method": "none",
                    "ranking_model": "fairness_weighted_greedy",
                    "optimizer_mode": "local_search",
                    "debias_mode": "none",
                    "variant_type": "interpretable_baseline",
                    "spread_estimator_search": "monte_carlo",
                    "spread_estimator_final": "monte_carlo",
                    "total_spread": 10.0,
                    "extra_spread": 2.0,
                    "mf": 0.50,
                    "dcv": 0.10,
                    "f_score": 0.120,
                    "runtime_seconds": 8.0,
                    "notes": "baseline",
                    "skip_reason": "",
                    "key_enabled_modules": "baseline",
                },
                {
                    "stack_name": "graphsage_fair_ris_hybrid",
                    "status": "ok",
                    "dataset": "toy_graph",
                    "protected_attribute": "group",
                    "budget": 4,
                    "diffusion_model": "ic",
                    "community_method": "leiden",
                    "embedding_method": "graphsage",
                    "clustering_method": "none",
                    "ranking_model": "graphsage",
                    "optimizer_mode": "hybrid_si_ea",
                    "debias_mode": "worst_group_boost",
                    "variant_type": "ml_guided",
                    "spread_estimator_search": "fairness_aware_ris",
                    "spread_estimator_final": "monte_carlo",
                    "total_spread": 10.7,
                    "extra_spread": 2.7,
                    "mf": 0.49,
                    "dcv": 0.12,
                    "f_score": 0.119,
                    "runtime_seconds": 4.0,
                    "notes": "ml",
                    "skip_reason": "",
                    "key_enabled_modules": "ml",
                },
                {
                    "stack_name": "graphcl_maximin",
                    "status": "skipped",
                    "dataset": "toy_graph",
                    "protected_attribute": "group",
                    "budget": 4,
                    "diffusion_model": "ic",
                    "community_method": "infomap",
                    "embedding_method": "graphcl",
                    "clustering_method": "spectral",
                    "ranking_model": "logistic_regression",
                    "optimizer_mode": "local_search",
                    "debias_mode": "none",
                    "variant_type": "exploratory_fairness",
                    "spread_estimator_search": "monte_carlo",
                    "spread_estimator_final": "monte_carlo",
                    "total_spread": pd.NA,
                    "extra_spread": pd.NA,
                    "mf": pd.NA,
                    "dcv": pd.NA,
                    "f_score": pd.NA,
                    "runtime_seconds": pd.NA,
                    "notes": "skipped",
                    "skip_reason": "optional dependency missing",
                    "key_enabled_modules": "graphcl",
                },
            ]
        )

        specs = resolve_ml_benchmark_specs(
            ml_stacks=["graphsage_fair_ris_hybrid"],
            include_baseline=True,
            embedding_methods=None,
            ranking_models=None,
            community_method=None,
            clustering_method=None,
            spread_estimator_search=None,
            spread_estimator_final="monte_carlo",
            optimizer_mode=None,
        )

        with _WorkspaceScratchDir() as temp_dir:
            with patch(
                "scripts.run_ml_fim_benchmark.run_fim_permutation_benchmark_from_config",
                return_value=FIMPermutationBenchmarkResult(summary_frame=fake_frame, results_by_permutation={}),
            ):
                result = run_ml_fim_benchmark(
                    dataset_config=object(),
                    specs=specs,
                    protected_attributes=["group"],
                    budgets=[4],
                    seeds=[7],
                    base_run_config=FIMPermutationRunConfig(
                        protected_attribute="group",
                        budget=4,
                        mc_runs_search=1,
                        mc_runs_eval=2,
                        random_seed=7,
                    ),
                    output_dir=temp_dir,
                    report_name="toy_ml_benchmark",
                    insight_thresholds=InsightThresholds(),
                    save_json=True,
                )

            self.assertTrue(result.comparison_csv_path is not None and result.comparison_csv_path.exists())
            self.assertTrue(result.normalized_csv_path is not None and result.normalized_csv_path.exists())
            self.assertTrue(result.ranked_csv_path is not None and result.ranked_csv_path.exists())
            self.assertTrue(result.report_path is not None and result.report_path.exists())
            self.assertTrue(result.json_path is not None and result.json_path.exists())
            self.assertIn("ML Benchmark Insights", result.report_text)
            self.assertIn("Best current overall ML stack", result.report_text)

            insight_lines, recommendations = build_ml_benchmark_insights(
                result.evaluation_result,
                result.raw_comparison_frame,
                thresholds=InsightThresholds(),
            )
            self.assertTrue(any("graphsage_fair_ris_hybrid" in line for line in insight_lines))
            self.assertEqual(recommendations["best_interpretable_baseline"], "community_aware_fair_greedy")


if __name__ == "__main__":
    unittest.main()
