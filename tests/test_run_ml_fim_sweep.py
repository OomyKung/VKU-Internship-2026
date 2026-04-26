"""Tests for the ML-FIM multi-budget sweep runner."""

from __future__ import annotations

from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

import networkx as nx
import pandas as pd

from fim_hybrid.data_loader import LoadedDataset
from fim_hybrid.permutations import FIMPermutationBenchmarkResult, FIMPermutationRunConfig
from scripts.evaluate_fim_results import InsightThresholds
from scripts.run_ml_fim_benchmark import resolve_ml_benchmark_specs
from scripts.run_ml_fim_sweep import (
    build_sweep_aggregate,
    parse_args,
    rank_sweep_aggregate,
    resolve_budget_values,
    resolve_protected_attributes,
    resolve_seed_values,
    run_ml_fim_sweep,
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


def _toy_dataset() -> LoadedDataset:
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
    attributes = pd.DataFrame(
        [
            {"node_id": 1, "region": "A"},
            {"node_id": 2, "region": "A"},
            {"node_id": 3, "region": "B"},
            {"node_id": 4, "region": "B"},
        ]
    ).set_index("node_id", drop=False)
    return LoadedDataset(name="toy_sweep", graph=graph, node_attributes=attributes)


def _raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "toy",
                "protected_attribute": "region",
                "budget": 2,
                "random_seed": 1,
                "stack_name": "community_aware_fair_greedy",
                "status": "ok",
                "variant_type": "interpretable_baseline",
                "embedding_method": "none",
                "total_spread": 5.0,
                "extra_spread": 3.0,
                "mf": 0.20,
                "dcv": 0.10,
                "f_score": 0.150,
                "runtime_seconds": 10.0,
            },
            {
                "dataset": "toy",
                "protected_attribute": "region",
                "budget": 2,
                "random_seed": 1,
                "stack_name": "graphsage_fair_ris_hybrid",
                "status": "ok",
                "variant_type": "ml_guided",
                "embedding_method": "graphsage",
                "total_spread": 5.5,
                "extra_spread": 3.5,
                "mf": 0.19,
                "dcv": 0.11,
                "f_score": 0.149,
                "runtime_seconds": 3.0,
            },
            {
                "dataset": "toy",
                "protected_attribute": "region",
                "budget": 2,
                "random_seed": 7,
                "stack_name": "community_aware_fair_greedy",
                "status": "ok",
                "variant_type": "interpretable_baseline",
                "embedding_method": "none",
                "total_spread": 5.1,
                "extra_spread": 3.1,
                "mf": 0.21,
                "dcv": 0.10,
                "f_score": 0.151,
                "runtime_seconds": 9.0,
            },
            {
                "dataset": "toy",
                "protected_attribute": "region",
                "budget": 2,
                "random_seed": 7,
                "stack_name": "graphsage_fair_ris_hybrid",
                "status": "skipped",
                "variant_type": "ml_guided",
                "embedding_method": "graphsage",
                "skip_reason": "optional dependency missing",
            },
        ]
    )


class RunMlFimSweepTestCase(unittest.TestCase):
    def test_cli_aliases_resolve_requested_sweep_axes(self) -> None:
        args = parse_args(
            [
                "--dataset",
                "graph_spa_500_0",
                "--protected-attribute",
                "region",
                "--protected-attributes",
                "ethnicity",
                "gender",
                "--budgets",
                "10",
                "20",
                "--random-seeds",
                "1",
                "7",
            ]
        )

        self.assertEqual(resolve_protected_attributes(args.protected_attribute, args.protected_attributes), ["region", "ethnicity", "gender"])
        self.assertEqual(resolve_budget_values(args.budget, args.budgets), [10, 20])
        self.assertEqual(resolve_seed_values(args.random_seed, args.random_seeds), [1, 7])

    def test_aggregate_computes_requested_columns_wins_and_rank(self) -> None:
        aggregate = build_sweep_aggregate(_raw_rows())
        ranked = rank_sweep_aggregate(aggregate)

        baseline = aggregate[aggregate["stack_name"].eq("community_aware_fair_greedy")].iloc[0]
        graphsage = aggregate[aggregate["stack_name"].eq("graphsage_fair_ris_hybrid")].iloc[0]
        self.assertAlmostEqual(float(baseline["mean_f_score"]), 0.1505, places=6)
        self.assertGreater(float(baseline["std_f_score"]), 0.0)
        self.assertEqual(int(baseline["win_count_by_f_score"]), 2)
        self.assertEqual(int(graphsage["win_count_by_runtime"]), 1)
        self.assertEqual(int(graphsage["number_of_successful_runs"]), 1)
        self.assertEqual(int(graphsage["number_of_failed_or_skipped_runs"]), 1)
        self.assertEqual(str(ranked.iloc[0]["stack_name"]), "community_aware_fair_greedy")

    def test_sweep_saves_outputs_and_preserves_skipped_and_failed_runs(self) -> None:
        dataset = _toy_dataset()
        specs = resolve_ml_benchmark_specs(
            ml_stacks=["line_fast_ml"],
            include_baseline=True,
            embedding_methods=None,
            ranking_models=None,
            community_method=None,
            clustering_method=None,
            spread_estimator_search=None,
            spread_estimator_final="monte_carlo",
            optimizer_mode=None,
        )

        def fake_run_fim_permutation_benchmark(*, dataset, protected_group_report, config, permutations):
            if int(config.budget) == 3:
                raise RuntimeError("simulated stack runner failure")
            rows = []
            for index, spec in enumerate(permutations):
                rows.append(
                    {
                        "stack_name": spec.name,
                        "status": "ok",
                        "dataset": dataset.name,
                        "protected_attribute": protected_group_report.protected_attribute,
                        "budget": int(config.budget),
                        "random_seed": int(config.random_seed),
                        "variant_type": "interpretable_baseline" if spec.name == "community_aware_fair_greedy" else "ml_guided",
                        "embedding_method": spec.embedding_method,
                        "ranking_model": spec.ranking_model,
                        "total_spread": 4.0 + index,
                        "extra_spread": 2.0 + index,
                        "mf": 0.2 - (0.01 * index),
                        "dcv": 0.1 + (0.01 * index),
                        "f_score": 0.15 - (0.001 * index),
                        "runtime_seconds": 5.0 - index,
                        "skip_reason": "",
                    }
                )
            return FIMPermutationBenchmarkResult(summary_frame=pd.DataFrame(rows), results_by_permutation={})

        with _WorkspaceScratchDir() as temp_dir:
            with patch("scripts.run_ml_fim_sweep.load_dataset", return_value=dataset):
                with patch(
                    "scripts.run_ml_fim_sweep.run_fim_permutation_benchmark",
                    side_effect=fake_run_fim_permutation_benchmark,
                ):
                    result = run_ml_fim_sweep(
                        dataset_config=object(),
                        specs=specs,
                        protected_attributes=["region"],
                        budgets=[2, 3, 100],
                        seeds=[7],
                        base_run_config=FIMPermutationRunConfig(
                            protected_attribute="region",
                            budget=2,
                            mc_runs_search=1,
                            mc_runs_eval=2,
                            random_seed=7,
                            continue_on_error=True,
                        ),
                        output_dir=temp_dir,
                        report_name="toy_budget9_sweep",
                        insight_thresholds=InsightThresholds(),
                        save_json=True,
                        print_progress=False,
                    )

            self.assertTrue(result.raw_csv_path.exists())
            self.assertTrue(result.aggregate_csv_path.exists())
            self.assertTrue(result.ranked_aggregate_csv_path.exists())
            self.assertTrue(result.report_path.exists())
            self.assertTrue(result.json_path is not None and result.json_path.exists())
            self.assertIn("report_name contains budget 9", "\n".join(result.warnings))
            self.assertTrue(result.raw_frame["skip_reason"].astype(str).str.contains("simulated stack runner failure", regex=False).any())
            self.assertTrue(result.raw_frame["skip_reason"].astype(str).str.contains("exceeds graph node count", regex=False).any())
            self.assertIn("mean_f_score", result.aggregate_frame.columns)
            self.assertIn("win_count_by_runtime", result.aggregate_frame.columns)
            self.assertIn("ML-FIM Sweep Report", result.report_text)


if __name__ == "__main__":
    unittest.main()
