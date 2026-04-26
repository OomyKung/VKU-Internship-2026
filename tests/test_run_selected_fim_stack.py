"""Tests for the selected-stack runner CLI helpers."""

from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import pandas as pd

from fim_hybrid.permutations import FIMPermutationBenchmarkResult
from scripts.run_selected_fim_stack import parse_args, run_selected_fim_stack


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


def _benchmark_csv(path: Path) -> Path:
    frame = pd.DataFrame(
        [
            {
                "stack_name": "community_aware_fair_greedy",
                "protected_attribute": "region",
                "budget": 40,
                "status": "ok",
                "mean_f_score": 0.100,
                "mean_mf": 0.2,
                "mean_dcv": 0.08,
                "mean_spread": 42.0,
                "mean_runtime": 50.0,
            },
            {
                "stack_name": "graphsage_fair_ris_hybrid",
                "protected_attribute": "region",
                "budget": 40,
                "status": "ok",
                "mean_f_score": 0.099,
                "mean_mf": 0.2,
                "mean_dcv": 0.08,
                "mean_spread": 42.0,
                "mean_runtime": 8.0,
            },
        ]
    )
    frame.to_csv(path, index=False)
    return path


def _fake_dataset(node_count: int = 100):
    return SimpleNamespace(graph=SimpleNamespace(number_of_nodes=lambda: node_count))


def _fake_result_row(
    stack_name: str,
    *,
    f_score: float,
    mf: float = 0.2,
    dcv: float = 0.08,
    runtime: float = 10.0,
) -> FIMPermutationBenchmarkResult:
    frame = pd.DataFrame(
        [
            {
                "stack_name": stack_name,
                "status": "ok",
                "dataset": "graph_spa_500_0",
                "protected_attribute": "region",
                "total_spread": 42.0,
                "extra_spread": 2.0,
                "mf": mf,
                "dcv": dcv,
                "f_score": f_score,
                "runtime_seconds": runtime,
                "final_spread_estimator": "monte_carlo",
            }
        ]
    )
    return FIMPermutationBenchmarkResult(summary_frame=frame, results_by_permutation={})


def _spec_name(permutations) -> str:
    return str(list(permutations)[0].name)


class RunSelectedFimStackTestCase(unittest.TestCase):
    def test_auto_selected_stack_exports_config_and_runs_one_stack(self) -> None:
        with _WorkspaceScratchDir() as temp_dir:
            benchmark_path = _benchmark_csv(temp_dir / "aggregate.csv")
            args = parse_args(
                [
                    "--dataset",
                    "graph_spa_500_0",
                    "--protected-attribute",
                    "region",
                    "--budget",
                    "40",
                    "--auto-select-stack-from",
                    str(benchmark_path),
                    "--output-dir",
                    str(temp_dir / "out"),
                    "--export-selected-stack-config",
                    str(temp_dir / "selected.json"),
                    "--save-json",
                ]
            )
            fake_frame = pd.DataFrame(
                [
                    {
                        "stack_name": "graphsage_fair_ris_hybrid",
                        "status": "ok",
                        "total_spread": 42.0,
                        "extra_spread": 2.0,
                        "mf": 0.2,
                        "dcv": 0.08,
                        "f_score": 0.10,
                        "runtime_seconds": 7.0,
                        "final_spread_estimator": "monte_carlo",
                    }
                ]
            )
            with patch(
                "scripts.run_selected_fim_stack.run_fim_permutation_benchmark_from_config",
                return_value=FIMPermutationBenchmarkResult(summary_frame=fake_frame, results_by_permutation={}),
            ) as runner:
                result = run_selected_fim_stack(args)

            self.assertEqual(result["selected"].selected_stack, "graphsage_fair_ris_hybrid")
            self.assertTrue(Path(result["selected_config_path"]).exists())
            self.assertTrue(Path(result["result_csv_path"]).exists())
            self.assertTrue(Path(result["report_path"]).exists())
            self.assertTrue(Path(result["json_path"]).exists())
            runner.assert_called_once()
            called_config = runner.call_args.kwargs["config"]
            self.assertEqual(called_config.spread_estimator_final if hasattr(called_config, "spread_estimator_final") else "monte_carlo", "monte_carlo")
            self.assertEqual(called_config.community_feature_mode, "basic")
            self.assertTrue(called_config.use_ml_scores_in_initialization)

    def test_loaded_selected_stack_config_executes(self) -> None:
        with _WorkspaceScratchDir() as temp_dir:
            benchmark_path = _benchmark_csv(temp_dir / "aggregate.csv")
            select_args = parse_args(
                [
                    "--dataset",
                    "graph_spa_500_0",
                    "--protected-attribute",
                    "region",
                    "--budget",
                    "40",
                    "--auto-select-stack-from",
                    str(benchmark_path),
                    "--output-dir",
                    str(temp_dir / "first"),
                    "--export-selected-stack-config",
                    str(temp_dir / "selected.json"),
                ]
            )
            fake_frame = pd.DataFrame(
                [
                    {
                        "stack_name": "graphsage_fair_ris_hybrid",
                        "status": "ok",
                        "total_spread": 42.0,
                        "extra_spread": 2.0,
                        "mf": 0.2,
                        "dcv": 0.08,
                        "f_score": 0.10,
                        "runtime_seconds": 7.0,
                        "final_spread_estimator": "monte_carlo",
                    }
                ]
            )
            with patch(
                "scripts.run_selected_fim_stack.run_fim_permutation_benchmark_from_config",
                return_value=FIMPermutationBenchmarkResult(summary_frame=fake_frame, results_by_permutation={}),
            ):
                run_selected_fim_stack(select_args)

            load_args = parse_args(
                [
                    "--dataset",
                    "graph_spa_500_0",
                    "--protected-attribute",
                    "region",
                    "--budget",
                    "40",
                    "--use-selected-stack-config",
                    str(temp_dir / "selected.json"),
                    "--output-dir",
                    str(temp_dir / "second"),
                ]
            )
            with patch(
                "scripts.run_selected_fim_stack.run_fim_permutation_benchmark_from_config",
                return_value=FIMPermutationBenchmarkResult(summary_frame=fake_frame, results_by_permutation={}),
            ):
                result = run_selected_fim_stack(load_args)

        self.assertEqual(result["selected"].selected_stack, "graphsage_fair_ris_hybrid")

    def test_auto_select_accepts_directory_and_chooses_likely_csv(self) -> None:
        with _WorkspaceScratchDir() as temp_dir:
            benchmark_dir = temp_dir / "benchmark_dir"
            benchmark_dir.mkdir()
            benchmark_path = _benchmark_csv(benchmark_dir / "custom_ranked_aggregate.csv")
            args = parse_args(
                [
                    "--dataset",
                    "graph_spa_500_0",
                    "--protected-attribute",
                    "region",
                    "--budget",
                    "40",
                    "--auto-select-stack-from",
                    str(benchmark_dir),
                    "--output-dir",
                    str(temp_dir / "out"),
                ]
            )
            fake_frame = pd.DataFrame(
                [
                    {
                        "stack_name": "graphsage_fair_ris_hybrid",
                        "status": "ok",
                        "total_spread": 42.0,
                        "extra_spread": 2.0,
                        "mf": 0.2,
                        "dcv": 0.08,
                        "f_score": 0.10,
                        "runtime_seconds": 7.0,
                        "final_spread_estimator": "monte_carlo",
                    }
                ]
            )

            with patch("scripts.run_selected_fim_stack.load_dataset", return_value=_fake_dataset()), patch(
                "scripts.run_selected_fim_stack.run_fim_permutation_benchmark_from_config",
                return_value=FIMPermutationBenchmarkResult(summary_frame=fake_frame, results_by_permutation={}),
            ):
                result = run_selected_fim_stack(args)

            self.assertTrue(benchmark_path.exists())
            self.assertEqual(result["initial_selected"].selected_stack, "graphsage_fair_ris_hybrid")

    def test_missing_benchmark_path_error_lists_nearby_csvs_and_example(self) -> None:
        with _WorkspaceScratchDir() as temp_dir:
            nearby = _benchmark_csv(temp_dir / "nearby_aggregate.csv")
            args = parse_args(
                [
                    "--dataset",
                    "graph_spa_500_0",
                    "--protected-attribute",
                    "region",
                    "--budget",
                    "40",
                    "--auto-select-stack-from",
                    str(temp_dir / "missing" / "aggregate.csv"),
                    "--output-dir",
                    str(temp_dir / "out"),
                ]
            )
            with patch("scripts.run_selected_fim_stack._nearby_results_csvs", return_value=[nearby]):
                with self.assertRaisesRegex(FileNotFoundError, "requested_path_exists=False"):
                    run_selected_fim_stack(args)

    def test_validation_accepts_candidate_and_saves_required_outputs(self) -> None:
        with _WorkspaceScratchDir() as temp_dir:
            benchmark_path = _benchmark_csv(temp_dir / "aggregate.csv")
            args = parse_args(
                [
                    "--dataset",
                    "graph_spa_500_0",
                    "--protected-attribute",
                    "region",
                    "--budget",
                    "40",
                    "--auto-select-stack-from",
                    str(benchmark_path),
                    "--validate-selected-stack",
                    "--validation-seeds",
                    "7",
                    "21",
                    "--output-dir",
                    str(temp_dir / "out"),
                ]
            )

            def fake_runner(*, dataset_config, config, permutations):
                stack_name = _spec_name(permutations)
                if "community_aware_fair_greedy" in stack_name:
                    return _fake_result_row(stack_name, f_score=0.100, dcv=0.080, runtime=50.0)
                return _fake_result_row(stack_name, f_score=0.099, dcv=0.085, runtime=8.0)

            with patch("scripts.run_selected_fim_stack.load_dataset", return_value=_fake_dataset()), patch(
                "scripts.run_selected_fim_stack.run_fim_permutation_benchmark_from_config",
                side_effect=fake_runner,
            ):
                result = run_selected_fim_stack(args)

            self.assertEqual(result["selected"].selected_stack, "graphsage_fair_ris_hybrid")
            self.assertEqual(result["fallback_selected"].selected_stack, "community_aware_fair_greedy")
            self.assertTrue(result["validation_summary"]["accepted"])
            self.assertEqual(result["validation_summary"]["fallback_stack"], "community_aware_fair_greedy")
            self.assertAlmostEqual(float(result["validation_summary"]["mf_ratio"]), 1.0)
            self.assertTrue(Path(result["initial_config_path"]).exists())
            self.assertTrue(Path(result["validation_csv_path"]).exists())
            self.assertTrue(Path(result["validation_summary_path"]).exists())
            self.assertTrue(Path(result["final_config_path"]).exists())
            self.assertTrue(Path(result["fixed_report_path"]).exists())

    def test_validation_rejects_candidate_when_fscore_drops(self) -> None:
        with _WorkspaceScratchDir() as temp_dir:
            benchmark_path = _benchmark_csv(temp_dir / "aggregate.csv")
            args = parse_args(
                [
                    "--dataset",
                    "graph_spa_500_0",
                    "--protected-attribute",
                    "region",
                    "--budget",
                    "40",
                    "--auto-select-stack-from",
                    str(benchmark_path),
                    "--validate-selected-stack",
                    "--validation-seeds",
                    "7",
                    "21",
                    "--output-dir",
                    str(temp_dir / "out"),
                ]
            )

            def fake_runner(*, dataset_config, config, permutations):
                stack_name = _spec_name(permutations)
                if "community_aware_fair_greedy" in stack_name:
                    return _fake_result_row(stack_name, f_score=0.100, dcv=0.080, runtime=50.0)
                return _fake_result_row(stack_name, f_score=0.060, dcv=0.085, runtime=8.0)

            with patch("scripts.run_selected_fim_stack.load_dataset", return_value=_fake_dataset()), patch(
                "scripts.run_selected_fim_stack.run_fim_permutation_benchmark_from_config",
                side_effect=fake_runner,
            ):
                result = run_selected_fim_stack(args)

            self.assertEqual(result["selected"].selected_stack, "community_aware_fair_greedy")
            self.assertFalse(result["validation_summary"]["accepted"])
            self.assertIn("F-score dropped", str(result["validation_summary"]["final_reason"]))

    def test_validation_rejects_candidate_when_dcv_delta_is_too_high(self) -> None:
        with _WorkspaceScratchDir() as temp_dir:
            benchmark_path = _benchmark_csv(temp_dir / "aggregate.csv")
            args = parse_args(
                [
                    "--dataset",
                    "graph_spa_500_0",
                    "--protected-attribute",
                    "region",
                    "--budget",
                    "40",
                    "--auto-select-stack-from",
                    str(benchmark_path),
                    "--validate-selected-stack",
                    "--validation-seeds",
                    "7",
                    "21",
                    "--output-dir",
                    str(temp_dir / "out"),
                ]
            )

            def fake_runner(*, dataset_config, config, permutations):
                stack_name = _spec_name(permutations)
                if "community_aware_fair_greedy" in stack_name:
                    return _fake_result_row(stack_name, f_score=0.100, dcv=0.080, runtime=50.0)
                return _fake_result_row(stack_name, f_score=0.099, dcv=0.120, runtime=8.0)

            with patch("scripts.run_selected_fim_stack.load_dataset", return_value=_fake_dataset()), patch(
                "scripts.run_selected_fim_stack.run_fim_permutation_benchmark_from_config",
                side_effect=fake_runner,
            ):
                result = run_selected_fim_stack(args)

            self.assertEqual(result["selected"].selected_stack, "community_aware_fair_greedy")
            self.assertFalse(result["validation_summary"]["accepted"])
            self.assertIn("DCV increased", str(result["validation_summary"]["final_reason"]))

    def test_validation_rejects_candidate_when_mf_ratio_is_too_low(self) -> None:
        with _WorkspaceScratchDir() as temp_dir:
            benchmark_path = _benchmark_csv(temp_dir / "aggregate.csv")
            args = parse_args(
                [
                    "--dataset",
                    "graph_spa_500_0",
                    "--protected-attribute",
                    "region",
                    "--budget",
                    "40",
                    "--auto-select-stack-from",
                    str(benchmark_path),
                    "--validate-selected-stack",
                    "--validation-seeds",
                    "7",
                    "21",
                    "--output-dir",
                    str(temp_dir / "out"),
                ]
            )

            def fake_runner(*, dataset_config, config, permutations):
                stack_name = _spec_name(permutations)
                if "community_aware_fair_greedy" in stack_name:
                    return _fake_result_row(stack_name, f_score=0.100, mf=0.200, dcv=0.080, runtime=50.0)
                return _fake_result_row(stack_name, f_score=0.099, mf=0.100, dcv=0.085, runtime=8.0)

            with patch("scripts.run_selected_fim_stack.load_dataset", return_value=_fake_dataset()), patch(
                "scripts.run_selected_fim_stack.run_fim_permutation_benchmark_from_config",
                side_effect=fake_runner,
            ):
                result = run_selected_fim_stack(args)

            self.assertEqual(result["selected"].selected_stack, "community_aware_fair_greedy")
            self.assertFalse(result["validation_summary"]["accepted"])
            self.assertIn("MF ratio", str(result["validation_summary"]["final_reason"]))

    def test_final_degradation_warning_is_reported(self) -> None:
        with _WorkspaceScratchDir() as temp_dir:
            benchmark_path = _benchmark_csv(temp_dir / "aggregate.csv")
            args = parse_args(
                [
                    "--dataset",
                    "graph_spa_500_0",
                    "--protected-attribute",
                    "region",
                    "--budget",
                    "40",
                    "--auto-select-stack-from",
                    str(benchmark_path),
                    "--output-dir",
                    str(temp_dir / "out"),
                ]
            )

            with patch("scripts.run_selected_fim_stack.load_dataset", return_value=_fake_dataset()), patch(
                "scripts.run_selected_fim_stack.run_fim_permutation_benchmark_from_config",
                return_value=_fake_result_row(
                    "graphsage_fair_ris_hybrid",
                    f_score=0.010,
                    mf=0.2,
                    dcv=0.200,
                    runtime=8.0,
                ),
            ):
                result = run_selected_fim_stack(args)

            self.assertTrue(result["validation_summary"]["final_validation_warning"])
            self.assertIn("final_validation_warning=True", result["report_text"])

    def test_validation_falls_back_when_candidate_runs_all_fail(self) -> None:
        with _WorkspaceScratchDir() as temp_dir:
            benchmark_path = _benchmark_csv(temp_dir / "aggregate.csv")
            args = parse_args(
                [
                    "--dataset",
                    "graph_spa_500_0",
                    "--protected-attribute",
                    "region",
                    "--budget",
                    "40",
                    "--auto-select-stack-from",
                    str(benchmark_path),
                    "--validate-selected-stack",
                    "--validation-seeds",
                    "7",
                    "--output-dir",
                    str(temp_dir / "out"),
                ]
            )

            def fake_runner(*, dataset_config, config, permutations):
                stack_name = _spec_name(permutations)
                if "community_aware_fair_greedy" in stack_name:
                    return _fake_result_row(stack_name, f_score=0.100, dcv=0.080, runtime=50.0)
                raise RuntimeError("candidate stack failed")

            with patch("scripts.run_selected_fim_stack.load_dataset", return_value=_fake_dataset()), patch(
                "scripts.run_selected_fim_stack.run_fim_permutation_benchmark_from_config",
                side_effect=fake_runner,
            ):
                result = run_selected_fim_stack(args)

            self.assertEqual(result["selected"].selected_stack, "community_aware_fair_greedy")
            self.assertFalse(result["validation_summary"]["accepted"])
            self.assertIn("all candidate validation runs failed", str(result["validation_summary"]["final_reason"]))

    def test_budget_larger_than_graph_fails_before_running(self) -> None:
        with _WorkspaceScratchDir() as temp_dir:
            benchmark_path = _benchmark_csv(temp_dir / "aggregate.csv")
            args = parse_args(
                [
                    "--dataset",
                    "graph_spa_500_0",
                    "--protected-attribute",
                    "region",
                    "--budget",
                    "40",
                    "--auto-select-stack-from",
                    str(benchmark_path),
                    "--output-dir",
                    str(temp_dir / "out"),
                ]
            )
            with patch("scripts.run_selected_fim_stack.load_dataset", return_value=_fake_dataset(node_count=10)):
                with self.assertRaisesRegex(ValueError, "exceeds graph node count"):
                    run_selected_fim_stack(args)


if __name__ == "__main__":
    unittest.main()
