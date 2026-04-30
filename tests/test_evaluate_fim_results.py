"""Tests for the FIM result evaluation CLI helpers."""

from __future__ import annotations

from pathlib import Path
import shutil
import unittest
from uuid import uuid4

import pandas as pd

from scripts.evaluate_fim_results import (
    InsightThresholds,
    build_evaluation_report,
    evaluate_result_frame,
    filter_result_frame,
    load_evaluation_inputs,
    normalize_result_frame,
    save_evaluation_outputs,
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


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _toy_result_rows(seed: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stack_name": "community_aware_fair_greedy",
                "dataset": "toy_graph",
                "protected_attribute": "region",
                "budget": 4,
                "random_seed": seed,
                "status": "ok",
                "variant_type": "interpretable_baseline",
                "embedding_method": "none",
                "ranking_model": "fairness_weighted_greedy",
                "total_spread": 10.0,
                "extra_spread": 2.0,
                "mf": 0.50,
                "dcv": 0.10,
                "f_score": 0.1200,
                "runtime_seconds": 8.0,
            },
            {
                "stack_name": "graphsage_fair_ris_hybrid",
                "dataset": "toy_graph",
                "protected_attribute": "region",
                "budget": 4,
                "random_seed": seed,
                "status": "ok",
                "variant_type": "ml_guided",
                "embedding_method": "graphsage",
                "ranking_model": "graphsage",
                "total_spread": 10.8,
                "extra_spread": 2.8,
                "mf": 0.49,
                "dcv": 0.12,
                "f_score": 0.1188,
                "runtime_seconds": 4.0,
            },
            {
                "stack_name": "infomap_graphcl_maximin",
                "dataset": "toy_graph",
                "protected_attribute": "region",
                "budget": 4,
                "random_seed": seed,
                "status": "ok",
                "variant_type": "exploratory_fairness",
                "embedding_method": "graphcl",
                "clustering_method": "spectral",
                "ranking_model": "maximin_greedy",
                "community_method": "infomap",
                "total_spread": 10.5,
                "extra_spread": 2.5,
                "mf": 0.0,
                "dcv": 0.55,
                "f_score": -0.20,
                "runtime_seconds": 12.0,
            },
        ]
    )


class EvaluateFimResultsScriptTestCase(unittest.TestCase):
    def test_load_evaluation_inputs_normalizes_legacy_permutation_csv(self) -> None:
        legacy_frame = pd.DataFrame(
            [
                {
                    "permutation_name": "legacy_stack",
                    "dataset": "toy_graph",
                    "protected_attribute": "region",
                    "status": "ok",
                    "ranking_mode": "graphsage_plus_fair_ris",
                    "bias_control": "worst_group_boost",
                    "spread": 12.0,
                    "extra": 2.0,
                    "MF": 0.4,
                    "DCV": 0.2,
                    "F-score": 0.3,
                    "runtime": 5.0,
                    "skipped_reason": "",
                }
            ]
        )

        with _WorkspaceScratchDir() as temp_dir:
            csv_path = _write_csv(
                temp_dir / "seed_7" / "toy_graph" / "region" / "toy_graph_budget4_permutation_comparison.csv",
                legacy_frame,
            )
            loaded = load_evaluation_inputs([temp_dir], glob_patterns=["*_permutation_comparison.csv"])

        row = loaded.iloc[0]
        self.assertEqual(str(row["stack_name"]), "legacy_stack")
        self.assertEqual(str(row["ranking_model"]), "graphsage_plus_fair_ris")
        self.assertEqual(str(row["debias_mode"]), "worst_group_boost")
        self.assertEqual(float(row["total_spread"]), 12.0)
        self.assertEqual(float(row["extra_spread"]), 2.0)
        self.assertEqual(float(row["runtime_seconds"]), 5.0)
        self.assertEqual(int(row["budget"]), 4)
        self.assertEqual(int(row["random_seed"]), 7)
        self.assertEqual(Path(str(row["source_path"])).name, csv_path.name)

    def test_normalize_result_frame_handles_experiment_runner_results(self) -> None:
        runner_frame = pd.DataFrame(
            [
                {
                    "method": "hybrid_siea",
                    "note": "history=file.csv",
                    "total_spread": 42.0,
                    "extra_spread": 2.0,
                    "mf": 0.07,
                    "dcv": 0.02,
                    "f_score": 0.03,
                    "runtime_seconds": 11.0,
                }
            ]
        )
        source_path = REPO_ROOT / "results" / "graph_spa_500_0" / "age" / "graph_spa_500_0_budget40_results.csv"
        normalized = normalize_result_frame(runner_frame, source_path=source_path)

        row = normalized.iloc[0]
        self.assertEqual(str(row["stack_name"]), "hybrid_siea")
        self.assertEqual(str(row["notes"]), "history=file.csv")
        self.assertEqual(str(row["dataset"]), "graph_spa_500_0")
        self.assertEqual(str(row["protected_attribute"]), "age")
        self.assertEqual(int(row["budget"]), 40)

    def test_load_evaluation_inputs_parses_current_permutation_report_text(self) -> None:
        report_text = """FIM Algorithm-Stack Permutation Comparison
------------------------------------------------------------------------
Protected attribute: region
Budget: 4
Final estimator: monte_carlo | mc_runs_eval=1000

1. community_aware_fair_greedy [ok]
   spread=10.1000 | extra=0.1000 | MF=0.5000 | DCV=0.1000 | F-score=0.1200 | runtime=8.000s
   modules: diffusion=ic | community=leiden | embedding=none | clustering=none | optimizer=local_search | ranking=fairness_weighted_greedy
   notes=baseline path

2. infomap_graphcl_maximin [ok]
   spread=10.5000 | extra=0.5000 | MF=0.0000 | DCV=0.5500 | F-score=-0.2000 | runtime=12.000s
   modules: diffusion=ic | community=infomap | embedding=graphcl | clustering=spectral | optimizer=local_search | ranking=maximin_greedy
   notes=exploratory path
"""

        with _WorkspaceScratchDir() as temp_dir:
            report_path = temp_dir / "toy_graph" / "region" / "toy_graph_budget4_permutation_report.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report_text, encoding="utf-8")
            loaded = load_evaluation_inputs([report_path])

        self.assertEqual(len(loaded), 2)
        self.assertEqual(str(loaded.iloc[0]["stack_name"]), "community_aware_fair_greedy")
        self.assertEqual(str(loaded.iloc[1]["clustering_method"]), "spectral")
        self.assertAlmostEqual(float(loaded.iloc[1]["f_score"]), -0.2, places=6)

    def test_evaluate_result_frame_aggregates_ranks_and_generates_insights(self) -> None:
        frame = pd.concat([_toy_result_rows(seed=1), _toy_result_rows(seed=2)], ignore_index=True)
        result = evaluate_result_frame(
            frame,
            rank_by="fim_default",
            group_by=["dataset", "protected_attribute", "budget"],
            thresholds=InsightThresholds(close_threshold=0.002, dcv_collapse_threshold=0.25, mf_collapse_threshold=0.001),
        )

        self.assertEqual(len(result.group_results), 1)
        group = result.group_results[0]
        self.assertEqual(group.recommendations["best_overall_method"], "community_aware_fair_greedy")
        self.assertEqual(group.recommendations["best_practical_choice"], "graphsage_fair_ris_hybrid")
        self.assertEqual(group.recommendations["best_interpretable_baseline"], "community_aware_fair_greedy")
        self.assertEqual(group.recommendations["best_exploratory_comparator"], "infomap_graphcl_maximin")
        self.assertEqual(int(group.ranked_frame.iloc[0]["run_count"]), 2)
        self.assertIn("close overall result", "\n".join(group.insight_lines))
        self.assertIn("fairness collapse", "\n".join(group.insight_lines).lower())
        self.assertIn("Best practical high-speed option", group.method_notes["graphsage_fair_ris_hybrid"])
        self.assertIn("Fairness collapse warning", group.method_notes["infomap_graphcl_maximin"])

    def test_professor_priority_does_not_select_faster_clear_fscore_loss(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "stack_name": "slow_fair",
                    "dataset": "toy_graph",
                    "protected_attribute": "region",
                    "budget": 4,
                    "status": "ok",
                    "total_spread": 10.0,
                    "extra_spread": 1.0,
                    "mf": 0.20,
                    "dcv": 0.05,
                    "f_score": 0.200,
                    "runtime_seconds": 100.0,
                    "scalability_pass": True,
                },
                {
                    "stack_name": "fast_less_fair",
                    "dataset": "toy_graph",
                    "protected_attribute": "region",
                    "budget": 4,
                    "status": "ok",
                    "total_spread": 15.0,
                    "extra_spread": 6.0,
                    "mf": 0.25,
                    "dcv": 0.04,
                    "f_score": 0.190,
                    "runtime_seconds": 1.0,
                    "scalability_pass": True,
                },
            ]
        )

        result = evaluate_result_frame(
            frame,
            rank_by="professor_priority",
            group_by=["dataset", "protected_attribute", "budget"],
            thresholds=InsightThresholds(close_threshold=0.003),
        )

        group = result.group_results[0]
        self.assertEqual(group.recommendations["final_professor_priority_recommendation"], "slow_fair")
        self.assertEqual(str(group.ranked_frame.iloc[0]["stack_name"]), "slow_fair")

    def test_custom_rank_and_output_saving_work(self) -> None:
        frame = _toy_result_rows(seed=3)
        result = evaluate_result_frame(
            frame,
            rank_by="custom",
            custom_rank="runtime:asc,F-score:desc",
            group_by=["dataset", "protected_attribute", "budget"],
        )

        with _WorkspaceScratchDir() as temp_dir:
            saved = save_evaluation_outputs(
                result,
                output_dir=temp_dir,
                report_name="toy_eval",
                input_paths=["toy.csv"],
                rank_by="custom",
                save_json=True,
            )
            report_text = saved["report_txt"].read_text(encoding="utf-8")
            self.assertTrue(saved["normalized_csv"].exists())
            self.assertTrue(saved["ranked_csv"].exists())
            self.assertTrue(saved["report_txt"].exists())
            self.assertTrue(saved["summary_json"].exists())
            self.assertIn("Instant Insights", report_text)
            self.assertIn("Ranking mode: custom", report_text)

    def test_filter_result_frame_errors_for_unavailable_requested_column(self) -> None:
        frame = _toy_result_rows(seed=1).copy()
        frame["dataset"] = pd.NA
        with self.assertRaisesRegex(ValueError, "dataset filter"):
            filter_result_frame(frame, datasets=["toy_graph"])

    def test_build_evaluation_report_mentions_method_notes(self) -> None:
        result = evaluate_result_frame(
            _toy_result_rows(seed=4),
            rank_by="fim_default",
            group_by=["dataset", "protected_attribute", "budget"],
        )
        report_text = build_evaluation_report(
            result,
            input_paths=["toy_graph_budget4_permutation_comparison.csv"],
            rank_by="fim_default",
            report_name="toy_eval",
        )
        self.assertIn("Per-Method Notes", report_text)
        self.assertIn("Best overall fairness-adjusted result", report_text)


if __name__ == "__main__":
    unittest.main()
