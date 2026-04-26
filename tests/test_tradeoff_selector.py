"""Tests for benchmark-driven stack trade-off selection."""

from __future__ import annotations

from pathlib import Path
import shutil
import unittest
from uuid import uuid4

import pandas as pd

from fim_hybrid.selection.tradeoff_selector import (
    TradeoffSelectionConfig,
    load_selected_stack_config,
    normalize_benchmark_frame,
    select_fairness_fallback,
    select_stack_from_benchmark_csv,
    select_stack_from_benchmark_frame,
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


def _aggregate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stack_name": "community_aware_fair_greedy",
                "protected_attribute": "region",
                "budget": 40,
                "status": "ok",
                "mean_f_score": 0.100,
                "mean_mf": 0.20,
                "mean_dcv": 0.08,
                "mean_spread": 42.0,
                "mean_runtime": 50.0,
            },
            {
                "stack_name": "graphsage_fair_ris_hybrid",
                "protected_attribute": "region",
                "budget": 40,
                "status": "ok",
                "mean_f_score": 0.098,
                "mean_mf": 0.19,
                "mean_dcv": 0.09,
                "mean_spread": 42.2,
                "mean_runtime": 12.0,
            },
            {
                "stack_name": "line_fast_ml",
                "protected_attribute": "region",
                "budget": 40,
                "status": "ok",
                "mean_f_score": -0.20,
                "mean_mf": 0.0,
                "mean_dcv": 0.60,
                "mean_spread": 43.0,
                "mean_runtime": 3.0,
            },
        ]
    )


class TradeoffSelectorTestCase(unittest.TestCase):
    def test_normalizes_raw_and_aggregate_aliases(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "permutation_name": "graphsage_fair_ris_hybrid",
                    "protected_attribute": "region",
                    "budget": 40,
                    "status": "ok",
                    "F-score": 0.1,
                    "MF": 0.2,
                    "DCV": 0.1,
                    "spread": 42.0,
                    "runtime": 5.0,
                }
            ]
        )

        normalized = normalize_benchmark_frame(raw)

        self.assertEqual(str(normalized.iloc[0]["stack_name"]), "graphsage_fair_ris_hybrid")
        self.assertAlmostEqual(float(normalized.iloc[0]["f_score"]), 0.1)
        self.assertAlmostEqual(float(normalized.iloc[0]["runtime_seconds"]), 5.0)

    def test_status_column_is_optional_and_defaults_to_ok(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "stack_name": "graphsage_fair_ris_hybrid",
                    "protected_attribute": "region",
                    "budget": 40,
                    "F-score": 0.1,
                    "MF": 0.2,
                    "DCV": 0.1,
                    "runtime": 5.0,
                }
            ]
        )

        normalized = normalize_benchmark_frame(raw)

        self.assertEqual(str(normalized.iloc[0]["status"]), "ok")

    def test_selects_runtime_winner_within_close_fscore_threshold_and_rejects_collapse(self) -> None:
        result = select_stack_from_benchmark_frame(
            _aggregate_rows(),
            protected_attribute="region",
            budget=40,
            config=TradeoffSelectionConfig(close_fscore_threshold=0.005),
        )

        self.assertEqual(result.selected_config.selected_stack, "graphsage_fair_ris_hybrid")
        self.assertIn("lowest runtime", result.selected_config.reason)
        self.assertEqual(len(result.rejected_frame), 1)
        self.assertEqual(result.selected_config.pipeline.spread_estimator_final, "monte_carlo")

    def test_strict_fairness_closeness_blocks_fast_dcv_regression(self) -> None:
        rows = _aggregate_rows()
        rows.loc[1, "mean_dcv"] = 0.12

        result = select_stack_from_benchmark_frame(
            rows,
            protected_attribute="region",
            budget=40,
            config=TradeoffSelectionConfig(close_fscore_threshold=0.005, max_dcv_delta_vs_best=0.01),
        )

        self.assertEqual(result.selected_config.selected_stack, "community_aware_fair_greedy")
        self.assertEqual(result.fallback_config.selected_stack, "graphsage_fair_ris_hybrid")
        self.assertLessEqual(float(result.selected_config.benchmark_row["dcv"]), 0.09)

    def test_quality_runtime_policy_is_explicit(self) -> None:
        result = select_stack_from_benchmark_frame(
            _aggregate_rows(),
            protected_attribute="region",
            budget=40,
            config=TradeoffSelectionConfig(selection_policy="quality_runtime", quality_runtime_lambda=1.5),
        )

        self.assertEqual(result.selected_config.selected_stack, "graphsage_fair_ris_hybrid")
        self.assertIn("quality_runtime", result.selected_config.reason)
        self.assertIn("quality_runtime_score", result.selected_config.benchmark_row)

    def test_missing_columns_and_missing_files_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required metric columns"):
            normalize_benchmark_frame(pd.DataFrame([{"stack_name": "x"}]))

        with self.assertRaises(FileNotFoundError):
            select_stack_from_benchmark_csv(REPO_ROOT / "does_not_exist.csv", protected_attribute="region", budget=40)

    def test_selected_config_exports_and_loads(self) -> None:
        result = select_stack_from_benchmark_frame(
            _aggregate_rows(),
            protected_attribute="region",
            budget=40,
        )

        with _WorkspaceScratchDir() as temp_dir:
            path = result.selected_config.to_json(temp_dir / "selected.json")
            loaded = load_selected_stack_config(path)

        self.assertEqual(loaded.selected_stack, result.selected_config.selected_stack)
        self.assertEqual(loaded.pipeline.embedding_method, "graphsage")
        self.assertEqual(loaded.budget, 40)

    def test_fairness_fallback_prefers_independent_best_stack(self) -> None:
        result = select_stack_from_benchmark_frame(
            _aggregate_rows(),
            protected_attribute="region",
            budget=40,
            config=TradeoffSelectionConfig(close_fscore_threshold=0.005),
        )

        self.assertEqual(result.selected_config.selected_stack, "graphsage_fair_ris_hybrid")
        self.assertEqual(result.fallback_config.selected_stack, "community_aware_fair_greedy")

    def test_forced_fallback_stack_must_be_valid_and_independent_when_possible(self) -> None:
        fallback = select_fairness_fallback(
            _aggregate_rows(),
            "graphsage_fair_ris_hybrid",
            TradeoffSelectionConfig(fallback_stack="community_aware_fair_greedy"),
            protected_attribute="region",
            budget=40,
        )

        self.assertEqual(fallback.selected_stack, "community_aware_fair_greedy")

        with self.assertRaisesRegex(ValueError, "equals candidate_stack"):
            select_fairness_fallback(
                _aggregate_rows(),
                "graphsage_fair_ris_hybrid",
                TradeoffSelectionConfig(fallback_stack="graphsage_fair_ris_hybrid"),
                protected_attribute="region",
                budget=40,
            )


if __name__ == "__main__":
    unittest.main()
