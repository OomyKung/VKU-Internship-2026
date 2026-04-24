"""Tests for the significance-test CLI helpers."""

from __future__ import annotations

from pathlib import Path
import shutil
import unittest
from uuid import uuid4

import pandas as pd

from scripts.run_significance_test import (
    build_significance_report,
    load_significance_input,
    run_significance_analysis,
    save_significance_report,
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


def _toy_comparison_frame(seed: int, challenger_shift: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stack_name": "baseline_stack",
                "status": "ok",
                "dataset": "toy_graph",
                "protected_attribute": "group",
                "f_score": 0.10,
                "total_spread": 10.0,
                "extra_spread": 2.0,
                "mf": 0.50,
                "dcv": 0.40,
                "runtime_seconds": 5.0,
            },
            {
                "stack_name": "challenger_stack",
                "status": "ok",
                "dataset": "toy_graph",
                "protected_attribute": "group",
                "f_score": 0.10 + challenger_shift,
                "total_spread": 10.0 + challenger_shift,
                "extra_spread": 2.0 + challenger_shift,
                "mf": 0.50 + challenger_shift,
                "dcv": 0.40 - challenger_shift,
                "runtime_seconds": 5.0 - challenger_shift,
            },
        ]
    ).assign(seed=seed)


class SignificanceTestScriptTestCase(unittest.TestCase):
    def test_load_significance_input_infers_seed_from_directory_names(self) -> None:
        with _WorkspaceScratchDir() as temp_dir:
            for seed in (1, 2):
                csv_dir = temp_dir / f"seed_{seed}" / "toy_graph" / "group"
                csv_dir.mkdir(parents=True, exist_ok=True)
                frame = _toy_comparison_frame(seed=seed, challenger_shift=0.1)
                frame.drop(columns=["seed"]).to_csv(
                    csv_dir / "toy_graph_budget4_permutation_comparison.csv",
                    index=False,
                )

            combined = load_significance_input(temp_dir)

        self.assertEqual(set(combined["seed"].astype(str)), {"1", "2"})
        self.assertIn("stack_name", combined.columns)

    def test_run_significance_analysis_computes_paired_rows(self) -> None:
        frame = pd.concat(
            [
                _toy_comparison_frame(seed=seed, challenger_shift=0.20)
                for seed in range(1, 11)
            ],
            ignore_index=True,
        )

        result = run_significance_analysis(
            frame,
            baseline="baseline_stack",
            metrics=["f_score", "dcv"],
            lower_is_better_metrics=["dcv"],
            min_pairs=5,
            p_adjustment="holm",
            bootstrap_reps=200,
            random_seed=7,
        )

        self.assertEqual(set(result["metric"]), {"f_score", "dcv"})
        self.assertTrue((result["status"] == "ok").all())
        self.assertTrue((result["pair_count"] == 10).all())
        self.assertTrue((pd.to_numeric(result["mean_delta"]) > 0.0).all())
        self.assertTrue((pd.to_numeric(result["adjusted_p"]) <= 0.05).all())
        self.assertTrue(result["significant"].all())

    def test_save_significance_report_writes_csv_and_text(self) -> None:
        frame = pd.concat(
            [
                _toy_comparison_frame(seed=seed, challenger_shift=0.15)
                for seed in range(1, 8)
            ],
            ignore_index=True,
        )
        result = run_significance_analysis(
            frame,
            baseline="baseline_stack",
            metrics=["f_score"],
            min_pairs=5,
            bootstrap_reps=100,
            random_seed=7,
        )

        with _WorkspaceScratchDir() as temp_dir:
            csv_path, txt_path = save_significance_report(
                result,
                output_dir=temp_dir,
                baseline="baseline_stack",
                p_adjustment="holm",
                alpha=0.05,
            )
            report_text = txt_path.read_text(encoding="utf-8")
            self.assertTrue(csv_path.exists())
            self.assertTrue(txt_path.exists())

        self.assertIn("FIM Significance Test Summary", report_text)
        self.assertIn("challenger_stack", report_text)
        self.assertIn("Wilcoxon", build_significance_report(result, baseline="baseline_stack", p_adjustment="holm", alpha=0.05))


if __name__ == "__main__":
    unittest.main()
