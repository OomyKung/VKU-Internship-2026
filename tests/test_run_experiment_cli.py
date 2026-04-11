"""Tests for the experiment CLI report formatting."""

from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from fim_hybrid.experiment_runner import ExperimentSettings
from scripts.run_experiment import format_results_report


KEPT_ML_LABEL = "hybrid_siea_ml_two_tier_tuned_swap_local_search"


class RunExperimentCliFormattingTestCase(unittest.TestCase):
    """Check that the CLI renders the cleaned comparison summary."""

    def test_format_results_report_includes_ranked_and_delta_sections(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "dataset": "toy_graph",
                    "diffusion_model": "ic",
                    "community_method": "leiden",
                    "method": KEPT_ML_LABEL,
                    "variant_type": "ml_guided",
                    "total_spread": 4.95,
                    "mf": 0.009000,
                    "dcv": 0.030303,
                    "f_score": -0.010652,
                    "runtime_seconds": 6.835000,
                    "search_runtime_seconds": 5.835000,
                    "final_eval_runtime_seconds": 1.000000,
                    "mc_runs_search": 20,
                    "mc_runs_eval": 1000,
                    "candidate_pool_size": 500,
                    "optimization_mode": "full",
                    "node2vec_enabled": False,
                    "node2vec_mode": "off",
                    "ml_guidance_mode": "two_tier",
                    "ml_validation_spearman": 0.302647,
                    "ml_validation_precision_at_budget": 0.25,
                    "community_modularity": 0.451674,
                    "zero_covered_groups_count": 0,
                    "bottom_3_avg_group_spread": 1.833333,
                    "fraction_groups_covered": 1.0,
                    "weakest_groups_note": "A,B,C",
                    "delta_f_score": 0.004297,
                },
                {
                    "dataset": "toy_graph",
                    "diffusion_model": "ic",
                    "community_method": "leiden",
                    "method": "hybrid_siea",
                    "variant_type": "proposed",
                    "total_spread": 4.65,
                    "mf": 0.008333,
                    "dcv": 0.038232,
                    "f_score": -0.014949,
                    "runtime_seconds": 3.157337,
                    "search_runtime_seconds": 2.657337,
                    "final_eval_runtime_seconds": 0.500000,
                    "mc_runs_search": 20,
                    "mc_runs_eval": 1000,
                    "candidate_pool_size": 500,
                    "optimization_mode": "full",
                    "node2vec_enabled": False,
                    "node2vec_mode": "off",
                    "ml_guidance_mode": "off",
                    "ml_validation_spearman": float("nan"),
                    "ml_validation_precision_at_budget": float("nan"),
                    "community_modularity": 0.451674,
                    "zero_covered_groups_count": 0,
                    "bottom_3_avg_group_spread": 1.700000,
                    "fraction_groups_covered": 1.0,
                    "weakest_groups_note": "A,B,C",
                    "delta_f_score": 0.0,
                },
                {
                    "dataset": "toy_graph",
                    "diffusion_model": "ic",
                    "community_method": "leiden",
                    "method": "cea_fim",
                    "variant_type": "comparator",
                    "total_spread": 4.70,
                    "mf": 0.008500,
                    "dcv": 0.034000,
                    "f_score": -0.012750,
                    "runtime_seconds": 4.000000,
                    "search_runtime_seconds": 3.250000,
                    "final_eval_runtime_seconds": 0.750000,
                    "mc_runs_search": 20,
                    "mc_runs_eval": 1000,
                    "candidate_pool_size": 500,
                    "optimization_mode": "full",
                    "node2vec_enabled": False,
                    "node2vec_mode": "off",
                    "ml_guidance_mode": "off",
                    "ml_validation_spearman": float("nan"),
                    "ml_validation_precision_at_budget": float("nan"),
                    "community_modularity": 0.451674,
                    "zero_covered_groups_count": 0,
                    "bottom_3_avg_group_spread": 1.750000,
                    "fraction_groups_covered": 1.0,
                    "weakest_groups_note": "A,B,C",
                    "delta_f_score": 0.002199,
                },
            ]
        )
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=4,
            community_method="leiden",
            mc_runs_search=20,
            mc_runs_eval=1000,
            random_seed=42,
            use_ml=True,
            ml_guidance_mode="two_tier",
            ml_singleton_runs=15,
        )

        report = format_results_report(frame, settings)

        self.assertIn("Fair Influence Maximization Experiment Summary", report)
        self.assertIn("Diffusion model: ic (Independent Cascade)", report)
        self.assertIn("MC runs: search=20, eval=1000", report)
        self.assertIn("Final evaluation seed: random_seed + 1000000", report)
        self.assertIn("Node2Vec: removed from the supported ML experiment surface", report)
        self.assertIn("Community: leiden", report)
        self.assertIn("Highlights", report)
        self.assertIn("Delta vs hybrid_siea", report)
        self.assertIn(KEPT_ML_LABEL, report)
        self.assertIn("two_tier", report)
        self.assertIn("0.302647", report)
        self.assertIn("search=5.835s", report)
        self.assertIn("eval=1.000s", report)

    def test_format_results_report_includes_protected_attribute_output_path(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "dataset": "toy_graph",
                    "diffusion_model": "ic",
                    "community_method": "leiden",
                    "method": "hybrid_siea",
                    "variant_type": "proposed",
                    "total_spread": 4.65,
                    "mf": 0.008333,
                    "dcv": 0.038232,
                    "f_score": -0.014949,
                    "runtime_seconds": 3.157337,
                    "search_runtime_seconds": 2.657337,
                    "final_eval_runtime_seconds": 0.500000,
                    "mc_runs_search": 20,
                    "mc_runs_eval": 20,
                    "candidate_pool_size": 500,
                    "optimization_mode": "full",
                    "node2vec_enabled": False,
                    "node2vec_mode": "off",
                    "ml_guidance_mode": "off",
                    "ml_validation_spearman": float("nan"),
                    "ml_validation_precision_at_budget": float("nan"),
                    "community_modularity": 0.451674,
                    "zero_covered_groups_count": 0,
                    "bottom_3_avg_group_spread": 1.700000,
                    "fraction_groups_covered": 1.0,
                    "weakest_groups_note": "A,B,C",
                    "delta_f_score": 0.0,
                }
            ]
        )
        settings = ExperimentSettings(
            protected_attribute="group/name",
            budget=4,
            community_method="leiden",
            mc_runs_search=20,
            mc_runs_eval=20,
            random_seed=42,
            output_dir=Path("results"),
        )

        report = format_results_report(frame, settings)

        self.assertIn(str(Path("results") / "toy_graph" / "group_name" / "toy_graph_budget4_results.csv"), report)

    def test_format_results_report_can_focus_on_best_ml_vs_cea_fim(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "dataset": "toy_graph",
                    "diffusion_model": "ic",
                    "community_method": "leiden",
                    "method": "cea_fim",
                    "variant_type": "comparator",
                    "total_spread": 4.70,
                    "mf": 0.008500,
                    "dcv": 0.034000,
                    "f_score": -0.012750,
                    "runtime_seconds": 4.000000,
                    "search_runtime_seconds": 3.250000,
                    "final_eval_runtime_seconds": 0.750000,
                    "mc_runs_search": 20,
                    "mc_runs_eval": 1000,
                    "candidate_pool_size": 500,
                    "optimization_mode": "full",
                    "node2vec_enabled": False,
                    "node2vec_mode": "off",
                    "ml_guidance_mode": "off",
                    "ml_validation_spearman": float("nan"),
                    "ml_validation_precision_at_budget": float("nan"),
                    "community_modularity": 0.451674,
                    "zero_covered_groups_count": 0,
                    "bottom_3_avg_group_spread": 1.750000,
                    "fraction_groups_covered": 1.0,
                    "weakest_groups_note": "A,B,C",
                    "delta_f_score": 0.002199,
                },
                {
                    "dataset": "toy_graph",
                    "diffusion_model": "ic",
                    "community_method": "leiden",
                    "method": KEPT_ML_LABEL,
                    "variant_type": "ml_guided",
                    "total_spread": 4.95,
                    "mf": 0.009000,
                    "dcv": 0.030303,
                    "f_score": -0.010652,
                    "runtime_seconds": 6.835000,
                    "search_runtime_seconds": 5.835000,
                    "final_eval_runtime_seconds": 1.000000,
                    "mc_runs_search": 20,
                    "mc_runs_eval": 1000,
                    "candidate_pool_size": 500,
                    "optimization_mode": "full",
                    "node2vec_enabled": False,
                    "node2vec_mode": "off",
                    "ml_guidance_mode": "two_tier",
                    "ml_validation_spearman": 0.302647,
                    "ml_validation_precision_at_budget": 0.25,
                    "community_modularity": 0.451674,
                    "zero_covered_groups_count": 0,
                    "bottom_3_avg_group_spread": 1.833333,
                    "fraction_groups_covered": 1.0,
                    "weakest_groups_note": "A,B,C",
                    "delta_f_score": 0.004297,
                },
                {
                    "dataset": "toy_graph",
                    "diffusion_model": "ic",
                    "community_method": "leiden",
                    "method": "hybrid_siea",
                    "variant_type": "proposed",
                    "total_spread": 4.65,
                    "mf": 0.008333,
                    "dcv": 0.038232,
                    "f_score": -0.014949,
                    "runtime_seconds": 3.157337,
                    "search_runtime_seconds": 2.657337,
                    "final_eval_runtime_seconds": 0.500000,
                    "mc_runs_search": 20,
                    "mc_runs_eval": 1000,
                    "candidate_pool_size": 500,
                    "optimization_mode": "full",
                    "node2vec_enabled": False,
                    "node2vec_mode": "off",
                    "ml_guidance_mode": "off",
                    "ml_validation_spearman": float("nan"),
                    "ml_validation_precision_at_budget": float("nan"),
                    "community_modularity": 0.451674,
                    "zero_covered_groups_count": 0,
                    "bottom_3_avg_group_spread": 1.700000,
                    "fraction_groups_covered": 1.0,
                    "weakest_groups_note": "A,B,C",
                    "delta_f_score": 0.0,
                },
            ]
        )
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=4,
            community_method="leiden",
            mc_runs_search=20,
            mc_runs_eval=1000,
            random_seed=42,
            use_ml=True,
            ml_guidance_mode="two_tier",
            ml_singleton_runs=15,
        )

        report = format_results_report(frame, settings, report_focus="best_ml_vs_cea_fim")

        self.assertIn("Report focus: best ML method vs CEA-FIM only", report)
        self.assertIn("Delta vs cea_fim", report)
        self.assertIn(KEPT_ML_LABEL, report)
        self.assertIn("cea_fim", report)
        self.assertNotIn("hybrid_siea [full]", report)


if __name__ == "__main__":
    unittest.main()
