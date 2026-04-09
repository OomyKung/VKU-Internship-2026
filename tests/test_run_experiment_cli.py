"""Tests for the experiment CLI report formatting."""

from __future__ import annotations

import unittest

import pandas as pd

from fim_hybrid.experiment_runner import ExperimentSettings
from scripts.run_experiment import format_results_report


class RunExperimentCliFormattingTestCase(unittest.TestCase):
    """Check that the CLI renders a readable comparison summary."""

    def test_format_results_report_includes_ranked_and_delta_sections(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "dataset": "toy_graph",
                    "community_method": "leiden",
                    "method": "hybrid_siea_ml_two_tier_tuned",
                    "variant_type": "ml_guided",
                    "total_spread": 4.90,
                    "mf": 0.008833,
                    "dcv": 0.032880,
                    "f_score": -0.012023,
                    "runtime_seconds": 6.376776,
                    "candidate_pool_size": 500,
                    "node2vec_enabled": False,
                    "node2vec_mode": "off",
                    "ml_guidance_mode": "two_tier",
                    "ml_validation_spearman": 0.302647,
                    "ml_validation_precision_at_budget": 0.25,
                    "community_modularity": 0.451674,
                },
                {
                    "dataset": "toy_graph",
                    "community_method": "leiden",
                    "method": "hybrid_siea",
                    "variant_type": "proposed",
                    "total_spread": 4.65,
                    "mf": 0.008333,
                    "dcv": 0.038232,
                    "f_score": -0.014949,
                    "runtime_seconds": 3.157337,
                    "candidate_pool_size": 500,
                    "node2vec_enabled": False,
                    "node2vec_mode": "off",
                    "ml_guidance_mode": "off",
                    "ml_validation_spearman": float("nan"),
                    "ml_validation_precision_at_budget": float("nan"),
                    "community_modularity": 0.451674,
                },
                {
                    "dataset": "toy_graph",
                    "community_method": "leiden",
                    "method": "hybrid_siea_ml_hard_filter",
                    "variant_type": "ml_guided",
                    "total_spread": 4.60,
                    "mf": 0.008167,
                    "dcv": 0.037440,
                    "f_score": -0.014636,
                    "runtime_seconds": 3.796700,
                    "candidate_pool_size": 125,
                    "node2vec_enabled": False,
                    "node2vec_mode": "off",
                    "ml_guidance_mode": "hard_filter",
                    "ml_validation_spearman": 0.302647,
                    "ml_validation_precision_at_budget": 0.25,
                    "community_modularity": 0.451674,
                },
            ]
        )
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=4,
            community_method="leiden",
            mc_runs=20,
            random_seed=42,
            use_ml=True,
            ml_guidance_mode="off",
            ml_singleton_runs=15,
        )

        report = format_results_report(frame, settings)

        self.assertIn("Fair Influence Maximization Experiment Summary", report)
        self.assertIn("Community: leiden", report)
        self.assertIn("Highlights", report)
        self.assertIn("Delta vs hybrid_siea", report)
        self.assertIn("hybrid_siea_ml_two_tier_tuned", report)
        self.assertIn("+0.002926", report)
        self.assertIn("0.302647", report)


if __name__ == "__main__":
    unittest.main()
