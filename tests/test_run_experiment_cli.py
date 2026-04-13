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
                    "extra_spread": 0.95,
                    "mf": 0.009000,
                    "dcv": 0.030303,
                    "f_score": -0.010652,
                    "runtime_seconds": 6.835000,
                    "search_runtime_seconds": 5.835000,
                    "final_eval_runtime_seconds": 1.000000,
                    "mc_runs_search": 20,
                    "mc_runs_eval": 1000,
                    "mc_runs_search_used": 20,
                    "mc_runs_eval_used": 1000,
                    "candidate_pool_size": 500,
                    "optimization_mode": "full",
                    "node2vec_enabled": False,
                    "node2vec_mode": "off",
                    "ml_guidance_mode": "two_tier",
                    "ml_backend": "tabular",
                    "gnn_model_type": pd.NA,
                    "ml_validation_spearman": 0.302647,
                    "ml_validation_precision_at_budget": 0.25,
                    "community_modularity": 0.451674,
                    "zero_covered_groups_count": 0,
                    "bottom_3_avg_group_spread": 1.833333,
                    "fraction_groups_covered": 1.0,
                    "weakest_groups_note": "A,B,C",
                    "delta_f_score": 0.004297,
                    "final_recheck_applied": True,
                    "final_recheck_top_k_rank": 1,
                    "final_recheck_mc_runs_used": 1000,
                    "final_recheck_total_spread": 5.02,
                    "final_recheck_extra_spread": 1.02,
                    "final_recheck_mf": 0.009100,
                    "final_recheck_dcv": 0.029900,
                    "final_recheck_f_score": -0.010400,
                    "final_recheck_runtime_seconds": 1.750000,
                },
                {
                    "dataset": "toy_graph",
                    "diffusion_model": "ic",
                    "community_method": "leiden",
                    "method": "hybrid_siea",
                    "variant_type": "proposed",
                    "total_spread": 4.65,
                    "extra_spread": 0.65,
                    "mf": 0.008333,
                    "dcv": 0.038232,
                    "f_score": -0.014949,
                    "runtime_seconds": 3.157337,
                    "search_runtime_seconds": 2.657337,
                    "final_eval_runtime_seconds": 0.500000,
                    "mc_runs_search": 20,
                    "mc_runs_eval": 1000,
                    "mc_runs_search_used": 20,
                    "mc_runs_eval_used": 1000,
                    "candidate_pool_size": 500,
                    "optimization_mode": "full",
                    "node2vec_enabled": False,
                    "node2vec_mode": "off",
                    "ml_guidance_mode": "off",
                    "ml_backend": "none",
                    "gnn_model_type": pd.NA,
                    "ml_validation_spearman": float("nan"),
                    "ml_validation_precision_at_budget": float("nan"),
                    "community_modularity": 0.451674,
                    "zero_covered_groups_count": 0,
                    "bottom_3_avg_group_spread": 1.700000,
                    "fraction_groups_covered": 1.0,
                    "weakest_groups_note": "A,B,C",
                    "delta_f_score": 0.0,
                    "final_recheck_applied": False,
                },
                {
                    "dataset": "toy_graph",
                    "diffusion_model": "ic",
                    "community_method": "leiden",
                    "method": "cea_fim",
                    "variant_type": "comparator",
                    "total_spread": 4.70,
                    "extra_spread": 0.70,
                    "mf": 0.008500,
                    "dcv": 0.034000,
                    "f_score": -0.012750,
                    "runtime_seconds": 4.000000,
                    "search_runtime_seconds": 3.250000,
                    "final_eval_runtime_seconds": 0.750000,
                    "mc_runs_search": 20,
                    "mc_runs_eval": 1000,
                    "mc_runs_search_used": 20,
                    "mc_runs_eval_used": 1000,
                    "candidate_pool_size": 500,
                    "optimization_mode": "full",
                    "node2vec_enabled": False,
                    "node2vec_mode": "off",
                    "ml_guidance_mode": "off",
                    "ml_backend": "none",
                    "gnn_model_type": pd.NA,
                    "ml_validation_spearman": float("nan"),
                    "ml_validation_precision_at_budget": float("nan"),
                    "community_modularity": 0.451674,
                    "zero_covered_groups_count": 0,
                    "bottom_3_avg_group_spread": 1.750000,
                    "fraction_groups_covered": 1.0,
                    "weakest_groups_note": "A,B,C",
                    "delta_f_score": 0.002199,
                    "final_recheck_applied": False,
                },
            ]
        )
        settings = ExperimentSettings(
            protected_attribute="group",
            budget=4,
            community_method="leiden",
            mc_runs_search=20,
            mc_runs_eval=1000,
            enable_final_recheck=True,
            final_recheck_mc_runs=1000,
            final_recheck_top_k=1,
            random_seed=42,
            use_ml=True,
            ml_backend="tabular",
            ml_guidance_mode="two_tier",
            ml_singleton_runs=15,
        )

        report = format_results_report(frame, settings)

        self.assertIn("Fair Influence Maximization Experiment Summary", report)
        self.assertIn("Diffusion model: ic (Independent Cascade)", report)
        self.assertIn("MC runs: search=20, eval=1000", report)
        self.assertIn("Final evaluation seed: random_seed + 1000000", report)
        self.assertIn("Spread semantics: total activated nodes including seed nodes; extra_spread = total_spread - budget", report)
        self.assertIn("Final recheck: enabled | mc_runs=1000 | top_k=1", report)
        self.assertIn("Final recheck seed: random_seed + 2000000", report)
        self.assertIn("Node2Vec: supported only as an optional GNN input feature source", report)
        self.assertIn("Community: leiden", report)
        self.assertIn("Highlights", report)
        self.assertIn("Delta vs hybrid_siea", report)
        self.assertIn(KEPT_ML_LABEL, report)
        self.assertIn("two_tier", report)
        self.assertIn("backend=tabular", report)
        self.assertIn("GNN=-", report)
        self.assertIn("0.302647", report)
        self.assertIn("search=5.835s", report)
        self.assertIn("eval=1.000s", report)
        self.assertIn("extra=0.950", report)
        self.assertIn("recheck@1000MC [rank=1]", report)

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
                    "ml_backend": "none",
                    "gnn_model_type": pd.NA,
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

    def test_format_results_report_shows_gnn_backend_metadata(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "dataset": "toy_graph",
                    "diffusion_model": "ic",
                    "community_method": "leiden",
                    "method": "hybrid_siea_ml_gnn_two_tier_tuned_swap_local_search",
                    "variant_type": "ml_guided",
                    "total_spread": 4.80,
                    "extra_spread": 0.80,
                    "mf": 0.008800,
                    "dcv": 0.031000,
                    "f_score": -0.011100,
                    "runtime_seconds": 7.000000,
                    "search_runtime_seconds": 6.000000,
                    "final_eval_runtime_seconds": 1.000000,
                    "mc_runs_search": 20,
                    "mc_runs_eval": 1000,
                    "candidate_pool_size": 500,
                    "optimization_mode": "full",
                    "node2vec_enabled": False,
                    "node2vec_mode": "off",
                    "ml_guidance_mode": "two_tier",
                    "ml_backend": "gnn",
                    "gnn_model_type": "graphsage",
                    "ml_validation_spearman": 0.280000,
                    "ml_validation_precision_at_budget": 0.20,
                    "community_modularity": 0.451674,
                    "zero_covered_groups_count": 0,
                    "bottom_3_avg_group_spread": 1.800000,
                    "fraction_groups_covered": 1.0,
                    "weakest_groups_note": "A,B,C",
                    "delta_f_score": 0.003000,
                }
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
            ml_backend="gnn",
            ml_guidance_mode="two_tier",
            gnn_model_type="graphsage",
            gnn_hidden_dim=64,
            gnn_num_layers=2,
            gnn_dropout=0.2,
            gnn_learning_rate=1e-3,
            gnn_weight_decay=5e-4,
            gnn_epochs=100,
            ml_singleton_runs=15,
        )

        report = format_results_report(frame, settings)

        self.assertIn("backend=gnn", report)
        self.assertIn("GNN=graphsage", report)
        self.assertIn("GNN config: type=graphsage", report)

    def test_format_results_report_shows_gnn_node2vec_metadata(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "dataset": "toy_graph",
                    "diffusion_model": "ic",
                    "community_method": "leiden",
                    "method": "hybrid_siea_ml_gnn_node2vec_two_tier_tuned_swap_local_search",
                    "variant_type": "ml_guided",
                    "total_spread": 4.85,
                    "extra_spread": 0.85,
                    "mf": 0.008900,
                    "dcv": 0.030500,
                    "f_score": -0.010800,
                    "runtime_seconds": 7.250000,
                    "search_runtime_seconds": 6.250000,
                    "final_eval_runtime_seconds": 1.000000,
                    "mc_runs_search": 20,
                    "mc_runs_eval": 1000,
                    "candidate_pool_size": 500,
                    "optimization_mode": "full",
                    "node2vec_enabled": True,
                    "node2vec_mode": "input_concat",
                    "ml_guidance_mode": "two_tier",
                    "ml_backend": "gnn",
                    "gnn_model_type": "graphsage",
                    "ml_validation_spearman": 0.310000,
                    "ml_validation_precision_at_budget": 0.25,
                    "community_modularity": 0.451674,
                    "zero_covered_groups_count": 0,
                    "bottom_3_avg_group_spread": 1.810000,
                    "fraction_groups_covered": 1.0,
                    "weakest_groups_note": "A,B,C",
                    "delta_f_score": 0.003300,
                }
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
            ml_backend="gnn",
            ml_guidance_mode="two_tier",
            gnn_model_type="graphsage",
            gnn_node2vec_mode="input_concat",
            gnn_hidden_dim=64,
            gnn_num_layers=2,
            gnn_dropout=0.2,
            gnn_learning_rate=1e-3,
            gnn_weight_decay=5e-4,
            gnn_epochs=100,
            node2vec_dimensions=8,
            node2vec_walk_length=20,
            node2vec_num_walks=10,
            node2vec_window=5,
            node2vec_p=1.0,
            node2vec_q=1.0,
            node2vec_scale_embeddings=False,
            node2vec_pca_components=None,
            ml_singleton_runs=15,
        )

        report = format_results_report(frame, settings)

        self.assertIn("Node2Vec: supported only as an optional GNN input feature source", report)
        self.assertIn("node2vec_mode=input_concat", report)
        self.assertIn("Node2Vec config: dimensions=8", report)
        self.assertIn("N2V=input_concat", report)

    def test_format_results_report_shows_ris_metadata(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "dataset": "toy_graph",
                    "diffusion_model": "ic",
                    "community_method": "leiden",
                    "method": "hybrid_siea_ml_gnn_ris_two_tier_tuned_swap_local_search",
                    "variant_type": "ml_guided",
                    "total_spread": 4.90,
                    "extra_spread": 0.90,
                    "mf": 0.009100,
                    "dcv": 0.030100,
                    "f_score": -0.010500,
                    "runtime_seconds": 7.400000,
                    "search_runtime_seconds": 6.200000,
                    "final_eval_runtime_seconds": 1.200000,
                    "mc_runs_search": 20,
                    "mc_runs_eval": 1000,
                    "candidate_pool_size": 500,
                    "optimization_mode": "full",
                    "node2vec_enabled": False,
                    "node2vec_mode": "off",
                    "ris_enabled": True,
                    "ris_mode": "weak_group_weighted",
                    "ml_guidance_mode": "two_tier",
                    "ml_backend": "gnn_ris",
                    "gnn_model_type": "graphsage",
                    "ml_validation_spearman": 0.280000,
                    "ml_validation_precision_at_budget": 0.20,
                    "community_modularity": 0.451674,
                    "zero_covered_groups_count": 0,
                    "bottom_3_avg_group_spread": 1.810000,
                    "fraction_groups_covered": 1.0,
                    "weakest_groups_note": "A,B,C",
                    "delta_f_score": 0.003500,
                }
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
            ml_backend="gnn_ris",
            ml_guidance_mode="two_tier",
            gnn_model_type="graphsage",
            gnn_hidden_dim=64,
            gnn_num_layers=2,
            gnn_dropout=0.2,
            gnn_learning_rate=1e-3,
            gnn_weight_decay=5e-4,
            gnn_epochs=100,
            ris_num_rr_sets=256,
            ris_mode="weak_group_weighted",
            ris_reuse_rr_sets=True,
            gnn_weight=1.0,
            ris_weight=1.0,
            fairness_urgency_weight=0.25,
            diversity_weight=0.15,
            ml_singleton_runs=15,
        )

        report = format_results_report(frame, settings)

        self.assertIn("backend=gnn_ris", report)
        self.assertIn("RIS=weak_group_weighted", report)
        self.assertIn("RIS config: num_rr_sets=256", report)
        self.assertIn("Guidance weights: gnn=1.0 | ris=1.0", report)

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
                    "ml_backend": "none",
                    "gnn_model_type": pd.NA,
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
                    "ml_backend": "tabular",
                    "gnn_model_type": pd.NA,
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
                    "ml_backend": "none",
                    "gnn_model_type": pd.NA,
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
            ml_backend="tabular",
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
