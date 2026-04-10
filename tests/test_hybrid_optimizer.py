"""Unit tests for the Phase 5 unified hybrid SI+EA optimizer."""

from __future__ import annotations

import unittest

import networkx as nx
import pandas as pd

from fim_hybrid.community_detection import (
    CommunityDetectionResult,
    CommunityStats,
    CommunityValidationReport,
    detect_communities,
)
from fim_hybrid.data_loader import LoadedDataset, verify_protected_groups
from fim_hybrid.diffusion import simulate_independent_cascade
from fim_hybrid.fairness import evaluate_fairness
from fim_hybrid.hybrid_optimizer import CandidateEvaluation, HybridSIEAConfig, HybridSIEAOptimizer


def _toy_optimizer_fixture() -> tuple[LoadedDataset, object, object]:
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
    node_groups = {
        1: "A",
        2: "A",
        3: "B",
        4: "B",
        5: "C",
        6: "C",
        7: "D",
        8: "D",
    }
    for node_id, group_name in node_groups.items():
        graph.nodes[node_id]["group"] = group_name

    node_attributes = pd.DataFrame(
        [{"node_id": node_id, "group": group_name} for node_id, group_name in sorted(node_groups.items())]
    ).set_index("node_id", drop=False)
    dataset = LoadedDataset(name="toy_optimizer", graph=graph, node_attributes=node_attributes)
    protected_group_report = verify_protected_groups(dataset, "group")
    community_result = detect_communities(graph, method="louvain", seed=42)
    return dataset, protected_group_report, community_result


class HybridOptimizerTestCase(unittest.TestCase):
    """Check invariants and reproducibility for the Phase 5 optimizer."""

    def _ml_node_scores(self) -> dict[int, float]:
        return {
            1: 0.10,
            2: 0.12,
            3: 0.20,
            4: 0.35,
            5: 0.48,
            6: 0.62,
            7: 0.78,
            8: 0.90,
        }

    def _node2vec_embeddings(self) -> dict[int, list[float]]:
        return {
            1: [1.0, 0.0, 0.0],
            2: [0.9, 0.1, 0.0],
            3: [0.8, 0.2, 0.0],
            4: [0.0, 1.0, 0.0],
            5: [0.0, 0.9, 0.1],
            6: [0.0, 0.8, 0.2],
            7: [0.0, 0.0, 1.0],
            8: [0.1, 0.0, 0.9],
        }

    def _fairness_stress_node_scores(self) -> dict[int, float]:
        return {
            1: 1.00,
            2: 0.95,
            3: 0.90,
            4: 0.88,
            5: 0.30,
            6: 0.28,
            7: 0.20,
            8: 0.18,
        }

    def _build_optimizer(
        self,
        candidate_nodes: list[int] | tuple[int, ...] | None = None,
        node_scores: dict[int, float] | None = None,
        ml_node_scores: dict[int, float] | None = None,
        node2vec_embeddings: dict[int, list[float]] | None = None,
        **config_overrides: object,
    ) -> HybridSIEAOptimizer:
        dataset, protected_group_report, community_result = _toy_optimizer_fixture()
        config = HybridSIEAConfig(
            budget=3,
            population_size=6,
            generations=4,
            crossover_probability=0.7,
            mutation_probability=0.4,
            elite_fraction=0.34,
            leader_guidance_fraction=0.34,
            propagation_probability=0.0,
            mc_runs=5,
            lambda_weight=0.5,
            random_seed=11,
        )
        for field_name, value in config_overrides.items():
            setattr(config, field_name, value)

        return HybridSIEAOptimizer(
            dataset=dataset,
            protected_group_report=protected_group_report,
            community_result=community_result,
            config=config,
            candidate_nodes=candidate_nodes,
            node_scores=node_scores,
            ml_node_scores=ml_node_scores,
            node2vec_embeddings=node2vec_embeddings,
        )

    def test_repair_prevents_duplicates_and_invalid_nodes(self) -> None:
        optimizer = self._build_optimizer()

        repaired = optimizer._repair_seed_set([1, 1, 99, 2])  # noqa: SLF001

        self.assertEqual(len(repaired), optimizer.config.budget)
        self.assertEqual(len(set(repaired)), len(repaired))
        self.assertTrue(set(repaired).issubset(set(optimizer.candidate_pool)))

    def test_guidance_changes_weak_individual(self) -> None:
        optimizer = self._build_optimizer(random_seed=5)
        current = (1, 2, 3)
        leader = (6, 7, 8)

        guided = optimizer._apply_leader_guidance(current, leader)  # noqa: SLF001

        self.assertNotEqual(guided, current)
        self.assertTrue(any(node in leader for node in guided))
        self.assertEqual(len(guided), optimizer.config.budget)

    def test_zero_guidance_fraction_is_no_op(self) -> None:
        optimizer = self._build_optimizer(leader_guidance_fraction=0.0, random_seed=5)

        guided = optimizer._apply_leader_guidance((1, 2, 3), (6, 7, 8))  # noqa: SLF001

        self.assertEqual(guided, (1, 2, 3))

    def test_crossover_and_mutation_preserve_seed_set_invariants(self) -> None:
        optimizer = self._build_optimizer(mutation_probability=1.0, random_seed=7)

        crossed = optimizer._crossover((1, 2, 3), (3, 4, 5))  # noqa: SLF001
        mutated = optimizer._mutate(crossed)  # noqa: SLF001

        self.assertEqual(len(crossed), optimizer.config.budget)
        self.assertEqual(len(set(crossed)), len(crossed))
        self.assertEqual(len(mutated), optimizer.config.budget)
        self.assertEqual(len(set(mutated)), len(mutated))

    def test_local_search_preserves_seed_set_invariants(self) -> None:
        optimizer = self._build_optimizer(local_search_steps=2, random_seed=7)

        refined = optimizer._local_search((1, 2, 3))  # noqa: SLF001

        self.assertEqual(len(refined), optimizer.config.budget)
        self.assertEqual(len(set(refined)), len(refined))
        self.assertTrue(set(refined).issubset(set(optimizer.candidate_pool)))

    def test_fitness_uses_verified_diffusion_and_fairness_pipeline(self) -> None:
        optimizer = self._build_optimizer()
        seed_set = (1, 4, 7)

        evaluation = optimizer._evaluate_seed_set(seed_set)  # noqa: SLF001
        diffusion_result = simulate_independent_cascade(
            dataset=optimizer.dataset,
            protected_group_report=optimizer.protected_group_report,
            seed_set=seed_set,
            propagation_probability=optimizer.config.propagation_probability,
            mc_runs=optimizer.config.mc_runs,
            random_seed=optimizer.config.random_seed,
        )
        fairness_metrics = evaluate_fairness(
            group_spread=diffusion_result.group_spread_mean,
            group_sizes=optimizer.protected_group_report.group_sizes,
            total_spread=diffusion_result.total_spread_mean,
            include_soft_mf=True,
        )
        expected_score = optimizer.config.lambda_weight * fairness_metrics.mf - (
            1.0 - optimizer.config.lambda_weight
        ) * fairness_metrics.dcv

        self.assertEqual(evaluation.total_spread_mean, diffusion_result.total_spread_mean)
        self.assertEqual(evaluation.fairness.mf, fairness_metrics.mf)
        self.assertEqual(evaluation.fairness.dcv, fairness_metrics.dcv)
        self.assertEqual(evaluation.score, expected_score)

    def test_optimize_is_reproducible(self) -> None:
        optimizer_a = self._build_optimizer(random_seed=13)
        optimizer_b = self._build_optimizer(random_seed=13)

        result_a = optimizer_a.optimize()
        result_b = optimizer_b.optimize()

        self.assertEqual(result_a.best_seed_set, result_b.best_seed_set)
        self.assertEqual(result_a.best_score, result_b.best_score)
        self.assertEqual(result_a.best_spread, result_b.best_spread)
        self.assertTrue(result_a.history.equals(result_b.history))

    def test_optimizer_runs_end_to_end_and_returns_valid_seed_set(self) -> None:
        optimizer = self._build_optimizer(debug_logging=True)

        result = optimizer.optimize()

        self.assertEqual(len(result.best_seed_set), optimizer.config.budget)
        self.assertEqual(len(set(result.best_seed_set)), len(result.best_seed_set))
        self.assertEqual(result.candidate_pool_size, len(optimizer.candidate_pool))
        self.assertGreaterEqual(len(result.history), 1)
        self.assertIn("generation", result.history.columns)
        self.assertIn("best_score", result.history.columns)
        self.assertIn("best_f_score", result.history.columns)
        self.assertIn("average_population_score", result.history.columns)
        self.assertIn("population_diversity", result.history.columns)
        self.assertIn("zero_covered_groups_count", result.history.columns)
        self.assertIn("bottom_3_avg_group_spread", result.history.columns)
        self.assertIn("fraction_groups_covered", result.history.columns)
        self.assertIn("weakest_groups_note", result.history.columns)

    def test_fairness_first_initialization_prioritizes_unrepresented_groups(self) -> None:
        optimizer = self._build_optimizer(
            node_scores=self._fairness_stress_node_scores(),
            fairness_first_init_enabled=True,
            fairness_first_init_slots=2,
            fairness_first_init_weight=1.5,
            random_seed=7,
        )

        ranked = optimizer._rank_fairness_first_candidates((1,))  # noqa: SLF001

        self.assertTrue(ranked)
        self.assertNotEqual(optimizer.node_group_by_node[ranked[0]], "A")

    def test_weakest_group_mutation_targets_zero_covered_groups(self) -> None:
        optimizer = self._build_optimizer(
            node_scores=self._fairness_stress_node_scores(),
            weakest_group_mutation_weight=1.2,
            zero_group_bonus_weight=1.0,
            bridge_to_weak_group_weight=0.5,
            random_seed=7,
        )
        seed_set = (1, 3, 5)
        evaluation = optimizer._evaluate_seed_set(seed_set)  # noqa: SLF001
        weak_context = optimizer._weak_group_context_for_seed_set(seed_set, evaluation=evaluation)  # noqa: SLF001

        ranked = optimizer._rank_external_candidates(  # noqa: SLF001
            seed_set,
            weak_group_context=weak_context,
            fairness_weight=optimizer.config.weakest_group_mutation_weight,
            zero_bonus_weight=optimizer.config.zero_group_bonus_weight,
            bridge_weight=optimizer.config.bridge_to_weak_group_weight,
        )

        self.assertTrue(ranked)
        self.assertEqual(optimizer.node_group_by_node[ranked[0]], "D")

    def test_fairness_aware_repair_prefers_weak_group_replacements(self) -> None:
        optimizer = self._build_optimizer(
            node_scores=self._fairness_stress_node_scores(),
            repair_fairness_weight=1.0,
            repair_bridge_weight=0.5,
            repair_centrality_weight=0.2,
            repair_diversity_weight=0.2,
            weakest_group_k=2,
            random_seed=7,
        )

        repaired = optimizer._repair_seed_set([1, 1, 3])  # noqa: SLF001
        added_nodes = [node_id for node_id in repaired if node_id not in {1, 3}]

        self.assertEqual(len(added_nodes), 1)
        self.assertIn(optimizer.node_group_by_node[added_nodes[0]], {"C", "D"})

    def test_worst_group_local_search_key_prioritizes_mf_then_dcv(self) -> None:
        optimizer = self._build_optimizer(
            local_search_focus_mode="worst_group",
            local_search_bottom_k_groups=2,
        )
        stronger_mf = CandidateEvaluation(
            seed_set=(1, 2, 3),
            total_spread_mean=4.0,
            total_spread_std=0.0,
            fairness=evaluate_fairness(
                group_spread={"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0},
                group_sizes=optimizer.protected_group_report.group_sizes,
                total_spread=4.0,
                include_soft_mf=True,
            ),
            score=-0.20,
        )
        weaker_mf_better_f = CandidateEvaluation(
            seed_set=(4, 5, 6),
            total_spread_mean=8.0,
            total_spread_std=0.0,
            fairness=evaluate_fairness(
                group_spread={"A": 2.0, "B": 2.0, "C": 2.0, "D": 0.5},
                group_sizes=optimizer.protected_group_report.group_sizes,
                total_spread=6.5,
                include_soft_mf=True,
            ),
            score=0.10,
        )

        self.assertGreater(
            optimizer._local_search_rank_key(stronger_mf),  # noqa: SLF001
            optimizer._local_search_rank_key(weaker_mf_better_f),  # noqa: SLF001
        )

    def test_marginal_gain_scoring_prioritizes_weak_group_candidates(self) -> None:
        optimizer = self._build_optimizer(
            node_scores=self._fairness_stress_node_scores(),
            marginal_gain_scoring_enabled=True,
            marginal_gain_delta_mf_weight=2.0,
            marginal_gain_delta_dcv_weight=2.0,
            marginal_gain_spread_weight=0.0,
            random_seed=7,
        )
        seed_set = (1, 2, 3)
        weak_context = optimizer._weak_group_context_for_seed_set(seed_set)  # noqa: SLF001

        ranked = optimizer._rank_external_candidates(  # noqa: SLF001
            seed_set,
            weak_group_context=weak_context,
        )

        self.assertTrue(ranked)
        self.assertIn(optimizer.node_group_by_node[ranked[0]], {"C", "D"})

    def test_urgency_weighting_changes_candidate_preference(self) -> None:
        base_optimizer = self._build_optimizer(
            node_scores=self._fairness_stress_node_scores(),
            marginal_gain_scoring_enabled=True,
            marginal_gain_delta_mf_weight=0.8,
            marginal_gain_delta_dcv_weight=0.6,
            marginal_gain_spread_weight=0.2,
            random_seed=7,
        )
        urgency_optimizer = self._build_optimizer(
            node_scores=self._fairness_stress_node_scores(),
            marginal_gain_scoring_enabled=True,
            marginal_gain_delta_mf_weight=0.8,
            marginal_gain_delta_dcv_weight=0.6,
            marginal_gain_spread_weight=0.2,
            urgency_weight_enabled=True,
            urgency_exponent=2.0,
            weak_group_focus_weight=0.5,
            random_seed=7,
        )
        seed_set = (1, 2, 3)
        weak_context = base_optimizer._weak_group_context_for_seed_set(seed_set)  # noqa: SLF001

        base_ranked = base_optimizer._rank_external_candidates(  # noqa: SLF001
            seed_set,
            weak_group_context=weak_context,
        )
        urgency_ranked = urgency_optimizer._rank_external_candidates(  # noqa: SLF001
            seed_set,
            weak_group_context=weak_context,
        )

        self.assertTrue(base_ranked)
        self.assertTrue(urgency_ranked)
        self.assertLess(
            urgency_ranked.index(7),
            base_ranked.index(7),
        )
        self.assertIn(urgency_optimizer.node_group_by_node[urgency_ranked[1]], {"C", "D"})

    def test_overlap_penalty_reduces_repeated_concentration(self) -> None:
        base_optimizer = self._build_optimizer(
            node_scores=self._fairness_stress_node_scores(),
            random_seed=7,
        )
        overlap_optimizer = self._build_optimizer(
            node_scores=self._fairness_stress_node_scores(),
            overlap_penalty_enabled=True,
            same_community_penalty_weight=5.0,
            neighborhood_overlap_penalty_weight=3.0,
            random_seed=7,
        )
        seed_set = (1, 2, 3)

        base_ranked = base_optimizer._rank_external_candidates(seed_set)  # noqa: SLF001
        overlap_ranked = overlap_optimizer._rank_external_candidates(seed_set)  # noqa: SLF001

        self.assertTrue(base_ranked)
        self.assertTrue(overlap_ranked)
        base_top_community = base_optimizer.community_result.community_id_by_node[base_ranked[0]]
        overlap_top_community = overlap_optimizer.community_result.community_id_by_node[overlap_ranked[0]]
        self.assertNotEqual(base_top_community, overlap_top_community)

    def test_swap_local_search_records_evaluated_replacements(self) -> None:
        optimizer = self._build_optimizer(
            local_search_focus_mode="worst_group",
            local_search_swap_trials=6,
            local_search_candidate_pool_size=4,
            local_search_delta_mf_weight=0.8,
            local_search_delta_dcv_weight=0.6,
            random_seed=7,
        )

        refined = optimizer._local_search((1, 2, 3))  # noqa: SLF001

        self.assertEqual(len(refined), optimizer.config.budget)
        self.assertGreater(optimizer.last_local_search_swap_evaluations, 0)

    def test_fitness_cache_reuses_evaluations_when_enabled(self) -> None:
        cached_optimizer = self._build_optimizer(enable_fitness_cache=True)
        uncached_optimizer = self._build_optimizer(enable_fitness_cache=False)

        cached_optimizer._evaluate_seed_set((1, 2, 3))  # noqa: SLF001
        cached_optimizer._evaluate_seed_set((1, 2, 3))  # noqa: SLF001
        uncached_optimizer._evaluate_seed_set((1, 2, 3))  # noqa: SLF001
        uncached_optimizer._evaluate_seed_set((1, 2, 3))  # noqa: SLF001

        self.assertGreater(cached_optimizer.fitness_cache_hits, 0)
        self.assertEqual(uncached_optimizer.fitness_cache_hits, 0)

    def test_marginal_cache_reuses_proxy_components_when_enabled(self) -> None:
        optimizer = self._build_optimizer(
            node_scores=self._fairness_stress_node_scores(),
            marginal_gain_scoring_enabled=True,
            marginal_gain_delta_mf_weight=0.8,
            marginal_gain_delta_dcv_weight=0.6,
            enable_marginal_cache=True,
            random_seed=7,
        )
        seed_set = (1, 2, 3)
        weak_context = optimizer._weak_group_context_for_seed_set(seed_set)  # noqa: SLF001

        optimizer._rank_external_candidates(  # noqa: SLF001
            seed_set,
            weak_group_context=weak_context,
            stage="marginal",
        )
        optimizer._rank_external_candidates(  # noqa: SLF001
            seed_set,
            weak_group_context=weak_context,
            stage="marginal",
        )

        self.assertGreater(optimizer.marginal_cache_hits, 0)

    def test_mutation_candidate_pool_size_limits_ranked_shortlist(self) -> None:
        optimizer = self._build_optimizer(
            node_scores=self._fairness_stress_node_scores(),
            marginal_gain_scoring_enabled=True,
            marginal_gain_delta_mf_weight=0.8,
            marginal_gain_delta_dcv_weight=0.6,
            mutation_candidate_pool_size=2,
            random_seed=7,
        )
        seed_set = (1, 2, 3)
        weak_context = optimizer._weak_group_context_for_seed_set(seed_set)  # noqa: SLF001

        ranked = optimizer._rank_external_candidates(  # noqa: SLF001
            seed_set,
            weak_group_context=weak_context,
            stage="mutation",
        )

        self.assertLessEqual(len(ranked), 2)

    def test_repair_candidate_pool_and_prefilter_controls_are_applied(self) -> None:
        optimizer = self._build_optimizer(
            repair_candidate_pool_size=2,
            candidate_prefilter_top_k=4,
            random_seed=7,
        )

        repair_limit = optimizer._candidate_pool_limit(len(optimizer.candidate_pool), stage="repair")  # noqa: SLF001
        prefilter_limit = optimizer._candidate_prefilter_limit(len(optimizer.candidate_pool), repair_limit, stage="repair")  # noqa: SLF001

        self.assertEqual(repair_limit, 2)
        self.assertEqual(prefilter_limit, 4)

    def test_balanced_and_fast_modes_reduce_local_search_budget(self) -> None:
        full_optimizer = self._build_optimizer(
            local_search_focus_mode="worst_group",
            local_search_candidate_pool_size=6,
            local_search_max_trials=10,
            local_search_early_stop_patience=6,
            local_search_delta_mf_weight=0.8,
            local_search_delta_dcv_weight=0.6,
            random_seed=7,
        )
        balanced_optimizer = self._build_optimizer(
            local_search_focus_mode="worst_group",
            local_search_candidate_pool_size=6,
            local_search_max_trials=10,
            local_search_early_stop_patience=6,
            local_search_delta_mf_weight=0.8,
            local_search_delta_dcv_weight=0.6,
            local_search_use_prefilter=True,
            optimization_mode="balanced",
            random_seed=7,
        )
        fast_optimizer = self._build_optimizer(
            local_search_focus_mode="worst_group",
            local_search_candidate_pool_size=6,
            local_search_max_trials=10,
            local_search_early_stop_patience=6,
            local_search_delta_mf_weight=0.8,
            local_search_delta_dcv_weight=0.6,
            local_search_use_prefilter=True,
            optimization_mode="fast",
            random_seed=7,
        )

        self.assertGreater(
            full_optimizer._effective_local_search_budget()[0],  # noqa: SLF001
            balanced_optimizer._effective_local_search_budget()[0],  # noqa: SLF001
        )
        self.assertGreater(
            balanced_optimizer._effective_local_search_budget()[0],  # noqa: SLF001
            fast_optimizer._effective_local_search_budget()[0],  # noqa: SLF001
        )

        full_optimizer._local_search((1, 2, 3))  # noqa: SLF001
        balanced_optimizer._local_search((1, 2, 3))  # noqa: SLF001
        fast_optimizer._local_search((1, 2, 3))  # noqa: SLF001

        self.assertLessEqual(
            balanced_optimizer.last_local_search_swap_evaluations,
            full_optimizer.last_local_search_swap_evaluations,
        )
        self.assertLessEqual(
            fast_optimizer.last_local_search_swap_evaluations,
            balanced_optimizer.last_local_search_swap_evaluations,
        )

    def test_swap_cache_reuses_evaluations_when_enabled(self) -> None:
        optimizer = self._build_optimizer(
            enable_swap_cache=True,
            random_seed=7,
        )

        first_evaluation, first_cached = optimizer._evaluate_swap_candidate((1, 2), 4)  # noqa: SLF001
        second_evaluation, second_cached = optimizer._evaluate_swap_candidate((1, 2), 4)  # noqa: SLF001

        self.assertFalse(first_cached)
        self.assertTrue(second_cached)
        self.assertEqual(first_evaluation.seed_set, second_evaluation.seed_set)
        self.assertGreater(optimizer.swap_cache_hits, 0)

    def test_staged_mc_uses_fast_screening_and_full_final_evaluation(self) -> None:
        optimizer = self._build_optimizer(
            local_search_focus_mode="worst_group",
            local_search_swap_trials=6,
            local_search_candidate_pool_size=4,
            local_search_delta_mf_weight=0.8,
            local_search_delta_dcv_weight=0.6,
            swap_candidate_pool_size=4,
            swap_prefilter_top_k=5,
            full_eval_top_k=2,
            use_staged_mc=True,
            mc_runs=5,
            mc_runs_fast=2,
            mc_runs_full=5,
            enable_swap_cache=True,
            random_seed=7,
        )

        refined = optimizer._local_search((1, 2, 3))  # noqa: SLF001
        final_evaluation = optimizer._evaluate_seed_set(refined)  # noqa: SLF001

        self.assertEqual(len(refined), optimizer.config.budget)
        self.assertEqual(final_evaluation.seed_set, refined)
        self.assertGreater(optimizer.screening_evaluation_calls, 0)
        self.assertGreater(optimizer.full_evaluation_calls, 0)
        self.assertEqual(optimizer.last_screening_mc_runs, 2)
        self.assertEqual(optimizer.last_full_mc_runs, 5)

    def test_swap_runtime_controls_reduce_expensive_local_search_evaluations(self) -> None:
        baseline_optimizer = self._build_optimizer(
            local_search_focus_mode="worst_group",
            local_search_swap_trials=8,
            local_search_candidate_pool_size=6,
            local_search_delta_mf_weight=0.8,
            local_search_delta_dcv_weight=0.6,
            random_seed=7,
        )
        optimized_optimizer = self._build_optimizer(
            local_search_focus_mode="worst_group",
            local_search_swap_trials=8,
            local_search_candidate_pool_size=6,
            local_search_delta_mf_weight=0.8,
            local_search_delta_dcv_weight=0.6,
            swap_candidate_pool_size=4,
            swap_prefilter_top_k=5,
            full_eval_top_k=2,
            enable_swap_cache=True,
            local_search_failed_patience=2,
            random_seed=7,
        )

        baseline_optimizer._local_search((1, 2, 3))  # noqa: SLF001
        optimized_optimizer._local_search((1, 2, 3))  # noqa: SLF001

        self.assertGreater(baseline_optimizer.last_local_search_swap_evaluations, 0)
        self.assertLessEqual(
            optimized_optimizer.last_local_search_swap_evaluations,
            baseline_optimizer.last_local_search_swap_evaluations,
        )

    def test_first_improvement_mode_reduces_local_search_evaluations(self) -> None:
        best_improvement_optimizer = self._build_optimizer(
            local_search_focus_mode="worst_group",
            local_search_swap_trials=8,
            local_search_candidate_pool_size=6,
            local_search_delta_mf_weight=0.8,
            local_search_delta_dcv_weight=0.6,
            swap_candidate_pool_size=4,
            swap_prefilter_top_k=5,
            full_eval_top_k=3,
            enable_swap_cache=True,
            random_seed=7,
        )
        first_improvement_optimizer = self._build_optimizer(
            local_search_focus_mode="worst_group",
            local_search_swap_trials=8,
            local_search_candidate_pool_size=6,
            local_search_delta_mf_weight=0.8,
            local_search_delta_dcv_weight=0.6,
            swap_candidate_pool_size=4,
            swap_prefilter_top_k=5,
            full_eval_top_k=3,
            enable_swap_cache=True,
            local_search_first_improvement=True,
            random_seed=7,
        )

        best_improvement_optimizer._local_search((1, 2, 3))  # noqa: SLF001
        first_improvement_optimizer._local_search((1, 2, 3))  # noqa: SLF001

        self.assertLessEqual(
            first_improvement_optimizer.last_local_search_swap_evaluations,
            best_improvement_optimizer.last_local_search_swap_evaluations,
        )

    def test_selective_local_search_applies_to_fewer_individuals(self) -> None:
        full_optimizer = self._build_optimizer(
            local_search_focus_mode="worst_group",
            local_search_swap_trials=6,
            local_search_candidate_pool_size=4,
            local_search_delta_mf_weight=0.8,
            local_search_delta_dcv_weight=0.6,
            random_seed=7,
        )
        selective_optimizer = self._build_optimizer(
            local_search_focus_mode="worst_group",
            local_search_swap_trials=6,
            local_search_candidate_pool_size=4,
            local_search_delta_mf_weight=0.8,
            local_search_delta_dcv_weight=0.6,
            local_search_elite_count=1,
            local_search_every_n_generations=2,
            random_seed=7,
        )

        full_result = full_optimizer.optimize()
        selective_result = selective_optimizer.optimize()

        self.assertIn("local_search_applied_count", full_result.history.columns)
        self.assertIn("screening_evaluation_calls", selective_result.history.columns)
        self.assertLess(
            int(selective_result.history["local_search_applied_count"].sum()),
            int(full_result.history["local_search_applied_count"].sum()),
        )

    def test_soft_bias_mode_preserves_full_candidate_pool(self) -> None:
        optimizer = self._build_optimizer(
            ml_node_scores=self._ml_node_scores(),
            ml_guidance_mode="soft_bias",
            ml_initialization_bias=1.0,
            random_seed=7,
        )

        result = optimizer.optimize()

        self.assertEqual(result.candidate_pool_size, optimizer.dataset.graph.number_of_nodes())
        self.assertEqual(len(result.best_seed_set), optimizer.config.budget)
        self.assertEqual(len(set(result.best_seed_set)), len(result.best_seed_set))

    def test_two_tier_mode_builds_primary_and_secondary_pools(self) -> None:
        optimizer = self._build_optimizer(
            ml_node_scores=self._ml_node_scores(),
            ml_guidance_mode="two_tier",
            ml_primary_pool_ratio=0.5,
            random_seed=7,
        )

        result = optimizer.optimize()

        self.assertTrue(optimizer.primary_ml_pool)
        self.assertTrue(optimizer.secondary_ml_pool)
        self.assertEqual(result.candidate_pool_size, optimizer.dataset.graph.number_of_nodes())
        self.assertEqual(len(set(result.best_seed_set)), len(result.best_seed_set))

    def test_tuned_two_tier_can_reach_secondary_pool(self) -> None:
        optimizer = self._build_optimizer(
            ml_node_scores=self._ml_node_scores(),
            ml_guidance_mode="two_tier",
            ml_primary_pool_ratio=0.5,
            ml_secondary_exploration_rate=1.0,
            random_seed=7,
        )

        sampled = optimizer._sample_unused_node(  # noqa: SLF001
            optimizer.candidate_pool,
            used_nodes=set(),
            ml_bias_weight=optimizer.config.ml_repair_bias_weight,
            primary_rate=0.0,
        )

        self.assertIsNotNone(sampled)
        self.assertIn(sampled, optimizer.secondary_ml_pool)

    def test_tuned_two_tier_primary_rate_changes_candidate_order(self) -> None:
        primary_optimizer = self._build_optimizer(
            ml_node_scores=self._ml_node_scores(),
            ml_guidance_mode="two_tier",
            ml_primary_pool_ratio=0.5,
            ml_secondary_exploration_rate=0.0,
            random_seed=7,
        )
        secondary_optimizer = self._build_optimizer(
            ml_node_scores=self._ml_node_scores(),
            ml_guidance_mode="two_tier",
            ml_primary_pool_ratio=0.5,
            ml_secondary_exploration_rate=0.0,
            random_seed=7,
        )

        primary_ranked = primary_optimizer._rank_external_candidates(  # noqa: SLF001
            (1, 2, 3),
            ml_bias_weight=primary_optimizer.config.ml_mutation_bias_weight,
            primary_rate=1.0,
        )
        secondary_ranked = secondary_optimizer._rank_external_candidates(  # noqa: SLF001
            (1, 2, 3),
            ml_bias_weight=secondary_optimizer.config.ml_mutation_bias_weight,
            primary_rate=0.0,
        )

        self.assertIn(primary_ranked[0], primary_optimizer.primary_ml_pool)
        self.assertIn(secondary_ranked[0], secondary_optimizer.secondary_ml_pool)

    def test_tuned_two_tier_preserves_seed_set_invariants(self) -> None:
        optimizer = self._build_optimizer(
            ml_node_scores=self._ml_node_scores(),
            ml_guidance_mode="two_tier",
            ml_primary_pool_ratio=0.5,
            ml_initialization_bias=1.0,
            mutation_probability=1.0,
            random_seed=7,
        )

        initialized = optimizer._initialize_individual(use_ml_bias=True)  # noqa: SLF001
        mutated = optimizer._mutate(initialized)  # noqa: SLF001
        repaired = optimizer._repair_seed_set(mutated + mutated[:1])  # noqa: SLF001
        refined = optimizer._local_search(repaired)  # noqa: SLF001

        for seed_set in [initialized, mutated, repaired, refined]:
            self.assertEqual(len(seed_set), optimizer.config.budget)
            self.assertEqual(len(set(seed_set)), len(seed_set))
            self.assertTrue(set(seed_set).issubset(set(optimizer.candidate_pool)))

    def test_hard_filter_mode_restricts_candidate_pool(self) -> None:
        candidate_nodes = (4, 5, 6, 7, 8)
        optimizer = self._build_optimizer(
            candidate_nodes=candidate_nodes,
            node_scores=self._ml_node_scores(),
            ml_guidance_mode="hard_filter",
            random_seed=7,
        )

        result = optimizer.optimize()

        self.assertEqual(result.candidate_pool_size, len(candidate_nodes))
        self.assertTrue(set(result.best_seed_set).issubset(set(candidate_nodes)))

    def test_soft_guidance_requires_ml_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "ml_node_scores"):
            self._build_optimizer(ml_guidance_mode="soft_bias")

    def test_node2vec_diversity_requires_embeddings(self) -> None:
        with self.assertRaisesRegex(ValueError, "node2vec_embeddings"):
            self._build_optimizer(node2vec_diversity_weight=0.2)

    def test_node2vec_diversity_scores_more_novel_nodes_higher(self) -> None:
        optimizer = self._build_optimizer(
            ml_node_scores=self._ml_node_scores(),
            ml_guidance_mode="two_tier",
            node2vec_embeddings=self._node2vec_embeddings(),
            node2vec_diversity_weight=0.5,
            random_seed=7,
        )

        moderate_novelty = optimizer._node2vec_diversity_score(4, {1, 2, 3})  # noqa: SLF001
        high_novelty = optimizer._node2vec_diversity_score(7, {1, 2, 3})  # noqa: SLF001

        self.assertGreater(high_novelty, moderate_novelty)

    def test_invalid_community_result_is_rejected(self) -> None:
        dataset, protected_group_report, _ = _toy_optimizer_fixture()
        invalid_result = CommunityDetectionResult(
            method="manual",
            community_id_by_node={1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 3},
            communities={0: (1, 2), 1: (3, 4), 2: (5, 6), 3: (7,)},
            stats=CommunityStats(
                num_communities=4,
                community_sizes={0: 2, 1: 2, 2: 2, 3: 1},
                largest_community_size=2,
                smallest_community_size=1,
                average_community_size=1.75,
            ),
            validation=CommunityValidationReport(
                every_node_assigned_exactly_once=False,
                mapping_matches_grouped_communities=False,
                has_empty_communities=False,
                missing_nodes=(8,),
                conflicting_nodes=(),
                extra_nodes=(),
                empty_community_ids=(),
            ),
            runtime_seconds=0.0,
        )

        with self.assertRaisesRegex(ValueError, "community_result validation"):
            HybridSIEAOptimizer(
                dataset=dataset,
                protected_group_report=protected_group_report,
                community_result=invalid_result,
                config=HybridSIEAConfig(
                    budget=3,
                    population_size=4,
                    generations=2,
                    propagation_probability=0.0,
                    mc_runs=2,
                    random_seed=11,
                ),
            )

    def test_optimizer_handles_small_unique_seed_space(self) -> None:
        graph = nx.Graph()
        graph.add_edge(1, 2)
        graph.nodes[1]["group"] = "A"
        graph.nodes[2]["group"] = "B"
        node_attributes = pd.DataFrame(
            [{"node_id": 1, "group": "A"}, {"node_id": 2, "group": "B"}]
        ).set_index("node_id", drop=False)
        dataset = LoadedDataset(name="tiny_optimizer", graph=graph, node_attributes=node_attributes)
        protected_group_report = verify_protected_groups(dataset, "group")
        community_result = detect_communities(graph, method="louvain", seed=7)
        optimizer = HybridSIEAOptimizer(
            dataset=dataset,
            protected_group_report=protected_group_report,
            community_result=community_result,
            config=HybridSIEAConfig(
                budget=2,
                population_size=4,
                generations=2,
                crossover_probability=0.0,
                mutation_probability=0.0,
                leader_guidance_fraction=0.0,
                propagation_probability=0.0,
                mc_runs=1,
                random_seed=7,
            ),
        )

        result = optimizer.optimize()

        self.assertEqual(result.best_seed_set, (1, 2))
        self.assertEqual(len(result.history), 2)


if __name__ == "__main__":
    unittest.main()
