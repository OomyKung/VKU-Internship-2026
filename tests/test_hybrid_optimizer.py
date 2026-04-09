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
from fim_hybrid.hybrid_optimizer import HybridSIEAConfig, HybridSIEAOptimizer


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

    def _build_optimizer(self, **config_overrides: object) -> HybridSIEAOptimizer:
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
        self.assertIn("average_population_score", result.history.columns)
        self.assertIn("population_diversity", result.history.columns)

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
