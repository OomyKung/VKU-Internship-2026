"""Invariant tests for the Phase 4 hybrid SI+EA optimizer."""

from __future__ import annotations

import unittest

import networkx as nx

from fim_hybrid.community_detection import CommunityDetectionResult
from fim_hybrid.config import FairnessConfig, OptimizerConfig
from fim_hybrid.diffusion import IndependentCascadeSimulator
from fim_hybrid.feature_extraction import compute_node_features
from fim_hybrid.hybrid_optimizer import HybridSIEAOptimizer


class HybridOptimizerTestCase(unittest.TestCase):
    """Small, deterministic tests for optimizer invariants."""

    def setUp(self) -> None:
        self.graph = nx.Graph()
        node_groups = {
            0: "red",
            1: "red",
            2: "blue",
            3: "blue",
            4: "red",
            5: "blue",
        }
        for node, color in node_groups.items():
            self.graph.add_node(node, color=color)

        edges = [
            (0, 1),
            (1, 2),
            (0, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (3, 5),
            (1, 4),
        ]
        self.graph.add_edges_from(edges)

        self.community_result = CommunityDetectionResult(
            method="manual",
            partition={0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1},
            communities=[[0, 1, 2], [3, 4, 5]],
            runtime_seconds=0.0,
        )
        self.feature_frame = compute_node_features(
            self.graph,
            self.community_result,
            protected_attribute="color",
        )
        self.simulator = IndependentCascadeSimulator(
            self.graph,
            propagation_probability=0.2,
            seed=11,
        )
        self.fairness_config = FairnessConfig(
            protected_attribute="color",
            lambda_weight=0.5,
        )
        self.optimizer_config = OptimizerConfig(
            budget=3,
            population_size=4,
            generations=3,
            crossover_rate=0.7,
            mutation_rate=0.4,
            elite_fraction=0.5,
            swarm_inertia_rate=0.35,
            swarm_cognitive_rate=0.30,
            swarm_social_rate=0.25,
            swarm_elite_rate=0.15,
            restart_rate=0.10,
            seed=11,
        )

    def _build_optimizer(self, **kwargs: object) -> HybridSIEAOptimizer:
        """Create an optimizer with stable defaults for tests."""

        optimizer_config = kwargs.pop("optimizer_config", self.optimizer_config)
        mc_runs = kwargs.pop("mc_runs", 3)
        return HybridSIEAOptimizer(
            graph=self.graph,
            community_result=self.community_result,
            feature_frame=self.feature_frame,
            simulator=self.simulator,
            fairness_config=self.fairness_config,
            optimizer_config=optimizer_config,
            mc_runs=mc_runs,
            **kwargs,
        )

    def test_repair_removes_duplicates_and_invalid_nodes(self) -> None:
        optimizer = self._build_optimizer()

        repaired = optimizer._repair_seed_set([0, 0, 99, 1])  # noqa: SLF001

        self.assertEqual(len(repaired), self.optimizer_config.budget)
        self.assertEqual(len(set(repaired)), len(repaired))
        self.assertTrue(set(repaired).issubset(set(optimizer.candidate_pool)))

    def test_mutation_preserves_seed_set_invariants(self) -> None:
        mutation_config = OptimizerConfig(
            budget=3,
            population_size=4,
            generations=3,
            crossover_rate=0.7,
            mutation_rate=1.0,
            elite_fraction=0.5,
            swarm_inertia_rate=0.35,
            swarm_cognitive_rate=0.30,
            swarm_social_rate=0.25,
            swarm_elite_rate=0.15,
            restart_rate=0.10,
            seed=11,
        )
        optimizer = self._build_optimizer(optimizer_config=mutation_config)

        mutated = optimizer._mutate((0, 1, 3))  # noqa: SLF001

        self.assertEqual(len(mutated), mutation_config.budget)
        self.assertEqual(len(set(mutated)), len(mutated))
        self.assertTrue(set(mutated).issubset(set(optimizer.candidate_pool)))

    def test_invalid_candidate_pool_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._build_optimizer(candidate_nodes=[0, 1, 2, 99])

    def test_optimize_is_reproducible_and_returns_valid_seed_set(self) -> None:
        optimizer_a = self._build_optimizer()
        optimizer_b = self._build_optimizer()

        result_a = optimizer_a.optimize()
        result_b = optimizer_b.optimize()

        self.assertEqual(result_a.best_seed_set, result_b.best_seed_set)
        self.assertEqual(result_a.best_score, result_b.best_score)
        self.assertEqual(len(result_a.best_seed_set), self.optimizer_config.budget)
        self.assertEqual(len(set(result_a.best_seed_set)), len(result_a.best_seed_set))
        self.assertGreaterEqual(len(result_a.history), 1)
        self.assertIn("population_diversity", result_a.history.columns)
        self.assertIn("best_seed_groups", result_a.history.columns)
        self.assertIn("best_seed_communities", result_a.history.columns)
        self.assertIn("best_mf_to_ideal", result_a.history.columns)

    def test_ablation_switches_still_produce_valid_seed_sets(self) -> None:
        ablation_config = OptimizerConfig(
            budget=3,
            population_size=4,
            generations=3,
            crossover_rate=0.7,
            mutation_rate=0.4,
            elite_fraction=0.5,
            swarm_inertia_rate=0.35,
            swarm_cognitive_rate=0.30,
            swarm_social_rate=0.25,
            swarm_elite_rate=0.15,
            restart_rate=0.10,
            fairness_repair_bias=0.35,
            disable_swarm_guidance=True,
            disable_crossover=True,
            disable_community_repair=True,
            seed=11,
        )
        optimizer = self._build_optimizer(optimizer_config=ablation_config)

        result = optimizer.optimize()

        self.assertEqual(len(result.best_seed_set), ablation_config.budget)
        self.assertEqual(len(set(result.best_seed_set)), len(result.best_seed_set))
        self.assertGreaterEqual(len(result.history), 1)


if __name__ == "__main__":
    unittest.main()
