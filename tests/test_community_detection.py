"""Unit tests for the Phase 4 community layer."""

from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

try:
    import igraph as ig
except ImportError:
    ig = None

from fim_hybrid.community_detection import (
    choose_communities,
    compute_community_quality_metrics,
    detect_communities,
    format_community_debug_report,
    get_community_stats,
    get_community_sizes,
    get_node_community,
    get_nodes_in_community,
    sample_community,
    sample_candidate_nodes_by_community,
    sample_node_from_community,
    score_communities_by_size,
    validate_communities,
    validate_community_assignments,
)


def _two_cluster_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (1, 2),
            (1, 3),
            (2, 3),
            (4, 5),
            (4, 6),
            (5, 6),
        ]
    )
    return graph


class CommunityDetectionTestCase(unittest.TestCase):
    """Check community detection, validation, and helper consistency."""

    def test_detect_communities_assigns_every_node_exactly_once(self) -> None:
        graph = _two_cluster_graph()

        result = detect_communities(graph, method="louvain", seed=42)

        self.assertTrue(result.validation.every_node_assigned_exactly_once)
        self.assertFalse(result.validation.has_empty_communities)
        self.assertEqual(set(result.community_id_by_node), set(graph.nodes()))
        flattened_nodes = {node for nodes in result.communities.values() for node in nodes}
        self.assertEqual(flattened_nodes, set(graph.nodes()))

    def test_validate_communities_alias_matches_assignment_validator(self) -> None:
        graph = _two_cluster_graph()
        validation = validate_communities(
            graph=graph,
            community_id_by_node={1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1},
            communities={0: (1, 2, 3), 1: (4, 5, 6)},
        )

        self.assertTrue(validation.every_node_assigned_exactly_once)

    def test_validate_community_assignments_detects_missing_nodes(self) -> None:
        graph = _two_cluster_graph()
        validation = validate_community_assignments(
            graph=graph,
            community_id_by_node={1: 0, 2: 0},
            communities={0: (1, 2, 3), 1: (4, 5)},
        )

        self.assertFalse(validation.every_node_assigned_exactly_once)
        self.assertIn(6, validation.missing_nodes)

    def test_validate_community_assignments_detects_conflicts(self) -> None:
        graph = _two_cluster_graph()
        validation = validate_community_assignments(
            graph=graph,
            community_id_by_node={1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1},
            communities={0: (1, 2, 3), 1: (3, 4, 5, 6)},
        )

        self.assertFalse(validation.every_node_assigned_exactly_once)
        self.assertIn(3, validation.conflicting_nodes)

    def test_validate_community_assignments_detects_empty_community(self) -> None:
        graph = _two_cluster_graph()
        validation = validate_community_assignments(
            graph=graph,
            community_id_by_node={1: 0, 2: 0, 3: 0, 4: 2, 5: 2, 6: 2},
            communities={0: (1, 2, 3), 1: (), 2: (4, 5, 6)},
        )

        self.assertTrue(validation.has_empty_communities)
        self.assertEqual(validation.empty_community_ids, (1,))

    def test_helpers_return_consistent_results(self) -> None:
        graph = _two_cluster_graph()
        result = detect_communities(graph, method="louvain", seed=42)

        community_sizes = get_community_sizes(result)
        self.assertEqual(sum(community_sizes.values()), graph.number_of_nodes())

        for node_id in graph.nodes():
            community_id = get_node_community(result, node_id)
            self.assertIn(node_id, get_nodes_in_community(result, community_id))

    def test_get_community_stats_matches_detection_result(self) -> None:
        graph = _two_cluster_graph()
        result = detect_communities(graph, method="louvain", seed=42)
        stats = get_community_stats(result.communities)

        self.assertEqual(stats.num_communities, result.stats.num_communities)
        self.assertEqual(stats.community_sizes, result.stats.community_sizes)

    def test_compute_community_quality_metrics_reports_modularity(self) -> None:
        graph = _two_cluster_graph()
        result = detect_communities(graph, method="louvain", seed=42)
        quality = compute_community_quality_metrics(graph, result)

        self.assertEqual(quality.num_communities, result.stats.num_communities)
        self.assertGreaterEqual(quality.modularity, 0.0)
        self.assertGreaterEqual(quality.community_size_std, 0.0)

    def test_choose_communities_is_deterministic(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from(
            [
                (1, 2),
                (1, 3),
                (2, 3),
                (4, 5),
                (5, 6),
                (6, 4),
                (7, 8),
            ]
        )
        result = detect_communities(graph, method="louvain", seed=42)

        first = choose_communities(result, count=2, mode="proportional_size", random_seed=7)
        second = choose_communities(result, count=2, mode="proportional_size", random_seed=7)

        self.assertEqual(first, second)

    def test_sample_community_is_deterministic_with_rng(self) -> None:
        graph = _two_cluster_graph()
        result = detect_communities(graph, method="louvain", seed=42)
        rng_one = np.random.default_rng(7)
        rng_two = np.random.default_rng(7)

        first = sample_community(result.communities, result.stats.community_sizes, rng_one)
        second = sample_community(result.communities, result.stats.community_sizes, rng_two)

        self.assertEqual(first, second)

    def test_sample_node_from_community_is_deterministic_with_rng(self) -> None:
        rng_one = np.random.default_rng(9)
        rng_two = np.random.default_rng(9)
        community_nodes = (4, 5, 6)

        first = sample_node_from_community(community_nodes, rng_one)
        second = sample_node_from_community(community_nodes, rng_two)

        self.assertEqual(first, second)

    def test_score_communities_by_size_returns_size_scores(self) -> None:
        graph = _two_cluster_graph()
        result = detect_communities(graph, method="louvain", seed=42)
        scores = score_communities_by_size(result.communities)

        self.assertEqual(scores, {0: 3.0, 1: 3.0})

    def test_sample_candidate_nodes_by_community_uses_scores_deterministically(self) -> None:
        graph = _two_cluster_graph()
        result = detect_communities(graph, method="louvain", seed=42)

        sampled = sample_candidate_nodes_by_community(
            result=result,
            community_ids=[0, 1],
            nodes_per_community=2,
            node_scores={1: 10.0, 2: 9.0, 3: 8.0, 4: 7.0, 5: 6.0, 6: 5.0},
        )

        self.assertEqual(sampled[0], (1, 2))
        self.assertEqual(sampled[1], (4, 5))

    def test_format_community_debug_report_contains_validation_status(self) -> None:
        graph = _two_cluster_graph()
        result = detect_communities(graph, method="louvain", seed=42)
        report = format_community_debug_report(result)

        self.assertIn("num_communities:", report)
        self.assertIn("every_node_assigned_exactly_once: True", report)
        self.assertIn("has_empty_communities: False", report)

    @unittest.skipUnless(ig is not None and hasattr(ig.Graph, "community_leiden"), "igraph Leiden is unavailable")
    def test_detect_communities_leiden_runs_when_available(self) -> None:
        graph = _two_cluster_graph()
        result = detect_communities(graph, method="leiden", seed=42)

        self.assertTrue(result.validation.every_node_assigned_exactly_once)
        self.assertGreaterEqual(result.stats.num_communities, 1)


if __name__ == "__main__":
    unittest.main()
