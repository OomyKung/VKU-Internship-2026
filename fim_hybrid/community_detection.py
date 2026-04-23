"""Phase 4 community detection and helper utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterable, Sequence
import random as py_random

try:
    import igraph as ig
except ImportError:
    ig = None

import networkx as nx
import numpy as np
import pandas as pd

from .clustering import ClusteringResult, cluster_nodes


@dataclass(slots=True)
class CommunityStats:
    """Summary statistics for a validated community partition."""

    num_communities: int
    community_sizes: dict[int, int]
    largest_community_size: int
    smallest_community_size: int
    average_community_size: float


@dataclass(slots=True)
class CommunityQualityMetrics:
    """Community quality metrics for experiment reporting."""

    modularity: float
    num_communities: int
    largest_community_size: int
    smallest_community_size: int
    average_community_size: float
    community_size_std: float


@dataclass(slots=True)
class CommunityValidationReport:
    """Validation outcome for community assignments."""

    every_node_assigned_exactly_once: bool
    mapping_matches_grouped_communities: bool
    has_empty_communities: bool
    missing_nodes: tuple[Any, ...]
    conflicting_nodes: tuple[Any, ...]
    extra_nodes: tuple[Any, ...]
    empty_community_ids: tuple[int, ...]


@dataclass(slots=True)
class CommunityDetectionResult:
    """Validated community detection output."""

    method: str
    community_id_by_node: dict[Any, int]
    communities: dict[int, tuple[Any, ...]]
    stats: CommunityStats
    validation: CommunityValidationReport
    runtime_seconds: float
    category: str = "graph_native"
    requested_input_mode: str = "graph"
    resolved_input_mode: str = "graph"
    metadata: dict[str, Any] = field(default_factory=dict)


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _normalize_communities(
    communities: Iterable[Iterable[Any]],
) -> dict[int, tuple[Any, ...]]:
    normalized_groups = [
        tuple(sorted(list(group), key=_sort_key))
        for group in communities
    ]

    non_empty_groups = [group for group in normalized_groups if group]
    if len(non_empty_groups) == len(normalized_groups):
        normalized_groups = sorted(non_empty_groups, key=lambda group: _sort_key(group[0]))

    return {
        community_id: group
        for community_id, group in enumerate(normalized_groups)
    }


def _coerce_communities(
    communities: dict[int, Sequence[Any]] | Iterable[Iterable[Any]],
) -> dict[int, tuple[Any, ...]]:
    if isinstance(communities, dict):
        normalized = {
            int(community_id): tuple(nodes)
            for community_id, nodes in communities.items()
        }
        return {
            community_id: tuple(sorted(nodes, key=_sort_key))
            for community_id, nodes in sorted(normalized.items())
        }
    return _normalize_communities(communities)


def validate_community_assignments(
    graph: nx.Graph,
    community_id_by_node: dict[Any, int],
    communities: dict[int, Sequence[Any]],
) -> CommunityValidationReport:
    """Validate that a community assignment covers each graph node exactly once."""

    if graph.number_of_nodes() == 0:
        raise ValueError("Community detection requires a graph with at least one node.")

    graph_nodes = set(graph.nodes())
    empty_community_ids = tuple(
        community_id
        for community_id, nodes in sorted(communities.items())
        if len(tuple(nodes)) == 0
    )

    grouped_assignment: dict[Any, int] = {}
    conflicting_nodes: set[Any] = set()
    for community_id, nodes in communities.items():
        for node_id in nodes:
            if node_id in grouped_assignment and grouped_assignment[node_id] != community_id:
                conflicting_nodes.add(node_id)
                continue
            grouped_assignment.setdefault(node_id, community_id)

    missing_nodes = tuple(sorted(
        (graph_nodes - set(community_id_by_node)) | (graph_nodes - set(grouped_assignment)),
        key=_sort_key,
    ))

    extra_nodes = tuple(sorted(
        (set(community_id_by_node) - graph_nodes) | (set(grouped_assignment) - graph_nodes),
        key=_sort_key,
    ))

    for node_id, community_id in community_id_by_node.items():
        if community_id not in communities:
            conflicting_nodes.add(node_id)
            continue
        if grouped_assignment.get(node_id) != community_id:
            conflicting_nodes.add(node_id)

    mapping_matches_grouped_communities = (
        not conflicting_nodes
        and not missing_nodes
        and not extra_nodes
        and set(community_id_by_node) == set(grouped_assignment)
    )
    every_node_assigned_exactly_once = (
        mapping_matches_grouped_communities
        and not empty_community_ids
        and len(grouped_assignment) == graph.number_of_nodes()
    )

    return CommunityValidationReport(
        every_node_assigned_exactly_once=every_node_assigned_exactly_once,
        mapping_matches_grouped_communities=mapping_matches_grouped_communities,
        has_empty_communities=bool(empty_community_ids),
        missing_nodes=missing_nodes,
        conflicting_nodes=tuple(sorted(conflicting_nodes, key=_sort_key)),
        extra_nodes=extra_nodes,
        empty_community_ids=empty_community_ids,
    )


def validate_communities(
    graph: nx.Graph,
    community_id_by_node: dict[Any, int],
    communities: dict[int, Sequence[Any]],
) -> CommunityValidationReport:
    """Validate that all graph nodes are assigned to exactly one non-empty community."""

    return validate_community_assignments(
        graph=graph,
        community_id_by_node=community_id_by_node,
        communities=communities,
    )


def _raise_for_invalid_validation(validation: CommunityValidationReport) -> None:
    if validation.has_empty_communities:
        raise ValueError(
            f"Community detection produced empty communities: {list(validation.empty_community_ids)}."
        )
    if validation.missing_nodes:
        raise ValueError(
            f"Community detection missed graph nodes: {list(validation.missing_nodes[:5])}."
        )
    if validation.conflicting_nodes:
        raise ValueError(
            f"Community detection produced conflicting assignments for nodes: "
            f"{list(validation.conflicting_nodes[:5])}."
        )
    if validation.extra_nodes:
        raise ValueError(
            f"Community detection produced nodes not present in the graph: {list(validation.extra_nodes[:5])}."
        )
    if not validation.mapping_matches_grouped_communities:
        raise ValueError("community_id_by_node does not match grouped communities.")


def _build_community_stats(communities: dict[int, tuple[Any, ...]]) -> CommunityStats:
    if not communities:
        raise ValueError("Community detection produced no communities.")

    community_sizes = {
        community_id: len(nodes)
        for community_id, nodes in communities.items()
    }
    sizes = list(community_sizes.values())
    return CommunityStats(
        num_communities=len(communities),
        community_sizes=community_sizes,
        largest_community_size=max(sizes),
        smallest_community_size=min(sizes),
        average_community_size=float(np.mean(sizes)),
    )


def get_community_stats(
    communities: dict[int, Sequence[Any]] | Iterable[Iterable[Any]],
) -> CommunityStats:
    """Compute basic community size statistics from grouped communities."""

    return _build_community_stats(_coerce_communities(communities))


def compute_community_quality_metrics(
    graph: nx.Graph,
    result: CommunityDetectionResult,
) -> CommunityQualityMetrics:
    """Compute compact community quality metrics for reporting."""

    work_graph = graph if not graph.is_directed() else graph.to_undirected()
    community_sets = [set(nodes) for nodes in result.communities.values()]
    modularity = 0.0
    if work_graph.number_of_edges() > 0:
        modularity = float(nx.community.modularity(work_graph, community_sets))

    sizes = np.asarray(list(result.stats.community_sizes.values()), dtype=float)
    return CommunityQualityMetrics(
        modularity=modularity,
        num_communities=result.stats.num_communities,
        largest_community_size=result.stats.largest_community_size,
        smallest_community_size=result.stats.smallest_community_size,
        average_community_size=result.stats.average_community_size,
        community_size_std=float(np.std(sizes)),
    )


def _build_community_result(
    graph: nx.Graph,
    method: str,
    raw_communities: Iterable[Iterable[Any]],
    runtime_seconds: float,
) -> CommunityDetectionResult:
    communities = _normalize_communities(raw_communities)
    community_id_by_node: dict[Any, int] = {}
    for community_id, nodes in communities.items():
        for node_id in nodes:
            if node_id not in community_id_by_node:
                community_id_by_node[node_id] = community_id

    validation = validate_communities(
        graph=graph,
        community_id_by_node=community_id_by_node,
        communities=communities,
    )
    _raise_for_invalid_validation(validation)

    return CommunityDetectionResult(
        method=method,
        community_id_by_node=community_id_by_node,
        communities=communities,
        stats=_build_community_stats(communities),
        validation=validation,
        runtime_seconds=runtime_seconds,
    )


def community_result_from_clustering(
    graph: nx.Graph,
    clustering_result: ClusteringResult,
) -> CommunityDetectionResult:
    """Adapt a generic clustering result into the validated community contract."""

    communities = {
        int(community_id): tuple(sorted(nodes, key=_sort_key))
        for community_id, nodes in sorted(clustering_result.clusters.items())
    }
    community_id_by_node = {
        node_id: int(community_id)
        for node_id, community_id in clustering_result.cluster_id_by_node.items()
    }
    validation = validate_communities(
        graph=graph,
        community_id_by_node=community_id_by_node,
        communities=communities,
    )
    _raise_for_invalid_validation(validation)
    metadata = dict(clustering_result.metadata)
    if clustering_result.modularity is not None:
        metadata["modularity"] = float(clustering_result.modularity)
    if clustering_result.mean_conductance is not None:
        metadata["mean_conductance"] = float(clustering_result.mean_conductance)
    if clustering_result.nmi is not None:
        metadata["nmi"] = float(clustering_result.nmi)
    if clustering_result.ari is not None:
        metadata["ari"] = float(clustering_result.ari)
    return CommunityDetectionResult(
        method=clustering_result.method,
        community_id_by_node=community_id_by_node,
        communities=communities,
        stats=_build_community_stats(communities),
        validation=validation,
        runtime_seconds=float(clustering_result.runtime_seconds),
        category=clustering_result.category,
        requested_input_mode=clustering_result.requested_input_mode,
        resolved_input_mode=clustering_result.resolved_input_mode,
        metadata=metadata,
    )


def _to_igraph(
    graph: nx.Graph,
    weight_attribute: str | None,
) -> tuple["ig.Graph", list[Any], list[float] | None]:
    if ig is None:
        raise ImportError(
            "Leiden community detection requires the 'igraph' package, but it is not installed."
        )

    nodes = sorted(graph.nodes(), key=_sort_key)
    node_to_index = {node_id: index for index, node_id in enumerate(nodes)}
    edges = [(node_to_index[source], node_to_index[target]) for source, target in graph.edges()]
    weights: list[float] | None = None
    if weight_attribute is not None:
        weights = [
            float(graph[source][target].get(weight_attribute, 1.0))
            for source, target in graph.edges()
        ]

    ig_graph = ig.Graph(
        n=len(nodes),
        edges=edges,
        directed=graph.is_directed(),
    )
    return ig_graph, nodes, weights


def detect_communities(
    graph: nx.Graph,
    method: str = "louvain",
    seed: int = 42,
    resolution: float = 1.0,
    weight_attribute: str | None = None,
    random_seed: int | None = None,
    embeddings: pd.DataFrame | np.ndarray | Sequence[Sequence[float]] | None = None,
    features: pd.DataFrame | np.ndarray | Sequence[Sequence[float]] | None = None,
    input_mode: str = "graph",
    config: dict[str, Any] | None = None,
) -> CommunityDetectionResult:
    """Detect and validate graph communities with a minimal wrapper."""

    if graph.number_of_nodes() == 0:
        raise ValueError("Community detection requires a graph with at least one node.")

    effective_seed = seed if random_seed is None else random_seed
    method_key = method.lower()
    method_config = dict(config or {})
    method_config.setdefault("resolution", float(resolution))
    if weight_attribute is not None:
        method_config.setdefault("weight_attribute", weight_attribute)
    clustering_result = cluster_nodes(
        graph,
        method_key,
        embeddings=embeddings,
        features=features,
        config=method_config,
        random_seed=int(effective_seed),
        input_mode=input_mode,
    )
    return community_result_from_clustering(graph, clustering_result)


def get_node_community(result: CommunityDetectionResult, node_id: Any) -> int:
    """Return the validated community id for a graph node."""

    if node_id not in result.community_id_by_node:
        raise KeyError(f"Node {node_id!r} is not present in the community assignment.")
    return result.community_id_by_node[node_id]


def get_nodes_in_community(result: CommunityDetectionResult, community_id: int) -> tuple[Any, ...]:
    """Return the nodes that belong to one community."""

    if community_id not in result.communities:
        raise KeyError(f"Community {community_id!r} is not present in the community assignment.")
    return result.communities[community_id]


def get_community_sizes(result: CommunityDetectionResult) -> dict[int, int]:
    """Return the validated community sizes."""

    return dict(result.stats.community_sizes)


def _rng_choice_index(
    rng: Any,
    size: int,
    probabilities: np.ndarray | None = None,
) -> int:
    if size < 1:
        raise ValueError("Cannot sample from an empty population.")

    if isinstance(rng, np.random.Generator):
        population = np.arange(size)
        return int(rng.choice(population, p=probabilities))

    if isinstance(rng, py_random.Random):
        population = list(range(size))
        if probabilities is None:
            return int(rng.choice(population))
        return int(rng.choices(population, weights=probabilities.tolist(), k=1)[0])

    if hasattr(rng, "choice"):
        population = np.arange(size)
        if probabilities is None:
            return int(rng.choice(population))
        return int(rng.choice(population, p=probabilities))

    raise TypeError(
        "rng must be a numpy.random.Generator, random.Random, or another object exposing choice()."
    )


def score_communities_by_size(
    communities: dict[int, Sequence[Any]] | Iterable[Iterable[Any]],
) -> dict[int, float]:
    """Return a simple size-based score for each community."""

    normalized = _coerce_communities(communities)
    return {
        community_id: float(len(nodes))
        for community_id, nodes in normalized.items()
    }


def sample_community(
    communities: dict[int, Sequence[Any]] | Iterable[Iterable[Any]],
    community_sizes: dict[int, int],
    rng: Any,
) -> int:
    """Sample one community proportionally to community size using the provided RNG."""

    normalized = _coerce_communities(communities)
    if not normalized:
        raise ValueError("communities must contain at least one community.")

    community_ids = sorted(normalized)
    missing_sizes = [community_id for community_id in community_ids if community_id not in community_sizes]
    if missing_sizes:
        raise ValueError(
            f"community_sizes is missing entries for communities: {missing_sizes[:5]}."
        )

    weights = np.asarray(
        [float(community_sizes[community_id]) for community_id in community_ids],
        dtype=float,
    )
    if np.any(weights <= 0.0):
        raise ValueError("community_sizes must contain only positive sizes.")
    weights = weights / weights.sum()

    return int(community_ids[_rng_choice_index(rng, size=len(community_ids), probabilities=weights)])


def sample_node_from_community(
    community_nodes: Sequence[Any],
    rng: Any,
) -> Any:
    """Sample one node from a community using the provided RNG."""

    normalized_nodes = tuple(sorted(tuple(community_nodes), key=_sort_key))
    if not normalized_nodes:
        raise ValueError("community_nodes must contain at least one node.")
    return normalized_nodes[_rng_choice_index(rng, size=len(normalized_nodes))]


def choose_communities(
    result: CommunityDetectionResult,
    count: int,
    mode: str = "proportional_size",
    random_seed: int = 42,
    community_scores: dict[int, float] | None = None,
) -> tuple[int, ...]:
    """Choose communities deterministically from a validated partition."""

    if count < 1:
        raise ValueError("count must be at least 1.")
    if count > result.stats.num_communities:
        raise ValueError("count cannot exceed the number of communities.")

    community_ids = sorted(result.communities)
    if mode == "proportional_size":
        weights = np.asarray(
            [float(result.stats.community_sizes[community_id]) for community_id in community_ids],
            dtype=float,
        )
        weights = weights / weights.sum()
        rng = np.random.default_rng(random_seed)
        chosen = rng.choice(np.asarray(community_ids, dtype=int), size=count, replace=False, p=weights)
        return tuple(sorted(int(community_id) for community_id in chosen.tolist()))

    if mode == "community_score":
        if community_scores is None:
            raise ValueError("community_scores is required when mode='community_score'.")
        missing_scores = [community_id for community_id in community_ids if community_id not in community_scores]
        if missing_scores:
            raise ValueError(
                f"community_scores is missing scores for communities: {missing_scores[:5]}."
            )
        ranked = sorted(
            community_ids,
            key=lambda community_id: (-float(community_scores[community_id]), community_id),
        )
        return tuple(ranked[:count])

    raise ValueError(
        f"Unsupported community choice mode '{mode}'. Supported modes: proportional_size, community_score."
    )


def sample_candidate_nodes_by_community(
    result: CommunityDetectionResult,
    community_ids: Sequence[int] | None = None,
    nodes_per_community: int = 1,
    random_seed: int = 42,
    node_scores: dict[Any, float] | None = None,
) -> dict[int, tuple[Any, ...]]:
    """Sample or rank candidate nodes within selected communities."""

    if nodes_per_community < 1:
        raise ValueError("nodes_per_community must be at least 1.")

    selected_community_ids = (
        tuple(sorted(result.communities))
        if community_ids is None
        else tuple(sorted(dict.fromkeys(community_ids)))
    )
    for community_id in selected_community_ids:
        if community_id not in result.communities:
            raise KeyError(f"Community {community_id!r} is not present in the community assignment.")

    rng = np.random.default_rng(random_seed)
    sampled_nodes: dict[int, tuple[Any, ...]] = {}
    for community_id in selected_community_ids:
        community_nodes = result.communities[community_id]
        if nodes_per_community > len(community_nodes):
            raise ValueError(
                f"nodes_per_community={nodes_per_community} exceeds size of community {community_id}."
            )

        if node_scores is None:
            chosen = rng.choice(
                np.asarray(community_nodes, dtype=object),
                size=nodes_per_community,
                replace=False,
            ).tolist()
            sampled_nodes[community_id] = tuple(sorted(chosen, key=_sort_key))
            continue

        ranked_nodes = sorted(
            community_nodes,
            key=lambda node_id: (-float(node_scores.get(node_id, 0.0)), _sort_key(node_id)),
        )
        sampled_nodes[community_id] = tuple(ranked_nodes[:nodes_per_community])

    return sampled_nodes


def format_community_debug_report(
    result: CommunityDetectionResult,
    max_items: int = 5,
) -> str:
    """Render a compact debug report for validated community assignments."""

    first_sizes = list(result.stats.community_sizes.items())[:max_items]
    first_assignments = list(sorted(result.community_id_by_node.items(), key=lambda item: _sort_key(item[0])))[:max_items]
    lines = [
        f"method: {result.method}",
        f"num_communities: {result.stats.num_communities}",
        f"largest_community_size: {result.stats.largest_community_size}",
        f"smallest_community_size: {result.stats.smallest_community_size}",
        f"average_community_size: {result.stats.average_community_size:.6f}",
        f"first_community_sizes: {first_sizes}",
        f"first_node_assignments: {first_assignments}",
        f"every_node_assigned_exactly_once: {result.validation.every_node_assigned_exactly_once}",
        f"mapping_matches_grouped_communities: {result.validation.mapping_matches_grouped_communities}",
        f"has_empty_communities: {result.validation.has_empty_communities}",
    ]
    return "\n".join(lines)
