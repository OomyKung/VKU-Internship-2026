"""Community detection wrappers for FIM experiments."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

try:
    import igraph as ig
except ImportError:
    ig = None

import networkx as nx

from .config import CommunityConfig


@dataclass(slots=True)
class CommunityDetectionResult:
    """Community detection output."""

    # `partition` stores node -> community id and `communities` stores the
    # inverse mapping as explicit node lists.
    method: str
    partition: dict[object, int]
    communities: list[list[object]]
    runtime_seconds: float


def _communities_to_partition(communities: Iterable[Iterable[object]]) -> tuple[dict[object, int], list[list[object]]]:
    # Normalize backend-specific outputs into one shared representation.
    normalized: list[list[object]] = [sorted(list(group)) for group in communities if group]
    partition: dict[object, int] = {}
    for idx, group in enumerate(normalized):
        for node in group:
            partition[node] = idx
    return partition, normalized


def _to_igraph(graph: nx.Graph) -> tuple["ig.Graph", list[object]]:
    # Leiden and Infomap are routed through igraph because it exposes robust
    # implementations for both algorithms.
    if ig is None:
        raise ImportError("igraph is required for the selected community detection method.")

    nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]
    ig_graph = ig.Graph(
        n=len(nodes),
        edges=edges,
        directed=graph.is_directed(),
    )
    return ig_graph, nodes


def detect_communities(graph: nx.Graph, config: CommunityConfig) -> CommunityDetectionResult:
    """Detect graph communities using the requested method."""

    method = config.method.lower()
    # Use an undirected working view to keep method comparisons simple.
    work_graph = graph if not graph.is_directed() else graph.to_undirected()
    start = perf_counter()

    if method == "louvain":
        communities = nx.community.louvain_communities(
            work_graph,
            weight=config.weight_attribute,
            resolution=config.resolution,
            seed=config.seed,
        )
        partition, normalized = _communities_to_partition(communities)
    elif method in {"label_propagation", "label-propagation", "lp"}:
        # Lightweight community baseline with essentially no tuning.
        communities = nx.community.label_propagation_communities(work_graph)
        partition, normalized = _communities_to_partition(communities)
    elif method == "leiden":
        ig_graph, nodes = _to_igraph(work_graph)
        # Use modularity here so the comparison with Louvain remains intuitive.
        clustering = ig_graph.community_leiden(
            objective_function="modularity",
            resolution=config.resolution,
            n_iterations=-1,
        )
        partition = {nodes[idx]: comm_idx for idx, comm_idx in enumerate(clustering.membership)}
        normalized = [sorted([nodes[idx] for idx in group]) for group in clustering]
    elif method == "infomap":
        ig_graph, nodes = _to_igraph(work_graph)
        # Infomap gives a contrasting flow-based view of community structure.
        clustering = ig_graph.community_infomap(edge_weights=None, vertex_weights=None)
        partition = {nodes[idx]: comm_idx for idx, comm_idx in enumerate(clustering.membership)}
        normalized = [sorted([nodes[idx] for idx in group]) for group in clustering]
    else:
        raise ValueError(
            f"Unsupported community detection method '{config.method}'. "
            "Supported methods: louvain, leiden, label_propagation, infomap."
        )

    runtime_seconds = perf_counter() - start
    return CommunityDetectionResult(
        method=config.method,
        partition=partition,
        communities=normalized,
        runtime_seconds=runtime_seconds,
    )
