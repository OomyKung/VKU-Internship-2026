"""Node feature extraction utilities for graph mining experiments."""

from __future__ import annotations

import math
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

from .community_detection import CommunityDetectionResult


def _shannon_entropy(values: list[object]) -> float:
    # Entropy measures how mixed a node's neighborhood is by attribute value.
    if not values:
        return 0.0

    counts = Counter(values)
    total = float(sum(counts.values()))
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability + 1e-12, 2)
    return entropy


def _minority_values(graph: nx.Graph, protected_attribute: str | None) -> set[object]:
    # Treat the least frequent protected-attribute values as minority groups.
    if not protected_attribute:
        return set()

    values = [
        attrs.get(protected_attribute)
        for _, attrs in graph.nodes(data=True)
        if protected_attribute in attrs and attrs.get(protected_attribute) is not None
    ]
    if not values:
        return set()

    counts = Counter(values)
    min_count = min(counts.values())
    return {value for value, count in counts.items() if count == min_count}


def compute_node_features(
    graph: nx.Graph,
    community_result: CommunityDetectionResult,
    protected_attribute: str | None = None,
) -> pd.DataFrame:
    """Compute structural, community, and neighborhood fairness features."""

    partition = community_result.partition
    communities = community_result.communities
    # Community size is reused often, so precompute it once.
    community_sizes = {idx: len(nodes) for idx, nodes in enumerate(communities)}
    minority_values = _minority_values(graph, protected_attribute)

    # Compute standard centrality measures once for the full graph.
    degree = dict(graph.degree())
    pagerank = nx.pagerank(graph)
    betweenness = nx.betweenness_centrality(graph)
    clustering = nx.clustering(graph)

    records: list[dict[str, object]] = []
    for node in graph.nodes():
        node_community = partition[node]
        neighbors = list(graph.neighbors(node))

        within_community_degree = 0
        cross_community_degree = 0
        neighboring_communities: set[int] = set()
        neighbor_attribute_values: list[object] = []

        for neighbor in neighbors:
            neighbor_community = partition[neighbor]
            if neighbor_community == node_community:
                within_community_degree += 1
            else:
                cross_community_degree += 1
            neighboring_communities.add(neighbor_community)

            if protected_attribute and protected_attribute in graph.nodes[neighbor]:
                neighbor_attribute_values.append(graph.nodes[neighbor][protected_attribute])

        # Exclude the node's own community when counting neighboring communities.
        if node_community in neighboring_communities:
            neighboring_communities.remove(node_community)

        minority_neighbor_ratio = 0.0
        if neighbor_attribute_values:
            # This feature captures whether the node is positioned near minority groups.
            minority_neighbor_ratio = (
                sum(1 for value in neighbor_attribute_values if value in minority_values)
                / len(neighbor_attribute_values)
            )

        record = {
            "node_id": node,
            "community_id": node_community,
            "degree": degree[node],
            "pagerank": pagerank[node],
            "betweenness": betweenness[node],
            "clustering_coefficient": clustering[node],
            "community_size": community_sizes[node_community],
            "within_community_degree": within_community_degree,
            "cross_community_degree": cross_community_degree,
            "neighboring_communities": len(neighboring_communities),
            "neighborhood_attribute_entropy": _shannon_entropy(neighbor_attribute_values),
            "minority_neighbor_ratio": minority_neighbor_ratio,
        }

        if protected_attribute:
            record[protected_attribute] = graph.nodes[node].get(protected_attribute)

        records.append(record)

    frame = pd.DataFrame(records).set_index("node_id", drop=False)
    numeric_columns = [
        "degree",
        "pagerank",
        "betweenness",
        "clustering_coefficient",
        "community_size",
        "within_community_degree",
        "cross_community_degree",
        "neighboring_communities",
        "neighborhood_attribute_entropy",
        "minority_neighbor_ratio",
    ]
    # Replace non-finite values so later ML code can assume clean numeric inputs.
    frame[numeric_columns] = frame[numeric_columns].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return frame
