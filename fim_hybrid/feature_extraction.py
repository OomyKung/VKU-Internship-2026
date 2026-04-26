"""Lightweight structural candidate scoring for community-aware FIM search."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .community_detection import CommunityDetectionResult
from .data_loader import LoadedDataset, ProtectedGroupReport
from .node2vec_embeddings import Node2VecConfig, generate_node2vec_embeddings


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _normalize_series(values: pd.Series) -> pd.Series:
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values.astype(float) - minimum) / (maximum - minimum)


def _compute_betweenness_centrality(graph: nx.Graph) -> dict[Any, float]:
    node_count = graph.number_of_nodes()
    if node_count <= 200:
        return nx.betweenness_centrality(graph, normalized=True)

    sample_size = min(100, node_count)
    return nx.betweenness_centrality(graph, k=sample_size, normalized=True, seed=42)


def _normalized_entropy(values: list[str], num_groups: int) -> float:
    if not values or num_groups <= 1:
        return 0.0

    counts = Counter(values)
    total = float(sum(counts.values()))
    entropy = 0.0
    for count in counts.values():
        probability = float(count) / total
        entropy -= probability * math.log(probability)

    normalizer = math.log(float(num_groups))
    if normalizer <= 0.0:
        return 0.0
    return float(entropy / normalizer)


def _safe_feature_token(value: object) -> str:
    token = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(value).strip()
    )
    return "_".join(part for part in token.split("_") if part) or "group"


def compute_node_features(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    community_result: CommunityDetectionResult,
    node2vec_config: Node2VecConfig | None = None,
    node2vec_cache_path: Path | None = None,
    use_community_features_for_ml: bool = True,
    community_feature_mode: str = "basic",
) -> pd.DataFrame:
    """Compute deterministic structural and fairness-aware node features."""

    resolved_mode = "none" if not bool(use_community_features_for_ml) else str(community_feature_mode).strip().lower()
    if resolved_mode not in {"none", "basic", "full"}:
        raise ValueError("community_feature_mode must be one of ['none', 'basic', 'full'].")
    if dataset.name != protected_group_report.dataset_name:
        raise ValueError("protected_group_report.dataset_name must match dataset.name.")
    if set(dataset.graph.nodes()) != set(community_result.community_id_by_node):
        raise ValueError("community_result must assign every graph node before feature extraction.")

    graph = dataset.graph
    work_graph = graph if not graph.is_directed() else graph.to_undirected()
    node_count = graph.number_of_nodes()
    max_degree_denominator = float(max(node_count - 1, 1))
    group_by_node = {
        node_id: group_name
        for group_name, node_ids in protected_group_report.protected_groups.items()
        for node_id in node_ids
    }
    num_groups = len(protected_group_report.group_sizes)
    global_group_frequency = {
        group_name: float(group_size) / float(node_count)
        for group_name, group_size in protected_group_report.group_sizes.items()
    }
    minority_frequency_threshold = 1.0 / float(max(num_groups, 1))
    community_group_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for node_id, community_id in community_result.community_id_by_node.items():
        community_group_counts[community_id][group_by_node[node_id]] += 1

    undercovered_groups_by_community: dict[int, set[str]] = {}
    community_group_fractions: dict[int, dict[str, float]] = {}
    community_group_entropies: dict[int, float] = {}
    for community_id, group_counts in community_group_counts.items():
        community_size = float(community_result.stats.community_sizes[community_id])
        community_group_fractions[community_id] = {
            group_name: float(group_counts.get(group_name, 0)) / community_size
            for group_name in protected_group_report.group_sizes
        }
        community_group_entropies[community_id] = _normalized_entropy(
            [
                group_name
                for group_name, count in group_counts.items()
                for _ in range(int(count))
            ],
            num_groups,
        )
        undercovered_groups_by_community[community_id] = {
            group_name
            for group_name, global_frequency in global_group_frequency.items()
            if (float(group_counts.get(group_name, 0)) / community_size) < global_frequency
        }
    community_size_rank = {
        community_id: rank
        for rank, (community_id, _) in enumerate(
            sorted(
                community_result.stats.community_sizes.items(),
                key=lambda item: (-int(item[1]), int(item[0])),
            ),
            start=1,
        )
    }

    pagerank_scores = nx.pagerank(graph)
    betweenness_scores = _compute_betweenness_centrality(work_graph)
    clustering_coefficients = nx.clustering(work_graph)

    records: list[dict[str, Any]] = []
    for node_id in sorted(graph.nodes(), key=_sort_key):
        community_id = community_result.community_id_by_node[node_id]
        neighbors = list(work_graph.neighbors(node_id))
        degree = float(work_graph.degree(node_id))
        neighboring_communities = {
            community_result.community_id_by_node[neighbor_id]
            for neighbor_id in neighbors
            if community_result.community_id_by_node[neighbor_id] != community_id
        }
        cross_community_degree = sum(
            1
            for neighbor_id in neighbors
            if community_result.community_id_by_node[neighbor_id] != community_id
        )
        within_community_degree = len(neighbors) - cross_community_degree
        group_name = group_by_node[node_id]
        group_size = protected_group_report.group_sizes[group_name]
        community_size = community_result.stats.community_sizes[community_id]
        neighbor_group_values = [group_by_node[neighbor_id] for neighbor_id in neighbors]
        undercovered_groups = undercovered_groups_by_community[community_id]
        undercovered_neighbor_count = sum(
            1
            for neighbor_id in neighbors
            if group_by_node[neighbor_id] in undercovered_groups
        )

        row = {
                "node_id": node_id,
                "community_id": community_id,
                "community_size": float(community_size),
                "protected_group": group_name,
                "degree": degree,
                "normalized_degree": degree / max_degree_denominator,
                "pagerank": float(pagerank_scores[node_id]),
                "betweenness": float(betweenness_scores[node_id]),
                "clustering_coefficient": float(clustering_coefficients[node_id]),
                "within_community_degree": float(within_community_degree),
                "cross_community_degree": float(cross_community_degree),
                "neighboring_communities": float(len(neighboring_communities)),
                "protected_group_frequency": float(group_size) / float(node_count),
                "minority_group_indicator": float(
                    global_group_frequency[group_name] < minority_frequency_threshold
                ),
                "neighborhood_group_entropy": _normalized_entropy(neighbor_group_values, num_groups),
                "fraction_neighbors_in_undercovered_groups": (
                    float(undercovered_neighbor_count) / float(len(neighbors))
                    if neighbors
                    else 0.0
                ),
                "inverse_community_size": 1.0 / float(community_size),
                "inverse_group_size": 1.0 / float(group_size),
            }
        if resolved_mode == "full":
            row["community_size_rank"] = float(community_size_rank[community_id])
            row["community_protected_group_entropy"] = float(community_group_entropies[community_id])
            for group_label in protected_group_report.group_sizes:
                token = _safe_feature_token(group_label)
                row[f"community_group_fraction_{token}"] = float(
                    community_group_fractions[community_id].get(group_label, 0.0)
                )
        records.append(row)

    frame = pd.DataFrame(records).set_index("node_id", drop=False)
    if node2vec_config is not None:
        embedding_result = generate_node2vec_embeddings(
            graph=graph,
            config=node2vec_config,
            cache_path=node2vec_cache_path,
        )
        embedding_frame = embedding_result.embedding_frame.set_index("node_id", drop=False)
        frame = frame.join(
            embedding_frame.drop(columns=["node_id"]),
            how="left",
            validate="one_to_one",
        )
        if frame.filter(like="node2vec_").isna().any().any():
            raise ValueError("Node2Vec embeddings must be available for every node.")

    numeric_columns = [
        "degree",
        "pagerank",
        "cross_community_degree",
        "neighboring_communities",
        "inverse_community_size",
        "inverse_group_size",
    ]
    normalized = {
        column_name: _normalize_series(frame[column_name])
        for column_name in numeric_columns
    }
    frame["structural_score"] = (
        0.30 * normalized["pagerank"]
        + 0.25 * normalized["degree"]
        + 0.20 * normalized["cross_community_degree"]
        + 0.15 * normalized["neighboring_communities"]
        + 0.05 * normalized["inverse_community_size"]
        + 0.05 * normalized["inverse_group_size"]
    )
    if resolved_mode == "none":
        explicit_community_columns = [
            "community_id",
            "community_size",
            "within_community_degree",
            "cross_community_degree",
            "neighboring_communities",
            "inverse_community_size",
        ]
        frame = frame.drop(columns=[column for column in explicit_community_columns if column in frame.columns])
    return frame


def compute_structural_node_scores(feature_frame: pd.DataFrame) -> dict[Any, float]:
    """Return deterministic node scores from a feature frame."""

    required_columns = {"node_id", "structural_score"}
    missing = required_columns.difference(feature_frame.columns)
    if missing:
        raise ValueError(f"feature_frame is missing required columns: {sorted(missing)}.")
    if feature_frame["node_id"].duplicated().any():
        raise ValueError("feature_frame contains duplicate node_id values.")

    return {
        row.node_id: float(row.structural_score)
        for row in feature_frame.itertuples(index=False)
    }
