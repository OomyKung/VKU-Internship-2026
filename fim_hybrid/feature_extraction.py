"""Lightweight structural candidate scoring for community-aware FIM search."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .community_detection import CommunityDetectionResult
from .data_loader import LoadedDataset, ProtectedGroupReport


def _normalize_series(values: pd.Series) -> pd.Series:
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values.astype(float) - minimum) / (maximum - minimum)


def compute_node_features(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    community_result: CommunityDetectionResult,
) -> pd.DataFrame:
    """Compute deterministic structural and fairness-aware node features."""

    if dataset.name != protected_group_report.dataset_name:
        raise ValueError("protected_group_report.dataset_name must match dataset.name.")
    if set(dataset.graph.nodes()) != set(community_result.community_id_by_node):
        raise ValueError("community_result must assign every graph node before feature extraction.")

    graph = dataset.graph
    work_graph = graph if not graph.is_directed() else graph.to_undirected()
    group_by_node = {
        node_id: group_name
        for group_name, node_ids in protected_group_report.protected_groups.items()
        for node_id in node_ids
    }
    pagerank_scores = nx.pagerank(graph)

    records: list[dict[str, Any]] = []
    for node_id in sorted(graph.nodes(), key=lambda value: (type(value).__name__, repr(value))):
        community_id = community_result.community_id_by_node[node_id]
        neighbors = list(work_graph.neighbors(node_id))
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
        group_name = group_by_node[node_id]
        group_size = protected_group_report.group_sizes[group_name]
        community_size = community_result.stats.community_sizes[community_id]

        records.append(
            {
                "node_id": node_id,
                "community_id": community_id,
                "protected_group": group_name,
                "degree": float(work_graph.degree(node_id)),
                "pagerank": float(pagerank_scores[node_id]),
                "cross_community_degree": float(cross_community_degree),
                "neighboring_communities": float(len(neighboring_communities)),
                "inverse_community_size": 1.0 / float(community_size),
                "inverse_group_size": 1.0 / float(group_size),
            }
        )

    frame = pd.DataFrame(records).set_index("node_id", drop=False)
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
