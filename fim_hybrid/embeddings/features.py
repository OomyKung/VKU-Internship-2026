"""Benchmark-specific feature preparation for embedding methods."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.preprocessing import StandardScaler

from fim_hybrid.data_loader import LoadedDataset

from .base import sort_key, sorted_node_ids


@dataclass(slots=True)
class PreparedFeatures:
    """Prepared node features aligned to one deterministic node ordering."""

    node_order: tuple[Any, ...]
    feature_frame: pd.DataFrame
    feature_matrix: np.ndarray
    feature_columns: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def _compute_betweenness(graph: nx.Graph, random_seed: int) -> dict[Any, float]:
    node_count = graph.number_of_nodes()
    if node_count <= 200:
        return nx.betweenness_centrality(graph, normalized=True)

    sample_size = min(100, node_count)
    return nx.betweenness_centrality(graph, k=sample_size, normalized=True, seed=random_seed)


def _safe_core_numbers(graph: nx.Graph) -> dict[Any, float]:
    if graph.number_of_edges() == 0:
        return {node_id: 0.0 for node_id in graph.nodes()}
    return {node_id: float(value) for node_id, value in nx.core_number(graph).items()}


def _structural_feature_frame(graph: nx.Graph, node_order: Sequence[Any], random_seed: int) -> pd.DataFrame:
    node_count = max(int(graph.number_of_nodes()), 1)
    pagerank = nx.pagerank(graph)
    clustering = nx.clustering(graph)
    betweenness = _compute_betweenness(graph, random_seed=random_seed)
    core_numbers = _safe_core_numbers(graph)
    rows: list[dict[str, float | Any]] = []
    max_degree = max((int(graph.degree(node_id)) for node_id in node_order), default=1)
    for node_id in node_order:
        degree = float(graph.degree(node_id))
        rows.append(
            {
                "node_id": node_id,
                "degree": degree,
                "normalized_degree": degree / float(max(max_degree, 1)),
                "pagerank": float(pagerank[node_id]),
                "clustering_coefficient": float(clustering[node_id]),
                "betweenness": float(betweenness[node_id]),
                "core_number": float(core_numbers[node_id]),
                "is_directed_graph": float(graph.is_directed()),
                "graph_density": float(nx.density(graph)) if node_count > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _encode_node_attributes(dataset: LoadedDataset, node_order: Sequence[Any]) -> pd.DataFrame:
    attribute_frame = dataset.node_attributes.copy()
    if "node_id" not in attribute_frame.columns:
        raise ValueError("dataset.node_attributes must contain a node_id column.")
    aligned = attribute_frame.set_index("node_id", drop=False).reindex(list(node_order))

    encoded_parts: list[pd.DataFrame] = [pd.DataFrame(index=range(len(node_order)))]
    for column_name in aligned.columns:
        if column_name == "node_id":
            continue
        series = aligned[column_name]
        if is_numeric_dtype(series) or is_bool_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            fill_value = float(numeric.median()) if numeric.notna().any() else 0.0
            encoded_parts.append(
                pd.DataFrame(
                    {f"attr__{column_name}": numeric.fillna(fill_value).astype(float).to_numpy()},
                    index=range(len(node_order)),
                )
            )
            continue

        categories = series.where(series.notna(), "__missing__").astype(str)
        dummies = pd.get_dummies(categories, prefix=f"attr__{column_name}", dtype=float)
        encoded_parts.append(dummies.reset_index(drop=True))

    encoded = pd.concat(encoded_parts, axis=1)
    encoded.insert(0, "node_id", list(node_order))
    return encoded


def _scale_feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...]]:
    feature_columns = tuple(column for column in frame.columns if column != "node_id")
    if not feature_columns:
        raise ValueError("Prepared features must contain at least one numeric feature column.")
    matrix = frame.loc[:, feature_columns].to_numpy(dtype=float, copy=True)
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(matrix)
    scaled_frame = pd.DataFrame(scaled_matrix, columns=list(feature_columns))
    scaled_frame.insert(0, "node_id", frame["node_id"].tolist())
    return scaled_frame, scaled_matrix, feature_columns


def prepare_benchmark_features(
    dataset: LoadedDataset,
    graph: nx.Graph | None = None,
    *,
    random_seed: int = 42,
) -> PreparedFeatures:
    """Build structural-plus-attribute features for embedding methods that need them."""

    work_graph = dataset.graph if graph is None else graph
    node_order = sorted_node_ids(work_graph)
    structural = _structural_feature_frame(work_graph, node_order=node_order, random_seed=random_seed)
    encoded_attributes = _encode_node_attributes(dataset, node_order=node_order)
    combined = structural.merge(encoded_attributes, on="node_id", how="left", validate="one_to_one")
    if combined.isna().any().any():
        raise ValueError("Prepared benchmark features contain missing values.")
    scaled_frame, scaled_matrix, feature_columns = _scale_feature_frame(combined)
    return PreparedFeatures(
        node_order=node_order,
        feature_frame=scaled_frame,
        feature_matrix=scaled_matrix,
        feature_columns=feature_columns,
        metadata={
            "uses_structural_features": True,
            "uses_node_attributes": len(dataset.node_attributes.columns) > 1,
        },
    )


def coerce_input_features(
    features: PreparedFeatures | pd.DataFrame | np.ndarray | Sequence[Sequence[float]] | None,
    node_order: Sequence[Any],
) -> PreparedFeatures | None:
    """Coerce custom features into the shared prepared-features structure."""

    if features is None:
        return None

    ordered_nodes = tuple(node_order)
    if isinstance(features, PreparedFeatures):
        if tuple(features.node_order) != ordered_nodes:
            raise ValueError("PreparedFeatures.node_order must match the graph node ordering.")
        return features

    if isinstance(features, pd.DataFrame):
        frame = features.copy()
        if "node_id" not in frame.columns:
            if len(frame) != len(ordered_nodes):
                raise ValueError("Feature frames without node_id must already match the graph node ordering.")
            frame.insert(0, "node_id", list(ordered_nodes))
        if frame["node_id"].duplicated().any():
            raise ValueError("Feature frames must not contain duplicate node_id values.")
        missing_nodes = [node_id for node_id in ordered_nodes if node_id not in set(frame["node_id"])]
        if missing_nodes:
            raise ValueError(f"Feature frames are missing graph nodes: {missing_nodes[:5]}.")
        aligned = (
            frame.set_index("node_id", drop=False)
            .reindex(list(ordered_nodes))
            .reset_index(drop=True)
        )
        feature_columns = tuple(column for column in aligned.columns if column != "node_id")
        if not feature_columns:
            raise ValueError("Feature frames must contain at least one non-node_id column.")
        numeric = aligned.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            raise ValueError("Feature frames must contain only numeric feature columns.")
        aligned.loc[:, feature_columns] = numeric
        return PreparedFeatures(
            node_order=ordered_nodes,
            feature_frame=aligned,
            feature_matrix=numeric.to_numpy(dtype=float, copy=True),
            feature_columns=feature_columns,
        )

    array = np.asarray(features, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] != len(ordered_nodes):
        raise ValueError("Feature arrays must have shape [num_nodes, num_features].")
    frame = pd.DataFrame({"node_id": list(ordered_nodes)})
    for feature_index in range(array.shape[1]):
        frame[f"feature_{feature_index}"] = array[:, feature_index]
    return PreparedFeatures(
        node_order=ordered_nodes,
        feature_frame=frame,
        feature_matrix=array.astype(float, copy=True),
        feature_columns=tuple(column for column in frame.columns if column != "node_id"),
    )

