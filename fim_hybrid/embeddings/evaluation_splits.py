"""Deterministic split helpers for embedding evaluation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split

from .base import sort_key, sorted_node_ids


def _canonical_edge(edge: tuple[Any, Any], *, directed: bool) -> tuple[Any, Any]:
    source, target = edge
    if directed:
        return source, target
    ordered = sorted((source, target), key=sort_key)
    return ordered[0], ordered[1]


def graph_edges_for_sampling(graph: nx.Graph) -> tuple[tuple[Any, Any], ...]:
    """Return deterministic canonical graph edges without self-loops."""

    edges = {
        _canonical_edge((source, target), directed=graph.is_directed())
        for source, target in graph.edges()
        if source != target
    }
    return tuple(sorted(edges, key=lambda edge: (sort_key(edge[0]), sort_key(edge[1]))))


@dataclass(frozen=True, slots=True)
class NodeClassificationSplit:
    """Deterministic node train/test split."""

    train_node_ids: tuple[Any, ...]
    test_node_ids: tuple[Any, ...]
    stratified: bool


@dataclass(frozen=True, slots=True)
class NodeTrainValidationSplit:
    """Deterministic node train/validation split."""

    train_node_ids: tuple[Any, ...]
    validation_node_ids: tuple[Any, ...]
    stratified: bool


@dataclass(frozen=True, slots=True)
class LinkPredictionSplit:
    """Deterministic positive/negative edge train/test split."""

    train_positive_edges: tuple[tuple[Any, Any], ...]
    test_positive_edges: tuple[tuple[Any, Any], ...]
    train_negative_edges: tuple[tuple[Any, Any], ...]
    test_negative_edges: tuple[tuple[Any, Any], ...]
    directed: bool


def build_node_classification_split(
    node_ids: list[Any] | tuple[Any, ...],
    labels: list[Any] | tuple[Any, ...] | np.ndarray,
    *,
    test_fraction: float = 0.25,
    random_seed: int = 42,
    use_stratified_split: bool = True,
) -> NodeClassificationSplit:
    """Create a reproducible node train/test split."""

    normalized_node_ids = tuple(node_ids)
    if len(normalized_node_ids) != len(labels):
        raise ValueError("node_ids and labels must have the same length.")
    if len(normalized_node_ids) < 4:
        raise ValueError("Node classification requires at least four labeled nodes.")
    if not 0.0 < float(test_fraction) < 1.0:
        raise ValueError("test_fraction must be between 0.0 and 1.0.")

    label_array = np.asarray(labels, dtype=object)
    unique_labels, counts = np.unique(label_array, return_counts=True)
    if unique_labels.size < 2:
        raise ValueError("Node classification requires at least two label classes.")

    test_size = max(1, int(round(len(normalized_node_ids) * float(test_fraction))))
    can_stratify = bool(use_stratified_split and np.all(counts >= 2))
    if can_stratify:
        test_size = max(test_size, int(unique_labels.size))
    test_size = min(test_size, len(normalized_node_ids) - 1)

    train_nodes, test_nodes = train_test_split(
        list(normalized_node_ids),
        test_size=test_size,
        random_state=int(random_seed),
        stratify=label_array if can_stratify else None,
    )
    return NodeClassificationSplit(
        train_node_ids=tuple(train_nodes),
        test_node_ids=tuple(test_nodes),
        stratified=can_stratify,
    )


def build_node_train_validation_split(
    node_ids: list[Any] | tuple[Any, ...],
    labels: list[Any] | tuple[Any, ...] | np.ndarray,
    *,
    validation_fraction: float = 0.2,
    random_seed: int = 42,
    use_stratified_split: bool = True,
) -> NodeTrainValidationSplit:
    """Create a reproducible node train/validation split."""

    normalized_node_ids = tuple(node_ids)
    if len(normalized_node_ids) != len(labels):
        raise ValueError("node_ids and labels must have the same length.")
    if len(normalized_node_ids) < 2:
        raise ValueError("Node train/validation splitting requires at least two labeled nodes.")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be between 0.0 and 1.0.")

    label_array = np.asarray(labels, dtype=object)
    validation_size = max(1, int(round(len(normalized_node_ids) * float(validation_fraction))))
    unique_labels, counts = np.unique(label_array, return_counts=True)
    can_stratify = bool(
        unique_labels.size >= 2
        and use_stratified_split
        and np.all(counts >= 2)
    )
    if can_stratify and len(normalized_node_ids) - 1 >= int(unique_labels.size):
        validation_size = max(validation_size, int(unique_labels.size))
    else:
        can_stratify = False
    validation_size = min(validation_size, len(normalized_node_ids) - 1)

    train_nodes, validation_nodes = train_test_split(
        list(normalized_node_ids),
        test_size=validation_size,
        random_state=int(random_seed),
        stratify=label_array if can_stratify else None,
    )
    return NodeTrainValidationSplit(
        train_node_ids=tuple(train_nodes),
        validation_node_ids=tuple(validation_nodes),
        stratified=can_stratify,
    )


def _possible_edge_count(node_count: int, *, directed: bool) -> int:
    if directed:
        return node_count * max(node_count - 1, 0)
    return node_count * max(node_count - 1, 0) // 2


def _sample_negative_edges(
    graph: nx.Graph,
    *,
    count: int,
    random_seed: int,
) -> tuple[tuple[Any, Any], ...]:
    if count < 1:
        return ()

    node_order = sorted_node_ids(graph)
    if len(node_order) < 2:
        raise ValueError("Link prediction requires at least two graph nodes.")

    directed = graph.is_directed()
    positive_edges = set(graph_edges_for_sampling(graph))
    available_negatives = _possible_edge_count(len(node_order), directed=directed) - len(positive_edges)
    if count > available_negatives:
        raise ValueError(
            "Requested more negative edges than the graph contains non-edges. "
            f"Requested={count}, available={available_negatives}."
        )

    rng = np.random.default_rng(int(random_seed))
    sampled: set[tuple[Any, Any]] = set()
    attempts = 0
    max_attempts = max(1000, count * 50)
    while len(sampled) < count and attempts < max_attempts:
        source_index = int(rng.integers(0, len(node_order)))
        target_index = int(rng.integers(0, len(node_order)))
        if source_index == target_index:
            attempts += 1
            continue
        edge = _canonical_edge((node_order[source_index], node_order[target_index]), directed=directed)
        if edge in positive_edges or edge in sampled:
            attempts += 1
            continue
        sampled.add(edge)
        attempts += 1

    if len(sampled) != count:
        raise ValueError(
            "Unable to sample the requested number of negative edges reproducibly. "
            "The graph may be too dense for the requested negative ratio."
        )

    return tuple(sorted(sampled, key=lambda edge: (sort_key(edge[0]), sort_key(edge[1]))))


def build_link_prediction_split(
    graph: nx.Graph,
    *,
    test_fraction: float = 0.25,
    negative_ratio: float = 1.0,
    random_seed: int = 42,
) -> LinkPredictionSplit:
    """Create a reproducible positive/negative edge train/test split."""

    if not 0.0 < float(test_fraction) < 1.0:
        raise ValueError("test_fraction must be between 0.0 and 1.0.")
    if float(negative_ratio) <= 0.0:
        raise ValueError("negative_ratio must be greater than 0.0.")

    positive_edges = graph_edges_for_sampling(graph)
    if len(positive_edges) < 2:
        raise ValueError("Link prediction requires at least two positive edges.")

    rng = np.random.default_rng(int(random_seed))
    positive_indices = rng.permutation(len(positive_edges))
    test_count = max(1, int(round(len(positive_edges) * float(test_fraction))))
    test_count = min(test_count, len(positive_edges) - 1)
    test_indices = positive_indices[:test_count]
    train_indices = positive_indices[test_count:]

    train_positive_edges = tuple(positive_edges[int(index)] for index in train_indices)
    test_positive_edges = tuple(positive_edges[int(index)] for index in test_indices)
    train_negative_count = max(1, int(round(len(train_positive_edges) * float(negative_ratio))))
    test_negative_count = max(1, int(round(len(test_positive_edges) * float(negative_ratio))))
    total_negative_count = train_negative_count + test_negative_count
    negative_edges = _sample_negative_edges(
        graph,
        count=total_negative_count,
        random_seed=int(random_seed) + 1,
    )
    train_negative_edges = negative_edges[:train_negative_count]
    test_negative_edges = negative_edges[train_negative_count:]

    return LinkPredictionSplit(
        train_positive_edges=train_positive_edges,
        test_positive_edges=test_positive_edges,
        train_negative_edges=train_negative_edges,
        test_negative_edges=test_negative_edges,
        directed=graph.is_directed(),
    )
