"""Lightweight deterministic Node2Vec-style embeddings for ML feature augmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


@dataclass(slots=True)
class Node2VecConfig:
    """Configuration for deterministic Node2Vec-style embedding generation."""

    dimensions: int = 8
    walk_length: int = 20
    num_walks: int = 10
    window: int = 5
    p: float = 1.0
    q: float = 1.0
    scale_embeddings: bool = False
    pca_components: int | None = None
    random_seed: int = 42


@dataclass(slots=True)
class Node2VecEmbeddingResult:
    """Embedding table plus runtime and cache metadata."""

    embedding_frame: pd.DataFrame
    runtime_seconds: float
    cache_path: Path | None
    loaded_from_cache: bool


def _validate_config(config: Node2VecConfig) -> None:
    if config.dimensions < 1:
        raise ValueError("Node2Vec dimensions must be at least 1.")
    if config.walk_length < 1:
        raise ValueError("Node2Vec walk_length must be at least 1.")
    if config.num_walks < 1:
        raise ValueError("Node2Vec num_walks must be at least 1.")
    if config.window < 1:
        raise ValueError("Node2Vec window must be at least 1.")
    if config.p <= 0.0:
        raise ValueError("Node2Vec p must be positive.")
    if config.q <= 0.0:
        raise ValueError("Node2Vec q must be positive.")
    if config.pca_components is not None and config.pca_components < 1:
        raise ValueError("Node2Vec pca_components must be at least 1 when provided.")


def _output_dimensions(config: Node2VecConfig, node_count: int) -> int:
    if config.pca_components is None:
        return config.dimensions
    upper_bound = max(1, min(config.pca_components, config.dimensions, node_count - 1))
    return upper_bound


def build_node2vec_cache_path(
    output_dir: Path | None,
    dataset_name: str,
    config: Node2VecConfig,
) -> Path | None:
    """Return a reusable cache path for a dataset/config pair when output_dir is available."""

    if output_dir is None:
        return None

    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{dataset_name}_node2vec_"
        f"d{config.dimensions}_"
        f"wl{config.walk_length}_"
        f"nw{config.num_walks}_"
        f"win{config.window}_"
        f"p{config.p:g}_"
        f"q{config.q:g}_"
        f"scale{int(config.scale_embeddings)}_"
        f"pca{config.pca_components if config.pca_components is not None else 'none'}_"
        f"seed{config.random_seed}.pkl"
    )
    return dataset_dir / filename


def _neighbor_lists(graph: nx.Graph) -> dict[Any, tuple[Any, ...]]:
    return {
        node_id: tuple(sorted(graph.neighbors(node_id), key=_sort_key))
        for node_id in graph.nodes()
    }


def _choose_next_node(
    graph: nx.Graph,
    neighbors_by_node: dict[Any, tuple[Any, ...]],
    previous_node: Any,
    current_node: Any,
    rng: np.random.Generator,
    config: Node2VecConfig,
) -> Any | None:
    candidates = neighbors_by_node[current_node]
    if not candidates:
        return None
    if previous_node is None or len(candidates) == 1:
        index = int(rng.integers(len(candidates)))
        return candidates[index]

    weights = np.asarray(
        [
            (1.0 / config.p)
            if neighbor_id == previous_node
            else (1.0 if graph.has_edge(neighbor_id, previous_node) else (1.0 / config.q))
            for neighbor_id in candidates
        ],
        dtype=float,
    )
    probabilities = weights / weights.sum()
    index = int(rng.choice(len(candidates), p=probabilities))
    return candidates[index]


def _generate_walks(graph: nx.Graph, config: Node2VecConfig) -> list[list[Any]]:
    rng = np.random.default_rng(config.random_seed)
    sorted_nodes = tuple(sorted(graph.nodes(), key=_sort_key))
    neighbors_by_node = _neighbor_lists(graph)
    walks: list[list[Any]] = []

    for _ in range(config.num_walks):
        walk_order = list(sorted_nodes)
        rng.shuffle(walk_order)
        for start_node in walk_order:
            walk = [start_node]
            previous_node: Any | None = None
            current_node = start_node
            for _ in range(config.walk_length - 1):
                next_node = _choose_next_node(
                    graph=graph,
                    neighbors_by_node=neighbors_by_node,
                    previous_node=previous_node,
                    current_node=current_node,
                    rng=rng,
                    config=config,
                )
                if next_node is None:
                    break
                walk.append(next_node)
                previous_node = current_node
                current_node = next_node
            walks.append(walk)
    return walks


def _build_cooccurrence_matrix(
    walks: list[list[Any]],
    node_index: dict[Any, int],
    window: int,
) -> csr_matrix:
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []

    for walk in walks:
        for center_index, center_node in enumerate(walk):
            center_position = node_index[center_node]
            left = max(0, center_index - window)
            right = min(len(walk), center_index + window + 1)
            for context_index in range(left, right):
                if context_index == center_index:
                    continue
                context_position = node_index[walk[context_index]]
                row_indices.append(center_position)
                col_indices.append(context_position)
                values.append(1.0)

    node_count = len(node_index)
    if not values:
        return csr_matrix((node_count, node_count), dtype=float)
    return csr_matrix((values, (row_indices, col_indices)), shape=(node_count, node_count), dtype=float)


def _compute_embeddings_from_walks(
    graph: nx.Graph,
    config: Node2VecConfig,
) -> pd.DataFrame:
    sorted_nodes = tuple(sorted(graph.nodes(), key=_sort_key))
    node_index = {node_id: index for index, node_id in enumerate(sorted_nodes)}
    walks = _generate_walks(graph, config)
    cooccurrence = _build_cooccurrence_matrix(walks, node_index=node_index, window=config.window)

    max_components = max(1, min(config.dimensions, cooccurrence.shape[0] - 1, cooccurrence.shape[1] - 1))
    if cooccurrence.nnz == 0 or max_components < 1:
        embedding_array = np.zeros((len(sorted_nodes), config.dimensions), dtype=float)
    else:
        svd = TruncatedSVD(n_components=max_components, random_state=config.random_seed)
        reduced = svd.fit_transform(cooccurrence)
        embedding_array = np.zeros((len(sorted_nodes), config.dimensions), dtype=float)
        embedding_array[:, :max_components] = reduced

    frame = pd.DataFrame({"node_id": list(sorted_nodes)})
    for dimension_index in range(config.dimensions):
        frame[f"node2vec_{dimension_index}"] = embedding_array[:, dimension_index]
    return frame


def transform_node2vec_embeddings(
    embedding_frame: pd.DataFrame,
    config: Node2VecConfig,
) -> pd.DataFrame:
    """Apply optional scaling and PCA to Node2Vec embeddings for feature use."""

    if "node_id" not in embedding_frame.columns:
        raise ValueError("embedding_frame must contain a node_id column.")

    feature_columns = [column_name for column_name in embedding_frame.columns if column_name.startswith("node2vec_")]
    transformed = embedding_frame[["node_id"]].copy()
    if not feature_columns:
        return transformed

    array = embedding_frame[feature_columns].to_numpy(dtype=float, copy=True)
    if config.scale_embeddings:
        scaler = StandardScaler()
        array = scaler.fit_transform(array)

    if config.pca_components is not None and array.shape[1] > 1:
        target_components = _output_dimensions(config, len(embedding_frame))
        if target_components < array.shape[1]:
            pca = PCA(n_components=target_components, random_state=config.random_seed)
            array = pca.fit_transform(array)

    for dimension_index in range(array.shape[1]):
        transformed[f"node2vec_{dimension_index}"] = array[:, dimension_index]
    return transformed


def _load_cached_embeddings(
    cache_path: Path,
    graph: nx.Graph,
    config: Node2VecConfig,
) -> pd.DataFrame:
    frame = pd.read_pickle(cache_path)
    output_dimensions = _output_dimensions(config, graph.number_of_nodes())
    required_columns = {"node_id"} | {f"node2vec_{index}" for index in range(output_dimensions)}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Cached Node2Vec embeddings are missing columns: {sorted(missing)}.")
    node_ids = tuple(frame["node_id"].tolist())
    expected_nodes = tuple(sorted(graph.nodes(), key=_sort_key))
    if set(node_ids) != set(expected_nodes):
        raise ValueError("Cached Node2Vec embeddings do not cover the current graph nodes.")
    return frame.sort_values("node_id", key=lambda values: values.map(_sort_key)).reset_index(drop=True)


def generate_node2vec_embeddings(
    graph: nx.Graph,
    config: Node2VecConfig,
    cache_path: Path | None = None,
) -> Node2VecEmbeddingResult:
    """Generate or load deterministic Node2Vec-style embeddings for every node.

    This implementation uses Node2Vec-style biased random walks and then applies
    truncated SVD to the resulting co-occurrence matrix. It is intentionally
    lightweight so the project can stay dependency-light while still gaining
    walk-based structural embeddings.
    """

    _validate_config(config)
    work_graph = graph if not graph.is_directed() else graph.to_undirected()
    start = perf_counter()

    if cache_path is not None and cache_path.exists():
        frame = _load_cached_embeddings(cache_path, work_graph, config)
        return Node2VecEmbeddingResult(
            embedding_frame=frame,
            runtime_seconds=perf_counter() - start,
            cache_path=cache_path,
            loaded_from_cache=True,
        )

    frame = _compute_embeddings_from_walks(work_graph, config)
    frame = transform_node2vec_embeddings(frame, config)
    if cache_path is not None:
        frame.to_pickle(cache_path)
    return Node2VecEmbeddingResult(
        embedding_frame=frame,
        runtime_seconds=perf_counter() - start,
        cache_path=cache_path,
        loaded_from_cache=False,
    )
