"""Lightweight LINE-style embeddings for static homogeneous graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from .base import GraphEmbeddingModel, embedding_frame_from_array, sorted_node_ids


def _validate_line_config(embedding_dim: int, order: str) -> None:
    if embedding_dim < 1:
        raise ValueError("LINE embedding_dim must be at least 1.")
    if order not in {"first", "second", "both"}:
        raise ValueError("LINE order must be one of ['first', 'second', 'both'].")
    if order == "both" and embedding_dim < 2:
        raise ValueError("LINE order='both' requires embedding_dim >= 2.")


def _svd_embedding(
    matrix: sparse.spmatrix,
    embedding_dim: int,
    random_seed: int,
) -> np.ndarray:
    node_count = matrix.shape[0]
    if node_count == 0:
        raise ValueError("LINE cannot embed an empty graph.")
    if matrix.nnz == 0 or node_count == 1:
        return np.zeros((node_count, embedding_dim), dtype=float)

    max_components = min(embedding_dim, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if max_components < 1:
        return np.zeros((node_count, embedding_dim), dtype=float)
    svd = TruncatedSVD(n_components=max_components, n_iter=10, random_state=random_seed)
    reduced = svd.fit_transform(matrix)
    embedding = np.zeros((node_count, embedding_dim), dtype=float)
    embedding[:, :max_components] = reduced
    return embedding


def _normalized_transition_matrix(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    row_sums = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.zeros_like(row_sums, dtype=float)
    non_zero_mask = row_sums > 0
    inverse[non_zero_mask] = 1.0 / row_sums[non_zero_mask]
    diagonal = sparse.diags(inverse)
    return diagonal @ adjacency


def _combine_orders(
    embeddings: Sequence[np.ndarray],
    embedding_dim: int,
) -> np.ndarray:
    combined = np.concatenate(list(embeddings), axis=1)
    if combined.shape[1] == embedding_dim:
        return combined
    if combined.shape[1] > embedding_dim:
        return combined[:, :embedding_dim]
    padding = np.zeros((combined.shape[0], embedding_dim - combined.shape[1]), dtype=float)
    return np.concatenate([combined, padding], axis=1)


class LINEEmbedding(GraphEmbeddingModel):
    """Approximate LINE via first- and second-order proximity matrices plus SVD."""

    method_name = "line"

    def _fit_impl(
        self,
        graph: nx.Graph,
        *,
        features: Any | None,
        config: Mapping[str, Any],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        embedding_dim = int(config.get("embedding_dim", 64))
        order = str(config.get("order", "both")).lower()
        random_seed = int(config.get("random_seed", 42))
        _validate_line_config(embedding_dim=embedding_dim, order=order)

        node_order = sorted_node_ids(graph)
        adjacency = nx.to_scipy_sparse_array(
            graph,
            nodelist=list(node_order),
            dtype=float,
            format="csr",
        )
        if not sparse.isspmatrix_csr(adjacency):
            adjacency = sparse.csr_matrix(adjacency)

        if order == "first":
            embedding_array = _svd_embedding(adjacency, embedding_dim=embedding_dim, random_seed=random_seed)
        elif order == "second":
            transition = _normalized_transition_matrix(adjacency)
            embedding_array = _svd_embedding(transition, embedding_dim=embedding_dim, random_seed=random_seed)
        else:
            first_dim = embedding_dim // 2
            second_dim = embedding_dim - first_dim
            first_order = _svd_embedding(adjacency, embedding_dim=first_dim, random_seed=random_seed)
            second_order = _svd_embedding(
                _normalized_transition_matrix(adjacency),
                embedding_dim=second_dim,
                random_seed=random_seed,
            )
            embedding_array = _combine_orders([first_order, second_order], embedding_dim=embedding_dim)

        return embedding_frame_from_array(node_order, embedding_array), {
            "order": order,
            "uses_sparse_svd": True,
        }

