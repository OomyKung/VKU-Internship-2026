"""Shared interfaces and validation helpers for graph embedding methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def sort_key(value: Any) -> tuple[str, str]:
    """Return a deterministic ordering key for mixed node identifier types."""

    return (type(value).__name__, repr(value))


def sorted_node_ids(graph: nx.Graph) -> tuple[Any, ...]:
    """Return graph node IDs in a deterministic order."""

    return tuple(sorted(graph.nodes(), key=sort_key))


class EmbeddingFrameworkError(ValueError):
    """Base error for embedding benchmark failures."""


class OptionalDependencyError(EmbeddingFrameworkError):
    """Raised when an embedding method requires an unavailable optional dependency."""


class UnsupportedGraphTypeError(EmbeddingFrameworkError):
    """Raised when an embedding method does not support the current graph type."""


@dataclass(slots=True)
class EmbeddingResult:
    """Normalized embedding output and execution metadata."""

    method_name: str
    embedding_frame: pd.DataFrame
    runtime_seconds: float
    node_count: int
    embedding_dim: int
    config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def vector_columns(self) -> list[str]:
        return embedding_columns(self.embedding_frame)


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    """Return normalized embedding columns in order."""

    return [column for column in frame.columns if column.startswith("embedding_")]


def embedding_frame_from_array(
    node_ids: Sequence[Any],
    embedding_array: np.ndarray | Sequence[Sequence[float]] | Sequence[float],
) -> pd.DataFrame:
    """Build the normalized embedding frame from raw vectors."""

    array = np.asarray(embedding_array, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("Embedding arrays must be one- or two-dimensional.")
    if array.shape[0] != len(node_ids):
        raise ValueError(
            "Embedding row count must match the node ordering. "
            f"Expected {len(node_ids)}, got {array.shape[0]}."
        )
    if array.shape[1] < 1:
        raise ValueError("Embedding arrays must contain at least one dimension.")

    frame = pd.DataFrame({"node_id": list(node_ids)})
    for dimension_index in range(array.shape[1]):
        frame[f"embedding_{dimension_index}"] = array[:, dimension_index]
    return frame


def validate_embedding_frame(
    node_ids: Sequence[Any],
    embedding_frame: pd.DataFrame,
    *,
    method_name: str,
) -> pd.DataFrame:
    """Validate and normalize the shared embedding output format."""

    if "node_id" not in embedding_frame.columns:
        raise ValueError(f"{method_name} did not return a node_id column.")
    if embedding_frame["node_id"].duplicated().any():
        raise ValueError(f"{method_name} returned duplicate node IDs.")

    vector_columns = embedding_columns(embedding_frame)
    if not vector_columns:
        raise ValueError(f"{method_name} did not return any embedding_* columns.")

    expected_nodes = tuple(node_ids)
    returned_nodes = tuple(embedding_frame["node_id"].tolist())
    if set(returned_nodes) != set(expected_nodes):
        missing_nodes = sorted(set(expected_nodes) - set(returned_nodes), key=sort_key)
        extra_nodes = sorted(set(returned_nodes) - set(expected_nodes), key=sort_key)
        raise ValueError(
            f"{method_name} embeddings do not match graph nodes. "
            f"Missing={missing_nodes[:5]} Extra={extra_nodes[:5]}."
        )

    frame = (
        embedding_frame.set_index("node_id", drop=False)
        .reindex(expected_nodes)
        .reset_index(drop=True)
        .copy()
    )
    numeric_frame = frame.loc[:, vector_columns].apply(pd.to_numeric, errors="coerce")
    if numeric_frame.isna().any().any():
        raise ValueError(f"{method_name} produced NaN values in the embedding matrix.")
    if not np.isfinite(numeric_frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{method_name} produced non-finite embedding values.")
    frame.loc[:, vector_columns] = numeric_frame
    return frame


class GraphEmbeddingModel(ABC):
    """Base class for embedding methods with a shared fit/transform API."""

    method_name: str = ""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self._result: EmbeddingResult | None = None

    def fit(
        self,
        graph: nx.Graph,
        features: Any | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> GraphEmbeddingModel:
        """Fit the method and keep the normalized embedding result in memory."""

        combined_config = dict(self.config)
        if config is not None:
            combined_config.update(dict(config))

        start = perf_counter()
        raw_frame, metadata = self._fit_impl(graph, features=features, config=combined_config)
        runtime_seconds = perf_counter() - start
        normalized_frame = validate_embedding_frame(
            sorted_node_ids(graph),
            raw_frame,
            method_name=self.method_name,
        )
        self._result = EmbeddingResult(
            method_name=self.method_name,
            embedding_frame=normalized_frame,
            runtime_seconds=runtime_seconds,
            node_count=int(graph.number_of_nodes()),
            embedding_dim=len(embedding_columns(normalized_frame)),
            config=combined_config,
            metadata=dict(metadata),
        )
        return self

    def transform(self, nodes: Sequence[Any] | None = None) -> pd.DataFrame:
        """Return all embeddings or a deterministic node subset."""

        if self._result is None:
            raise ValueError(f"{self.method_name} must be fitted before transform().")
        frame = self._result.embedding_frame
        if nodes is None:
            return frame.copy()

        requested_nodes = list(nodes)
        lookup = frame.set_index("node_id", drop=False)
        missing_nodes = [node_id for node_id in requested_nodes if node_id not in lookup.index]
        if missing_nodes:
            raise ValueError(
                f"{self.method_name} transform() requested nodes without embeddings: {missing_nodes[:5]}."
            )
        return lookup.loc[requested_nodes].reset_index(drop=True).copy()

    def fit_transform(
        self,
        graph: nx.Graph,
        features: Any | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Fit the method and return the normalized embedding frame."""

        self.fit(graph, features=features, config=config)
        return self.transform()

    @property
    def result(self) -> EmbeddingResult:
        """Return the fitted result object."""

        if self._result is None:
            raise ValueError(f"{self.method_name} must be fitted before result access.")
        return self._result

    @abstractmethod
    def _fit_impl(
        self,
        graph: nx.Graph,
        *,
        features: Any | None,
        config: Mapping[str, Any],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fit the method and return a normalized-or-normalizable embedding frame."""

