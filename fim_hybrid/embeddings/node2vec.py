"""Benchmark wrapper for deterministic Node2Vec embeddings."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from fim_hybrid.node2vec_embeddings import Node2VecConfig, generate_node2vec_embeddings

from .base import GraphEmbeddingModel


def _renamed_node2vec_frame(frame: pd.DataFrame) -> pd.DataFrame:
    renamed_columns = {
        column_name: column_name.replace("node2vec_", "embedding_")
        for column_name in frame.columns
        if column_name.startswith("node2vec_")
    }
    return frame.rename(columns=renamed_columns)


class Node2VecEmbedding(GraphEmbeddingModel):
    """Generate walk-based structural embeddings with deterministic settings."""

    method_name = "node2vec"

    def _fit_impl(
        self,
        graph: nx.Graph,
        *,
        features: Any | None,
        config: Mapping[str, Any],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        node2vec_config = Node2VecConfig(
            dimensions=int(config.get("embedding_dim", 64)),
            walk_length=int(config.get("walk_length", 20)),
            num_walks=int(config.get("num_walks", 10)),
            window=int(config.get("window_size", 5)),
            p=float(config.get("p", 1.0)),
            q=float(config.get("q", 1.0)),
            scale_embeddings=bool(config.get("scale_embeddings", False)),
            pca_components=(
                None
                if config.get("pca_components") is None
                else int(config["pca_components"])
            ),
            random_seed=int(config.get("random_seed", 42)),
        )
        cache_path = None if config.get("cache_path") is None else Path(config["cache_path"])
        result = generate_node2vec_embeddings(graph, node2vec_config, cache_path=cache_path)
        return _renamed_node2vec_frame(result.embedding_frame), {
            "loaded_from_cache": result.loaded_from_cache,
            "cache_path": None if result.cache_path is None else str(result.cache_path),
            "base_runtime_seconds": result.runtime_seconds,
        }

