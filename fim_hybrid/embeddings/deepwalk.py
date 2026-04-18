"""Benchmark wrapper for DeepWalk embeddings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx
import pandas as pd

from .base import GraphEmbeddingModel
from .node2vec import Node2VecEmbedding


class DeepWalkEmbedding(GraphEmbeddingModel):
    """DeepWalk implemented as Node2Vec with p=q=1."""

    method_name = "deepwalk"

    def _fit_impl(
        self,
        graph: nx.Graph,
        *,
        features: Any | None,
        config: Mapping[str, Any],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        node2vec_model = Node2VecEmbedding()
        node2vec_frame = node2vec_model.fit_transform(
            graph,
            config={
                **dict(config),
                "p": 1.0,
                "q": 1.0,
            },
        )
        return node2vec_frame, {"base_method": "node2vec", "deepwalk_mode": True}

