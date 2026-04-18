"""Heterogeneous-only metapath2vec placeholder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx
import pandas as pd

from .base import GraphEmbeddingModel, UnsupportedGraphTypeError


class MetaPath2VecEmbedding(GraphEmbeddingModel):
    """Clear placeholder for future heterogeneous graph support."""

    method_name = "metapath2vec"

    def _fit_impl(
        self,
        graph: nx.Graph,
        *,
        features: Any | None,
        config: Mapping[str, Any],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        raise UnsupportedGraphTypeError(
            "metapath2vec requires a heterogeneous graph with explicit node and edge types. "
            "The current benchmark graph is homogeneous, so this method is only stubbed."
        )

