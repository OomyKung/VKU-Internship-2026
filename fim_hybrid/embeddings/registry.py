"""Registry and capability metadata for embedding methods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib.util
from typing import Any

import networkx as nx

from .base import GraphEmbeddingModel
from .deepwalk import DeepWalkEmbedding
from .line import LINEEmbedding
from .node2vec import Node2VecEmbedding


@dataclass(frozen=True, slots=True)
class EmbeddingMethodSpec:
    """Static metadata describing one supported embedding method."""

    name: str
    model_cls: type[GraphEmbeddingModel]
    needs_features: bool
    optional_dependencies: tuple[str, ...] = ()
    supports_homogeneous_graphs: bool = True
    description: str = ""
    unsupported_reason: str | None = None


def _pyg_available() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("torch_geometric") is not None
    )


def _build_registry() -> dict[str, EmbeddingMethodSpec]:
    from .dgi import DGIEmbedding
    from .gcn import GCNEmbedding
    from .graphcl import GraphCLEmbedding
    from .graphsage import GraphSAGEEmbedding
    from .metapath2vec import MetaPath2VecEmbedding
    from .vgae import VGAEEmbedding

    return {
        "deepwalk": EmbeddingMethodSpec(
            name="deepwalk",
            model_cls=DeepWalkEmbedding,
            needs_features=False,
            description="Random-walk based shallow embeddings using unbiased walks.",
        ),
        "node2vec": EmbeddingMethodSpec(
            name="node2vec",
            model_cls=Node2VecEmbedding,
            needs_features=False,
            description="Biased random-walk shallow embeddings.",
        ),
        "line": EmbeddingMethodSpec(
            name="line",
            model_cls=LINEEmbedding,
            needs_features=False,
            description="First- and second-order proximity embeddings.",
        ),
        "vgae": EmbeddingMethodSpec(
            name="vgae",
            model_cls=VGAEEmbedding,
            needs_features=True,
            optional_dependencies=("torch", "torch_geometric"),
            description="Variational graph autoencoder with a GCN encoder.",
        ),
        "gcn": EmbeddingMethodSpec(
            name="gcn",
            model_cls=GCNEmbedding,
            needs_features=True,
            optional_dependencies=("torch", "torch_geometric"),
            description="Unsupervised graph autoencoder with a GCN encoder.",
        ),
        "graphsage": EmbeddingMethodSpec(
            name="graphsage",
            model_cls=GraphSAGEEmbedding,
            needs_features=True,
            optional_dependencies=("torch", "torch_geometric"),
            description="Unsupervised graph autoencoder with a GraphSAGE encoder.",
        ),
        "dgi": EmbeddingMethodSpec(
            name="dgi",
            model_cls=DGIEmbedding,
            needs_features=True,
            optional_dependencies=("torch", "torch_geometric"),
            description="Deep Graph Infomax self-supervised embeddings.",
        ),
        "graphcl": EmbeddingMethodSpec(
            name="graphcl",
            model_cls=GraphCLEmbedding,
            needs_features=True,
            optional_dependencies=("torch", "torch_geometric"),
            description="Graph contrastive learning with two augmented views.",
        ),
        "metapath2vec": EmbeddingMethodSpec(
            name="metapath2vec",
            model_cls=MetaPath2VecEmbedding,
            needs_features=False,
            optional_dependencies=("torch", "torch_geometric"),
            supports_homogeneous_graphs=False,
            description="Heterogeneous graph embeddings over typed metapaths.",
            unsupported_reason=(
                "metapath2vec requires a heterogeneous graph with node/edge types; "
                "the current benchmark graph is homogeneous."
            ),
        ),
    }


_REGISTRY = _build_registry()


def available_embedding_methods() -> tuple[str, ...]:
    """Return all registered embedding method names."""

    return tuple(_REGISTRY)


def resolve_method_names(methods: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize selected methods, including the 'all' alias."""

    if methods is None or not methods:
        return tuple(_REGISTRY)
    normalized = [str(method).strip().lower() for method in methods]
    if "all" in normalized:
        return tuple(_REGISTRY)
    unknown = sorted(set(normalized) - set(_REGISTRY))
    if unknown:
        raise ValueError(
            f"Unsupported embedding methods: {unknown}. Supported methods: {sorted(_REGISTRY)}."
        )
    return tuple(dict.fromkeys(normalized))


def get_method_spec(method_name: str) -> EmbeddingMethodSpec:
    """Return the static registry entry for one method."""

    key = str(method_name).strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown embedding method '{method_name}'.")
    return _REGISTRY[key]


def create_embedding_model(
    method_name: str,
    config: Mapping[str, Any] | None = None,
) -> GraphEmbeddingModel:
    """Instantiate one embedding method from the registry."""

    spec = get_method_spec(method_name)
    return spec.model_cls(config=config)


def missing_dependency_reason(method_name: str) -> str | None:
    """Return a skip reason when the method's optional stack is unavailable."""

    spec = get_method_spec(method_name)
    if not spec.optional_dependencies:
        return None
    if spec.optional_dependencies == ("torch", "torch_geometric") and _pyg_available():
        return None
    missing = [
        dependency_name
        for dependency_name in spec.optional_dependencies
        if importlib.util.find_spec(dependency_name) is None
    ]
    if not missing:
        return None
    return f"{method_name} requires optional dependencies: {', '.join(missing)}."


def unsupported_graph_reason(method_name: str, graph: nx.Graph) -> str | None:
    """Return a graph-type skip reason when the method is not applicable."""

    spec = get_method_spec(method_name)
    if spec.supports_homogeneous_graphs:
        return None
    if isinstance(graph, nx.Graph):
        return spec.unsupported_reason
    return None


def method_requires_features(method_name: str) -> bool:
    """Return whether the embedding method expects node features."""

    return get_method_spec(method_name).needs_features

