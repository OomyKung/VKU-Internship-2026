"""Shared PyTorch Geometric helpers for embedding methods."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import importlib.util
import random
from typing import Any

import networkx as nx
import numpy as np

from .base import OptionalDependencyError, sorted_node_ids
from .features import PreparedFeatures, coerce_input_features


@dataclass(slots=True)
class PYGGraphData:
    """Prepared graph tensors for PyG-based methods."""

    node_order: tuple[Any, ...]
    x: Any
    edge_index: Any

    @property
    def input_dim(self) -> int:
        return int(self.x.shape[1])

    @property
    def node_count(self) -> int:
        return int(self.x.shape[0])


def pyg_dependencies_available() -> bool:
    """Return whether torch and torch_geometric are importable."""

    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("torch_geometric") is not None
    )


def require_pyg_dependencies() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """Import the optional PyG stack or raise a clear benchmark error."""

    if not pyg_dependencies_available():
        raise OptionalDependencyError(
            "This embedding method requires optional dependencies 'torch' and 'torch_geometric'."
        )

    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415
    from torch_geometric.data import Data  # noqa: PLC0415
    from torch_geometric.nn import DeepGraphInfomax, GAE, VGAE, GCNConv, SAGEConv  # noqa: PLC0415

    return torch, nn, Data, GAE, VGAE, DeepGraphInfomax, GCNConv, SAGEConv


def seed_torch(torch: Any, random_seed: int) -> None:
    """Set deterministic seeds across Python, NumPy, and torch."""

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(int(random_seed))
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def _edge_index_tensor(
    graph: nx.Graph,
    node_order: Sequence[Any],
    torch: Any,
) -> Any:
    node_to_index = {node_id: index for index, node_id in enumerate(node_order)}
    edge_pairs: list[tuple[int, int]] = []
    for source_node, target_node in graph.edges():
        source_index = node_to_index[source_node]
        target_index = node_to_index[target_node]
        edge_pairs.append((source_index, target_index))
        if source_index != target_index:
            edge_pairs.append((target_index, source_index))

    if not edge_pairs:
        self_loops = [(index, index) for index in range(len(node_order))]
        edge_pairs = self_loops

    return torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()


def build_pyg_graph_data(
    graph: nx.Graph,
    features: PreparedFeatures | Any | None,
    torch: Any,
) -> PYGGraphData:
    """Convert aligned graph/features into tensors for PyG methods."""

    node_order = sorted_node_ids(graph)
    prepared_features = coerce_input_features(features, node_order)
    if prepared_features is None:
        raise ValueError("This embedding method requires numeric node features.")

    x = torch.tensor(prepared_features.feature_matrix, dtype=torch.float32)
    edge_index = _edge_index_tensor(graph, node_order=node_order, torch=torch)
    return PYGGraphData(node_order=node_order, x=x, edge_index=edge_index)


def build_gnn_encoder(
    nn_module: Any,
    conv_cls: Any,
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    dropout: float,
) -> Any:
    """Build a stacked message-passing encoder."""

    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")

    class Encoder(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            layer_dims: list[tuple[int, int]] = []
            if num_layers == 1:
                layer_dims.append((input_dim, output_dim))
            else:
                layer_dims.append((input_dim, hidden_dim))
                layer_dims.extend((hidden_dim, hidden_dim) for _ in range(num_layers - 2))
                layer_dims.append((hidden_dim, output_dim))
            self.convs = nn_module.ModuleList(
                [conv_cls(in_dim, out_dim) for in_dim, out_dim in layer_dims]
            )
            self.dropout = float(dropout)

        def forward(self, x: Any, edge_index: Any) -> Any:
            output = x
            for layer_index, conv in enumerate(self.convs):
                output = conv(output, edge_index)
                if layer_index < len(self.convs) - 1:
                    output = nn_module.functional.relu(output)
                    if self.dropout > 0.0:
                        output = nn_module.functional.dropout(
                            output,
                            p=self.dropout,
                            training=self.training,
                        )
            return output

    return Encoder()


def build_variational_gcn_encoder(
    nn_module: Any,
    gcn_conv_cls: Any,
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    dropout: float,
) -> Any:
    """Build a variational GCN encoder for VGAE."""

    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")

    class VariationalEncoder(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden_layers = max(1, num_layers - 1)
            layer_dims: list[tuple[int, int]] = [(input_dim, hidden_dim)]
            layer_dims.extend((hidden_dim, hidden_dim) for _ in range(hidden_layers - 1))
            self.shared_convs = nn_module.ModuleList(
                [gcn_conv_cls(in_dim, out_dim) for in_dim, out_dim in layer_dims]
            )
            self.mu_conv = gcn_conv_cls(hidden_dim, output_dim)
            self.logstd_conv = gcn_conv_cls(hidden_dim, output_dim)
            self.dropout = float(dropout)

        def forward(self, x: Any, edge_index: Any) -> tuple[Any, Any]:
            output = x
            for conv in self.shared_convs:
                output = conv(output, edge_index)
                output = nn_module.functional.relu(output)
                if self.dropout > 0.0:
                    output = nn_module.functional.dropout(
                        output,
                        p=self.dropout,
                        training=self.training,
                    )
            return self.mu_conv(output, edge_index), self.logstd_conv(output, edge_index)

    return VariationalEncoder()


def dropout_edge_index(edge_index: Any, dropout_probability: float, generator: Any, torch: Any) -> Any:
    """Randomly drop edges while keeping a non-empty graph."""

    if dropout_probability <= 0.0 or int(edge_index.shape[1]) == 0:
        return edge_index.clone()
    mask = torch.rand(int(edge_index.shape[1]), generator=generator) >= float(dropout_probability)
    if int(mask.sum().item()) == 0:
        return edge_index.clone()
    return edge_index[:, mask]


def mask_feature_matrix(x: Any, mask_probability: float, generator: Any, torch: Any) -> Any:
    """Randomly zero feature entries for contrastive augmentations."""

    if mask_probability <= 0.0:
        return x.clone()
    mask = torch.rand(x.shape, generator=generator) >= float(mask_probability)
    return x * mask.to(dtype=x.dtype)

