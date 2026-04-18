"""Unsupervised GCN embeddings via a graph autoencoder objective."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx

from ._pyg_common import build_gnn_encoder, build_pyg_graph_data, require_pyg_dependencies, seed_torch
from .base import GraphEmbeddingModel, embedding_frame_from_array


class GCNEmbedding(GraphEmbeddingModel):
    """Learn node embeddings with a GCN encoder and reconstruction loss."""

    method_name = "gcn"

    def _fit_impl(
        self,
        graph: nx.Graph,
        *,
        features: Any | None,
        config: Mapping[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        torch, nn_module, _, gae_cls, _, _, gcn_conv_cls, _ = require_pyg_dependencies()
        seed_torch(torch, int(config.get("random_seed", 42)))
        graph_data = build_pyg_graph_data(graph, features=features, torch=torch)

        embedding_dim = int(config.get("embedding_dim", 64))
        hidden_dim = int(config.get("hidden_dim", 128))
        num_layers = int(config.get("num_layers", 2))
        dropout = float(config.get("dropout", 0.2))
        learning_rate = float(config.get("learning_rate", 1e-3))
        weight_decay = float(config.get("weight_decay", 5e-4))
        epochs = int(config.get("epochs", 100))

        encoder = build_gnn_encoder(
            nn_module,
            gcn_conv_cls,
            input_dim=graph_data.input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        model = gae_cls(encoder)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            z = model.encode(graph_data.x, graph_data.edge_index)
            loss = model.recon_loss(z, graph_data.edge_index)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            embedding_array = model.encode(graph_data.x, graph_data.edge_index).detach().cpu().numpy()
        return embedding_frame_from_array(graph_data.node_order, embedding_array), {
            "input_dim": graph_data.input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        }

