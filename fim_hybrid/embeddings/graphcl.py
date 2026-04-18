"""GraphCL-style node embeddings from two augmented graph views."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx

from ._pyg_common import (
    build_gnn_encoder,
    build_pyg_graph_data,
    dropout_edge_index,
    mask_feature_matrix,
    require_pyg_dependencies,
    seed_torch,
)
from .base import GraphEmbeddingModel, embedding_frame_from_array


def _nt_xent_loss(z1: Any, z2: Any, temperature: float, nn_module: Any, torch: Any) -> Any:
    if temperature <= 0.0:
        raise ValueError("GraphCL temperature must be positive.")

    batch_size = int(z1.size(0))
    representations = torch.cat([z1, z2], dim=0)
    logits = torch.matmul(representations, representations.T) / float(temperature)
    mask = torch.eye(int(logits.size(0)), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(mask, -1e9)
    targets = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=logits.device),
            torch.arange(0, batch_size, device=logits.device),
        ]
    )
    return nn_module.functional.cross_entropy(logits, targets)


class GraphCLEmbedding(GraphEmbeddingModel):
    """Learn node embeddings with simple feature-masking and edge-drop augmentations."""

    method_name = "graphcl"

    def _fit_impl(
        self,
        graph: nx.Graph,
        *,
        features: Any | None,
        config: Mapping[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        torch, nn_module, _, _, _, _, _, sage_conv_cls = require_pyg_dependencies()
        random_seed = int(config.get("random_seed", 42))
        seed_torch(torch, random_seed)
        graph_data = build_pyg_graph_data(graph, features=features, torch=torch)

        embedding_dim = int(config.get("embedding_dim", 64))
        hidden_dim = int(config.get("hidden_dim", 128))
        projection_dim = int(config.get("projection_dim", embedding_dim))
        num_layers = int(config.get("num_layers", 2))
        dropout = float(config.get("dropout", 0.2))
        learning_rate = float(config.get("learning_rate", 1e-3))
        weight_decay = float(config.get("weight_decay", 5e-4))
        epochs = int(config.get("epochs", 100))
        temperature = float(config.get("temperature", 0.5))
        edge_dropout_probability = float(config.get("edge_dropout_probability", 0.2))
        feature_mask_probability = float(config.get("feature_mask_probability", 0.2))

        encoder = build_gnn_encoder(
            nn_module,
            sage_conv_cls,
            input_dim=graph_data.input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        projector = nn_module.Sequential(
            nn_module.Linear(embedding_dim, projection_dim),
            nn_module.ReLU(),
            nn_module.Linear(projection_dim, embedding_dim),
        )
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(projector.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        generator = torch.Generator()
        generator.manual_seed(random_seed)

        for _ in range(epochs):
            encoder.train()
            projector.train()
            optimizer.zero_grad()

            view_one_x = mask_feature_matrix(
                graph_data.x,
                mask_probability=feature_mask_probability,
                generator=generator,
                torch=torch,
            )
            view_two_x = mask_feature_matrix(
                graph_data.x,
                mask_probability=feature_mask_probability,
                generator=generator,
                torch=torch,
            )
            view_one_edges = dropout_edge_index(
                graph_data.edge_index,
                dropout_probability=edge_dropout_probability,
                generator=generator,
                torch=torch,
            )
            view_two_edges = dropout_edge_index(
                graph_data.edge_index,
                dropout_probability=edge_dropout_probability,
                generator=generator,
                torch=torch,
            )

            h1 = encoder(view_one_x, view_one_edges)
            h2 = encoder(view_two_x, view_two_edges)
            z1 = nn_module.functional.normalize(projector(h1), dim=1)
            z2 = nn_module.functional.normalize(projector(h2), dim=1)
            loss = _nt_xent_loss(z1, z2, temperature=temperature, nn_module=nn_module, torch=torch)
            loss.backward()
            optimizer.step()

        encoder.eval()
        with torch.no_grad():
            embedding_array = encoder(graph_data.x, graph_data.edge_index).detach().cpu().numpy()
        return embedding_frame_from_array(graph_data.node_order, embedding_array), {
            "input_dim": graph_data.input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "projection_dim": projection_dim,
            "temperature": temperature,
        }

