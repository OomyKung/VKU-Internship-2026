"""Optional GNN training utilities for ML-guided candidate ranking in fair FIM."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import LoadedDataset
from .ml_training import (
    _build_feature_matrix,
    _precision_at_k,
    _rank_nodes,
    _safe_spearman,
    _split_training_frame,
    _sort_key,
    select_ml_candidate_nodes,
)


_LABEL_EXCLUDED_COLUMNS = {
    "node_id",
    "singleton_total_spread",
    "singleton_mf",
    "singleton_soft_mf",
    "singleton_dcv",
    "singleton_soft_fair_score",
    "spread_norm",
    "soft_fair_norm",
    "label_score",
    "gnn_label_score",
    "marginal_proxy_score",
    "marginal_proxy_norm",
}
_CATEGORICAL_COLUMNS = ("community_id", "protected_group")
_SUPPORTED_GNN_MODEL_TYPES = {"graphsage", "gcn"}


@dataclass(slots=True)
class GNNTrainingResult:
    """Fitted GNN outputs and ranking diagnostics for ML-guided FIM."""

    model: Any | None
    training_frame: pd.DataFrame
    predicted_scores: dict[Any, float]
    ranked_nodes: tuple[Any, ...]
    candidate_nodes: tuple[Any, ...]
    validation_spearman: float
    validation_precision_at_budget: float
    runtime_seconds: float
    model_type: str
    cache_path: Path | None
    loaded_from_cache: bool
    feature_matrix_shape: tuple[int, int]
    edge_index_shape: tuple[int, int]


def gnn_dependencies_available() -> bool:
    """Return whether the optional torch/PyG stack is importable."""

    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torch_geometric") is not None


def require_gnn_dependencies() -> None:
    """Raise a clear error when the optional GNN stack is unavailable."""

    if gnn_dependencies_available():
        return

    raise ValueError(
        "ml_backend includes 'gnn', but optional dependencies are unavailable. "
        "Install both 'torch' and 'torch_geometric' to enable the GNN backend."
    )


def _import_gnn_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    require_gnn_dependencies()

    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415
    from torch_geometric.data import Data  # noqa: PLC0415
    from torch_geometric.nn import GCNConv, SAGEConv  # noqa: PLC0415

    return torch, nn, Data, SAGEConv, GCNConv


def _validate_training_parameters(
    model_type: str,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
) -> None:
    if model_type not in _SUPPORTED_GNN_MODEL_TYPES:
        raise ValueError(f"gnn_model_type must be one of {sorted(_SUPPORTED_GNN_MODEL_TYPES)}.")
    if hidden_dim < 1:
        raise ValueError("hidden_dim must be at least 1.")
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in the interval [0.0, 1.0).")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")


def _one_hot_feature_frame(training_frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column_name
        for column_name in training_frame.columns
        if column_name not in _LABEL_EXCLUDED_COLUMNS and column_name not in _CATEGORICAL_COLUMNS
    ]
    numeric_frame = training_frame.loc[:, numeric_columns].astype(float)
    categorical_frame = pd.get_dummies(
        training_frame.loc[:, list(_CATEGORICAL_COLUMNS)].astype(str),
        columns=list(_CATEGORICAL_COLUMNS),
        dtype=float,
    )
    feature_frame = pd.concat([numeric_frame, categorical_frame], axis=1)
    return feature_frame.reindex(sorted(feature_frame.columns), axis=1)


def _build_edge_index(
    dataset: LoadedDataset,
    node_to_index: dict[Any, int],
) -> np.ndarray:
    work_graph = dataset.graph if not dataset.graph.is_directed() else dataset.graph.to_undirected()
    edge_pairs: list[tuple[int, int]] = []
    for source_node, target_node in sorted(work_graph.edges(), key=lambda item: (_sort_key(item[0]), _sort_key(item[1]))):
        source_index = node_to_index[source_node]
        target_index = node_to_index[target_node]
        edge_pairs.append((source_index, target_index))
        edge_pairs.append((target_index, source_index))

    if not edge_pairs:
        return np.empty((2, 0), dtype=np.int64)

    return np.asarray(edge_pairs, dtype=np.int64).T


def _fingerprint_value(value: Any) -> dict[str, str]:
    return {
        "type": type(value).__name__,
        "repr": repr(value),
    }


def _build_cache_fingerprint(
    node_ids: list[Any],
    feature_matrix: np.ndarray,
    label_scores: np.ndarray,
    edge_index: np.ndarray,
    cache_config: dict[str, Any],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(cache_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    hasher.update(
        json.dumps(
            [_fingerprint_value(node_id) for node_id in node_ids],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    hasher.update(feature_matrix.astype(np.float64).tobytes())
    hasher.update(label_scores.astype(np.float64).tobytes())
    hasher.update(edge_index.astype(np.int64).tobytes())
    return hasher.hexdigest()


def _load_cached_training_result(
    cache_path: Path,
    expected_fingerprint: str,
    training_frame: pd.DataFrame,
) -> GNNTrainingResult | None:
    if not cache_path.exists():
        return None

    with cache_path.open("rb") as handle:
        cached_payload = pickle.load(handle)

    if not isinstance(cached_payload, dict):
        return None

    metadata = cached_payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    if metadata.get("fingerprint") != expected_fingerprint:
        return None

    predicted_scores = cached_payload.get("predicted_scores")
    if not isinstance(predicted_scores, dict):
        return None

    ranked_nodes = tuple(cached_payload.get("ranked_nodes", ()))
    candidate_nodes = tuple(cached_payload.get("candidate_nodes", ()))
    if set(predicted_scores) != set(training_frame["node_id"]):
        return None
    if set(ranked_nodes) != set(training_frame["node_id"]):
        return None
    if not set(candidate_nodes).issubset(set(training_frame["node_id"])):
        return None

    return GNNTrainingResult(
        model=None,
        training_frame=training_frame.copy(),
        predicted_scores={
            node_id: float(predicted_scores[node_id])
            for node_id in training_frame["node_id"]
        },
        ranked_nodes=ranked_nodes,
        candidate_nodes=candidate_nodes,
        validation_spearman=float(metadata["validation_spearman"]),
        validation_precision_at_budget=float(metadata["validation_precision_at_budget"]),
        runtime_seconds=float(metadata["runtime_seconds"]),
        model_type=str(metadata["model_type"]),
        cache_path=cache_path,
        loaded_from_cache=True,
        feature_matrix_shape=tuple(metadata["feature_matrix_shape"]),
        edge_index_shape=tuple(metadata["edge_index_shape"]),
    )


def _store_cached_training_result(
    cache_path: Path,
    fingerprint: str,
    result: GNNTrainingResult,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "fingerprint": fingerprint,
            "validation_spearman": float(result.validation_spearman),
            "validation_precision_at_budget": float(result.validation_precision_at_budget),
            "runtime_seconds": float(result.runtime_seconds),
            "model_type": result.model_type,
            "feature_matrix_shape": list(result.feature_matrix_shape),
            "edge_index_shape": list(result.edge_index_shape),
        },
        "predicted_scores": result.predicted_scores,
        "ranked_nodes": list(result.ranked_nodes),
        "candidate_nodes": list(result.candidate_nodes),
    }
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle)


class _NodeScoreGNNModel:
    """Thin wrapper that builds a small GraphSAGE or GCN regressor."""

    def __init__(
        self,
        model_type: str,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        nn_module: Any,
        sage_conv_cls: Any,
        gcn_conv_cls: Any,
    ) -> None:
        conv_cls = sage_conv_cls if model_type == "graphsage" else gcn_conv_cls
        self.module = self._build_module(
            conv_cls=conv_cls,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            nn_module=nn_module,
        )

    @staticmethod
    def _build_module(
        conv_cls: Any,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        nn_module: Any,
    ) -> Any:
        class GNNRegressor(nn_module.Module):
            def __init__(self) -> None:
                super().__init__()
                layer_dims: list[tuple[int, int]] = []
                if num_layers == 1:
                    layer_dims.append((input_dim, 1))
                else:
                    layer_dims.append((input_dim, hidden_dim))
                    layer_dims.extend((hidden_dim, hidden_dim) for _ in range(num_layers - 2))
                    layer_dims.append((hidden_dim, 1))
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
                return output.squeeze(-1)

        return GNNRegressor()


def train_gnn_node_utility_model(
    dataset: LoadedDataset,
    feature_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    budget: int,
    model_type: str = "graphsage",
    target_column: str = "label_score",
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-4,
    epochs: int = 100,
    top_fraction: float | None = None,
    top_n: int | None = None,
    max_nodes: int | None = None,
    random_seed: int = 42,
    cache_path: Path | None = None,
    node2vec_mode: str = "off",
    node2vec_config: dict[str, Any] | None = None,
) -> GNNTrainingResult:
    """Train a GNN regressor and derive a filtered candidate pool."""

    _validate_training_parameters(
        model_type=model_type,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
    )
    torch, nn_module, data_cls, sage_conv_cls, gcn_conv_cls = _import_gnn_dependencies()

    start = perf_counter()
    training_frame = _build_feature_matrix(feature_frame, label_frame)
    if target_column not in training_frame.columns:
        raise ValueError(f"label_frame must contain the requested target_column='{target_column}'.")

    ordered_frame = training_frame.sort_values(
        by="node_id",
        key=lambda values: values.map(_sort_key),
    ).reset_index(drop=True)
    node_ids = ordered_frame["node_id"].tolist()
    encoded_features = _one_hot_feature_frame(ordered_frame)
    feature_matrix = encoded_features.to_numpy(dtype=np.float32, copy=True)
    label_scores = ordered_frame[target_column].to_numpy(dtype=np.float32, copy=True)
    node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    edge_index = _build_edge_index(dataset, node_to_index)

    cache_config = {
        "backend": "gnn",
        "model_type": model_type,
        "target_column": target_column,
        "budget": int(budget),
        "top_fraction": None if top_fraction is None else float(top_fraction),
        "top_n": None if top_n is None else int(top_n),
        "max_nodes": None if max_nodes is None else int(max_nodes),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "epochs": int(epochs),
        "random_seed": int(random_seed),
        "node2vec_mode": node2vec_mode,
        "node2vec_config": node2vec_config,
        "feature_columns": encoded_features.columns.tolist(),
    }
    fingerprint = _build_cache_fingerprint(
        node_ids=node_ids,
        feature_matrix=feature_matrix,
        label_scores=label_scores,
        edge_index=edge_index,
        cache_config=cache_config,
    )
    if cache_path is not None:
        cached_result = _load_cached_training_result(
            cache_path=cache_path,
            expected_fingerprint=fingerprint,
            training_frame=ordered_frame,
        )
        if cached_result is not None:
            return cached_result

    torch.manual_seed(int(random_seed))
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    data = data_cls(
        x=torch.tensor(feature_matrix, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(label_scores, dtype=torch.float32),
    )

    train_frame, validation_frame = _split_training_frame(ordered_frame, random_seed=random_seed)
    train_indices = torch.tensor(
        [node_to_index[node_id] for node_id in train_frame["node_id"]],
        dtype=torch.long,
    )
    validation_indices = torch.tensor(
        [node_to_index[node_id] for node_id in validation_frame["node_id"]],
        dtype=torch.long,
    )

    model_builder = _NodeScoreGNNModel(
        model_type=model_type,
        input_dim=int(feature_matrix.shape[1]),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        nn_module=nn_module,
        sage_conv_cls=sage_conv_cls,
        gcn_conv_cls=gcn_conv_cls,
    )
    model = model_builder.module
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    loss_fn = nn_module.MSELoss()

    best_state: dict[str, Any] | None = None
    best_validation_loss = float("inf")
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        predictions = model(data.x, data.edge_index)
        train_loss = loss_fn(predictions[train_indices], data.y[train_indices])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_predictions = model(data.x, data.edge_index)
            validation_loss = float(
                loss_fn(
                    validation_predictions[validation_indices],
                    data.y[validation_indices],
                ).item()
            )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        predicted_tensor = model(data.x, data.edge_index).detach().cpu()

    predicted_array = predicted_tensor.numpy().astype(float, copy=False)
    validation_prediction_array = predicted_array[validation_indices.detach().cpu().numpy()]
    validation_predictions = pd.Series(
        validation_prediction_array,
        index=validation_frame.index,
        dtype=float,
    )
    validation_spearman = _safe_spearman(validation_frame[target_column], validation_predictions)
    validation_precision_at_budget = _precision_at_k(
        node_ids=validation_frame["node_id"],
        true_scores=validation_frame[target_column],
        predicted_scores=validation_predictions,
        k=min(budget, len(validation_frame)),
    )

    predicted_scores = {
        node_id: float(score)
        for node_id, score in zip(node_ids, predicted_array.tolist(), strict=True)
    }
    ranked_nodes = _rank_nodes(predicted_scores)
    candidate_nodes = select_ml_candidate_nodes(
        ranked_nodes=ranked_nodes,
        budget=budget,
        top_fraction=top_fraction,
        top_n=top_n,
        max_nodes=max_nodes,
    )
    runtime_seconds = perf_counter() - start
    result = GNNTrainingResult(
        model=model,
        training_frame=ordered_frame,
        predicted_scores=predicted_scores,
        ranked_nodes=ranked_nodes,
        candidate_nodes=candidate_nodes,
        validation_spearman=validation_spearman,
        validation_precision_at_budget=validation_precision_at_budget,
        runtime_seconds=runtime_seconds,
        model_type=model_type,
        cache_path=cache_path,
        loaded_from_cache=False,
        feature_matrix_shape=(int(feature_matrix.shape[0]), int(feature_matrix.shape[1])),
        edge_index_shape=(int(edge_index.shape[0]), int(edge_index.shape[1])),
    )
    if cache_path is not None:
        _store_cached_training_result(cache_path=cache_path, fingerprint=fingerprint, result=result)
    return result
