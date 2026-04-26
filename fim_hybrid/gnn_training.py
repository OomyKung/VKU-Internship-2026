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
    _is_protected_feature_column,
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
_SUPPORTED_GNN_DEBIAS_MODES = {
    "none",
    "off",
    "class_weighted",
    "focal_loss",
    "worst_group_boost",
    "group_dro",
    "adversarial",
}


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
    target_column: str = "label_score"
    debias_mode: str = "none"


def _normalize_debias_mode(debias_mode: str) -> str:
    normalized = str(debias_mode).strip().lower()
    if normalized not in _SUPPORTED_GNN_DEBIAS_MODES:
        raise ValueError(
            f"debias_mode must be one of {sorted(_SUPPORTED_GNN_DEBIAS_MODES)}."
        )
    return "none" if normalized == "off" else normalized


def _binary_priority_targets(
    training_frame: pd.DataFrame,
    *,
    budget: int,
    target_column: str,
) -> pd.Series:
    if target_column not in training_frame.columns:
        raise ValueError(f"training_frame must contain target_column='{target_column}'.")
    if budget < 1:
        raise ValueError("budget must be at least 1.")
    if len(training_frame) < 2:
        raise ValueError("GNN ranking requires at least two nodes.")
    positive_count = min(max(int(budget), 1), len(training_frame) - 1)
    ordered = training_frame.sort_values(by=[target_column, "node_id"], ascending=[False, True])
    targets = pd.Series(0, index=training_frame.index, dtype=int)
    targets.loc[ordered.index[:positive_count]] = 1
    if targets.nunique() < 2:
        raise ValueError("Priority targets are degenerate; ranking requires both positive and negative labels.")
    return targets


def _binary_class_weights(binary_targets: np.ndarray) -> np.ndarray:
    counts = np.bincount(binary_targets.astype(int), minlength=2).astype(float)
    weights = np.ones_like(counts, dtype=float)
    nonzero = counts > 0
    if not np.all(nonzero):
        raise ValueError("Binary priority targets must contain both classes.")
    weights[nonzero] = float(np.sum(counts)) / (float(len(counts)) * counts[nonzero])
    return weights


def _weighted_mean(values: Any, weights: Any) -> Any:
    weight_sum = weights.sum()
    if float(weight_sum.detach().cpu().item()) <= 0.0:
        return values.mean()
    return (values * weights).sum() / weight_sum


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


def _one_hot_feature_frame(
    training_frame: pd.DataFrame,
    *,
    allow_protected_features_in_ml: bool = False,
) -> pd.DataFrame:
    categorical_columns = (
        list(_CATEGORICAL_COLUMNS)
        if bool(allow_protected_features_in_ml)
        else [column_name for column_name in _CATEGORICAL_COLUMNS if column_name != "protected_group"]
    )
    numeric_columns = [
        column_name
        for column_name in training_frame.columns
        if column_name not in _LABEL_EXCLUDED_COLUMNS and column_name not in _CATEGORICAL_COLUMNS
        and (bool(allow_protected_features_in_ml) or not _is_protected_feature_column(column_name))
    ]
    numeric_frame = training_frame.loc[:, numeric_columns].astype(float)
    categorical_frame = pd.get_dummies(
        training_frame.loc[:, categorical_columns].astype(str),
        columns=categorical_columns,
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
        target_column=str(metadata.get("target_column", "label_score")),
        debias_mode=str(metadata.get("debias_mode", "none")),
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
            "target_column": result.target_column,
            "debias_mode": result.debias_mode,
        },
        "predicted_scores": result.predicted_scores,
        "ranked_nodes": list(result.ranked_nodes),
        "candidate_nodes": list(result.candidate_nodes),
    }
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle)


class _NodeScoreGNNModel:
    """Thin wrapper that builds a small GraphSAGE or GCN encoder plus score head."""

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
                output_dim = max(int(hidden_dim), 1)
                layer_dims.append((input_dim, output_dim))
                layer_dims.extend((output_dim, output_dim) for _ in range(max(0, num_layers - 1)))
                self.convs = nn_module.ModuleList(
                    [conv_cls(in_dim, out_dim) for in_dim, out_dim in layer_dims]
                )
                self.dropout = float(dropout)
                self.score_head = nn_module.Linear(output_dim, 1)

            def encode(self, x: Any, edge_index: Any) -> Any:
                output = x
                for layer_index, conv in enumerate(self.convs):
                    output = conv(output, edge_index)
                    output = nn_module.functional.relu(output)
                    if self.dropout > 0.0 and layer_index < len(self.convs) - 1:
                        output = nn_module.functional.dropout(
                            output,
                            p=self.dropout,
                            training=self.training,
                        )
                return output

            def predict_scores(self, embeddings: Any) -> Any:
                return self.score_head(embeddings).squeeze(-1)

            def forward(self, x: Any, edge_index: Any) -> Any:
                output = self.encode(x, edge_index)
                output = self.predict_scores(output)
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
    debias_mode: str = "none",
    protected_attribute_column: str | None = "protected_group",
    focal_gamma: float = 2.0,
    group_robust_weight: float = 0.25,
    worst_group_boost_factor: float = 2.0,
    adversary_loss_weight: float = 0.1,
    allow_protected_features_in_ml: bool = False,
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
    resolved_debias_mode = _normalize_debias_mode(debias_mode)
    if float(group_robust_weight) < 0.0:
        raise ValueError("group_robust_weight must be non-negative.")
    if float(worst_group_boost_factor) < 1.0:
        raise ValueError("worst_group_boost_factor must be at least 1.0.")
    if float(focal_gamma) < 0.0:
        raise ValueError("focal_gamma must be non-negative.")
    if float(adversary_loss_weight) < 0.0:
        raise ValueError("adversary_loss_weight must be non-negative.")

    start = perf_counter()
    training_frame = _build_feature_matrix(feature_frame, label_frame)
    if target_column not in training_frame.columns:
        raise ValueError(f"label_frame must contain the requested target_column='{target_column}'.")

    ordered_frame = training_frame.sort_values(
        by="node_id",
        key=lambda values: values.map(_sort_key),
    ).reset_index(drop=True)
    node_ids = ordered_frame["node_id"].tolist()
    encoded_features = _one_hot_feature_frame(
        ordered_frame,
        allow_protected_features_in_ml=bool(allow_protected_features_in_ml),
    )
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
        "debias_mode": resolved_debias_mode,
        "protected_attribute_column": protected_attribute_column,
        "allow_protected_features_in_ml": bool(allow_protected_features_in_ml),
        "focal_gamma": float(focal_gamma),
        "group_robust_weight": float(group_robust_weight),
        "worst_group_boost_factor": float(worst_group_boost_factor),
        "adversary_loss_weight": float(adversary_loss_weight),
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
    priority_targets = _binary_priority_targets(
        ordered_frame,
        budget=budget,
        target_column=target_column,
    )
    priority_tensor = torch.tensor(priority_targets.to_numpy(dtype=np.int64), dtype=torch.long)
    binary_class_weights = torch.tensor(
        _binary_class_weights(priority_targets.to_numpy(dtype=np.int64)),
        dtype=torch.float32,
    )
    protected_labels = None
    if protected_attribute_column is not None:
        if protected_attribute_column in ordered_frame.columns:
            protected_labels = ordered_frame[protected_attribute_column].astype(str).copy()
        elif protected_attribute_column in dataset.node_attributes.columns:
            protected_labels = (
                dataset.node_attributes.set_index("node_id", drop=False)
                .reindex(node_ids)[protected_attribute_column]
                .astype(str)
            )
    if protected_labels is not None and protected_labels.isna().any():
        protected_labels = None
    if resolved_debias_mode in {"worst_group_boost", "group_dro", "adversarial"} and protected_labels is None:
        raise ValueError(
            f"debias_mode='{resolved_debias_mode}' requires protected labels via protected_attribute_column."
        )

    protected_tensor = None
    train_worst_group_weights = None
    if protected_labels is not None:
        protected_names = tuple(sorted(protected_labels.unique().tolist()))
        protected_to_index = {group_name: index for index, group_name in enumerate(protected_names)}
        protected_indices = np.asarray(
            [protected_to_index[group_name] for group_name in protected_labels.tolist()],
            dtype=np.int64,
        )
        protected_tensor = torch.tensor(protected_indices, dtype=torch.long)
        if resolved_debias_mode == "worst_group_boost":
            train_group_mean = (
                train_frame.assign(_protected_group=protected_labels.loc[train_frame.index].tolist())
                .groupby("_protected_group", observed=False)[target_column]
                .mean()
            )
            worst_mean = float(train_group_mean.min())
            boosted_groups = {
                str(group_name)
                for group_name, mean_value in train_group_mean.items()
                if float(mean_value) <= worst_mean + 1e-12
            }
            train_worst_group_weights = torch.tensor(
                [
                    float(worst_group_boost_factor)
                    if str(protected_labels.iloc[int(index)]) in boosted_groups
                    else 1.0
                    for index in train_indices.detach().cpu().numpy().tolist()
                ],
                dtype=torch.float32,
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
    embedding_dim = int(model.score_head.in_features)
    auxiliary_head = None if resolved_debias_mode != "focal_loss" else nn_module.Linear(embedding_dim, 1)
    adversary_head = (
        None
        if resolved_debias_mode != "adversarial" or protected_tensor is None
        else nn_module.Linear(embedding_dim, int(len(protected_to_index)))
    )

    trainable_parameters = list(model.parameters())
    if auxiliary_head is not None:
        trainable_parameters.extend(auxiliary_head.parameters())
    if adversary_head is not None:
        trainable_parameters.extend(adversary_head.parameters())
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    class _GradientReversalFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, input_tensor: Any, lambda_value: float) -> Any:
            ctx.lambda_value = float(lambda_value)
            return input_tensor.view_as(input_tensor)

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, None]:
            return grad_output.neg() * ctx.lambda_value, None

    def _apply_gradient_reversal(input_tensor: Any, *, lambda_value: float) -> Any:
        return _GradientReversalFunction.apply(input_tensor, float(lambda_value))

    best_state: dict[str, Any] | None = None
    best_validation_loss = float("inf")
    group_dro_state = (
        torch.ones(int(len(protected_to_index)), dtype=torch.float32)
        if resolved_debias_mode == "group_dro" and protected_tensor is not None
        else None
    )
    for _ in range(int(epochs)):
        model.train()
        if auxiliary_head is not None:
            auxiliary_head.train()
        if adversary_head is not None:
            adversary_head.train()
        optimizer.zero_grad()
        embeddings = model.encode(data.x, data.edge_index)
        predictions = model.predict_scores(embeddings)
        train_predictions = predictions[train_indices]
        train_targets = data.y[train_indices]
        loss_vector = nn_module.functional.mse_loss(
            train_predictions,
            train_targets,
            reduction="none",
        )
        train_loss = loss_vector.mean()

        if resolved_debias_mode == "class_weighted":
            sample_weights = binary_class_weights[priority_tensor[train_indices]]
            train_loss = _weighted_mean(loss_vector, sample_weights)
        elif resolved_debias_mode == "worst_group_boost":
            if train_worst_group_weights is None:
                raise ValueError("worst_group_boost requires protected labels.")
            train_loss = _weighted_mean(loss_vector, train_worst_group_weights)
        elif resolved_debias_mode == "group_dro":
            if protected_tensor is None or group_dro_state is None:
                raise ValueError("group_dro requires protected labels.")
            train_group_ids = protected_tensor[train_indices]
            active_group_indices: list[int] = []
            group_losses: list[Any] = []
            for group_index in range(int(group_dro_state.shape[0])):
                mask = train_group_ids == int(group_index)
                if not bool(mask.any().detach().cpu().item()):
                    continue
                active_group_indices.append(group_index)
                group_losses.append(loss_vector[mask].mean())
            if group_losses:
                group_loss_tensor = torch.stack(group_losses)
                active_state = group_dro_state[active_group_indices]
                active_state = active_state * torch.exp(float(group_robust_weight) * group_loss_tensor.detach())
                active_state = active_state / active_state.sum()
                updated_state = group_dro_state.clone()
                updated_state[active_group_indices] = active_state
                group_dro_state = updated_state
                train_loss = (group_loss_tensor * active_state).sum()

        if resolved_debias_mode == "focal_loss":
            if auxiliary_head is None:
                raise RuntimeError("focal_loss requires an auxiliary priority head.")
            aux_logits = auxiliary_head(embeddings).squeeze(-1)
            train_priority_targets = priority_tensor[train_indices].to(dtype=torch.float32)
            train_aux_logits = aux_logits[train_indices]
            bce_vector = nn_module.functional.binary_cross_entropy_with_logits(
                train_aux_logits,
                train_priority_targets,
                reduction="none",
            )
            positive_weight = float(binary_class_weights[1].item())
            negative_weight = float(binary_class_weights[0].item())
            sample_weights = torch.where(
                train_priority_targets > 0.5,
                torch.full_like(train_priority_targets, positive_weight),
                torch.full_like(train_priority_targets, negative_weight),
            )
            probabilities = torch.sigmoid(train_aux_logits)
            pt = torch.where(train_priority_targets > 0.5, probabilities, 1.0 - probabilities)
            focal_factor = torch.pow(1.0 - pt.clamp(min=1e-6, max=1.0), float(focal_gamma))
            auxiliary_loss = _weighted_mean(bce_vector * focal_factor, sample_weights)
            train_loss = train_loss + auxiliary_loss

        if resolved_debias_mode == "adversarial":
            if adversary_head is None or protected_tensor is None:
                raise ValueError("adversarial debiasing requires protected labels.")
            adversary_logits = adversary_head(
                _apply_gradient_reversal(embeddings[train_indices], lambda_value=1.0)
            )
            adversary_loss = nn_module.functional.cross_entropy(
                adversary_logits,
                protected_tensor[train_indices],
            )
            train_loss = train_loss + (float(adversary_loss_weight) * adversary_loss)

        train_loss.backward()
        optimizer.step()

        model.eval()
        if auxiliary_head is not None:
            auxiliary_head.eval()
        if adversary_head is not None:
            adversary_head.eval()
        with torch.no_grad():
            validation_embeddings = model.encode(data.x, data.edge_index)
            validation_predictions = model.predict_scores(validation_embeddings)
            validation_loss = float(
                nn_module.functional.mse_loss(
                    validation_predictions[validation_indices],
                    data.y[validation_indices],
                ).item()
            )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                "model": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "auxiliary_head": (
                    None
                    if auxiliary_head is None
                    else {
                        key: value.detach().cpu().clone()
                        for key, value in auxiliary_head.state_dict().items()
                    }
                ),
                "adversary_head": (
                    None
                    if adversary_head is None
                    else {
                        key: value.detach().cpu().clone()
                        for key, value in adversary_head.state_dict().items()
                    }
                ),
            }

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        if auxiliary_head is not None and best_state["auxiliary_head"] is not None:
            auxiliary_head.load_state_dict(best_state["auxiliary_head"])
        if adversary_head is not None and best_state["adversary_head"] is not None:
            adversary_head.load_state_dict(best_state["adversary_head"])

    model.eval()
    with torch.no_grad():
        final_embeddings = model.encode(data.x, data.edge_index)
        predicted_tensor = model.predict_scores(final_embeddings).detach().cpu()

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
        target_column=target_column,
        debias_mode=resolved_debias_mode,
    )
    if cache_path is not None:
        _store_cached_training_result(cache_path=cache_path, fingerprint=fingerprint, result=result)
    return result
