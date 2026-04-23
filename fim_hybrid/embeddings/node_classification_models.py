"""Supervised node-classification helpers for embedding evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from ._pyg_common import require_pyg_dependencies, seed_torch
from .base import sorted_node_ids
from .evaluation_splits import NodeClassificationSplit, NodeTrainValidationSplit
from .features import coerce_input_features

SUPPORTED_NODE_CLASSIFICATION_MODELS = ("logistic_regression", "graphsage")
SUPPORTED_TRAINING_MODES = (
    "baseline",
    "imbalance_only",
    "group_robust",
    "adversarial",
    "adversarial_group_robust",
    "support_aware_group_robust",
    "anti_collapse_group_robust",
    "anti_collapse_group_robust_with_mild_adversarial",
)
SUPPORTED_DEBIAS_MODES = ("none", "adversarial")
SUPPORTED_IMBALANCE_MODES = ("none", "class_weighted", "focal_loss", "weighted_sampler")
SUPPORTED_GROUP_WEIGHT_MODES = ("none", "inverse_frequency", "worst_group_boost", "group_dro", "min_support_boost")
SUPPORTED_PROBE_MODEL_TYPES = ("linear", "nonlinear")
SUPPORTED_EARLY_STOP_METRICS = (
    "accuracy",
    "macro_f1",
    "worst_group_f1",
    "worst_group_f1_raw",
    "worst_group_f1_supported",
    "fairness_score",
)

_DEFAULT_HIDDEN_DIM = 64
_DEFAULT_NUM_LAYERS = 2
_DEFAULT_DROPOUT = 0.2
_DEFAULT_LEARNING_RATE = 1e-3
_DEFAULT_WEIGHT_DECAY = 5e-4
_DEFAULT_EPOCHS = 100
_DEFAULT_UNDERREPRESENTED_COUNT_THRESHOLD = 5
_DEFAULT_MIN_GROUP_SUPPORT_TRAIN = 10
_DEFAULT_MIN_GROUP_SUPPORT_EVAL = 10
_DEFAULT_MIN_SUPPORT_BOOST_FACTOR = 2.0

_FIXED_TRAINING_MODE_BUNDLES: dict[str, dict[str, object]] = {
    "support_aware_group_robust": {
        "imbalance_mode": "none",
        "debias_mode": "none",
        "group_weight_mode": "inverse_frequency",
        "group_robust_weight": 0.25,
        "rebalance_batches_by_group": False,
        "early_stop_metric": "worst_group_f1_supported",
    },
    "anti_collapse_group_robust": {
        "imbalance_mode": "none",
        "debias_mode": "none",
        "group_weight_mode": "min_support_boost",
        "group_robust_weight": 0.25,
        "min_support_boost_factor": _DEFAULT_MIN_SUPPORT_BOOST_FACTOR,
        "rebalance_batches_by_group": True,
        "early_stop_metric": "worst_group_f1_raw",
    },
    "anti_collapse_group_robust_with_mild_adversarial": {
        "imbalance_mode": "none",
        "debias_mode": "adversarial",
        "group_weight_mode": "min_support_boost",
        "group_robust_weight": 0.25,
        "min_support_boost_factor": _DEFAULT_MIN_SUPPORT_BOOST_FACTOR,
        "rebalance_batches_by_group": True,
        "early_stop_metric": "fairness_score",
        "adversary_loss_weight": 0.02,
        "gradient_reversal_lambda": 0.02,
        "adversary_hidden_dim": _DEFAULT_HIDDEN_DIM,
        "adversary_num_layers": 1,
        "adversary_dropout": 0.1,
    },
}


@dataclass(frozen=True, slots=True)
class GraphSAGENodeClassificationConfig:
    """Configuration for the supervised GraphSAGE evaluator."""

    training_mode: str | None = None
    imbalance_mode: str = "none"
    focal_gamma: float = 2.0
    early_stop_metric: str = "accuracy"
    early_stop_patience: int = 0
    class_weight_smoothing: float = 0.0
    debias_mode: str = "none"
    group_robust_weight: float = 0.0
    group_weight_mode: str = "none"
    worst_group_boost_factor: float = 2.0
    min_support_boost_factor: float = _DEFAULT_MIN_SUPPORT_BOOST_FACTOR
    min_group_support_threshold: int | None = None
    min_group_support_train: int | None = None
    min_group_support_eval: int | None = None
    report_small_group_metrics: bool = False
    rebalance_batches_by_group: bool = False
    fairness_score_alpha: float = 0.25
    fairness_score_beta: float = 0.25
    probe_model_type: str = "linear"
    adversary_loss_weight: float = 1.0
    gradient_reversal_lambda: float = 1.0
    adversary_warmup_epochs: int = 0
    group_robust_warmup_epochs: int = 0
    adversary_hidden_dim: int = _DEFAULT_HIDDEN_DIM
    adversary_num_layers: int = 1
    adversary_dropout: float = _DEFAULT_DROPOUT
    hidden_dim: int = _DEFAULT_HIDDEN_DIM
    num_layers: int = _DEFAULT_NUM_LAYERS
    dropout: float = _DEFAULT_DROPOUT
    learning_rate: float = _DEFAULT_LEARNING_RATE
    weight_decay: float = _DEFAULT_WEIGHT_DECAY
    epochs: int = _DEFAULT_EPOCHS
    random_seed: int = 42


@dataclass(frozen=True, slots=True)
class GraphSAGENodeClassificationResult:
    """Predictions plus training metadata for the GraphSAGE evaluator."""

    predictions: np.ndarray
    node_embeddings: np.ndarray
    validation_loss: float
    validation_metric_value: float
    validation_metric_name: str
    validation_count: int
    train_count: int
    validation_stratified: bool
    best_epoch: int
    training_history_json: str


def resolve_node_classification_model(model_name: str) -> str:
    """Validate and normalize a node-classification model name."""

    normalized = str(model_name).strip().lower()
    if normalized not in SUPPORTED_NODE_CLASSIFICATION_MODELS:
        raise ValueError(
            f"Unsupported node_classification_model '{model_name}'. "
            f"Supported values: {list(SUPPORTED_NODE_CLASSIFICATION_MODELS)}."
        )
    return normalized


def _group_robust_enabled(*, group_weight_mode: str, group_robust_weight: float) -> bool:
    """Return whether protected-group robust weighting is active."""

    return str(group_weight_mode).strip().lower() != "none" and float(group_robust_weight) > 0.0


def _normalize_training_mode_name(training_mode: str | None) -> str | None:
    if training_mode is None:
        return None
    return str(training_mode).strip().lower()


def resolve_graphsage_training_mode_settings(
    *,
    training_mode: str | None,
    imbalance_mode: str,
    debias_mode: str,
    group_weight_mode: str,
    group_robust_weight: float,
    worst_group_boost_factor: float,
    min_support_boost_factor: float,
    rebalance_batches_by_group: bool,
    early_stop_metric: str,
    adversary_loss_weight: float,
    gradient_reversal_lambda: float,
    adversary_hidden_dim: int,
    adversary_num_layers: int,
    adversary_dropout: float,
) -> dict[str, object]:
    """Apply any fixed training-mode bundle overrides before validation."""

    normalized_training_mode = _normalize_training_mode_name(training_mode)
    resolved: dict[str, object] = {
        "training_mode": normalized_training_mode,
        "imbalance_mode": imbalance_mode,
        "debias_mode": debias_mode,
        "group_weight_mode": group_weight_mode,
        "group_robust_weight": float(group_robust_weight),
        "worst_group_boost_factor": float(worst_group_boost_factor),
        "min_support_boost_factor": float(min_support_boost_factor),
        "rebalance_batches_by_group": bool(rebalance_batches_by_group),
        "early_stop_metric": early_stop_metric,
        "adversary_loss_weight": float(adversary_loss_weight),
        "gradient_reversal_lambda": float(gradient_reversal_lambda),
        "adversary_hidden_dim": int(adversary_hidden_dim),
        "adversary_num_layers": int(adversary_num_layers),
        "adversary_dropout": float(adversary_dropout),
    }
    if normalized_training_mode in _FIXED_TRAINING_MODE_BUNDLES:
        resolved.update(_FIXED_TRAINING_MODE_BUNDLES[normalized_training_mode])
        resolved["training_mode"] = normalized_training_mode
    return resolved


def infer_training_mode(
    *,
    imbalance_mode: str,
    debias_mode: str,
    group_weight_mode: str,
    group_robust_weight: float,
) -> str:
    """Infer the high-level training mode from low-level knobs."""

    group_robust_active = _group_robust_enabled(
        group_weight_mode=group_weight_mode,
        group_robust_weight=group_robust_weight,
    )
    if debias_mode == "adversarial" and group_robust_active:
        return "adversarial_group_robust"
    if debias_mode == "adversarial":
        return "adversarial"
    if group_robust_active:
        return "group_robust"
    if imbalance_mode != "none":
        return "imbalance_only"
    return "baseline"


def resolve_training_mode(
    training_mode: str | None,
    *,
    imbalance_mode: str,
    debias_mode: str,
    group_weight_mode: str,
    group_robust_weight: float,
) -> str:
    """Validate one training mode and ensure it matches the low-level configuration."""

    inferred_mode = infer_training_mode(
        imbalance_mode=imbalance_mode,
        debias_mode=debias_mode,
        group_weight_mode=group_weight_mode,
        group_robust_weight=group_robust_weight,
    )
    if training_mode is None:
        return inferred_mode
    normalized = _normalize_training_mode_name(training_mode)
    if normalized in {"", "auto"}:
        return inferred_mode
    if normalized not in SUPPORTED_TRAINING_MODES:
        raise ValueError(
            f"Unsupported training_mode '{training_mode}'. "
            f"Supported values: {list(SUPPORTED_TRAINING_MODES)}."
        )
    if normalized in _FIXED_TRAINING_MODE_BUNDLES:
        return normalized
    if normalized in {"group_robust", "adversarial_group_robust"}:
        if normalized == "adversarial_group_robust" and debias_mode != "adversarial":
            raise ValueError(
                "training_mode='adversarial_group_robust' requires debias_mode='adversarial'."
            )
        if normalized == "group_robust" and debias_mode == "adversarial":
            raise ValueError(
                "training_mode='group_robust' cannot be combined with debias_mode='adversarial'. "
                "Use training_mode='adversarial_group_robust' instead."
            )
        if not _group_robust_enabled(
            group_weight_mode=group_weight_mode,
            group_robust_weight=group_robust_weight,
        ):
            raise ValueError(
                f"training_mode='{normalized}' requires group_weight_mode to be active "
                "and group_robust_weight > 0."
            )
        return inferred_mode
    if normalized != inferred_mode:
        raise ValueError(
            "training_mode is inconsistent with the requested imbalance/debias/group-robust settings. "
            f"Expected training_mode='{inferred_mode}' for the current configuration."
        )
    return normalized


def derive_comparison_mode(
    *,
    debias_mode: str,
    group_weight_mode: str,
    group_robust_weight: float,
    training_mode: str | None = None,
) -> str:
    """Return the high-level comparison bucket for one GraphSAGE configuration."""

    normalized_training_mode = _normalize_training_mode_name(training_mode)
    if normalized_training_mode in _FIXED_TRAINING_MODE_BUNDLES:
        return str(normalized_training_mode)
    if _group_robust_enabled(
        group_weight_mode=group_weight_mode,
        group_robust_weight=group_robust_weight,
    ):
        if debias_mode == "adversarial":
            return "adversarial_group_robust"
        return "group_robust"
    if debias_mode == "adversarial":
        return "adversarial"
    return "baseline"


def resolve_imbalance_mode(imbalance_mode: str) -> str:
    """Validate and normalize one imbalance mode."""

    normalized = str(imbalance_mode).strip().lower()
    if normalized not in SUPPORTED_IMBALANCE_MODES:
        raise ValueError(
            f"Unsupported imbalance_mode '{imbalance_mode}'. "
            f"Supported values: {list(SUPPORTED_IMBALANCE_MODES)}."
        )
    return normalized


def resolve_debias_mode(debias_mode: str) -> str:
    """Validate and normalize one debias mode."""

    normalized = str(debias_mode).strip().lower()
    if normalized not in SUPPORTED_DEBIAS_MODES:
        raise ValueError(
            f"Unsupported debias_mode '{debias_mode}'. "
            f"Supported values: {list(SUPPORTED_DEBIAS_MODES)}."
        )
    return normalized


def resolve_group_weight_mode(group_weight_mode: str) -> str:
    """Validate and normalize one protected-group weighting mode."""

    normalized = str(group_weight_mode).strip().lower()
    if normalized not in SUPPORTED_GROUP_WEIGHT_MODES:
        raise ValueError(
            f"Unsupported group_weight_mode '{group_weight_mode}'. "
            f"Supported values: {list(SUPPORTED_GROUP_WEIGHT_MODES)}."
        )
    return normalized


def resolve_probe_model_type(model_type: str) -> str:
    """Validate and normalize one embedding-probe model type."""

    normalized = str(model_type).strip().lower()
    if normalized not in SUPPORTED_PROBE_MODEL_TYPES:
        raise ValueError(
            f"Unsupported probe_model_type '{model_type}'. "
            f"Supported values: {list(SUPPORTED_PROBE_MODEL_TYPES)}."
        )
    return normalized


def resolve_group_support_thresholds(
    *,
    min_group_support_threshold: int | None = None,
    min_group_support_train: int | None = None,
    min_group_support_eval: int | None = None,
) -> tuple[int, int]:
    """Resolve legacy and explicit support thresholds into train/eval values."""

    legacy_threshold = None if min_group_support_threshold is None else int(min_group_support_threshold)
    train_threshold = (
        int(min_group_support_train)
        if min_group_support_train is not None
        else legacy_threshold if legacy_threshold is not None else _DEFAULT_MIN_GROUP_SUPPORT_TRAIN
    )
    eval_threshold = (
        int(min_group_support_eval)
        if min_group_support_eval is not None
        else legacy_threshold if legacy_threshold is not None else _DEFAULT_MIN_GROUP_SUPPORT_EVAL
    )
    if train_threshold < 0:
        raise ValueError("min_group_support_train must be non-negative.")
    if eval_threshold < 0:
        raise ValueError("min_group_support_eval must be non-negative.")
    return train_threshold, eval_threshold


def resolve_early_stop_metric(metric_name: str) -> str:
    """Validate and normalize the early-stopping metric name."""

    normalized = str(metric_name).strip().lower()
    if normalized == "worst_group_f1":
        return "worst_group_f1_raw"
    if normalized not in SUPPORTED_EARLY_STOP_METRICS:
        raise ValueError(
            f"Unsupported early_stop_metric '{metric_name}'. "
            f"Supported values: {list(SUPPORTED_EARLY_STOP_METRICS)}."
        )
    return normalized


def validate_graphsage_node_classification_config(
    config: GraphSAGENodeClassificationConfig,
) -> GraphSAGENodeClassificationConfig:
    """Validate one GraphSAGE node-classification config."""

    resolved_settings = resolve_graphsage_training_mode_settings(
        training_mode=config.training_mode,
        imbalance_mode=config.imbalance_mode,
        debias_mode=config.debias_mode,
        group_weight_mode=config.group_weight_mode,
        group_robust_weight=config.group_robust_weight,
        worst_group_boost_factor=config.worst_group_boost_factor,
        min_support_boost_factor=config.min_support_boost_factor,
        rebalance_batches_by_group=config.rebalance_batches_by_group,
        early_stop_metric=config.early_stop_metric,
        adversary_loss_weight=config.adversary_loss_weight,
        gradient_reversal_lambda=config.gradient_reversal_lambda,
        adversary_hidden_dim=config.adversary_hidden_dim,
        adversary_num_layers=config.adversary_num_layers,
        adversary_dropout=config.adversary_dropout,
    )
    imbalance_mode = resolve_imbalance_mode(str(resolved_settings["imbalance_mode"]))
    debias_mode = resolve_debias_mode(str(resolved_settings["debias_mode"]))
    group_weight_mode = resolve_group_weight_mode(str(resolved_settings["group_weight_mode"]))
    probe_model_type = resolve_probe_model_type(config.probe_model_type)
    early_stop_metric = resolve_early_stop_metric(str(resolved_settings["early_stop_metric"]))
    min_group_support_train, min_group_support_eval = resolve_group_support_thresholds(
        min_group_support_threshold=config.min_group_support_threshold,
        min_group_support_train=config.min_group_support_train,
        min_group_support_eval=config.min_group_support_eval,
    )
    training_mode = resolve_training_mode(
        resolved_settings["training_mode"],
        imbalance_mode=imbalance_mode,
        debias_mode=debias_mode,
        group_weight_mode=group_weight_mode,
        group_robust_weight=float(resolved_settings["group_robust_weight"]),
    )
    if config.focal_gamma < 0.0:
        raise ValueError("focal_gamma must be non-negative.")
    if config.early_stop_patience < 0:
        raise ValueError("early_stop_patience must be non-negative.")
    if config.class_weight_smoothing < 0.0:
        raise ValueError("class_weight_smoothing must be non-negative.")
    if float(resolved_settings["group_robust_weight"]) < 0.0:
        raise ValueError("group_robust_weight must be non-negative.")
    if float(resolved_settings["worst_group_boost_factor"]) < 1.0:
        raise ValueError("worst_group_boost_factor must be at least 1.0.")
    if float(resolved_settings["min_support_boost_factor"]) < 1.0:
        raise ValueError("min_support_boost_factor must be at least 1.0.")
    if config.fairness_score_alpha < 0.0:
        raise ValueError("fairness_score_alpha must be non-negative.")
    if config.fairness_score_beta < 0.0:
        raise ValueError("fairness_score_beta must be non-negative.")
    if float(resolved_settings["adversary_loss_weight"]) < 0.0:
        raise ValueError("adversary_loss_weight must be non-negative.")
    if float(resolved_settings["gradient_reversal_lambda"]) < 0.0:
        raise ValueError("gradient_reversal_lambda must be non-negative.")
    if config.adversary_warmup_epochs < 0:
        raise ValueError("adversary_warmup_epochs must be non-negative.")
    if config.group_robust_warmup_epochs < 0:
        raise ValueError("group_robust_warmup_epochs must be non-negative.")
    if int(resolved_settings["adversary_hidden_dim"]) < 1:
        raise ValueError("adversary_hidden_dim must be at least 1.")
    if int(resolved_settings["adversary_num_layers"]) < 1:
        raise ValueError("adversary_num_layers must be at least 1.")
    if not 0.0 <= float(resolved_settings["adversary_dropout"]) < 1.0:
        raise ValueError("adversary_dropout must be in the interval [0.0, 1.0).")
    if config.hidden_dim < 1:
        raise ValueError("hidden_dim must be at least 1.")
    if config.num_layers < 1:
        raise ValueError("num_layers must be at least 1.")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in the interval [0.0, 1.0).")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1.")
    return GraphSAGENodeClassificationConfig(
        training_mode=training_mode,
        imbalance_mode=imbalance_mode,
        focal_gamma=float(config.focal_gamma),
        early_stop_metric=early_stop_metric,
        early_stop_patience=int(config.early_stop_patience),
        class_weight_smoothing=float(config.class_weight_smoothing),
        debias_mode=debias_mode,
        group_robust_weight=float(resolved_settings["group_robust_weight"]),
        group_weight_mode=group_weight_mode,
        worst_group_boost_factor=float(resolved_settings["worst_group_boost_factor"]),
        min_support_boost_factor=float(resolved_settings["min_support_boost_factor"]),
        min_group_support_threshold=(
            None
            if config.min_group_support_threshold is None
            else int(config.min_group_support_threshold)
        ),
        min_group_support_train=int(min_group_support_train),
        min_group_support_eval=int(min_group_support_eval),
        report_small_group_metrics=bool(config.report_small_group_metrics),
        rebalance_batches_by_group=bool(resolved_settings["rebalance_batches_by_group"]),
        fairness_score_alpha=float(config.fairness_score_alpha),
        fairness_score_beta=float(config.fairness_score_beta),
        probe_model_type=probe_model_type,
        adversary_loss_weight=float(resolved_settings["adversary_loss_weight"]),
        gradient_reversal_lambda=float(resolved_settings["gradient_reversal_lambda"]),
        adversary_warmup_epochs=int(config.adversary_warmup_epochs),
        group_robust_warmup_epochs=int(config.group_robust_warmup_epochs),
        adversary_hidden_dim=int(resolved_settings["adversary_hidden_dim"]),
        adversary_num_layers=int(resolved_settings["adversary_num_layers"]),
        adversary_dropout=float(resolved_settings["adversary_dropout"]),
        hidden_dim=int(config.hidden_dim),
        num_layers=int(config.num_layers),
        dropout=float(config.dropout),
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        epochs=int(config.epochs),
        random_seed=int(config.random_seed),
    )


def compute_classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
) -> dict[str, float]:
    """Compute the shared classification metrics for one prediction set."""

    resolved_labels = _resolve_metric_labels(y_true, y_pred, labels=labels)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=resolved_labels,
        average="macro",
        zero_division=0.0,
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=resolved_labels,
        average="micro",
        zero_division=0.0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=resolved_labels,
        average="weighted",
        zero_division=0.0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": _balanced_accuracy_from_labels(y_true, y_pred, resolved_labels),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
    }


def _resolve_metric_labels(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
) -> list[str]:
    """Return deterministic classification labels for metric computation."""

    if labels is not None:
        return [str(label) for label in labels]
    return sorted({str(label) for label in y_true} | {str(label) for label in y_pred})


def _json_dumps(payload: Any) -> str:
    """Serialize one metrics payload deterministically."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _string_series(values: Sequence[Any], *, name: str) -> pd.Series:
    """Normalize one sequence into a deterministic string series."""

    return pd.Series([str(value) for value in values], dtype=str, name=name)


def _count_payload(series: pd.Series) -> dict[str, int]:
    """Return deterministic string counts for one series."""

    counts = series.value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _matrix_payload(
    row_values: Sequence[Any],
    column_values: Sequence[Any],
    *,
    row_name: str,
    column_name: str,
) -> dict[str, Any]:
    """Return a deterministic contingency-table payload."""

    row_series = _string_series(row_values, name=row_name)
    column_series = _string_series(column_values, name=column_name)
    if len(row_series) != len(column_series):
        raise ValueError(f"{row_name} and {column_name} must have the same length.")
    table = pd.crosstab(row_series, column_series, dropna=False)
    row_labels = [str(label) for label in table.index.tolist()]
    column_labels = [str(label) for label in table.columns.tolist()]
    return {
        "row_label": row_name,
        "column_label": column_name,
        "rows": row_labels,
        "columns": column_labels,
        "matrix": table.to_numpy(dtype=int).tolist(),
    }


def _balanced_accuracy_from_labels(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any],
) -> float:
    """Compute balanced accuracy deterministically across explicit labels."""

    _, recalls, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[str(label) for label in labels],
        average=None,
        zero_division=0.0,
    )
    if len(recalls) == 0:
        return 0.0
    return float(np.mean(recalls))


def per_class_metrics_payload(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic per-class precision/recall/F1 metrics."""

    resolved_labels = _resolve_metric_labels(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=resolved_labels,
        average=None,
        zero_division=0.0,
    )
    return [
        {
            "label": resolved_labels[index],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in range(len(resolved_labels))
    ]


def confusion_matrix_payload(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Return a deterministic confusion-matrix payload."""

    resolved_labels = _resolve_metric_labels(y_true, y_pred, labels=labels)
    true_categorical = pd.Categorical([str(value) for value in y_true], categories=resolved_labels)
    pred_categorical = pd.Categorical([str(value) for value in y_pred], categories=resolved_labels)
    matrix = pd.crosstab(true_categorical, pred_categorical, dropna=False)
    return {
        "labels": resolved_labels,
        "matrix": matrix.to_numpy(dtype=int).tolist(),
    }


def class_distribution_payload(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Return deterministic true and predicted class distributions."""

    resolved_labels = _resolve_metric_labels(y_true, y_pred, labels=labels)
    true_counts = pd.Series([str(label) for label in y_true]).value_counts().to_dict()
    predicted_counts = pd.Series([str(label) for label in y_pred]).value_counts().to_dict()
    return {
        "labels": resolved_labels,
        "true_counts": {label: int(true_counts.get(label, 0)) for label in resolved_labels},
        "predicted_counts": {label: int(predicted_counts.get(label, 0)) for label in resolved_labels},
    }


def serialize_per_class_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
) -> str:
    """Serialize deterministic per-class precision/recall/F1 metrics."""

    return _json_dumps(per_class_metrics_payload(y_true, y_pred, labels=labels))


def serialize_confusion_matrix_summary(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
) -> str:
    """Serialize a deterministic confusion-matrix summary."""

    return _json_dumps(confusion_matrix_payload(y_true, y_pred, labels=labels))


def serialize_class_distribution_summary(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
) -> str:
    """Serialize true and predicted class distributions deterministically."""

    return _json_dumps(class_distribution_payload(y_true, y_pred, labels=labels))


def split_distribution_payload(
    label_values: Sequence[Any],
    *,
    protected_values: Sequence[Any] | None = None,
    underrepresented_threshold: int = _DEFAULT_UNDERREPRESENTED_COUNT_THRESHOLD,
) -> dict[str, Any]:
    """Return deterministic label/protected distribution diagnostics for one split."""

    label_series = _string_series(label_values, name="label")
    payload: dict[str, Any] = {
        "count": int(len(label_series)),
        "label_counts": _count_payload(label_series),
        "underrepresented_threshold": int(underrepresented_threshold),
    }
    if protected_values is None:
        return payload

    protected_series = _string_series(protected_values, name="protected_group")
    if len(label_series) != len(protected_series):
        raise ValueError("label_values and protected_values must have the same length.")
    protected_counts = _count_payload(protected_series)
    intersection_payload = _matrix_payload(
        label_series.tolist(),
        protected_series.tolist(),
        row_name="label",
        column_name="protected_group",
    )
    underrepresented_groups = [
        {"group": group_name, "count": count}
        for group_name, count in protected_counts.items()
        if count < int(underrepresented_threshold)
    ]
    underrepresented_intersections: list[dict[str, Any]] = []
    for row_index, label_name in enumerate(intersection_payload["rows"]):
        for column_index, group_name in enumerate(intersection_payload["columns"]):
            count = int(intersection_payload["matrix"][row_index][column_index])
            if count < int(underrepresented_threshold):
                underrepresented_intersections.append(
                    {
                        "label": label_name,
                        "group": group_name,
                        "count": count,
                    }
                )
    payload.update(
        {
            "protected_counts": protected_counts,
            "label_by_protected_counts": intersection_payload,
            "underrepresented_protected_groups": underrepresented_groups,
            "underrepresented_intersections": underrepresented_intersections,
        }
    )
    return payload


def serialize_split_distribution_summary(
    label_values: Sequence[Any],
    *,
    protected_values: Sequence[Any] | None = None,
    underrepresented_threshold: int = _DEFAULT_UNDERREPRESENTED_COUNT_THRESHOLD,
) -> str:
    """Serialize one split distribution payload deterministically."""

    return _json_dumps(
        split_distribution_payload(
            label_values,
            protected_values=protected_values,
            underrepresented_threshold=underrepresented_threshold,
        )
    )


def serialize_split_diagnostics(payload_by_split: Mapping[str, dict[str, Any]]) -> str:
    """Serialize deterministic diagnostics for train/validation/test splits."""

    normalized_payload = {str(split_name): dict(payload) for split_name, payload in payload_by_split.items()}
    return _json_dumps(normalized_payload)


def _prediction_entropy(counts: Mapping[str, int]) -> float:
    """Return the base-2 entropy of one predicted-label distribution."""

    total = float(sum(int(count) for count in counts.values()))
    if total <= 0.0:
        return 0.0
    probabilities = np.asarray(
        [float(count) / total for count in counts.values() if int(count) > 0],
        dtype=float,
    )
    if probabilities.size == 0:
        return 0.0
    return float(-(probabilities * np.log2(probabilities)).sum())


def _top_confusion_pairs(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> list[dict[str, Any]]:
    """Return deterministic non-diagonal confusion pairs ordered by count."""

    pair_counts: dict[tuple[str, str], int] = {}
    for true_label, predicted_label in zip(y_true, y_pred, strict=False):
        true_name = str(true_label)
        predicted_name = str(predicted_label)
        if true_name == predicted_name:
            continue
        pair_counts[(true_name, predicted_name)] = pair_counts.get((true_name, predicted_name), 0) + 1
    ordered_pairs = sorted(
        pair_counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )
    return [
        {"true_label": true_label, "predicted_label": predicted_label, "count": int(count)}
        for (true_label, predicted_label), count in ordered_pairs
    ]


def summarize_group_classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    group_values: Sequence[Any],
    *,
    min_group_support_threshold: int = 0,
    report_small_group_metrics: bool = False,
) -> tuple[dict[str, float | str], str, str, str]:
    """Return summary, per-group metrics, confusions, and error diagnostics."""

    true_series = pd.Series([str(value) for value in y_true], dtype=str)
    pred_series = pd.Series([str(value) for value in y_pred], dtype=str)
    group_series = pd.Series([str(value) for value in group_values], dtype=str)
    if not (len(true_series) == len(pred_series) == len(group_series)):
        raise ValueError("y_true, y_pred, and group_values must have the same length.")

    group_payloads: list[dict[str, Any]] = []
    group_confusion_payloads: list[dict[str, Any]] = []
    group_error_payloads: list[dict[str, Any]] = []
    accuracy_by_group: dict[str, float] = {}
    macro_f1_by_group: dict[str, float] = {}
    count_by_group: dict[str, int] = {}
    support_threshold = max(0, int(min_group_support_threshold))
    for group_name in sorted(group_series.unique().tolist()):
        mask = group_series == group_name
        group_true = true_series.loc[mask].tolist()
        group_pred = pred_series.loc[mask].tolist()
        labels = _resolve_metric_labels(group_true, group_pred)
        group_metrics = compute_classification_metrics(group_true, group_pred, labels=labels)
        group_distribution = class_distribution_payload(group_true, group_pred, labels=labels)
        group_confusion = confusion_matrix_payload(group_true, group_pred, labels=labels)
        misclassified_count = int(np.sum(true_series.loc[mask].to_numpy(dtype=object) != pred_series.loc[mask].to_numpy(dtype=object)))
        prediction_entropy = _prediction_entropy(group_distribution["predicted_counts"])
        predicted_unique_labels = sum(
            1 for count in group_distribution["predicted_counts"].values() if int(count) > 0
        )
        group_count = int(mask.sum())
        supported_group = group_count >= support_threshold
        accuracy_by_group[group_name] = float(group_metrics["accuracy"])
        macro_f1_by_group[group_name] = float(group_metrics["macro_f1"])
        count_by_group[group_name] = group_count
        group_payloads.append(
            {
                "group": group_name,
                "count": group_count,
                "supported": bool(supported_group),
                "below_support_threshold": bool(not supported_group),
                "accuracy": float(group_metrics["accuracy"]),
                "macro_f1": float(group_metrics["macro_f1"]),
                "weighted_f1": float(group_metrics["weighted_f1"]),
                "balanced_accuracy": float(group_metrics["balanced_accuracy"]),
                "macro_precision": float(group_metrics["macro_precision"]),
                "macro_recall": float(group_metrics["macro_recall"]),
            }
        )
        group_confusion_payloads.append(
            {
                "group": group_name,
                "count": group_count,
                "supported": bool(supported_group),
                "below_support_threshold": bool(not supported_group),
                **group_confusion,
            }
        )
        group_error_payloads.append(
            {
                "group": group_name,
                "count": group_count,
                "supported": bool(supported_group),
                "below_support_threshold": bool(not supported_group),
                "misclassified_count": misclassified_count,
                "misclassification_rate": float(misclassified_count / max(1, group_count)),
                "true_counts": group_distribution["true_counts"],
                "predicted_counts": group_distribution["predicted_counts"],
                "top_confusions": _top_confusion_pairs(group_true, group_pred),
                "prediction_entropy": prediction_entropy,
                "predicted_unique_labels": int(predicted_unique_labels),
                "prediction_collapse": bool(predicted_unique_labels <= 1 or prediction_entropy <= 1e-12),
            }
        )

    def _metric_extrema_summary(
        metric_by_group: Mapping[str, float],
        group_names: Sequence[str],
    ) -> dict[str, Any]:
        if not group_names:
            return {
                "best_value": pd.NA,
                "worst_value": pd.NA,
                "gap": pd.NA,
                "best_name": pd.NA,
                "best_count": pd.NA,
                "worst_name": pd.NA,
                "worst_count": pd.NA,
            }
        best_name = min(
            group_names,
            key=lambda group_name: (-metric_by_group[group_name], group_name),
        )
        worst_name = min(
            group_names,
            key=lambda group_name: (metric_by_group[group_name], group_name),
        )
        return {
            "best_value": float(metric_by_group[best_name]),
            "worst_value": float(metric_by_group[worst_name]),
            "gap": float(metric_by_group[best_name] - metric_by_group[worst_name]),
            "best_name": str(best_name),
            "best_count": int(count_by_group[best_name]),
            "worst_name": str(worst_name),
            "worst_count": int(count_by_group[worst_name]),
        }

    ordered_groups = sorted(count_by_group)
    supported_groups = [group_name for group_name in ordered_groups if count_by_group[group_name] >= support_threshold]
    below_support_groups = [group_name for group_name in ordered_groups if count_by_group[group_name] < support_threshold]
    raw_accuracy_summary = _metric_extrema_summary(accuracy_by_group, ordered_groups)
    raw_f1_summary = _metric_extrema_summary(macro_f1_by_group, ordered_groups)
    supported_accuracy_summary = _metric_extrema_summary(accuracy_by_group, supported_groups)
    supported_f1_summary = _metric_extrema_summary(macro_f1_by_group, supported_groups)
    group_support_payload = [
        {
            "group": group_name,
            "count": int(count_by_group[group_name]),
            "supported": bool(group_name in supported_groups),
        }
        for group_name in ordered_groups
    ]
    groups_below_support_payload = [
        {
            "group": group_name,
            "count": int(count_by_group[group_name]),
        }
        for group_name in below_support_groups
    ]
    summary = {
        "support_threshold_used": int(support_threshold),
        "supported_group_count": int(len(supported_groups)),
        "small_group_count": int(len(below_support_groups)),
        "group_support_counts_json": _json_dumps(group_support_payload),
        "groups_below_support_threshold_json": _json_dumps(groups_below_support_payload),
        "best_group_accuracy": raw_accuracy_summary["best_value"],
        "worst_group_accuracy": raw_accuracy_summary["worst_value"],
        "group_accuracy_gap": raw_accuracy_summary["gap"],
        "accuracy_gap": raw_accuracy_summary["gap"],
        "best_group_accuracy_name": raw_accuracy_summary["best_name"],
        "best_group_accuracy_count": raw_accuracy_summary["best_count"],
        "worst_group_accuracy_name": raw_accuracy_summary["worst_name"],
        "worst_group_accuracy_count": raw_accuracy_summary["worst_count"],
        "best_group_macro_f1": raw_f1_summary["best_value"],
        "worst_group_macro_f1": raw_f1_summary["worst_value"],
        "group_macro_f1_gap": raw_f1_summary["gap"],
        "best_group_f1": raw_f1_summary["best_value"],
        "worst_group_f1": raw_f1_summary["worst_value"],
        "macro_f1_gap": raw_f1_summary["gap"],
        "best_group_f1_name": raw_f1_summary["best_name"],
        "best_group_f1_count": raw_f1_summary["best_count"],
        "worst_group_f1_name": raw_f1_summary["worst_name"],
        "worst_group_f1_count": raw_f1_summary["worst_count"],
        "worst_group_accuracy_raw": raw_accuracy_summary["worst_value"],
        "worst_group_f1_raw": raw_f1_summary["worst_value"],
        "accuracy_gap_raw": raw_accuracy_summary["gap"],
        "macro_f1_gap_raw": raw_f1_summary["gap"],
        "worst_group_accuracy_supported": supported_accuracy_summary["worst_value"],
        "worst_group_f1_supported": supported_f1_summary["worst_value"],
        "accuracy_gap_supported": supported_accuracy_summary["gap"],
        "macro_f1_gap_supported": supported_f1_summary["gap"],
        "best_group_accuracy_supported_name": supported_accuracy_summary["best_name"],
        "best_group_accuracy_supported_count": supported_accuracy_summary["best_count"],
        "worst_group_accuracy_supported_name": supported_accuracy_summary["worst_name"],
        "worst_group_accuracy_supported_count": supported_accuracy_summary["worst_count"],
        "best_group_f1_supported_name": supported_f1_summary["best_name"],
        "best_group_f1_supported_count": supported_f1_summary["best_count"],
        "worst_group_f1_supported_name": supported_f1_summary["worst_name"],
        "worst_group_f1_supported_count": supported_f1_summary["worst_count"],
    }
    if report_small_group_metrics:
        summary["small_group_metrics_json"] = _json_dumps(
            [payload for payload in group_payloads if bool(payload["below_support_threshold"])]
        )
    else:
        summary["small_group_metrics_json"] = pd.NA
    return (
        summary,
        _json_dumps(group_payloads),
        _json_dumps(group_confusion_payloads),
        _json_dumps(group_error_payloads),
    )


def _probe_split_viability_reason(
    train_labels: Sequence[Any],
    test_labels: Sequence[Any],
) -> str:
    """Return one reason why a train/test probe split is not runnable."""

    train_label_series = pd.Series([str(label) for label in train_labels], dtype=str)
    test_label_series = pd.Series([str(label) for label in test_labels], dtype=str)
    if train_label_series.empty or test_label_series.empty:
        return "Probe requires non-empty train and test splits."
    if train_label_series.nunique() < 2:
        return "Probe requires at least two classes in the training split."
    if test_label_series.nunique() < 2:
        return "Probe requires at least two classes in the test split."
    unseen_test_labels = sorted(set(test_label_series) - set(train_label_series))
    if unseen_test_labels:
        return (
            "Probe test split contains classes not present in training: "
            f"{unseen_test_labels}."
        )
    return ""


def _resolve_validation_probe_subset(
    train_labels: Sequence[Any],
    test_labels: Sequence[Any],
    *,
    min_group_support_threshold: int,
    prefer_support_aware_subset: bool,
) -> dict[str, Any]:
    """Return one runnable validation-probe subset, falling back conservatively when needed."""

    train_label_series = pd.Series([str(label) for label in train_labels], dtype=str)
    test_label_series = pd.Series([str(label) for label in test_labels], dtype=str)
    full_reason = _probe_split_viability_reason(
        train_label_series.tolist(),
        test_label_series.tolist(),
    )
    if not full_reason:
        return {
            "reason": "",
            "train_mask": np.ones(len(train_label_series), dtype=bool),
            "test_mask": np.ones(len(test_label_series), dtype=bool),
            "scope": "all",
        }

    train_label_set = set(train_label_series.tolist())
    candidate_label_sets: list[tuple[str, set[str]]] = []
    if prefer_support_aware_subset:
        supported_test_labels = {
            str(label)
            for label, count in test_label_series.value_counts().items()
            if int(count) >= int(min_group_support_threshold)
        }
        candidate_label_sets.append(
            ("supported_shared", supported_test_labels & train_label_set)
        )
    candidate_label_sets.append(
        ("shared", set(test_label_series.tolist()) & train_label_set)
    )
    for scope, candidate_labels in candidate_label_sets:
        if len(candidate_labels) < 2:
            continue
        train_mask = train_label_series.isin(candidate_labels).to_numpy(dtype=bool)
        test_mask = test_label_series.isin(candidate_labels).to_numpy(dtype=bool)
        candidate_reason = _probe_split_viability_reason(
            train_label_series.loc[train_mask].tolist(),
            test_label_series.loc[test_mask].tolist(),
        )
        if not candidate_reason:
            return {
                "reason": "",
                "train_mask": train_mask,
                "test_mask": test_mask,
                "scope": scope,
            }

    return {
        "reason": full_reason,
        "train_mask": np.zeros(len(train_label_series), dtype=bool),
        "test_mask": np.zeros(len(test_label_series), dtype=bool),
        "scope": "none",
    }


def _build_probe_pipeline(*, random_seed: int, model_type: str) -> Pipeline:
    """Build one deterministic probe pipeline."""

    resolved_model_type = resolve_probe_model_type(model_type)
    if resolved_model_type == "linear":
        model: Any = LogisticRegression(max_iter=1000, random_state=int(random_seed))
    else:
        model = MLPClassifier(
            hidden_layer_sizes=(64,),
            max_iter=300,
            random_state=int(random_seed),
        )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def run_train_test_embedding_probe(
    *,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_labels: Sequence[Any],
    test_labels: Sequence[Any],
    random_seed: int,
    model_type: str = "linear",
) -> dict[str, Any]:
    """Train one simple probe on embeddings and evaluate it on a held-out split."""

    resolved_model_type = resolve_probe_model_type(model_type)
    train_label_series = pd.Series([str(label) for label in train_labels], dtype=str)
    test_label_series = pd.Series([str(label) for label in test_labels], dtype=str)
    viability_reason = _probe_split_viability_reason(train_label_series.tolist(), test_label_series.tolist())
    if viability_reason:
        return {
            "status": "skipped",
            "accuracy": pd.NA,
            "macro_f1": pd.NA,
            "reason": viability_reason,
            "model_type": resolved_model_type,
        }

    probe = _build_probe_pipeline(random_seed=int(random_seed), model_type=resolved_model_type)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        probe.fit(np.asarray(train_embeddings, dtype=float), train_label_series.to_numpy(dtype=object))
    predictions = probe.predict(np.asarray(test_embeddings, dtype=float))
    metrics = compute_classification_metrics(
        test_label_series.to_numpy(dtype=object),
        predictions,
    )
    return {
        "status": "ok",
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "reason": "",
        "model_type": resolved_model_type,
    }


def _class_weight_tensor(
    encoded_labels: np.ndarray,
    *,
    class_count: int,
    smoothing: float,
    torch: Any,
) -> Any:
    counts = np.bincount(encoded_labels, minlength=class_count).astype(np.float64, copy=False)
    smoothed_counts = counts + float(smoothing)
    weights = 1.0 / smoothed_counts
    weights *= float(class_count) / float(weights.sum())
    return torch.tensor(weights, dtype=torch.float32)


def _sample_weight_tensor(
    encoded_labels: np.ndarray,
    *,
    class_count: int,
    smoothing: float,
    torch: Any,
) -> Any:
    class_weights = _class_weight_tensor(
        encoded_labels,
        class_count=class_count,
        smoothing=smoothing,
        torch=torch,
    )
    sample_weights = class_weights[torch.tensor(encoded_labels, dtype=torch.long)]
    return sample_weights.to(dtype=torch.float32)


def _group_rebalanced_sample_offsets(
    group_targets: Any,
    *,
    min_group_support_threshold: int,
    epoch_index: int,
    random_seed: int,
    rebalance_all_groups: bool = False,
    torch: Any,
) -> Any:
    """Return deterministic training offsets after mild protected-group rebalancing."""

    group_targets_cpu = group_targets.detach().cpu().numpy()
    if group_targets_cpu.size == 0:
        return torch.zeros(0, dtype=torch.long, device=group_targets.device)
    unique_groups, counts = np.unique(group_targets_cpu, return_counts=True)
    if rebalance_all_groups:
        supported_groups = [int(group_value) for group_value in unique_groups.tolist()]
    else:
        supported_groups = [
            int(group_value)
            for group_value, count in zip(unique_groups.tolist(), counts.tolist(), strict=False)
            if int(count) >= int(min_group_support_threshold)
        ]
    if len(supported_groups) < 2:
        return torch.arange(len(group_targets_cpu), dtype=torch.long, device=group_targets.device)

    rng = np.random.default_rng(int(random_seed) + (int(epoch_index) * 10007) + 17)
    target_count = max(
        int(count)
        for group_value, count in zip(unique_groups.tolist(), counts.tolist(), strict=False)
        if int(group_value) in supported_groups
    )
    sampled_offsets: list[np.ndarray] = []
    for group_value, count in zip(unique_groups.tolist(), counts.tolist(), strict=False):
        member_offsets = np.flatnonzero(group_targets_cpu == int(group_value))
        if int(group_value) in supported_groups:
            sampled = rng.choice(member_offsets, size=int(target_count), replace=True)
        else:
            sampled = member_offsets
        sampled_offsets.append(np.asarray(sampled, dtype=np.int64))
    merged_offsets = np.concatenate(sampled_offsets, axis=0)
    shuffled_offsets = merged_offsets[rng.permutation(len(merged_offsets))]
    return torch.tensor(shuffled_offsets, dtype=torch.long, device=group_targets.device)


def _focal_loss_vector(
    logits: Any,
    targets: Any,
    *,
    gamma: float,
    class_weights: Any | None,
    nn_functional: Any,
) -> Any:
    log_probs = nn_functional.log_softmax(logits, dim=-1)
    ce_loss = nn_functional.nll_loss(
        log_probs,
        targets,
        reduction="none",
        weight=class_weights,
    )
    target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = target_log_probs.exp()
    return ((1.0 - pt) ** float(gamma)) * ce_loss


def _main_task_loss_vector(
    logits: Any,
    targets: Any,
    *,
    imbalance_mode: str,
    focal_gamma: float,
    class_weights: Any | None,
    nn_functional: Any,
) -> Any:
    """Return unreduced per-sample main-task loss values."""

    if imbalance_mode == "focal_loss":
        return _focal_loss_vector(
            logits,
            targets,
            gamma=focal_gamma,
            class_weights=class_weights,
            nn_functional=nn_functional,
        )
    return nn_functional.cross_entropy(
        logits,
        targets,
        reduction="none",
        weight=class_weights,
    )


def _group_loss_statistics(
    group_targets: Any,
    *,
    loss_vector: Any,
    group_count: int,
    torch: Any,
) -> tuple[Any, Any]:
    """Return per-group mean loss and sample counts for one batch."""

    group_means = torch.ones(
        int(group_count),
        dtype=torch.float32,
        device=loss_vector.device,
    )
    group_counts = torch.bincount(group_targets, minlength=int(group_count)).to(dtype=torch.float32)
    detached_losses = loss_vector.detach()
    for group_index in range(int(group_count)):
        mask = group_targets == int(group_index)
        if bool(mask.any()):
            group_means[int(group_index)] = detached_losses[mask].mean()
    return group_means, group_counts


def _eligible_group_mask(
    group_counts: Any,
    *,
    min_group_support_threshold: int,
) -> Any:
    """Return a boolean mask for groups eligible for robust weighting."""

    return group_counts >= int(min_group_support_threshold)


def _collapsed_group_indices(
    group_targets: Any,
    *,
    predicted_targets: Any,
    group_count: int,
    min_group_support_threshold: int,
    torch: Any,
) -> list[int]:
    """Return eligible group indices whose predictions collapsed to one label."""

    counts = torch.bincount(group_targets, minlength=int(group_count)).to(dtype=torch.int64)
    collapsed: list[int] = []
    for group_index in range(int(group_count)):
        if int(counts[int(group_index)].item()) < int(min_group_support_threshold):
            continue
        mask = group_targets == int(group_index)
        if not bool(mask.any()):
            continue
        unique_predictions = torch.unique(predicted_targets[mask])
        if int(unique_predictions.numel()) <= 1:
            collapsed.append(int(group_index))
    return collapsed


def _group_robust_sample_weights(
    group_targets: Any,
    *,
    loss_vector: Any,
    predicted_targets: Any,
    group_count: int,
    group_weight_mode: str,
    group_robust_weight: float,
    worst_group_boost_factor: float,
    min_support_boost_factor: float,
    min_group_support_threshold: int,
    group_state: Any | None,
    torch: Any,
) -> dict[str, Any] | None:
    """Return protected-group weighting outputs for one training epoch."""

    if not _group_robust_enabled(
        group_weight_mode=group_weight_mode,
        group_robust_weight=group_robust_weight,
    ):
        return None
    group_means, group_counts = _group_loss_statistics(
        group_targets,
        loss_vector=loss_vector,
        group_count=group_count,
        torch=torch,
    )
    eligible_mask = _eligible_group_mask(
        group_counts,
        min_group_support_threshold=min_group_support_threshold,
    )
    collapsed_indices = _collapsed_group_indices(
        group_targets,
        predicted_targets=predicted_targets,
        group_count=group_count,
        min_group_support_threshold=min_group_support_threshold,
        torch=torch,
    )
    if group_weight_mode == "min_support_boost":
        weightable_mask = group_counts > 0
    else:
        weightable_mask = eligible_mask
    if not bool(weightable_mask.any()):
        return {
            "sample_weights": None,
            "group_state": group_state,
            "collapsed_group_indices": collapsed_indices,
            "eligible_group_indices": [],
            "boosted_group_indices": [],
        }
    if group_weight_mode == "inverse_frequency":
        base_weights = torch.ones(
            int(group_count),
            dtype=torch.float32,
            device=loss_vector.device,
        )
        base_weights[eligible_mask] = 1.0 / group_counts[eligible_mask].clamp_min(1.0)
        sample_weights = base_weights[group_targets]
        sample_weights = sample_weights.to(dtype=torch.float32)
        sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-12)
        sample_weights = 1.0 + (float(group_robust_weight) * (sample_weights - 1.0))
        boosted_group_indices: list[int] = []
    elif group_weight_mode == "min_support_boost":
        base_weights = torch.ones(
            int(group_count),
            dtype=torch.float32,
            device=loss_vector.device,
        )
        present_mask = group_counts > 0
        base_weights[present_mask] = 1.0 / group_counts[present_mask].clamp_min(1.0)
        small_group_mask = present_mask & (~eligible_mask)
        if bool(small_group_mask.any()):
            base_weights[small_group_mask] = (
                base_weights[small_group_mask] * float(min_support_boost_factor)
            )
        sample_weights = base_weights[group_targets]
        sample_weights = sample_weights.to(dtype=torch.float32)
        sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-12)
        sample_weights = 1.0 + (float(group_robust_weight) * (sample_weights - 1.0))
        boosted_group_indices = [
            int(index.item())
            for index in torch.nonzero(small_group_mask, as_tuple=False).view(-1)
        ]
    elif group_weight_mode == "worst_group_boost":
        base_weights = torch.ones(
            int(group_count),
            dtype=torch.float32,
            device=loss_vector.device,
        )
        if collapsed_indices:
            boosted_group_indices = collapsed_indices
        else:
            eligible_group_indices = torch.nonzero(eligible_mask, as_tuple=False).view(-1)
            max_group_loss = torch.max(group_means[eligible_mask])
            boosted_group_indices = [
                int(index.item())
                for index in eligible_group_indices
                if abs(float(group_means[int(index.item())].item()) - float(max_group_loss.item())) <= 1e-12
            ]
        for group_index in boosted_group_indices:
            base_weights[int(group_index)] = float(worst_group_boost_factor)
        sample_weights = base_weights[group_targets]
        sample_weights = 1.0 + (float(group_robust_weight) * (sample_weights - 1.0))
    elif group_weight_mode == "group_dro":
        if group_state is None:
            current_group_state = torch.ones(
                int(group_count),
                dtype=torch.float32,
                device=loss_vector.device,
            )
        else:
            current_group_state = group_state.to(dtype=torch.float32, device=loss_vector.device).clone()
        current_group_state[~eligible_mask] = 1.0
        current_group_state = current_group_state / current_group_state.mean().clamp_min(1e-12)
        sample_weights = current_group_state[group_targets]
        updated_group_state = current_group_state.clone()
        if bool(eligible_mask.any()):
            exponent = torch.clamp(
                float(group_robust_weight) * group_means[eligible_mask],
                min=-20.0,
                max=20.0,
            )
            updated_group_state[eligible_mask] = updated_group_state[eligible_mask] * torch.exp(exponent)
            updated_group_state[~eligible_mask] = 1.0
            updated_group_state = updated_group_state / updated_group_state.mean().clamp_min(1e-12)
        return {
            "sample_weights": sample_weights.to(dtype=torch.float32),
            "group_state": updated_group_state.detach(),
            "collapsed_group_indices": collapsed_indices,
            "eligible_group_indices": [
                int(index.item())
                for index in torch.nonzero(eligible_mask, as_tuple=False).view(-1)
            ],
            "boosted_group_indices": [],
        }
    else:
        raise ValueError(f"Unsupported group_weight_mode '{group_weight_mode}'.")
    return {
        "sample_weights": sample_weights.to(dtype=torch.float32),
        "group_state": group_state,
        "collapsed_group_indices": collapsed_indices,
        "eligible_group_indices": [
            int(index.item())
            for index in torch.nonzero(eligible_mask, as_tuple=False).view(-1)
        ],
        "boosted_group_indices": boosted_group_indices,
    }


def _validation_metric_value(
    metric_name: str,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_values: np.ndarray | None = None,
    min_group_support_eval: int,
    protected_probe_macro_f1: float | None = None,
    fairness_score_alpha: float = 0.25,
    fairness_score_beta: float = 0.25,
) -> float:
    if metric_name == "worst_group_f1_raw":
        if group_values is None:
            raise ValueError("worst_group_f1_raw early stopping requires validation protected-group labels.")
        group_summary, _, _, _ = summarize_group_classification_metrics(
            y_true,
            y_pred,
            group_values,
            min_group_support_threshold=min_group_support_eval,
        )
        return float(group_summary["worst_group_f1_raw"])
    if metric_name == "worst_group_f1_supported":
        if group_values is None:
            raise ValueError("worst_group_f1_supported early stopping requires validation protected-group labels.")
        group_summary, _, _, _ = summarize_group_classification_metrics(
            y_true,
            y_pred,
            group_values,
            min_group_support_threshold=min_group_support_eval,
        )
        if pd.isna(group_summary["worst_group_f1_supported"]):
            raise ValueError(
                "worst_group_f1_supported early stopping requires at least one validation protected group "
                f"with support >= {int(min_group_support_eval)}."
            )
        return float(group_summary["worst_group_f1_supported"])
    if metric_name == "fairness_score":
        if group_values is None:
            raise ValueError("fairness_score early stopping requires validation protected-group labels.")
        if protected_probe_macro_f1 is None:
            raise ValueError("fairness_score early stopping requires a validation protected-attribute probe score.")
        metrics = compute_classification_metrics(y_true, y_pred)
        group_summary, _, _, _ = summarize_group_classification_metrics(
            y_true,
            y_pred,
            group_values,
            min_group_support_threshold=min_group_support_eval,
        )
        if pd.isna(group_summary["macro_f1_gap_supported"]):
            raise ValueError(
                "fairness_score early stopping requires at least one validation protected group "
                f"with support >= {int(min_group_support_eval)}."
            )
        return float(
            float(metrics["macro_f1"])
            - (float(fairness_score_alpha) * float(group_summary["macro_f1_gap_supported"]))
            - (float(fairness_score_beta) * float(protected_probe_macro_f1))
        )
    metrics = compute_classification_metrics(y_true, y_pred)
    return float(metrics[metric_name])


def _warmup_scale(*, epoch_index: int, warmup_epochs: int) -> float:
    """Return a linear warm-up multiplier for one epoch."""

    if int(warmup_epochs) <= 0:
        return 1.0
    return float(min(1.0, float(epoch_index + 1) / float(warmup_epochs)))


def _is_better_validation_result(
    metric_name: str,
    *,
    best_metric: float,
    best_loss: float,
    candidate_metric: float,
    candidate_loss: float,
) -> bool:
    if candidate_metric > best_metric + 1e-12:
        return True
    if abs(candidate_metric - best_metric) <= 1e-12 and candidate_loss < best_loss - 1e-12:
        return True
    return False


def train_graphsage_node_classifier(
    graph: nx.Graph,
    embedding_frame: pd.DataFrame,
    labels: pd.Series,
    split: NodeClassificationSplit,
    train_validation_split: NodeTrainValidationSplit,
    *,
    protected_labels: pd.Series | None = None,
    config: GraphSAGENodeClassificationConfig,
) -> GraphSAGENodeClassificationResult:
    """Train a supervised GraphSAGE classifier and return test predictions."""

    torch, nn_module, _, _, _, _, _, sage_conv_cls = require_pyg_dependencies()
    resolved_config = validate_graphsage_node_classification_config(config)
    seed_torch(torch, resolved_config.random_seed)

    node_order = sorted_node_ids(graph)
    prepared_features = coerce_input_features(embedding_frame, node_order)
    if prepared_features is None:
        raise ValueError("GraphSAGE node classification requires node-aligned embedding features.")

    label_lookup = labels.reindex(list(node_order))
    if label_lookup.isna().any():
        raise ValueError("Node-classification labels must align to every graph node.")
    protected_lookup: pd.Series | None = None
    if protected_labels is not None:
        protected_lookup = protected_labels.reindex(list(node_order))
        if protected_lookup.isna().any():
            raise ValueError("Protected-attribute labels must align to every graph node without missing values.")

    class_names = tuple(np.unique(label_lookup.to_numpy(dtype=object)))
    class_to_index = {label_name: index for index, label_name in enumerate(class_names)}
    encoded_labels = np.asarray(
        [class_to_index[str(label_lookup.loc[node_id])] for node_id in node_order],
        dtype=np.int64,
    )
    encoded_protected_labels: np.ndarray | None = None
    protected_class_names: tuple[Any, ...] = ()
    if protected_lookup is not None:
        protected_class_names = tuple(np.unique(protected_lookup.to_numpy(dtype=object)))
        if len(protected_class_names) < 2:
            raise ValueError("Protected-attribute labels must contain at least two classes.")
        protected_to_index = {
            label_name: index for index, label_name in enumerate(protected_class_names)
        }
        encoded_protected_labels = np.asarray(
            [protected_to_index[str(protected_lookup.loc[node_id])] for node_id in node_order],
            dtype=np.int64,
        )
    node_to_index = {node_id: index for index, node_id in enumerate(node_order)}

    work_graph = graph if not graph.is_directed() else graph.to_undirected()
    edge_pairs: list[tuple[int, int]] = []
    for source_node, target_node in work_graph.edges():
        source_index = node_to_index[source_node]
        target_index = node_to_index[target_node]
        edge_pairs.append((source_index, target_index))
        if source_index != target_index:
            edge_pairs.append((target_index, source_index))
    if not edge_pairs:
        edge_pairs = [(index, index) for index in range(len(node_order))]

    x = torch.tensor(prepared_features.feature_matrix, dtype=torch.float32)
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    y = torch.tensor(encoded_labels, dtype=torch.long)
    protected_targets_tensor = (
        None
        if encoded_protected_labels is None
        else torch.tensor(encoded_protected_labels, dtype=torch.long)
    )

    train_indices = torch.tensor(
        [node_to_index[node_id] for node_id in train_validation_split.train_node_ids],
        dtype=torch.long,
    )
    validation_indices = torch.tensor(
        [node_to_index[node_id] for node_id in train_validation_split.validation_node_ids],
        dtype=torch.long,
    )
    test_indices = torch.tensor(
        [node_to_index[node_id] for node_id in split.test_node_ids],
        dtype=torch.long,
    )

    train_targets = encoded_labels[train_indices.detach().cpu().numpy()]
    train_group_values = None
    validation_group_values = None
    if protected_lookup is not None:
        train_group_values = (
            protected_lookup.loc[list(train_validation_split.train_node_ids)]
            .astype(str)
            .to_numpy(dtype=object)
        )
        validation_group_values = (
            protected_lookup.loc[list(train_validation_split.validation_node_ids)]
            .astype(str)
            .to_numpy(dtype=object)
        )

    if resolved_config.debias_mode == "adversarial" and protected_targets_tensor is None:
        raise ValueError(
            "debias_mode='adversarial' requires a protected_attribute_column with non-missing labels."
        )
    if _group_robust_enabled(
        group_weight_mode=resolved_config.group_weight_mode,
        group_robust_weight=resolved_config.group_robust_weight,
    ) and protected_targets_tensor is None:
        raise ValueError(
            "Protected-group robust weighting requires a protected_attribute_column with non-missing labels."
        )
    if resolved_config.rebalance_batches_by_group and protected_targets_tensor is None:
        raise ValueError(
            "rebalance_batches_by_group requires a protected_attribute_column with non-missing labels."
        )
    if resolved_config.early_stop_metric in {"worst_group_f1_raw", "worst_group_f1_supported"} and protected_lookup is None:
        raise ValueError(
            f"early_stop_metric='{resolved_config.early_stop_metric}' requires a protected_attribute_column with non-missing labels."
        )
    if resolved_config.early_stop_metric == "fairness_score":
        if protected_lookup is None:
            raise ValueError(
                "early_stop_metric='fairness_score' requires a protected_attribute_column with non-missing labels."
            )
        validation_probe_resolution = _resolve_validation_probe_subset(
            train_group_values if train_group_values is not None else (),
            validation_group_values if validation_group_values is not None else (),
            min_group_support_threshold=resolved_config.min_group_support_eval,
            prefer_support_aware_subset=True,
        )
        if validation_probe_resolution["reason"]:
            raise ValueError(
                "early_stop_metric='fairness_score' requires a runnable validation protected-attribute probe. "
                + str(validation_probe_resolution["reason"])
            )
    if (
        validation_group_values is not None
        and resolved_config.early_stop_metric in {"worst_group_f1_supported", "fairness_score"}
    ):
        validation_group_counts = pd.Series(validation_group_values, dtype=str).value_counts()
        supported_validation_group_count = int(
            (validation_group_counts >= int(resolved_config.min_group_support_eval)).sum()
        )
        if supported_validation_group_count < 1:
            raise ValueError(
                f"early_stop_metric='{resolved_config.early_stop_metric}' requires at least one validation "
                f"protected group with support >= {int(resolved_config.min_group_support_eval)}."
            )
    class_weights = None
    if resolved_config.imbalance_mode in {"class_weighted", "focal_loss"}:
        class_weights = _class_weight_tensor(
            train_targets,
            class_count=len(class_names),
            smoothing=resolved_config.class_weight_smoothing,
            torch=torch,
        )

    class GraphSAGEEncoder(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            layer_dims: list[tuple[int, int]] = []
            input_dim = int(x.shape[1])
            layer_dims.append((input_dim, resolved_config.hidden_dim))
            layer_dims.extend(
                (resolved_config.hidden_dim, resolved_config.hidden_dim)
                for _ in range(max(0, resolved_config.num_layers - 1))
            )
            self.convs = nn_module.ModuleList(
                [sage_conv_cls(in_dim, out_dim) for in_dim, out_dim in layer_dims]
            )

        def forward(self, feature_matrix: Any, full_edge_index: Any) -> Any:
            output = feature_matrix
            for layer_index, conv in enumerate(self.convs):
                output = conv(output, full_edge_index)
                if layer_index < len(self.convs) - 1:
                    output = nn_module.functional.relu(output)
                    if resolved_config.dropout > 0.0:
                        output = nn_module.functional.dropout(
                            output,
                            p=resolved_config.dropout,
                            training=self.training,
                        )
            return output

    class MLPHead(nn_module.Module):
        def __init__(
            self,
            *,
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.dropout = float(dropout)
            layers: list[Any] = []
            current_dim = int(input_dim)
            for _ in range(max(0, num_layers - 1)):
                layers.append(nn_module.Linear(current_dim, hidden_dim))
                current_dim = int(hidden_dim)
            self.hidden_layers = nn_module.ModuleList(layers)
            self.output_layer = nn_module.Linear(current_dim, int(output_dim))

        def forward(self, features: Any) -> Any:
            output = features
            for layer in self.hidden_layers:
                output = nn_module.functional.relu(layer(output))
                if self.dropout > 0.0:
                    output = nn_module.functional.dropout(
                        output,
                        p=self.dropout,
                        training=self.training,
                    )
            return self.output_layer(output)

    class GraphSAGEClassifier(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = GraphSAGEEncoder()
            self.main_head = nn_module.Linear(resolved_config.hidden_dim, len(class_names))
            self.adversary_head = (
                None
                if resolved_config.debias_mode != "adversarial"
                else MLPHead(
                    input_dim=resolved_config.hidden_dim,
                    output_dim=len(protected_class_names),
                    hidden_dim=resolved_config.adversary_hidden_dim,
                    num_layers=resolved_config.adversary_num_layers,
                    dropout=resolved_config.adversary_dropout,
                )
            )

        def encode(self, feature_matrix: Any, full_edge_index: Any) -> Any:
            return self.encoder(feature_matrix, full_edge_index)

        def predict_main(self, embeddings: Any) -> Any:
            return self.main_head(embeddings)

        def predict_adversary(self, embeddings: Any) -> Any:
            if self.adversary_head is None:
                raise RuntimeError("Adversary head is not enabled.")
            return self.adversary_head(embeddings)

    model = GraphSAGEClassifier()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=resolved_config.learning_rate,
        weight_decay=resolved_config.weight_decay,
    )

    best_state: dict[str, Any] | None = None
    best_validation_loss = float("inf")
    best_validation_metric = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    training_history: list[dict[str, Any]] = []

    train_sample_weights = None
    if resolved_config.imbalance_mode == "weighted_sampler":
        train_sample_weights = _sample_weight_tensor(
            train_targets,
            class_count=len(class_names),
            smoothing=resolved_config.class_weight_smoothing,
            torch=torch,
        )
    validation_probe_viability_reason = ""
    validation_probe_train_mask = np.zeros(
        len(train_group_values) if train_group_values is not None else 0,
        dtype=bool,
    )
    validation_probe_test_mask = np.zeros(
        len(validation_group_values) if validation_group_values is not None else 0,
        dtype=bool,
    )
    if protected_lookup is not None:
        validation_probe_resolution = _resolve_validation_probe_subset(
            train_group_values if train_group_values is not None else (),
            validation_group_values if validation_group_values is not None else (),
            min_group_support_threshold=resolved_config.min_group_support_eval,
            prefer_support_aware_subset=(resolved_config.early_stop_metric == "fairness_score"),
        )
        validation_probe_viability_reason = str(validation_probe_resolution["reason"])
        validation_probe_train_mask = np.asarray(
            validation_probe_resolution["train_mask"],
            dtype=bool,
        )
        validation_probe_test_mask = np.asarray(
            validation_probe_resolution["test_mask"],
            dtype=bool,
        )
    can_run_validation_probe = protected_lookup is not None and not validation_probe_viability_reason
    group_dro_state = None
    if (
        protected_targets_tensor is not None
        and resolved_config.group_weight_mode == "group_dro"
        and _group_robust_enabled(
            group_weight_mode=resolved_config.group_weight_mode,
            group_robust_weight=resolved_config.group_robust_weight,
        )
    ):
        group_dro_state = torch.ones(len(protected_class_names), dtype=torch.float32)

    class _GradientReversalFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, input_tensor: Any, lambda_value: float) -> Any:
            ctx.lambda_value = float(lambda_value)
            return input_tensor.view_as(input_tensor)

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, None]:
            return grad_output.neg() * ctx.lambda_value, None

    def _apply_gradient_reversal(input_tensor: Any, *, lambda_value: float) -> Any:
        return _GradientReversalFunction.apply(
            input_tensor,
            float(lambda_value),
        )

    for epoch_index in range(resolved_config.epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(x, edge_index)
        logits = model.predict_main(embeddings)
        effective_train_indices = train_indices
        effective_train_sample_weights = train_sample_weights
        if resolved_config.rebalance_batches_by_group and protected_targets_tensor is not None:
            rebalance_offsets = _group_rebalanced_sample_offsets(
                protected_targets_tensor[train_indices],
                min_group_support_threshold=resolved_config.min_group_support_train,
                epoch_index=epoch_index,
                random_seed=resolved_config.random_seed,
                rebalance_all_groups=resolved_config.group_weight_mode == "min_support_boost",
                torch=torch,
            )
            effective_train_indices = train_indices[rebalance_offsets]
            if effective_train_sample_weights is not None:
                effective_train_sample_weights = effective_train_sample_weights[rebalance_offsets]
        if effective_train_sample_weights is not None:
            sampled_offsets = torch.multinomial(
                effective_train_sample_weights,
                num_samples=int(effective_train_indices.shape[0]),
                replacement=True,
            )
            effective_train_indices = effective_train_indices[sampled_offsets]
        train_loss_vector = _main_task_loss_vector(
            logits[effective_train_indices],
            y[effective_train_indices],
            imbalance_mode=resolved_config.imbalance_mode,
            focal_gamma=resolved_config.focal_gamma,
            class_weights=class_weights,
            nn_functional=nn_module.functional,
        )
        train_loss = train_loss_vector.mean()
        train_collapsed_group_names: list[str] = []
        group_robust_scale = _warmup_scale(
            epoch_index=epoch_index,
            warmup_epochs=resolved_config.group_robust_warmup_epochs,
        )
        scaled_group_robust_weight = float(resolved_config.group_robust_weight) * float(group_robust_scale)
        if _group_robust_enabled(
            group_weight_mode=resolved_config.group_weight_mode,
            group_robust_weight=scaled_group_robust_weight,
        ):
            effective_group_targets = protected_targets_tensor[effective_train_indices]
            group_weight_result = _group_robust_sample_weights(
                effective_group_targets,
                loss_vector=train_loss_vector,
                predicted_targets=logits[effective_train_indices].argmax(dim=1).detach(),
                group_count=len(protected_class_names),
                group_weight_mode=resolved_config.group_weight_mode,
                group_robust_weight=scaled_group_robust_weight,
                worst_group_boost_factor=resolved_config.worst_group_boost_factor,
                min_support_boost_factor=resolved_config.min_support_boost_factor,
                min_group_support_threshold=resolved_config.min_group_support_train,
                group_state=group_dro_state,
                torch=torch,
            )
            if group_weight_result is not None:
                group_dro_state = group_weight_result["group_state"]
                group_weights = group_weight_result["sample_weights"]
                train_collapsed_group_names = [
                    str(protected_class_names[int(group_index)])
                    for group_index in group_weight_result["collapsed_group_indices"]
                ]
                if group_weights is not None:
                    train_loss = (train_loss_vector * group_weights).mean()
        adversary_scale = _warmup_scale(
            epoch_index=epoch_index,
            warmup_epochs=resolved_config.adversary_warmup_epochs,
        )
        if resolved_config.debias_mode == "adversarial" and adversary_scale > 0.0:
            adversary_logits = model.predict_adversary(
                _apply_gradient_reversal(
                    embeddings[effective_train_indices],
                    lambda_value=float(resolved_config.gradient_reversal_lambda) * float(adversary_scale),
                )
            )
            adversary_loss = nn_module.functional.cross_entropy(
                adversary_logits,
                protected_targets_tensor[effective_train_indices],
            )
            train_loss = train_loss + (
                (float(resolved_config.adversary_loss_weight) * float(adversary_scale)) * adversary_loss
            )
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_embeddings = model.encode(x, edge_index)
            validation_logits = model.predict_main(validation_embeddings)
            validation_loss = float(
                nn_module.functional.cross_entropy(
                    validation_logits[validation_indices],
                    y[validation_indices],
                ).item()
            )
            validation_predictions = (
                validation_logits[validation_indices].argmax(dim=1).detach().cpu().numpy()
            )
            validation_targets = y[validation_indices].detach().cpu().numpy()
            validation_probe_accuracy = None
            validation_probe_macro_f1 = None
            if can_run_validation_probe:
                train_probe_embeddings = validation_embeddings[train_indices].detach().cpu().numpy()
                validation_probe_embeddings = validation_embeddings[validation_indices].detach().cpu().numpy()
                train_probe_labels = np.asarray(
                    train_group_values if train_group_values is not None else (),
                    dtype=object,
                )
                validation_probe_labels = np.asarray(
                    validation_group_values if validation_group_values is not None else (),
                    dtype=object,
                )
                validation_probe_result = run_train_test_embedding_probe(
                    train_embeddings=train_probe_embeddings[validation_probe_train_mask],
                    test_embeddings=validation_probe_embeddings[validation_probe_test_mask],
                    train_labels=train_probe_labels[validation_probe_train_mask],
                    test_labels=validation_probe_labels[validation_probe_test_mask],
                    random_seed=resolved_config.random_seed,
                    model_type=resolved_config.probe_model_type,
                )
                if validation_probe_result["status"] != "ok":
                    if resolved_config.early_stop_metric == "fairness_score":
                        raise ValueError(
                            "Validation protected-attribute probe became unrunnable while computing fairness_score: "
                            f"{validation_probe_result['reason']}"
                        )
                else:
                    validation_probe_accuracy = float(validation_probe_result["accuracy"])
                    validation_probe_macro_f1 = float(validation_probe_result["macro_f1"])
            elif resolved_config.early_stop_metric == "fairness_score":
                raise ValueError(
                    "Validation protected-attribute probe became unrunnable while computing fairness_score: "
                    f"{validation_probe_viability_reason}"
                )
            validation_metric = _validation_metric_value(
                resolved_config.early_stop_metric,
                y_true=validation_targets,
                y_pred=validation_predictions,
                group_values=validation_group_values,
                min_group_support_eval=resolved_config.min_group_support_eval,
                protected_probe_macro_f1=validation_probe_macro_f1,
                fairness_score_alpha=resolved_config.fairness_score_alpha,
                fairness_score_beta=resolved_config.fairness_score_beta,
            )

            validation_metrics = compute_classification_metrics(validation_targets, validation_predictions)
            validation_group_summary = None
            validation_group_errors: list[dict[str, Any]] = []
            if validation_group_values is not None:
                validation_group_summary, _, _, validation_group_error_json = summarize_group_classification_metrics(
                    validation_targets,
                    validation_predictions,
                    validation_group_values,
                    min_group_support_threshold=resolved_config.min_group_support_eval,
                    report_small_group_metrics=resolved_config.report_small_group_metrics,
                )
                validation_group_errors = json.loads(validation_group_error_json)
            collapsed_groups = [
                str(group_payload.get("group"))
                for group_payload in validation_group_errors
                if bool(group_payload.get("prediction_collapse"))
            ]
            history_entry: dict[str, Any] = {
                "epoch": int(epoch_index + 1),
                "train_loss": float(train_loss.detach().cpu().item()),
                "validation_loss": float(validation_loss),
                "macro_f1": float(validation_metrics["macro_f1"]),
                "early_stop_metric_value": float(validation_metric),
                "collapsed_groups": collapsed_groups,
                "group_robust_weight_scale": float(group_robust_scale),
                "adversary_weight_scale": float(adversary_scale),
            }
            if validation_group_summary is not None:
                history_entry.update(
                    {
                        "worst_group_accuracy": float(validation_group_summary["worst_group_accuracy"]),
                        "worst_group_f1": float(validation_group_summary["worst_group_f1"]),
                        "accuracy_gap": float(validation_group_summary["accuracy_gap"]),
                        "macro_f1_gap": float(validation_group_summary["macro_f1_gap"]),
                        "worst_group_accuracy_raw": float(validation_group_summary["worst_group_accuracy_raw"]),
                        "worst_group_f1_raw": float(validation_group_summary["worst_group_f1_raw"]),
                        "accuracy_gap_raw": float(validation_group_summary["accuracy_gap_raw"]),
                        "macro_f1_gap_raw": float(validation_group_summary["macro_f1_gap_raw"]),
                        "small_group_count": int(validation_group_summary["small_group_count"]),
                        "support_threshold_used": int(validation_group_summary["support_threshold_used"]),
                    }
                )
                if not pd.isna(validation_group_summary["worst_group_accuracy_supported"]):
                    history_entry["worst_group_accuracy_supported"] = float(
                        validation_group_summary["worst_group_accuracy_supported"]
                    )
                if not pd.isna(validation_group_summary["worst_group_f1_supported"]):
                    history_entry["worst_group_f1_supported"] = float(
                        validation_group_summary["worst_group_f1_supported"]
                    )
                if not pd.isna(validation_group_summary["accuracy_gap_supported"]):
                    history_entry["accuracy_gap_supported"] = float(
                        validation_group_summary["accuracy_gap_supported"]
                    )
                if not pd.isna(validation_group_summary["macro_f1_gap_supported"]):
                    history_entry["macro_f1_gap_supported"] = float(
                        validation_group_summary["macro_f1_gap_supported"]
                    )
            if validation_probe_accuracy is not None and validation_probe_macro_f1 is not None:
                history_entry.update(
                    {
                        "protected_probe_accuracy": float(validation_probe_accuracy),
                        "protected_probe_macro_f1": float(validation_probe_macro_f1),
                    }
                )
            if train_collapsed_group_names:
                history_entry["train_collapsed_groups"] = train_collapsed_group_names
            training_history.append(history_entry)

        if resolved_config.early_stop_patience > 0:
            if _is_better_validation_result(
                resolved_config.early_stop_metric,
                best_metric=best_validation_metric,
                best_loss=best_validation_loss,
                candidate_metric=validation_metric,
                candidate_loss=validation_loss,
            ):
                best_validation_metric = validation_metric
                best_validation_loss = validation_loss
                best_epoch = int(epoch_index + 1)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= resolved_config.early_stop_patience:
                    break
        else:
            best_validation_metric = validation_metric
            best_validation_loss = validation_loss
            best_epoch = int(epoch_index + 1)

    if resolved_config.early_stop_patience > 0 and best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        final_embeddings = model.encode(x, edge_index)
        final_logits = model.predict_main(final_embeddings)
        test_predictions = final_logits[test_indices].argmax(dim=1).detach().cpu().numpy()

    decoded_predictions = np.asarray(
        [class_names[int(index)] for index in test_predictions],
        dtype=object,
    )
    return GraphSAGENodeClassificationResult(
        predictions=decoded_predictions,
        node_embeddings=final_embeddings.detach().cpu().numpy(),
        validation_loss=best_validation_loss,
        validation_metric_value=best_validation_metric,
        validation_metric_name=resolved_config.early_stop_metric,
        validation_count=int(validation_indices.shape[0]),
        train_count=int(train_indices.shape[0]),
        validation_stratified=train_validation_split.stratified,
        best_epoch=int(best_epoch),
        training_history_json=_json_dumps(training_history),
    )
