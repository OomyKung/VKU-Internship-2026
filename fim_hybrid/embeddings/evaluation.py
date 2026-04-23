"""Shared downstream evaluation helpers for graph embeddings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from time import perf_counter
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fim_hybrid.clustering import ClusteringFrameworkError, cluster_nodes
from fim_hybrid.data_loader import LoadedDataset

from .base import EmbeddingResult, embedding_columns, sorted_node_ids, validate_embedding_frame
from .evaluation_splits import (
    LinkPredictionSplit,
    NodeClassificationSplit,
    build_node_train_validation_split,
    build_link_prediction_split,
    build_node_classification_split,
)
from .node_classification_models import (
    GraphSAGENodeClassificationConfig,
    compute_classification_metrics,
    derive_comparison_mode,
    resolve_graphsage_training_mode_settings,
    resolve_debias_mode,
    resolve_early_stop_metric,
    resolve_group_support_thresholds,
    resolve_group_weight_mode,
    resolve_imbalance_mode,
    resolve_node_classification_model,
    resolve_probe_model_type,
    resolve_training_mode,
    run_train_test_embedding_probe,
    serialize_class_distribution_summary,
    serialize_confusion_matrix_summary,
    serialize_per_class_metrics,
    serialize_split_diagnostics,
    split_distribution_payload,
    summarize_group_classification_metrics,
    train_graphsage_node_classifier,
)

if TYPE_CHECKING:
    from .benchmark import BenchmarkRunResult


SUPPORTED_EVALUATION_TASKS = (
    "node_classification",
    "link_prediction",
    "node_clustering",
)
SUPPORTED_LINK_EDGE_FEATURES = ("hadamard", "abs_diff", "concat", "dot")
EVALUATION_RESULT_COLUMNS = [
    "dataset",
    "method",
    "task",
    "status",
    "evaluation_base_seed",
    "evaluation_repeat_index",
    "effective_random_seed",
    "evaluation_seed_count",
    "repeated_split_count",
    "runtime_seconds",
    "embedding_runtime_seconds",
    "embedding_dim",
    "node_count",
    "evaluated_count",
    "train_count",
    "test_count",
    "label_column",
    "classifier",
    "edge_feature",
    "clustering_method",
    "clustering_category",
    "clustering_requested_input_mode",
    "clustering_input_mode",
    "num_clusters_found",
    "cluster_size_summary",
    "silhouette_score",
    "davies_bouldin_score",
    "calinski_harabasz_score",
    "comparison_mode",
    "training_mode",
    "imbalance_mode",
    "early_stop_metric",
    "debias_mode",
    "group_robust_weight",
    "group_weight_mode",
    "worst_group_boost_factor",
    "min_support_boost_factor",
    "min_group_support_threshold",
    "min_group_support_train",
    "min_group_support_eval",
    "report_small_group_metrics",
    "rebalance_batches_by_group",
    "fairness_score_alpha",
    "fairness_score_beta",
    "adversary_loss_weight",
    "gradient_reversal_lambda",
    "adversary_warmup_epochs",
    "group_robust_warmup_epochs",
    "adversary_hidden_dim",
    "adversary_num_layers",
    "adversary_dropout",
    "accuracy",
    "macro_f1",
    "micro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "roc_auc",
    "average_precision",
    "nmi",
    "ari",
    "protected_attribute_column",
    "best_group_accuracy",
    "worst_group_accuracy",
    "group_accuracy_gap",
    "accuracy_gap",
    "worst_group_accuracy_raw",
    "accuracy_gap_raw",
    "worst_group_accuracy_supported",
    "accuracy_gap_supported",
    "best_group_accuracy_name",
    "best_group_accuracy_count",
    "worst_group_accuracy_name",
    "worst_group_accuracy_count",
    "best_group_accuracy_supported_name",
    "best_group_accuracy_supported_count",
    "worst_group_accuracy_supported_name",
    "worst_group_accuracy_supported_count",
    "best_group_macro_f1",
    "worst_group_macro_f1",
    "group_macro_f1_gap",
    "best_group_f1",
    "worst_group_f1",
    "macro_f1_gap",
    "worst_group_f1_raw",
    "macro_f1_gap_raw",
    "worst_group_f1_supported",
    "macro_f1_gap_supported",
    "best_group_f1_name",
    "best_group_f1_count",
    "worst_group_f1_name",
    "worst_group_f1_count",
    "best_group_f1_supported_name",
    "best_group_f1_supported_count",
    "worst_group_f1_supported_name",
    "worst_group_f1_supported_count",
    "support_threshold_used",
    "supported_group_count",
    "protected_probe_status",
    "protected_probe_accuracy",
    "protected_probe_macro_f1",
    "protected_probe_model_type",
    "protected_probe_reason",
    "per_class_metrics_json",
    "confusion_matrix_json",
    "class_distribution_json",
    "per_group_metrics_json",
    "per_group_confusion_json",
    "group_error_diagnostics_json",
    "split_diagnostics_json",
    "group_support_counts_json",
    "groups_below_support_threshold_json",
    "small_group_metrics_json",
    "collapsed_groups",
    "small_group_count",
    "max_group_error",
    "best_epoch",
    "training_history_json",
    "notes",
    "unavailable_reason",
    "skipped_reason",
]


@dataclass(slots=True)
class _EvaluationContext:
    graph: nx.Graph
    tasks: tuple[str, ...]
    random_seed: int
    classification_label_column: str | None
    clustering_label_column: str | None
    link_prediction_edge_feature: str
    node_classification_model: str
    clustering_method: str
    clustering_input_mode: str
    clustering_n_clusters: int | None
    clustering_min_cluster_size: int | None
    clustering_method_config: dict[str, Any]
    training_mode: str
    imbalance_mode: str
    use_stratified_split: bool
    focal_gamma: float
    early_stop_metric: str
    early_stop_patience: int
    class_weight_smoothing: float
    node_validation_fraction: float
    debias_mode: str
    group_robust_weight: float
    group_weight_mode: str
    worst_group_boost_factor: float
    min_support_boost_factor: float
    min_group_support_threshold: int | None
    min_group_support_train: int
    min_group_support_eval: int
    report_small_group_metrics: bool
    rebalance_batches_by_group: bool
    fairness_score_alpha: float
    fairness_score_beta: float
    probe_model_type: str
    adversary_loss_weight: float
    gradient_reversal_lambda: float
    adversary_warmup_epochs: int
    group_robust_warmup_epochs: int
    adversary_hidden_dim: int
    adversary_num_layers: int
    adversary_dropout: float
    protected_attribute_column: str | None
    node_classification_labels: pd.Series | None
    node_classification_split: NodeClassificationSplit | None
    node_classification_skip_reason: str
    node_clustering_labels: pd.Series | None
    node_clustering_skip_reason: str
    link_prediction_split: LinkPredictionSplit | None
    link_prediction_skip_reason: str


def available_evaluation_tasks() -> tuple[str, ...]:
    """Return the supported downstream evaluation tasks."""

    return SUPPORTED_EVALUATION_TASKS


def resolve_evaluation_tasks(tasks: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize and validate requested evaluation tasks."""

    if tasks is None:
        return ()
    normalized = tuple(dict.fromkeys(str(task).strip().lower() for task in tasks))
    if not normalized:
        return ()
    if "all" in normalized:
        return SUPPORTED_EVALUATION_TASKS
    invalid = sorted(set(normalized) - set(SUPPORTED_EVALUATION_TASKS))
    if invalid:
        raise ValueError(
            f"Unsupported evaluation tasks: {invalid}. Supported values: {list(SUPPORTED_EVALUATION_TASKS)}."
        )
    return normalized


def _resolve_edge_feature_name(edge_feature: str) -> str:
    normalized = str(edge_feature).strip().lower()
    if normalized not in SUPPORTED_LINK_EDGE_FEATURES:
        raise ValueError(
            f"Unsupported link_prediction_edge_feature '{edge_feature}'. "
            f"Supported values: {list(SUPPORTED_LINK_EDGE_FEATURES)}."
        )
    return normalized


def _evaluation_graph(
    dataset: LoadedDataset,
    *,
    graph: nx.Graph | None = None,
    symmetrize_directed: bool = True,
) -> nx.Graph:
    if graph is not None:
        return graph.copy()
    if dataset.graph.is_directed() and symmetrize_directed:
        return dataset.graph.to_undirected()
    return dataset.graph.copy()


def _empty_result_row(
    *,
    dataset_name: str,
    method_name: str,
    task: str,
    embedding_runtime_seconds: float,
    embedding_dim: int,
    node_count: int,
    label_column: str | None = None,
    classifier: str | None = None,
    edge_feature: str | None = None,
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "method": method_name,
        "task": task,
        "status": "ok",
        "evaluation_base_seed": pd.NA,
        "evaluation_repeat_index": pd.NA,
        "effective_random_seed": pd.NA,
        "evaluation_seed_count": pd.NA,
        "repeated_split_count": pd.NA,
        "runtime_seconds": 0.0,
        "embedding_runtime_seconds": float(embedding_runtime_seconds),
        "embedding_dim": int(embedding_dim),
        "node_count": int(node_count),
        "evaluated_count": pd.NA,
        "train_count": pd.NA,
        "test_count": pd.NA,
        "label_column": label_column if label_column is not None else pd.NA,
        "classifier": classifier if classifier is not None else pd.NA,
        "edge_feature": edge_feature if edge_feature is not None else pd.NA,
        "clustering_method": pd.NA,
        "clustering_category": pd.NA,
        "clustering_requested_input_mode": pd.NA,
        "clustering_input_mode": pd.NA,
        "num_clusters_found": pd.NA,
        "cluster_size_summary": pd.NA,
        "silhouette_score": pd.NA,
        "davies_bouldin_score": pd.NA,
        "calinski_harabasz_score": pd.NA,
        "comparison_mode": pd.NA,
        "training_mode": pd.NA,
        "imbalance_mode": pd.NA,
        "early_stop_metric": pd.NA,
        "debias_mode": pd.NA,
        "group_robust_weight": pd.NA,
        "group_weight_mode": pd.NA,
        "worst_group_boost_factor": pd.NA,
        "min_support_boost_factor": pd.NA,
        "min_group_support_threshold": pd.NA,
        "min_group_support_train": pd.NA,
        "min_group_support_eval": pd.NA,
        "report_small_group_metrics": pd.NA,
        "rebalance_batches_by_group": pd.NA,
        "fairness_score_alpha": pd.NA,
        "fairness_score_beta": pd.NA,
        "adversary_loss_weight": pd.NA,
        "gradient_reversal_lambda": pd.NA,
        "adversary_warmup_epochs": pd.NA,
        "group_robust_warmup_epochs": pd.NA,
        "adversary_hidden_dim": pd.NA,
        "adversary_num_layers": pd.NA,
        "adversary_dropout": pd.NA,
        "accuracy": pd.NA,
        "macro_f1": pd.NA,
        "micro_f1": pd.NA,
        "weighted_f1": pd.NA,
        "balanced_accuracy": pd.NA,
        "macro_precision": pd.NA,
        "macro_recall": pd.NA,
        "roc_auc": pd.NA,
        "average_precision": pd.NA,
        "nmi": pd.NA,
        "ari": pd.NA,
        "protected_attribute_column": pd.NA,
        "best_group_accuracy": pd.NA,
        "worst_group_accuracy": pd.NA,
        "group_accuracy_gap": pd.NA,
        "accuracy_gap": pd.NA,
        "worst_group_accuracy_raw": pd.NA,
        "accuracy_gap_raw": pd.NA,
        "worst_group_accuracy_supported": pd.NA,
        "accuracy_gap_supported": pd.NA,
        "best_group_accuracy_name": pd.NA,
        "best_group_accuracy_count": pd.NA,
        "worst_group_accuracy_name": pd.NA,
        "worst_group_accuracy_count": pd.NA,
        "best_group_accuracy_supported_name": pd.NA,
        "best_group_accuracy_supported_count": pd.NA,
        "worst_group_accuracy_supported_name": pd.NA,
        "worst_group_accuracy_supported_count": pd.NA,
        "best_group_macro_f1": pd.NA,
        "worst_group_macro_f1": pd.NA,
        "group_macro_f1_gap": pd.NA,
        "best_group_f1": pd.NA,
        "worst_group_f1": pd.NA,
        "macro_f1_gap": pd.NA,
        "worst_group_f1_raw": pd.NA,
        "macro_f1_gap_raw": pd.NA,
        "worst_group_f1_supported": pd.NA,
        "macro_f1_gap_supported": pd.NA,
        "best_group_f1_name": pd.NA,
        "best_group_f1_count": pd.NA,
        "worst_group_f1_name": pd.NA,
        "worst_group_f1_count": pd.NA,
        "best_group_f1_supported_name": pd.NA,
        "best_group_f1_supported_count": pd.NA,
        "worst_group_f1_supported_name": pd.NA,
        "worst_group_f1_supported_count": pd.NA,
        "support_threshold_used": pd.NA,
        "supported_group_count": pd.NA,
        "protected_probe_status": pd.NA,
        "protected_probe_accuracy": pd.NA,
        "protected_probe_macro_f1": pd.NA,
        "protected_probe_model_type": pd.NA,
        "protected_probe_reason": "",
        "per_class_metrics_json": pd.NA,
        "confusion_matrix_json": pd.NA,
        "class_distribution_json": pd.NA,
        "per_group_metrics_json": pd.NA,
        "per_group_confusion_json": pd.NA,
        "group_error_diagnostics_json": pd.NA,
        "split_diagnostics_json": pd.NA,
        "group_support_counts_json": pd.NA,
        "groups_below_support_threshold_json": pd.NA,
        "small_group_metrics_json": pd.NA,
        "collapsed_groups": pd.NA,
        "small_group_count": pd.NA,
        "max_group_error": pd.NA,
        "best_epoch": pd.NA,
        "training_history_json": pd.NA,
        "notes": "",
        "unavailable_reason": "",
        "skipped_reason": "",
    }


def _skip_row(
    *,
    dataset_name: str,
    method_name: str,
    task: str,
    reason: str,
    embedding_runtime_seconds: float,
    embedding_dim: int,
    node_count: int,
    label_column: str | None = None,
    classifier: str | None = None,
    edge_feature: str | None = None,
) -> dict[str, Any]:
    row = _empty_result_row(
        dataset_name=dataset_name,
        method_name=method_name,
        task=task,
        embedding_runtime_seconds=embedding_runtime_seconds,
        embedding_dim=embedding_dim,
        node_count=node_count,
        label_column=label_column,
        classifier=classifier,
        edge_feature=edge_feature,
    )
    row["status"] = "skipped"
    row["skipped_reason"] = reason
    return row


def _attribute_lookup(dataset: LoadedDataset) -> pd.DataFrame:
    if "node_id" not in dataset.node_attributes.columns:
        raise ValueError("dataset.node_attributes must contain a node_id column.")
    return dataset.node_attributes.set_index("node_id", drop=False)


def _aligned_label_series(
    dataset: LoadedDataset,
    *,
    node_order: Sequence[Any],
    label_column: str,
) -> tuple[pd.Series | None, str]:
    if label_column not in dataset.node_attributes.columns:
        return None, f"Label column '{label_column}' is not present in dataset.node_attributes."
    label_series = _attribute_lookup(dataset).reindex(list(node_order))[label_column]
    if label_series.isna().any():
        return None, f"Label column '{label_column}' contains missing values."
    if label_series.nunique() < 2:
        return None, f"Label column '{label_column}' has fewer than two classes."
    return label_series.astype(str), ""


def _coerce_embedding_result(
    embedding_result_or_frame: EmbeddingResult | pd.DataFrame,
    *,
    graph: nx.Graph,
    method_name: str | None = None,
    embedding_runtime_seconds: float | None = None,
) -> tuple[str, pd.DataFrame, float, int, int]:
    node_order = sorted_node_ids(graph)
    if isinstance(embedding_result_or_frame, EmbeddingResult):
        result = embedding_result_or_frame
        return (
            result.method_name,
            result.embedding_frame.copy(),
            float(result.runtime_seconds),
            int(result.embedding_dim),
            int(result.node_count),
        )

    resolved_method_name = str(method_name or "embedding")
    frame = validate_embedding_frame(
        node_order,
        embedding_result_or_frame.copy(),
        method_name=resolved_method_name,
    )
    return (
        resolved_method_name,
        frame,
        float(0.0 if embedding_runtime_seconds is None else embedding_runtime_seconds),
        len(embedding_columns(frame)),
        len(node_order),
    )


def _resolve_label_columns(
    *,
    label_column: str | None,
    classification_label_column: str | None,
    clustering_label_column: str | None,
) -> tuple[str | None, str | None]:
    return (
        classification_label_column if classification_label_column is not None else label_column,
        clustering_label_column if clustering_label_column is not None else label_column,
    )


def _validate_node_classification_configuration(
    *,
    node_classification_model: str,
    training_mode: str | None,
    imbalance_mode: str,
    debias_mode: str,
    focal_gamma: float,
    early_stop_metric: str,
    early_stop_patience: int,
    class_weight_smoothing: float,
    node_validation_fraction: float,
    group_robust_weight: float,
    group_weight_mode: str,
    worst_group_boost_factor: float,
    min_support_boost_factor: float,
    min_group_support_threshold: int | None,
    min_group_support_train: int | None,
    min_group_support_eval: int | None,
    report_small_group_metrics: bool,
    rebalance_batches_by_group: bool,
    fairness_score_alpha: float,
    fairness_score_beta: float,
    probe_model_type: str,
    adversary_loss_weight: float,
    gradient_reversal_lambda: float,
    adversary_warmup_epochs: int,
    group_robust_warmup_epochs: int,
    adversary_hidden_dim: int,
    adversary_num_layers: int,
    adversary_dropout: float,
) -> dict[str, Any]:
    resolved_model = resolve_node_classification_model(node_classification_model)
    resolved_training_settings = resolve_graphsage_training_mode_settings(
        training_mode=training_mode,
        imbalance_mode=imbalance_mode,
        debias_mode=debias_mode,
        group_weight_mode=group_weight_mode,
        group_robust_weight=group_robust_weight,
        worst_group_boost_factor=worst_group_boost_factor,
        min_support_boost_factor=min_support_boost_factor,
        rebalance_batches_by_group=rebalance_batches_by_group,
        early_stop_metric=early_stop_metric,
        adversary_loss_weight=adversary_loss_weight,
        gradient_reversal_lambda=gradient_reversal_lambda,
        adversary_hidden_dim=adversary_hidden_dim,
        adversary_num_layers=adversary_num_layers,
        adversary_dropout=adversary_dropout,
    )
    resolved_imbalance_mode = resolve_imbalance_mode(str(resolved_training_settings["imbalance_mode"]))
    resolved_debias_mode = resolve_debias_mode(str(resolved_training_settings["debias_mode"]))
    resolved_group_weight_mode = resolve_group_weight_mode(str(resolved_training_settings["group_weight_mode"]))
    resolved_probe_model_type = resolve_probe_model_type(probe_model_type)
    resolved_early_stop_metric = resolve_early_stop_metric(str(resolved_training_settings["early_stop_metric"]))
    resolved_min_group_support_train, resolved_min_group_support_eval = resolve_group_support_thresholds(
        min_group_support_threshold=min_group_support_threshold,
        min_group_support_train=min_group_support_train,
        min_group_support_eval=min_group_support_eval,
    )
    resolved_training_mode = resolve_training_mode(
        resolved_training_settings["training_mode"],
        imbalance_mode=resolved_imbalance_mode,
        debias_mode=resolved_debias_mode,
        group_weight_mode=resolved_group_weight_mode,
        group_robust_weight=float(resolved_training_settings["group_robust_weight"]),
    )
    if float(focal_gamma) < 0.0:
        raise ValueError("focal_gamma must be non-negative.")
    if int(early_stop_patience) < 0:
        raise ValueError("early_stop_patience must be non-negative.")
    if float(class_weight_smoothing) < 0.0:
        raise ValueError("class_weight_smoothing must be non-negative.")
    if not 0.0 < float(node_validation_fraction) < 1.0:
        raise ValueError("node_validation_fraction must be between 0.0 and 1.0.")
    if float(resolved_training_settings["group_robust_weight"]) < 0.0:
        raise ValueError("group_robust_weight must be non-negative.")
    if float(resolved_training_settings["worst_group_boost_factor"]) < 1.0:
        raise ValueError("worst_group_boost_factor must be at least 1.0.")
    if float(resolved_training_settings["min_support_boost_factor"]) < 1.0:
        raise ValueError("min_support_boost_factor must be at least 1.0.")
    if float(fairness_score_alpha) < 0.0:
        raise ValueError("fairness_score_alpha must be non-negative.")
    if float(fairness_score_beta) < 0.0:
        raise ValueError("fairness_score_beta must be non-negative.")
    if float(resolved_training_settings["adversary_loss_weight"]) < 0.0:
        raise ValueError("adversary_loss_weight must be non-negative.")
    if float(resolved_training_settings["gradient_reversal_lambda"]) < 0.0:
        raise ValueError("gradient_reversal_lambda must be non-negative.")
    if int(adversary_warmup_epochs) < 0:
        raise ValueError("adversary_warmup_epochs must be non-negative.")
    if int(group_robust_warmup_epochs) < 0:
        raise ValueError("group_robust_warmup_epochs must be non-negative.")
    if int(resolved_training_settings["adversary_hidden_dim"]) < 1:
        raise ValueError("adversary_hidden_dim must be at least 1.")
    if int(resolved_training_settings["adversary_num_layers"]) < 1:
        raise ValueError("adversary_num_layers must be at least 1.")
    if not 0.0 <= float(resolved_training_settings["adversary_dropout"]) < 1.0:
        raise ValueError("adversary_dropout must be in the interval [0.0, 1.0).")

    uses_graphsage_only_option = (
        resolved_training_mode != "baseline"
        or resolved_imbalance_mode != "none"
        or resolved_debias_mode != "none"
        or abs(float(focal_gamma) - 2.0) > 1e-12
        or resolved_early_stop_metric != "accuracy"
        or int(early_stop_patience) != 0
        or abs(float(class_weight_smoothing)) > 1e-12
        or abs(float(node_validation_fraction) - 0.2) > 1e-12
        or abs(float(resolved_training_settings["group_robust_weight"])) > 1e-12
        or resolved_group_weight_mode != "none"
        or abs(float(resolved_training_settings["worst_group_boost_factor"]) - 2.0) > 1e-12
        or (
            resolved_group_weight_mode == "min_support_boost"
            and abs(float(resolved_training_settings["min_support_boost_factor"]) - 2.0) > 1e-12
        )
        or bool(resolved_training_settings["rebalance_batches_by_group"])
        or abs(float(fairness_score_alpha) - 0.25) > 1e-12
        or abs(float(fairness_score_beta) - 0.25) > 1e-12
        or abs(float(resolved_training_settings["adversary_loss_weight"]) - 1.0) > 1e-12
        or abs(float(resolved_training_settings["gradient_reversal_lambda"]) - 1.0) > 1e-12
        or int(adversary_warmup_epochs) != 0
        or int(group_robust_warmup_epochs) != 0
        or int(resolved_training_settings["adversary_hidden_dim"]) != 64
        or int(resolved_training_settings["adversary_num_layers"]) != 1
        or abs(float(resolved_training_settings["adversary_dropout"]) - 0.2) > 1e-12
    )
    if resolved_model == "logistic_regression" and uses_graphsage_only_option:
        raise ValueError(
            "node_classification_model='logistic_regression' does not support "
            "imbalance, debiasing, validation, or early-stopping options. "
            "Use node_classification_model='graphsage' or reset those flags to defaults."
        )

    return {
        "node_classification_model": resolved_model,
        "training_mode": resolved_training_mode,
        "imbalance_mode": resolved_imbalance_mode,
        "early_stop_metric": resolved_early_stop_metric,
        "debias_mode": resolved_debias_mode,
        "probe_model_type": resolved_probe_model_type,
        "group_robust_weight": float(resolved_training_settings["group_robust_weight"]),
        "group_weight_mode": resolved_group_weight_mode,
        "worst_group_boost_factor": float(resolved_training_settings["worst_group_boost_factor"]),
        "min_support_boost_factor": float(resolved_training_settings["min_support_boost_factor"]),
        "rebalance_batches_by_group": bool(resolved_training_settings["rebalance_batches_by_group"]),
        "adversary_loss_weight": float(resolved_training_settings["adversary_loss_weight"]),
        "gradient_reversal_lambda": float(resolved_training_settings["gradient_reversal_lambda"]),
        "adversary_hidden_dim": int(resolved_training_settings["adversary_hidden_dim"]),
        "adversary_num_layers": int(resolved_training_settings["adversary_num_layers"]),
        "adversary_dropout": float(resolved_training_settings["adversary_dropout"]),
        "min_group_support_train": int(resolved_min_group_support_train),
        "min_group_support_eval": int(resolved_min_group_support_eval),
    }


def _build_context(
    dataset: LoadedDataset,
    *,
    tasks: Sequence[str] | None,
    graph: nx.Graph | None = None,
    symmetrize_directed: bool = True,
    random_seed: int = 42,
    label_column: str | None = None,
    classification_label_column: str | None = None,
    clustering_label_column: str | None = None,
    node_test_fraction: float = 0.25,
    link_test_fraction: float = 0.25,
    link_negative_ratio: float = 1.0,
    link_prediction_edge_feature: str = "hadamard",
    node_classification_model: str = "logistic_regression",
    clustering_method: str = "kmeans",
    clustering_input_mode: str = "embedding",
    clustering_n_clusters: int | None = None,
    clustering_min_cluster_size: int | None = None,
    clustering_method_config: dict[str, Any] | None = None,
    training_mode: str | None = None,
    imbalance_mode: str = "none",
    debias_mode: str = "none",
    focal_gamma: float = 2.0,
    use_stratified_split: bool = True,
    early_stop_metric: str = "accuracy",
    early_stop_patience: int = 0,
    class_weight_smoothing: float = 0.0,
    node_validation_fraction: float = 0.2,
    group_robust_weight: float = 0.0,
    group_weight_mode: str = "none",
    worst_group_boost_factor: float = 2.0,
    min_support_boost_factor: float = 2.0,
    min_group_support_threshold: int | None = None,
    min_group_support_train: int | None = None,
    min_group_support_eval: int | None = None,
    report_small_group_metrics: bool = False,
    rebalance_batches_by_group: bool = False,
    fairness_score_alpha: float = 0.25,
    fairness_score_beta: float = 0.25,
    probe_model_type: str = "linear",
    adversary_loss_weight: float = 1.0,
    gradient_reversal_lambda: float = 1.0,
    adversary_warmup_epochs: int = 0,
    group_robust_warmup_epochs: int = 0,
    adversary_hidden_dim: int = 64,
    adversary_num_layers: int = 1,
    adversary_dropout: float = 0.2,
    protected_attribute_column: str | None = None,
) -> _EvaluationContext:
    normalized_tasks = resolve_evaluation_tasks(tasks)
    evaluation_graph = _evaluation_graph(dataset, graph=graph, symmetrize_directed=symmetrize_directed)
    classification_column, clustering_column = _resolve_label_columns(
        label_column=label_column,
        classification_label_column=classification_label_column,
        clustering_label_column=clustering_label_column,
    )
    node_order = sorted_node_ids(evaluation_graph)
    edge_feature = _resolve_edge_feature_name(link_prediction_edge_feature)
    resolved_node_classification_model = resolve_node_classification_model(node_classification_model)
    resolved_clustering_method = str(clustering_method).strip().lower()
    resolved_clustering_input_mode = str(clustering_input_mode).strip().lower()
    resolved_training_mode = "baseline"
    resolved_imbalance_mode = resolve_imbalance_mode(imbalance_mode)
    resolved_debias_mode = resolve_debias_mode(debias_mode)
    resolved_early_stop_metric = resolve_early_stop_metric(early_stop_metric)
    resolved_group_weight_mode = resolve_group_weight_mode(group_weight_mode)
    resolved_group_robust_weight = float(group_robust_weight)
    resolved_worst_group_boost_factor = float(worst_group_boost_factor)
    resolved_min_support_boost_factor = float(min_support_boost_factor)
    resolved_rebalance_batches_by_group = bool(rebalance_batches_by_group)
    resolved_probe_model_type = resolve_probe_model_type(probe_model_type)
    resolved_adversary_loss_weight = float(adversary_loss_weight)
    resolved_gradient_reversal_lambda = float(gradient_reversal_lambda)
    resolved_adversary_hidden_dim = int(adversary_hidden_dim)
    resolved_adversary_num_layers = int(adversary_num_layers)
    resolved_adversary_dropout = float(adversary_dropout)
    resolved_min_group_support_train, resolved_min_group_support_eval = resolve_group_support_thresholds(
        min_group_support_threshold=min_group_support_threshold,
        min_group_support_train=min_group_support_train,
        min_group_support_eval=min_group_support_eval,
    )

    node_classification_labels: pd.Series | None = None
    node_classification_split: NodeClassificationSplit | None = None
    node_classification_skip_reason = ""
    if "node_classification" in normalized_tasks:
        resolved_node_configuration = _validate_node_classification_configuration(
            node_classification_model=node_classification_model,
            training_mode=training_mode,
            imbalance_mode=imbalance_mode,
            debias_mode=debias_mode,
            focal_gamma=focal_gamma,
            early_stop_metric=early_stop_metric,
            early_stop_patience=early_stop_patience,
            class_weight_smoothing=class_weight_smoothing,
            node_validation_fraction=node_validation_fraction,
            group_robust_weight=group_robust_weight,
            group_weight_mode=group_weight_mode,
            worst_group_boost_factor=worst_group_boost_factor,
            min_support_boost_factor=min_support_boost_factor,
            min_group_support_threshold=min_group_support_threshold,
            min_group_support_train=min_group_support_train,
            min_group_support_eval=min_group_support_eval,
            report_small_group_metrics=report_small_group_metrics,
            rebalance_batches_by_group=rebalance_batches_by_group,
            fairness_score_alpha=fairness_score_alpha,
            fairness_score_beta=fairness_score_beta,
            probe_model_type=probe_model_type,
            adversary_loss_weight=adversary_loss_weight,
            gradient_reversal_lambda=gradient_reversal_lambda,
            adversary_warmup_epochs=adversary_warmup_epochs,
            group_robust_warmup_epochs=group_robust_warmup_epochs,
            adversary_hidden_dim=adversary_hidden_dim,
            adversary_num_layers=adversary_num_layers,
            adversary_dropout=adversary_dropout,
        )
        resolved_node_classification_model = str(resolved_node_configuration["node_classification_model"])
        resolved_training_mode = str(resolved_node_configuration["training_mode"])
        resolved_imbalance_mode = str(resolved_node_configuration["imbalance_mode"])
        resolved_early_stop_metric = str(resolved_node_configuration["early_stop_metric"])
        resolved_debias_mode = str(resolved_node_configuration["debias_mode"])
        resolved_probe_model_type = str(resolved_node_configuration["probe_model_type"])
        resolved_group_robust_weight = float(resolved_node_configuration["group_robust_weight"])
        resolved_group_weight_mode = str(resolved_node_configuration["group_weight_mode"])
        resolved_worst_group_boost_factor = float(resolved_node_configuration["worst_group_boost_factor"])
        resolved_min_support_boost_factor = float(resolved_node_configuration["min_support_boost_factor"])
        resolved_rebalance_batches_by_group = bool(
            resolved_node_configuration["rebalance_batches_by_group"]
        )
        resolved_adversary_loss_weight = float(resolved_node_configuration["adversary_loss_weight"])
        resolved_gradient_reversal_lambda = float(
            resolved_node_configuration["gradient_reversal_lambda"]
        )
        resolved_adversary_hidden_dim = int(resolved_node_configuration["adversary_hidden_dim"])
        resolved_adversary_num_layers = int(resolved_node_configuration["adversary_num_layers"])
        resolved_adversary_dropout = float(resolved_node_configuration["adversary_dropout"])
        resolved_min_group_support_train = int(resolved_node_configuration["min_group_support_train"])
        resolved_min_group_support_eval = int(resolved_node_configuration["min_group_support_eval"])
        if resolved_debias_mode == "adversarial" and protected_attribute_column is None:
            raise ValueError(
                "debias_mode='adversarial' requires protected_attribute_column to be set."
            )
        if (
            resolved_group_weight_mode != "none"
            and float(resolved_group_robust_weight) > 0.0
            and protected_attribute_column is None
        ):
            raise ValueError(
                "Protected-group robust weighting requires protected_attribute_column to be set."
            )
        if (
            resolved_early_stop_metric in {"worst_group_f1_raw", "worst_group_f1_supported"}
            and protected_attribute_column is None
        ):
            raise ValueError(
                f"early_stop_metric='{resolved_early_stop_metric}' requires protected_attribute_column to be set."
            )
        if resolved_early_stop_metric == "fairness_score" and protected_attribute_column is None:
            raise ValueError(
                "early_stop_metric='fairness_score' requires protected_attribute_column to be set."
            )
        if (
            protected_attribute_column is not None
            and (
                resolved_debias_mode == "adversarial"
                or (
                    resolved_group_weight_mode != "none"
                    and float(resolved_group_robust_weight) > 0.0
                )
                or bool(resolved_rebalance_batches_by_group)
                or resolved_early_stop_metric in {"worst_group_f1_raw", "worst_group_f1_supported", "fairness_score"}
            )
        ):
            if protected_attribute_column not in dataset.node_attributes.columns:
                raise ValueError(
                    f"Protected attribute column '{protected_attribute_column}' is not present in dataset.node_attributes."
                )
            protected_series = _attribute_lookup(dataset).reindex(list(node_order))[protected_attribute_column]
            if protected_series.isna().any():
                raise ValueError(
                    f"Protected attribute column '{protected_attribute_column}' contains missing values."
                )
            if protected_series.nunique() < 2:
                raise ValueError(
                    f"Protected attribute column '{protected_attribute_column}' has fewer than two classes."
                )
        if classification_column is None:
            node_classification_skip_reason = "Node classification requires a label column."
        else:
            node_classification_labels, node_classification_skip_reason = _aligned_label_series(
                dataset,
                node_order=node_order,
                label_column=classification_column,
            )
            if node_classification_labels is not None:
                try:
                    node_classification_split = build_node_classification_split(
                        node_order,
                        node_classification_labels.to_numpy(dtype=object),
                        test_fraction=node_test_fraction,
                        random_seed=random_seed,
                        use_stratified_split=use_stratified_split,
                    )
                except ValueError as exc:
                    node_classification_skip_reason = str(exc)
                    node_classification_labels = None

    node_clustering_labels: pd.Series | None = None
    node_clustering_skip_reason = ""
    if "node_clustering" in normalized_tasks:
        if clustering_column is None:
            node_clustering_skip_reason = "Node clustering requires a label column."
        else:
            node_clustering_labels, node_clustering_skip_reason = _aligned_label_series(
                dataset,
                node_order=node_order,
                label_column=clustering_column,
            )

    link_prediction_split: LinkPredictionSplit | None = None
    link_prediction_skip_reason = ""
    if "link_prediction" in normalized_tasks:
        try:
            link_prediction_split = build_link_prediction_split(
                evaluation_graph,
                test_fraction=link_test_fraction,
                negative_ratio=link_negative_ratio,
                random_seed=random_seed,
            )
        except ValueError as exc:
            link_prediction_skip_reason = str(exc)

    return _EvaluationContext(
        graph=evaluation_graph,
        tasks=normalized_tasks,
        random_seed=int(random_seed),
        classification_label_column=classification_column,
        clustering_label_column=clustering_column,
        link_prediction_edge_feature=edge_feature,
        node_classification_model=resolved_node_classification_model,
        clustering_method=resolved_clustering_method,
        clustering_input_mode=resolved_clustering_input_mode,
        clustering_n_clusters=(
            None if clustering_n_clusters is None else int(clustering_n_clusters)
        ),
        clustering_min_cluster_size=(
            None if clustering_min_cluster_size is None else int(clustering_min_cluster_size)
        ),
        clustering_method_config=dict(clustering_method_config or {}),
        training_mode=resolved_training_mode,
        imbalance_mode=resolved_imbalance_mode,
        use_stratified_split=bool(use_stratified_split),
        focal_gamma=float(focal_gamma),
        early_stop_metric=resolved_early_stop_metric,
        early_stop_patience=int(early_stop_patience),
        class_weight_smoothing=float(class_weight_smoothing),
        node_validation_fraction=float(node_validation_fraction),
        debias_mode=resolved_debias_mode,
        group_robust_weight=float(resolved_group_robust_weight),
        group_weight_mode=resolved_group_weight_mode,
        worst_group_boost_factor=float(resolved_worst_group_boost_factor),
        min_support_boost_factor=float(resolved_min_support_boost_factor),
        min_group_support_threshold=(
            None if min_group_support_threshold is None else int(min_group_support_threshold)
        ),
        min_group_support_train=int(resolved_min_group_support_train),
        min_group_support_eval=int(resolved_min_group_support_eval),
        report_small_group_metrics=bool(report_small_group_metrics),
        rebalance_batches_by_group=bool(resolved_rebalance_batches_by_group),
        fairness_score_alpha=float(fairness_score_alpha),
        fairness_score_beta=float(fairness_score_beta),
        probe_model_type=resolved_probe_model_type,
        adversary_loss_weight=float(resolved_adversary_loss_weight),
        gradient_reversal_lambda=float(resolved_gradient_reversal_lambda),
        adversary_warmup_epochs=int(adversary_warmup_epochs),
        group_robust_warmup_epochs=int(group_robust_warmup_epochs),
        adversary_hidden_dim=int(resolved_adversary_hidden_dim),
        adversary_num_layers=int(resolved_adversary_num_layers),
        adversary_dropout=float(resolved_adversary_dropout),
        protected_attribute_column=(
            None if protected_attribute_column is None else str(protected_attribute_column).strip()
        ) or None,
        node_classification_labels=node_classification_labels,
        node_classification_split=node_classification_split,
        node_classification_skip_reason=node_classification_skip_reason,
        node_clustering_labels=node_clustering_labels,
        node_clustering_skip_reason=node_clustering_skip_reason,
        link_prediction_split=link_prediction_split,
        link_prediction_skip_reason=link_prediction_skip_reason,
    )


def _node_vectors_lookup(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index("node_id", drop=False)


def _split_diagnostics_for_nodes(
    *,
    labels: pd.Series,
    node_ids: Sequence[Any],
    protected_series: pd.Series | None = None,
    underrepresented_threshold: int = 0,
) -> dict[str, Any]:
    """Return label/protected diagnostics for one explicit node split."""

    label_subset = labels.loc[list(node_ids)].astype(str)
    if protected_series is None:
        return split_distribution_payload(
            label_subset.to_numpy(dtype=object),
            underrepresented_threshold=underrepresented_threshold,
        )

    protected_subset = protected_series.reindex(list(node_ids))
    payload = split_distribution_payload(
        label_subset.to_numpy(dtype=object),
        protected_values=protected_subset.fillna("<missing>").astype(str).to_numpy(dtype=object),
        underrepresented_threshold=underrepresented_threshold,
    )
    payload["protected_missing_count"] = int(protected_subset.isna().sum())
    payload["protected_has_missing"] = bool(protected_subset.isna().any())
    return payload


def _edge_feature_matrix(
    embedding_lookup: pd.DataFrame,
    edges: Sequence[tuple[Any, Any]],
    *,
    feature_name: str,
) -> np.ndarray:
    vector_columns = embedding_columns(embedding_lookup.reset_index(drop=True))
    if not edges:
        return np.empty((0, len(vector_columns)), dtype=float)

    source_vectors = embedding_lookup.loc[[edge[0] for edge in edges], vector_columns].to_numpy(dtype=float)
    target_vectors = embedding_lookup.loc[[edge[1] for edge in edges], vector_columns].to_numpy(dtype=float)
    if feature_name == "hadamard":
        return source_vectors * target_vectors
    if feature_name == "abs_diff":
        return np.abs(source_vectors - target_vectors)
    if feature_name == "concat":
        return np.concatenate([source_vectors, target_vectors], axis=1)
    if feature_name == "dot":
        return np.sum(source_vectors * target_vectors, axis=1, keepdims=True)
    raise ValueError(f"Unsupported edge feature '{feature_name}'.")


def _evaluate_node_classification(
    *,
    dataset: LoadedDataset,
    method_name: str,
    embedding_frame: pd.DataFrame,
    embedding_runtime_seconds: float,
    embedding_dim: int,
    node_count: int,
    context: _EvaluationContext,
) -> dict[str, Any]:
    classifier_name = context.node_classification_model
    if context.node_classification_labels is None or context.node_classification_split is None:
        return _skip_row(
            dataset_name=dataset.name,
            method_name=method_name,
            task="node_classification",
            reason=context.node_classification_skip_reason or "Node classification could not be prepared.",
            embedding_runtime_seconds=embedding_runtime_seconds,
            embedding_dim=embedding_dim,
            node_count=node_count,
            label_column=context.classification_label_column,
            classifier=classifier_name,
        )

    start = perf_counter()
    row = _empty_result_row(
        dataset_name=dataset.name,
        method_name=method_name,
        task="node_classification",
        embedding_runtime_seconds=embedding_runtime_seconds,
        embedding_dim=embedding_dim,
        node_count=node_count,
        label_column=context.classification_label_column,
        classifier=classifier_name,
    )
    embedding_lookup = _node_vectors_lookup(embedding_frame)
    vector_columns = embedding_columns(embedding_frame)
    split = context.node_classification_split
    labels = context.node_classification_labels
    node_order = list(sorted_node_ids(context.graph))
    node_index_lookup = {node_id: index for index, node_id in enumerate(node_order)}
    protected_series_aligned: pd.Series | None = None
    if context.protected_attribute_column is not None and context.protected_attribute_column in dataset.node_attributes.columns:
        protected_series_aligned = _attribute_lookup(dataset).reindex(node_order)[context.protected_attribute_column]
    y_test = labels.loc[list(split.test_node_ids)].to_numpy(dtype=object)
    predictions: np.ndarray
    probe_train_embeddings: np.ndarray | None = None
    probe_test_embeddings: np.ndarray | None = None
    split_diagnostics_payloads: dict[str, dict[str, Any]] = {
        "train": _split_diagnostics_for_nodes(
            labels=labels,
            node_ids=split.train_node_ids,
            protected_series=protected_series_aligned,
            underrepresented_threshold=context.min_group_support_train,
        ),
        "test": _split_diagnostics_for_nodes(
            labels=labels,
            node_ids=split.test_node_ids,
            protected_series=protected_series_aligned,
            underrepresented_threshold=context.min_group_support_eval,
        ),
    }
    notes_parts = [
        f"classes={int(labels.nunique())}",
        f"stratified={split.stratified}",
    ]
    row["comparison_mode"] = (
        derive_comparison_mode(
            training_mode=context.training_mode,
            debias_mode=context.debias_mode,
            group_weight_mode=context.group_weight_mode,
            group_robust_weight=context.group_robust_weight,
        )
        if classifier_name == "graphsage"
        else pd.NA
    )
    row["debias_mode"] = context.debias_mode if classifier_name == "graphsage" else "none"
    row["training_mode"] = context.training_mode if classifier_name == "graphsage" else "baseline"
    row["imbalance_mode"] = context.imbalance_mode if classifier_name == "graphsage" else "none"
    row["early_stop_metric"] = context.early_stop_metric if classifier_name == "graphsage" else "accuracy"
    row["group_robust_weight"] = (
        float(context.group_robust_weight) if classifier_name == "graphsage" else pd.NA
    )
    row["group_weight_mode"] = (
        context.group_weight_mode if classifier_name == "graphsage" else pd.NA
    )
    row["worst_group_boost_factor"] = (
        float(context.worst_group_boost_factor) if classifier_name == "graphsage" else pd.NA
    )
    row["min_support_boost_factor"] = (
        float(context.min_support_boost_factor) if classifier_name == "graphsage" else pd.NA
    )
    row["min_group_support_threshold"] = (
        (
            int(context.min_group_support_threshold)
            if context.min_group_support_threshold is not None
            else int(context.min_group_support_eval)
            if context.min_group_support_train == context.min_group_support_eval
            else pd.NA
        )
        if classifier_name == "graphsage"
        else pd.NA
    )
    row["min_group_support_train"] = (
        int(context.min_group_support_train) if classifier_name == "graphsage" else pd.NA
    )
    row["min_group_support_eval"] = (
        int(context.min_group_support_eval) if classifier_name == "graphsage" else pd.NA
    )
    row["report_small_group_metrics"] = (
        bool(context.report_small_group_metrics) if classifier_name == "graphsage" else pd.NA
    )
    row["rebalance_batches_by_group"] = (
        bool(context.rebalance_batches_by_group) if classifier_name == "graphsage" else pd.NA
    )
    row["fairness_score_alpha"] = (
        float(context.fairness_score_alpha) if classifier_name == "graphsage" else pd.NA
    )
    row["fairness_score_beta"] = (
        float(context.fairness_score_beta) if classifier_name == "graphsage" else pd.NA
    )
    row["adversary_loss_weight"] = (
        float(context.adversary_loss_weight) if classifier_name == "graphsage" else pd.NA
    )
    row["gradient_reversal_lambda"] = (
        float(context.gradient_reversal_lambda) if classifier_name == "graphsage" else pd.NA
    )
    row["adversary_warmup_epochs"] = (
        int(context.adversary_warmup_epochs) if classifier_name == "graphsage" else pd.NA
    )
    row["group_robust_warmup_epochs"] = (
        int(context.group_robust_warmup_epochs) if classifier_name == "graphsage" else pd.NA
    )
    row["adversary_hidden_dim"] = (
        int(context.adversary_hidden_dim) if classifier_name == "graphsage" else pd.NA
    )
    row["adversary_num_layers"] = (
        int(context.adversary_num_layers) if classifier_name == "graphsage" else pd.NA
    )
    row["adversary_dropout"] = (
        float(context.adversary_dropout) if classifier_name == "graphsage" else pd.NA
    )
    if classifier_name == "logistic_regression":
        x_train = embedding_lookup.loc[list(split.train_node_ids), vector_columns].to_numpy(dtype=float)
        x_test = embedding_lookup.loc[list(split.test_node_ids), vector_columns].to_numpy(dtype=float)
        y_train = labels.loc[list(split.train_node_ids)].to_numpy(dtype=object)
        probe_train_embeddings = x_train
        probe_test_embeddings = x_test

        classifier = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=context.random_seed)),
            ]
        )
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
    else:
        protected_series_for_training: pd.Series | None = None
        if context.protected_attribute_column is not None:
            protected_series_candidate = _attribute_lookup(dataset).reindex(node_order)[context.protected_attribute_column]
            if not protected_series_candidate.isna().any():
                protected_series_for_training = protected_series_candidate.astype(str)
        try:
            train_validation_split = build_node_train_validation_split(
                list(split.train_node_ids),
                labels.loc[list(split.train_node_ids)].to_numpy(dtype=object),
                validation_fraction=context.node_validation_fraction,
                random_seed=context.random_seed,
                use_stratified_split=context.use_stratified_split,
            )
        except ValueError as exc:
            return _skip_row(
                dataset_name=dataset.name,
                method_name=method_name,
                task="node_classification",
                reason=str(exc),
                embedding_runtime_seconds=embedding_runtime_seconds,
                embedding_dim=embedding_dim,
                node_count=node_count,
                label_column=context.classification_label_column,
                classifier=classifier_name,
            )
        split_diagnostics_payloads["train_full"] = split_diagnostics_payloads["train"]
        split_diagnostics_payloads["train"] = _split_diagnostics_for_nodes(
            labels=labels,
            node_ids=train_validation_split.train_node_ids,
            protected_series=protected_series_aligned,
            underrepresented_threshold=context.min_group_support_train,
        )
        split_diagnostics_payloads["validation"] = _split_diagnostics_for_nodes(
            labels=labels,
            node_ids=train_validation_split.validation_node_ids,
            protected_series=protected_series_aligned,
            underrepresented_threshold=context.min_group_support_train,
        )
        split_diagnostics_payloads["test"] = _split_diagnostics_for_nodes(
            labels=labels,
            node_ids=split.test_node_ids,
            protected_series=protected_series_aligned,
            underrepresented_threshold=context.min_group_support_eval,
        )

        classifier_result = train_graphsage_node_classifier(
            context.graph,
            embedding_frame,
            labels,
            split,
            train_validation_split,
            protected_labels=protected_series_for_training,
            config=GraphSAGENodeClassificationConfig(
                imbalance_mode=context.imbalance_mode,
                debias_mode=context.debias_mode,
                focal_gamma=context.focal_gamma,
                early_stop_metric=context.early_stop_metric,
                early_stop_patience=context.early_stop_patience,
                class_weight_smoothing=context.class_weight_smoothing,
                training_mode=context.training_mode,
                group_robust_weight=context.group_robust_weight,
                group_weight_mode=context.group_weight_mode,
                worst_group_boost_factor=context.worst_group_boost_factor,
                min_support_boost_factor=context.min_support_boost_factor,
                min_group_support_threshold=context.min_group_support_threshold,
                min_group_support_train=context.min_group_support_train,
                min_group_support_eval=context.min_group_support_eval,
                report_small_group_metrics=context.report_small_group_metrics,
                rebalance_batches_by_group=context.rebalance_batches_by_group,
                fairness_score_alpha=context.fairness_score_alpha,
                fairness_score_beta=context.fairness_score_beta,
                probe_model_type=context.probe_model_type,
                adversary_loss_weight=context.adversary_loss_weight,
                gradient_reversal_lambda=context.gradient_reversal_lambda,
                adversary_warmup_epochs=context.adversary_warmup_epochs,
                group_robust_warmup_epochs=context.group_robust_warmup_epochs,
                adversary_hidden_dim=context.adversary_hidden_dim,
                adversary_num_layers=context.adversary_num_layers,
                adversary_dropout=context.adversary_dropout,
                random_seed=context.random_seed,
            ),
        )
        predictions = classifier_result.predictions
        probe_embeddings = classifier_result.node_embeddings
        probe_train_embeddings = probe_embeddings[
            [node_index_lookup[node_id] for node_id in split.train_node_ids]
        ]
        probe_test_embeddings = probe_embeddings[
            [node_index_lookup[node_id] for node_id in split.test_node_ids]
        ]
        notes_parts.extend(
            [
                f"comparison_mode={row['comparison_mode']}",
                f"val={classifier_result.validation_count}",
                f"val_stratified={classifier_result.validation_stratified}",
                f"training_mode={context.training_mode}",
                f"debias_mode={context.debias_mode}",
                f"imbalance_mode={context.imbalance_mode}",
                f"early_stop_metric={context.early_stop_metric}",
                f"min_group_support_train={context.min_group_support_train}",
                f"min_group_support_eval={context.min_group_support_eval}",
            ]
        )
        if context.rebalance_batches_by_group:
            notes_parts.append("rebalance_batches_by_group=true")
        if context.report_small_group_metrics:
            notes_parts.append("report_small_group_metrics=true")
        row["best_epoch"] = int(classifier_result.best_epoch)
        row["training_history_json"] = classifier_result.training_history_json
        if context.group_weight_mode != "none" and float(context.group_robust_weight) > 0.0:
            notes_parts.extend(
                [
                    f"group_weight_mode={context.group_weight_mode}",
                    f"group_robust_weight={context.group_robust_weight}",
                    f"worst_group_boost_factor={context.worst_group_boost_factor}",
                    f"min_group_support_train={context.min_group_support_train}",
                ]
            )
            if context.group_weight_mode == "min_support_boost":
                notes_parts.append(f"min_support_boost_factor={context.min_support_boost_factor}")
            if int(context.group_robust_warmup_epochs) > 0:
                notes_parts.append(
                    f"group_robust_warmup_epochs={context.group_robust_warmup_epochs}"
                )
        if context.debias_mode == "adversarial":
            notes_parts.extend(
                [
                    f"adversary_loss_weight={context.adversary_loss_weight}",
                    f"gradient_reversal_lambda={context.gradient_reversal_lambda}",
                ]
            )
            if int(context.adversary_warmup_epochs) > 0:
                notes_parts.append(f"adversary_warmup_epochs={context.adversary_warmup_epochs}")
        if context.early_stop_metric == "fairness_score":
            notes_parts.extend(
                [
                    f"fairness_score_alpha={context.fairness_score_alpha}",
                    f"fairness_score_beta={context.fairness_score_beta}",
                    f"probe_model_type={context.probe_model_type}",
                ]
            )

    row["runtime_seconds"] = float(perf_counter() - start)
    row["evaluated_count"] = int(len(y_test))
    row["train_count"] = int(len(split.train_node_ids))
    row["test_count"] = int(len(y_test))
    row["split_diagnostics_json"] = serialize_split_diagnostics(split_diagnostics_payloads)
    test_split_payload = split_diagnostics_payloads.get("test", {})
    if isinstance(test_split_payload, dict):
        underrepresented_groups = test_split_payload.get("underrepresented_protected_groups", [])
        if isinstance(underrepresented_groups, list):
            row["small_group_count"] = int(len(underrepresented_groups))
    metric_labels = sorted({str(label) for label in y_test} | {str(label) for label in predictions})
    metrics = compute_classification_metrics(y_test, predictions, labels=metric_labels)
    row.update(metrics)
    row["per_class_metrics_json"] = serialize_per_class_metrics(y_test, predictions, labels=metric_labels)
    row["confusion_matrix_json"] = serialize_confusion_matrix_summary(y_test, predictions, labels=metric_labels)
    row["class_distribution_json"] = serialize_class_distribution_summary(y_test, predictions, labels=metric_labels)
    if context.protected_attribute_column is not None:
        if context.protected_attribute_column not in dataset.node_attributes.columns:
            row["protected_attribute_column"] = context.protected_attribute_column
            row["unavailable_reason"] = (
                f"Protected attribute column '{context.protected_attribute_column}' is not present in dataset.node_attributes."
            )
            row["protected_probe_status"] = "skipped"
            row["protected_probe_reason"] = row["unavailable_reason"]
        else:
            protected_series = _attribute_lookup(dataset).reindex(node_order)[context.protected_attribute_column]
            test_groups = protected_series.reindex(list(split.test_node_ids))
            train_groups = protected_series.reindex(list(split.train_node_ids))
            row["protected_attribute_column"] = context.protected_attribute_column
            row["protected_probe_model_type"] = context.probe_model_type
            if test_groups.isna().any():
                row["unavailable_reason"] = (
                    f"Protected attribute column '{context.protected_attribute_column}' contains missing values for the test split."
                )
                row["protected_probe_status"] = "skipped"
                row["protected_probe_reason"] = row["unavailable_reason"]
            else:
                group_summary, per_group_json, per_group_confusion_json, group_error_diagnostics_json = summarize_group_classification_metrics(
                    y_test,
                    predictions,
                    test_groups.astype(str).to_numpy(dtype=object),
                    min_group_support_threshold=context.min_group_support_eval,
                    report_small_group_metrics=context.report_small_group_metrics,
                )
                row.update(group_summary)
                row["per_group_metrics_json"] = per_group_json
                row["per_group_confusion_json"] = per_group_confusion_json
                row["group_error_diagnostics_json"] = group_error_diagnostics_json
                group_error_payload = json.loads(group_error_diagnostics_json)
                if isinstance(group_error_payload, list):
                    collapsed_groups = [
                        str(item.get("group"))
                        for item in group_error_payload
                        if isinstance(item, dict) and bool(item.get("prediction_collapse"))
                    ]
                    row["collapsed_groups"] = ",".join(collapsed_groups) if collapsed_groups else ""
                    if group_error_payload:
                        row["max_group_error"] = max(
                            float(item.get("misclassification_rate", 0.0) or 0.0)
                            for item in group_error_payload
                            if isinstance(item, dict)
                        )
                if train_groups.isna().any():
                    row["protected_probe_status"] = "skipped"
                    row["protected_probe_reason"] = (
                        f"Protected attribute column '{context.protected_attribute_column}' contains missing values for the training split."
                    )
                elif probe_train_embeddings is not None and probe_test_embeddings is not None:
                    probe_result = run_train_test_embedding_probe(
                        train_embeddings=probe_train_embeddings,
                        test_embeddings=probe_test_embeddings,
                        train_labels=train_groups.astype(str).to_numpy(dtype=object),
                        test_labels=test_groups.astype(str).to_numpy(dtype=object),
                        random_seed=context.random_seed,
                        model_type=context.probe_model_type,
                    )
                    row["protected_probe_status"] = probe_result["status"]
                    row["protected_probe_accuracy"] = probe_result["accuracy"]
                    row["protected_probe_macro_f1"] = probe_result["macro_f1"]
                    row["protected_probe_model_type"] = probe_result["model_type"]
                    row["protected_probe_reason"] = probe_result["reason"]
    row["notes"] = "; ".join(notes_parts)
    return row


def _evaluate_link_prediction(
    *,
    dataset: LoadedDataset,
    method_name: str,
    embedding_frame: pd.DataFrame,
    embedding_runtime_seconds: float,
    embedding_dim: int,
    node_count: int,
    context: _EvaluationContext,
) -> dict[str, Any]:
    if context.link_prediction_split is None:
        return _skip_row(
            dataset_name=dataset.name,
            method_name=method_name,
            task="link_prediction",
            reason=context.link_prediction_skip_reason or "Link prediction split could not be prepared.",
            embedding_runtime_seconds=embedding_runtime_seconds,
            embedding_dim=embedding_dim,
            node_count=node_count,
            classifier="logistic_regression",
            edge_feature=context.link_prediction_edge_feature,
        )

    start = perf_counter()
    row = _empty_result_row(
        dataset_name=dataset.name,
        method_name=method_name,
        task="link_prediction",
        embedding_runtime_seconds=embedding_runtime_seconds,
        embedding_dim=embedding_dim,
        node_count=node_count,
        classifier="logistic_regression",
        edge_feature=context.link_prediction_edge_feature,
    )
    embedding_lookup = _node_vectors_lookup(embedding_frame)
    split = context.link_prediction_split
    train_positive = split.train_positive_edges
    train_negative = split.train_negative_edges
    test_positive = split.test_positive_edges
    test_negative = split.test_negative_edges
    x_train = np.vstack(
        [
            _edge_feature_matrix(embedding_lookup, train_positive, feature_name=context.link_prediction_edge_feature),
            _edge_feature_matrix(embedding_lookup, train_negative, feature_name=context.link_prediction_edge_feature),
        ]
    )
    y_train = np.concatenate(
        [
            np.ones(len(train_positive), dtype=int),
            np.zeros(len(train_negative), dtype=int),
        ]
    )
    x_test = np.vstack(
        [
            _edge_feature_matrix(embedding_lookup, test_positive, feature_name=context.link_prediction_edge_feature),
            _edge_feature_matrix(embedding_lookup, test_negative, feature_name=context.link_prediction_edge_feature),
        ]
    )
    y_test = np.concatenate(
        [
            np.ones(len(test_positive), dtype=int),
            np.zeros(len(test_negative), dtype=int),
        ]
    )

    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=context.random_seed)),
        ]
    )
    classifier.fit(x_train, y_train)
    probabilities = classifier.predict_proba(x_test)[:, 1]

    row["runtime_seconds"] = float(perf_counter() - start)
    row["evaluated_count"] = int(len(y_test))
    row["train_count"] = int(len(y_train))
    row["test_count"] = int(len(y_test))
    row["roc_auc"] = float(roc_auc_score(y_test, probabilities))
    row["average_precision"] = float(average_precision_score(y_test, probabilities))
    row["notes"] = (
        f"train_pos={len(train_positive)}; train_neg={len(train_negative)}; "
        f"test_pos={len(test_positive)}; test_neg={len(test_negative)}"
    )
    return row


def _evaluate_node_clustering(
    *,
    dataset: LoadedDataset,
    method_name: str,
    embedding_frame: pd.DataFrame,
    embedding_runtime_seconds: float,
    embedding_dim: int,
    node_count: int,
    context: _EvaluationContext,
) -> dict[str, Any]:
    classifier_name = context.clustering_method
    if context.node_clustering_labels is None:
        return _skip_row(
            dataset_name=dataset.name,
            method_name=method_name,
            task="node_clustering",
            reason=context.node_clustering_skip_reason or "Node clustering labels could not be prepared.",
            embedding_runtime_seconds=embedding_runtime_seconds,
            embedding_dim=embedding_dim,
            node_count=node_count,
            label_column=context.clustering_label_column,
            classifier=classifier_name,
        )

    start = perf_counter()
    row = _empty_result_row(
        dataset_name=dataset.name,
        method_name=method_name,
        task="node_clustering",
        embedding_runtime_seconds=embedding_runtime_seconds,
        embedding_dim=embedding_dim,
        node_count=node_count,
        label_column=context.clustering_label_column,
        classifier=classifier_name,
    )
    labels = context.node_clustering_labels
    cluster_count = int(labels.nunique())
    if cluster_count < 2 or len(labels) < cluster_count:
        return _skip_row(
            dataset_name=dataset.name,
            method_name=method_name,
            task="node_clustering",
            reason="Node clustering requires at least two label classes and at least one node per class.",
            embedding_runtime_seconds=embedding_runtime_seconds,
            embedding_dim=embedding_dim,
            node_count=node_count,
            label_column=context.clustering_label_column,
            classifier=classifier_name,
        )

    clustering_config = dict(context.clustering_method_config)
    if context.clustering_n_clusters is not None:
        clustering_config.setdefault("n_clusters", int(context.clustering_n_clusters))
    else:
        clustering_config.setdefault("n_clusters", cluster_count)
    if context.clustering_min_cluster_size is not None:
        clustering_config.setdefault("min_cluster_size", int(context.clustering_min_cluster_size))
    try:
        clustering_result = cluster_nodes(
            context.graph,
            context.clustering_method,
            embeddings=embedding_frame,
            labels=labels,
            config=clustering_config,
            random_seed=context.random_seed,
            input_mode=context.clustering_input_mode,
            dataset=dataset,
        )
    except ClusteringFrameworkError as exc:
        return _skip_row(
            dataset_name=dataset.name,
            method_name=method_name,
            task="node_clustering",
            reason=str(exc),
            embedding_runtime_seconds=embedding_runtime_seconds,
            embedding_dim=embedding_dim,
            node_count=node_count,
            label_column=context.clustering_label_column,
            classifier=classifier_name,
        )

    row["runtime_seconds"] = float(perf_counter() - start)
    row["evaluated_count"] = int(len(labels))
    row["train_count"] = int(len(labels))
    row["test_count"] = pd.NA
    row["clustering_method"] = context.clustering_method
    row["clustering_category"] = clustering_result.category
    row["clustering_requested_input_mode"] = clustering_result.requested_input_mode
    row["clustering_input_mode"] = clustering_result.resolved_input_mode
    row["num_clusters_found"] = int(clustering_result.num_clusters)
    row["cluster_size_summary"] = str(clustering_result.metadata.get("cluster_size_summary", ""))
    row["silhouette_score"] = (
        pd.NA if clustering_result.silhouette_score is None else float(clustering_result.silhouette_score)
    )
    row["davies_bouldin_score"] = (
        pd.NA if clustering_result.davies_bouldin_score is None else float(clustering_result.davies_bouldin_score)
    )
    row["calinski_harabasz_score"] = (
        pd.NA
        if clustering_result.calinski_harabasz_score is None
        else float(clustering_result.calinski_harabasz_score)
    )
    row["nmi"] = pd.NA if clustering_result.nmi is None else float(clustering_result.nmi)
    row["ari"] = pd.NA if clustering_result.ari is None else float(clustering_result.ari)
    row["notes"] = (
        f"clusters={int(clustering_result.num_clusters)}; "
        f"category={clustering_result.category}; "
        f"input_mode={clustering_result.resolved_input_mode}"
    )
    return row


def _ordered_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=EVALUATION_RESULT_COLUMNS)
    frame = pd.DataFrame(rows)
    for column in EVALUATION_RESULT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame.loc[:, EVALUATION_RESULT_COLUMNS].copy()
    return frame.sort_values(by=["method", "task"], ascending=[True, True]).reset_index(drop=True)


def _evaluate_with_context(
    *,
    dataset: LoadedDataset,
    embedding_result_or_frame: EmbeddingResult | pd.DataFrame,
    context: _EvaluationContext,
    method_name: str | None = None,
    embedding_runtime_seconds: float | None = None,
) -> pd.DataFrame:
    if not context.tasks:
        return _ordered_frame([])

    resolved_method_name, embedding_frame, resolved_runtime, resolved_dim, resolved_node_count = _coerce_embedding_result(
        embedding_result_or_frame,
        graph=context.graph,
        method_name=method_name,
        embedding_runtime_seconds=embedding_runtime_seconds,
    )

    rows: list[dict[str, Any]] = []
    for task in context.tasks:
        if task == "node_classification":
            rows.append(
                _evaluate_node_classification(
                    dataset=dataset,
                    method_name=resolved_method_name,
                    embedding_frame=embedding_frame,
                    embedding_runtime_seconds=resolved_runtime,
                    embedding_dim=resolved_dim,
                    node_count=resolved_node_count,
                    context=context,
                )
            )
        elif task == "link_prediction":
            rows.append(
                _evaluate_link_prediction(
                    dataset=dataset,
                    method_name=resolved_method_name,
                    embedding_frame=embedding_frame,
                    embedding_runtime_seconds=resolved_runtime,
                    embedding_dim=resolved_dim,
                    node_count=resolved_node_count,
                    context=context,
                )
            )
        elif task == "node_clustering":
            rows.append(
                _evaluate_node_clustering(
                    dataset=dataset,
                    method_name=resolved_method_name,
                    embedding_frame=embedding_frame,
                    embedding_runtime_seconds=resolved_runtime,
                    embedding_dim=resolved_dim,
                    node_count=resolved_node_count,
                    context=context,
                )
            )
    return _ordered_frame(rows)


def evaluate_embedding_result(
    dataset: LoadedDataset,
    embedding_result_or_frame: EmbeddingResult | pd.DataFrame,
    *,
    tasks: Sequence[str] | None = None,
    graph: nx.Graph | None = None,
    symmetrize_directed: bool = True,
    method_name: str | None = None,
    embedding_runtime_seconds: float | None = None,
    random_seed: int = 42,
    label_column: str | None = None,
    classification_label_column: str | None = None,
    clustering_label_column: str | None = None,
    node_test_fraction: float = 0.25,
    link_test_fraction: float = 0.25,
    link_negative_ratio: float = 1.0,
    link_prediction_edge_feature: str = "hadamard",
    node_classification_model: str = "logistic_regression",
    clustering_method: str = "kmeans",
    clustering_input_mode: str = "embedding",
    clustering_n_clusters: int | None = None,
    clustering_min_cluster_size: int | None = None,
    clustering_method_config: dict[str, Any] | None = None,
    training_mode: str | None = None,
    imbalance_mode: str = "none",
    debias_mode: str = "none",
    focal_gamma: float = 2.0,
    use_stratified_split: bool = True,
    early_stop_metric: str = "accuracy",
    early_stop_patience: int = 0,
    class_weight_smoothing: float = 0.0,
    node_validation_fraction: float = 0.2,
    group_robust_weight: float = 0.0,
    group_weight_mode: str = "none",
    worst_group_boost_factor: float = 2.0,
    min_support_boost_factor: float = 2.0,
    min_group_support_threshold: int | None = None,
    min_group_support_train: int | None = None,
    min_group_support_eval: int | None = None,
    report_small_group_metrics: bool = False,
    rebalance_batches_by_group: bool = False,
    fairness_score_alpha: float = 0.25,
    fairness_score_beta: float = 0.25,
    probe_model_type: str = "linear",
    adversary_loss_weight: float = 1.0,
    gradient_reversal_lambda: float = 1.0,
    adversary_warmup_epochs: int = 0,
    group_robust_warmup_epochs: int = 0,
    adversary_hidden_dim: int = 64,
    adversary_num_layers: int = 1,
    adversary_dropout: float = 0.2,
    protected_attribute_column: str | None = None,
) -> pd.DataFrame:
    """Evaluate one embedding result across requested downstream tasks."""

    context = _build_context(
        dataset,
        tasks=tasks,
        graph=graph,
        symmetrize_directed=symmetrize_directed,
        random_seed=random_seed,
        label_column=label_column,
        classification_label_column=classification_label_column,
        clustering_label_column=clustering_label_column,
        node_test_fraction=node_test_fraction,
        link_test_fraction=link_test_fraction,
        link_negative_ratio=link_negative_ratio,
        link_prediction_edge_feature=link_prediction_edge_feature,
        node_classification_model=node_classification_model,
        clustering_method=clustering_method,
        clustering_input_mode=clustering_input_mode,
        clustering_n_clusters=clustering_n_clusters,
        clustering_min_cluster_size=clustering_min_cluster_size,
        clustering_method_config=clustering_method_config,
        training_mode=training_mode,
        imbalance_mode=imbalance_mode,
        debias_mode=debias_mode,
        focal_gamma=focal_gamma,
        use_stratified_split=use_stratified_split,
        early_stop_metric=early_stop_metric,
        early_stop_patience=early_stop_patience,
        class_weight_smoothing=class_weight_smoothing,
        node_validation_fraction=node_validation_fraction,
        group_robust_weight=group_robust_weight,
        group_weight_mode=group_weight_mode,
        worst_group_boost_factor=worst_group_boost_factor,
        min_support_boost_factor=min_support_boost_factor,
        min_group_support_threshold=min_group_support_threshold,
        min_group_support_train=min_group_support_train,
        min_group_support_eval=min_group_support_eval,
        report_small_group_metrics=report_small_group_metrics,
        rebalance_batches_by_group=rebalance_batches_by_group,
        fairness_score_alpha=fairness_score_alpha,
        fairness_score_beta=fairness_score_beta,
        probe_model_type=probe_model_type,
        adversary_loss_weight=adversary_loss_weight,
        gradient_reversal_lambda=gradient_reversal_lambda,
        adversary_warmup_epochs=adversary_warmup_epochs,
        group_robust_warmup_epochs=group_robust_warmup_epochs,
        adversary_hidden_dim=adversary_hidden_dim,
        adversary_num_layers=adversary_num_layers,
        adversary_dropout=adversary_dropout,
        protected_attribute_column=protected_attribute_column,
    )
    return _evaluate_with_context(
        dataset=dataset,
        embedding_result_or_frame=embedding_result_or_frame,
        context=context,
        method_name=method_name,
        embedding_runtime_seconds=embedding_runtime_seconds,
    )


def evaluate_embedding_benchmark(
    dataset: LoadedDataset,
    benchmark_result: BenchmarkRunResult,
    *,
    tasks: Sequence[str] | None = None,
    graph: nx.Graph | None = None,
    symmetrize_directed: bool = True,
    random_seed: int = 42,
    label_column: str | None = None,
    classification_label_column: str | None = None,
    clustering_label_column: str | None = None,
    node_test_fraction: float = 0.25,
    link_test_fraction: float = 0.25,
    link_negative_ratio: float = 1.0,
    link_prediction_edge_feature: str = "hadamard",
    node_classification_model: str = "logistic_regression",
    clustering_method: str = "kmeans",
    clustering_input_mode: str = "embedding",
    clustering_n_clusters: int | None = None,
    clustering_min_cluster_size: int | None = None,
    clustering_method_config: dict[str, Any] | None = None,
    training_mode: str | None = None,
    imbalance_mode: str = "none",
    debias_mode: str = "none",
    focal_gamma: float = 2.0,
    use_stratified_split: bool = True,
    early_stop_metric: str = "accuracy",
    early_stop_patience: int = 0,
    class_weight_smoothing: float = 0.0,
    node_validation_fraction: float = 0.2,
    group_robust_weight: float = 0.0,
    group_weight_mode: str = "none",
    worst_group_boost_factor: float = 2.0,
    min_support_boost_factor: float = 2.0,
    min_group_support_threshold: int | None = None,
    min_group_support_train: int | None = None,
    min_group_support_eval: int | None = None,
    report_small_group_metrics: bool = False,
    rebalance_batches_by_group: bool = False,
    fairness_score_alpha: float = 0.25,
    fairness_score_beta: float = 0.25,
    probe_model_type: str = "linear",
    adversary_loss_weight: float = 1.0,
    gradient_reversal_lambda: float = 1.0,
    adversary_warmup_epochs: int = 0,
    group_robust_warmup_epochs: int = 0,
    adversary_hidden_dim: int = 64,
    adversary_num_layers: int = 1,
    adversary_dropout: float = 0.2,
    protected_attribute_column: str | None = None,
) -> pd.DataFrame:
    """Evaluate every method in a benchmark run with shared splits."""

    context = _build_context(
        dataset,
        tasks=tasks,
        graph=graph,
        symmetrize_directed=symmetrize_directed,
        random_seed=random_seed,
        label_column=label_column,
        classification_label_column=classification_label_column,
        clustering_label_column=clustering_label_column,
        node_test_fraction=node_test_fraction,
        link_test_fraction=link_test_fraction,
        link_negative_ratio=link_negative_ratio,
        link_prediction_edge_feature=link_prediction_edge_feature,
        node_classification_model=node_classification_model,
        clustering_method=clustering_method,
        clustering_input_mode=clustering_input_mode,
        clustering_n_clusters=clustering_n_clusters,
        clustering_min_cluster_size=clustering_min_cluster_size,
        clustering_method_config=clustering_method_config,
        training_mode=training_mode,
        imbalance_mode=imbalance_mode,
        debias_mode=debias_mode,
        focal_gamma=focal_gamma,
        use_stratified_split=use_stratified_split,
        early_stop_metric=early_stop_metric,
        early_stop_patience=early_stop_patience,
        class_weight_smoothing=class_weight_smoothing,
        node_validation_fraction=node_validation_fraction,
        group_robust_weight=group_robust_weight,
        group_weight_mode=group_weight_mode,
        worst_group_boost_factor=worst_group_boost_factor,
        min_support_boost_factor=min_support_boost_factor,
        min_group_support_threshold=min_group_support_threshold,
        min_group_support_train=min_group_support_train,
        min_group_support_eval=min_group_support_eval,
        report_small_group_metrics=report_small_group_metrics,
        rebalance_batches_by_group=rebalance_batches_by_group,
        fairness_score_alpha=fairness_score_alpha,
        fairness_score_beta=fairness_score_beta,
        probe_model_type=probe_model_type,
        adversary_loss_weight=adversary_loss_weight,
        gradient_reversal_lambda=gradient_reversal_lambda,
        adversary_warmup_epochs=adversary_warmup_epochs,
        group_robust_warmup_epochs=group_robust_warmup_epochs,
        adversary_hidden_dim=adversary_hidden_dim,
        adversary_num_layers=adversary_num_layers,
        adversary_dropout=adversary_dropout,
        protected_attribute_column=protected_attribute_column,
    )
    if not context.tasks:
        return _ordered_frame([])

    rows: list[dict[str, Any]] = []
    summary_frame = benchmark_result.summary_frame.copy()
    for summary_row in summary_frame.to_dict(orient="records"):
        method_name = str(summary_row["method"])
        status = str(summary_row.get("status", "")).strip().lower()
        if status != "ok" or method_name not in benchmark_result.results_by_method:
            reason = str(summary_row.get("skip_reason", "") or summary_row.get("error_message", "")).strip()
            if not reason:
                reason = f"Embedding generation status is '{status or 'unknown'}'."
            for task in context.tasks:
                rows.append(
                    _skip_row(
                        dataset_name=dataset.name,
                        method_name=method_name,
                        task=task,
                        reason=reason,
                        embedding_runtime_seconds=float(summary_row.get("runtime_seconds", 0.0) or 0.0),
                        embedding_dim=0 if pd.isna(summary_row.get("embedding_dim", pd.NA)) else int(summary_row["embedding_dim"]),
                        node_count=int(summary_row.get("node_count", dataset.graph.number_of_nodes())),
                        label_column=(
                            context.classification_label_column
                            if task == "node_classification"
                            else context.clustering_label_column if task == "node_clustering" else None
                        ),
                        classifier=(
                            context.clustering_method
                            if task == "node_clustering"
                            else context.node_classification_model if task == "node_classification" else "logistic_regression"
                        ),
                        edge_feature=context.link_prediction_edge_feature if task == "link_prediction" else None,
                    )
                )
            continue

        rows.extend(
            _evaluate_with_context(
                dataset=dataset,
                embedding_result_or_frame=benchmark_result.results_by_method[method_name],
                context=context,
            ).to_dict(orient="records")
        )

    return _ordered_frame(rows)


def _normalize_evaluation_random_seeds(
    random_seeds: Sequence[int] | None,
    *,
    fallback_seed: int,
) -> tuple[int, ...]:
    if random_seeds is None:
        return (int(fallback_seed),)
    normalized = tuple(dict.fromkeys(int(seed) for seed in random_seeds))
    if not normalized:
        raise ValueError("evaluation_random_seeds must contain at least one seed when provided.")
    return normalized


def _aggregate_repeated_evaluation_frame(per_run_frame: pd.DataFrame) -> pd.DataFrame:
    if per_run_frame.empty:
        return pd.DataFrame()

    group_columns = [
        column
        for column in [
            "dataset",
            "method",
            "task",
            "label_column",
            "classifier",
            "comparison_mode",
            "training_mode",
            "imbalance_mode",
            "early_stop_metric",
            "debias_mode",
            "group_robust_weight",
            "group_weight_mode",
            "worst_group_boost_factor",
            "min_support_boost_factor",
            "min_group_support_threshold",
            "min_group_support_train",
            "min_group_support_eval",
            "report_small_group_metrics",
            "rebalance_batches_by_group",
            "fairness_score_alpha",
            "fairness_score_beta",
            "adversary_loss_weight",
            "gradient_reversal_lambda",
            "adversary_warmup_epochs",
            "group_robust_warmup_epochs",
            "adversary_hidden_dim",
            "adversary_num_layers",
            "adversary_dropout",
            "protected_attribute_column",
            "protected_probe_model_type",
        ]
        if column in per_run_frame.columns
    ]
    metric_columns = {
        "macro_f1": "macro_f1",
        "worst_group_f1_raw": "worst_group_f1_raw",
        "worst_group_f1_supported": "worst_group_f1_supported",
        "macro_f1_gap_raw": "macro_f1_gap_raw",
        "macro_f1_gap_supported": "macro_f1_gap_supported",
        "protected_probe_macro_f1": "protected_probe_macro_f1",
    }
    aggregate_rows: list[dict[str, Any]] = []
    grouped = per_run_frame.groupby(group_columns, dropna=False, sort=True) if group_columns else [((), per_run_frame)]
    for _, group_frame in grouped:
        row = {column: group_frame.iloc[0][column] for column in group_columns}
        status_values = [
            str(value).strip()
            for value in group_frame.get("status", pd.Series(dtype=object)).tolist()
            if str(value).strip()
        ]
        unique_statuses = sorted(set(status_values))
        row["status"] = unique_statuses[0] if len(unique_statuses) == 1 else "mixed"
        row["run_count"] = int(len(group_frame))
        row["evaluation_seed_count"] = int(
            pd.to_numeric(group_frame.get("evaluation_seed_count", pd.Series(dtype=float)), errors="coerce")
            .dropna()
            .max()
            if "evaluation_seed_count" in group_frame
            else 0
        )
        row["repeated_split_count"] = int(
            pd.to_numeric(group_frame.get("repeated_split_count", pd.Series(dtype=float)), errors="coerce")
            .dropna()
            .max()
            if "repeated_split_count" in group_frame
            else 0
        )
        for source_column, metric_name in metric_columns.items():
            metric_values = pd.to_numeric(group_frame.get(source_column, pd.Series(dtype=float)), errors="coerce").dropna()
            row[f"mean_{metric_name}"] = float(metric_values.mean()) if not metric_values.empty else pd.NA
            row[f"std_{metric_name}"] = float(metric_values.std(ddof=0)) if not metric_values.empty else pd.NA
        collapsed_groups: set[str] = set()
        for value in group_frame.get("collapsed_groups", pd.Series(dtype=object)).tolist():
            if value is None or pd.isna(value):
                continue
            for group_name in str(value).split(","):
                normalized = group_name.strip()
                if normalized:
                    collapsed_groups.add(normalized)
        row["collapsed_groups_notes"] = ",".join(sorted(collapsed_groups)) if collapsed_groups else ""
        row["collapsed_group_run_count"] = int(
            sum(
                1
                for value in group_frame.get("collapsed_groups", pd.Series(dtype=object)).tolist()
                if value is not None and not pd.isna(value) and str(value).strip()
            )
        )
        aggregate_rows.append(row)

    aggregate_frame = pd.DataFrame(aggregate_rows)
    if aggregate_frame.empty:
        return aggregate_frame
    ordered_columns = [
        *group_columns,
        "status",
        "run_count",
        "evaluation_seed_count",
        "repeated_split_count",
        "mean_macro_f1",
        "std_macro_f1",
        "mean_worst_group_f1_raw",
        "std_worst_group_f1_raw",
        "mean_worst_group_f1_supported",
        "std_worst_group_f1_supported",
        "mean_macro_f1_gap_raw",
        "std_macro_f1_gap_raw",
        "mean_macro_f1_gap_supported",
        "std_macro_f1_gap_supported",
        "mean_protected_probe_macro_f1",
        "std_protected_probe_macro_f1",
        "collapsed_groups_notes",
        "collapsed_group_run_count",
    ]
    return aggregate_frame.loc[:, [column for column in ordered_columns if column in aggregate_frame.columns]].copy()


def evaluate_embedding_benchmark_repeated(
    dataset: LoadedDataset,
    benchmark_result: BenchmarkRunResult,
    *,
    evaluation_random_seeds: Sequence[int] | None = None,
    repeated_split_count: int = 1,
    tasks: Sequence[str] | None = None,
    graph: nx.Graph | None = None,
    symmetrize_directed: bool = True,
    random_seed: int = 42,
    label_column: str | None = None,
    classification_label_column: str | None = None,
    clustering_label_column: str | None = None,
    node_test_fraction: float = 0.25,
    link_test_fraction: float = 0.25,
    link_negative_ratio: float = 1.0,
    link_prediction_edge_feature: str = "hadamard",
    node_classification_model: str = "logistic_regression",
    clustering_method: str = "kmeans",
    clustering_input_mode: str = "embedding",
    clustering_n_clusters: int | None = None,
    clustering_min_cluster_size: int | None = None,
    clustering_method_config: dict[str, Any] | None = None,
    training_mode: str | None = None,
    imbalance_mode: str = "none",
    debias_mode: str = "none",
    focal_gamma: float = 2.0,
    use_stratified_split: bool = True,
    early_stop_metric: str = "accuracy",
    early_stop_patience: int = 0,
    class_weight_smoothing: float = 0.0,
    node_validation_fraction: float = 0.2,
    group_robust_weight: float = 0.0,
    group_weight_mode: str = "none",
    worst_group_boost_factor: float = 2.0,
    min_support_boost_factor: float = 2.0,
    min_group_support_threshold: int | None = None,
    min_group_support_train: int | None = None,
    min_group_support_eval: int | None = None,
    report_small_group_metrics: bool = False,
    rebalance_batches_by_group: bool = False,
    fairness_score_alpha: float = 0.25,
    fairness_score_beta: float = 0.25,
    probe_model_type: str = "linear",
    adversary_loss_weight: float = 1.0,
    gradient_reversal_lambda: float = 1.0,
    adversary_warmup_epochs: int = 0,
    group_robust_warmup_epochs: int = 0,
    adversary_hidden_dim: int = 64,
    adversary_num_layers: int = 1,
    adversary_dropout: float = 0.2,
    protected_attribute_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run deterministic repeated seeded/split evaluation and return per-run plus aggregate frames."""

    normalized_random_seeds = _normalize_evaluation_random_seeds(
        evaluation_random_seeds,
        fallback_seed=random_seed,
    )
    if int(repeated_split_count) < 1:
        raise ValueError("repeated_split_count must be at least 1.")

    per_run_frames: list[pd.DataFrame] = []
    for base_seed in normalized_random_seeds:
        for repeat_index in range(int(repeated_split_count)):
            effective_seed = int(base_seed) + (int(repeat_index) * 1000)
            frame = evaluate_embedding_benchmark(
                dataset,
                benchmark_result,
                tasks=tasks,
                graph=graph,
                symmetrize_directed=symmetrize_directed,
                random_seed=effective_seed,
                label_column=label_column,
                classification_label_column=classification_label_column,
                clustering_label_column=clustering_label_column,
                node_test_fraction=node_test_fraction,
                link_test_fraction=link_test_fraction,
                link_negative_ratio=link_negative_ratio,
                link_prediction_edge_feature=link_prediction_edge_feature,
                node_classification_model=node_classification_model,
                clustering_method=clustering_method,
                clustering_input_mode=clustering_input_mode,
                clustering_n_clusters=clustering_n_clusters,
                clustering_min_cluster_size=clustering_min_cluster_size,
                clustering_method_config=clustering_method_config,
                training_mode=training_mode,
                imbalance_mode=imbalance_mode,
                debias_mode=debias_mode,
                focal_gamma=focal_gamma,
                use_stratified_split=use_stratified_split,
                early_stop_metric=early_stop_metric,
                early_stop_patience=early_stop_patience,
                class_weight_smoothing=class_weight_smoothing,
                node_validation_fraction=node_validation_fraction,
                group_robust_weight=group_robust_weight,
                group_weight_mode=group_weight_mode,
                worst_group_boost_factor=worst_group_boost_factor,
                min_support_boost_factor=min_support_boost_factor,
                min_group_support_threshold=min_group_support_threshold,
                min_group_support_train=min_group_support_train,
                min_group_support_eval=min_group_support_eval,
                report_small_group_metrics=report_small_group_metrics,
                rebalance_batches_by_group=rebalance_batches_by_group,
                fairness_score_alpha=fairness_score_alpha,
                fairness_score_beta=fairness_score_beta,
                probe_model_type=probe_model_type,
                adversary_loss_weight=adversary_loss_weight,
                gradient_reversal_lambda=gradient_reversal_lambda,
                adversary_warmup_epochs=adversary_warmup_epochs,
                group_robust_warmup_epochs=group_robust_warmup_epochs,
                adversary_hidden_dim=adversary_hidden_dim,
                adversary_num_layers=adversary_num_layers,
                adversary_dropout=adversary_dropout,
                protected_attribute_column=protected_attribute_column,
            )
            if frame.empty:
                continue
            frame = frame.copy()
            frame["evaluation_base_seed"] = int(base_seed)
            frame["evaluation_repeat_index"] = int(repeat_index)
            frame["effective_random_seed"] = int(effective_seed)
            frame["evaluation_seed_count"] = int(len(normalized_random_seeds))
            frame["repeated_split_count"] = int(repeated_split_count)
            per_run_frames.append(frame)

    if not per_run_frames:
        empty_frame = _ordered_frame([])
        return empty_frame, pd.DataFrame()

    per_run_frame = pd.concat(per_run_frames, ignore_index=True)
    per_run_frame = per_run_frame.loc[:, [column for column in EVALUATION_RESULT_COLUMNS if column in per_run_frame.columns]].copy()
    aggregate_frame = _aggregate_repeated_evaluation_frame(per_run_frame)
    return per_run_frame, aggregate_frame


def evaluation_result_columns() -> list[str]:
    """Return the normalized evaluation result columns in order."""

    return list(EVALUATION_RESULT_COLUMNS)
