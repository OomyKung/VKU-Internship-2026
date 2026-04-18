"""Shared downstream evaluation helpers for graph embeddings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    average_precision_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fim_hybrid.data_loader import LoadedDataset

from .base import EmbeddingResult, embedding_columns, sorted_node_ids, validate_embedding_frame
from .evaluation_splits import (
    LinkPredictionSplit,
    NodeClassificationSplit,
    build_link_prediction_split,
    build_node_classification_split,
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
    "accuracy",
    "macro_f1",
    "micro_f1",
    "roc_auc",
    "average_precision",
    "nmi",
    "ari",
    "notes",
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
        "accuracy": pd.NA,
        "macro_f1": pd.NA,
        "micro_f1": pd.NA,
        "roc_auc": pd.NA,
        "average_precision": pd.NA,
        "nmi": pd.NA,
        "ari": pd.NA,
        "notes": "",
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

    node_classification_labels: pd.Series | None = None
    node_classification_split: NodeClassificationSplit | None = None
    node_classification_skip_reason = ""
    if "node_classification" in normalized_tasks:
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
            classifier="logistic_regression",
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
        classifier="logistic_regression",
    )
    embedding_lookup = _node_vectors_lookup(embedding_frame)
    vector_columns = embedding_columns(embedding_frame)
    split = context.node_classification_split
    labels = context.node_classification_labels
    x_train = embedding_lookup.loc[list(split.train_node_ids), vector_columns].to_numpy(dtype=float)
    x_test = embedding_lookup.loc[list(split.test_node_ids), vector_columns].to_numpy(dtype=float)
    y_train = labels.loc[list(split.train_node_ids)].to_numpy(dtype=object)
    y_test = labels.loc[list(split.test_node_ids)].to_numpy(dtype=object)

    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=context.random_seed)),
        ]
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    row["runtime_seconds"] = float(perf_counter() - start)
    row["evaluated_count"] = int(len(y_test))
    row["train_count"] = int(len(y_train))
    row["test_count"] = int(len(y_test))
    row["accuracy"] = float(accuracy_score(y_test, predictions))
    row["macro_f1"] = float(f1_score(y_test, predictions, average="macro"))
    row["micro_f1"] = float(f1_score(y_test, predictions, average="micro"))
    row["notes"] = f"classes={int(labels.nunique())}; stratified={split.stratified}"
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
            classifier="kmeans",
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
        classifier="kmeans",
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
            classifier="kmeans",
        )

    vectors = embedding_frame[embedding_columns(embedding_frame)].to_numpy(dtype=float)
    clustering = KMeans(n_clusters=cluster_count, n_init=10, random_state=context.random_seed)
    predicted_clusters = clustering.fit_predict(vectors)

    row["runtime_seconds"] = float(perf_counter() - start)
    row["evaluated_count"] = int(len(labels))
    row["train_count"] = int(len(labels))
    row["test_count"] = pd.NA
    row["nmi"] = float(normalized_mutual_info_score(labels.to_numpy(dtype=object), predicted_clusters))
    row["ari"] = float(adjusted_rand_score(labels.to_numpy(dtype=object), predicted_clusters))
    row["notes"] = f"clusters={cluster_count}"
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
                        classifier="kmeans" if task == "node_clustering" else "logistic_regression",
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


def evaluation_result_columns() -> list[str]:
    """Return the normalized evaluation result columns in order."""

    return list(EVALUATION_RESULT_COLUMNS)
