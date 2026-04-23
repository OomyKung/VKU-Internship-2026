"""Benchmark runner and export helpers for graph embeddings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from fim_hybrid.data_loader import LoadedDataset

from .base import EmbeddingResult, embedding_columns, sorted_node_ids
from .evaluation import evaluate_embedding_benchmark, resolve_evaluation_tasks
from .features import PreparedFeatures, coerce_input_features, prepare_benchmark_features
from .node_classification_models import resolve_probe_model_type, run_train_test_embedding_probe
from .registry import (
    create_embedding_model,
    get_method_spec,
    method_requires_features,
    missing_dependency_reason,
    resolve_method_names,
    unsupported_graph_reason,
)


@dataclass(slots=True)
class BenchmarkRunResult:
    """Benchmark summary plus any successful embedding results."""

    dataset_name: str
    summary_frame: pd.DataFrame
    results_by_method: dict[str, EmbeddingResult]
    output_dir: Path | None
    evaluation_frame: pd.DataFrame = field(default_factory=pd.DataFrame)


def _benchmark_graph(graph: nx.Graph, *, symmetrize_directed: bool) -> tuple[nx.Graph, bool]:
    if graph.is_directed() and symmetrize_directed:
        return graph.to_undirected(), True
    return graph.copy(), False


def _normalize_export_formats(export_formats: Sequence[str] | None) -> tuple[str, ...]:
    if export_formats is None:
        return ()
    normalized = tuple(dict.fromkeys(str(value).strip().lower() for value in export_formats))
    valid = {"csv", "pickle", "npy"}
    invalid = sorted(set(normalized) - valid)
    if invalid:
        raise ValueError(f"Unsupported export formats: {invalid}. Supported values: {sorted(valid)}.")
    return normalized


def _embedding_output_base(output_dir: Path, dataset_name: str, method_name: str) -> Path:
    benchmark_dir = output_dir / dataset_name / "embeddings"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    return benchmark_dir / f"{dataset_name}_{method_name}_embeddings"


def _evaluation_summary_path(output_dir: Path, dataset_name: str) -> Path:
    benchmark_dir = output_dir / dataset_name / "embeddings"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    return benchmark_dir / f"{dataset_name}_embedding_evaluation_summary.csv"


def _json_safe_node_id(node_id: Any) -> Any:
    if isinstance(node_id, (str, int, float, bool)) or node_id is None:
        return node_id
    return {"type": type(node_id).__name__, "repr": repr(node_id)}


def _save_embedding_outputs(
    result: EmbeddingResult,
    *,
    dataset_name: str,
    output_dir: Path,
    export_formats: Sequence[str],
) -> dict[str, str]:
    base_path = _embedding_output_base(output_dir, dataset_name, result.method_name)
    saved_paths: dict[str, str] = {}
    if "csv" in export_formats:
        csv_path = base_path.with_suffix(".csv")
        result.embedding_frame.to_csv(csv_path, index=False)
        saved_paths["csv"] = str(csv_path)
    if "pickle" in export_formats:
        pickle_path = base_path.with_suffix(".pkl")
        result.embedding_frame.to_pickle(pickle_path)
        saved_paths["pickle"] = str(pickle_path)
    if "npy" in export_formats:
        npy_path = base_path.with_suffix(".npy")
        node_path = base_path.with_name(f"{base_path.name}_node_order.json")
        np.save(npy_path, result.embedding_frame[result.vector_columns].to_numpy(dtype=float))
        node_path.write_text(
            json.dumps([_json_safe_node_id(node_id) for node_id in result.embedding_frame["node_id"].tolist()]),
            encoding="utf-8",
        )
        saved_paths["npy"] = str(npy_path)
        saved_paths["npy_node_order"] = str(node_path)
    return saved_paths


def _pairwise_cosine_mean(frame: pd.DataFrame) -> float:
    vectors = frame[embedding_columns(frame)].to_numpy(dtype=float)
    if len(vectors) < 2:
        return 0.0
    sample = vectors[: min(128, len(vectors))]
    similarity = cosine_similarity(sample)
    upper = similarity[np.triu_indices_from(similarity, k=1)]
    if upper.size == 0:
        return 0.0
    return float(np.mean(upper))


def _embedding_probe(
    embedding_frame: pd.DataFrame,
    dataset: LoadedDataset,
    target_column: str | None,
    *,
    random_seed: int,
    requested: bool,
    model_type: str,
) -> dict[str, Any]:
    resolved_model_type = resolve_probe_model_type(model_type)
    if not requested:
        return {
            "status": "not_requested",
            "column": pd.NA,
            "accuracy": pd.NA,
            "macro_f1": pd.NA,
            "model_type": pd.NA,
            "reason": "",
        }
    if target_column is None:
        return {
            "status": "skipped",
            "column": pd.NA,
            "accuracy": pd.NA,
            "macro_f1": pd.NA,
            "model_type": resolved_model_type,
            "reason": "Probe requires a target column.",
        }
    if target_column not in dataset.node_attributes.columns:
        return {
            "status": "skipped",
            "column": target_column,
            "accuracy": pd.NA,
            "macro_f1": pd.NA,
            "model_type": resolved_model_type,
            "reason": f"Target column '{target_column}' is not present in dataset.node_attributes.",
        }

    label_series = (
        dataset.node_attributes.set_index("node_id", drop=False)
        .reindex(embedding_frame["node_id"].tolist())[target_column]
    )
    if label_series.isna().any():
        return {
            "status": "skipped",
            "column": target_column,
            "accuracy": pd.NA,
            "macro_f1": pd.NA,
            "model_type": resolved_model_type,
            "reason": f"Target column '{target_column}' contains missing values.",
        }
    if label_series.nunique() < 2:
        return {
            "status": "skipped",
            "column": target_column,
            "accuracy": pd.NA,
            "macro_f1": pd.NA,
            "model_type": resolved_model_type,
            "reason": f"Target column '{target_column}' has fewer than two classes.",
        }

    vectors = embedding_frame[embedding_columns(embedding_frame)].to_numpy(dtype=float)
    labels = label_series.astype(str).to_numpy()
    if len(labels) < 4:
        return {
            "status": "skipped",
            "column": target_column,
            "accuracy": pd.NA,
            "macro_f1": pd.NA,
            "model_type": resolved_model_type,
            "reason": "Probe requires at least four labeled nodes.",
        }

    unique_counts = label_series.value_counts()
    can_stratify = bool((unique_counts >= 2).all())
    test_size = max(1, int(round(len(labels) * 0.25)))
    if can_stratify:
        test_size = max(test_size, int(label_series.nunique()))
    test_size = min(test_size, len(labels) - 1)
    x_train, x_test, y_train, y_test = train_test_split(
        vectors,
        labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels if can_stratify else None,
    )
    probe_result = run_train_test_embedding_probe(
        train_embeddings=x_train,
        test_embeddings=x_test,
        train_labels=y_train,
        test_labels=y_test,
        random_seed=random_seed,
        model_type=resolved_model_type,
    )
    return {
        "status": probe_result["status"],
        "column": target_column,
        "accuracy": probe_result["accuracy"],
        "macro_f1": probe_result["macro_f1"],
        "model_type": probe_result["model_type"],
        "reason": probe_result["reason"],
    }


def _prefix_probe_result(prefix: str, probe_result: dict[str, Any]) -> dict[str, Any]:
    """Apply one output prefix to a generic probe result."""

    return {
        f"{prefix}_status": probe_result["status"],
        f"{prefix}_column": probe_result["column"],
        f"{prefix}_accuracy": probe_result["accuracy"],
        f"{prefix}_macro_f1": probe_result["macro_f1"],
        f"{prefix}_model_type": probe_result["model_type"],
        f"{prefix}_reason": probe_result["reason"],
    }


def _summary_row_for_result(
    dataset: LoadedDataset,
    result: EmbeddingResult,
    *,
    symmetrized_directed_graph: bool,
    label_column: str | None,
    protected_attribute_column: str | None,
    run_protected_attribute_probe: bool,
    random_seed: int,
    probe_model_type: str,
) -> dict[str, Any]:
    frame = result.embedding_frame
    vector_matrix = frame[result.vector_columns].to_numpy(dtype=float)
    row: dict[str, Any] = {
        "dataset": dataset.name,
        "method": result.method_name,
        "status": "ok",
        "runtime_seconds": float(result.runtime_seconds),
        "node_count": int(result.node_count),
        "embedding_dim": int(result.embedding_dim),
        "graph_was_directed": bool(dataset.graph.is_directed()),
        "graph_was_symmetrized": bool(symmetrized_directed_graph),
        "all_nodes_embedded": len(frame) == dataset.graph.number_of_nodes(),
        "all_finite": bool(np.isfinite(vector_matrix).all()),
        "ml_ready": bool(np.isfinite(vector_matrix).all() and np.var(vector_matrix) > 0.0),
        "mean_vector_norm": float(np.linalg.norm(vector_matrix, axis=1).mean()),
        "std_vector_norm": float(np.linalg.norm(vector_matrix, axis=1).std(ddof=0)),
        "mean_pairwise_cosine": _pairwise_cosine_mean(frame),
        "skip_reason": "",
        "error_message": "",
        "csv_path": result.metadata.get("export_paths", {}).get("csv", pd.NA),
        "pickle_path": result.metadata.get("export_paths", {}).get("pickle", pd.NA),
        "npy_path": result.metadata.get("export_paths", {}).get("npy", pd.NA),
    }
    row.update(
        _prefix_probe_result(
            "label_probe",
            _embedding_probe(
                frame,
                dataset=dataset,
                target_column=label_column,
                random_seed=random_seed,
                requested=label_column is not None,
                model_type=probe_model_type,
            ),
        )
    )
    row.update(
        _prefix_probe_result(
            "protected_probe",
            _embedding_probe(
                frame,
                dataset=dataset,
                target_column=protected_attribute_column,
                random_seed=random_seed,
                requested=bool(run_protected_attribute_probe),
                model_type=probe_model_type,
            ),
        )
    )
    return row


def _build_feature_input(
    dataset: LoadedDataset,
    graph: nx.Graph,
    *,
    selected_methods: Sequence[str],
    features: PreparedFeatures | pd.DataFrame | np.ndarray | Sequence[Sequence[float]] | None,
    random_seed: int,
) -> PreparedFeatures | None:
    if features is not None:
        return coerce_input_features(features, sorted_node_ids(graph))
    if any(method_requires_features(method_name) for method_name in selected_methods):
        return prepare_benchmark_features(dataset, graph=graph, random_seed=random_seed)
    return None


def _skip_summary_row(
    dataset: LoadedDataset,
    *,
    method_name: str,
    graph: nx.Graph,
    symmetrized: bool,
    needs_features: bool,
    reason: str,
    protected_attribute_column: str | None,
    run_protected_attribute_probe: bool,
    probe_model_type: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset.name,
        "method": method_name,
        "status": "skipped",
        "runtime_seconds": 0.0,
        "node_count": int(graph.number_of_nodes()),
        "embedding_dim": pd.NA,
        "graph_was_directed": bool(dataset.graph.is_directed()),
        "graph_was_symmetrized": bool(symmetrized),
        "all_nodes_embedded": False,
        "all_finite": False,
        "ml_ready": False,
        "mean_vector_norm": pd.NA,
        "std_vector_norm": pd.NA,
        "mean_pairwise_cosine": pd.NA,
        "label_probe_status": "not_run",
        "label_probe_accuracy": pd.NA,
        "label_probe_macro_f1": pd.NA,
        "label_probe_model_type": pd.NA,
        "label_probe_column": pd.NA,
        "label_probe_reason": "",
        "protected_probe_status": "not_run" if run_protected_attribute_probe else "not_requested",
        "protected_probe_column": protected_attribute_column if protected_attribute_column is not None else pd.NA,
        "protected_probe_accuracy": pd.NA,
        "protected_probe_macro_f1": pd.NA,
        "protected_probe_model_type": resolve_probe_model_type(probe_model_type) if run_protected_attribute_probe else pd.NA,
        "protected_probe_reason": "",
        "skip_reason": reason,
        "error_message": "",
        "csv_path": pd.NA,
        "pickle_path": pd.NA,
        "npy_path": pd.NA,
        "uses_features": bool(needs_features),
    }


def _failed_summary_row(
    dataset: LoadedDataset,
    *,
    method_name: str,
    graph: nx.Graph,
    symmetrized: bool,
    needs_features: bool,
    error_message: str,
    protected_attribute_column: str | None,
    run_protected_attribute_probe: bool,
    probe_model_type: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset.name,
        "method": method_name,
        "status": "failed",
        "runtime_seconds": 0.0,
        "node_count": int(graph.number_of_nodes()),
        "embedding_dim": pd.NA,
        "graph_was_directed": bool(dataset.graph.is_directed()),
        "graph_was_symmetrized": bool(symmetrized),
        "all_nodes_embedded": False,
        "all_finite": False,
        "ml_ready": False,
        "mean_vector_norm": pd.NA,
        "std_vector_norm": pd.NA,
        "mean_pairwise_cosine": pd.NA,
        "label_probe_status": "not_run",
        "label_probe_accuracy": pd.NA,
        "label_probe_macro_f1": pd.NA,
        "label_probe_model_type": pd.NA,
        "label_probe_column": pd.NA,
        "label_probe_reason": "",
        "protected_probe_status": "not_run" if run_protected_attribute_probe else "not_requested",
        "protected_probe_column": protected_attribute_column if protected_attribute_column is not None else pd.NA,
        "protected_probe_accuracy": pd.NA,
        "protected_probe_macro_f1": pd.NA,
        "protected_probe_model_type": resolve_probe_model_type(probe_model_type) if run_protected_attribute_probe else pd.NA,
        "protected_probe_reason": "",
        "skip_reason": "",
        "error_message": error_message,
        "csv_path": pd.NA,
        "pickle_path": pd.NA,
        "npy_path": pd.NA,
        "uses_features": bool(needs_features),
    }


def _run_benchmark_method_job(
    dataset: LoadedDataset,
    *,
    method_name: str,
    config: Mapping[str, Any],
    output_dir: Path | None,
    export_formats: Sequence[str],
    features: PreparedFeatures | None,
    symmetrize_directed: bool,
    label_column: str | None,
    protected_attribute_column: str | None,
    run_protected_attribute_probe: bool,
    symmetrized: bool,
    needs_features: bool,
    probe_model_type: str,
) -> tuple[str, EmbeddingResult, dict[str, Any]]:
    result = run_embedding_method(
        dataset,
        method_name,
        config=config,
        output_dir=output_dir,
        export_formats=export_formats,
        features=features,
        symmetrize_directed=symmetrize_directed,
    )
    summary_row = _summary_row_for_result(
        dataset,
        result,
        symmetrized_directed_graph=symmetrized,
        label_column=label_column,
        protected_attribute_column=protected_attribute_column,
        run_protected_attribute_probe=run_protected_attribute_probe,
        random_seed=int(config.get("random_seed", 42)),
        probe_model_type=probe_model_type,
    )
    summary_row["uses_features"] = bool(needs_features)
    return method_name, result, summary_row


def _resolve_max_workers(max_workers: int | None, runnable_method_count: int) -> int:
    if runnable_method_count <= 1:
        return 1
    if max_workers is None or int(max_workers) == 0:
        return runnable_method_count
    if int(max_workers) < 0:
        raise ValueError("max_workers must be zero, positive, or None.")
    return min(int(max_workers), runnable_method_count)


def run_embedding_method(
    dataset: LoadedDataset,
    method_name: str,
    *,
    config: Mapping[str, Any] | None = None,
    output_dir: Path | str | None = None,
    export_formats: Sequence[str] | None = ("csv", "pickle"),
    features: PreparedFeatures | pd.DataFrame | np.ndarray | Sequence[Sequence[float]] | None = None,
    symmetrize_directed: bool = True,
) -> EmbeddingResult:
    """Run one embedding method and optionally export its outputs."""

    graph, symmetrized = _benchmark_graph(dataset.graph, symmetrize_directed=symmetrize_directed)
    method_key = str(method_name).strip().lower()
    dependency_reason = missing_dependency_reason(method_key)
    if dependency_reason is not None:
        raise ValueError(dependency_reason)
    graph_reason = unsupported_graph_reason(method_key, graph)
    if graph_reason is not None:
        raise ValueError(graph_reason)

    method_features = None
    if method_requires_features(method_key):
        method_features = (
            coerce_input_features(features, sorted_node_ids(graph))
            if features is not None
            else prepare_benchmark_features(
                dataset,
                graph=graph,
                random_seed=int((config or {}).get("random_seed", 42)),
            )
        )

    model = create_embedding_model(method_key, config=config)
    model.fit(graph, features=method_features)
    result = model.result
    result.metadata["graph_was_symmetrized"] = bool(symmetrized)

    normalized_formats = _normalize_export_formats(export_formats)
    if output_dir is not None and normalized_formats:
        result.metadata["export_paths"] = _save_embedding_outputs(
            result,
            dataset_name=dataset.name,
            output_dir=Path(output_dir),
            export_formats=normalized_formats,
        )
    return result


def run_embedding_benchmark(
    dataset: LoadedDataset,
    methods: Sequence[str] | None = None,
    *,
    method_configs: Mapping[str, Mapping[str, Any]] | None = None,
    output_dir: Path | str | None = None,
    export_formats: Sequence[str] | None = ("csv", "pickle"),
    label_column: str | None = None,
    evaluation_tasks: Sequence[str] | None = None,
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
    run_protected_attribute_probe: bool = False,
    features: PreparedFeatures | pd.DataFrame | np.ndarray | Sequence[Sequence[float]] | None = None,
    max_workers: int | None = None,
    symmetrize_directed: bool = True,
    continue_on_error: bool = True,
) -> BenchmarkRunResult:
    """Run one or more embedding methods through the shared benchmark pipeline."""

    selected_methods = resolve_method_names(methods)
    graph, symmetrized = _benchmark_graph(dataset.graph, symmetrize_directed=symmetrize_directed)
    method_configs = {str(key).lower(): dict(value) for key, value in (method_configs or {}).items()}
    shared_random_seed = int(next(iter(method_configs.values()), {}).get("random_seed", 42))
    feature_input = _build_feature_input(
        dataset,
        graph,
        selected_methods=selected_methods,
        features=features,
        random_seed=shared_random_seed,
    )

    results_by_method: dict[str, EmbeddingResult] = {}
    summary_rows_by_method: dict[str, dict[str, Any]] = {}
    normalized_formats = _normalize_export_formats(export_formats)
    resolved_output_dir = None if output_dir is None else Path(output_dir)
    runnable_methods: list[tuple[str, dict[str, Any], bool, PreparedFeatures | None]] = []

    for method_name in selected_methods:
        config = method_configs.get(method_name, {})
        spec = get_method_spec(method_name)
        dependency_reason = missing_dependency_reason(method_name)
        graph_reason = unsupported_graph_reason(method_name, graph)
        if dependency_reason is not None or graph_reason is not None:
            summary_rows_by_method[method_name] = _skip_summary_row(
                dataset,
                method_name=method_name,
                graph=graph,
                symmetrized=symmetrized,
                needs_features=bool(spec.needs_features),
                reason=dependency_reason or graph_reason or "",
                protected_attribute_column=protected_attribute_column,
                run_protected_attribute_probe=run_protected_attribute_probe,
                probe_model_type=probe_model_type,
            )
            continue
        runnable_methods.append((method_name, config, bool(spec.needs_features), feature_input if spec.needs_features else None))

    resolved_max_workers = _resolve_max_workers(max_workers, len(runnable_methods))
    if resolved_max_workers == 1:
        for method_name, config, needs_features, method_feature_input in runnable_methods:
            try:
                _, result, summary_row = _run_benchmark_method_job(
                    dataset,
                    method_name=method_name,
                    config=config,
                    output_dir=resolved_output_dir,
                    export_formats=normalized_formats,
                    features=method_feature_input,
                    symmetrize_directed=symmetrize_directed,
                    label_column=label_column,
                    protected_attribute_column=protected_attribute_column,
                    run_protected_attribute_probe=run_protected_attribute_probe,
                    symmetrized=symmetrized,
                    needs_features=needs_features,
                    probe_model_type=probe_model_type,
                )
                results_by_method[method_name] = result
                summary_rows_by_method[method_name] = summary_row
            except Exception as exc:
                if not continue_on_error:
                    raise
                summary_rows_by_method[method_name] = _failed_summary_row(
                    dataset,
                    method_name=method_name,
                    graph=graph,
                    symmetrized=symmetrized,
                    needs_features=needs_features,
                    error_message=str(exc),
                    protected_attribute_column=protected_attribute_column,
                    run_protected_attribute_probe=run_protected_attribute_probe,
                    probe_model_type=probe_model_type,
                )
    else:
        future_to_method: dict[Any, tuple[str, bool]] = {}
        with ThreadPoolExecutor(max_workers=resolved_max_workers, thread_name_prefix="embedding-benchmark") as executor:
            for method_name, config, needs_features, method_feature_input in runnable_methods:
                future = executor.submit(
                    _run_benchmark_method_job,
                    dataset,
                    method_name=method_name,
                    config=config,
                    output_dir=resolved_output_dir,
                    export_formats=normalized_formats,
                    features=method_feature_input,
                    symmetrize_directed=symmetrize_directed,
                    label_column=label_column,
                    protected_attribute_column=protected_attribute_column,
                    run_protected_attribute_probe=run_protected_attribute_probe,
                    symmetrized=symmetrized,
                    needs_features=needs_features,
                    probe_model_type=probe_model_type,
                )
                future_to_method[future] = (method_name, needs_features)

            for future in as_completed(future_to_method):
                method_name, needs_features = future_to_method[future]
                try:
                    completed_method_name, result, summary_row = future.result()
                except Exception as exc:
                    if not continue_on_error:
                        for pending_future in future_to_method:
                            pending_future.cancel()
                        raise
                    summary_rows_by_method[method_name] = _failed_summary_row(
                        dataset,
                        method_name=method_name,
                        graph=graph,
                        symmetrized=symmetrized,
                        needs_features=needs_features,
                        error_message=str(exc),
                        protected_attribute_column=protected_attribute_column,
                        run_protected_attribute_probe=run_protected_attribute_probe,
                        probe_model_type=probe_model_type,
                    )
                    continue
                results_by_method[completed_method_name] = result
                summary_rows_by_method[completed_method_name] = summary_row

    summary_rows = [summary_rows_by_method[method_name] for method_name in selected_methods if method_name in summary_rows_by_method]
    summary_frame = pd.DataFrame(summary_rows)
    if resolved_output_dir is not None:
        summary_dir = resolved_output_dir / dataset.name / "embeddings"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"{dataset.name}_embedding_benchmark_summary.csv"
        summary_frame.to_csv(summary_path, index=False)

    benchmark_result = BenchmarkRunResult(
        dataset_name=dataset.name,
        summary_frame=summary_frame,
        results_by_method=results_by_method,
        output_dir=resolved_output_dir,
    )
    normalized_evaluation_tasks = resolve_evaluation_tasks(evaluation_tasks)
    if normalized_evaluation_tasks:
        evaluation_frame = evaluate_embedding_benchmark(
            dataset,
            benchmark_result,
            tasks=normalized_evaluation_tasks,
            graph=graph,
            symmetrize_directed=symmetrize_directed,
            random_seed=shared_random_seed,
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
        benchmark_result.evaluation_frame = evaluation_frame
        if resolved_output_dir is not None and not evaluation_frame.empty:
            evaluation_frame.to_csv(_evaluation_summary_path(resolved_output_dir, dataset.name), index=False)
    return benchmark_result
