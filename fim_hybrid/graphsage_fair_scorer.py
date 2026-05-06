"""Fairness-aware GraphSAGE scorer for the active FIM stack."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import networkx as nx
import numpy as np
import pandas as pd

from .baselines import select_baseline_seed_set
from .community_detection import CommunityDetectionResult
from .data_loader import LoadedDataset, ProtectedGroupReport
from .evaluation import compute_ideal_influences_proportional
from .gnn_training import _build_edge_index, gnn_dependencies_available
from .ml_training import RankingTrainingResult, _rank_nodes, select_ml_candidate_nodes
from .safe_math import safe_divide, safe_minmax_normalize
from .search_evaluator import SearchObjectiveEvaluator


_TRAINING_TARGETS = {
    "fairness_ris_score",
    "shortfall_gain",
    "combined_fairness_gain",
}
_LOSSES = {"mse", "smooth_l1"}


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _normalize_map(scores: Mapping[Any, float], ordered_nodes: list[Any]) -> dict[Any, float]:
    values = [float(scores.get(node_id, 0.0)) for node_id in ordered_nodes]
    normalized = safe_minmax_normalize(
        values,
        default=0.0,
        context="GraphSAGE fairness scorer component normalization",
    )
    return {
        node_id: float(value)
        for node_id, value in zip(ordered_nodes, normalized, strict=True)
    }


def _normalize_array(values: np.ndarray) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    clean = np.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = safe_minmax_normalize(
        clean.tolist(),
        default=0.0,
        context="GraphSAGE fairness scorer array normalization",
    )
    return np.asarray(normalized, dtype=np.float32)


def _safe_corr(left: np.ndarray, right: np.ndarray, method: str) -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    left_series = pd.Series(left.astype(float))
    right_series = pd.Series(right.astype(float))
    if float(left_series.std(ddof=0)) <= 1e-12 or float(right_series.std(ddof=0)) <= 1e-12:
        return 0.0
    corr = left_series.corr(right_series, method=method)
    if pd.isna(corr):
        return 0.0
    return float(corr)


def _precision_at_k(
    node_ids: list[Any],
    labels: np.ndarray,
    predictions: np.ndarray,
    k: int,
) -> float:
    if k < 1:
        return 0.0
    limit = min(int(k), len(node_ids))
    predicted_order = np.argsort(-predictions, kind="mergesort")[:limit]
    label_order = np.argsort(-labels, kind="mergesort")[:limit]
    return safe_divide(
        float(len(set(predicted_order.tolist()) & set(label_order.tolist()))),
        float(limit),
        default=0.0,
        context="GraphSAGE precision@k",
    )


def _ndcg_at_k(labels: np.ndarray, predictions: np.ndarray, k: int) -> float:
    if k < 1 or labels.size == 0:
        return 0.0
    limit = min(int(k), int(labels.size))
    order = np.argsort(-predictions, kind="mergesort")[:limit]
    ideal_order = np.argsort(-labels, kind="mergesort")[:limit]
    discounts = 1.0 / np.log2(np.arange(2, limit + 2, dtype=float))
    dcg = float(np.sum(labels[order] * discounts))
    ideal = float(np.sum(labels[ideal_order] * discounts))
    return safe_divide(dcg, ideal, default=0.0, context="GraphSAGE ndcg@k")


def _coverage_ratio(values: list[Any], denominator: int) -> tuple[int, float]:
    unique_count = len(set(str(value) for value in values))
    return unique_count, safe_divide(
        float(unique_count),
        float(max(1, denominator)),
        default=0.0,
        context="GraphSAGE top-k coverage",
    )


def _component_stats(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for column_name in columns:
        values = pd.to_numeric(frame[column_name], errors="coerce").dropna()
        if values.empty:
            stats[column_name] = {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
            continue
        stats[column_name] = {
            "count": int(values.count()),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return stats


@dataclass(frozen=True, slots=True)
class GraphSAGEFairScorerConfig:
    """Runtime controls for fairness-aware GraphSAGE node scoring."""

    training_target: str = "combined_fairness_gain"
    hidden_dim: int = 32
    embedding_dim: int = 64
    epochs: int = 100
    learning_rate: float = 0.005
    dropout: float = 0.2
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    loss: str = "smooth_l1"
    use_cache: bool = True
    regenerate_cache: bool = False
    allow_structural_fallback: bool = False
    allow_protected_features_in_ml: bool = False
    score_weight: float = 0.8
    auto_downweight_bad_ml: bool = True
    disable_graphsage_guidance: bool = False
    budget: int = 1
    random_seed: int = 42
    propagation_probability: float = 0.01
    mc_runs_search: int = 20
    diffusion_model: str = "ic"
    top_fraction: float | None = None
    top_n: int | None = None
    max_nodes: int | None = None
    output_dir: Path | None = None


@dataclass(slots=True)
class GraphSAGEFairScorerResult:
    """Artifacts from fairness-aware GraphSAGE training."""

    training_result: RankingTrainingResult
    feature_frame: pd.DataFrame
    label_frame: pd.DataFrame
    prediction_frame: pd.DataFrame
    training_history: pd.DataFrame
    metrics: dict[str, float]
    effective_score_weight: float
    warnings: list[str] = field(default_factory=list)
    inactive_components: list[str] = field(default_factory=list)
    output_paths: dict[str, str] = field(default_factory=dict)
    component_score_maps: dict[str, dict[Any, float]] = field(default_factory=dict)
    label_generation_runtime_seconds: float = 0.0


def _validate_config(config: GraphSAGEFairScorerConfig) -> None:
    if config.training_target not in _TRAINING_TARGETS:
        raise ValueError(f"graphsage_training_target must be one of {sorted(_TRAINING_TARGETS)}.")
    if config.hidden_dim < 1:
        raise ValueError("graphsage_hidden_dim must be at least 1.")
    if config.embedding_dim < 1:
        raise ValueError("graphsage_embedding_dim must be at least 1.")
    if config.epochs < 1:
        raise ValueError("graphsage_epochs must be at least 1.")
    if config.learning_rate <= 0.0:
        raise ValueError("graphsage_learning_rate must be positive.")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("graphsage_dropout must be in the interval [0.0, 1.0).")
    if config.weight_decay < 0.0:
        raise ValueError("graphsage_weight_decay must be non-negative.")
    if config.early_stopping_patience < 1:
        raise ValueError("graphsage_early_stopping_patience must be at least 1.")
    if not 0.0 < config.train_ratio < 1.0:
        raise ValueError("graphsage_train_ratio must be in the interval (0.0, 1.0).")
    if not 0.0 <= config.val_ratio < 1.0:
        raise ValueError("graphsage_val_ratio must be in the interval [0.0, 1.0).")
    if config.train_ratio + config.val_ratio >= 1.0:
        raise ValueError("graphsage_train_ratio + graphsage_val_ratio must be less than 1.0.")
    if config.loss not in _LOSSES:
        raise ValueError(f"graphsage_loss must be one of {sorted(_LOSSES)}.")
    if config.budget < 1:
        raise ValueError("budget must be at least 1.")


def _output_path(output_dir: Path | None, filename: str) -> Path | None:
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def _write_frame(output_dir: Path | None, filename: str, frame: pd.DataFrame) -> str:
    path = _output_path(output_dir, filename)
    if path is None:
        return ""
    frame.to_csv(path, index=False)
    return str(path)


def _write_json(output_dir: Path | None, filename: str, payload: Mapping[str, Any]) -> str:
    path = _output_path(output_dir, filename)
    if path is None:
        return ""
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _build_safe_node_features(
    *,
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    community_result: CommunityDetectionResult,
    ris_scores: Mapping[Any, float] | None,
    allow_protected_features_in_ml: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    graph = dataset.graph
    work_graph = graph if not graph.is_directed() else graph.to_undirected()
    ordered_nodes = sorted(graph.nodes(), key=_sort_key)
    node_count = max(1, len(ordered_nodes))
    degree_map = {node_id: float(work_graph.degree(node_id)) for node_id in ordered_nodes}
    max_degree = max(degree_map.values(), default=1.0)
    pagerank = nx.pagerank(work_graph) if work_graph.number_of_nodes() else {}
    clustering = nx.clustering(work_graph)
    if work_graph.number_of_edges() > 0:
        try:
            core_numbers = {node_id: float(value) for node_id, value in nx.core_number(work_graph).items()}
        except nx.NetworkXError:
            core_numbers = {node_id: 0.0 for node_id in ordered_nodes}
    else:
        core_numbers = {node_id: 0.0 for node_id in ordered_nodes}
    group_by_node = {
        node_id: str(group_name)
        for group_name, node_ids in protected_group_report.protected_groups.items()
        for node_id in node_ids
    }
    community_sizes = dict(community_result.stats.community_sizes)
    community_ids = sorted(set(community_result.community_id_by_node.values()))
    community_index = {community_id: index for index, community_id in enumerate(community_ids)}
    max_community_index = max(1, len(community_ids) - 1)
    rows: list[dict[str, Any]] = []
    for node_id in ordered_nodes:
        neighbors = list(work_graph.neighbors(node_id))
        neighbor_degrees = [degree_map.get(neighbor_id, 0.0) for neighbor_id in neighbors]
        community_id = community_result.community_id_by_node[node_id]
        rows.append(
            {
                "node_id": node_id,
                "protected_group": group_by_node.get(node_id, ""),
                "community_id": community_id,
                "degree": safe_divide(degree_map[node_id], max_degree, default=0.0, context="GraphSAGE degree feature"),
                "normalized_degree": safe_divide(
                    degree_map[node_id],
                    float(max(1, node_count - 1)),
                    default=0.0,
                    context="GraphSAGE normalized degree feature",
                ),
                "pagerank": float(pagerank.get(node_id, 0.0)),
                "clustering_coefficient": float(clustering.get(node_id, 0.0)),
                "k_core_number": float(core_numbers.get(node_id, 0.0)),
                "community_size": float(community_sizes.get(community_id, 1)),
                "community_id_encoded": safe_divide(
                    float(community_index.get(community_id, 0)),
                    float(max_community_index),
                    default=0.0,
                    context="GraphSAGE community id encoding",
                ),
                "neighbor_degree_mean": float(np.mean(neighbor_degrees)) if neighbor_degrees else 0.0,
                "neighbor_degree_max": float(max(neighbor_degrees)) if neighbor_degrees else 0.0,
                "ris_coverage_proxy": float((ris_scores or {}).get(node_id, 0.0)),
            }
        )
    frame = pd.DataFrame(rows)
    numeric_columns = [
        "degree",
        "normalized_degree",
        "pagerank",
        "clustering_coefficient",
        "k_core_number",
        "community_size",
        "community_id_encoded",
        "neighbor_degree_mean",
        "neighbor_degree_max",
        "ris_coverage_proxy",
    ]
    for column_name in numeric_columns:
        frame[column_name] = _normalize_array(pd.to_numeric(frame[column_name], errors="coerce").to_numpy(dtype=float))
    frame = frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    feature_columns = list(numeric_columns)
    model_frame = frame[["node_id", *numeric_columns]].copy()
    community_dummies = pd.get_dummies(frame["community_id"].astype(str), prefix="community", dtype=float)
    model_frame = pd.concat([model_frame, community_dummies], axis=1)
    feature_columns.extend(community_dummies.columns.tolist())
    protected_feature_columns: list[str] = []
    if allow_protected_features_in_ml:
        protected_dummies = pd.get_dummies(frame["protected_group"].astype(str), prefix="protected_group", dtype=float)
        model_frame = pd.concat([model_frame, protected_dummies], axis=1)
        protected_feature_columns = protected_dummies.columns.tolist()
        feature_columns.extend(protected_feature_columns)
    return frame, model_frame, feature_columns, protected_feature_columns


def _baseline_group_influence(
    *,
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    community_result: CommunityDetectionResult,
    search_evaluator: SearchObjectiveEvaluator | None,
    config: GraphSAGEFairScorerConfig,
) -> tuple[dict[str, float], tuple[Any, ...]]:
    if search_evaluator is None:
        return {str(group_name): 0.0 for group_name in protected_group_report.group_sizes}, ()
    try:
        baseline_seed_set = select_baseline_seed_set(
            dataset=dataset,
            method="community_round_robin",
            budget=int(config.budget),
            protected_group_report=protected_group_report,
            propagation_probability=float(config.propagation_probability),
            mc_runs=max(1, int(config.mc_runs_search)),
            lambda_weight=0.5,
            community_result=community_result,
            random_seed=int(config.random_seed),
            diffusion_model=str(config.diffusion_model),
        )
        payload = search_evaluator.evaluate_seed_set(baseline_seed_set)
        influence = dict(payload.get("approx_group_influence", {}) or {})
        return {str(group): float(influence.get(group, 0.0)) for group in protected_group_report.group_sizes}, tuple(baseline_seed_set)
    except Exception:
        return {str(group_name): 0.0 for group_name in protected_group_report.group_sizes}, ()


def _build_fairness_ris_labels(
    *,
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    community_result: CommunityDetectionResult,
    ris_artifact: Any | None,
    search_evaluator: SearchObjectiveEvaluator | None,
    ideal_influences: Mapping[str, float] | None,
    config: GraphSAGEFairScorerConfig,
) -> tuple[pd.DataFrame, dict[str, Any], float]:
    start = perf_counter()
    ordered_nodes = sorted(dataset.graph.nodes(), key=_sort_key)
    if ris_artifact is None:
        raise RuntimeError("Fairness-aware GraphSAGE labels require RIS/Fair RIS scores.")
    ris_result = getattr(ris_artifact, "ris_result", None)
    if ris_result is None:
        raise RuntimeError("Fairness-aware GraphSAGE labels require reusable RR sets.")

    if ideal_influences:
        resolved_ideal = {str(group): float(value) for group, value in ideal_influences.items()}
    else:
        resolved_ideal = compute_ideal_influences_proportional(
            dataset=dataset,
            protected_group_report=protected_group_report,
            budget=int(config.budget),
            propagation_probability=float(config.propagation_probability),
            mc_runs=max(10, int(config.mc_runs_search)),
            random_seed=int(config.random_seed),
            diffusion_model=str(config.diffusion_model),
        )
    baseline_influence, baseline_seed_set = _baseline_group_influence(
        dataset=dataset,
        protected_group_report=protected_group_report,
        community_result=community_result,
        search_evaluator=search_evaluator,
        config=config,
    )
    shortfall_ratio = {
        str(group_name): (
            max(0.0, (float(resolved_ideal.get(str(group_name), 0.0)) - float(baseline_influence.get(str(group_name), 0.0)))
                / max(float(resolved_ideal.get(str(group_name), 0.0)), 1e-9))
        )
        for group_name in protected_group_report.group_sizes
    }
    group_weights = {
        group_name: max(0.1, 1.0 + float(shortfall_ratio.get(str(group_name), 0.0)))
        for group_name in protected_group_report.group_sizes
    }
    fair_scores = getattr(ris_artifact, "fair_scores", None)
    if not fair_scores:
        fair_scores = ris_result.weighted_node_scores(group_weights)
    ris_scores = getattr(ris_artifact, "global_scores", None) or ris_result.global_node_scores

    raw_shortfall_gain: dict[Any, float] = {}
    rr_denominator = float(max(1, len(ris_result.rr_sets)))
    for node_id in ordered_nodes:
        group_counts = dict(ris_result.node_group_rr_counts.get(node_id, {}) or {})
        raw_shortfall_gain[node_id] = safe_divide(
            sum(
                float(count) * float(shortfall_ratio.get(str(group_name), 0.0))
                for group_name, count in group_counts.items()
            ),
            rr_denominator,
            default=0.0,
            context="GraphSAGE shortfall gain label",
        )

    community_seed_counts: dict[Any, int] = {}
    for seed_node in baseline_seed_set:
        community_id = community_result.community_id_by_node.get(seed_node)
        community_seed_counts[community_id] = int(community_seed_counts.get(community_id, 0)) + 1
    community_deficit: dict[Any, float] = {}
    total_nodes = max(1, int(dataset.graph.number_of_nodes()))
    for community_id, community_size in community_result.stats.community_sizes.items():
        target = float(config.budget) * float(community_size) / float(total_nodes)
        community_deficit[community_id] = max(
            0.0,
            safe_divide(
                target - float(community_seed_counts.get(community_id, 0)),
                max(target, 1e-9),
                default=0.0,
                context="GraphSAGE community diversity label",
            ),
        )

    work_graph = dataset.graph if not dataset.graph.is_directed() else dataset.graph.to_undirected()
    pagerank = nx.pagerank(work_graph) if work_graph.number_of_nodes() else {}
    degrees = {node_id: float(work_graph.degree(node_id)) for node_id in ordered_nodes}
    degree_scores = _normalize_map(degrees, ordered_nodes)
    pagerank_scores = _normalize_map(pagerank, ordered_nodes)
    ris_norm = _normalize_map(ris_scores, ordered_nodes)
    fair_norm = _normalize_map(fair_scores, ordered_nodes)
    shortfall_norm = _normalize_map(raw_shortfall_gain, ordered_nodes)
    community_raw = {
        node_id: float(community_deficit.get(community_result.community_id_by_node[node_id], 0.0))
        for node_id in ordered_nodes
    }
    community_norm = _normalize_map(community_raw, ordered_nodes)
    spread_proxy_raw = {
        node_id: (
            0.45 * float(degree_scores.get(node_id, 0.0))
            + 0.35 * float(pagerank_scores.get(node_id, 0.0))
            + 0.20 * float(ris_norm.get(node_id, 0.0))
        )
        for node_id in ordered_nodes
    }
    spread_proxy_norm = _normalize_map(spread_proxy_raw, ordered_nodes)

    rows = []
    for node_id in ordered_nodes:
        combined = (
            2.0 * float(fair_norm.get(node_id, 0.0))
            + 3.0 * float(shortfall_norm.get(node_id, 0.0))
            + 1.0 * float(ris_norm.get(node_id, 0.0))
            + 0.5 * float(community_norm.get(node_id, 0.0))
            + 0.3 * float(spread_proxy_norm.get(node_id, 0.0))
        )
        rows.append(
            {
                "node_id": node_id,
                "ris_score": float(ris_norm.get(node_id, 0.0)),
                "fair_ris_score": float(fair_norm.get(node_id, 0.0)),
                "shortfall_gain": float(shortfall_norm.get(node_id, 0.0)),
                "community_diversity_score": float(community_norm.get(node_id, 0.0)),
                "spread_proxy_score": float(spread_proxy_norm.get(node_id, 0.0)),
                "combined_fairness_gain_raw": combined,
            }
        )
    label_frame = pd.DataFrame(rows)
    label_frame["combined_fairness_gain"] = _normalize_array(label_frame["combined_fairness_gain_raw"].to_numpy(dtype=float))
    label_frame["fairness_ris_score"] = label_frame["fair_ris_score"].astype(float)
    label_frame["label_score"] = label_frame[str(config.training_target)].astype(float)
    label_frame = label_frame.drop(columns=["combined_fairness_gain_raw"])
    stats = {
        "training_target": str(config.training_target),
        "ideal_influences": resolved_ideal,
        "baseline_group_influence": baseline_influence,
        "baseline_seed_count": int(len(baseline_seed_set)),
        "shortfall_ratio": shortfall_ratio,
        "label_component_stats": _component_stats(
            label_frame,
            [
                "ris_score",
                "fair_ris_score",
                "shortfall_gain",
                "community_diversity_score",
                "spread_proxy_score",
                "combined_fairness_gain",
                "label_score",
            ],
        ),
    }
    return label_frame, stats, float(perf_counter() - start)


def _build_feature_matrix(
    model_feature_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    merged = model_feature_frame.merge(
        label_frame[["node_id", target_column, "label_score"]],
        on="node_id",
        how="inner",
        validate="one_to_one",
    )
    if merged["node_id"].duplicated().any():
        raise ValueError("GraphSAGE feature table contains duplicate node_id values.")
    missing = set(model_feature_frame["node_id"]) ^ set(label_frame["node_id"])
    if missing:
        preview = sorted(missing, key=_sort_key)[:5]
        raise ValueError(f"GraphSAGE features and labels cover different nodes: {preview}.")
    for column_name in feature_columns:
        merged[column_name] = pd.to_numeric(merged[column_name], errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0)
    merged[target_column] = pd.to_numeric(merged[target_column], errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return merged


def _split_indices(node_count: int, config: GraphSAGEFairScorerConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(node_count, dtype=np.int64)
    rng = np.random.default_rng(int(config.random_seed))
    rng.shuffle(indices)
    if node_count < 4:
        return indices, indices, indices
    train_count = max(1, int(round(float(config.train_ratio) * float(node_count))))
    val_count = max(1, int(round(float(config.val_ratio) * float(node_count))))
    if train_count + val_count >= node_count:
        train_count = max(1, node_count - 2)
        val_count = 1
    train_idx = indices[:train_count]
    val_idx = indices[train_count:train_count + val_count]
    test_idx = indices[train_count + val_count:]
    if test_idx.size == 0:
        test_idx = val_idx.copy()
    return train_idx, val_idx, test_idx


def _build_fingerprint(
    *,
    node_ids: list[Any],
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    edge_index: np.ndarray,
    feature_columns: list[str],
    config: GraphSAGEFairScorerConfig,
) -> str:
    payload = {
        "node_ids": [repr(node_id) for node_id in node_ids],
        "feature_columns": feature_columns,
        "training_target": config.training_target,
        "hidden_dim": int(config.hidden_dim),
        "embedding_dim": int(config.embedding_dim),
        "epochs": int(config.epochs),
        "learning_rate": float(config.learning_rate),
        "dropout": float(config.dropout),
        "weight_decay": float(config.weight_decay),
        "loss": str(config.loss),
        "allow_protected_features_in_ml": bool(config.allow_protected_features_in_ml),
        "random_seed": int(config.random_seed),
    }
    hasher = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
    hasher.update(feature_matrix.astype(np.float32).tobytes())
    hasher.update(labels.astype(np.float32).tobytes())
    hasher.update(edge_index.astype(np.int64).tobytes())
    return hasher.hexdigest()


def _cache_files(output_dir: Path | None) -> dict[str, Path | None]:
    return {
        "metadata": _output_path(output_dir, "graphsage_training_metadata.json"),
        "predictions": _output_path(output_dir, "graphsage_predictions.csv"),
        "history": _output_path(output_dir, "graphsage_training_history.csv"),
        "embeddings": _output_path(output_dir, "embeddings.npy"),
    }


def _load_cached_result(
    *,
    output_dir: Path | None,
    fingerprint: str,
    ordered_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    budget: int,
    top_fraction: float | None,
    top_n: int | None,
    max_nodes: int | None,
    feature_shape: tuple[int, int],
    edge_shape: tuple[int, int],
) -> tuple[RankingTrainingResult, pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, Any]] | None:
    files = _cache_files(output_dir)
    metadata_path = files["metadata"]
    predictions_path = files["predictions"]
    history_path = files["history"]
    embeddings_path = files["embeddings"]
    if (
        metadata_path is None
        or predictions_path is None
        or history_path is None
        or embeddings_path is None
        or not metadata_path.exists()
        or not predictions_path.exists()
        or not history_path.exists()
        or not embeddings_path.exists()
    ):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if metadata.get("cache_fingerprint") != fingerprint:
        return None
    try:
        prediction_frame = pd.read_csv(predictions_path)
        history_frame = pd.read_csv(history_path)
        embeddings = np.load(embeddings_path)
    except Exception:
        return None
    if len(prediction_frame) != len(ordered_frame):
        return None
    node_ids = ordered_frame["node_id"].tolist()
    scores = prediction_frame["graphsage_pred_score"].astype(float).to_numpy()
    predicted_scores = {
        node_id: float(score)
        for node_id, score in zip(node_ids, scores.tolist(), strict=True)
    }
    ranked_nodes = _rank_nodes(predicted_scores)
    candidate_nodes = select_ml_candidate_nodes(
        ranked_nodes=ranked_nodes,
        budget=int(budget),
        top_fraction=top_fraction,
        top_n=top_n,
        max_nodes=max_nodes,
    )
    metrics = dict(metadata.get("metrics", {}) or {})
    training_result = RankingTrainingResult(
        model=None,
        training_frame=ordered_frame.copy(),
        predicted_scores=predicted_scores,
        ranked_nodes=ranked_nodes,
        candidate_nodes=candidate_nodes,
        validation_spearman=float(metrics.get("spearman", 0.0)),
        validation_precision_at_budget=float(metrics.get("precision_at_budget", 0.0)),
        runtime_seconds=float(metadata.get("runtime_seconds", 0.0)),
        model_type="graphsage",
        backend="gnn",
        target_type="regression",
        target_column=str(metadata.get("training_target", "combined_fairness_gain")),
        loaded_from_cache=True,
        feature_matrix_shape=feature_shape,
        edge_index_shape=edge_shape,
        debias_mode="none",
        metadata=dict(metadata),
    )
    return training_result, prediction_frame, history_frame, embeddings, metadata


def _train_graphsage_regressor(
    *,
    dataset: LoadedDataset,
    ordered_frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    config: GraphSAGEFairScorerConfig,
    fingerprint: str,
    edge_index: np.ndarray,
) -> tuple[Any, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any], float, bool]:
    if not gnn_dependencies_available():
        if not bool(config.allow_structural_fallback):
            raise RuntimeError(
                "GraphSAGE fairness scorer requires optional dependencies 'torch' and 'torch_geometric'. "
                "Install them or pass --allow-structural-fallback."
            )
        labels = ordered_frame[target_column].to_numpy(dtype=np.float32, copy=True)
        predictions = _normalize_array(ordered_frame.get("ris_coverage_proxy", ordered_frame[target_column]).to_numpy(dtype=float))
        embeddings = ordered_frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
        history = pd.DataFrame([{"epoch": 0, "train_loss": 0.0, "validation_loss": 0.0}])
        metadata = {
            "cache_fingerprint": fingerprint,
            "fallback": "structural",
            "warning": "GraphSAGE dependencies unavailable; structural fallback was explicitly allowed.",
        }
        return None, predictions, embeddings, history, metadata, 0.0, False

    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415
    from torch_geometric.nn import SAGEConv  # noqa: PLC0415

    start = perf_counter()
    torch.manual_seed(int(config.random_seed))
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    feature_matrix = ordered_frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
    labels = ordered_frame[target_column].to_numpy(dtype=np.float32, copy=True)
    train_idx, val_idx, test_idx = _split_indices(len(ordered_frame), config)

    class GraphSAGERegressor(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float) -> None:
            super().__init__()
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, embedding_dim)
            self.dropout = float(dropout)
            self.score_head = nn.Linear(embedding_dim, 1)

        def encode(self, x: Any, edge_index_tensor: Any) -> Any:
            h = self.conv1(x, edge_index_tensor)
            h = nn.functional.relu(h)
            h = nn.functional.dropout(h, p=self.dropout, training=self.training)
            h = self.conv2(h, edge_index_tensor)
            return nn.functional.relu(h)

        def forward(self, x: Any, edge_index_tensor: Any) -> Any:
            embeddings = self.encode(x, edge_index_tensor)
            return self.score_head(embeddings).squeeze(-1)

    model = GraphSAGERegressor(
        input_dim=int(feature_matrix.shape[1]),
        hidden_dim=int(config.hidden_dim),
        embedding_dim=int(config.embedding_dim),
        dropout=float(config.dropout),
    )
    x = torch.tensor(feature_matrix, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    edge_tensor = torch.tensor(edge_index, dtype=torch.long)
    train_tensor = torch.tensor(train_idx, dtype=torch.long)
    val_tensor = torch.tensor(val_idx, dtype=torch.long)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    loss_fn = nn.SmoothL1Loss() if config.loss == "smooth_l1" else nn.MSELoss()
    best_state: dict[str, Any] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    rows: list[dict[str, float | int]] = []
    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        optimizer.zero_grad()
        predictions = model(x, edge_tensor)
        train_loss = loss_fn(predictions[train_tensor], y[train_tensor])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            epoch_predictions = model(x, edge_tensor)
            val_loss = float(loss_fn(epoch_predictions[val_tensor], y[val_tensor]).item())
        train_loss_value = float(train_loss.detach().cpu().item())
        rows.append(
            {
                "epoch": int(epoch),
                "train_loss": train_loss_value,
                "validation_loss": val_loss,
            }
        )
        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_epoch = int(epoch)
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        if stale_epochs >= int(config.early_stopping_patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_embeddings_tensor = model.encode(x, edge_tensor)
        final_predictions_tensor = model.score_head(final_embeddings_tensor).squeeze(-1)
    raw_predictions = final_predictions_tensor.detach().cpu().numpy().astype(float, copy=False)
    predictions = _normalize_array(raw_predictions)
    embeddings = final_embeddings_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    history = pd.DataFrame(rows)
    metadata = {
        "cache_fingerprint": fingerprint,
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(history)),
        "train_indices": train_idx.tolist(),
        "validation_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "best_validation_loss": float(best_val_loss),
    }
    return model, predictions, embeddings, history, metadata, float(perf_counter() - start), False


def _evaluate_graphsage_quality(
    *,
    ordered_frame: pd.DataFrame,
    prediction_scores: np.ndarray,
    embeddings: np.ndarray,
    feature_metadata: pd.DataFrame,
    target_column: str,
    budget: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    labels = ordered_frame[target_column].to_numpy(dtype=float, copy=True)
    predictions = np.asarray(prediction_scores, dtype=float)
    residual = predictions - labels
    node_ids = ordered_frame["node_id"].tolist()
    prediction_frame = pd.DataFrame(
        {
            "node_id": node_ids,
            "graphsage_pred_score": predictions,
            "graphsage_raw_label": labels,
            "graphsage_embedding_norm": _normalize_array(np.linalg.norm(embeddings, axis=1)),
        }
    )
    prediction_frame = prediction_frame.merge(
        feature_metadata[["node_id", "protected_group", "community_id"]],
        on="node_id",
        how="left",
        validate="one_to_one",
    )
    top_k = max(1, min(int(budget), len(prediction_frame)))
    ordered_top = prediction_frame.sort_values(
        by=["graphsage_pred_score", "node_id"],
        ascending=[False, True],
    ).head(top_k)
    protected_count, protected_ratio = _coverage_ratio(
        ordered_top["protected_group"].astype(str).tolist(),
        int(feature_metadata["protected_group"].astype(str).nunique()),
    )
    community_count, community_ratio = _coverage_ratio(
        ordered_top["community_id"].astype(str).tolist(),
        int(feature_metadata["community_id"].astype(str).nunique()),
    )
    metrics = {
        "spearman": _safe_corr(labels, predictions, "spearman"),
        "pearson": _safe_corr(labels, predictions, "pearson"),
        "mae": float(np.mean(np.abs(residual))) if residual.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(residual)))) if residual.size else 0.0,
        "precision_at_budget": _precision_at_k(node_ids, labels, predictions, int(budget)),
        "precision_at_2budget": _precision_at_k(node_ids, labels, predictions, int(2 * budget)),
        "ndcg_at_budget": _ndcg_at_k(labels, predictions, int(budget)),
        "topk_protected_group_count": float(protected_count),
        "topk_protected_group_coverage": float(protected_ratio),
        "topk_community_count": float(community_count),
        "topk_community_coverage": float(community_ratio),
        "prediction_std": float(np.std(predictions)) if predictions.size else 0.0,
        "label_std": float(np.std(labels)) if labels.size else 0.0,
    }
    return prediction_frame, metrics


def _effective_graphsage_weight(
    config: GraphSAGEFairScorerConfig,
    metrics: Mapping[str, float],
) -> tuple[float, list[str], list[str]]:
    warnings: list[str] = []
    inactive: list[str] = []
    configured = 0.0 if bool(config.disable_graphsage_guidance) else float(config.score_weight)
    effective = configured
    if bool(config.disable_graphsage_guidance):
        inactive.append("graphsage_guidance")
        return 0.0, warnings, inactive
    prediction_std = float(metrics.get("prediction_std", 0.0))
    spearman = float(metrics.get("spearman", 0.0))
    if bool(config.auto_downweight_bad_ml):
        if prediction_std <= 1e-6:
            effective = 0.0
            warnings.append("GraphSAGE predictions collapsed to near-constant scores; GraphSAGE guidance weight set to 0.0.")
            inactive.append("graphsage_guidance")
        elif spearman < 0.05:
            effective = min(effective, 0.2)
            warnings.append("GraphSAGE Spearman correlation is poor; GraphSAGE guidance weight capped at 0.2.")
    return float(effective), warnings, inactive


def train_graphsage_fair_candidate_scorer(
    *,
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    community_result: CommunityDetectionResult,
    ris_artifact: Any | None,
    search_evaluator: SearchObjectiveEvaluator | None,
    ideal_influences: Mapping[str, float] | None,
    config: GraphSAGEFairScorerConfig,
) -> GraphSAGEFairScorerResult:
    """Train GraphSAGE to predict Fair RIS/shortfall-aware node utility."""

    _validate_config(config)
    if bool(config.allow_protected_features_in_ml):
        print("Protected attributes are used as ML features. Use only for ablation/debug, not default fairness claim.")

    feature_metadata, model_feature_frame, feature_columns, protected_feature_columns = _build_safe_node_features(
        dataset=dataset,
        protected_group_report=protected_group_report,
        community_result=community_result,
        ris_scores=None if ris_artifact is None else getattr(ris_artifact, "global_scores", None),
        allow_protected_features_in_ml=bool(config.allow_protected_features_in_ml),
    )
    label_frame, label_stats, label_runtime = _build_fairness_ris_labels(
        dataset=dataset,
        protected_group_report=protected_group_report,
        community_result=community_result,
        ris_artifact=ris_artifact,
        search_evaluator=search_evaluator,
        ideal_influences=ideal_influences,
        config=config,
    )
    target_column = str(config.training_target)
    ordered_frame = _build_feature_matrix(
        model_feature_frame=model_feature_frame,
        label_frame=label_frame,
        feature_columns=feature_columns,
        target_column=target_column,
    ).sort_values(by="node_id", key=lambda values: values.map(_sort_key)).reset_index(drop=True)
    node_ids = ordered_frame["node_id"].tolist()
    node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    edge_index = _build_edge_index(dataset, node_to_index)
    feature_matrix = ordered_frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
    labels = ordered_frame[target_column].to_numpy(dtype=np.float32, copy=True)
    fingerprint = _build_fingerprint(
        node_ids=node_ids,
        feature_matrix=feature_matrix,
        labels=labels,
        edge_index=edge_index,
        feature_columns=feature_columns,
        config=config,
    )
    paths: dict[str, str] = {}
    paths["node_features_csv"] = _write_frame(config.output_dir, "node_features.csv", feature_metadata)
    paths["graphsage_training_labels_csv"] = _write_frame(config.output_dir, "graphsage_training_labels.csv", label_frame)
    paths["graphsage_label_stats_json"] = _write_json(config.output_dir, "graphsage_label_component_stats.json", label_stats)

    cached = None
    if bool(config.use_cache) and not bool(config.regenerate_cache):
        cached = _load_cached_result(
            output_dir=config.output_dir,
            fingerprint=fingerprint,
            ordered_frame=ordered_frame,
            label_frame=label_frame,
            budget=int(config.budget),
            top_fraction=config.top_fraction,
            top_n=config.top_n,
            max_nodes=config.max_nodes,
            feature_shape=(int(feature_matrix.shape[0]), int(feature_matrix.shape[1])),
            edge_shape=(int(edge_index.shape[0]), int(edge_index.shape[1])),
        )

    if cached is None:
        model, predictions, embeddings, history_frame, train_metadata, runtime_seconds, _ = _train_graphsage_regressor(
            dataset=dataset,
            ordered_frame=ordered_frame,
            feature_columns=feature_columns,
            target_column=target_column,
            config=config,
            fingerprint=fingerprint,
            edge_index=edge_index,
        )
        prediction_frame, metrics = _evaluate_graphsage_quality(
            ordered_frame=ordered_frame,
            prediction_scores=predictions,
            embeddings=embeddings,
            feature_metadata=feature_metadata,
            target_column=target_column,
            budget=int(config.budget),
        )
        effective_weight, warnings, inactive = _effective_graphsage_weight(config, metrics)
        ranked_nodes = _rank_nodes(
            {
                node_id: float(score)
                for node_id, score in zip(node_ids, predictions.tolist(), strict=True)
            }
        )
        candidate_nodes = select_ml_candidate_nodes(
            ranked_nodes=ranked_nodes,
            budget=int(config.budget),
            top_fraction=config.top_fraction,
            top_n=config.top_n,
            max_nodes=config.max_nodes,
        )
        metadata = {
            **train_metadata,
            "training_target": target_column,
            "graphsage_hidden_dim": int(config.hidden_dim),
            "graphsage_embedding_dim": int(config.embedding_dim),
            "graphsage_epochs_configured": int(config.epochs),
            "graphsage_learning_rate": float(config.learning_rate),
            "graphsage_dropout": float(config.dropout),
            "graphsage_weight_decay": float(config.weight_decay),
            "graphsage_loss": str(config.loss),
            "graphsage_train_ratio": float(config.train_ratio),
            "graphsage_val_ratio": float(config.val_ratio),
            "allow_protected_features_in_ml": bool(config.allow_protected_features_in_ml),
            "protected_feature_columns": protected_feature_columns,
            "feature_columns": feature_columns,
            "metrics": metrics,
            "effective_graphsage_weight": float(effective_weight),
            "warnings": warnings,
            "inactive_guidance_components": inactive,
            "label_stats": label_stats,
            "runtime_seconds": float(runtime_seconds + label_runtime),
            "loaded_from_cache": False,
        }
        paths["graphsage_training_history_csv"] = _write_frame(config.output_dir, "graphsage_training_history.csv", history_frame)
        paths["graphsage_predictions_csv"] = _write_frame(config.output_dir, "graphsage_predictions.csv", prediction_frame)
        paths["node_predictions_csv"] = _write_frame(config.output_dir, "node_predictions.csv", prediction_frame)
        embeddings_path = _output_path(config.output_dir, "embeddings.npy")
        if embeddings_path is not None:
            np.save(embeddings_path, embeddings)
            paths["embeddings_npy"] = str(embeddings_path)
        if model is not None:
            import torch  # noqa: PLC0415

            checkpoint_path = _output_path(config.output_dir, "graphsage_model.pt")
            if checkpoint_path is not None:
                torch.save({"state_dict": model.state_dict(), "metadata": _json_ready(metadata)}, checkpoint_path)
                paths["graphsage_model_checkpoint"] = str(checkpoint_path)
        paths["graphsage_training_metadata_json"] = _write_json(config.output_dir, "graphsage_training_metadata.json", metadata)
        training_result = RankingTrainingResult(
            model=model,
            training_frame=ordered_frame.copy(),
            predicted_scores={
                node_id: float(score)
                for node_id, score in zip(node_ids, predictions.tolist(), strict=True)
            },
            ranked_nodes=ranked_nodes,
            candidate_nodes=candidate_nodes,
            validation_spearman=float(metrics.get("spearman", 0.0)),
            validation_precision_at_budget=float(metrics.get("precision_at_budget", 0.0)),
            runtime_seconds=float(runtime_seconds + label_runtime),
            model_type="graphsage",
            backend="gnn",
            target_type="regression",
            target_column=target_column,
            loaded_from_cache=False,
            feature_matrix_shape=(int(feature_matrix.shape[0]), int(feature_matrix.shape[1])),
            edge_index_shape=(int(edge_index.shape[0]), int(edge_index.shape[1])),
            debias_mode="none",
            metadata=metadata,
        )
    else:
        training_result, prediction_frame, history_frame, embeddings, metadata = cached
        metrics = dict(metadata.get("metrics", {}) or {})
        effective_weight = float(metadata.get("effective_graphsage_weight", config.score_weight))
        warnings = list(metadata.get("warnings", []) or [])
        inactive = list(metadata.get("inactive_guidance_components", []) or [])
        paths["graphsage_training_history_csv"] = str(_cache_files(config.output_dir)["history"] or "")
        paths["graphsage_predictions_csv"] = str(_cache_files(config.output_dir)["predictions"] or "")
        paths["node_predictions_csv"] = str(_cache_files(config.output_dir)["predictions"] or "")
        paths["embeddings_npy"] = str(_cache_files(config.output_dir)["embeddings"] or "")
        paths["graphsage_training_metadata_json"] = str(_cache_files(config.output_dir)["metadata"] or "")
        training_result.metadata.update({"loaded_from_cache": True})

    embedding_norm_map = {
        node_id: float(score)
        for node_id, score in zip(
            node_ids,
            _normalize_array(np.linalg.norm(embeddings, axis=1)).tolist(),
            strict=True,
        )
    }
    component_score_maps = {
        "shortfall_gain": {
            row.node_id: float(row.shortfall_gain)
            for row in label_frame[["node_id", "shortfall_gain"]].itertuples(index=False)
        },
        "community_diversity_score": {
            row.node_id: float(row.community_diversity_score)
            for row in label_frame[["node_id", "community_diversity_score"]].itertuples(index=False)
        },
        "spread_proxy_score": {
            row.node_id: float(row.spread_proxy_score)
            for row in label_frame[["node_id", "spread_proxy_score"]].itertuples(index=False)
        },
        "graphsage_embedding_norm": embedding_norm_map,
    }
    return GraphSAGEFairScorerResult(
        training_result=training_result,
        feature_frame=feature_metadata,
        label_frame=label_frame,
        prediction_frame=prediction_frame,
        training_history=history_frame,
        metrics={str(key): float(value) for key, value in metrics.items()},
        effective_score_weight=float(effective_weight),
        warnings=warnings,
        inactive_components=inactive,
        output_paths=paths,
        component_score_maps=component_score_maps,
        label_generation_runtime_seconds=float(label_runtime),
    )
