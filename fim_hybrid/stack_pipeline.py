"""Shared ML-ready stack adapters for unified FIM comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import pandas as pd

from .clustering import ClusteringFrameworkError, ClusteringResult, cluster_nodes
from .community_detection import CommunityDetectionResult
from .data_loader import LoadedDataset, ProtectedGroupReport
from .embeddings.base import validate_embedding_frame
from .embeddings.benchmark import run_embedding_method
from .feature_extraction import compute_node_features
from .label_generation import NodeUtilityLabelResult, generate_singleton_node_utility_labels
from .ml_training import RankingTrainingResult, train_ranking_model as train_backend_ranking_model
from .ris_guidance import RISConfig, RISGuidanceResult, generate_ris_guidance
from .safe_math import safe_minmax_normalize


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _normalize_score_map(scores: Mapping[Any, float]) -> dict[Any, float]:
    if not scores:
        return {}
    ordered_nodes = sorted(scores, key=_sort_key)
    normalized = safe_minmax_normalize(
        [float(scores[node_id]) for node_id in ordered_nodes],
        default=0.0,
        context="stack score normalization",
    )
    return {
        node_id: float(score)
        for node_id, score in zip(ordered_nodes, normalized, strict=True)
    }


@dataclass(slots=True)
class StackEmbeddingArtifact:
    method_name: str
    embedding_frame: pd.DataFrame | None
    output_paths: dict[str, str]
    loaded_from_cache: bool
    notes: str = ""


@dataclass(slots=True)
class StackClusteringArtifact:
    method_name: str
    clustering_result: ClusteringResult | None
    assignments_csv_path: Path | None = None
    cluster_sizes_csv_path: Path | None = None
    notes: str = ""


@dataclass(slots=True)
class StackRISArtifact:
    ris_result: RISGuidanceResult
    global_scores: dict[Any, float]
    fair_scores: dict[Any, float]
    scores_csv_path: Path | None = None
    notes: str = ""


@dataclass(slots=True)
class StackRankingArtifact:
    training_result: RankingTrainingResult
    node_scores_csv_path: Path | None = None
    notes: str = ""


def _dataset_artifact_dir(output_dir: Path | str | None, dataset_name: str, subdir: str) -> Path | None:
    if output_dir is None:
        return None
    path = Path(output_dir) / dataset_name / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _attribute_artifact_dir(
    output_dir: Path | str | None,
    dataset_name: str,
    protected_attribute: str,
) -> Path | None:
    if output_dir is None:
        return None
    safe_attribute = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in protected_attribute.strip()
    ).strip("._-") or "protected_attribute"
    path = Path(output_dir) / dataset_name / safe_attribute
    path.mkdir(parents=True, exist_ok=True)
    return path


def _embedding_pickle_path(output_dir: Path | str | None, dataset_name: str, method_name: str) -> Path | None:
    dataset_dir = _dataset_artifact_dir(output_dir, dataset_name, "embeddings")
    if dataset_dir is None:
        return None
    return dataset_dir / f"{dataset_name}_{method_name}_embeddings.pkl"


def _score_table_path(
    output_dir: Path | str | None,
    dataset_name: str,
    protected_attribute: str,
    stack_name: str,
    suffix: str,
) -> Path | None:
    attribute_dir = _attribute_artifact_dir(output_dir, dataset_name, protected_attribute)
    if attribute_dir is None:
        return None
    return attribute_dir / f"{dataset_name}_{stack_name}_{suffix}.csv"


def _structural_clustering_features(dataset: LoadedDataset) -> pd.DataFrame:
    """Build structural-only fallback features for optional embedding-space clustering."""

    graph = dataset.graph
    work_graph = graph if not graph.is_directed() else graph.to_undirected()
    node_order = tuple(sorted(graph.nodes(), key=_sort_key))
    node_count = max(1, int(graph.number_of_nodes()))
    max_degree = max((float(work_graph.degree(node_id)) for node_id in node_order), default=1.0)
    pagerank = nx.pagerank(work_graph) if work_graph.number_of_nodes() else {}
    clustering = nx.clustering(work_graph)
    core_numbers = (
        {node_id: float(value) for node_id, value in nx.core_number(work_graph).items()}
        if work_graph.number_of_edges() > 0
        else {node_id: 0.0 for node_id in node_order}
    )
    rows: list[dict[str, object]] = []
    for node_id in node_order:
        degree = float(work_graph.degree(node_id))
        rows.append(
            {
                "node_id": node_id,
                "degree": degree,
                "normalized_degree": degree / float(max(max_degree, 1.0)),
                "pagerank": float(pagerank.get(node_id, 0.0)),
                "clustering_coefficient": float(clustering.get(node_id, 0.0)),
                "core_number": float(core_numbers.get(node_id, 0.0)),
                "graph_density": float(nx.density(work_graph)) if node_count > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def prepare_embedding_frame(
    dataset: LoadedDataset,
    method_name: str,
    *,
    output_dir: Path | str | None = None,
    random_seed: int = 42,
    method_config: Mapping[str, Any] | None = None,
    export_formats: Sequence[str] = ("csv", "pickle"),
    allow_cache: bool = True,
) -> StackEmbeddingArtifact:
    """Load or compute one shared embedding frame for a stack."""

    normalized_method = str(method_name).strip().lower()
    if normalized_method in {"", "none", "off"}:
        return StackEmbeddingArtifact(
            method_name="none",
            embedding_frame=None,
            output_paths={},
            loaded_from_cache=False,
            notes="embedding=none",
        )

    config = {"random_seed": int(random_seed), **dict(method_config or {})}
    pickle_path = _embedding_pickle_path(output_dir, dataset.name, normalized_method)
    if allow_cache and pickle_path is not None and pickle_path.is_file():
        frame = pd.read_pickle(pickle_path)
        validated = validate_embedding_frame(
            tuple(sorted(dataset.graph.nodes(), key=_sort_key)),
            frame,
            method_name=normalized_method,
        )
        return StackEmbeddingArtifact(
            method_name=normalized_method,
            embedding_frame=validated,
            output_paths={"pickle": str(pickle_path)},
            loaded_from_cache=True,
            notes="embedding_cache=hit",
        )

    result = run_embedding_method(
        dataset,
        normalized_method,
        config=config,
        output_dir=output_dir,
        export_formats=export_formats,
    )
    output_paths = {
        str(key): str(value)
        for key, value in dict(result.metadata.get("export_paths", {})).items()
    }
    return StackEmbeddingArtifact(
        method_name=normalized_method,
        embedding_frame=result.embedding_frame.copy(),
        output_paths=output_paths,
        loaded_from_cache=False,
        notes="embedding_cache=miss",
    )


def prepare_optional_clustering(
    dataset: LoadedDataset,
    method_name: str,
    *,
    input_mode: str = "auto",
    embeddings: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
    labels: pd.Series | Sequence[Any] | Mapping[Any, Any] | None = None,
    output_dir: Path | str | None = None,
    random_seed: int = 42,
    use_structural_fallback: bool = True,
) -> StackClusteringArtifact:
    """Compute one optional clustering artifact and persist assignments when requested."""

    normalized_method = str(method_name).strip().lower()
    if normalized_method in {"", "none", "off"}:
        return StackClusteringArtifact(method_name="none", clustering_result=None, notes="clustering=none")

    resolved_features = features
    notes = []
    if embeddings is None and resolved_features is None and bool(use_structural_fallback):
        resolved_features = _structural_clustering_features(dataset)
        notes.append("clustering_features=structural_fallback")
    try:
        result = cluster_nodes(
            dataset.graph,
            normalized_method,
            embeddings=embeddings,
            features=resolved_features,
            labels=labels,
            config=dict(config or {}),
            random_seed=int(random_seed),
            input_mode=input_mode,
            dataset=None if resolved_features is not None else dataset,
        )
    except ClusteringFrameworkError as exc:
        notes.append(f"clustering_skipped={type(exc).__name__}: {exc}")
        return StackClusteringArtifact(
            method_name=normalized_method,
            clustering_result=None,
            notes="; ".join(notes),
        )
    dataset_dir = _dataset_artifact_dir(output_dir, dataset.name, "clustering")
    assignments_csv_path = None
    cluster_sizes_csv_path = None
    if dataset_dir is not None:
        assignments_csv_path = dataset_dir / f"{dataset.name}_{normalized_method}_cluster_assignments.csv"
        cluster_sizes_csv_path = dataset_dir / f"{dataset.name}_{normalized_method}_cluster_sizes.csv"
        result.assignment_frame.to_csv(assignments_csv_path, index=False)
        pd.DataFrame(
            [
                {"cluster_id": int(cluster_id), "size": int(size)}
                for cluster_id, size in sorted(result.cluster_sizes.items())
            ]
        ).to_csv(cluster_sizes_csv_path, index=False)
    return StackClusteringArtifact(
        method_name=normalized_method,
        clustering_result=result,
        assignments_csv_path=assignments_csv_path,
        cluster_sizes_csv_path=cluster_sizes_csv_path,
        notes="; ".join([*notes, f"clustering_input_mode={result.resolved_input_mode}"]),
    )


def _ris_group_weights(
    feature_frame: pd.DataFrame,
    protected_group_report: ProtectedGroupReport,
    mode: str = "weak_group_weighted",
) -> dict[str, float]:
    normalized_mode = str(mode or "weak_group_weighted").strip().lower()
    if normalized_mode in {"standard", "global"}:
        return {group_name: 1.0 for group_name in protected_group_report.group_sizes}
    if normalized_mode == "group_balanced":
        positive_sizes = [
            int(size)
            for size in protected_group_report.group_sizes.values()
            if int(size) > 0
        ]
        largest = max(positive_sizes, default=1)
        return {
            group_name: float(largest) / float(max(1, int(group_size)))
            for group_name, group_size in protected_group_report.group_sizes.items()
        }
    if normalized_mode != "weak_group_weighted":
        raise ValueError("ris_mode must be one of ['standard', 'global', 'weak_group_weighted', 'group_balanced'].")
    required_columns = {
        "protected_group",
        "fraction_neighbors_in_undercovered_groups",
        "inverse_group_size",
        "minority_group_indicator",
    }
    missing = required_columns.difference(feature_frame.columns)
    if missing:
        raise ValueError(f"feature_frame is missing RIS group-weight columns: {sorted(missing)}.")
    group_frame = (
        feature_frame
        .groupby("protected_group", sort=True)
        .agg(
            avg_undercovered_neighbors=("fraction_neighbors_in_undercovered_groups", "mean"),
            avg_inverse_group_size=("inverse_group_size", "mean"),
            avg_minority_indicator=("minority_group_indicator", "mean"),
        )
        .reset_index()
    )
    group_frame["raw_weight"] = (
        0.45 * group_frame["avg_undercovered_neighbors"].astype(float)
        + 0.40 * group_frame["avg_inverse_group_size"].astype(float)
        + 0.15 * group_frame["avg_minority_indicator"].astype(float)
    )
    normalized_weights = _normalize_score_map(
        {
            row.protected_group: float(row.raw_weight)
            for row in group_frame.itertuples(index=False)
        }
    )
    return {
        group_name: 1.0 + float(normalized_weights.get(group_name, 0.0))
        for group_name in protected_group_report.group_sizes
    }


def prepare_ris_guidance(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    *,
    propagation_probability: float,
    feature_frame: pd.DataFrame,
    output_dir: Path | str | None = None,
    protected_attribute: str | None = None,
    stack_name: str | None = None,
    config: RISConfig | None = None,
) -> StackRISArtifact:
    """Compute RIS guidance and save node-level summaries when requested."""

    resolved_config = RISConfig() if config is None else config
    ris_result = generate_ris_guidance(
        dataset=dataset,
        protected_group_report=protected_group_report,
        propagation_probability=propagation_probability,
        config=resolved_config,
    )
    global_scores = dict(ris_result.global_node_scores)
    fair_scores = ris_result.weighted_node_scores(
        _ris_group_weights(feature_frame, protected_group_report, resolved_config.mode)
    )
    scores_csv_path = None
    if protected_attribute is not None and stack_name is not None:
        scores_csv_path = _score_table_path(
            output_dir,
            dataset.name,
            protected_attribute,
            stack_name,
            "ris_scores",
        )
        if scores_csv_path is not None:
            pd.DataFrame(
                {
                    "node_id": list(sorted(global_scores, key=_sort_key)),
                    "ris_global_score": [
                        float(global_scores[node_id])
                        for node_id in sorted(global_scores, key=_sort_key)
                    ],
                    "fair_ris_score": [
                        float(fair_scores[node_id])
                        for node_id in sorted(global_scores, key=_sort_key)
                    ],
                }
            ).to_csv(scores_csv_path, index=False)
    return StackRISArtifact(
        ris_result=ris_result,
        global_scores=global_scores,
        fair_scores=fair_scores,
        scores_csv_path=scores_csv_path,
        notes=f"ris_num_rr_sets={resolved_config.num_rr_sets}; ris_mode={resolved_config.mode}",
    )


def build_ranking_feature_frame(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    community_result: CommunityDetectionResult,
    *,
    embedding_frame: pd.DataFrame | None = None,
    clustering_result: ClusteringResult | None = None,
    ris_scores: Mapping[Any, float] | None = None,
    fair_ris_scores: Mapping[Any, float] | None = None,
    extra_score_maps: Mapping[str, Mapping[Any, float]] | None = None,
    use_community_features_for_ml: bool = True,
    community_feature_mode: str = "basic",
    use_clustering_features: bool = False,
) -> pd.DataFrame:
    """Build the shared feature table consumed by tabular and GNN rankers."""

    feature_frame = compute_node_features(
        dataset=dataset,
        protected_group_report=protected_group_report,
        community_result=community_result,
        use_community_features_for_ml=use_community_features_for_ml,
        community_feature_mode=community_feature_mode,
    ).reset_index(drop=True)
    merged = feature_frame.copy()

    if embedding_frame is not None:
        merged = merged.merge(
            embedding_frame,
            on="node_id",
            how="left",
            validate="one_to_one",
        )
    if clustering_result is not None and bool(use_clustering_features):
        assignment_frame = clustering_result.assignment_frame.rename(
            columns={"cluster_id": "clustering_cluster_id"}
        )
        merged = merged.merge(
            assignment_frame,
            on="node_id",
            how="left",
            validate="one_to_one",
        )
        cluster_size_by_node = {
            row.node_id: int(clustering_result.cluster_sizes[int(row.clustering_cluster_id)])
            for row in assignment_frame.itertuples(index=False)
        }
        merged["clustering_cluster_size"] = [
            float(cluster_size_by_node[row.node_id])
            for row in merged.itertuples(index=False)
        ]
    if ris_scores is not None:
        merged["ris_score"] = [
            float(ris_scores.get(node_id, 0.0))
            for node_id in merged["node_id"].tolist()
        ]
    if fair_ris_scores is not None:
        merged["fair_ris_score"] = [
            float(fair_ris_scores.get(node_id, 0.0))
            for node_id in merged["node_id"].tolist()
        ]
    for column_name, score_map in dict(extra_score_maps or {}).items():
        merged[column_name] = [
            float(score_map.get(node_id, 0.0))
            for node_id in merged["node_id"].tolist()
        ]

    if merged.isna().any().any():
        missing_columns = [
            column_name
            for column_name in merged.columns
            if merged[column_name].isna().any()
        ]
        raise ValueError(f"ranking feature frame contains missing values: {sorted(missing_columns)}.")
    return merged


def build_stack_label_frame(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    *,
    propagation_probability: float,
    mc_runs: int,
    lambda_weight: float,
    random_seed: int,
) -> NodeUtilityLabelResult:
    """Build the canonical singleton label frame used by ML-guided stacks."""

    return generate_singleton_node_utility_labels(
        dataset=dataset,
        protected_group_report=protected_group_report,
        propagation_probability=propagation_probability,
        mc_runs=mc_runs,
        lambda_weight=lambda_weight,
        random_seed=random_seed,
    )


def train_ranking_model(
    dataset: LoadedDataset,
    *,
    feature_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    budget: int,
    model_type: str,
    target_column: str = "label_score",
    protected_attribute: str | None = None,
    stack_name: str | None = None,
    output_dir: Path | str | None = None,
    random_seed: int = 42,
    top_fraction: float | None = None,
    top_n: int | None = None,
    max_nodes: int | None = None,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-4,
    epochs: int = 100,
    cache_path: Path | None = None,
    node2vec_mode: str = "off",
    node2vec_config: dict[str, Any] | None = None,
    debias_mode: str = "none",
    protected_attribute_column: str | None = "protected_group",
    focal_gamma: float = 2.0,
    group_robust_weight: float = 0.25,
    worst_group_boost_factor: float = 2.0,
    allow_protected_features_in_ml: bool = False,
) -> StackRankingArtifact:
    """Train one ranking model and persist its node score table."""

    result = train_backend_ranking_model(
        dataset=dataset,
        feature_frame=feature_frame,
        label_frame=label_frame,
        budget=budget,
        model_type=model_type,
        target_column=target_column,
        top_fraction=top_fraction,
        top_n=top_n,
        max_nodes=max_nodes,
        random_seed=random_seed,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        cache_path=cache_path,
        node2vec_mode=node2vec_mode,
        node2vec_config=node2vec_config,
        debias_mode=debias_mode,
        protected_attribute_column=protected_attribute_column,
        focal_gamma=focal_gamma,
        group_robust_weight=group_robust_weight,
        worst_group_boost_factor=worst_group_boost_factor,
        allow_protected_features_in_ml=bool(allow_protected_features_in_ml),
    )
    node_scores_csv_path = _score_table_path(
        output_dir,
        dataset.name,
        protected_attribute if protected_attribute is not None else "protected_attribute",
        stack_name if stack_name is not None else model_type,
        "node_scores",
    ) if protected_attribute is not None and stack_name is not None else None
    if node_scores_csv_path is not None:
        ordered_nodes = list(sorted(result.predicted_scores, key=_sort_key))
        pd.DataFrame(
            {
                "node_id": ordered_nodes,
                "predicted_score": [float(result.predicted_scores[node_id]) for node_id in ordered_nodes],
                "ranking_position": list(range(1, len(ordered_nodes) + 1)),
            }
        ).to_csv(node_scores_csv_path, index=False)
    return StackRankingArtifact(
        training_result=result,
        node_scores_csv_path=node_scores_csv_path,
        notes=(
            f"backend={result.backend}; target_column={result.target_column}; "
            f"debias_mode={result.debias_mode}; "
            f"allow_protected_features_in_ml={bool(allow_protected_features_in_ml)}"
        ),
    )
