"""Shared clustering and community-detection helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any
import random as py_random

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

try:
    import igraph as ig
except ImportError:
    ig = None

try:
    import hdbscan
except ImportError:
    hdbscan = None


GRAPH_NATIVE_CLUSTERING_METHODS = (
    "louvain",
    "leiden",
    "multilevel",
    "infomap",
    "label_propagation",
    "walktrap",
)
EMBEDDING_SPACE_CLUSTERING_METHODS = (
    "kmeans",
    "spectral",
    "agglomerative",
    "dbscan_or_hdbscan",
    "gmm",
)
SUPPORTED_CLUSTERING_METHODS = GRAPH_NATIVE_CLUSTERING_METHODS + EMBEDDING_SPACE_CLUSTERING_METHODS
SUPPORTED_CLUSTERING_INPUT_MODES = ("graph", "embedding", "auto")
CLUSTERING_SUMMARY_COLUMNS = [
    "dataset",
    "method",
    "category",
    "status",
    "requested_input_mode",
    "resolved_input_mode",
    "runtime_seconds",
    "num_clusters",
    "largest_cluster_size",
    "smallest_cluster_size",
    "average_cluster_size",
    "cluster_size_std",
    "cluster_size_summary",
    "modularity",
    "mean_conductance",
    "silhouette_score",
    "davies_bouldin_score",
    "calinski_harabasz_score",
    "nmi",
    "ari",
    "assignments_csv_path",
    "cluster_sizes_csv_path",
    "skip_reason",
]


class ClusteringFrameworkError(ValueError):
    """Base error for clustering-framework failures."""


class UnsupportedClusteringMethodError(ClusteringFrameworkError):
    """Raised when the requested clustering method is unsupported."""


class ClusteringDependencyError(ClusteringFrameworkError):
    """Raised when a clustering method requires an unavailable dependency."""


class ClusteringInputError(ClusteringFrameworkError):
    """Raised when the requested clustering input cannot be resolved."""


@dataclass(frozen=True, slots=True)
class ClusteringMethodSpec:
    """Static metadata for one clustering method."""

    name: str
    category: str
    requires_fixed_cluster_count: bool
    preferred_input_mode: str
    requires_igraph: bool = False


@dataclass(slots=True)
class ClusteringResult:
    """Normalized clustering output with additive diagnostics."""

    method: str
    category: str
    requested_input_mode: str
    resolved_input_mode: str
    assignment_frame: pd.DataFrame
    cluster_id_by_node: dict[Any, int]
    clusters: dict[int, tuple[Any, ...]]
    cluster_sizes: dict[int, int]
    num_clusters: int
    runtime_seconds: float
    modularity: float | None = None
    mean_conductance: float | None = None
    silhouette_score: float | None = None
    davies_bouldin_score: float | None = None
    calinski_harabasz_score: float | None = None
    nmi: float | None = None
    ari: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClusteringBenchmarkResult:
    """Batch clustering benchmark output."""

    dataset_name: str
    summary_frame: pd.DataFrame
    results_by_method: dict[str, ClusteringResult]
    output_dir: Path | None = None


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _sorted_node_ids(graph: nx.Graph) -> tuple[Any, ...]:
    return tuple(sorted(graph.nodes(), key=_sort_key))


def _normalize_input_mode(method_spec: ClusteringMethodSpec, input_mode: str) -> str:
    normalized = str(input_mode).strip().lower()
    if normalized not in SUPPORTED_CLUSTERING_INPUT_MODES:
        raise ValueError(
            f"Unsupported clustering_input_mode '{input_mode}'. "
            f"Supported values: {list(SUPPORTED_CLUSTERING_INPUT_MODES)}."
        )
    if method_spec.category == "graph_native":
        return "graph"
    return normalized


def _normalize_clusters(
    raw_clusters: Sequence[Sequence[Any]] | Mapping[int, Sequence[Any]],
) -> tuple[dict[int, tuple[Any, ...]], dict[Any, int]]:
    if isinstance(raw_clusters, Mapping):
        cluster_nodes = [tuple(nodes) for _, nodes in sorted(raw_clusters.items())]
    else:
        cluster_nodes = [tuple(nodes) for nodes in raw_clusters]
    normalized_groups = [
        tuple(sorted(tuple(nodes), key=_sort_key))
        for nodes in cluster_nodes
        if len(tuple(nodes)) > 0
    ]
    normalized_groups = sorted(
        normalized_groups,
        key=lambda nodes: (_sort_key(nodes[0]), len(nodes), tuple(_sort_key(node) for node in nodes)),
    )
    clusters = {cluster_id: nodes for cluster_id, nodes in enumerate(normalized_groups)}
    cluster_id_by_node: dict[Any, int] = {}
    for cluster_id, nodes in clusters.items():
        for node_id in nodes:
            cluster_id_by_node[node_id] = cluster_id
    return clusters, cluster_id_by_node


def _normalize_labels_from_assignments(
    node_order: Sequence[Any],
    labels: Sequence[int] | np.ndarray,
) -> tuple[dict[int, tuple[Any, ...]], dict[Any, int]]:
    if len(node_order) != len(labels):
        raise ClusteringInputError(
            "Predicted cluster labels must align with the graph node ordering."
        )
    cluster_nodes: dict[int, list[Any]] = {}
    label_remap: dict[int, int] = {}
    next_label = 0
    for node_id, raw_label in zip(node_order, labels, strict=True):
        source_label = int(raw_label)
        if source_label not in label_remap:
            label_remap[source_label] = next_label
            next_label += 1
        normalized_label = label_remap[source_label]
        cluster_nodes.setdefault(normalized_label, []).append(node_id)
    clusters, _ = _normalize_clusters(cluster_nodes)
    cluster_id_by_node: dict[Any, int] = {}
    for cluster_id, nodes in clusters.items():
        for node_id in nodes:
            cluster_id_by_node[node_id] = cluster_id
    return clusters, cluster_id_by_node


def _cluster_size_summary(cluster_sizes: Mapping[int, int], *, max_items: int = 6) -> str:
    ordered = sorted(cluster_sizes.items(), key=lambda item: (-int(item[1]), int(item[0])))
    summary = [f"{cluster_id}:{size}" for cluster_id, size in ordered[:max_items]]
    if len(ordered) > max_items:
        summary.append("...")
    return ",".join(summary)


def _igraph_required(method_name: str) -> "ig.Graph":
    if ig is None:
        raise ClusteringDependencyError(
            f"Clustering method '{method_name}' requires the 'igraph' package, but it is not installed."
        )
    return ig


def _to_igraph(
    graph: nx.Graph,
    weight_attribute: str | None = None,
) -> tuple["ig.Graph", list[Any], list[float] | None]:
    ig_module = _igraph_required("igraph")
    node_order = list(_sorted_node_ids(graph))
    node_to_index = {node_id: index for index, node_id in enumerate(node_order)}
    work_graph = graph if not graph.is_directed() else graph.to_undirected()
    edges = [(node_to_index[source], node_to_index[target]) for source, target in work_graph.edges()]
    weights: list[float] | None = None
    if weight_attribute is not None:
        weights = [
            float(work_graph[source][target].get(weight_attribute, 1.0))
            for source, target in work_graph.edges()
        ]
    ig_graph = ig_module.Graph(n=len(node_order), edges=edges, directed=work_graph.is_directed())
    return ig_graph, node_order, weights


def available_clustering_methods() -> tuple[str, ...]:
    """Return the supported clustering methods."""

    return SUPPORTED_CLUSTERING_METHODS


def clustering_summary_columns() -> list[str]:
    """Return the normalized clustering benchmark columns."""

    return list(CLUSTERING_SUMMARY_COLUMNS)


def _method_registry() -> dict[str, ClusteringMethodSpec]:
    return {
        "louvain": ClusteringMethodSpec("louvain", "graph_native", False, "graph"),
        "leiden": ClusteringMethodSpec("leiden", "graph_native", False, "graph", requires_igraph=True),
        "multilevel": ClusteringMethodSpec("multilevel", "graph_native", False, "graph", requires_igraph=True),
        "infomap": ClusteringMethodSpec("infomap", "graph_native", False, "graph", requires_igraph=True),
        "label_propagation": ClusteringMethodSpec("label_propagation", "graph_native", False, "graph"),
        "walktrap": ClusteringMethodSpec("walktrap", "graph_native", False, "graph", requires_igraph=True),
        "kmeans": ClusteringMethodSpec("kmeans", "embedding_space", True, "embedding"),
        "spectral": ClusteringMethodSpec("spectral", "embedding_space", True, "embedding"),
        "agglomerative": ClusteringMethodSpec("agglomerative", "embedding_space", True, "embedding"),
        "dbscan_or_hdbscan": ClusteringMethodSpec("dbscan_or_hdbscan", "embedding_space", False, "embedding"),
        "gmm": ClusteringMethodSpec("gmm", "embedding_space", True, "embedding"),
    }


def get_clustering_method_spec(method: str) -> ClusteringMethodSpec:
    """Return the static method metadata for one clustering method."""

    method_key = str(method).strip().lower()
    registry = _method_registry()
    if method_key not in registry:
        raise UnsupportedClusteringMethodError(
            f"Unsupported clustering method '{method}'. Supported methods: {list(registry)}."
        )
    return registry[method_key]


def _aligned_label_array(
    node_order: Sequence[Any],
    labels: pd.Series | Sequence[Any] | Mapping[Any, Any] | None,
) -> np.ndarray | None:
    if labels is None:
        return None
    if isinstance(labels, pd.Series):
        series = labels
        if "node_id" in series.index.names:
            series = series.reindex(list(node_order))
        elif list(series.index) == list(node_order):
            pass
        elif len(series) == len(node_order):
            series = pd.Series(series.to_list(), index=list(node_order))
        else:
            missing = [node_id for node_id in node_order if node_id not in series.index]
            if missing:
                raise ClusteringInputError(f"Labels are missing graph nodes: {missing[:5]}.")
            series = series.reindex(list(node_order))
        return series.astype(str).to_numpy(dtype=object)
    if isinstance(labels, Mapping):
        missing = [node_id for node_id in node_order if node_id not in labels]
        if missing:
            raise ClusteringInputError(f"Labels are missing graph nodes: {missing[:5]}.")
        return np.asarray([str(labels[node_id]) for node_id in node_order], dtype=object)
    label_array = np.asarray(labels, dtype=object)
    if label_array.shape[0] != len(node_order):
        raise ClusteringInputError("Labels must align with the graph node ordering.")
    return label_array.astype(str)


def _resolve_fixed_cluster_count(
    method: str,
    *,
    config: Mapping[str, Any],
    labels: np.ndarray | None,
) -> int:
    candidates = [
        config.get("method_n_clusters"),
        config.get("n_clusters"),
        config.get("num_clusters"),
    ]
    for value in candidates:
        if value is None:
            continue
        resolved = int(value)
        if resolved < 1:
            raise ClusteringInputError("n_clusters must be at least 1 when provided.")
        return resolved
    if labels is not None:
        unique_count = int(len(pd.unique(pd.Series(labels))))
        if unique_count >= 1:
            return unique_count
    raise ClusteringInputError(
        f"Clustering method '{method}' requires n_clusters. "
        "Provide it explicitly or supply labels so the class count can be reused."
    )


def _resolve_embedding_frame(
    graph: nx.Graph,
    data: pd.DataFrame | np.ndarray | Sequence[Sequence[float]] | None,
) -> tuple[pd.DataFrame, np.ndarray, str] | None:
    if data is None:
        return None
    node_order = list(_sorted_node_ids(graph))
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        if "node_id" not in frame.columns:
            raise ClusteringInputError("Embedding frames must contain a node_id column.")
        if frame["node_id"].duplicated().any():
            raise ClusteringInputError("Embedding frames must not contain duplicate node_id values.")
        missing = [node_id for node_id in node_order if node_id not in set(frame["node_id"])]
        if missing:
            raise ClusteringInputError(f"Embedding frames are missing graph nodes: {missing[:5]}.")
        aligned = frame.set_index("node_id", drop=False).reindex(node_order).reset_index(drop=True)
        value_columns = [column for column in aligned.columns if column != "node_id"]
        if not value_columns:
            raise ClusteringInputError("Embedding frames must contain at least one feature column.")
        numeric = aligned.loc[:, value_columns].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            raise ClusteringInputError("Embedding frames must contain only numeric values.")
        return aligned.loc[:, ["node_id", *value_columns]], numeric.to_numpy(dtype=float, copy=True), "embedding"
    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] != len(node_order):
        raise ClusteringInputError(
            "Embedding arrays must have shape [num_nodes, num_embedding_dimensions]."
        )
    frame = pd.DataFrame({"node_id": node_order})
    for feature_index in range(array.shape[1]):
        frame[f"value_{feature_index}"] = array[:, feature_index]
    return frame, array.astype(float, copy=True), "embedding"


def _resolve_feature_frame(
    graph: nx.Graph,
    features: pd.DataFrame | np.ndarray | Sequence[Sequence[float]] | None,
) -> tuple[pd.DataFrame, np.ndarray, str] | None:
    if features is None:
        return None
    node_order = list(_sorted_node_ids(graph))
    if isinstance(features, pd.DataFrame):
        frame = features.copy()
        if "node_id" not in frame.columns:
            if len(frame) != len(node_order):
                raise ClusteringInputError(
                    "Feature frames without node_id must already match the graph node ordering."
                )
            frame.insert(0, "node_id", node_order)
        if frame["node_id"].duplicated().any():
            raise ClusteringInputError("Feature frames must not contain duplicate node_id values.")
        missing = [node_id for node_id in node_order if node_id not in set(frame["node_id"])]
        if missing:
            raise ClusteringInputError(f"Feature frames are missing graph nodes: {missing[:5]}.")
        aligned = frame.set_index("node_id", drop=False).reindex(node_order).reset_index(drop=True)
        value_columns = [column for column in aligned.columns if column != "node_id"]
        if not value_columns:
            raise ClusteringInputError("Feature frames must contain at least one numeric column.")
        numeric = aligned.loc[:, value_columns].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            raise ClusteringInputError("Feature frames must contain only numeric values.")
        return aligned.loc[:, ["node_id", *value_columns]], numeric.to_numpy(dtype=float, copy=True), "feature"
    array = np.asarray(features, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] != len(node_order):
        raise ClusteringInputError("Feature arrays must have shape [num_nodes, num_features].")
    frame = pd.DataFrame({"node_id": node_order})
    for feature_index in range(array.shape[1]):
        frame[f"feature_{feature_index}"] = array[:, feature_index]
    return frame, array.astype(float, copy=True), "feature"


def _resolve_embedding_input(
    graph: nx.Graph,
    *,
    requested_input_mode: str,
    embeddings: pd.DataFrame | np.ndarray | Sequence[Sequence[float]] | None,
    features: pd.DataFrame | np.ndarray | Sequence[Sequence[float]] | None,
    dataset: Any | None,
    random_seed: int,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    resolved_embeddings = _resolve_embedding_frame(graph, embeddings)
    resolved_features = _resolve_feature_frame(graph, features)

    if requested_input_mode == "embedding":
        if resolved_embeddings is not None:
            return resolved_embeddings
        if resolved_features is not None:
            return resolved_features
        if dataset is not None:
            from .embeddings.features import prepare_benchmark_features

            prepared = prepare_benchmark_features(dataset, graph=graph, random_seed=random_seed)
            return prepared.feature_frame.copy(), prepared.feature_matrix.copy(), "feature"
        raise ClusteringInputError(
            "Embedding-space clustering requires embeddings or numeric feature vectors."
        )

    if requested_input_mode == "auto":
        if resolved_embeddings is not None:
            return resolved_embeddings
        if resolved_features is not None:
            return resolved_features
        if dataset is not None:
            from .embeddings.features import prepare_benchmark_features

            prepared = prepare_benchmark_features(dataset, graph=graph, random_seed=random_seed)
            return prepared.feature_frame.copy(), prepared.feature_matrix.copy(), "feature"
        raise ClusteringInputError(
            "Auto input mode could not resolve embeddings or fallback features."
        )

    raise ClusteringInputError(
        f"Embedding-space clustering does not support requested input mode '{requested_input_mode}'."
    )


def _graph_native_clusters(
    graph: nx.Graph,
    *,
    method: str,
    random_seed: int,
    config: Mapping[str, Any],
) -> Sequence[Sequence[Any]]:
    work_graph = graph if not graph.is_directed() else graph.to_undirected()
    resolution = float(config.get("resolution", 1.0))
    weight_attribute = config.get("weight_attribute")
    method_key = str(method).strip().lower()
    if method_key == "louvain":
        return nx.community.louvain_communities(
            work_graph,
            weight=weight_attribute,
            resolution=resolution,
            seed=random_seed,
        )
    if method_key == "label_propagation":
        return list(nx.community.asyn_lpa_communities(work_graph, weight=weight_attribute, seed=random_seed))

    ig_graph, node_order, weights = _to_igraph(work_graph, weight_attribute)
    ig.set_random_number_generator(py_random.Random(random_seed))
    if method_key == "leiden":
        communities = ig_graph.community_leiden(
            objective_function="modularity",
            weights=weights,
            resolution=resolution,
        )
    elif method_key == "multilevel":
        communities = ig_graph.community_multilevel(weights=weights)
    elif method_key == "infomap":
        communities = ig_graph.community_infomap(edge_weights=weights)
    elif method_key == "walktrap":
        communities = ig_graph.community_walktrap(weights=weights).as_clustering()
    else:
        raise UnsupportedClusteringMethodError(
            f"Unsupported graph-native clustering method '{method}'."
        )
    return [[node_order[index] for index in community] for community in communities]


def _embedding_space_assignments(
    matrix: np.ndarray,
    *,
    method: str,
    random_seed: int,
    config: Mapping[str, Any],
    labels: np.ndarray | None,
) -> np.ndarray:
    method_key = str(method).strip().lower()
    if method_key in {"kmeans", "spectral", "agglomerative", "gmm"}:
        n_clusters = _resolve_fixed_cluster_count(method_key, config=config, labels=labels)
    if method_key == "kmeans":
        model = KMeans(
            n_clusters=n_clusters,
            n_init=int(config.get("n_init", 10)),
            random_state=random_seed,
        )
        return model.fit_predict(matrix)
    if method_key == "spectral":
        if n_clusters < 2:
            raise ClusteringInputError("spectral clustering requires n_clusters >= 2.")
        n_neighbors = int(config.get("n_neighbors", min(10, max(1, matrix.shape[0] - 1))))
        model = SpectralClustering(
            n_clusters=n_clusters,
            random_state=random_seed,
            assign_labels=str(config.get("assign_labels", "kmeans")),
            affinity=str(config.get("affinity", "nearest_neighbors")),
            n_neighbors=n_neighbors,
        )
        return model.fit_predict(matrix)
    if method_key == "agglomerative":
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=str(config.get("linkage", "ward")),
        )
        return model.fit_predict(matrix)
    if method_key == "dbscan_or_hdbscan":
        min_cluster_size = int(config.get("min_cluster_size", config.get("min_samples", 5)))
        if hdbscan is not None:
            model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=int(config.get("min_samples", min_cluster_size)),
            )
            return model.fit_predict(matrix)
        model = DBSCAN(
            eps=float(config.get("eps", 0.5)),
            min_samples=int(config.get("min_samples", min_cluster_size)),
            metric=str(config.get("metric", "euclidean")),
        )
        return model.fit_predict(matrix)
    if method_key == "gmm":
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type=str(config.get("covariance_type", "full")),
            random_state=random_seed,
        )
        return model.fit_predict(matrix)
    raise UnsupportedClusteringMethodError(
        f"Unsupported embedding-space clustering method '{method}'."
    )


def _mean_conductance(
    graph: nx.Graph,
    clusters: Mapping[int, Sequence[Any]],
) -> float | None:
    work_graph = graph if not graph.is_directed() else graph.to_undirected()
    if work_graph.number_of_edges() == 0 or len(clusters) < 2:
        return None
    scores: list[float] = []
    for nodes in clusters.values():
        node_set = set(nodes)
        if not node_set or len(node_set) == work_graph.number_of_nodes():
            continue
        try:
            scores.append(float(nx.conductance(work_graph, node_set)))
        except nx.NetworkXError:
            continue
    if not scores:
        return None
    return float(np.mean(scores))


def _modularity(
    graph: nx.Graph,
    clusters: Mapping[int, Sequence[Any]],
) -> float | None:
    work_graph = graph if not graph.is_directed() else graph.to_undirected()
    if work_graph.number_of_edges() == 0 or not clusters:
        return None
    return float(nx.community.modularity(work_graph, [set(nodes) for nodes in clusters.values()]))


def _embedding_quality_metrics(
    matrix: np.ndarray,
    cluster_id_by_node: Mapping[Any, int],
    node_order: Sequence[Any],
) -> tuple[float | None, float | None, float | None]:
    labels = np.asarray([cluster_id_by_node[node_id] for node_id in node_order], dtype=int)
    unique_labels = np.unique(labels)
    if matrix.shape[0] < 3 or len(unique_labels) < 2 or len(unique_labels) >= matrix.shape[0]:
        return None, None, None
    silhouette = float(silhouette_score(matrix, labels))
    davies_bouldin = float(davies_bouldin_score(matrix, labels))
    calinski_harabasz = float(calinski_harabasz_score(matrix, labels))
    return silhouette, davies_bouldin, calinski_harabasz


def cluster_nodes(
    graph: nx.Graph,
    method: str,
    *,
    embeddings: pd.DataFrame | np.ndarray | Sequence[Sequence[float]] | None = None,
    features: pd.DataFrame | np.ndarray | Sequence[Sequence[float]] | None = None,
    labels: pd.Series | Sequence[Any] | Mapping[Any, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    random_seed: int = 42,
    input_mode: str = "auto",
    dataset: Any | None = None,
) -> ClusteringResult:
    """Cluster graph nodes with a graph-native or embedding-space method."""

    if graph.number_of_nodes() == 0:
        raise ClusteringInputError("Clustering requires a graph with at least one node.")

    method_spec = get_clustering_method_spec(method)
    requested_input_mode = _normalize_input_mode(method_spec, input_mode)
    method_config = dict(config or {})
    node_order = _sorted_node_ids(graph)
    aligned_labels = _aligned_label_array(node_order, labels)
    start = perf_counter()

    if method_spec.category == "graph_native":
        raw_clusters = _graph_native_clusters(
            graph,
            method=method_spec.name,
            random_seed=int(random_seed),
            config=method_config,
        )
        clusters, cluster_id_by_node = _normalize_clusters(raw_clusters)
        resolved_input_mode = "graph"
        silhouette = None
        davies_bouldin = None
        calinski_harabasz = None
    else:
        _, matrix, resolved_input_mode = _resolve_embedding_input(
            graph,
            requested_input_mode=requested_input_mode,
            embeddings=embeddings,
            features=features,
            dataset=dataset,
            random_seed=int(random_seed),
        )
        predicted_labels = _embedding_space_assignments(
            matrix,
            method=method_spec.name,
            random_seed=int(random_seed),
            config=method_config,
            labels=aligned_labels,
        )
        clusters, cluster_id_by_node = _normalize_labels_from_assignments(node_order, predicted_labels)
        silhouette, davies_bouldin, calinski_harabasz = _embedding_quality_metrics(
            matrix,
            cluster_id_by_node,
            node_order,
        )

    if set(cluster_id_by_node) != set(node_order):
        missing_nodes = sorted(set(node_order) - set(cluster_id_by_node), key=_sort_key)
        raise ClusteringInputError(
            f"Clustering method '{method}' did not assign all graph nodes. Missing: {missing_nodes[:5]}."
        )

    assignment_frame = pd.DataFrame(
        {
            "node_id": list(node_order),
            "cluster_id": [int(cluster_id_by_node[node_id]) for node_id in node_order],
        }
    )
    cluster_sizes = {cluster_id: len(nodes) for cluster_id, nodes in clusters.items()}
    runtime_seconds = perf_counter() - start
    modularity = _modularity(graph, clusters)
    mean_conductance = _mean_conductance(graph, clusters)
    nmi = None
    ari = None
    if aligned_labels is not None:
        predicted = np.asarray([cluster_id_by_node[node_id] for node_id in node_order], dtype=int)
        nmi = float(normalized_mutual_info_score(aligned_labels, predicted))
        ari = float(adjusted_rand_score(aligned_labels, predicted))

    return ClusteringResult(
        method=method_spec.name,
        category=method_spec.category,
        requested_input_mode=requested_input_mode,
        resolved_input_mode=resolved_input_mode,
        assignment_frame=assignment_frame,
        cluster_id_by_node=cluster_id_by_node,
        clusters=clusters,
        cluster_sizes=cluster_sizes,
        num_clusters=len(clusters),
        runtime_seconds=runtime_seconds,
        modularity=modularity,
        mean_conductance=mean_conductance,
        silhouette_score=silhouette,
        davies_bouldin_score=davies_bouldin,
        calinski_harabasz_score=calinski_harabasz,
        nmi=nmi,
        ari=ari,
        metadata={
            "cluster_size_summary": _cluster_size_summary(cluster_sizes),
            "largest_cluster_size": max(cluster_sizes.values()) if cluster_sizes else 0,
            "smallest_cluster_size": min(cluster_sizes.values()) if cluster_sizes else 0,
            "average_cluster_size": float(np.mean(list(cluster_sizes.values()))) if cluster_sizes else 0.0,
            "cluster_size_std": float(np.std(list(cluster_sizes.values()), ddof=0)) if cluster_sizes else 0.0,
        },
    )


def _summary_row(
    dataset_name: str,
    method_name: str,
    *,
    result: ClusteringResult | None = None,
    status: str,
    assignments_csv_path: Path | None = None,
    cluster_sizes_csv_path: Path | None = None,
    skip_reason: str = "",
) -> dict[str, Any]:
    if result is None:
        return {
            "dataset": dataset_name,
            "method": method_name,
            "category": pd.NA,
            "status": status,
            "requested_input_mode": pd.NA,
            "resolved_input_mode": pd.NA,
            "runtime_seconds": 0.0,
            "num_clusters": pd.NA,
            "largest_cluster_size": pd.NA,
            "smallest_cluster_size": pd.NA,
            "average_cluster_size": pd.NA,
            "cluster_size_std": pd.NA,
            "cluster_size_summary": "",
            "modularity": pd.NA,
            "mean_conductance": pd.NA,
            "silhouette_score": pd.NA,
            "davies_bouldin_score": pd.NA,
            "calinski_harabasz_score": pd.NA,
            "nmi": pd.NA,
            "ari": pd.NA,
            "assignments_csv_path": pd.NA,
            "cluster_sizes_csv_path": pd.NA,
            "skip_reason": skip_reason,
        }
    return {
        "dataset": dataset_name,
        "method": method_name,
        "category": result.category,
        "status": status,
        "requested_input_mode": result.requested_input_mode,
        "resolved_input_mode": result.resolved_input_mode,
        "runtime_seconds": float(result.runtime_seconds),
        "num_clusters": int(result.num_clusters),
        "largest_cluster_size": int(result.metadata["largest_cluster_size"]),
        "smallest_cluster_size": int(result.metadata["smallest_cluster_size"]),
        "average_cluster_size": float(result.metadata["average_cluster_size"]),
        "cluster_size_std": float(result.metadata["cluster_size_std"]),
        "cluster_size_summary": str(result.metadata["cluster_size_summary"]),
        "modularity": pd.NA if result.modularity is None else float(result.modularity),
        "mean_conductance": pd.NA if result.mean_conductance is None else float(result.mean_conductance),
        "silhouette_score": pd.NA if result.silhouette_score is None else float(result.silhouette_score),
        "davies_bouldin_score": pd.NA if result.davies_bouldin_score is None else float(result.davies_bouldin_score),
        "calinski_harabasz_score": (
            pd.NA if result.calinski_harabasz_score is None else float(result.calinski_harabasz_score)
        ),
        "nmi": pd.NA if result.nmi is None else float(result.nmi),
        "ari": pd.NA if result.ari is None else float(result.ari),
        "assignments_csv_path": pd.NA if assignments_csv_path is None else str(assignments_csv_path),
        "cluster_sizes_csv_path": pd.NA if cluster_sizes_csv_path is None else str(cluster_sizes_csv_path),
        "skip_reason": skip_reason,
    }


def _cluster_sizes_frame(result: ClusteringResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"cluster_id": int(cluster_id), "size": int(size)}
            for cluster_id, size in sorted(result.cluster_sizes.items())
        ]
    )


def run_clustering_benchmark(
    dataset: Any,
    *,
    methods: Sequence[str] | None = None,
    embeddings: pd.DataFrame | np.ndarray | Sequence[Sequence[float]] | None = None,
    features: pd.DataFrame | np.ndarray | Sequence[Sequence[float]] | None = None,
    labels: pd.Series | Sequence[Any] | Mapping[Any, Any] | None = None,
    output_dir: Path | None = None,
    input_mode: str = "auto",
    method_configs: Mapping[str, Mapping[str, Any]] | None = None,
    random_seed: int = 42,
    continue_on_error: bool = True,
    save_cluster_assignments: bool = True,
) -> ClusteringBenchmarkResult:
    """Run a deterministic clustering benchmark over multiple methods."""

    graph = dataset.graph
    selected_methods = (
        list(available_clustering_methods())
        if methods is None
        else [str(method).strip().lower() for method in methods]
    )
    results_by_method: dict[str, ClusteringResult] = {}
    rows: list[dict[str, Any]] = []
    resolved_output_dir = Path(output_dir) if output_dir is not None else None
    benchmark_dir: Path | None = None
    if resolved_output_dir is not None:
        benchmark_dir = resolved_output_dir / dataset.name / "clustering"
        benchmark_dir.mkdir(parents=True, exist_ok=True)

    for method_name in selected_methods:
        assignments_csv_path: Path | None = None
        cluster_sizes_csv_path: Path | None = None
        try:
            result = cluster_nodes(
                graph,
                method_name,
                embeddings=embeddings,
                features=features,
                labels=labels,
                config=(method_configs or {}).get(method_name, {}),
                random_seed=int(random_seed),
                input_mode=input_mode,
                dataset=dataset,
            )
            if benchmark_dir is not None and save_cluster_assignments:
                assignments_csv_path = benchmark_dir / f"{dataset.name}_{method_name}_cluster_assignments.csv"
                cluster_sizes_csv_path = benchmark_dir / f"{dataset.name}_{method_name}_cluster_sizes.csv"
                result.assignment_frame.to_csv(assignments_csv_path, index=False)
                _cluster_sizes_frame(result).to_csv(cluster_sizes_csv_path, index=False)
            results_by_method[method_name] = result
            rows.append(
                _summary_row(
                    dataset.name,
                    method_name,
                    result=result,
                    status="ok",
                    assignments_csv_path=assignments_csv_path,
                    cluster_sizes_csv_path=cluster_sizes_csv_path,
                )
            )
        except (ClusteringDependencyError, ClusteringInputError, UnsupportedClusteringMethodError) as exc:
            rows.append(
                _summary_row(
                    dataset.name,
                    method_name,
                    status="skipped",
                    skip_reason=str(exc),
                )
            )
            if not continue_on_error:
                raise
        except Exception:
            if not continue_on_error:
                raise
            rows.append(
                _summary_row(
                    dataset.name,
                    method_name,
                    status="skipped",
                    skip_reason=f"Unexpected failure while clustering with '{method_name}'.",
                )
            )

    summary_frame = pd.DataFrame(rows)
    if not summary_frame.empty:
        summary_frame = summary_frame.loc[:, [column for column in CLUSTERING_SUMMARY_COLUMNS if column in summary_frame.columns]]
    if benchmark_dir is not None and not summary_frame.empty:
        summary_frame.to_csv(benchmark_dir / f"{dataset.name}_clustering_summary.csv", index=False)
    return ClusteringBenchmarkResult(
        dataset_name=dataset.name,
        summary_frame=summary_frame,
        results_by_method=results_by_method,
        output_dir=resolved_output_dir,
    )


def cluster_size_distribution_summary(cluster_sizes: Mapping[int, int], *, max_items: int = 6) -> str:
    """Render a compact cluster-size summary."""

    return _cluster_size_summary(cluster_sizes, max_items=max_items)
