"""Named end-to-end FIM stack comparisons with shared final evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from .baselines import select_baseline_seed_set
from .community_detection import (
    CommunityDetectionResult,
    compute_community_quality_metrics,
    detect_communities,
)
from .config import DatasetConfig
from .data_loader import LoadedDataset, ProtectedGroupReport, load_dataset, verify_protected_groups
from .diffusion import DEFAULT_DIFFUSION_MODEL, validate_diffusion_model
from .evaluation import SeedSetEvaluation, evaluate_seed_set
from .experiment_runner import ExperimentSettings, run_loaded_experiment
from .feature_extraction import compute_structural_node_scores
from .hybrid_optimizer import HybridSIEAConfig, HybridSIEAOptimizer
from .ris_guidance import RISConfig
from .stack_pipeline import (
    build_ranking_feature_frame,
    build_stack_label_frame,
    prepare_embedding_frame,
    prepare_optional_clustering,
    prepare_ris_guidance,
    train_ranking_model,
)

_FINAL_EVAL_RANDOM_SEED_OFFSET = 1_000_000
_EXPERIMENT_RUNNER_GNN_RIS_METHOD = "hybrid_siea_ml_gnn_ris_two_tier_tuned_swap_local_search"
_FIM_VARIANT_TYPES = {
    "baseline": "interpretable_baseline",
    "ml": "ml_guided",
    "fairness": "exploratory_fairness",
    "clustering": "clustering_enhanced",
}
_BASELINE_RANKING_MODELS = {"none", "greedy", "fairness_weighted_greedy", "maximin_greedy"}


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _normalized_seed_set(seed_set: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(sorted(tuple(seed_set), key=_sort_key))


def _seed_tuple_key(seed_set: Iterable[Any]) -> tuple[tuple[str, str], ...]:
    return tuple(_sort_key(node_id) for node_id in _normalized_seed_set(seed_set))


def _normalize_score_map(scores: Mapping[Any, float]) -> dict[Any, float]:
    if not scores:
        return {}
    ordered_nodes = sorted(scores, key=_sort_key)
    values = pd.Series([float(scores[node_id]) for node_id in ordered_nodes], dtype=float)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return {node_id: 0.0 for node_id in ordered_nodes}
    return {
        node_id: float((float(scores[node_id]) - minimum) / (maximum - minimum))
        for node_id in ordered_nodes
    }


def _resolved_eval_random_seed(config: FIMPermutationRunConfig) -> int:
    return int(config.random_seed) + _FINAL_EVAL_RANDOM_SEED_OFFSET


@dataclass(frozen=True, slots=True)
class FIMPermutationSpec:
    """Static metadata for one named FIM stack."""

    name: str
    description: str
    runner_kind: str
    diffusion_model: str
    community_method: str
    community_input_mode: str = "graph"
    spread_estimator_search: str = "monte_carlo"
    spread_estimator_final: str = "monte_carlo"
    embedding_method: str = "none"
    embedding_config: Mapping[str, Any] = field(default_factory=dict)
    clustering_method: str = "none"
    clustering_input_mode: str = "none"
    clustering_config: Mapping[str, Any] = field(default_factory=dict)
    ranking_model: str = "none"
    optimizer_mode: str = "greedy"
    debias_mode: str = "none"
    fairness_objective: str = "f_score"
    variant_family: str = "baseline"
    candidate_top_fraction: float | None = 0.5
    candidate_top_n: int | None = None
    candidate_max_nodes: int | None = None
    use_ris_guidance: bool = False
    use_fair_ris: bool = False
    ranking_weight: float = 1.0
    ris_weight: float = 1.0
    fair_ris_weight: float = 0.5
    notes: str = ""


@dataclass(frozen=True, slots=True)
class FIMPermutationRunConfig:
    """Shared runtime controls for a permutation benchmark."""

    protected_attribute: str
    budget: int
    propagation_probability: float = 0.01
    mc_runs_search: int = 20
    mc_runs_eval: int = 100
    lambda_weight: float = 0.5
    random_seed: int = 42
    output_dir: Path | None = None
    continue_on_error: bool = True
    swap_candidate_pool_size: int = 24
    local_search_steps: int = 1
    population_size: int = 8
    generations: int = 5
    gnn_epochs: int = 30
    gnn_hidden_dim: int = 32
    gnn_num_layers: int = 2
    gnn_dropout: float = 0.2
    gnn_learning_rate: float = 1e-3
    gnn_weight_decay: float = 5e-4
    ris_num_rr_sets: int = 128
    ranking_top_fraction: float | None = 0.5
    ranking_top_n: int | None = None
    ranking_max_nodes: int | None = None
    clustering_n_clusters: int | None = None
    clustering_min_cluster_size: int | None = None
    alternate_diffusion_model: str | None = None
    permutation3_diffusion_model: str = DEFAULT_DIFFUSION_MODEL
    embedding_dim: int = 64
    use_embedding_cache: bool = True
    use_score_cache: bool = True


@dataclass(slots=True)
class FIMPermutationBenchmarkResult:
    """Benchmark output plus artifact paths."""

    summary_frame: pd.DataFrame
    results_by_permutation: dict[str, pd.DataFrame]
    comparison_csv_path: Path | None = None
    report_path: Path | None = None


def _stack_registry() -> dict[str, FIMPermutationSpec]:
    return {
        "community_aware_fair_greedy": FIMPermutationSpec(
            name="community_aware_fair_greedy",
            description=(
                "Leiden communities with community round-robin initialization, fairness-weighted "
                "greedy selection, and bounded swap local search."
            ),
            runner_kind="community_aware_fair_greedy",
            diffusion_model="ic",
            community_method="leiden",
            ranking_model="fairness_weighted_greedy",
            optimizer_mode="local_search",
            fairness_objective="f_score",
            variant_family="baseline",
        ),
        "leiden_graphsage_fair_ris_hybrid": FIMPermutationSpec(
            name="leiden_graphsage_fair_ris_hybrid",
            description=(
                "Leiden communities with GraphSAGE ranking, fair RIS guidance, and the existing "
                "hybrid SI+EA optimizer."
            ),
            runner_kind="experiment_runner_gnn_ris",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            embedding_method="graphsage",
            ranking_model="graphsage",
            optimizer_mode="hybrid_si_ea",
            debias_mode="worst_group_boost",
            fairness_objective="f_score",
            variant_family="ml",
            use_ris_guidance=True,
            use_fair_ris=True,
        ),
        "leiden_gcn_fair_ris_hybrid": FIMPermutationSpec(
            name="leiden_gcn_fair_ris_hybrid",
            description=(
                "Leiden communities with GCN ranking, fair RIS guidance, and the existing hybrid "
                "SI+EA optimizer."
            ),
            runner_kind="experiment_runner_gnn_ris",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            embedding_method="gcn",
            ranking_model="gcn",
            optimizer_mode="hybrid_si_ea",
            debias_mode="none",
            fairness_objective="f_score",
            variant_family="ml",
            use_ris_guidance=True,
            use_fair_ris=True,
        ),
        "leiden_line_logreg_greedy": FIMPermutationSpec(
            name="leiden_line_logreg_greedy",
            description=(
                "Leiden communities with LINE embeddings, logistic-regression ranking, and "
                "candidate-restricted greedy search."
            ),
            runner_kind="ranked_greedy",
            diffusion_model="ic",
            community_method="leiden",
            embedding_method="line",
            ranking_model="logistic_regression",
            optimizer_mode="greedy",
            debias_mode="none",
            fairness_objective="f_score",
            variant_family="ml",
            candidate_top_fraction=0.5,
        ),
        "infomap_graphcl_logreg_maximin": FIMPermutationSpec(
            name="infomap_graphcl_logreg_maximin",
            description=(
                "Infomap communities with GraphCL embeddings, spectral clustering features, "
                "logistic-regression ranking, and maximin local search."
            ),
            runner_kind="ranked_maximin",
            diffusion_model="ic",
            community_method="infomap",
            embedding_method="graphcl",
            clustering_method="spectral",
            clustering_input_mode="embedding",
            ranking_model="logistic_regression",
            optimizer_mode="local_search",
            debias_mode="none",
            fairness_objective="maximin",
            variant_family="fairness",
            candidate_top_fraction=0.5,
        ),
        "leiden_node2vec_xgboost_hybrid": FIMPermutationSpec(
            name="leiden_node2vec_xgboost_hybrid",
            description=(
                "Leiden communities with Node2Vec embeddings, XGBoost ranking, and a hybrid "
                "SI+EA optimizer guided by learned node scores."
            ),
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            embedding_method="node2vec",
            ranking_model="xgboost",
            optimizer_mode="hybrid_si_ea",
            debias_mode="none",
            fairness_objective="f_score",
            variant_family="ml",
            candidate_top_fraction=0.5,
        ),
        "leiden_node2vec_kmeans_logreg_hybrid": FIMPermutationSpec(
            name="leiden_node2vec_kmeans_logreg_hybrid",
            description=(
                "Leiden communities with Node2Vec embeddings, KMeans clustering features, "
                "logistic-regression ranking, fair RIS guidance, and a hybrid SI+EA optimizer."
            ),
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            embedding_method="node2vec",
            clustering_method="kmeans",
            clustering_input_mode="embedding",
            ranking_model="logistic_regression",
            optimizer_mode="hybrid_si_ea",
            debias_mode="none",
            fairness_objective="f_score",
            variant_family="clustering",
            candidate_top_fraction=0.5,
            use_ris_guidance=True,
            use_fair_ris=True,
        ),
    }


_STACK_ALIASES = {
    "graphsage_fair_ris_hybrid": "leiden_graphsage_fair_ris_hybrid",
    "infomap_graphcl_maximin": "infomap_graphcl_logreg_maximin",
}


def available_fim_permutations() -> tuple[str, ...]:
    """Return supported canonical stack names in deterministic order."""

    return tuple(_stack_registry())


def get_fim_permutation_spec(name: str) -> FIMPermutationSpec:
    """Return metadata for one stack, accepting legacy aliases."""

    key = str(name).strip().lower()
    resolved_key = _STACK_ALIASES.get(key, key)
    registry = _stack_registry()
    if resolved_key not in registry:
        supported = ", ".join(sorted(tuple(registry) + tuple(_STACK_ALIASES)))
        raise ValueError(f"Unknown FIM permutation '{name}'. Supported permutations: {supported}.")
    return registry[resolved_key]


def _resolve_fim_permutation_spec(value: str | FIMPermutationSpec) -> FIMPermutationSpec:
    if isinstance(value, FIMPermutationSpec):
        return value
    return get_fim_permutation_spec(value)


def fim_permutation_specs() -> tuple[FIMPermutationSpec, ...]:
    """Return all canonical stack specs."""

    return tuple(_stack_registry().values())


def permutation_summary_columns() -> list[str]:
    """Return the normalized comparison columns written by this benchmark."""

    return [
        "stack_name",
        "permutation_name",
        "status",
        "dataset",
        "protected_attribute",
        "diffusion_model",
        "community_method",
        "community_input_mode",
        "clustering_method",
        "clustering_input_mode",
        "embedding_method",
        "ranking_model",
        "optimizer_mode",
        "debias_mode",
        "fairness_objective",
        "spread_estimator_search",
        "spread_estimator_final",
        "search_spread_estimator",
        "search_guidance_estimator",
        "final_spread_estimator",
        "method",
        "variant_type",
        "seed_set",
        "total_spread",
        "extra_spread",
        "mf",
        "dcv",
        "f_score",
        "MF",
        "DCV",
        "F-score",
        "runtime_seconds",
        "search_runtime_seconds",
        "final_eval_runtime_seconds",
        "mc_runs_search",
        "mc_runs_eval",
        "key_enabled_modules",
        "notes",
        "skip_reason",
        "skipped_reason",
    ]


def _permutation_output_dir(output_dir: Path | None, dataset_name: str, protected_attribute: str) -> Path | None:
    if output_dir is None:
        return None
    safe_attribute = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in protected_attribute.strip()
    ).strip("._-") or "protected_attribute"
    path = Path(output_dir) / dataset_name / safe_attribute
    path.mkdir(parents=True, exist_ok=True)
    return path


def _score_cache_path(
    output_dir: Path | None,
    dataset_name: str,
    protected_attribute: str,
    stack_name: str,
) -> Path | None:
    attribute_dir = _permutation_output_dir(output_dir, dataset_name, protected_attribute)
    if attribute_dir is None:
        return None
    return attribute_dir / f"{dataset_name}_{stack_name}_node_scores.pkl"


def _optimizer_history_path(
    output_dir: Path | None,
    dataset_name: str,
    protected_attribute: str,
    stack_name: str,
) -> Path | None:
    attribute_dir = _permutation_output_dir(output_dir, dataset_name, protected_attribute)
    if attribute_dir is None:
        return None
    return attribute_dir / f"{dataset_name}_{stack_name}_history.csv"


def _community_result(
    dataset: LoadedDataset,
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
):
    return detect_communities(
        dataset.graph,
        method=spec.community_method,
        seed=int(config.random_seed),
        input_mode=spec.community_input_mode,
    )


def _final_evaluate(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    seed_set: Iterable[Any],
    diffusion_model: str,
    config: FIMPermutationRunConfig,
) -> SeedSetEvaluation:
    return evaluate_seed_set(
        dataset=dataset,
        protected_group_report=protected_group_report,
        seed_set=_normalized_seed_set(seed_set),
        propagation_probability=float(config.propagation_probability),
        mc_runs=int(config.mc_runs_eval),
        random_seed=_resolved_eval_random_seed(config),
        lambda_weight=float(config.lambda_weight),
        include_soft_mf=True,
        diffusion_model=diffusion_model,
    )


def _search_evaluate(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    seed_set: Iterable[Any],
    diffusion_model: str,
    config: FIMPermutationRunConfig,
    seed_offset: int,
) -> SeedSetEvaluation:
    return evaluate_seed_set(
        dataset=dataset,
        protected_group_report=protected_group_report,
        seed_set=_normalized_seed_set(seed_set),
        propagation_probability=float(config.propagation_probability),
        mc_runs=int(config.mc_runs_search),
        random_seed=int(config.random_seed) + int(seed_offset),
        lambda_weight=float(config.lambda_weight),
        include_soft_mf=True,
        diffusion_model=diffusion_model,
    )


def _candidate_pool(
    dataset: LoadedDataset,
    seed_set: tuple[Any, ...],
    max_size: int,
) -> list[Any]:
    excluded = set(seed_set)
    degree_view = dataset.graph.out_degree() if dataset.graph.is_directed() else dataset.graph.degree()
    degree_scores = {node_id: float(score) for node_id, score in degree_view if node_id not in excluded}
    candidates = sorted(degree_scores, key=lambda node_id: (-degree_scores[node_id], _sort_key(node_id)))
    if max_size > 0:
        candidates = candidates[: int(max_size)]
    return candidates


def _rank_search_evaluation(mode: str, evaluation: SeedSetEvaluation) -> tuple[float, float, float, float]:
    if mode == "maximin":
        return (
            float(evaluation.fairness.mf),
            float(evaluation.f_score),
            float(evaluation.total_spread_mean),
            -float(evaluation.fairness.dcv),
        )
    return (
        float(evaluation.f_score),
        float(evaluation.fairness.mf),
        float(evaluation.total_spread_mean),
        -float(evaluation.fairness.dcv),
    )


def _swap_local_search(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    initial_seed_set: tuple[Any, ...],
    diffusion_model: str,
    config: FIMPermutationRunConfig,
    objective_mode: str,
) -> tuple[tuple[Any, ...], SeedSetEvaluation]:
    current = _normalized_seed_set(initial_seed_set)
    current_eval = _search_evaluate(
        dataset,
        protected_group_report,
        current,
        diffusion_model,
        config,
        seed_offset=101,
    )
    cache: dict[tuple[Any, ...], SeedSetEvaluation] = {current: current_eval}

    for step_index in range(max(0, int(config.local_search_steps))):
        best_seed_set = current
        best_eval = current_eval
        best_key = _rank_search_evaluation(objective_mode, current_eval)
        for removed_node in current:
            retained_nodes = [node_id for node_id in current if node_id != removed_node]
            for candidate_node in _candidate_pool(dataset, current, int(config.swap_candidate_pool_size)):
                if candidate_node in retained_nodes:
                    continue
                trial_seed_set = _normalized_seed_set([*retained_nodes, candidate_node])
                evaluation = cache.get(trial_seed_set)
                if evaluation is None:
                    evaluation = _search_evaluate(
                        dataset,
                        protected_group_report,
                        trial_seed_set,
                        diffusion_model,
                        config,
                        seed_offset=211 + step_index,
                    )
                    cache[trial_seed_set] = evaluation
                rank_key = _rank_search_evaluation(objective_mode, evaluation)
                if rank_key > best_key or (
                    rank_key == best_key and _seed_tuple_key(trial_seed_set) < _seed_tuple_key(best_seed_set)
                ):
                    best_seed_set = trial_seed_set
                    best_eval = evaluation
                    best_key = rank_key
        if best_seed_set == current:
            break
        current = best_seed_set
        current_eval = best_eval
    return current, current_eval


def _search_guidance_estimator(spec: FIMPermutationSpec) -> str:
    if spec.use_fair_ris:
        return "fair_ris_guidance"
    if spec.use_ris_guidance:
        return "ris_guidance"
    return "none"


def _key_enabled_modules(spec: FIMPermutationSpec, clustering_input_mode: str) -> str:
    parts = [
        f"diffusion={spec.diffusion_model}",
        f"community={spec.community_method}",
        f"embedding={spec.embedding_method}",
        f"clustering={spec.clustering_method}",
        f"clustering_input_mode={clustering_input_mode}",
        f"ranking={spec.ranking_model}",
        f"optimizer={spec.optimizer_mode}",
        f"debias_mode={spec.debias_mode}",
        f"search_estimator={spec.spread_estimator_search}",
        f"final_estimator={spec.spread_estimator_final}",
    ]
    return " | ".join(parts)


def _spread_dict(evaluation: SeedSetEvaluation, budget: int) -> dict[str, object]:
    total_spread = float(evaluation.total_spread_mean)
    mf = float(evaluation.fairness.mf)
    dcv = float(evaluation.fairness.dcv)
    f_score = float(evaluation.f_score)
    return {
        "seed_set": json.dumps(list(evaluation.seed_set)),
        "total_spread": total_spread,
        "extra_spread": float(total_spread - float(budget)),
        "mf": mf,
        "dcv": dcv,
        "f_score": f_score,
        "MF": mf,
        "DCV": dcv,
        "F-score": f_score,
        "final_eval_runtime_seconds": float(evaluation.runtime_seconds),
    }


def _result_row(
    *,
    spec: FIMPermutationSpec,
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    config: FIMPermutationRunConfig,
    evaluation: SeedSetEvaluation,
    search_runtime_seconds: float,
    status: str = "ok",
    notes: str = "",
    skip_reason: str = "",
    clustering_input_mode: str | None = None,
    method: str,
) -> dict[str, object]:
    resolved_clustering_input_mode = (
        "none"
        if spec.clustering_method == "none"
        else str(clustering_input_mode or spec.clustering_input_mode)
    )
    row = {
        "stack_name": spec.name,
        "permutation_name": spec.name,
        "status": status,
        "dataset": dataset.name,
        "protected_attribute": protected_group_report.protected_attribute,
        "diffusion_model": spec.diffusion_model,
        "community_method": spec.community_method,
        "community_input_mode": spec.community_input_mode,
        "clustering_method": spec.clustering_method,
        "clustering_input_mode": resolved_clustering_input_mode,
        "embedding_method": spec.embedding_method,
        "ranking_model": spec.ranking_model,
        "optimizer_mode": spec.optimizer_mode,
        "debias_mode": spec.debias_mode,
        "fairness_objective": spec.fairness_objective,
        "spread_estimator_search": spec.spread_estimator_search,
        "spread_estimator_final": spec.spread_estimator_final,
        "search_spread_estimator": (
            "ris_guidance"
            if spec.spread_estimator_search == "fairness_aware_ris"
            else spec.spread_estimator_search
        ),
        "search_guidance_estimator": _search_guidance_estimator(spec),
        "final_spread_estimator": spec.spread_estimator_final,
        "method": method,
        "variant_type": _FIM_VARIANT_TYPES[spec.variant_family],
        "runtime_seconds": float(search_runtime_seconds + evaluation.runtime_seconds),
        "search_runtime_seconds": float(search_runtime_seconds),
        "mc_runs_search": int(config.mc_runs_search),
        "mc_runs_eval": int(config.mc_runs_eval),
        "key_enabled_modules": _key_enabled_modules(spec, resolved_clustering_input_mode),
        "notes": "; ".join(part for part in [spec.notes, notes] if part),
        "skip_reason": skip_reason,
        "skipped_reason": skip_reason,
    }
    row.update(_spread_dict(evaluation, int(config.budget)))
    return row


def _skipped_row(
    *,
    spec: FIMPermutationSpec,
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    config: FIMPermutationRunConfig,
    skip_reason: str,
) -> dict[str, object]:
    row = {column: pd.NA for column in permutation_summary_columns()}
    row.update(
        {
            "stack_name": spec.name,
            "permutation_name": spec.name,
            "status": "skipped",
            "dataset": dataset.name,
            "protected_attribute": protected_group_report.protected_attribute,
            "diffusion_model": spec.diffusion_model,
            "community_method": spec.community_method,
            "community_input_mode": spec.community_input_mode,
            "clustering_method": spec.clustering_method,
            "clustering_input_mode": spec.clustering_input_mode,
            "embedding_method": spec.embedding_method,
            "ranking_model": spec.ranking_model,
            "optimizer_mode": spec.optimizer_mode,
            "debias_mode": spec.debias_mode,
            "fairness_objective": spec.fairness_objective,
            "spread_estimator_search": spec.spread_estimator_search,
            "spread_estimator_final": spec.spread_estimator_final,
            "search_spread_estimator": (
                "ris_guidance"
                if spec.spread_estimator_search == "fairness_aware_ris"
                else spec.spread_estimator_search
            ),
            "search_guidance_estimator": _search_guidance_estimator(spec),
            "final_spread_estimator": spec.spread_estimator_final,
            "mc_runs_search": int(config.mc_runs_search),
            "mc_runs_eval": int(config.mc_runs_eval),
            "key_enabled_modules": _key_enabled_modules(spec, spec.clustering_input_mode),
            "notes": spec.notes,
            "skip_reason": skip_reason,
            "skipped_reason": skip_reason,
        }
    )
    return row


def _clustering_config(spec: FIMPermutationSpec, config: FIMPermutationRunConfig) -> dict[str, object]:
    resolved = dict(spec.clustering_config)
    if config.clustering_n_clusters is not None:
        resolved["n_clusters"] = int(config.clustering_n_clusters)
    elif "n_clusters" not in resolved and spec.clustering_method in {"kmeans", "spectral", "agglomerative", "gmm"}:
        resolved["n_clusters"] = max(2, min(int(config.budget), 8))
    if config.clustering_min_cluster_size is not None:
        resolved["min_cluster_size"] = int(config.clustering_min_cluster_size)
    return resolved


def _combine_guidance_scores(
    spec: FIMPermutationSpec,
    ranking_scores: Mapping[Any, float] | None,
    ris_scores: Mapping[Any, float] | None,
    fair_ris_scores: Mapping[Any, float] | None,
) -> dict[Any, float]:
    normalized_ranking = _normalize_score_map(ranking_scores or {})
    normalized_ris = _normalize_score_map(ris_scores or {})
    normalized_fair_ris = _normalize_score_map(fair_ris_scores or {})
    all_nodes = sorted(
        set(normalized_ranking) | set(normalized_ris) | set(normalized_fair_ris),
        key=_sort_key,
    )
    if not all_nodes:
        return {}
    combined: dict[Any, float] = {}
    for node_id in all_nodes:
        combined[node_id] = float(
            spec.ranking_weight * normalized_ranking.get(node_id, 0.0)
            + spec.ris_weight * normalized_ris.get(node_id, 0.0)
            + spec.fair_ris_weight * normalized_fair_ris.get(node_id, 0.0)
        )
    return combined


def _resolved_candidate_controls(
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
) -> tuple[float | None, int | None, int | None]:
    return (
        spec.candidate_top_fraction if spec.candidate_top_fraction is not None else config.ranking_top_fraction,
        spec.candidate_top_n if spec.candidate_top_n is not None else config.ranking_top_n,
        spec.candidate_max_nodes if spec.candidate_max_nodes is not None else config.ranking_max_nodes,
    )


def _build_shared_stack_inputs(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
) -> dict[str, Any]:
    embedding_config = dict(spec.embedding_config)
    if spec.embedding_method not in {"", "none", "off"}:
        embedding_config.setdefault("embedding_dim", int(config.embedding_dim))
        if spec.embedding_method in {"graphsage", "gcn", "dgi", "vgae", "graphcl"}:
            embedding_config.setdefault("hidden_dim", int(config.gnn_hidden_dim))
        if spec.embedding_method == "graphcl":
            embedding_config.setdefault("projection_dim", int(config.embedding_dim))

    community_result = _community_result(dataset, spec, config)
    embedding_artifact = prepare_embedding_frame(
        dataset=dataset,
        method_name=spec.embedding_method,
        output_dir=config.output_dir,
        random_seed=int(config.random_seed),
        method_config=embedding_config,
        allow_cache=bool(config.use_embedding_cache),
    )
    clustering_artifact = prepare_optional_clustering(
        dataset=dataset,
        method_name=spec.clustering_method,
        input_mode=spec.clustering_input_mode,
        embeddings=embedding_artifact.embedding_frame,
        config=_clustering_config(spec, config),
        output_dir=config.output_dir,
        random_seed=int(config.random_seed),
    )
    base_feature_frame = build_ranking_feature_frame(
        dataset=dataset,
        protected_group_report=protected_group_report,
        community_result=community_result,
        embedding_frame=embedding_artifact.embedding_frame,
        clustering_result=clustering_artifact.clustering_result,
    )
    ris_artifact = None
    if spec.use_ris_guidance:
        ris_artifact = prepare_ris_guidance(
            dataset=dataset,
            protected_group_report=protected_group_report,
            propagation_probability=float(config.propagation_probability),
            feature_frame=base_feature_frame,
            output_dir=config.output_dir,
            protected_attribute=protected_group_report.protected_attribute,
            stack_name=spec.name,
            config=RISConfig(
                num_rr_sets=int(config.ris_num_rr_sets),
                random_seed=int(config.random_seed),
                mode="weak_group_weighted" if spec.use_fair_ris else "global",
                reuse_rr_sets=True,
            ),
        )
    feature_frame = build_ranking_feature_frame(
        dataset=dataset,
        protected_group_report=protected_group_report,
        community_result=community_result,
        embedding_frame=embedding_artifact.embedding_frame,
        clustering_result=clustering_artifact.clustering_result,
        ris_scores=None if ris_artifact is None else ris_artifact.global_scores,
        fair_ris_scores=None if ris_artifact is None else ris_artifact.fair_scores,
    )
    label_result = build_stack_label_frame(
        dataset=dataset,
        protected_group_report=protected_group_report,
        propagation_probability=float(config.propagation_probability),
        mc_runs=int(config.mc_runs_search),
        lambda_weight=float(config.lambda_weight),
        random_seed=int(config.random_seed),
    )
    top_fraction, top_n, max_nodes = _resolved_candidate_controls(spec, config)
    ranking_artifact = train_ranking_model(
        dataset=dataset,
        feature_frame=feature_frame,
        label_frame=label_result.label_frame,
        budget=int(config.budget),
        model_type=spec.ranking_model,
        target_column="label_score",
        protected_attribute=protected_group_report.protected_attribute,
        stack_name=spec.name,
        output_dir=config.output_dir,
        random_seed=int(config.random_seed),
        top_fraction=top_fraction,
        top_n=top_n,
        max_nodes=max_nodes,
        hidden_dim=int(config.gnn_hidden_dim),
        num_layers=int(config.gnn_num_layers),
        dropout=float(config.gnn_dropout),
        learning_rate=float(config.gnn_learning_rate),
        weight_decay=float(config.gnn_weight_decay),
        epochs=int(config.gnn_epochs),
        cache_path=_score_cache_path(
            config.output_dir,
            dataset.name,
            protected_group_report.protected_attribute,
            spec.name,
        ) if bool(config.use_score_cache) else None,
        debias_mode=spec.debias_mode,
    )
    combined_scores = _combine_guidance_scores(
        spec,
        ranking_artifact.training_result.predicted_scores,
        None if ris_artifact is None else ris_artifact.global_scores,
        None if ris_artifact is None else ris_artifact.fair_scores,
    )
    return {
        "community_result": community_result,
        "embedding_artifact": embedding_artifact,
        "clustering_artifact": clustering_artifact,
        "feature_frame": feature_frame,
        "label_result": label_result,
        "ranking_artifact": ranking_artifact,
        "combined_guidance_scores": combined_scores,
    }


def _stack_notes(
    spec: FIMPermutationSpec,
    shared: Mapping[str, Any],
    quality_metrics,
    extra_notes: str = "",
) -> str:
    ranking_artifact = shared["ranking_artifact"]
    embedding_artifact = shared["embedding_artifact"]
    clustering_artifact = shared["clustering_artifact"]
    notes_parts = [
        f"communities={quality_metrics.num_communities}",
        f"modularity={quality_metrics.modularity:.6f}",
        f"embedding_cache={'hit' if embedding_artifact.loaded_from_cache else 'miss'}",
        f"target={ranking_artifact.training_result.target_column}",
        f"backend={ranking_artifact.training_result.backend}",
        f"candidate_pool={len(ranking_artifact.training_result.candidate_nodes)}",
        f"validation_spearman={ranking_artifact.training_result.validation_spearman:.4f}",
        f"validation_precision_at_budget={ranking_artifact.training_result.validation_precision_at_budget:.4f}",
    ]
    if clustering_artifact.clustering_result is not None:
        notes_parts.append(f"clusters={len(clustering_artifact.clustering_result.clusters)}")
        notes_parts.append(f"clustering_input_mode={clustering_artifact.clustering_result.resolved_input_mode}")
    if extra_notes:
        notes_parts.append(extra_notes)
    return "; ".join(notes_parts)


def _run_community_aware_fair_greedy(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
) -> pd.DataFrame:
    start = perf_counter()
    community_result = _community_result(dataset, spec, config)
    initial_seed_set = select_baseline_seed_set(
        dataset=dataset,
        method="community_round_robin",
        budget=int(config.budget),
        protected_group_report=protected_group_report,
        propagation_probability=float(config.propagation_probability),
        mc_runs=int(config.mc_runs_search),
        lambda_weight=float(config.lambda_weight),
        community_result=community_result,
        random_seed=int(config.random_seed),
        diffusion_model=spec.diffusion_model,
    )
    greedy_seed_set = select_baseline_seed_set(
        dataset=dataset,
        method="fairness_weighted_greedy",
        budget=int(config.budget),
        protected_group_report=protected_group_report,
        propagation_probability=float(config.propagation_probability),
        mc_runs=int(config.mc_runs_search),
        lambda_weight=float(config.lambda_weight),
        community_result=community_result,
        random_seed=int(config.random_seed),
        diffusion_model=spec.diffusion_model,
    )
    initial_eval = _search_evaluate(
        dataset,
        protected_group_report,
        initial_seed_set,
        spec.diffusion_model,
        config,
        seed_offset=17,
    )
    greedy_eval = _search_evaluate(
        dataset,
        protected_group_report,
        greedy_seed_set,
        spec.diffusion_model,
        config,
        seed_offset=19,
    )
    chosen_start = (
        greedy_seed_set
        if _rank_search_evaluation("f_score", greedy_eval) >= _rank_search_evaluation("f_score", initial_eval)
        else initial_seed_set
    )
    refined_seed_set, _ = _swap_local_search(
        dataset=dataset,
        protected_group_report=protected_group_report,
        initial_seed_set=chosen_start,
        diffusion_model=spec.diffusion_model,
        config=config,
        objective_mode="f_score",
    )
    search_runtime = perf_counter() - start
    final_eval = _final_evaluate(dataset, protected_group_report, refined_seed_set, spec.diffusion_model, config)
    quality = compute_community_quality_metrics(dataset.graph, community_result)
    notes = (
        f"initial_rr={list(initial_seed_set)}; greedy={list(greedy_seed_set)}; "
        f"communities={quality.num_communities}; modularity={quality.modularity:.6f}"
    )
    return pd.DataFrame(
        [
            _result_row(
                spec=spec,
                dataset=dataset,
                protected_group_report=protected_group_report,
                config=config,
                evaluation=final_eval,
                search_runtime_seconds=search_runtime,
                clustering_input_mode="none",
                method="community_round_robin+fairness_weighted_greedy+swap_local_search",
                notes=notes,
            )
        ]
    )


def _run_experiment_runner_gnn_ris_stack(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
) -> pd.DataFrame:
    legacy_debias_mode = "off" if spec.debias_mode in {"none", "off"} else spec.debias_mode
    settings = ExperimentSettings(
        protected_attribute=config.protected_attribute,
        budget=int(config.budget),
        diffusion_model=spec.diffusion_model,
        spread_estimator="ris_guidance",
        community_method=spec.community_method,
        propagation_probability=float(config.propagation_probability),
        mc_runs_search=int(config.mc_runs_search),
        mc_runs_eval=int(config.mc_runs_eval),
        lambda_weight=float(config.lambda_weight),
        population_size=int(config.population_size),
        generations=int(config.generations),
        local_search_steps=max(1, int(config.local_search_steps)),
        random_seed=int(config.random_seed),
        output_dir=config.output_dir,
        use_ml=True,
        ml_backend="gnn_ris",
        ml_guidance_mode="two_tier",
        debias_mode=legacy_debias_mode,
        gnn_model_type=spec.ranking_model,
        gnn_hidden_dim=int(config.gnn_hidden_dim),
        gnn_num_layers=int(config.gnn_num_layers),
        gnn_dropout=float(config.gnn_dropout),
        gnn_learning_rate=float(config.gnn_learning_rate),
        gnn_weight_decay=float(config.gnn_weight_decay),
        gnn_epochs=int(config.gnn_epochs),
        ris_num_rr_sets=int(config.ris_num_rr_sets),
        ris_random_seed=int(config.random_seed),
        ris_mode="weak_group_weighted",
        ris_reuse_rr_sets=True,
        gnn_weight=float(spec.ranking_weight),
        ris_weight=float(spec.ris_weight),
        fair_ris_weight=float(spec.fair_ris_weight),
        fairness_urgency_weight=0.20,
        repair_fairness_weight=0.40,
        repair_bridge_weight=0.20,
        weakest_group_k=3,
        weakest_group_mutation_weight=0.35,
        zero_group_bonus_weight=0.25,
        bridge_to_weak_group_weight=0.15,
        local_search_focus_mode="worst_group",
        local_search_bottom_k_groups=3,
        local_search_swap_trials=max(1, int(config.local_search_steps)),
        local_search_candidate_pool_size=max(4, int(config.swap_candidate_pool_size)),
        swap_candidate_pool_size=max(4, int(config.swap_candidate_pool_size)),
    )
    frame = run_loaded_experiment(
        dataset=dataset,
        protected_group_report=protected_group_report,
        settings=settings,
        community_methods=[spec.community_method],
        baseline_methods=[],
        selected_methods=[_EXPERIMENT_RUNNER_GNN_RIS_METHOD],
        include_ablations=False,
    )
    if frame.empty:
        raise RuntimeError(f"{spec.name} produced no result rows.")
    row = frame.iloc[0].to_dict()
    row.update(
        {
            "stack_name": spec.name,
            "permutation_name": spec.name,
            "status": "ok",
            "community_input_mode": "graph",
            "clustering_method": "none",
            "clustering_input_mode": "none",
            "embedding_method": spec.embedding_method,
            "ranking_model": spec.ranking_model,
            "optimizer_mode": spec.optimizer_mode,
            "debias_mode": spec.debias_mode,
            "fairness_objective": spec.fairness_objective,
            "spread_estimator_search": spec.spread_estimator_search,
            "spread_estimator_final": spec.spread_estimator_final,
            "search_spread_estimator": "ris_guidance",
            "search_guidance_estimator": "fair_ris_guidance",
            "final_spread_estimator": spec.spread_estimator_final,
            "key_enabled_modules": _key_enabled_modules(spec, "none"),
            "MF": float(row["mf"]),
            "DCV": float(row["dcv"]),
            "F-score": float(row["f_score"]),
            "notes": "; ".join(
                part
                for part in [
                    spec.notes,
                    str(row.get("note", "")),
                    "Delegated to existing gnn_ris hybrid path with shared final Monte Carlo evaluation.",
                ]
                if part
            ),
            "skip_reason": "",
            "skipped_reason": "",
        }
    )
    return pd.DataFrame([row])


def _run_ranked_greedy_stack(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
) -> pd.DataFrame:
    start = perf_counter()
    shared = _build_shared_stack_inputs(dataset, protected_group_report, spec, config)
    seed_set = select_baseline_seed_set(
        dataset=dataset,
        method="greedy",
        budget=int(config.budget),
        protected_group_report=protected_group_report,
        propagation_probability=float(config.propagation_probability),
        mc_runs=int(config.mc_runs_search),
        lambda_weight=float(config.lambda_weight),
        community_result=shared["community_result"],
        random_seed=int(config.random_seed),
        diffusion_model=spec.diffusion_model,
        candidate_nodes=shared["ranking_artifact"].training_result.candidate_nodes,
        candidate_scores=shared["combined_guidance_scores"],
    )
    search_runtime = perf_counter() - start
    final_eval = _final_evaluate(dataset, protected_group_report, seed_set, spec.diffusion_model, config)
    quality = compute_community_quality_metrics(dataset.graph, shared["community_result"])
    notes = _stack_notes(spec, shared, quality)
    return pd.DataFrame(
        [
            _result_row(
                spec=spec,
                dataset=dataset,
                protected_group_report=protected_group_report,
                config=config,
                evaluation=final_eval,
                search_runtime_seconds=search_runtime,
                clustering_input_mode=(
                    "none"
                    if shared["clustering_artifact"].clustering_result is None
                    else shared["clustering_artifact"].clustering_result.resolved_input_mode
                ),
                method="embedding+ranking+greedy",
                notes=notes,
            )
        ]
    )


def _run_ranked_maximin_stack(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
) -> pd.DataFrame:
    start = perf_counter()
    shared = _build_shared_stack_inputs(dataset, protected_group_report, spec, config)
    initial_seed_set = select_baseline_seed_set(
        dataset=dataset,
        method="maximin_greedy",
        budget=int(config.budget),
        protected_group_report=protected_group_report,
        propagation_probability=float(config.propagation_probability),
        mc_runs=int(config.mc_runs_search),
        lambda_weight=float(config.lambda_weight),
        community_result=shared["community_result"],
        random_seed=int(config.random_seed),
        diffusion_model=spec.diffusion_model,
        candidate_nodes=shared["ranking_artifact"].training_result.candidate_nodes,
        candidate_scores=shared["combined_guidance_scores"],
    )
    refined_seed_set, _ = _swap_local_search(
        dataset=dataset,
        protected_group_report=protected_group_report,
        initial_seed_set=initial_seed_set,
        diffusion_model=spec.diffusion_model,
        config=config,
        objective_mode="maximin",
    )
    search_runtime = perf_counter() - start
    final_eval = _final_evaluate(dataset, protected_group_report, refined_seed_set, spec.diffusion_model, config)
    quality = compute_community_quality_metrics(dataset.graph, shared["community_result"])
    notes = _stack_notes(spec, shared, quality, extra_notes=f"initial_maximin={list(initial_seed_set)}")
    return pd.DataFrame(
        [
            _result_row(
                spec=spec,
                dataset=dataset,
                protected_group_report=protected_group_report,
                config=config,
                evaluation=final_eval,
                search_runtime_seconds=search_runtime,
                clustering_input_mode=(
                    "none"
                    if shared["clustering_artifact"].clustering_result is None
                    else shared["clustering_artifact"].clustering_result.resolved_input_mode
                ),
                method="embedding+ranking+maximin_greedy+swap_local_search",
                notes=notes,
            )
        ]
    )


def _hybrid_optimizer_config(spec: FIMPermutationSpec, config: FIMPermutationRunConfig) -> HybridSIEAConfig:
    return HybridSIEAConfig(
        budget=int(config.budget),
        population_size=int(config.population_size),
        generations=int(config.generations),
        propagation_probability=float(config.propagation_probability),
        mc_runs=int(config.mc_runs_search),
        diffusion_model=spec.diffusion_model,
        lambda_weight=float(config.lambda_weight),
        random_seed=int(config.random_seed),
        local_search_steps=max(1, int(config.local_search_steps)),
        ml_guidance_mode="two_tier",
        ml_primary_pool_ratio=0.50,
        ml_secondary_exploration_rate=0.10,
        ml_initialization_bias=0.25,
        ml_initialization_primary_rate=0.90,
        ml_mutation_primary_rate=0.80,
        ml_repair_primary_rate=0.70,
        ml_local_search_primary_rate=0.60,
        ml_mutation_bias_weight=0.20,
        ml_repair_bias_weight=0.15,
        ml_local_search_bias_weight=0.25,
        fairness_first_init_enabled=spec.use_fair_ris,
        fairness_first_init_slots=min(2, int(config.budget)),
        fairness_first_init_weight=0.75 if spec.use_fair_ris else 0.0,
        weakest_group_k=3 if spec.use_fair_ris else 1,
        weakest_group_mutation_weight=0.35 if spec.use_fair_ris else 0.0,
        zero_group_bonus_weight=0.25 if spec.use_fair_ris else 0.0,
        bridge_to_weak_group_weight=0.15 if spec.use_fair_ris else 0.0,
        repair_fairness_weight=0.40 if spec.use_fair_ris else 0.0,
        repair_bridge_weight=0.20 if spec.use_fair_ris else 0.0,
        local_search_focus_mode="worst_group" if spec.use_fair_ris else "default",
        local_search_bottom_k_groups=3,
        local_search_swap_trials=max(1, int(config.local_search_steps)),
        local_search_candidate_pool_size=max(4, int(config.swap_candidate_pool_size)),
        swap_candidate_pool_size=max(4, int(config.swap_candidate_pool_size)),
        optimization_mode="full",
    )


def _run_ranked_hybrid_stack(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
) -> pd.DataFrame:
    start = perf_counter()
    shared = _build_shared_stack_inputs(dataset, protected_group_report, spec, config)
    feature_frame = shared["feature_frame"]
    structural_scores = compute_structural_node_scores(feature_frame)
    optimizer = HybridSIEAOptimizer(
        dataset=dataset,
        protected_group_report=protected_group_report,
        community_result=shared["community_result"],
        config=_hybrid_optimizer_config(spec, config),
        candidate_nodes=shared["ranking_artifact"].training_result.candidate_nodes,
        node_scores=structural_scores,
        ml_node_scores=shared["combined_guidance_scores"],
    )
    optimization_result = optimizer.optimize()
    history_path = _optimizer_history_path(
        config.output_dir,
        dataset.name,
        protected_group_report.protected_attribute,
        spec.name,
    )
    if history_path is not None:
        optimization_result.history.to_csv(history_path, index=False)
    search_runtime = perf_counter() - start
    final_eval = _final_evaluate(
        dataset,
        protected_group_report,
        optimization_result.best_seed_set,
        spec.diffusion_model,
        config,
    )
    quality = compute_community_quality_metrics(dataset.graph, shared["community_result"])
    notes = _stack_notes(
        spec,
        shared,
        quality,
        extra_notes=(
            f"history={history_path.name}" if history_path is not None else ""
        ),
    )
    return pd.DataFrame(
        [
            _result_row(
                spec=spec,
                dataset=dataset,
                protected_group_report=protected_group_report,
                config=config,
                evaluation=final_eval,
                search_runtime_seconds=search_runtime,
                clustering_input_mode=(
                    "none"
                    if shared["clustering_artifact"].clustering_result is None
                    else shared["clustering_artifact"].clustering_result.resolved_input_mode
                ),
                method="embedding+ranking+hybrid_si_ea",
                notes=notes,
            )
        ]
    )


_RUNNERS: Mapping[str, Callable[[LoadedDataset, ProtectedGroupReport, FIMPermutationSpec, FIMPermutationRunConfig], pd.DataFrame]] = {
    "community_aware_fair_greedy": _run_community_aware_fair_greedy,
    "experiment_runner_gnn_ris": _run_experiment_runner_gnn_ris_stack,
    "ranked_greedy": _run_ranked_greedy_stack,
    "ranked_maximin": _run_ranked_maximin_stack,
    "ranked_hybrid": _run_ranked_hybrid_stack,
}


def _normalize_summary_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in permutation_summary_columns():
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, permutation_summary_columns()].copy()


def _recommend_stack_name(frame: pd.DataFrame, mask: pd.Series) -> str | None:
    candidates = frame.loc[mask].copy()
    if candidates.empty:
        return None
    for column in ["f_score", "mf", "total_spread", "runtime_seconds"]:
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    ordered = candidates.sort_values(
        ["f_score", "mf", "total_spread", "runtime_seconds"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    if ordered.empty:
        return None
    return str(ordered.iloc[0]["stack_name"])


def recommend_fim_stacks(frame: pd.DataFrame) -> dict[str, str | None]:
    """Return the best stack names under the configured recommendation rules."""

    if frame.empty:
        return {
            "best_interpretable_baseline": None,
            "best_practical_ml_default": None,
            "best_exploratory_fairness_heavy": None,
        }
    ok_mask = frame["status"].astype(str) == "ok"
    debias_mask = ~frame["debias_mode"].astype(str).isin({"none", "off"})
    fairness_search_mask = frame["fairness_objective"].astype(str).eq("maximin") | frame["spread_estimator_search"].astype(str).eq("fairness_aware_ris")
    interpretable_mask = ok_mask & frame["variant_type"].astype(str).eq(_FIM_VARIANT_TYPES["baseline"])
    ml_mask = ok_mask & ~frame["ranking_model"].astype(str).isin(_BASELINE_RANKING_MODELS)
    fairness_mask = ok_mask & (debias_mask | fairness_search_mask)
    return {
        "best_interpretable_baseline": _recommend_stack_name(frame, interpretable_mask),
        "best_practical_ml_default": _recommend_stack_name(frame, ml_mask),
        "best_exploratory_fairness_heavy": _recommend_stack_name(frame, fairness_mask),
    }


def format_fim_permutation_report(frame: pd.DataFrame, config: FIMPermutationRunConfig) -> str:
    """Format a compact terminal/report comparison table."""

    lines = [
        "FIM Algorithm-Stack Permutation Comparison",
        "-" * 72,
        f"Protected attribute: {config.protected_attribute}",
        f"Budget: {config.budget}",
        f"Final estimator: monte_carlo | mc_runs_eval={config.mc_runs_eval}",
        "",
    ]
    if frame.empty:
        lines.append("No permutation rows were produced.")
        return "\n".join(lines)

    sortable = frame.copy()
    for column in ["f_score", "mf", "total_spread", "runtime_seconds"]:
        sortable[column] = pd.to_numeric(sortable[column], errors="coerce")
    ok_rows = sortable[sortable["status"] == "ok"].sort_values(
        ["f_score", "mf", "total_spread", "runtime_seconds"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    skipped_rows = sortable[sortable["status"] != "ok"]
    ordered = pd.concat([ok_rows, skipped_rows], ignore_index=True)
    for index, row in ordered.iterrows():
        status = str(row.get("status", "unknown"))
        lines.append(f"{index + 1}. {row['stack_name']} [{status}]")
        if status == "ok":
            lines.append(
                "   "
                f"spread={float(row['total_spread']):.4f} | "
                f"extra={float(row['extra_spread']):.4f} | "
                f"MF={float(row['mf']):.4f} | "
                f"DCV={float(row['dcv']):.4f} | "
                f"F-score={float(row['f_score']):.4f} | "
                f"runtime={float(row['runtime_seconds']):.3f}s"
            )
        else:
            lines.append(f"   skip_reason={row.get('skip_reason', '')}")
        lines.append(
            "   "
            f"modules: diffusion={row.get('diffusion_model')} | "
            f"community={row.get('community_method')} | "
            f"embedding={row.get('embedding_method')} | "
            f"clustering={row.get('clustering_method')} | "
            f"ranking={row.get('ranking_model')} | "
            f"optimizer={row.get('optimizer_mode')} | "
            f"debias={row.get('debias_mode')}"
        )
        notes = str(row.get("notes", "")).strip()
        if notes and notes != "<NA>":
            lines.append(f"   notes={notes}")
        lines.append("")

    recommendations = recommend_fim_stacks(frame)
    lines.append("Recommendations")
    lines.append("-" * 72)
    lines.append(
        f"best_interpretable_baseline={recommendations['best_interpretable_baseline'] or 'n/a'}"
    )
    lines.append(
        f"best_practical_ml_default={recommendations['best_practical_ml_default'] or 'n/a'}"
    )
    lines.append(
        f"best_exploratory_fairness_heavy={recommendations['best_exploratory_fairness_heavy'] or 'n/a'}"
    )
    return "\n".join(lines).rstrip()


def run_fim_permutation_benchmark(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    config: FIMPermutationRunConfig,
    permutations: Iterable[str | FIMPermutationSpec] | None = None,
) -> FIMPermutationBenchmarkResult:
    """Run named FIM stacks with shared final Monte Carlo evaluation."""

    requested = tuple(permutations or available_fim_permutations())
    rows: list[dict[str, object]] = []
    results_by_permutation: dict[str, pd.DataFrame] = {}

    for requested_value in requested:
        spec = _resolve_fim_permutation_spec(requested_value)
        if spec.spread_estimator_final != "monte_carlo":
            raise ValueError("All permutation specs must use spread_estimator_final='monte_carlo'.")
        runner = _RUNNERS[spec.runner_kind]
        try:
            frame = runner(dataset, protected_group_report, spec, config)
        except Exception as exc:  # noqa: BLE001 - benchmark rows should skip cleanly.
            if not config.continue_on_error:
                raise
            frame = pd.DataFrame(
                [
                    _skipped_row(
                        spec=spec,
                        dataset=dataset,
                        protected_group_report=protected_group_report,
                        config=config,
                        skip_reason=f"{type(exc).__name__}: {exc}",
                    )
                ]
            )
        results_by_permutation[spec.name] = frame.copy()
        rows.extend(frame.to_dict(orient="records"))

    summary_frame = _normalize_summary_frame(rows)
    output_dir = _permutation_output_dir(config.output_dir, dataset.name, protected_group_report.protected_attribute)
    comparison_csv_path = None
    report_path = None
    if output_dir is not None:
        comparison_csv_path = output_dir / f"{dataset.name}_budget{config.budget}_permutation_comparison.csv"
        summary_frame.to_csv(comparison_csv_path, index=False)
        report_path = output_dir / f"{dataset.name}_budget{config.budget}_permutation_report.txt"
        report_path.write_text(format_fim_permutation_report(summary_frame, config), encoding="utf-8")

    return FIMPermutationBenchmarkResult(
        summary_frame=summary_frame,
        results_by_permutation=results_by_permutation,
        comparison_csv_path=comparison_csv_path,
        report_path=report_path,
    )


def run_fim_permutation_benchmark_from_config(
    dataset_config: DatasetConfig,
    config: FIMPermutationRunConfig,
    permutations: Iterable[str | FIMPermutationSpec] | None = None,
) -> FIMPermutationBenchmarkResult:
    """Load a dataset and run the named permutation benchmark."""

    dataset = load_dataset(dataset_config)
    protected_group_report = verify_protected_groups(dataset, config.protected_attribute)
    return run_fim_permutation_benchmark(
        dataset=dataset,
        protected_group_report=protected_group_report,
        config=config,
        permutations=permutations,
    )
