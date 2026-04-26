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
    use_community_features_for_ml: bool = True
    community_feature_mode: str = "basic"
    allow_protected_features_in_ml: bool = False
    use_ml_scores_in_initialization: bool | None = None
    use_ml_scores_in_mutation: bool | None = None
    use_ml_scores_in_crossover: bool | None = None
    use_ml_scores_in_repair: bool | None = None
    use_ml_scores_in_local_search: bool | None = None
    ml_score_weight: float | None = None
    ris_score_weight: float | None = None
    fair_ris_score_weight: float | None = None
    fairness_bonus_weight: float = 0.2
    diversity_bonus_weight: float = 0.2
    community_balance_enabled: bool = True
    protected_group_balance_enabled: bool = True
    repair_mode: str = "balanced"


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
        "fim_stack",
        "pipeline_mode",
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
        "method_type",
        "repair_enabled",
        "swap_local_search_enabled",
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
        "use_community_features",
        "community_feature_mode",
        "allow_protected_features_in_ml",
        "ml_score_weight",
        "ris_score_weight",
        "fair_ris_score_weight",
        "fairness_bonus_weight",
        "community_diversity_weight",
        "community_balance_enabled",
        "protected_group_balance_enabled",
        "repair_mode",
        "initial_seed_source",
        "repaired_seed_sets",
        "successful_swaps",
        "final_community_coverage",
        "final_protected_group_coverage",
        "score_table_path",
        "embeddings_cache_path",
        "community_assignments_path",
        "community_sizes_path",
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


def _method_type(spec: FIMPermutationSpec) -> str:
    if spec.variant_family == "baseline" or spec.embedding_method in {"", "none"}:
        return "non_ml_baseline"
    if spec.name in {"line_fast_ml", "deepwalk_mlp"} or "fairness_collapse_risk=true" in spec.notes:
        return "weak_baseline"
    if spec.optimizer_mode == "hybrid_si_ea":
        return "ml_guided_hybrid"
    return "ml_guided"


def _embedding_artifact_path(embedding_artifact) -> str:
    if embedding_artifact is None:
        return ""
    output_paths = dict(getattr(embedding_artifact, "output_paths", {}) or {})
    for key in ("pickle", "csv"):
        if key in output_paths:
            return str(output_paths[key])
    if output_paths:
        return str(next(iter(output_paths.values())))
    return ""


def _write_community_artifacts(
    *,
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    community_result: CommunityDetectionResult,
    config: FIMPermutationRunConfig,
    spec: FIMPermutationSpec,
) -> dict[str, str]:
    output_dir = _permutation_output_dir(config.output_dir, dataset.name, protected_group_report.protected_attribute)
    if output_dir is None:
        return {"community_assignments_path": "", "community_sizes_path": ""}
    assignment_path = output_dir / f"{dataset.name}_{spec.name}_{spec.community_method}_community_assignments.csv"
    sizes_path = output_dir / f"{dataset.name}_{spec.name}_{spec.community_method}_community_sizes.csv"
    pd.DataFrame(
        [
            {
                "node_id": node_id,
                "community_id": community_id,
                "community_size": int(community_result.stats.community_sizes[community_id]),
            }
            for node_id, community_id in sorted(
                community_result.community_id_by_node.items(),
                key=lambda item: _sort_key(item[0]),
            )
        ]
    ).to_csv(assignment_path, index=False)
    pd.DataFrame(
        [
            {"community_id": community_id, "community_size": int(size)}
            for community_id, size in sorted(community_result.stats.community_sizes.items())
        ]
    ).to_csv(sizes_path, index=False)
    return {
        "community_assignments_path": str(assignment_path),
        "community_sizes_path": str(sizes_path),
    }


def _write_combined_score_table(
    *,
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
    score_frame: pd.DataFrame,
) -> str:
    output_dir = _permutation_output_dir(config.output_dir, dataset.name, protected_group_report.protected_attribute)
    if output_dir is None or score_frame.empty:
        return ""
    path = output_dir / f"{dataset.name}_{spec.name}_combined_candidate_scores.csv"
    score_frame.to_csv(path, index=False)
    return str(path)


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
    extra_fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    resolved_clustering_input_mode = (
        "none"
        if spec.clustering_method == "none"
        else str(clustering_input_mode or spec.clustering_input_mode)
    )
    row = {
        "stack_name": spec.name,
        "fim_stack": spec.name,
        "pipeline_mode": "ml_guided_community_siea" if spec.runner_kind in {"ranked_hybrid", "no_ml_hybrid"} else "baseline",
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
        "method_type": _method_type(spec),
        "repair_enabled": spec.optimizer_mode in {"hybrid_si_ea", "local_search"},
        "swap_local_search_enabled": int(config.local_search_steps) > 0,
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
        "use_community_features": bool(config.use_community_features_for_ml),
        "community_feature_mode": str(config.community_feature_mode),
        "allow_protected_features_in_ml": bool(config.allow_protected_features_in_ml),
        "ml_score_weight": config.ml_score_weight if config.ml_score_weight is not None else spec.ranking_weight,
        "ris_score_weight": config.ris_score_weight if config.ris_score_weight is not None else spec.ris_weight,
        "fair_ris_score_weight": config.fair_ris_score_weight if config.fair_ris_score_weight is not None else spec.fair_ris_weight,
        "fairness_bonus_weight": float(config.fairness_bonus_weight),
        "community_diversity_weight": float(config.diversity_bonus_weight),
        "community_balance_enabled": bool(config.community_balance_enabled),
        "protected_group_balance_enabled": bool(config.protected_group_balance_enabled),
        "repair_mode": str(config.repair_mode),
        "initial_seed_source": "combined_candidate_score" if spec.optimizer_mode == "hybrid_si_ea" else spec.ranking_model,
        "repaired_seed_sets": pd.NA,
        "successful_swaps": pd.NA,
        "final_community_coverage": pd.NA,
        "final_protected_group_coverage": pd.NA,
        "score_table_path": "",
        "embeddings_cache_path": "",
        "community_assignments_path": "",
        "community_sizes_path": "",
        "notes": "; ".join(part for part in [spec.notes, notes] if part),
        "skip_reason": skip_reason,
        "skipped_reason": skip_reason,
    }
    row.update(_spread_dict(evaluation, int(config.budget)))
    if extra_fields:
        row.update(dict(extra_fields))
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
            "method_type": _method_type(spec),
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
            "use_community_features": bool(config.use_community_features_for_ml),
            "community_feature_mode": str(config.community_feature_mode),
            "allow_protected_features_in_ml": bool(config.allow_protected_features_in_ml),
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


def _combined_guidance_score_frame(
    spec: FIMPermutationSpec,
    ranking_scores: Mapping[Any, float] | None,
    ris_scores: Mapping[Any, float] | None,
    fair_ris_scores: Mapping[Any, float] | None,
    *,
    ml_score_weight: float | None = None,
    ris_score_weight: float | None = None,
    fair_ris_score_weight: float | None = None,
    fairness_bonus_scores: Mapping[Any, float] | None = None,
    diversity_bonus_scores: Mapping[Any, float] | None = None,
    fairness_bonus_weight: float = 0.0,
    diversity_bonus_weight: float = 0.0,
) -> pd.DataFrame:
    normalized_ranking = _normalize_score_map(ranking_scores or {})
    normalized_ris = _normalize_score_map(ris_scores or {})
    normalized_fair_ris = _normalize_score_map(fair_ris_scores or {})
    normalized_fairness_bonus = _normalize_score_map(fairness_bonus_scores or {})
    normalized_diversity_bonus = _normalize_score_map(diversity_bonus_scores or {})
    all_nodes = sorted(
        set(normalized_ranking)
        | set(normalized_ris)
        | set(normalized_fair_ris)
        | set(normalized_fairness_bonus)
        | set(normalized_diversity_bonus),
        key=_sort_key,
    )
    if not all_nodes:
        return pd.DataFrame(
            columns=[
                "node_id",
                "ml_score",
                "ris_score",
                "fair_ris_score",
                "fairness_bonus",
                "community_diversity_bonus",
                "combined_score",
            ]
        )
    resolved_ml_weight = spec.ranking_weight if ml_score_weight is None else float(ml_score_weight)
    resolved_ris_weight = spec.ris_weight if ris_score_weight is None else float(ris_score_weight)
    resolved_fair_ris_weight = spec.fair_ris_weight if fair_ris_score_weight is None else float(fair_ris_score_weight)
    rows: list[dict[str, object]] = []
    for node_id in all_nodes:
        ml_score = float(normalized_ranking.get(node_id, 0.0))
        ris_score = float(normalized_ris.get(node_id, 0.0))
        fair_ris_score = float(normalized_fair_ris.get(node_id, 0.0))
        fairness_bonus = float(normalized_fairness_bonus.get(node_id, 0.0))
        community_diversity_bonus = float(normalized_diversity_bonus.get(node_id, 0.0))
        combined_score = float(
            resolved_ml_weight * normalized_ranking.get(node_id, 0.0)
            + resolved_ris_weight * normalized_ris.get(node_id, 0.0)
            + resolved_fair_ris_weight * normalized_fair_ris.get(node_id, 0.0)
            + float(fairness_bonus_weight) * normalized_fairness_bonus.get(node_id, 0.0)
            + float(diversity_bonus_weight) * normalized_diversity_bonus.get(node_id, 0.0)
        )
        rows.append(
            {
                "node_id": node_id,
                "ml_score": ml_score,
                "ris_score": ris_score,
                "fair_ris_score": fair_ris_score,
                "fairness_bonus": fairness_bonus,
                "community_diversity_bonus": community_diversity_bonus,
                "combined_score": combined_score,
            }
        )
    return pd.DataFrame(rows)


def _combine_guidance_scores(
    spec: FIMPermutationSpec,
    ranking_scores: Mapping[Any, float] | None,
    ris_scores: Mapping[Any, float] | None,
    fair_ris_scores: Mapping[Any, float] | None,
    *,
    ml_score_weight: float | None = None,
    ris_score_weight: float | None = None,
    fair_ris_score_weight: float | None = None,
    fairness_bonus_scores: Mapping[Any, float] | None = None,
    diversity_bonus_scores: Mapping[Any, float] | None = None,
    fairness_bonus_weight: float = 0.0,
    diversity_bonus_weight: float = 0.0,
) -> dict[Any, float]:
    frame = _combined_guidance_score_frame(
        spec,
        ranking_scores,
        ris_scores,
        fair_ris_scores,
        ml_score_weight=ml_score_weight,
        ris_score_weight=ris_score_weight,
        fair_ris_score_weight=fair_ris_score_weight,
        fairness_bonus_scores=fairness_bonus_scores,
        diversity_bonus_scores=diversity_bonus_scores,
        fairness_bonus_weight=fairness_bonus_weight,
        diversity_bonus_weight=diversity_bonus_weight,
    )
    if frame.empty:
        return {}
    return {
        row.node_id: float(row.combined_score)
        for row in frame[["node_id", "combined_score"]].itertuples(index=False)
    }


def _feature_score_map(feature_frame: pd.DataFrame, column_name: str) -> dict[Any, float]:
    if column_name not in feature_frame.columns:
        return {}
    return {
        row.node_id: float(getattr(row, column_name))
        for row in feature_frame[["node_id", column_name]].itertuples(index=False)
    }


def _seed_community_coverage(seed_set: Sequence[Any], community_result) -> str:
    counts: dict[str, int] = {}
    for node_id in seed_set:
        community_id = community_result.community_id_by_node.get(node_id)
        key = str(community_id)
        counts[key] = counts.get(key, 0) + 1
    return json.dumps(dict(sorted(counts.items(), key=lambda item: item[0])))


def _seed_protected_group_coverage(seed_set: Sequence[Any], protected_group_report: ProtectedGroupReport) -> str:
    group_by_node = {
        node_id: group_name
        for group_name, node_ids in protected_group_report.protected_groups.items()
        for node_id in node_ids
    }
    counts = {str(group_name): 0 for group_name in protected_group_report.group_sizes}
    for node_id in seed_set:
        group_name = group_by_node.get(node_id)
        if group_name is not None:
            counts[str(group_name)] = counts.get(str(group_name), 0) + 1
    return json.dumps(dict(sorted(counts.items(), key=lambda item: item[0])))


def _optimizer_diagnostics_fields(
    optimization_result,
    protected_group_report: ProtectedGroupReport,
    community_result,
) -> dict[str, object]:
    return {
        "repaired_seed_sets": int(getattr(optimization_result, "repaired_seed_sets", 0)),
        "successful_swaps": int(getattr(optimization_result, "successful_swaps", 0)),
        "final_community_coverage": _seed_community_coverage(
            optimization_result.best_seed_set,
            community_result,
        ),
        "final_protected_group_coverage": _seed_protected_group_coverage(
            optimization_result.best_seed_set,
            protected_group_report,
        ),
    }


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
        use_community_features_for_ml=bool(config.use_community_features_for_ml),
        community_feature_mode=str(config.community_feature_mode),
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
        allow_protected_features_in_ml=bool(config.allow_protected_features_in_ml),
    )
    fairness_bonus_scores = _feature_score_map(feature_frame, "fraction_neighbors_in_undercovered_groups")
    diversity_bonus_scores = _feature_score_map(feature_frame, "neighboring_communities")
    score_frame = _combined_guidance_score_frame(
        spec,
        ranking_artifact.training_result.predicted_scores,
        None if ris_artifact is None else ris_artifact.global_scores,
        None if ris_artifact is None else ris_artifact.fair_scores,
        ml_score_weight=config.ml_score_weight,
        ris_score_weight=config.ris_score_weight,
        fair_ris_score_weight=config.fair_ris_score_weight,
        fairness_bonus_scores=fairness_bonus_scores,
        diversity_bonus_scores=diversity_bonus_scores,
        fairness_bonus_weight=float(config.fairness_bonus_weight),
        diversity_bonus_weight=float(config.diversity_bonus_weight),
    )
    combined_scores = {
        row.node_id: float(row.combined_score)
        for row in score_frame[["node_id", "combined_score"]].itertuples(index=False)
    }
    score_table_path = _write_combined_score_table(
        dataset=dataset,
        protected_group_report=protected_group_report,
        spec=spec,
        config=config,
        score_frame=score_frame,
    )
    community_paths = _write_community_artifacts(
        dataset=dataset,
        protected_group_report=protected_group_report,
        community_result=community_result,
        config=config,
        spec=spec,
    )
    return {
        "community_result": community_result,
        "embedding_artifact": embedding_artifact,
        "clustering_artifact": clustering_artifact,
        "feature_frame": feature_frame,
        "label_result": label_result,
        "ranking_artifact": ranking_artifact,
        "combined_guidance_scores": combined_scores,
        "score_table_path": score_table_path,
        "community_assignments_path": community_paths["community_assignments_path"],
        "community_sizes_path": community_paths["community_sizes_path"],
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
    fit_warnings = ranking_artifact.training_result.metadata.get("fit_warnings", [])
    if fit_warnings:
        notes_parts.append("training_warnings=" + " | ".join(str(value) for value in fit_warnings))
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
    community_paths = _write_community_artifacts(
        dataset=dataset,
        protected_group_report=protected_group_report,
        community_result=community_result,
        config=config,
        spec=spec,
    )
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
                extra_fields={
                    "final_community_coverage": _seed_community_coverage(refined_seed_set, community_result),
                    "final_protected_group_coverage": _seed_protected_group_coverage(refined_seed_set, protected_group_report),
                    "community_assignments_path": community_paths["community_assignments_path"],
                    "community_sizes_path": community_paths["community_sizes_path"],
                },
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
        local_search_steps=max(0, int(config.local_search_steps)),
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
    use_ml_initialization = True if config.use_ml_scores_in_initialization is None else bool(config.use_ml_scores_in_initialization)
    use_ml_mutation = True if config.use_ml_scores_in_mutation is None else bool(config.use_ml_scores_in_mutation)
    use_ml_crossover = True if config.use_ml_scores_in_crossover is None else bool(config.use_ml_scores_in_crossover)
    use_ml_repair = True if config.use_ml_scores_in_repair is None else bool(config.use_ml_scores_in_repair)
    use_ml_local_search = True if config.use_ml_scores_in_local_search is None else bool(config.use_ml_scores_in_local_search)
    protected_balance = bool(config.protected_group_balance_enabled)
    return HybridSIEAConfig(
        budget=int(config.budget),
        population_size=int(config.population_size),
        generations=int(config.generations),
        propagation_probability=float(config.propagation_probability),
        mc_runs=int(config.mc_runs_search),
        diffusion_model=spec.diffusion_model,
        lambda_weight=float(config.lambda_weight),
        random_seed=int(config.random_seed),
        local_search_steps=max(0, int(config.local_search_steps)),
        disable_local_search=int(config.local_search_steps) <= 0,
        disable_community_aware_mutation=not bool(config.community_balance_enabled),
        use_ml_scores_in_crossover=use_ml_crossover,
        ml_guidance_mode="two_tier",
        ml_primary_pool_ratio=0.50,
        ml_secondary_exploration_rate=0.10,
        ml_initialization_bias=0.25 if use_ml_initialization else 0.0,
        ml_initialization_primary_rate=0.90 if use_ml_initialization else 0.0,
        ml_mutation_primary_rate=0.80 if use_ml_mutation else 0.0,
        ml_repair_primary_rate=0.70 if use_ml_repair else 0.0,
        ml_local_search_primary_rate=0.60 if use_ml_local_search else 0.0,
        ml_mutation_bias_weight=0.20 if use_ml_mutation else 0.0,
        ml_repair_bias_weight=0.15 if use_ml_repair else 0.0,
        ml_local_search_bias_weight=0.25 if use_ml_local_search else 0.0,
        fairness_first_init_enabled=bool(spec.use_fair_ris and protected_balance),
        fairness_first_init_slots=min(2, int(config.budget)),
        fairness_first_init_weight=0.75 if spec.use_fair_ris and protected_balance else 0.0,
        weakest_group_k=3 if spec.use_fair_ris and protected_balance else 1,
        weakest_group_mutation_weight=0.35 if spec.use_fair_ris and protected_balance else 0.0,
        zero_group_bonus_weight=0.25 if spec.use_fair_ris and protected_balance else 0.0,
        bridge_to_weak_group_weight=0.15 if spec.use_fair_ris and protected_balance else 0.0,
        repair_fairness_weight=0.40 if spec.use_fair_ris and protected_balance and config.repair_mode != "basic" else 0.0,
        repair_bridge_weight=0.20 if spec.use_fair_ris and protected_balance and config.repair_mode != "basic" else 0.0,
        local_search_focus_mode="worst_group" if spec.use_fair_ris and protected_balance else "default",
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
                extra_fields={
                    **_optimizer_diagnostics_fields(
                        optimization_result,
                        protected_group_report,
                        shared["community_result"],
                    ),
                    "score_table_path": shared.get("score_table_path", ""),
                    "embeddings_cache_path": _embedding_artifact_path(shared.get("embedding_artifact")),
                    "community_assignments_path": shared.get("community_assignments_path", ""),
                    "community_sizes_path": shared.get("community_sizes_path", ""),
                },
            )
        ]
    )


def _run_no_ml_hybrid_stack(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
) -> pd.DataFrame:
    start = perf_counter()
    community_result = _community_result(dataset, spec, config)
    feature_frame = build_ranking_feature_frame(
        dataset=dataset,
        protected_group_report=protected_group_report,
        community_result=community_result,
        use_community_features_for_ml=bool(config.use_community_features_for_ml),
        community_feature_mode=str(config.community_feature_mode),
    )
    structural_scores = compute_structural_node_scores(feature_frame)
    fairness_bonus_scores = _feature_score_map(feature_frame, "fraction_neighbors_in_undercovered_groups")
    diversity_bonus_scores = _feature_score_map(feature_frame, "neighboring_communities")
    score_frame = _combined_guidance_score_frame(
        spec,
        structural_scores,
        None,
        None,
        ml_score_weight=config.ml_score_weight,
        fairness_bonus_scores=fairness_bonus_scores,
        diversity_bonus_scores=diversity_bonus_scores,
        fairness_bonus_weight=float(config.fairness_bonus_weight),
        diversity_bonus_weight=float(config.diversity_bonus_weight),
    )
    combined_scores = {
        row.node_id: float(row.combined_score)
        for row in score_frame[["node_id", "combined_score"]].itertuples(index=False)
    }
    score_table_path = _write_combined_score_table(
        dataset=dataset,
        protected_group_report=protected_group_report,
        spec=spec,
        config=config,
        score_frame=score_frame,
    )
    community_paths = _write_community_artifacts(
        dataset=dataset,
        protected_group_report=protected_group_report,
        community_result=community_result,
        config=config,
        spec=spec,
    )
    optimizer = HybridSIEAOptimizer(
        dataset=dataset,
        protected_group_report=protected_group_report,
        community_result=community_result,
        config=_hybrid_optimizer_config(spec, config),
        node_scores=structural_scores,
        ml_node_scores=combined_scores,
    )
    optimization_result = optimizer.optimize()
    search_runtime = perf_counter() - start
    final_eval = _final_evaluate(
        dataset,
        protected_group_report,
        optimization_result.best_seed_set,
        spec.diffusion_model,
        config,
    )
    quality = compute_community_quality_metrics(dataset.graph, community_result)
    notes = f"communities={quality.num_communities}; modularity={quality.modularity:.6f}; scoring=no_ml_structural_community"
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
                method="community_structural+hybrid_si_ea",
                notes=notes,
                extra_fields={
                    **_optimizer_diagnostics_fields(
                        optimization_result,
                        protected_group_report,
                        community_result,
                    ),
                    "score_table_path": score_table_path,
                    "community_assignments_path": community_paths["community_assignments_path"],
                    "community_sizes_path": community_paths["community_sizes_path"],
                },
            )
        ]
    )


_RUNNERS: Mapping[str, Callable[[LoadedDataset, ProtectedGroupReport, FIMPermutationSpec, FIMPermutationRunConfig], pd.DataFrame]] = {
    "community_aware_fair_greedy": _run_community_aware_fair_greedy,
    "experiment_runner_gnn_ris": _run_experiment_runner_gnn_ris_stack,
    "ranked_greedy": _run_ranked_greedy_stack,
    "ranked_maximin": _run_ranked_maximin_stack,
    "ranked_hybrid": _run_ranked_hybrid_stack,
    "no_ml_hybrid": _run_no_ml_hybrid_stack,
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
        if "pipeline_mode" in row.index and str(row.get("pipeline_mode", "")).strip():
            lines.append(
                "   "
                f"pipeline: fim_stack={row.get('fim_stack', row.get('stack_name'))} | "
                f"mode={row.get('pipeline_mode')} | "
                f"method_type={row.get('method_type')} | "
                f"repair={row.get('repair_enabled')} | "
                f"swap_local_search={row.get('swap_local_search_enabled')}"
            )
            lines.append(
                "   "
                f"features: use_community_features={row.get('use_community_features')} | "
                f"community_feature_mode={row.get('community_feature_mode')} | "
                f"allow_protected_features_in_ml={row.get('allow_protected_features_in_ml')}"
            )
            lines.append(
                "   "
                f"guidance: ml={row.get('ml_score_weight')} | "
                f"ris={row.get('ris_score_weight')} | "
                f"fair_ris={row.get('fair_ris_score_weight')} | "
                f"fairness={row.get('fairness_bonus_weight')} | "
                f"community_diversity={row.get('community_diversity_weight')}"
            )
            lines.append(
                "   "
                f"optimizer_diagnostics: initial_seed_source={row.get('initial_seed_source')} | "
                f"repaired_seed_sets={row.get('repaired_seed_sets')} | "
                f"successful_swaps={row.get('successful_swaps')} | "
                f"community_coverage={row.get('final_community_coverage')} | "
                f"protected_group_coverage={row.get('final_protected_group_coverage')}"
            )
            lines.append(
                "   "
                f"artifacts: score_table={row.get('score_table_path')} | "
                f"embeddings={row.get('embeddings_cache_path')} | "
                f"community_assignments={row.get('community_assignments_path')}"
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
    node_count = dataset.graph.number_of_nodes()
    if int(config.budget) > int(node_count):
        raise ValueError(f"Budget {config.budget} exceeds graph node count {node_count} for dataset '{dataset.name}'.")
    protected_group_report = verify_protected_groups(dataset, config.protected_attribute)
    return run_fim_permutation_benchmark(
        dataset=dataset,
        protected_group_report=protected_group_report,
        config=config,
        permutations=permutations,
    )
