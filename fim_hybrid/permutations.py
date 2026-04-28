"""Named end-to-end FIM stack comparisons with shared final evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
from time import perf_counter
import traceback
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
from .safe_math import safe_minmax_normalize
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
PROFESSOR_PRIORITY = "professor_priority"
FAIRNESS_FIRST_PRIORITY = "fairness_first_priority"
_PRIORITY_POLICIES = {PROFESSOR_PRIORITY, FAIRNESS_FIRST_PRIORITY}
_RANKING_MODEL_TRAINING_ALIASES = {
    "graphsage_plus_fair_ris": "graphsage",
    "gcn_plus_fair_ris": "gcn",
}
_FAIRNESS_FIRST_STACKS = {
    "fairness_first_scalable_ml_siea",
    "node2vec_xgboost_fair_siea",
    "gcn_fair_siea",
    "community_fair_greedy_baseline",
    "community_aware_fair_greedy",
    "graphsage_community_siea",
    "gcn_community_siea",
    "node2vec_xgboost_community_siea",
    "graphsage_fair_ris_hybrid",
    "gcn_fair_ris_hybrid",
    "node2vec_xgboost",
}


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _training_ranking_model(model_name: str) -> str:
    return _RANKING_MODEL_TRAINING_ALIASES.get(str(model_name).strip().lower(), str(model_name).strip().lower())


def _normalize_ranking_policy(policy: object) -> str:
    value = str(policy).strip().lower()
    if value == FAIRNESS_FIRST_PRIORITY:
        return PROFESSOR_PRIORITY
    return value or "fim_default"


def _normalized_seed_set(seed_set: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(sorted(tuple(seed_set), key=_sort_key))


def _seed_tuple_key(seed_set: Iterable[Any]) -> tuple[tuple[str, str], ...]:
    return tuple(_sort_key(node_id) for node_id in _normalized_seed_set(seed_set))


def _normalize_score_map(scores: Mapping[Any, float]) -> dict[Any, float]:
    if not scores:
        return {}
    ordered_nodes = sorted(scores, key=_sort_key)
    values = [float(scores[node_id]) for node_id in ordered_nodes]
    normalized = safe_minmax_normalize(
        values,
        default=0.0,
        context="permutation score normalization",
    )
    return {
        node_id: float(score)
        for node_id, score in zip(ordered_nodes, normalized, strict=True)
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
    ranking_policy: str = "fim_default"
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
    debug_errors: bool = False
    raise_errors: bool = False
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
    weak_group_bonus_weight: float | None = None
    diversity_bonus_weight: float = 0.2
    protected_group_coverage_weight: float = 0.0
    ranking_policy: str = "fim_default"
    min_f_score: float = 0.0
    min_mf: float = 0.0001
    max_dcv: float = 0.25
    min_fraction_groups_covered: float = 0.80
    fairness_close_threshold: float = 0.003
    scalability_required: bool = True
    runtime_tiebreak_only: bool = True
    warn_only_fairness_gates: bool = False
    fscore_weight: float = 3.0
    mf_weight: float = 2.0
    dcv_weight: float = 2.0
    group_coverage_weight: float = 1.0
    scalability_weight: float = 0.5
    spread_weight: float = 0.5
    runtime_penalty_weight: float = 0.05
    runtime_weight: float = 0.05
    fairness_tolerance_dcv: float = 0.005
    fairness_tolerance_fscore_drop: float = 0.001
    use_fairness_first_swap_acceptance: bool = False
    use_professor_priority_fitness: bool = False
    fair_greedy_objective: str = "default"
    dcv_penalty_weight: float = 2.5
    community_coverage_weight: float = 0.5
    use_ris_greedy_approximation: bool = False
    scalability_mode: str = "auto"
    max_candidate_pool_size: int = 500
    candidate_pool_fraction: float = 0.30
    adaptive_ris_rr_sets: bool = True
    community_cache: bool = True
    ris_cache: bool = True
    spread_proxy_weight: float = 0.0
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
        "community_fair_greedy_baseline": FIMPermutationSpec(
            name="community_fair_greedy_baseline",
            description=(
                "Leiden communities with fairness-weighted greedy search, swap local search, "
                "and fairness-first priority ranking."
            ),
            runner_kind="community_aware_fair_greedy",
            diffusion_model="ic",
            community_method="leiden",
            ranking_model="fairness_weighted_greedy",
            optimizer_mode="local_search",
            fairness_objective="f_score",
            ranking_policy=FAIRNESS_FIRST_PRIORITY,
            variant_family="baseline",
        ),
        "fairness_first_scalable_ml_siea": FIMPermutationSpec(
            name="fairness_first_scalable_ml_siea",
            description=(
                "Leiden communities with GraphSAGE guidance, Fair RIS search guidance, "
                "hybrid SI+EA, fairness-first repair, swap search, and final Monte Carlo evaluation."
            ),
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            embedding_method="graphsage",
            ranking_model="graphsage_plus_fair_ris",
            optimizer_mode="hybrid_si_ea",
            debias_mode="worst_group_boost",
            fairness_objective="f_score",
            ranking_policy=FAIRNESS_FIRST_PRIORITY,
            variant_family="ml",
            candidate_top_fraction=None,
            use_ris_guidance=True,
            use_fair_ris=True,
            ranking_weight=0.8,
            ris_weight=0.8,
            fair_ris_weight=1.2,
        ),
        "node2vec_xgboost_fair_siea": FIMPermutationSpec(
            name="node2vec_xgboost_fair_siea",
            description=(
                "Leiden communities with Node2Vec embeddings, XGBoost ranking, Fair RIS guidance, "
                "hybrid SI+EA, and fairness-first priority ranking."
            ),
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            embedding_method="node2vec",
            ranking_model="xgboost",
            optimizer_mode="hybrid_si_ea",
            debias_mode="worst_group_boost",
            fairness_objective="f_score",
            ranking_policy=FAIRNESS_FIRST_PRIORITY,
            variant_family="ml",
            candidate_top_fraction=None,
            use_ris_guidance=True,
            use_fair_ris=True,
            ranking_weight=0.8,
            ris_weight=0.8,
            fair_ris_weight=1.2,
        ),
        "gcn_fair_siea": FIMPermutationSpec(
            name="gcn_fair_siea",
            description=(
                "Leiden communities with GCN guidance, Fair RIS search guidance, hybrid SI+EA, "
                "and fairness-first priority ranking."
            ),
            runner_kind="ranked_hybrid",
            diffusion_model="ic",
            community_method="leiden",
            spread_estimator_search="fairness_aware_ris",
            embedding_method="gcn",
            ranking_model="gcn_plus_fair_ris",
            optimizer_mode="hybrid_si_ea",
            debias_mode="worst_group_boost",
            fairness_objective="f_score",
            ranking_policy=FAIRNESS_FIRST_PRIORITY,
            variant_family="ml",
            candidate_top_fraction=None,
            use_ris_guidance=True,
            use_fair_ris=True,
            ranking_weight=0.8,
            ris_weight=0.8,
            fair_ris_weight=1.2,
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
        "budget",
        "diffusion_model",
        "ranking_policy",
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
        "scalability_mode",
        "scalability_pass",
        "candidate_pool_size",
        "ris_num_rr_sets",
        "effective_ris_num_rr_sets",
        "zero_covered_groups_count",
        "fraction_groups_covered",
        "protected_group_coverage_summary",
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
        "weak_group_bonus_weight",
        "community_diversity_weight",
        "protected_group_coverage_weight",
        "spread_proxy_weight",
        "community_balance_enabled",
        "protected_group_balance_enabled",
        "repair_mode",
        "initial_seed_source",
        "repaired_seed_sets",
        "successful_swaps",
        "final_community_coverage",
        "final_seed_count_per_community",
        "community_coverage_ratio",
        "final_protected_group_coverage",
        "num_communities",
        "community_modularity",
        "score_table_path",
        "embeddings_cache_path",
        "embedding_cache_status",
        "community_cache_status",
        "ris_cache_status",
        "score_cache_status",
        "fitness_cache_hits",
        "marginal_cache_hits",
        "swap_cache_hits",
        "community_assignments_path",
        "community_sizes_path",
        "diagnostics_path",
        "candidate_score_components_constant",
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


def _effective_ranking_policy(spec: FIMPermutationSpec, config: FIMPermutationRunConfig) -> str:
    config_policy = _normalize_ranking_policy(config.ranking_policy)
    if config_policy != "fim_default":
        return config_policy
    return _normalize_ranking_policy(spec.ranking_policy)


def _fairness_first_enabled(spec: FIMPermutationSpec, config: FIMPermutationRunConfig) -> bool:
    return (
        _effective_ranking_policy(spec, config) in _PRIORITY_POLICIES
        or spec.name in _FAIRNESS_FIRST_STACKS
    )


def _weak_group_bonus_weight(config: FIMPermutationRunConfig) -> float:
    if config.weak_group_bonus_weight is not None:
        return float(config.weak_group_bonus_weight)
    return float(config.fairness_bonus_weight)


def _professor_priority_weight_defaults(config: FIMPermutationRunConfig, spec: FIMPermutationSpec) -> dict[str, float]:
    if _fairness_first_enabled(spec, config):
        return {
            "ml_score_weight": 0.7,
            "ris_score_weight": 0.7,
            "fair_ris_score_weight": 1.2,
            "weak_group_bonus_weight": 1.0,
            "protected_group_coverage_weight": 1.0,
            "community_diversity_weight": 0.5,
            "spread_proxy_weight": 0.4,
        }
    return {
        "ml_score_weight": spec.ranking_weight,
        "ris_score_weight": spec.ris_weight,
        "fair_ris_score_weight": spec.fair_ris_weight,
        "weak_group_bonus_weight": _weak_group_bonus_weight(config),
        "protected_group_coverage_weight": float(config.protected_group_coverage_weight),
        "community_diversity_weight": float(config.diversity_bonus_weight),
        "spread_proxy_weight": float(config.spread_proxy_weight),
    }


def _resolved_candidate_score_weights(spec: FIMPermutationSpec, config: FIMPermutationRunConfig) -> dict[str, float]:
    defaults = _professor_priority_weight_defaults(config, spec)
    return {
        "ml_score_weight": defaults["ml_score_weight"] if config.ml_score_weight is None else float(config.ml_score_weight),
        "ris_score_weight": defaults["ris_score_weight"] if config.ris_score_weight is None else float(config.ris_score_weight),
        "fair_ris_score_weight": defaults["fair_ris_score_weight"] if config.fair_ris_score_weight is None else float(config.fair_ris_score_weight),
        "weak_group_bonus_weight": defaults["weak_group_bonus_weight"] if config.weak_group_bonus_weight is None else float(config.weak_group_bonus_weight),
        "protected_group_coverage_weight": defaults["protected_group_coverage_weight"] if config.protected_group_coverage_weight == 0.0 and _fairness_first_enabled(spec, config) else float(config.protected_group_coverage_weight),
        "community_diversity_weight": defaults["community_diversity_weight"] if config.diversity_bonus_weight == 0.2 and _fairness_first_enabled(spec, config) else float(config.diversity_bonus_weight),
        "spread_proxy_weight": defaults["spread_proxy_weight"] if config.spread_proxy_weight == 0.0 and _fairness_first_enabled(spec, config) else float(config.spread_proxy_weight),
    }


def _scalability_pass(dataset: LoadedDataset, config: FIMPermutationRunConfig, candidate_pool_size: int | None) -> bool:
    if str(config.scalability_mode) == "off":
        return True
    if candidate_pool_size is None:
        return True
    node_count = max(1, int(dataset.graph.number_of_nodes()))
    return int(candidate_pool_size) <= min(int(config.max_candidate_pool_size), node_count)


def _effective_candidate_pool_size(dataset: LoadedDataset, config: FIMPermutationRunConfig) -> int:
    node_count = int(dataset.graph.number_of_nodes())
    if str(config.scalability_mode) == "off":
        return node_count
    adaptive_size = int(math.ceil(max(0.0, float(config.candidate_pool_fraction)) * float(node_count)))
    return min(node_count, max(int(config.budget), min(int(config.max_candidate_pool_size), node_count, adaptive_size)))


def _effective_ris_rr_sets(dataset: LoadedDataset, config: FIMPermutationRunConfig) -> int:
    base_rr_sets = max(1, int(config.ris_num_rr_sets))
    if not bool(config.adaptive_ris_rr_sets) or int(dataset.graph.number_of_nodes()) < 1000:
        return base_rr_sets
    moderate_multiplier = 1.5 if str(config.scalability_mode) == "large_graph" else 1.25
    return int(min(max(base_rr_sets, math.ceil(base_rr_sets * moderate_multiplier)), max(base_rr_sets, 512)))


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


_SCORE_COMPONENT_COLUMNS = (
    "ml_score",
    "ris_score",
    "fair_ris_score",
    "fairness_bonus",
    "weak_group_bonus",
    "community_diversity_bonus",
    "protected_group_coverage_bonus",
    "spread_proxy_score",
    "combined_score",
)


def _is_constant_score_component(score_frame: pd.DataFrame, column_name: str) -> bool | None:
    if score_frame is None or score_frame.empty or column_name not in score_frame.columns:
        return None
    numeric_values = pd.to_numeric(score_frame[column_name], errors="coerce").dropna()
    if numeric_values.empty:
        return True
    return float(numeric_values.max()) <= float(numeric_values.min())


def _score_component_diagnostics(score_frame: pd.DataFrame | None) -> dict[str, object]:
    component_flags = {
        column_name: constant
        for column_name in _SCORE_COMPONENT_COLUMNS
        if (constant := _is_constant_score_component(score_frame, column_name)) is not None
    }
    all_constant = None if not component_flags else bool(all(component_flags.values()))
    return {
        "all_candidate_score_components_constant": all_constant,
        "constant_components": component_flags,
    }


def _diagnostics_payload(
    *,
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
    community_result: CommunityDetectionResult | None = None,
    score_frame: pd.DataFrame | None = None,
    score_table_path: str = "",
) -> dict[str, object]:
    community_sizes = (
        list(community_result.stats.community_sizes.values())
        if community_result is not None
        else []
    )
    payload: dict[str, object] = {
        "stack_name": spec.name,
        "dataset": dataset.name,
        "protected_attribute": protected_group_report.protected_attribute,
        "budget": int(config.budget),
        "num_nodes": int(dataset.graph.number_of_nodes()),
        "num_edges": int(dataset.graph.number_of_edges()),
        "protected_group_counts": dict(protected_group_report.group_sizes),
        "num_communities": int(community_result.stats.num_communities) if community_result is not None else None,
        "smallest_community_size": int(min(community_sizes)) if community_sizes else None,
        "largest_community_size": int(max(community_sizes)) if community_sizes else None,
        "score_table_path": score_table_path,
    }
    payload.update(_score_component_diagnostics(score_frame))
    return payload


def _diagnostics_path(
    *,
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
) -> Path | None:
    output_dir = _permutation_output_dir(config.output_dir, dataset.name, protected_group_report.protected_attribute)
    if output_dir is None:
        return None
    return output_dir / f"{dataset.name}_{spec.name}_diagnostics.json"


def _print_diagnostics(payload: Mapping[str, object]) -> None:
    constant_components = payload.get("constant_components", {})
    if isinstance(constant_components, Mapping) and constant_components:
        score_constant_text = json.dumps(dict(constant_components), sort_keys=True)
    else:
        score_constant_text = "n/a"
    print(
        "Benchmark diagnostics | "
        f"stack={payload.get('stack_name')} | "
        f"dataset={payload.get('dataset')} | "
        f"protected_attribute={payload.get('protected_attribute')} | "
        f"budget={payload.get('budget')} | "
        f"nodes={payload.get('num_nodes')} | "
        f"edges={payload.get('num_edges')} | "
        f"protected_group_counts={json.dumps(payload.get('protected_group_counts', {}), sort_keys=True)} | "
        f"communities={payload.get('num_communities')} | "
        f"smallest_community_size={payload.get('smallest_community_size')} | "
        f"largest_community_size={payload.get('largest_community_size')} | "
        f"all_candidate_score_components_constant={payload.get('all_candidate_score_components_constant')} | "
        f"constant_components={score_constant_text}"
    )


def _emit_stack_diagnostics(
    *,
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
    community_result: CommunityDetectionResult | None = None,
    score_frame: pd.DataFrame | None = None,
    score_table_path: str = "",
) -> dict[str, object]:
    payload = _diagnostics_payload(
        dataset=dataset,
        protected_group_report=protected_group_report,
        spec=spec,
        config=config,
        community_result=community_result,
        score_frame=score_frame,
        score_table_path=score_table_path,
    )
    path = _diagnostics_path(
        dataset=dataset,
        protected_group_report=protected_group_report,
        spec=spec,
        config=config,
    )
    if path is not None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _print_diagnostics(payload)
    return {
        "diagnostics_path": "" if path is None else str(path),
        "candidate_score_components_constant": payload["all_candidate_score_components_constant"],
    }


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


def _community_cache_path(
    output_dir: Path | None,
    dataset_name: str,
    protected_attribute: str,
    stack_name: str,
    community_method: str,
) -> Path | None:
    attribute_dir = _permutation_output_dir(output_dir, dataset_name, protected_attribute)
    if attribute_dir is None:
        return None
    return attribute_dir / f"{dataset_name}_{stack_name}_{community_method}_community_cache.pkl"


def _ris_cache_path(
    output_dir: Path | None,
    dataset_name: str,
    protected_attribute: str,
    stack_name: str,
    rr_sets: int,
) -> Path | None:
    attribute_dir = _permutation_output_dir(output_dir, dataset_name, protected_attribute)
    if attribute_dir is None:
        return None
    return attribute_dir / f"{dataset_name}_{stack_name}_ris_rr{int(rr_sets)}_cache.pkl"


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
    cache_path = _community_cache_path(
        config.output_dir,
        dataset.name,
        config.protected_attribute,
        spec.name,
        spec.community_method,
    )
    if bool(config.community_cache) and cache_path is not None and cache_path.exists():
        try:
            cached = pd.read_pickle(cache_path)
            if set(cached.community_id_by_node) == set(dataset.graph.nodes()):
                cached.metadata["community_cache_status"] = "hit"
                return cached
        except Exception:
            pass
    result = detect_communities(
        dataset.graph,
        method=spec.community_method,
        seed=int(config.random_seed),
        input_mode=spec.community_input_mode,
    )
    result.metadata["community_cache_status"] = "miss" if bool(config.community_cache) else "disabled"
    if bool(config.community_cache) and cache_path is not None:
        try:
            pd.to_pickle(result, cache_path)
        except Exception:
            result.metadata["community_cache_status"] = "miss_unwritten"
    return result


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
        "budget": int(config.budget),
        "diffusion_model": spec.diffusion_model,
        "ranking_policy": _effective_ranking_policy(spec, config),
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
        "scalability_mode": str(config.scalability_mode),
        "scalability_pass": pd.NA,
        "candidate_pool_size": pd.NA,
        "ris_num_rr_sets": int(config.ris_num_rr_sets),
        "effective_ris_num_rr_sets": int(config.ris_num_rr_sets),
        "mc_runs_search": int(config.mc_runs_search),
        "mc_runs_eval": int(config.mc_runs_eval),
        "key_enabled_modules": _key_enabled_modules(spec, resolved_clustering_input_mode),
        "use_community_features": bool(config.use_community_features_for_ml),
        "community_feature_mode": str(config.community_feature_mode),
        "allow_protected_features_in_ml": bool(config.allow_protected_features_in_ml),
        "ml_score_weight": _resolved_candidate_score_weights(spec, config)["ml_score_weight"],
        "ris_score_weight": _resolved_candidate_score_weights(spec, config)["ris_score_weight"],
        "fair_ris_score_weight": _resolved_candidate_score_weights(spec, config)["fair_ris_score_weight"],
        "fairness_bonus_weight": float(config.fairness_bonus_weight),
        "weak_group_bonus_weight": _resolved_candidate_score_weights(spec, config)["weak_group_bonus_weight"],
        "community_diversity_weight": _resolved_candidate_score_weights(spec, config)["community_diversity_weight"],
        "protected_group_coverage_weight": _resolved_candidate_score_weights(spec, config)["protected_group_coverage_weight"],
        "spread_proxy_weight": _resolved_candidate_score_weights(spec, config)["spread_proxy_weight"],
        "community_balance_enabled": bool(config.community_balance_enabled),
        "protected_group_balance_enabled": bool(config.protected_group_balance_enabled),
        "repair_mode": str(config.repair_mode),
        "initial_seed_source": "combined_candidate_score" if spec.optimizer_mode == "hybrid_si_ea" else spec.ranking_model,
        "repaired_seed_sets": pd.NA,
        "successful_swaps": pd.NA,
        "final_community_coverage": pd.NA,
        "final_seed_count_per_community": pd.NA,
        "community_coverage_ratio": pd.NA,
        "final_protected_group_coverage": pd.NA,
        "protected_group_coverage_summary": pd.NA,
        "num_communities": pd.NA,
        "community_modularity": pd.NA,
        "score_table_path": "",
        "embeddings_cache_path": "",
        "embedding_cache_status": "",
        "community_cache_status": "",
        "ris_cache_status": "",
        "score_cache_status": "",
        "fitness_cache_hits": pd.NA,
        "marginal_cache_hits": pd.NA,
        "swap_cache_hits": pd.NA,
        "community_assignments_path": "",
        "community_sizes_path": "",
        "diagnostics_path": "",
        "candidate_score_components_constant": pd.NA,
        "notes": "; ".join(part for part in [spec.notes, notes] if part),
        "skip_reason": skip_reason,
        "skipped_reason": skip_reason,
    }
    row.update(_spread_dict(evaluation, int(config.budget)))
    row.update(_fairness_coverage_fields(evaluation))
    if extra_fields:
        row.update(dict(extra_fields))
    candidate_pool_raw = pd.to_numeric(pd.Series([row.get("candidate_pool_size", pd.NA)]), errors="coerce").iloc[0]
    row["scalability_pass"] = _scalability_pass(
        dataset,
        config,
        None if pd.isna(candidate_pool_raw) else int(candidate_pool_raw),
    )
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
            "budget": int(config.budget),
            "diffusion_model": spec.diffusion_model,
            "ranking_policy": _effective_ranking_policy(spec, config),
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
            "scalability_mode": str(config.scalability_mode),
            "ris_num_rr_sets": int(config.ris_num_rr_sets),
            "effective_ris_num_rr_sets": _effective_ris_rr_sets(dataset, config),
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
    weak_group_bonus_scores: Mapping[Any, float] | None = None,
    diversity_bonus_scores: Mapping[Any, float] | None = None,
    protected_group_coverage_scores: Mapping[Any, float] | None = None,
    spread_proxy_scores: Mapping[Any, float] | None = None,
    fairness_bonus_weight: float = 0.0,
    weak_group_bonus_weight: float | None = None,
    diversity_bonus_weight: float = 0.0,
    protected_group_coverage_weight: float = 0.0,
    spread_proxy_weight: float = 0.0,
) -> pd.DataFrame:
    normalized_ranking = _normalize_score_map(ranking_scores or {})
    normalized_ris = _normalize_score_map(ris_scores or {})
    normalized_fair_ris = _normalize_score_map(fair_ris_scores or {})
    normalized_weak_group_bonus = _normalize_score_map(weak_group_bonus_scores or fairness_bonus_scores or {})
    normalized_diversity_bonus = _normalize_score_map(diversity_bonus_scores or {})
    normalized_protected_group_coverage = _normalize_score_map(protected_group_coverage_scores or {})
    normalized_spread_proxy = _normalize_score_map(spread_proxy_scores or {})
    all_nodes = sorted(
        set(normalized_ranking)
        | set(normalized_ris)
        | set(normalized_fair_ris)
        | set(normalized_weak_group_bonus)
        | set(normalized_diversity_bonus)
        | set(normalized_protected_group_coverage)
        | set(normalized_spread_proxy),
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
                "weak_group_bonus",
                "community_diversity_bonus",
                "protected_group_coverage_bonus",
                "spread_proxy_score",
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
        weak_group_bonus = float(normalized_weak_group_bonus.get(node_id, 0.0))
        community_diversity_bonus = float(normalized_diversity_bonus.get(node_id, 0.0))
        protected_group_coverage_bonus = float(normalized_protected_group_coverage.get(node_id, 0.0))
        spread_proxy_score = float(normalized_spread_proxy.get(node_id, 0.0))
        resolved_weak_group_weight = (
            float(fairness_bonus_weight)
            if weak_group_bonus_weight is None
            else float(weak_group_bonus_weight)
        )
        combined_score = float(
            resolved_ml_weight * normalized_ranking.get(node_id, 0.0)
            + resolved_ris_weight * normalized_ris.get(node_id, 0.0)
            + resolved_fair_ris_weight * normalized_fair_ris.get(node_id, 0.0)
            + resolved_weak_group_weight * normalized_weak_group_bonus.get(node_id, 0.0)
            + float(diversity_bonus_weight) * normalized_diversity_bonus.get(node_id, 0.0)
            + float(protected_group_coverage_weight) * normalized_protected_group_coverage.get(node_id, 0.0)
            + float(spread_proxy_weight) * normalized_spread_proxy.get(node_id, 0.0)
        )
        rows.append(
            {
                "node_id": node_id,
                "ml_score": ml_score,
                "ris_score": ris_score,
                "fair_ris_score": fair_ris_score,
                "fairness_bonus": weak_group_bonus,
                "weak_group_bonus": weak_group_bonus,
                "community_diversity_bonus": community_diversity_bonus,
                "protected_group_coverage_bonus": protected_group_coverage_bonus,
                "spread_proxy_score": spread_proxy_score,
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
    weak_group_bonus_scores: Mapping[Any, float] | None = None,
    diversity_bonus_scores: Mapping[Any, float] | None = None,
    protected_group_coverage_scores: Mapping[Any, float] | None = None,
    spread_proxy_scores: Mapping[Any, float] | None = None,
    fairness_bonus_weight: float = 0.0,
    weak_group_bonus_weight: float | None = None,
    diversity_bonus_weight: float = 0.0,
    protected_group_coverage_weight: float = 0.0,
    spread_proxy_weight: float = 0.0,
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
        weak_group_bonus_scores=weak_group_bonus_scores,
        diversity_bonus_scores=diversity_bonus_scores,
        protected_group_coverage_scores=protected_group_coverage_scores,
        spread_proxy_scores=spread_proxy_scores,
        fairness_bonus_weight=fairness_bonus_weight,
        weak_group_bonus_weight=weak_group_bonus_weight,
        diversity_bonus_weight=diversity_bonus_weight,
        protected_group_coverage_weight=protected_group_coverage_weight,
        spread_proxy_weight=spread_proxy_weight,
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


def _protected_group_coverage_bonus_scores(
    feature_frame: pd.DataFrame,
    protected_group_report: ProtectedGroupReport,
) -> dict[Any, float]:
    group_by_node = {
        node_id: str(group_name)
        for group_name, node_ids in protected_group_report.protected_groups.items()
        for node_id in node_ids
    }
    largest_group = max((int(size) for size in protected_group_report.group_sizes.values()), default=1)
    scores: dict[Any, float] = {}
    for node_id in feature_frame["node_id"].tolist() if "node_id" in feature_frame.columns else []:
        group_name = group_by_node.get(node_id)
        group_size = int(protected_group_report.group_sizes.get(group_name, largest_group))
        scores[node_id] = float(largest_group) / float(max(1, group_size))
    return scores


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


def _seed_community_coverage_ratio(seed_set: Sequence[Any], community_result) -> float:
    covered_communities = {
        community_result.community_id_by_node[node_id]
        for node_id in seed_set
        if node_id in community_result.community_id_by_node
    }
    return float(len(covered_communities)) / float(max(1, int(community_result.stats.num_communities)))


def _fairness_coverage_fields(evaluation: SeedSetEvaluation) -> dict[str, object]:
    group_spread = dict(evaluation.fairness.group_spread)
    covered_groups = [group_name for group_name, value in group_spread.items() if float(value) > 0.0]
    total_groups = max(1, len(group_spread))
    fraction_groups_covered = float(len(covered_groups)) / float(total_groups)
    return {
        "zero_covered_groups_count": int(total_groups - len(covered_groups)),
        "fraction_groups_covered": fraction_groups_covered,
    }


def _optimizer_diagnostics_fields(
    optimization_result,
    protected_group_report: ProtectedGroupReport,
    community_result,
) -> dict[str, object]:
    community_coverage = _seed_community_coverage(
        optimization_result.best_seed_set,
        community_result,
    )
    protected_coverage = _seed_protected_group_coverage(
        optimization_result.best_seed_set,
        protected_group_report,
    )
    return {
        "repaired_seed_sets": int(getattr(optimization_result, "repaired_seed_sets", 0)),
        "successful_swaps": int(getattr(optimization_result, "successful_swaps", 0)),
        "final_community_coverage": community_coverage,
        "final_seed_count_per_community": community_coverage,
        "community_coverage_ratio": _seed_community_coverage_ratio(optimization_result.best_seed_set, community_result),
        "final_protected_group_coverage": protected_coverage,
        "protected_group_coverage_summary": protected_coverage,
        "candidate_pool_size": int(getattr(optimization_result, "candidate_pool_size", 0)),
        "fitness_cache_hits": int(getattr(optimization_result, "fitness_cache_hits", 0)),
        "marginal_cache_hits": int(getattr(optimization_result, "marginal_cache_hits", 0)),
        "swap_cache_hits": int(getattr(optimization_result, "swap_cache_hits", 0)),
    }


def _resolved_candidate_controls(
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
    dataset: LoadedDataset,
) -> tuple[float | None, int | None, int | None]:
    adaptive_max_nodes = _effective_candidate_pool_size(dataset, config)
    configured_max_nodes = spec.candidate_max_nodes if spec.candidate_max_nodes is not None else config.ranking_max_nodes
    if configured_max_nodes is None or _fairness_first_enabled(spec, config) or str(config.scalability_mode) != "off":
        max_nodes = adaptive_max_nodes
    else:
        max_nodes = int(configured_max_nodes)
    return (
        spec.candidate_top_fraction if spec.candidate_top_fraction is not None else config.ranking_top_fraction,
        spec.candidate_top_n if spec.candidate_top_n is not None else config.ranking_top_n,
        max_nodes,
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
    ris_cache_status = "disabled"
    effective_rr_sets = _effective_ris_rr_sets(dataset, config)
    if spec.use_ris_guidance:
        cache_path = _ris_cache_path(
            config.output_dir,
            dataset.name,
            protected_group_report.protected_attribute,
            spec.name,
            effective_rr_sets,
        )
        if bool(config.ris_cache) and cache_path is not None and cache_path.exists():
            try:
                ris_artifact = pd.read_pickle(cache_path)
                ris_cache_status = "hit"
            except Exception:
                ris_artifact = None
                ris_cache_status = "miss"
        if ris_artifact is None:
            ris_cache_status = "miss" if bool(config.ris_cache) else "disabled"
            ris_artifact = prepare_ris_guidance(
                dataset=dataset,
                protected_group_report=protected_group_report,
                propagation_probability=float(config.propagation_probability),
                feature_frame=base_feature_frame,
                output_dir=config.output_dir,
                protected_attribute=protected_group_report.protected_attribute,
                stack_name=spec.name,
                config=RISConfig(
                    num_rr_sets=effective_rr_sets,
                    random_seed=int(config.random_seed),
                    mode="weak_group_weighted" if spec.use_fair_ris else "global",
                    reuse_rr_sets=True,
                ),
            )
            if bool(config.ris_cache) and cache_path is not None:
                try:
                    pd.to_pickle(ris_artifact, cache_path)
                except Exception:
                    ris_cache_status = "miss_unwritten"
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
    top_fraction, top_n, max_nodes = _resolved_candidate_controls(spec, config, dataset)
    ranking_artifact = train_ranking_model(
        dataset=dataset,
        feature_frame=feature_frame,
        label_frame=label_result.label_frame,
        budget=int(config.budget),
        model_type=_training_ranking_model(spec.ranking_model),
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
    protected_group_coverage_scores = _protected_group_coverage_bonus_scores(feature_frame, protected_group_report)
    spread_proxy_scores = compute_structural_node_scores(feature_frame)
    score_weights = _resolved_candidate_score_weights(spec, config)
    score_frame = _combined_guidance_score_frame(
        spec,
        ranking_artifact.training_result.predicted_scores,
        None if ris_artifact is None else ris_artifact.global_scores,
        None if ris_artifact is None else ris_artifact.fair_scores,
        ml_score_weight=score_weights["ml_score_weight"],
        ris_score_weight=score_weights["ris_score_weight"],
        fair_ris_score_weight=score_weights["fair_ris_score_weight"],
        fairness_bonus_scores=fairness_bonus_scores,
        weak_group_bonus_scores=fairness_bonus_scores,
        diversity_bonus_scores=diversity_bonus_scores,
        protected_group_coverage_scores=protected_group_coverage_scores,
        spread_proxy_scores=spread_proxy_scores,
        fairness_bonus_weight=float(config.fairness_bonus_weight),
        weak_group_bonus_weight=score_weights["weak_group_bonus_weight"],
        diversity_bonus_weight=score_weights["community_diversity_weight"],
        protected_group_coverage_weight=score_weights["protected_group_coverage_weight"],
        spread_proxy_weight=score_weights["spread_proxy_weight"],
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
    diagnostics_fields = _emit_stack_diagnostics(
        dataset=dataset,
        protected_group_report=protected_group_report,
        spec=spec,
        config=config,
        community_result=community_result,
        score_frame=score_frame,
        score_table_path=score_table_path,
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
        "embedding_cache_status": "hit" if embedding_artifact.loaded_from_cache else ("disabled" if not config.use_embedding_cache else "miss"),
        "community_cache_status": community_result.metadata.get("community_cache_status", "disabled"),
        "ris_cache_status": ris_cache_status,
        "score_cache_status": "hit" if ranking_artifact.training_result.loaded_from_cache else ("disabled" if not config.use_score_cache else "miss"),
        "effective_ris_num_rr_sets": effective_rr_sets,
        "candidate_pool_size": len(ranking_artifact.training_result.candidate_nodes),
        "diagnostics_fields": diagnostics_fields,
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


def _shared_stack_reporting_fields(
    shared: Mapping[str, Any],
    quality_metrics,
) -> dict[str, object]:
    return {
        "candidate_pool_size": shared.get("candidate_pool_size", pd.NA),
        "effective_ris_num_rr_sets": shared.get("effective_ris_num_rr_sets", pd.NA),
        "num_communities": getattr(quality_metrics, "num_communities", pd.NA),
        "community_modularity": getattr(quality_metrics, "modularity", pd.NA),
        "embedding_cache_status": shared.get("embedding_cache_status", ""),
        "community_cache_status": shared.get("community_cache_status", ""),
        "ris_cache_status": shared.get("ris_cache_status", ""),
        "score_cache_status": shared.get("score_cache_status", ""),
    }


def _run_community_aware_fair_greedy(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
) -> pd.DataFrame:
    start = perf_counter()
    community_result = _community_result(dataset, spec, config)
    diagnostics_fields = _emit_stack_diagnostics(
        dataset=dataset,
        protected_group_report=protected_group_report,
        spec=spec,
        config=config,
        community_result=community_result,
    )
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
                    **diagnostics_fields,
                    "final_community_coverage": _seed_community_coverage(refined_seed_set, community_result),
                    "final_seed_count_per_community": _seed_community_coverage(refined_seed_set, community_result),
                    "community_coverage_ratio": _seed_community_coverage_ratio(refined_seed_set, community_result),
                    "final_protected_group_coverage": _seed_protected_group_coverage(refined_seed_set, protected_group_report),
                    "protected_group_coverage_summary": _seed_protected_group_coverage(refined_seed_set, protected_group_report),
                    "candidate_pool_size": dataset.graph.number_of_nodes(),
                    "num_communities": quality.num_communities,
                    "community_modularity": quality.modularity,
                    "community_cache_status": community_result.metadata.get("community_cache_status", "disabled"),
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
            "budget": int(config.budget),
            "ranking_policy": _effective_ranking_policy(spec, config),
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
            "scalability_mode": str(config.scalability_mode),
            "ris_num_rr_sets": int(config.ris_num_rr_sets),
            "effective_ris_num_rr_sets": int(config.ris_num_rr_sets),
            "candidate_pool_size": pd.NA,
            "scalability_pass": True,
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
                extra_fields={
                    **shared.get("diagnostics_fields", {}),
                    **_shared_stack_reporting_fields(shared, quality),
                    "final_community_coverage": _seed_community_coverage(seed_set, shared["community_result"]),
                    "final_seed_count_per_community": _seed_community_coverage(seed_set, shared["community_result"]),
                    "community_coverage_ratio": _seed_community_coverage_ratio(seed_set, shared["community_result"]),
                    "final_protected_group_coverage": _seed_protected_group_coverage(seed_set, protected_group_report),
                    "protected_group_coverage_summary": _seed_protected_group_coverage(seed_set, protected_group_report),
                    "score_table_path": shared.get("score_table_path", ""),
                    "embeddings_cache_path": _embedding_artifact_path(shared.get("embedding_artifact")),
                    "community_assignments_path": shared.get("community_assignments_path", ""),
                    "community_sizes_path": shared.get("community_sizes_path", ""),
                },
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
                extra_fields={
                    **shared.get("diagnostics_fields", {}),
                    **_shared_stack_reporting_fields(shared, quality),
                    "final_community_coverage": _seed_community_coverage(refined_seed_set, shared["community_result"]),
                    "final_seed_count_per_community": _seed_community_coverage(refined_seed_set, shared["community_result"]),
                    "community_coverage_ratio": _seed_community_coverage_ratio(refined_seed_set, shared["community_result"]),
                    "final_protected_group_coverage": _seed_protected_group_coverage(refined_seed_set, protected_group_report),
                    "protected_group_coverage_summary": _seed_protected_group_coverage(refined_seed_set, protected_group_report),
                    "score_table_path": shared.get("score_table_path", ""),
                    "embeddings_cache_path": _embedding_artifact_path(shared.get("embedding_artifact")),
                    "community_assignments_path": shared.get("community_assignments_path", ""),
                    "community_sizes_path": shared.get("community_sizes_path", ""),
                },
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
    fairness_first = _fairness_first_enabled(spec, config)
    fair_guidance = bool((spec.use_fair_ris or fairness_first) and protected_balance)
    return HybridSIEAConfig(
        budget=int(config.budget),
        population_size=int(config.population_size),
        generations=int(config.generations),
        propagation_probability=float(config.propagation_probability),
        mc_runs=int(config.mc_runs_search),
        diffusion_model=spec.diffusion_model,
        lambda_weight=float(config.lambda_weight),
        random_seed=int(config.random_seed),
        fitness_policy="fairness_first" if fairness_first else "f_score",
        fscore_weight=float(config.fscore_weight),
        mf_weight=float(config.mf_weight),
        dcv_weight=float(config.dcv_weight),
        spread_weight=float(config.spread_weight),
        runtime_penalty_weight=float(config.runtime_penalty_weight),
        fairness_tolerance_dcv=float(config.fairness_tolerance_dcv),
        fairness_tolerance_fscore_drop=float(config.fairness_tolerance_fscore_drop),
        use_fairness_first_swap_acceptance=bool(config.use_fairness_first_swap_acceptance or fairness_first),
        local_search_steps=max(0, int(config.local_search_steps)),
        disable_local_search=int(config.local_search_steps) <= 0,
        disable_community_aware_mutation=not bool(config.community_balance_enabled),
        use_ml_scores_in_crossover=use_ml_crossover,
        ml_guidance_mode="two_tier",
        ml_primary_pool_ratio=0.50,
        ml_secondary_exploration_rate=0.10,
        ml_initialization_bias=(0.40 if fairness_first else 0.25) if use_ml_initialization else 0.0,
        ml_initialization_primary_rate=0.90 if use_ml_initialization else 0.0,
        ml_mutation_primary_rate=0.80 if use_ml_mutation else 0.0,
        ml_repair_primary_rate=0.70 if use_ml_repair else 0.0,
        ml_local_search_primary_rate=0.60 if use_ml_local_search else 0.0,
        ml_mutation_bias_weight=0.20 if use_ml_mutation else 0.0,
        ml_repair_bias_weight=0.15 if use_ml_repair else 0.0,
        ml_local_search_bias_weight=0.25 if use_ml_local_search else 0.0,
        fairness_first_init_enabled=fair_guidance,
        fairness_first_init_slots=min(3 if fairness_first else 2, int(config.budget)) if fair_guidance else 0,
        fairness_first_init_weight=1.0 if fairness_first and fair_guidance else (0.75 if fair_guidance else 0.0),
        weakest_group_k=3 if fair_guidance else 1,
        weakest_group_mutation_weight=(1.0 if fairness_first else 0.35) if fair_guidance else 0.0,
        zero_group_bonus_weight=(0.50 if fairness_first else 0.25) if fair_guidance else 0.0,
        bridge_to_weak_group_weight=(0.30 if fairness_first else 0.15) if fair_guidance else 0.0,
        repair_fairness_weight=(0.75 if fairness_first else 0.40) if fair_guidance and config.repair_mode != "basic" else 0.0,
        repair_bridge_weight=(0.35 if fairness_first else 0.20) if fair_guidance and config.repair_mode != "basic" else 0.0,
        repair_diversity_weight=0.20 if fairness_first and bool(config.community_balance_enabled) else 0.0,
        local_search_focus_mode="worst_group" if fair_guidance else "default",
        local_search_bottom_k_groups=3,
        local_search_swap_trials=max(1, int(config.local_search_steps)),
        local_search_candidate_pool_size=max(4, int(config.swap_candidate_pool_size)),
        swap_candidate_pool_size=max(4, int(config.swap_candidate_pool_size)),
        enable_fitness_cache=True,
        enable_marginal_cache=fairness_first,
        enable_swap_cache=fairness_first,
        marginal_gain_scoring_enabled=fairness_first,
        marginal_gain_delta_mf_weight=1.0 if fairness_first else 0.0,
        marginal_gain_delta_dcv_weight=1.0 if fairness_first else 0.0,
        marginal_gain_spread_weight=0.25 if fairness_first else 0.0,
        local_search_delta_mf_weight=1.0 if fairness_first else 0.0,
        local_search_delta_dcv_weight=1.0 if fairness_first else 0.0,
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
                    **shared.get("diagnostics_fields", {}),
                    **_shared_stack_reporting_fields(shared, quality),
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
    protected_group_coverage_scores = _protected_group_coverage_bonus_scores(feature_frame, protected_group_report)
    score_frame = _combined_guidance_score_frame(
        spec,
        structural_scores,
        None,
        None,
        ml_score_weight=config.ml_score_weight,
        fairness_bonus_scores=fairness_bonus_scores,
        weak_group_bonus_scores=fairness_bonus_scores,
        diversity_bonus_scores=diversity_bonus_scores,
        protected_group_coverage_scores=protected_group_coverage_scores,
        fairness_bonus_weight=float(config.fairness_bonus_weight),
        weak_group_bonus_weight=_weak_group_bonus_weight(config),
        diversity_bonus_weight=float(config.diversity_bonus_weight),
        protected_group_coverage_weight=float(config.protected_group_coverage_weight),
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
    diagnostics_fields = _emit_stack_diagnostics(
        dataset=dataset,
        protected_group_report=protected_group_report,
        spec=spec,
        config=config,
        community_result=community_result,
        score_frame=score_frame,
        score_table_path=score_table_path,
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
                    **diagnostics_fields,
                    **_optimizer_diagnostics_fields(
                        optimization_result,
                        protected_group_report,
                        community_result,
                    ),
                    "num_communities": quality.num_communities,
                    "community_modularity": quality.modularity,
                    "community_cache_status": community_result.metadata.get("community_cache_status", "disabled"),
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


def _print_stack_exception(
    *,
    exc: Exception,
    spec: FIMPermutationSpec,
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    config: FIMPermutationRunConfig,
) -> None:
    print(
        "Stack failure traceback | "
        f"stack_name={spec.name} | "
        f"dataset={dataset.name} | "
        f"protected_attribute={protected_group_report.protected_attribute} | "
        f"budget={int(config.budget)}",
        file=sys.stderr,
    )
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _fairness_valid_mask(frame: pd.DataFrame, *, config: FIMPermutationRunConfig | None = None) -> pd.Series:
    min_f_score = 0.0 if config is None else float(config.min_f_score)
    min_mf = 0.0001 if config is None else float(config.min_mf)
    max_dcv = 0.25 if config is None else float(config.max_dcv)
    min_fraction = 0.80 if config is None else float(config.min_fraction_groups_covered)
    ok = frame["status"].astype(str).eq("ok") if "status" in frame.columns else pd.Series([True] * len(frame), index=frame.index)
    mask = (
        ok
        & (_numeric_series(frame, "f_score") >= min_f_score)
        & (_numeric_series(frame, "mf") > min_mf)
        & (_numeric_series(frame, "dcv", default=1.0) < max_dcv)
    )
    if "fraction_groups_covered" in frame.columns:
        fraction = pd.to_numeric(frame["fraction_groups_covered"], errors="coerce")
        mask &= fraction.isna() | (fraction >= min_fraction)
    return mask


def _fairness_priority_order(frame: pd.DataFrame, *, config: FIMPermutationRunConfig | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    sortable = frame.copy()
    sortable["_f_score_sort"] = _numeric_series(sortable, "f_score")
    sortable["_mf_sort"] = _numeric_series(sortable, "mf")
    sortable["_dcv_sort"] = _numeric_series(sortable, "dcv", default=1.0)
    sortable["_zero_groups_sort"] = _numeric_series(sortable, "zero_covered_groups_count", default=0.0)
    sortable["_fraction_groups_sort"] = _numeric_series(sortable, "fraction_groups_covered", default=1.0)
    scalability_values = sortable["scalability_pass"] if "scalability_pass" in sortable.columns else pd.Series([True] * len(sortable), index=sortable.index)
    sortable["_scalability_sort"] = scalability_values.fillna(True).map(
        lambda value: 1 if bool(value) and str(value) != "<NA>" else 0
    )
    sortable["_spread_sort"] = _numeric_series(sortable, "total_spread")
    sortable["_runtime_sort"] = _numeric_series(sortable, "runtime_seconds", default=float("inf"))
    valid_mask = _fairness_valid_mask(sortable, config=config)
    sortable["_fairness_valid_sort"] = valid_mask.astype(int)
    ordered = sortable.sort_values(
        [
            "_fairness_valid_sort",
            "_f_score_sort",
            "_mf_sort",
            "_dcv_sort",
            "_zero_groups_sort",
            "_fraction_groups_sort",
            "_scalability_sort",
            "_spread_sort",
            "_runtime_sort",
        ],
        ascending=[False, False, False, True, True, False, False, False, True],
        na_position="last",
    )
    return ordered.drop(columns=[column for column in ordered.columns if column.endswith("_sort")])


def _recommend_stack_name(frame: pd.DataFrame, mask: pd.Series, *, config: FIMPermutationRunConfig | None = None) -> str | None:
    candidates = frame.loc[mask].copy()
    if candidates.empty:
        return None
    if config is not None and str(config.ranking_policy) == FAIRNESS_FIRST_PRIORITY:
        ordered = _fairness_priority_order(candidates, config=config)
        return str(ordered.iloc[0]["stack_name"]) if not ordered.empty else None
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


def recommend_fim_stacks(frame: pd.DataFrame, config: FIMPermutationRunConfig | None = None) -> dict[str, str | None]:
    """Return the best stack names under the configured recommendation rules."""

    if frame.empty:
        return {
            "best_interpretable_baseline": None,
            "best_practical_ml_default": None,
            "best_exploratory_fairness_heavy": None,
            "best_fairness_quality_method": None,
            "best_scalable_method": None,
            "best_spread_method": None,
            "best_runtime_method": None,
            "final_professor_priority_recommendation": None,
        }
    ok_mask = frame["status"].astype(str) == "ok"
    debias_mask = ~frame["debias_mode"].astype(str).isin({"none", "off"})
    fairness_search_mask = frame["fairness_objective"].astype(str).eq("maximin") | frame["spread_estimator_search"].astype(str).eq("fairness_aware_ris")
    interpretable_mask = ok_mask & frame["variant_type"].astype(str).eq(_FIM_VARIANT_TYPES["baseline"])
    ml_mask = ok_mask & ~frame["ranking_model"].astype(str).isin(_BASELINE_RANKING_MODELS)
    fairness_mask = ok_mask & (debias_mask | fairness_search_mask)
    valid_fairness_mask = _fairness_valid_mask(frame, config=config) if config is not None else ok_mask
    strict_fairness_first = config is not None and str(config.ranking_policy) == FAIRNESS_FIRST_PRIORITY
    if not bool(valid_fairness_mask.any()) and not strict_fairness_first:
        valid_fairness_mask = ok_mask
    if strict_fairness_first:
        interpretable_mask &= valid_fairness_mask
        ml_mask &= valid_fairness_mask
        fairness_mask &= valid_fairness_mask
    scalability_values = frame["scalability_pass"] if "scalability_pass" in frame.columns else pd.Series([True] * len(frame), index=frame.index)
    scalability_mask = scalability_values.fillna(True).map(
        lambda value: str(value).strip().lower() not in {"false", "0", "no"}
    )
    scalable_mask = valid_fairness_mask & scalability_mask
    spread_candidates = frame.loc[valid_fairness_mask].copy()
    runtime_candidates = _fairness_priority_order(frame.loc[valid_fairness_mask].copy(), config=config) if bool(valid_fairness_mask.any()) else pd.DataFrame()
    runtime_close_mask = valid_fairness_mask
    if config is not None and not runtime_candidates.empty and bool(config.runtime_tiebreak_only):
        best_f_score = pd.to_numeric(runtime_candidates.iloc[0:1]["f_score"], errors="coerce").fillna(0.0).iloc[0]
        runtime_close_mask = valid_fairness_mask & (_numeric_series(frame, "f_score") >= float(best_f_score) - float(config.fairness_close_threshold))
    best_spread_method = None
    if not spread_candidates.empty:
        for column in ["total_spread", "f_score", "mf"]:
            spread_candidates[column] = pd.to_numeric(spread_candidates[column], errors="coerce")
        best_spread_method = str(
            spread_candidates.sort_values(["total_spread", "f_score", "mf"], ascending=[False, False, False], na_position="last").iloc[0]["stack_name"]
        )
    runtime_subset = frame.loc[runtime_close_mask].copy()
    best_runtime_method = None
    if not runtime_subset.empty:
        runtime_subset["_runtime_sort"] = _numeric_series(runtime_subset, "runtime_seconds", default=float("inf"))
        runtime_subset["_f_score_sort"] = _numeric_series(runtime_subset, "f_score")
        runtime_subset = runtime_subset.sort_values(
            ["_runtime_sort", "_f_score_sort"],
            ascending=[True, False],
            na_position="last",
        )
        best_runtime_method = str(runtime_subset.iloc[0]["stack_name"])
    final_professor = _recommend_stack_name(frame, valid_fairness_mask, config=config)
    return {
        "best_interpretable_baseline": _recommend_stack_name(frame, interpretable_mask, config=config),
        "best_practical_ml_default": _recommend_stack_name(frame, ml_mask, config=config),
        "best_exploratory_fairness_heavy": _recommend_stack_name(frame, fairness_mask, config=config),
        "best_fairness_quality_method": final_professor,
        "best_scalable_method": _recommend_stack_name(frame, scalable_mask, config=config),
        "best_spread_method": best_spread_method,
        "best_runtime_method": best_runtime_method,
        "final_professor_priority_recommendation": final_professor,
    }


def format_fim_permutation_report(frame: pd.DataFrame, config: FIMPermutationRunConfig) -> str:
    """Format a compact terminal/report comparison table."""

    lines = [
        "FIM Algorithm-Stack Permutation Comparison",
        "-" * 72,
        f"Protected attribute: {config.protected_attribute}",
        f"Budget: {config.budget}",
        f"Ranking policy: {config.ranking_policy}",
        f"Final estimator: monte_carlo | mc_runs_eval={config.mc_runs_eval}",
        "",
    ]
    if frame.empty:
        lines.append("No permutation rows were produced.")
        return "\n".join(lines)

    sortable = frame.copy()
    for column in ["f_score", "mf", "total_spread", "runtime_seconds"]:
        sortable[column] = pd.to_numeric(sortable[column], errors="coerce")
    ok_slice = sortable[sortable["status"] == "ok"]
    if str(config.ranking_policy) == FAIRNESS_FIRST_PRIORITY:
        ok_rows = _fairness_priority_order(ok_slice, config=config)
    else:
        ok_rows = ok_slice.sort_values(
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
            lines.append(
                "   "
                f"zero_covered_groups={row.get('zero_covered_groups_count')} | "
                f"fraction_groups_covered={row.get('fraction_groups_covered')} | "
                f"scalability_pass={row.get('scalability_pass')} | "
                f"candidate_pool={row.get('candidate_pool_size')} | "
                f"ris_rr_sets={row.get('effective_ris_num_rr_sets')}"
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
                f"ranking_policy={row.get('ranking_policy')} | "
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
                f"weak_group={row.get('weak_group_bonus_weight')} | "
                f"community_diversity={row.get('community_diversity_weight')} | "
                f"protected_group_coverage={row.get('protected_group_coverage_weight')}"
            )
            lines.append(
                "   "
                f"optimizer_diagnostics: initial_seed_source={row.get('initial_seed_source')} | "
                f"repaired_seed_sets={row.get('repaired_seed_sets')} | "
                f"successful_swaps={row.get('successful_swaps')} | "
                f"community_coverage={row.get('final_community_coverage')} | "
                f"community_coverage_ratio={row.get('community_coverage_ratio')} | "
                f"protected_group_coverage={row.get('final_protected_group_coverage')}"
            )
            lines.append(
                "   "
                f"caches: embedding={row.get('embedding_cache_status')} | "
                f"community={row.get('community_cache_status')} | "
                f"ris={row.get('ris_cache_status')} | "
                f"score={row.get('score_cache_status')} | "
                f"fitness_hits={row.get('fitness_cache_hits')} | "
                f"swap_hits={row.get('swap_cache_hits')}"
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

    recommendations = recommend_fim_stacks(frame, config=config)
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
    lines.append(
        f"best_fairness_quality_method={recommendations['best_fairness_quality_method'] or 'n/a'}"
    )
    lines.append(f"best_scalable_method={recommendations['best_scalable_method'] or 'n/a'}")
    lines.append(f"best_spread_method={recommendations['best_spread_method'] or 'n/a'}")
    lines.append(f"best_runtime_method={recommendations['best_runtime_method'] or 'n/a'}")
    lines.append(
        "final_professor_priority_recommendation="
        f"{recommendations['final_professor_priority_recommendation'] or 'n/a'}"
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
            if config.debug_errors:
                _print_stack_exception(
                    exc=exc,
                    spec=spec,
                    dataset=dataset,
                    protected_group_report=protected_group_report,
                    config=config,
                )
            if config.raise_errors or not config.continue_on_error:
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
