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
from .priority_policy import (
    FAIRNESS_FIRST_PRIORITY,
    PROFESSOR_PRIORITY,
    fairness_gate_failures,
    fairness_valid_mask as _policy_fairness_valid_mask,
    normalize_ranking_policy as _policy_normalize_ranking_policy,
    professor_priority_config_from_object,
    professor_priority_warning,
    rank_frame_professor_priority,
)
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
    return _policy_normalize_ranking_policy(policy)


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
    use_ris: bool = False
    use_fair_ris: bool = False
    force_ris_for_all_stacks: bool = False
    require_ris: bool = False
    ris_mode: str = "weak_group_weighted"
    ris_reuse_rr_sets: bool = True
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
    fscore_weight: float = 4.0
    mf_weight: float = 2.0
    dcv_weight: float = 2.5
    group_coverage_weight: float = 1.0
    scalability_weight: float = 0.5
    spread_weight: float = 0.5
    runtime_penalty_weight: float = 0.05
    runtime_weight: float = 0.05
    fairness_tolerance_dcv: float = 0.005
    fairness_tolerance_fscore_drop: float = 0.001
    use_fairness_first_swap_acceptance: bool = True
    use_professor_priority_fitness: bool = False
    fair_greedy_objective: str = "default"
    dcv_penalty_weight: float = 2.5
    community_coverage_weight: float = 0.5
    use_ris_greedy_approximation: bool = False
    adaptive_fairness_weights: bool = True
    imbalance_threshold_medium: float = 5.0
    imbalance_threshold_high: float = 10.0
    adaptive_fairness_multiplier_medium: float = 1.5
    adaptive_fairness_multiplier_high: float = 2.0
    large_imbalance_fairness_mode: str = "off"
    large_imbalance_threshold: float = 5.0
    large_graph_threshold: int = 1000
    use_group_stratified_candidate_pool: bool | None = None
    min_group_candidate_floor: int = 20
    group_candidate_multiplier: float = 3.0
    use_protected_group_quota_initialization: bool | None = None
    small_group_seed_fraction: float = 0.10
    initialization_quota_mode: str = "sqrt"
    score_normalization: str = "global"
    large_imbalance_ml_score_weight: float = 0.4
    large_imbalance_ris_score_weight: float = 0.5
    large_imbalance_fair_ris_score_weight: float = 2.0
    large_imbalance_weak_group_bonus_weight: float = 2.0
    large_imbalance_protected_group_coverage_weight: float = 2.0
    large_imbalance_community_diversity_weight: float = 0.8
    large_imbalance_spread_proxy_weight: float = 0.2
    use_group_quota_repair: bool = True
    min_seeds_per_protected_group: int = 1
    quota_mode: str = "support_aware"
    quota_min_group_support: int = 5
    use_fairness_first_repair: bool = False
    weak_group_repair_rounds: int = 0
    majority_overconcentration_threshold: float = 0.60
    swap_reject_spread_gain_if_fairness_collapses: bool = False
    min_budget_node_ratio_warning: float = 0.02
    min_seeds_per_group_warning: int = 5
    imbalance_ratio_warning_threshold: float = 5.0
    warn_if_communities_exceed_budget: bool = True
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
    print_experiment_header: bool = True
    print_budget_check: bool = True
    print_stack_summary: bool = True
    print_runtime_breakdown: bool = True
    print_seed_diagnostics: bool = True
    print_group_influence: bool = True
    print_score_diagnostics: bool = True
    print_optimizer_diagnostics: bool = True
    print_delta_vs_baseline: bool = True
    print_decision_trace: bool = True
    print_collapse_explanations: bool = True
    print_raw_diagnostics: bool = False
    debug_diagnostics: bool = False


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
        "graph_embedding_algorithm",
        "ml_ranking_algorithm",
        "community_detection_algorithm",
        "clustering_algorithm",
        "diffusion_model_name",
        "search_time_spread_estimator",
        "fair_ris_mode",
        "fairness_influence_objective",
        "fair_influence_optimizer",
        "repair_strategy",
        "local_refinement",
        "final_evaluator",
        "ranking_policy_name",
        "repair_enabled",
        "swap_local_search_enabled",
        "debias_mode",
        "fairness_objective",
        "search_estimator",
        "final_estimator",
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
        "time_dataset_loading",
        "time_preprocessing",
        "time_community_detection",
        "time_embedding",
        "time_ris",
        "time_candidate_scoring",
        "time_optimizer",
        "time_repair",
        "time_local_search",
        "time_final_mc",
        "time_reporting",
        "runtime_breakdown_json",
        "scalability_mode",
        "scalability_pass",
        "candidate_pool_size",
        "use_ris",
        "use_fair_ris",
        "fair_ris_enabled",
        "ris_mode",
        "ris_num_rr_sets",
        "effective_ris_num_rr_sets",
        "ris_reuse_rr_sets",
        "ris_score_nonzero_count",
        "ris_score_std",
        "fair_ris_score_nonzero_count",
        "fair_ris_score_std",
        "ris_verified",
        "fair_ris_verified",
        "ris_active_verified",
        "fair_ris_active_verified",
        "ris_verification_warnings",
        "force_ris_for_all_stacks",
        "require_ris",
        "zero_covered_groups_count",
        "fraction_groups_covered",
        "zero_covered_groups",
        "fairness_gate_status",
        "fairness_gate_failures",
        "protected_group_coverage_summary",
        "protected_group_counts",
        "seed_count",
        "expected_seed_count",
        "duplicate_seed_count",
        "seed_group_counts",
        "seed_community_counts",
        "seed_communities_covered",
        "largest_community_seed_fraction",
        "smallest_group_seed_count",
        "largest_group_seed_count",
        "group_imbalance_ratio",
        "budget_per_group_estimate",
        "group_influence_distribution",
        "normalized_group_influence_distribution",
        "protected_group_influence_json",
        "protected_group_normalized_influence_json",
        "weakest_protected_group",
        "strongest_protected_group",
        "weakest_group",
        "strongest_group",
        "negative_f_score_reason",
        "adaptive_fairness_multiplier",
        "large_imbalance_fairness_active",
        "large_imbalance_fairness_mode",
        "score_normalization",
        "candidate_pool_group_counts",
        "candidate_pool_group_quota",
        "candidate_pool_group_quota_shortfall",
        "group_stratified_candidate_pool",
        "budget_adequacy_warning",
        "seed_quota_per_protected_group",
        "initial_population_group_coverage_summary",
        "repair_attempts",
        "weak_group_repairs",
        "duplicate_repairs",
        "protected_group_seed_counts_before_repair",
        "protected_group_seed_counts_after_repair",
        "swap_attempts",
        "swap_accepted",
        "swap_rejected_fairness_degradation",
        "rejected_fairness_drops",
        "swap_accepted_fscore_improvement",
        "swap_accepted_mf_improvement",
        "swap_accepted_spread_fairness_preserved",
        "best_generation",
        "final_fitness",
        "optimizer_diagnostics_json",
        "mc_runs_search",
        "mc_runs_eval",
        "population_size",
        "generations",
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
        "diagnostics_json",
        "candidate_score_components_constant",
        "score_component_stats_json",
        "constant_score_components",
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


def _protected_group_imbalance(protected_group_report: ProtectedGroupReport | None) -> dict[str, float]:
    if protected_group_report is None or not protected_group_report.group_sizes:
        return {
            "largest_group_size": 0.0,
            "smallest_group_size": 0.0,
            "imbalance_ratio": 1.0,
        }
    positive_sizes = [int(size) for size in protected_group_report.group_sizes.values() if int(size) > 0]
    if not positive_sizes:
        return {
            "largest_group_size": 0.0,
            "smallest_group_size": 0.0,
            "imbalance_ratio": 1.0,
        }
    largest = float(max(positive_sizes))
    smallest = float(min(positive_sizes))
    return {
        "largest_group_size": largest,
        "smallest_group_size": smallest,
        "imbalance_ratio": largest / smallest if smallest > 0.0 else float("inf"),
    }


def _large_imbalance_fairness_active(
    dataset: LoadedDataset | None,
    protected_group_report: ProtectedGroupReport | None,
    config: FIMPermutationRunConfig,
) -> bool:
    mode = str(config.large_imbalance_fairness_mode).strip().lower()
    if mode == "force":
        return True
    if mode == "off":
        return False
    imbalance_ratio = float(_protected_group_imbalance(protected_group_report)["imbalance_ratio"])
    node_count = int(dataset.graph.number_of_nodes()) if dataset is not None else 0
    return (
        imbalance_ratio >= float(config.large_imbalance_threshold)
        or node_count >= int(config.large_graph_threshold)
    )


def _effective_score_normalization(
    dataset: LoadedDataset | None,
    protected_group_report: ProtectedGroupReport | None,
    config: FIMPermutationRunConfig,
) -> str:
    configured = str(config.score_normalization).strip().lower()
    if configured not in {"global", "per_group", "hybrid"}:
        configured = "global"
    if configured == "global" and _large_imbalance_fairness_active(dataset, protected_group_report, config):
        return "per_group"
    return configured


def _effective_group_stratified_candidate_pool(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    config: FIMPermutationRunConfig,
) -> bool:
    if config.use_group_stratified_candidate_pool is not None:
        return bool(config.use_group_stratified_candidate_pool)
    return _large_imbalance_fairness_active(dataset, protected_group_report, config)


def _effective_quota_initialization(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    config: FIMPermutationRunConfig,
) -> bool:
    if config.use_protected_group_quota_initialization is not None:
        return bool(config.use_protected_group_quota_initialization)
    return _large_imbalance_fairness_active(dataset, protected_group_report, config)


def _adaptive_fairness_multiplier(
    config: FIMPermutationRunConfig,
    protected_group_report: ProtectedGroupReport | None,
) -> float:
    if not bool(config.adaptive_fairness_weights):
        return 1.0
    imbalance_ratio = float(_protected_group_imbalance(protected_group_report)["imbalance_ratio"])
    if imbalance_ratio > float(config.imbalance_threshold_high):
        return float(config.adaptive_fairness_multiplier_high)
    if imbalance_ratio > float(config.imbalance_threshold_medium):
        return float(config.adaptive_fairness_multiplier_medium)
    return 1.0


_METHOD_RECOMMENDED_WEIGHT_DEFAULTS: dict[str, dict[str, float]] = {
    "node2vec_xgboost_community_siea": {
        "ml_score_weight": 0.8,
        "fair_ris_score_weight": 1.2,
        "weak_group_bonus_weight": 1.2,
        "protected_group_coverage_weight": 1.2,
    },
    "graphsage_community_siea": {
        "ml_score_weight": 0.7,
        "fair_ris_score_weight": 1.3,
        "weak_group_bonus_weight": 1.2,
        "dcv_penalty_weight": 2.5,
    },
    "gcn_community_siea": {
        "ml_score_weight": 0.6,
        "fair_ris_score_weight": 1.4,
        "weak_group_bonus_weight": 1.3,
        "dcv_penalty_weight": 2.5,
    },
}


def _professor_priority_weight_defaults(config: FIMPermutationRunConfig, spec: FIMPermutationSpec) -> dict[str, float]:
    if _fairness_first_enabled(spec, config):
        defaults = {
            "ml_score_weight": 0.7,
            "ris_score_weight": 0.7,
            "fair_ris_score_weight": 1.2,
            "weak_group_bonus_weight": 1.0,
            "protected_group_coverage_weight": 1.0,
            "community_diversity_weight": 0.5,
            "spread_proxy_weight": 0.4,
            "dcv_penalty_weight": float(config.dcv_penalty_weight),
        }
    else:
        defaults = {
            "ml_score_weight": spec.ranking_weight,
            "ris_score_weight": spec.ris_weight,
            "fair_ris_score_weight": spec.fair_ris_weight,
            "weak_group_bonus_weight": _weak_group_bonus_weight(config),
            "protected_group_coverage_weight": float(config.protected_group_coverage_weight),
            "community_diversity_weight": float(config.diversity_bonus_weight),
            "spread_proxy_weight": float(config.spread_proxy_weight),
            "dcv_penalty_weight": float(config.dcv_penalty_weight),
        }
    defaults.update(_METHOD_RECOMMENDED_WEIGHT_DEFAULTS.get(spec.name, {}))
    return defaults


def _resolved_candidate_score_weights(
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
    protected_group_report: ProtectedGroupReport | None = None,
    dataset: LoadedDataset | None = None,
) -> dict[str, float]:
    defaults = _professor_priority_weight_defaults(config, spec)
    large_imbalance_active = _large_imbalance_fairness_active(dataset, protected_group_report, config)
    if large_imbalance_active:
        defaults.update(
            {
                "ml_score_weight": float(config.large_imbalance_ml_score_weight),
                "ris_score_weight": float(config.large_imbalance_ris_score_weight),
                "fair_ris_score_weight": float(config.large_imbalance_fair_ris_score_weight),
                "weak_group_bonus_weight": float(config.large_imbalance_weak_group_bonus_weight),
                "protected_group_coverage_weight": float(config.large_imbalance_protected_group_coverage_weight),
                "community_diversity_weight": float(config.large_imbalance_community_diversity_weight),
                "spread_proxy_weight": float(config.large_imbalance_spread_proxy_weight),
            }
        )
    weights = {
        "ml_score_weight": defaults["ml_score_weight"] if config.ml_score_weight is None else float(config.ml_score_weight),
        "ris_score_weight": defaults["ris_score_weight"] if config.ris_score_weight is None else float(config.ris_score_weight),
        "fair_ris_score_weight": defaults["fair_ris_score_weight"] if config.fair_ris_score_weight is None else float(config.fair_ris_score_weight),
        "weak_group_bonus_weight": defaults["weak_group_bonus_weight"] if config.weak_group_bonus_weight is None else float(config.weak_group_bonus_weight),
        "protected_group_coverage_weight": defaults["protected_group_coverage_weight"] if config.protected_group_coverage_weight == 0.0 and (_fairness_first_enabled(spec, config) or large_imbalance_active) else float(config.protected_group_coverage_weight),
        "community_diversity_weight": defaults["community_diversity_weight"] if config.diversity_bonus_weight == 0.2 and (_fairness_first_enabled(spec, config) or large_imbalance_active) else float(config.diversity_bonus_weight),
        "spread_proxy_weight": defaults["spread_proxy_weight"] if config.spread_proxy_weight == 0.0 and (_fairness_first_enabled(spec, config) or large_imbalance_active) else float(config.spread_proxy_weight),
        "dcv_penalty_weight": defaults["dcv_penalty_weight"],
    }
    multiplier = _adaptive_fairness_multiplier(config, protected_group_report)
    if multiplier > 1.0:
        adaptive_weight_keys = []
        if config.weak_group_bonus_weight is None:
            adaptive_weight_keys.append("weak_group_bonus_weight")
        if config.fair_ris_score_weight is None:
            adaptive_weight_keys.append("fair_ris_score_weight")
        if float(config.protected_group_coverage_weight) == 0.0:
            adaptive_weight_keys.append("protected_group_coverage_weight")
        adaptive_weight_keys.append("dcv_penalty_weight")
        for key in adaptive_weight_keys:
            weights[key] = float(weights[key]) * multiplier
    if not _spec_uses_ris(spec):
        weights["ris_score_weight"] = 0.0
        weights["fair_ris_score_weight"] = 0.0
    elif not _fair_ris_enabled(spec):
        weights["fair_ris_score_weight"] = 0.0
    return weights


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


def _candidate_pool_group_counts(
    candidate_nodes: Sequence[Any],
    protected_group_report: ProtectedGroupReport,
) -> dict[str, int]:
    group_by_node = {
        node_id: group_name
        for group_name, node_ids in protected_group_report.protected_groups.items()
        for node_id in node_ids
    }
    counts = {str(group_name): 0 for group_name in protected_group_report.group_sizes}
    for node_id in candidate_nodes:
        group_name = group_by_node.get(node_id)
        if group_name is not None:
            counts[str(group_name)] = int(counts.get(str(group_name), 0)) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _group_stratified_candidate_pool(
    *,
    score_frame: pd.DataFrame,
    protected_group_report: ProtectedGroupReport,
    config: FIMPermutationRunConfig,
    target_size: int,
) -> tuple[list[Any], dict[str, object]]:
    if score_frame.empty or "node_id" not in score_frame.columns:
        return [], {
            "group_stratified_candidate_pool": False,
            "candidate_pool_group_counts": {},
            "candidate_pool_group_quota": 0,
            "candidate_pool_group_quota_shortfall": {},
        }

    if "combined_score" not in score_frame.columns:
        working = score_frame.assign(combined_score=0.0)
    else:
        working = score_frame.copy()
    score_by_node = {
        row.node_id: float(row.combined_score)
        for row in working[["node_id", "combined_score"]].itertuples(index=False)
    }
    group_by_node = {
        node_id: group_name
        for group_name, node_ids in protected_group_report.protected_groups.items()
        for node_id in node_ids
    }
    group_names = [
        group_name
        for group_name in sorted(protected_group_report.group_sizes, key=_sort_key)
        if int(protected_group_report.group_sizes[group_name]) > 0
    ]
    num_groups = max(1, len(group_names))
    group_quota = max(
        int(config.min_group_candidate_floor),
        int(math.ceil(float(config.group_candidate_multiplier) * float(config.budget) / float(num_groups))),
    )
    selected: dict[Any, None] = {}
    shortfalls: dict[str, int] = {}
    for group_name in group_names:
        group_nodes = [
            node_id
            for node_id, mapped_group in group_by_node.items()
            if mapped_group == group_name and node_id in score_by_node
        ]
        ranked_group_nodes = sorted(
            group_nodes,
            key=lambda node_id: (-float(score_by_node.get(node_id, 0.0)), _sort_key(node_id)),
        )
        chosen = ranked_group_nodes[:group_quota]
        shortfalls[str(group_name)] = max(0, int(group_quota) - len(chosen))
        for node_id in chosen:
            selected.setdefault(node_id, None)

    ranked_global = sorted(
        score_by_node,
        key=lambda node_id: (-float(score_by_node.get(node_id, 0.0)), _sort_key(node_id)),
    )
    fill_target = min(
        len(ranked_global),
        max(int(target_size), int(config.budget), len(selected)),
    )
    for node_id in ranked_global:
        if len(selected) >= fill_target:
            break
        selected.setdefault(node_id, None)

    candidate_nodes = list(selected)
    return candidate_nodes, {
        "group_stratified_candidate_pool": True,
        "candidate_pool_group_counts": _candidate_pool_group_counts(candidate_nodes, protected_group_report),
        "candidate_pool_group_quota": int(group_quota),
        "candidate_pool_group_quota_shortfall": dict(sorted(shortfalls.items(), key=lambda item: item[0])),
    }


def _budget_adequacy_warnings(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    config: FIMPermutationRunConfig,
    community_result: CommunityDetectionResult | None = None,
) -> list[str]:
    warnings: list[str] = []
    num_groups = max(1, len([size for size in protected_group_report.group_sizes.values() if int(size) > 0]))
    node_count = max(1, int(dataset.graph.number_of_nodes()))
    imbalance = _protected_group_imbalance(protected_group_report)
    positive_sizes = [int(size) for size in protected_group_report.group_sizes.values() if int(size) > 0]
    if int(config.budget) < int(num_groups) * int(config.min_seeds_per_group_warning):
        warnings.append("budget below protected-group seed warning threshold")
    if float(config.budget) / float(node_count) < float(config.min_budget_node_ratio_warning):
        warnings.append("budget/node ratio below warning threshold")
    if int(config.budget) < int(num_groups):
        warnings.append("protected groups exceed budget")
    if positive_sizes and min(positive_sizes) < int(config.min_seeds_per_group_warning):
        warnings.append("smallest protected group is very small")
    if float(imbalance["imbalance_ratio"]) >= float(config.imbalance_ratio_warning_threshold):
        warnings.append("protected-group imbalance ratio above warning threshold")
    if (
        bool(config.warn_if_communities_exceed_budget)
        and community_result is not None
        and int(community_result.stats.num_communities) > int(config.budget)
    ):
        warnings.append("number of communities exceeds budget")
    return warnings


def _budget_adequacy_warning_text(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    config: FIMPermutationRunConfig,
    community_result: CommunityDetectionResult | None = None,
) -> str:
    warnings = _budget_adequacy_warnings(dataset, protected_group_report, config, community_result)
    if not warnings:
        return ""
    return "Budget may be too small for stable fairness on this graph. Reasons: " + "; ".join(warnings)


def _effective_ris_rr_sets(
    dataset: LoadedDataset,
    config: FIMPermutationRunConfig,
    protected_group_report: ProtectedGroupReport | None = None,
) -> int:
    base_rr_sets = max(1, int(config.ris_num_rr_sets))
    node_count = int(dataset.graph.number_of_nodes())
    imbalance_ratio = (
        float(_protected_group_imbalance(protected_group_report)["imbalance_ratio"])
        if protected_group_report is not None
        else 1.0
    )
    large_imbalance_or_graph = (
        str(config.scalability_mode).strip().lower() in {"auto", "large_graph"}
        and (
            node_count >= int(config.large_graph_threshold)
            or imbalance_ratio >= float(config.large_imbalance_threshold)
        )
    )
    if not bool(config.adaptive_ris_rr_sets):
        return base_rr_sets
    if large_imbalance_or_graph:
        return max(base_rr_sets, 256)
    if node_count < 1000:
        return base_rr_sets
    moderate_multiplier = 1.5 if str(config.scalability_mode) == "large_graph" else 1.25
    return int(min(max(base_rr_sets, math.ceil(base_rr_sets * moderate_multiplier)), max(base_rr_sets, 512)))


def _emit_ris_rr_set_adjustment_warning(
    *,
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
    effective_rr_sets: int,
) -> None:
    base_rr_sets = max(1, int(config.ris_num_rr_sets))
    if int(effective_rr_sets) <= base_rr_sets:
        return
    print(
        "WARNING: "
        f"RIS RR-set count raised for stack={spec.name} from {base_rr_sets} to {int(effective_rr_sets)} "
        "because adaptive_ris_rr_sets is enabled for a large or imbalanced graph.",
        file=sys.stderr,
    )


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


def _is_missing_value(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _json_ready_mapping(mapping: Mapping[object, object] | None) -> dict[str, object]:
    if not mapping:
        return {}
    ready: dict[str, object] = {}
    for key, value in mapping.items():
        if _is_missing_value(value):
            ready[str(key)] = None
        elif isinstance(value, (int, float, str, bool)):
            ready[str(key)] = value
        else:
            ready[str(key)] = str(value)
    return ready


def _compact_json(value: object) -> str:
    if _is_missing_value(value):
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped == "<NA>":
            return ""
        return stripped
    if isinstance(value, Mapping):
        return json.dumps(_json_ready_mapping(value), sort_keys=True)
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), sort_keys=True, default=str)
    return json.dumps(value, sort_keys=True, default=str)


def _score_component_diagnostics(score_frame: pd.DataFrame | None) -> dict[str, object]:
    component_flags = {
        column_name: constant
        for column_name in _SCORE_COMPONENT_COLUMNS
        if (constant := _is_constant_score_component(score_frame, column_name)) is not None
    }
    all_constant = None if not component_flags else bool(all(component_flags.values()))
    component_stats: dict[str, dict[str, float | bool | int]] = {}
    if score_frame is not None and not score_frame.empty:
        for column_name in _SCORE_COMPONENT_COLUMNS:
            if column_name not in score_frame.columns:
                continue
            values = pd.to_numeric(score_frame[column_name], errors="coerce").dropna()
            if values.empty:
                component_stats[column_name] = {
                    "count": 0,
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "constant": True,
                }
                continue
            component_stats[column_name] = {
                "count": int(values.count()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "constant": bool(float(values.max()) <= float(values.min())),
            }
    constant_components = [
        component_name
        for component_name, is_constant in component_flags.items()
        if bool(is_constant)
    ]
    return {
        "all_candidate_score_components_constant": all_constant,
        "constant_components": component_flags,
        "score_component_stats_json": json.dumps(component_stats, sort_keys=True),
        "constant_score_components": json.dumps(constant_components, sort_keys=True),
    }


def _score_column_stats(score_frame: pd.DataFrame | None, column_name: str) -> dict[str, object]:
    if score_frame is None or score_frame.empty or column_name not in score_frame.columns:
        return {
            "present": False,
            "nonzero_count": 0,
            "std": 0.0,
            "constant_zero": True,
            "constant": True,
        }
    values = pd.to_numeric(score_frame[column_name], errors="coerce").fillna(0.0)
    if values.empty:
        return {
            "present": True,
            "nonzero_count": 0,
            "std": 0.0,
            "constant_zero": True,
            "constant": True,
        }
    std = float(values.std(ddof=0)) if len(values) > 1 else 0.0
    nonzero_count = int((values.abs() > 1e-12).sum())
    return {
        "present": True,
        "nonzero_count": nonzero_count,
        "std": std,
        "constant_zero": nonzero_count == 0,
        "constant": std <= 1e-12,
    }


def _verify_ris_score_table(
    *,
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
    score_frame: pd.DataFrame | None,
) -> dict[str, object]:
    ris_stats = _score_column_stats(score_frame, "ris_score")
    fair_stats = _score_column_stats(score_frame, "fair_ris_score")
    search_estimator = _canonical_search_estimator(spec.spread_estimator_search)
    final_estimator = str(spec.spread_estimator_final or "").strip().lower()
    ris_requested = _spec_uses_ris(spec)
    ris_required = bool(config.require_ris) or ris_requested
    fair_requested = _fair_ris_enabled(spec)
    warnings: list[str] = []
    hard_failures: list[str] = []

    if final_estimator != "monte_carlo":
        message = f"spread_estimator_final is {final_estimator or 'unset'}; final evaluation must remain monte_carlo."
        warnings.append(message)
        hard_failures.append(message)
    if ris_required and search_estimator == "monte_carlo":
        message = "spread_estimator_search is monte_carlo; RIS is not active."
        warnings.append(message)
        hard_failures.append(message)
    if ris_required and search_estimator not in {"ris", "fairness_aware_ris"}:
        message = f"spread_estimator_search is {search_estimator}; expected ris or fairness_aware_ris."
        warnings.append(message)
        hard_failures.append(message)
    if fair_requested and search_estimator != "fairness_aware_ris":
        message = f"Fair RIS requested but spread_estimator_search is {search_estimator}."
        warnings.append(message)
        hard_failures.append(message)
    if ris_required and not bool(ris_stats["present"]):
        message = "RIS requested but ris_score column is missing."
        warnings.append(message)
        hard_failures.append(message)
    if ris_required and bool(ris_stats["constant_zero"]):
        message = "RIS requested but ris_score is constant zero."
        warnings.append(message)
        if bool(config.require_ris):
            hard_failures.append(message)
    elif ris_required and bool(ris_stats["constant"]):
        warnings.append("RIS requested but ris_score is constant.")

    if fair_requested and not bool(fair_stats["present"]):
        message = "Fair RIS requested but fair_ris_score is missing."
        warnings.append(message)
        hard_failures.append(message)
    if fair_requested and bool(fair_stats["constant_zero"]):
        message = "Fair RIS requested but fair_ris_score is constant zero."
        warnings.append(message)
        if bool(config.require_ris):
            hard_failures.append(message)
    elif fair_requested and bool(fair_stats["constant"]):
        warnings.append("Fair RIS requested but fair_ris_score is constant.")

    for warning in warnings:
        print(f"WARNING: stack={spec.name} | {warning}", file=sys.stderr)

    if bool(config.require_ris) and hard_failures:
        raise RuntimeError(
            f"RIS verification failed for stack {spec.name}: " + " ".join(hard_failures)
        )

    ris_active_verified = bool(
        ris_required
        and search_estimator != "monte_carlo"
        and bool(ris_stats["present"])
        and int(ris_stats["nonzero_count"]) > 0
        and float(ris_stats["std"]) > 1e-12
    )
    fair_ris_active_verified = bool(
        fair_requested
        and bool(fair_stats["present"])
        and int(fair_stats["nonzero_count"]) > 0
        and float(fair_stats["std"]) > 1e-12
    )
    return {
        "ris_score_nonzero_count": int(ris_stats["nonzero_count"]),
        "ris_score_std": float(ris_stats["std"]),
        "fair_ris_score_nonzero_count": int(fair_stats["nonzero_count"]),
        "fair_ris_score_std": float(fair_stats["std"]),
        "ris_verified": ris_active_verified,
        "fair_ris_verified": fair_ris_active_verified,
        "ris_active_verified": ris_active_verified,
        "fair_ris_active_verified": fair_ris_active_verified,
        "ris_verification_warnings": " ".join(dict.fromkeys(warnings)),
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
    imbalance = _protected_group_imbalance(protected_group_report)
    total_group_support = max(1, int(sum(protected_group_report.group_sizes.values())))
    budget_per_group_estimate = {
        group_name: float(config.budget) * float(group_size) / float(total_group_support)
        for group_name, group_size in protected_group_report.group_sizes.items()
    }
    payload: dict[str, object] = {
        "stack_name": spec.name,
        "dataset": dataset.name,
        "protected_attribute": protected_group_report.protected_attribute,
        "budget": int(config.budget),
        "num_nodes": int(dataset.graph.number_of_nodes()),
        "num_edges": int(dataset.graph.number_of_edges()),
        "protected_group_counts": dict(protected_group_report.group_sizes),
        "largest_group_size": int(imbalance["largest_group_size"]),
        "smallest_group_size": int(imbalance["smallest_group_size"]),
        "imbalance_ratio": float(imbalance["imbalance_ratio"]),
        "budget_per_group_estimate": budget_per_group_estimate,
        "adaptive_fairness_multiplier": _adaptive_fairness_multiplier(config, protected_group_report),
        "large_imbalance_fairness_active": _large_imbalance_fairness_active(dataset, protected_group_report, config),
        "large_imbalance_fairness_mode": str(config.large_imbalance_fairness_mode),
        "score_normalization": _effective_score_normalization(dataset, protected_group_report, config),
        "spread_estimator_search": _canonical_search_estimator(spec.spread_estimator_search),
        "spread_estimator_final": spec.spread_estimator_final,
        "use_ris": _spec_uses_ris(spec),
        "use_fair_ris": bool(spec.use_fair_ris),
        "fair_ris_enabled": _fair_ris_enabled(spec),
        "ris_mode": _reported_ris_mode(config),
        "ris_num_rr_sets": int(config.ris_num_rr_sets),
        "effective_ris_num_rr_sets": _effective_ris_rr_sets(dataset, config, protected_group_report),
        "ris_reuse_rr_sets": bool(config.ris_reuse_rr_sets),
        "force_ris_for_all_stacks": bool(config.force_ris_for_all_stacks),
        "require_ris": bool(config.require_ris),
        "budget_adequacy_warning": _budget_adequacy_warning_text(dataset, protected_group_report, config, community_result),
        "num_communities": int(community_result.stats.num_communities) if community_result is not None else None,
        "smallest_community_size": int(min(community_sizes)) if community_sizes else None,
        "largest_community_size": int(max(community_sizes)) if community_sizes else None,
        "time_community_detection": (
            float(community_result.runtime_seconds)
            if community_result is not None and hasattr(community_result, "runtime_seconds")
            else None
        ),
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


def _format_raw_diagnostics_line(payload: Mapping[str, object]) -> str:
    constant_components = payload.get("constant_components", {})
    if isinstance(constant_components, Mapping) and constant_components:
        score_constant_text = json.dumps(dict(constant_components), sort_keys=True)
    else:
        score_constant_text = "n/a"
    return (
        "Benchmark diagnostics | "
        f"stack={payload.get('stack_name')} | "
        f"dataset={payload.get('dataset')} | "
        f"protected_attribute={payload.get('protected_attribute')} | "
        f"budget={payload.get('budget')} | "
        f"nodes={payload.get('num_nodes')} | "
        f"edges={payload.get('num_edges')} | "
        f"protected_group_counts={json.dumps(payload.get('protected_group_counts', {}), sort_keys=True)} | "
        f"imbalance_ratio={payload.get('imbalance_ratio')} | "
        f"large_imbalance_fairness_active={payload.get('large_imbalance_fairness_active')} | "
        f"search_estimator={payload.get('spread_estimator_search')} | "
        f"final_estimator={payload.get('spread_estimator_final')} | "
        f"fair_ris={payload.get('fair_ris_enabled')} | "
        f"ris_mode={payload.get('ris_mode')} | "
        f"ris_rr_sets={payload.get('effective_ris_num_rr_sets')} | "
        f"force_ris={payload.get('force_ris_for_all_stacks')} | "
        f"require_ris={payload.get('require_ris')} | "
        f"score_normalization={payload.get('score_normalization')} | "
        f"adaptive_fairness_multiplier={payload.get('adaptive_fairness_multiplier')} | "
        f"communities={payload.get('num_communities')} | "
        f"smallest_community_size={payload.get('smallest_community_size')} | "
        f"largest_community_size={payload.get('largest_community_size')} | "
        f"all_candidate_score_components_constant={payload.get('all_candidate_score_components_constant')} | "
        f"constant_components={score_constant_text}"
    )


def _print_diagnostics_warning(payload: Mapping[str, object]) -> None:
    warning = str(payload.get("budget_adequacy_warning", "") or "").strip()
    if warning:
        print(warning)


def _print_diagnostics(payload: Mapping[str, object]) -> None:
    print(_format_raw_diagnostics_line(payload))
    _print_diagnostics_warning(payload)


def _readable_diagnostics_covered_by_sections(config: FIMPermutationRunConfig) -> bool:
    return any(
        bool(value)
        for value in (
            config.print_experiment_header,
            config.print_budget_check,
            config.print_stack_summary,
            config.print_score_diagnostics,
            config.print_group_influence,
            config.print_optimizer_diagnostics,
        )
    )


def _print_diagnostics_summary(payload: Mapping[str, object]) -> None:
    group_counts = _parse_json_mapping(payload.get("protected_group_counts", {}))
    constant_components = _parse_json_sequence(payload.get("constant_score_components", []))
    print("Benchmark Diagnostics Summary")
    print("-" * 60)
    print(f"{'Stack':<30}: {_format_diagnostic_value(payload.get('stack_name'))}")
    print(f"{'Dataset':<30}: {_format_diagnostic_value(payload.get('dataset'))}")
    print(f"{'Protected Attribute':<30}: {_format_diagnostic_value(payload.get('protected_attribute'))}")
    print(f"{'Budget':<30}: {_format_diagnostic_value(payload.get('budget'))}")
    print(f"{'Nodes / Edges':<30}: {_format_diagnostic_value(payload.get('num_nodes'))} / {_format_diagnostic_value(payload.get('num_edges'))}")
    if group_counts and len(group_counts) <= 8:
        print("Protected Groups:")
        for group_name in sorted(group_counts):
            print(f"  - {group_name}: {group_counts[group_name]}")
    elif group_counts:
        print(f"{'Protected Groups':<30}: {len(group_counts)} groups; full diagnostics saved to JSON report")
    else:
        print(f"{'Protected Groups':<30}: n/a")
    print(f"{'Imbalance Ratio':<30}: {_format_diagnostic_value(payload.get('imbalance_ratio'))}")
    print(f"{'Communities':<30}: {_format_diagnostic_value(payload.get('num_communities'))}")
    print(f"{'Smallest / Largest Comm.':<30}: {_format_diagnostic_value(payload.get('smallest_community_size'))} / {_format_diagnostic_value(payload.get('largest_community_size'))}")
    print(f"{'Search Estimator':<30}: {_format_diagnostic_value(payload.get('spread_estimator_search'))}")
    print(f"{'Final Estimator':<30}: {_format_diagnostic_value(payload.get('spread_estimator_final'))}")
    print(f"{'Fair RIS':<30}: {_format_diagnostic_value(payload.get('fair_ris_enabled'))}")
    print(f"{'RIS RR Sets':<30}: {_format_diagnostic_value(payload.get('effective_ris_num_rr_sets'))}")
    print(f"{'Score Normalization':<30}: {_format_diagnostic_value(payload.get('score_normalization'))}")
    print(f"{'Constant Score Components':<30}: {constant_components if constant_components else 'None'}")
    if payload.get("diagnostics_path"):
        print(f"{'Full Diagnostics':<30}: saved to JSON report")
    print("")
    _print_diagnostics_warning(payload)


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
        payload["diagnostics_path"] = str(path)
    diagnostics_json = json.dumps(payload, sort_keys=True, default=str)
    if bool(config.print_raw_diagnostics) or bool(config.debug_diagnostics):
        _print_diagnostics(payload)
    elif _readable_diagnostics_covered_by_sections(config):
        _print_diagnostics_warning(payload)
    else:
        _print_diagnostics_summary(payload)
    return {
        "diagnostics_path": "" if path is None else str(path),
        "diagnostics_json": diagnostics_json,
        "candidate_score_components_constant": payload["all_candidate_score_components_constant"],
        "large_imbalance_fairness_active": payload["large_imbalance_fairness_active"],
        "large_imbalance_fairness_mode": payload["large_imbalance_fairness_mode"],
        "score_normalization": payload["score_normalization"],
        "budget_adequacy_warning": payload["budget_adequacy_warning"],
        "time_community_detection": payload.get("time_community_detection", pd.NA),
        "score_component_stats_json": payload.get("score_component_stats_json", ""),
        "constant_score_components": payload.get("constant_score_components", "[]"),
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
    ris_mode: str = "global",
) -> Path | None:
    attribute_dir = _permutation_output_dir(output_dir, dataset_name, protected_attribute)
    if attribute_dir is None:
        return None
    safe_mode = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in str(ris_mode).strip().lower()
    ).strip("._-") or "global"
    return attribute_dir / f"{dataset_name}_{stack_name}_ris_{safe_mode}_rr{int(rr_sets)}_cache.pkl"


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
    protected_group_report: ProtectedGroupReport | None = None,
    config: FIMPermutationRunConfig | None = None,
) -> list[Any]:
    excluded = set(seed_set)
    degree_view = dataset.graph.out_degree() if dataset.graph.is_directed() else dataset.graph.degree()
    degree_scores = {node_id: float(score) for node_id, score in degree_view if node_id not in excluded}
    candidates = sorted(degree_scores, key=lambda node_id: (-degree_scores[node_id], _sort_key(node_id)))
    if (
        protected_group_report is not None
        and config is not None
        and _effective_group_stratified_candidate_pool(dataset, protected_group_report, config)
    ):
        score_frame = pd.DataFrame(
            [{"node_id": node_id, "combined_score": score} for node_id, score in degree_scores.items()]
        )
        stratified_candidates, _ = _group_stratified_candidate_pool(
            score_frame=score_frame,
            protected_group_report=protected_group_report,
            config=config,
            target_size=max_size if max_size > 0 else len(candidates),
        )
        candidates = stratified_candidates
    if max_size > 0:
        candidates = candidates[: int(max_size)]
    return candidates


def _rank_search_evaluation(mode: str, evaluation: SeedSetEvaluation) -> tuple[float, float, float, float]:
    normalized_mode = _normalize_ranking_policy(mode)
    if normalized_mode == "maximin":
        return (
            float(evaluation.fairness.mf),
            float(evaluation.f_score),
            float(evaluation.total_spread_mean),
            -float(evaluation.fairness.dcv),
        )
    if normalized_mode == PROFESSOR_PRIORITY:
        return (
            float(evaluation.f_score),
            float(evaluation.fairness.mf),
            -float(evaluation.fairness.dcv),
            float(evaluation.total_spread_mean),
        )
    return (
        float(evaluation.f_score),
        float(evaluation.fairness.mf),
        float(evaluation.total_spread_mean),
        -float(evaluation.fairness.dcv),
    )


def _professor_priority_search_accepts(
    incumbent: SeedSetEvaluation,
    candidate: SeedSetEvaluation,
    config: FIMPermutationRunConfig,
) -> bool:
    """Fairness-first swap acceptance used by community greedy/local-search paths."""

    epsilon = 1e-12
    if float(candidate.f_score) > float(incumbent.f_score) + epsilon:
        return True
    if (
        float(candidate.fairness.mf) > float(incumbent.fairness.mf) + epsilon
        and float(candidate.fairness.dcv) <= float(incumbent.fairness.dcv) + float(config.fairness_tolerance_dcv)
    ):
        return True
    if (
        float(candidate.total_spread_mean) > float(incumbent.total_spread_mean) + epsilon
        and float(candidate.f_score) >= float(incumbent.f_score) - float(config.fairness_tolerance_fscore_drop)
        and float(candidate.fairness.dcv) <= float(incumbent.fairness.dcv) + float(config.fairness_tolerance_dcv)
    ):
        if (
            bool(config.swap_reject_spread_gain_if_fairness_collapses)
            and float(candidate.f_score) < 0.0
            and float(candidate.f_score) < float(incumbent.f_score)
        ):
            return False
        return True
    return False


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
    priority_acceptance = (
        _normalize_ranking_policy(objective_mode) == PROFESSOR_PRIORITY
        or bool(config.use_fairness_first_swap_acceptance)
    )

    for step_index in range(max(0, int(config.local_search_steps))):
        best_seed_set = current
        best_eval = current_eval
        best_key = _rank_search_evaluation(objective_mode, current_eval)
        for removed_node in current:
            retained_nodes = [node_id for node_id in current if node_id != removed_node]
            for candidate_node in _candidate_pool(
                dataset,
                current,
                int(config.swap_candidate_pool_size),
                protected_group_report=protected_group_report,
                config=config,
            ):
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
                if (
                    (priority_acceptance and _professor_priority_search_accepts(best_eval, evaluation, config))
                    or (not priority_acceptance and rank_key > best_key)
                    or (
                        rank_key == best_key
                        and _seed_tuple_key(trial_seed_set) < _seed_tuple_key(best_seed_set)
                    )
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


def _canonical_search_estimator(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "ris_guidance":
        return "ris"
    if normalized in {"ris", "fairness_aware_ris", "monte_carlo"}:
        return normalized
    return normalized or "monte_carlo"


def _spec_uses_ris(spec: FIMPermutationSpec) -> bool:
    return bool(spec.use_ris_guidance) or _canonical_search_estimator(spec.spread_estimator_search) in {
        "ris",
        "fairness_aware_ris",
    }


def _fair_ris_enabled(spec: FIMPermutationSpec) -> bool:
    return bool(spec.use_fair_ris) or _canonical_search_estimator(spec.spread_estimator_search) == "fairness_aware_ris"


def _reported_ris_mode(config: FIMPermutationRunConfig) -> str:
    mode = str(config.ris_mode).strip().lower()
    if mode == "global":
        return "standard"
    return mode or "standard"


def _ris_config_mode(config: FIMPermutationRunConfig) -> str:
    mode = _reported_ris_mode(config)
    return "global" if mode == "standard" else mode


_METHOD_TYPE_NAMES = {
    "non_ml_baseline": "Non-ML baseline",
    "weak_baseline": "Weak ML baseline",
    "ml_guided_hybrid": "ML-guided hybrid",
    "ml_guided": "ML-guided",
}
_EMBEDDING_NAMES = {
    "": "None",
    "none": "None",
    "graphsage": "GraphSAGE",
    "gcn": "Graph Convolutional Network",
    "node2vec": "Node2Vec",
    "deepwalk": "DeepWalk",
    "line": "LINE",
    "graphcl": "GraphCL",
    "dgi": "Deep Graph Infomax",
    "vgae": "Variational Graph Autoencoder",
}
_COMMUNITY_NAMES = {
    "": "None",
    "none": "None",
    "leiden": "Leiden",
    "louvain": "Louvain",
    "multilevel": "Multilevel Modularity",
    "infomap": "Infomap",
    "walktrap": "Walktrap",
}
_CLUSTERING_NAMES = {
    "": "None",
    "none": "None",
    "false": "Disabled",
    "kmeans": "K-Means",
    "spectral": "Spectral Clustering",
    "agglomerative": "Agglomerative Clustering",
    "gmm": "Gaussian Mixture Model",
}
_DIFFUSION_NAMES = {
    "ic": "Independent Cascade",
    "independent_cascade": "Independent Cascade",
    "lt": "Linear Threshold",
    "wc": "Weighted Cascade",
}
_SEARCH_ESTIMATOR_NAMES = {
    "monte_carlo": "Monte Carlo",
    "ris": "Reverse Influence Sampling",
    "fairness_aware_ris": "Fair Reverse Influence Sampling / Fair RIS",
    "fair_ris": "Fair Reverse Influence Sampling / Fair RIS",
}
_RANKING_MODEL_NAMES = {
    "": "None",
    "none": "None",
    "graphsage": "GraphSAGE node scorer",
    "graphsage_plus_fair_ris": "GraphSAGE + Fair RIS scoring",
    "gcn": "GCN node scorer",
    "gcn_plus_fair_ris": "GCN + Fair RIS scoring",
    "node2vec": "Node2Vec node scorer",
    "xgboost": "XGBoost node scorer",
    "logistic_regression": "Logistic regression node scorer",
    "random_forest": "Random forest node scorer",
    "mlp": "MLP node scorer",
    "fairness_weighted_greedy": "Fairness-weighted greedy score",
    "maximin_greedy": "Maximin greedy score",
    "structural_community_score": "Structural community score",
}
_OPTIMIZER_NAMES = {
    "": "None",
    "none": "None",
    "hybrid_si_ea": "Swarm Intelligence + Evolutionary Algorithm",
    "local_search": "Local Search",
    "greedy": "Greedy",
    "fairness_weighted_greedy": "Fairness-Weighted Greedy",
    "maximin_greedy": "Maximin Greedy",
}
_REPAIR_NAMES = {
    "": "None",
    "none": "None",
    "false": "Disabled",
    "off": "Disabled",
    "basic": "Standard repair",
    "standard": "Standard repair",
    "balanced": "Balanced repair",
    "fairness_first": "Fairness-first repair",
    "group_quota": "Group quota repair",
}
_LOCAL_REFINEMENT_NAMES = {
    "": "None",
    "none": "None",
    "false": "Disabled",
    "off": "Disabled",
    "swap_local_search": "Swap Local Search",
    "local_search": "Local Search",
}
_FINAL_EVALUATOR_NAMES = {
    "monte_carlo": "Monte Carlo simulation",
}
_RANKING_POLICY_NAMES = {
    "fim_default": "FIM default ranking",
    "fairness_first": "Fairness-first ranking",
    "fairness_first_priority": "Fairness-first ranking",
    PROFESSOR_PRIORITY: "Professor-priority ranking",
    "spread_first": "Spread-first ranking",
    "runtime_first": "Runtime-first ranking",
}
_OBJECTIVE_NAMES = {
    "f_score": "F-score fairness / influence objective",
    "default": "Default fairness / influence objective",
    "maximin": "Maximin fairness objective",
    PROFESSOR_PRIORITY: "Professor-priority objective",
    "fairness_first": "Fairness-first objective",
}


def _readable_name(value: object, mapping: Mapping[str, str]) -> str:
    text = "" if value is None else str(value).strip()
    key = text.lower()
    if key in mapping:
        return mapping[key]
    return text or "None"


def _safe_display_value(value: object) -> str:
    if value is None:
        return "None"
    try:
        if pd.isna(value):
            return "None"
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else "None"


def _ml_ranking_algorithm_name(spec: FIMPermutationSpec) -> str:
    model = str(spec.ranking_model or "none").strip().lower()
    if _method_type(spec) == "non_ml_baseline" and str(spec.embedding_method).strip().lower() in {"", "none"}:
        return "None"
    if model == "graphsage" and _fair_ris_enabled(spec):
        return "GraphSAGE + Fair RIS scoring"
    if model == "gcn" and _fair_ris_enabled(spec):
        return "GCN + Fair RIS scoring"
    return _readable_name(model, _RANKING_MODEL_NAMES)


def _search_estimator_algorithm_name(spec: FIMPermutationSpec) -> str:
    if _fair_ris_enabled(spec):
        return _SEARCH_ESTIMATOR_NAMES["fairness_aware_ris"]
    if _spec_uses_ris(spec):
        return _SEARCH_ESTIMATOR_NAMES["ris"]
    return _readable_name(_canonical_search_estimator(spec.spread_estimator_search), _SEARCH_ESTIMATOR_NAMES)


def _fair_influence_optimizer_name(spec: FIMPermutationSpec, config: FIMPermutationRunConfig) -> str:
    if spec.runner_kind == "community_aware_fair_greedy":
        return "Fairness-weighted greedy + local search" if int(config.local_search_steps) > 0 else "Fairness-weighted greedy"
    return _readable_name(spec.optimizer_mode, _OPTIMIZER_NAMES)


def _repair_strategy_name(spec: FIMPermutationSpec, config: FIMPermutationRunConfig) -> str:
    repair_enabled = spec.optimizer_mode in {"hybrid_si_ea", "local_search"} or bool(config.use_group_quota_repair)
    if not repair_enabled:
        return "None"
    if bool(config.use_fairness_first_repair) or _fairness_first_enabled(spec, config):
        return _REPAIR_NAMES["fairness_first"]
    if bool(config.use_group_quota_repair):
        return _REPAIR_NAMES["group_quota"]
    repair_mode = str(config.repair_mode or "").strip().lower()
    if repair_mode:
        return _readable_name(repair_mode, _REPAIR_NAMES)
    return "Enabled"


def _local_refinement_name(spec: FIMPermutationSpec, config: FIMPermutationRunConfig) -> str:
    if int(config.local_search_steps) <= 0:
        return "Disabled"
    if spec.optimizer_mode in {"hybrid_si_ea", "local_search"}:
        return _LOCAL_REFINEMENT_NAMES["swap_local_search"]
    return _LOCAL_REFINEMENT_NAMES["local_search"]


def _clustering_algorithm_name(spec: FIMPermutationSpec, clustering_input_mode: str) -> str:
    base = _readable_name(spec.clustering_method, _CLUSTERING_NAMES)
    if str(spec.clustering_method or "").strip().lower() in {"", "none", "false", "off"}:
        return base
    configured_input = str(spec.clustering_input_mode or "").strip().lower()
    resolved_input = str(clustering_input_mode or configured_input or "none").strip().lower()
    if configured_input == "auto" and resolved_input and resolved_input != "auto":
        return f"{base} ({resolved_input}, auto)"
    return base


def _candidate_pool_strategy_name(
    dataset: LoadedDataset | None,
    protected_group_report: ProtectedGroupReport | None,
    config: FIMPermutationRunConfig,
) -> str:
    auto_suffix = " (auto)" if config.use_group_stratified_candidate_pool is None else ""
    if dataset is not None and protected_group_report is not None:
        stratified = _effective_group_stratified_candidate_pool(dataset, protected_group_report, config)
        base = "Group-stratified candidate pool" if stratified else "Standard candidate pool"
    elif config.use_group_stratified_candidate_pool is None:
        base = "Auto"
    else:
        base = "Group-stratified candidate pool" if bool(config.use_group_stratified_candidate_pool) else "Standard candidate pool"
    return (
        f"{base}{auto_suffix} "
        f"(fraction={float(config.candidate_pool_fraction):.2f}, max={int(config.max_candidate_pool_size)})"
    )


def _score_normalization_name(
    dataset: LoadedDataset | None,
    protected_group_report: ProtectedGroupReport | None,
    config: FIMPermutationRunConfig,
) -> str:
    configured = str(config.score_normalization).strip().lower()
    effective = _effective_score_normalization(dataset, protected_group_report, config)
    if configured != effective:
        return f"{effective} (auto from {configured})"
    return effective


def _score_weights_summary(
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
    protected_group_report: ProtectedGroupReport | None,
    dataset: LoadedDataset | None,
) -> str:
    weights = _resolved_candidate_score_weights(spec, config, protected_group_report, dataset)
    keys = (
        ("ml", "ml_score_weight"),
        ("ris", "ris_score_weight"),
        ("fair_ris", "fair_ris_score_weight"),
        ("weak_group", "weak_group_bonus_weight"),
        ("coverage", "protected_group_coverage_weight"),
        ("diversity", "community_diversity_weight"),
    )
    return ", ".join(f"{label}={float(weights[key]):.3g}" for label, key in keys)


def stack_algorithm_summary_fields(
    stack_name: str | FIMPermutationSpec,
    config: FIMPermutationRunConfig,
    resolved_modules: Mapping[str, object] | None = None,
    *,
    dataset: LoadedDataset | None = None,
    protected_group_report: ProtectedGroupReport | None = None,
    clustering_input_mode: str | None = None,
) -> dict[str, object]:
    """Return human-readable algorithm composition for a resolved stack config."""

    spec = _resolve_fim_permutation_spec(stack_name)
    resolved_modules = dict(resolved_modules or {})
    resolved_clustering_input = str(
        resolved_modules.get(
            "clustering_input_mode",
            "none"
            if str(spec.clustering_method or "").strip().lower() in {"", "none"}
            else clustering_input_mode or spec.clustering_input_mode,
        )
    )
    ranking_policy = _effective_ranking_policy(spec, config)
    fair_ris_mode = _reported_ris_mode(config) if _spec_uses_ris(spec) else "None"
    return {
        "fim_stack": spec.name,
        "method_type_name": _readable_name(_method_type(spec), _METHOD_TYPE_NAMES),
        "graph_embedding_algorithm": _readable_name(spec.embedding_method, _EMBEDDING_NAMES),
        "ml_ranking_algorithm": _ml_ranking_algorithm_name(spec),
        "community_detection_algorithm": _readable_name(spec.community_method, _COMMUNITY_NAMES),
        "clustering_algorithm": _clustering_algorithm_name(spec, resolved_clustering_input),
        "diffusion_model_name": _readable_name(spec.diffusion_model, _DIFFUSION_NAMES),
        "search_time_spread_estimator": _search_estimator_algorithm_name(spec),
        "fair_ris_mode": fair_ris_mode,
        "fair_ris_enabled": _fair_ris_enabled(spec),
        "fairness_influence_objective": _readable_name(spec.fairness_objective, _OBJECTIVE_NAMES),
        "fair_influence_optimizer": _fair_influence_optimizer_name(spec, config),
        "repair_strategy": _repair_strategy_name(spec, config),
        "local_refinement": _local_refinement_name(spec, config),
        "final_evaluator": _readable_name(spec.spread_estimator_final, _FINAL_EVALUATOR_NAMES),
        "ranking_policy_name": _readable_name(ranking_policy, _RANKING_POLICY_NAMES),
        "scalability_mode_name": str(config.scalability_mode),
        "candidate_pool_strategy": _candidate_pool_strategy_name(dataset, protected_group_report, config),
        "score_normalization_name": _score_normalization_name(dataset, protected_group_report, config),
        "protected_attribute_usage_in_ml": "Allowed" if bool(config.allow_protected_features_in_ml) else "Disabled",
        "large_imbalance_fairness_mode_name": str(config.large_imbalance_fairness_mode),
        "score_weights_summary": _score_weights_summary(spec, config, protected_group_report, dataset),
        "budget": int(config.budget),
        "dataset": _safe_display_value(dataset.name if dataset is not None else resolved_modules.get("dataset")),
        "protected_attribute": _safe_display_value(
            protected_group_report.protected_attribute
            if protected_group_report is not None
            else config.protected_attribute
        ),
    }


def _algorithm_composition_fields(
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
    dataset: LoadedDataset | None,
    protected_group_report: ProtectedGroupReport | None,
    clustering_input_mode: str | None = None,
) -> dict[str, object]:
    fields = stack_algorithm_summary_fields(
        spec,
        config,
        dataset=dataset,
        protected_group_report=protected_group_report,
        clustering_input_mode=clustering_input_mode,
    )
    return {
        "graph_embedding_algorithm": fields["graph_embedding_algorithm"],
        "ml_ranking_algorithm": fields["ml_ranking_algorithm"],
        "community_detection_algorithm": fields["community_detection_algorithm"],
        "clustering_algorithm": fields["clustering_algorithm"],
        "diffusion_model_name": fields["diffusion_model_name"],
        "search_time_spread_estimator": fields["search_time_spread_estimator"],
        "fair_ris_mode": fields["fair_ris_mode"],
        "fairness_influence_objective": fields["fairness_influence_objective"],
        "fair_influence_optimizer": fields["fair_influence_optimizer"],
        "repair_strategy": fields["repair_strategy"],
        "local_refinement": fields["local_refinement"],
        "final_evaluator": fields["final_evaluator"],
        "ranking_policy_name": fields["ranking_policy_name"],
    }


def print_stack_algorithm_summary(
    stack_name: str | FIMPermutationSpec,
    config: FIMPermutationRunConfig,
    resolved_modules: Mapping[str, object] | None = None,
    *,
    dataset: LoadedDataset | None = None,
    protected_group_report: ProtectedGroupReport | None = None,
    clustering_input_mode: str | None = None,
) -> None:
    """Print a compact pre-run algorithm composition summary for one FIM stack."""

    fields = stack_algorithm_summary_fields(
        stack_name,
        config,
        resolved_modules=resolved_modules,
        dataset=dataset,
        protected_group_report=protected_group_report,
        clustering_input_mode=clustering_input_mode,
    )
    rows = [
        ("Method Type", "method_type_name"),
        ("Graph Embedding", "graph_embedding_algorithm"),
        ("ML Ranking Model", "ml_ranking_algorithm"),
        ("Community Detection", "community_detection_algorithm"),
        ("Clustering", "clustering_algorithm"),
        ("Diffusion Model", "diffusion_model_name"),
        ("Search-time Spread Estimator", "search_time_spread_estimator"),
        ("RIS Mode", "fair_ris_mode"),
        ("Fair RIS Enabled", "fair_ris_enabled"),
        ("Fairness / Influence Objective", "fairness_influence_objective"),
        ("Fair Influence Optimizer", "fair_influence_optimizer"),
        ("Repair Strategy", "repair_strategy"),
        ("Local Refinement", "local_refinement"),
        ("Final Evaluator", "final_evaluator"),
        ("Ranking Policy", "ranking_policy_name"),
        ("Scalability Mode", "scalability_mode_name"),
        ("Candidate Pool Strategy", "candidate_pool_strategy"),
        ("Score Normalization", "score_normalization_name"),
        ("Protected Attribute Usage in ML", "protected_attribute_usage_in_ml"),
        ("Large Imbalance Fairness Mode", "large_imbalance_fairness_mode_name"),
        ("Score Weights", "score_weights_summary"),
        ("Budget", "budget"),
        ("Dataset", "dataset"),
        ("Protected Attribute", "protected_attribute"),
    ]
    print("=" * 60)
    print(f"Running FIM Stack: {fields['fim_stack']}")
    print("=" * 60)
    for label, key in rows:
        print(f"{label:<32}: {fields[key]}")
    print("=" * 60)


def _safe_numeric(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _safe_int_value(value: object) -> int | None:
    numeric = _safe_numeric(value)
    if numeric is None:
        return None
    return int(numeric)


def _format_diagnostic_value(value: object) -> str:
    if _is_missing_value(value):
        return "n/a"
    text = str(value).strip()
    return text if text and text != "<NA>" else "n/a"


def _format_seconds(value: object) -> str:
    numeric = _safe_numeric(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.2f}s"


def _format_percent(value: object) -> str:
    numeric = _safe_numeric(value)
    if numeric is None:
        return "n/a"
    return f"{100.0 * numeric:.2f}%"


def _parse_json_sequence(value: object) -> list[object]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if _is_missing_value(value):
        return []
    text = str(value).strip()
    if not text or text == "<NA>":
        return []
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001 - diagnostics are best effort.
        return []
    return list(parsed) if isinstance(parsed, list) else []


def _parse_json_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if _is_missing_value(value):
        return {}
    text = str(value).strip()
    if not text or text == "<NA>":
        return {}
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001 - diagnostics are best effort.
        return {}
    if isinstance(parsed, Mapping):
        return {str(key): item for key, item in parsed.items()}
    return {}


def _format_compact_mapping(value: object) -> str:
    mapping = _parse_json_mapping(value)
    if not mapping:
        return "n/a"
    return "{" + ", ".join(f"{key}:{mapping[key]}" for key in sorted(mapping)) + "}"


def _format_table_number(value: object, digits: int = 4) -> str:
    numeric = _safe_numeric(value)
    if numeric is None:
        return _format_diagnostic_value(value)
    return f"{numeric:.{digits}f}"


def _first_resolved_spec(permutations: Iterable[str | FIMPermutationSpec]) -> FIMPermutationSpec | None:
    for value in permutations:
        try:
            return _resolve_fim_permutation_spec(value)
        except Exception:
            continue
    return None


def _shared_spec_value(
    specs: Sequence[FIMPermutationSpec],
    getter: Callable[[FIMPermutationSpec], object],
    fallback: str = "mixed",
) -> object:
    values = [getter(spec) for spec in specs]
    if not values:
        return "n/a"
    first = values[0]
    if all(str(value) == str(first) for value in values):
        return first
    return fallback


def print_experiment_setup_header(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    config: FIMPermutationRunConfig,
    permutations: Iterable[str | FIMPermutationSpec],
    *,
    dataset_loading_seconds: float | None = None,
    preprocessing_seconds: float | None = None,
) -> None:
    specs = [
        _resolve_fim_permutation_spec(value)
        for value in permutations
    ]
    first_spec = specs[0] if specs else None
    diffusion = (
        _readable_name(_shared_spec_value(specs, lambda spec: spec.diffusion_model), _DIFFUSION_NAMES)
        if first_spec is not None
        else "n/a"
    )
    search = (
        _search_estimator_algorithm_name(first_spec)
        if first_spec is not None and _shared_spec_value(specs, lambda spec: _canonical_search_estimator(spec.spread_estimator_search)) != "mixed"
        else "mixed"
    )
    node_count = max(1, int(dataset.graph.number_of_nodes()))
    budget_ratio = float(config.budget) / float(node_count)
    print("Experiment Setup")
    print("-" * 60)
    print(f"{'Dataset':<30}: {dataset.name}")
    print(f"{'Nodes / Edges':<30}: {dataset.graph.number_of_nodes()} / {dataset.graph.number_of_edges()}")
    print(f"{'Protected Attribute':<30}: {protected_group_report.protected_attribute}")
    print(f"{'Protected Groups':<30}: {len(protected_group_report.group_sizes)}")
    print(f"{'Group Counts':<30}: {_format_compact_mapping(protected_group_report.group_sizes)}")
    print(f"{'Imbalance Ratio':<30}: {_protected_group_imbalance(protected_group_report)['imbalance_ratio']:.2f}")
    print(f"{'Budget':<30}: {int(config.budget)}")
    print(f"{'Budget / Nodes':<30}: {100.0 * budget_ratio:.2f}%")
    print(f"{'Communities':<30}: computed per stack")
    print(f"{'Ranking Policy':<30}: {_effective_ranking_policy(first_spec, config) if first_spec is not None else config.ranking_policy}")
    print(f"{'Diffusion Model':<30}: {diffusion}")
    print(f"{'Search Estimator':<30}: {search}")
    print(f"{'Final Evaluator':<30}: Monte Carlo")
    print(f"{'MC Runs Search':<30}: {int(config.mc_runs_search)}")
    print(f"{'MC Runs Eval':<30}: {int(config.mc_runs_eval)}")
    print(f"{'Random Seed':<30}: {int(config.random_seed)}")
    print(f"{'Output Directory':<30}: {_format_diagnostic_value(config.output_dir)}")
    print(f"{'Dataset Loading Time':<30}: {_format_seconds(dataset_loading_seconds)}")
    print(f"{'Preprocessing Time':<30}: {_format_seconds(preprocessing_seconds)}")
    print("")


def print_budget_adequacy_check(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    config: FIMPermutationRunConfig,
    *,
    community_count: int | None = None,
) -> None:
    node_count = max(1, int(dataset.graph.number_of_nodes()))
    budget_ratio = float(config.budget) / float(node_count)
    group_sizes = [int(size) for size in protected_group_report.group_sizes.values() if int(size) > 0]
    imbalance = _protected_group_imbalance(protected_group_report)
    warnings = _budget_adequacy_warnings(dataset, protected_group_report, config, None)
    if community_count is not None and bool(config.warn_if_communities_exceed_budget) and int(community_count) > int(config.budget):
        warnings.append("number of communities exceeds budget")
    print("Budget Adequacy Check")
    print("-" * 60)
    print(f"{'Budget / Nodes':<30}: {100.0 * budget_ratio:.2f}%")
    communities_text = (
        "Yes" if community_count is not None and int(community_count) > int(config.budget)
        else ("No" if community_count is not None else "computed per stack")
    )
    print(f"{'Communities > Budget?':<30}: {communities_text}")
    print(f"{'Protected Groups > Budget?':<30}: {'Yes' if len(group_sizes) > int(config.budget) else 'No'}")
    print(f"{'Smallest Group Size':<30}: {min(group_sizes) if group_sizes else 'n/a'}")
    print(f"{'Largest Group Size':<30}: {max(group_sizes) if group_sizes else 'n/a'}")
    print(f"{'Imbalance Ratio':<30}: {float(imbalance['imbalance_ratio']):.2f}")
    print(f"{'Warning':<30}: {'; '.join(dict.fromkeys(warnings)) if warnings else 'n/a'}")
    print("")


def _runtime_breakdown_from_row(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "Dataset Loading": row.get("time_dataset_loading", pd.NA),
        "Preprocessing": row.get("time_preprocessing", pd.NA),
        "Community Detection": row.get("time_community_detection", pd.NA),
        "Embedding Generation": row.get("time_embedding", pd.NA),
        "RIS / RR-set Generation": row.get("time_ris", pd.NA),
        "Candidate Scoring": row.get("time_candidate_scoring", pd.NA),
        "SI+EA Optimization": row.get("time_optimizer", pd.NA),
        "Repair": row.get("time_repair", pd.NA),
        "Swap Local Search": row.get("time_local_search", pd.NA),
        "Final MC Evaluation": row.get("time_final_mc", row.get("final_eval_runtime_seconds", pd.NA)),
        "Reporting": row.get("time_reporting", pd.NA),
        "Total Runtime": row.get("runtime_seconds", pd.NA),
    }


def print_runtime_breakdown(row: Mapping[str, object]) -> None:
    print("Runtime Breakdown")
    print("-" * 60)
    for label, value in _runtime_breakdown_from_row(row).items():
        print(f"{label:<30}: {_format_seconds(value)}")
    print("")


def print_final_seed_set_diagnostics(row: Mapping[str, object]) -> None:
    seed_set = _parse_json_sequence(row.get("seed_set", []))
    seed_group_counts = _parse_json_mapping(row.get("seed_group_counts", row.get("final_protected_group_coverage", {})))
    community_counts = _parse_json_mapping(row.get("seed_community_counts", row.get("final_seed_count_per_community", {})))
    duplicate_count = len(seed_set) - len(set(str(value) for value in seed_set)) if seed_set else _safe_int_value(row.get("duplicate_seed_count")) or 0
    covered_communities = sum(1 for value in community_counts.values() if (_safe_numeric(value) or 0.0) > 0.0)
    largest_community = max((_safe_numeric(value) or 0.0 for value in community_counts.values()), default=0.0)
    expected = _safe_int_value(row.get("budget")) or _safe_int_value(row.get("expected_seed_count")) or 0
    group_values = [int(_safe_numeric(value) or 0) for value in seed_group_counts.values()]
    print("Final Seed Set Diagnostics")
    print("-" * 60)
    print(f"{'Seed Count':<30}: {len(seed_set) if seed_set else _format_diagnostic_value(row.get('seed_count'))} / {expected or 'n/a'}")
    print(f"{'Duplicate Seeds':<30}: {duplicate_count}")
    print(f"{'Seed Groups':<30}: {_format_compact_mapping(seed_group_counts)}")
    print(f"{'Seed Communities Covered':<30}: {covered_communities if community_counts else _format_diagnostic_value(row.get('seed_communities_covered'))} / {_format_diagnostic_value(row.get('num_communities'))}")
    largest_pct = f"{100.0 * largest_community / max(1, expected):.1f}%" if community_counts and expected else "n/a"
    print(f"{'Largest Community Seed %':<30}: {largest_pct}")
    print(f"{'Smallest Group Seed Count':<30}: {min(group_values) if group_values else 'n/a'}")
    print(f"{'Largest Group Seed Count':<30}: {max(group_values) if group_values else 'n/a'}")
    print("")


def print_protected_group_influence(row: Mapping[str, object]) -> None:
    group_sizes = _parse_json_mapping(row.get("protected_group_counts", {}))
    seed_counts = _parse_json_mapping(row.get("seed_group_counts", row.get("final_protected_group_coverage", {})))
    influence = _parse_json_mapping(row.get("protected_group_influence_json", row.get("group_influence_distribution", {})))
    normalized = _parse_json_mapping(row.get("protected_group_normalized_influence_json", row.get("normalized_group_influence_distribution", {})))
    groups = sorted(set(group_sizes) | set(seed_counts) | set(influence) | set(normalized))
    print("Protected Group Influence")
    print("-" * 60)
    print(f"{'Group':<10}{'Size':<10}{'Seed Count':<14}{'Influence':<18}{'Normalized Influence'}")
    if groups:
        for group_name in groups:
            print(
                f"{group_name:<10}"
                f"{_format_diagnostic_value(group_sizes.get(group_name)):<10}"
                f"{_format_diagnostic_value(seed_counts.get(group_name)):<14}"
                f"{_format_table_number(influence.get(group_name)):<18}"
                f"{_format_table_number(normalized.get(group_name))}"
            )
    else:
        print("n/a")
    print("")
    print(f"{'Weakest Group':<30}: {_format_diagnostic_value(row.get('weakest_group', row.get('weakest_protected_group')))}")
    print(f"{'Strongest Group':<30}: {_format_diagnostic_value(row.get('strongest_group', row.get('strongest_protected_group')))}")
    print(f"{'MF':<30}: {_format_diagnostic_value(row.get('mf'))}")
    print(f"{'DCV':<30}: {_format_diagnostic_value(row.get('dcv'))}")
    print(f"{'F-score':<30}: {_format_diagnostic_value(row.get('f_score'))}")
    print("")


def print_candidate_score_diagnostics(row: Mapping[str, object]) -> None:
    stats = _parse_json_mapping(row.get("score_component_stats_json", {}))
    constants = _parse_json_sequence(row.get("constant_score_components", []))
    print("Candidate Score Diagnostics")
    print("-" * 60)
    print(f"{'Score Normalization':<30}: {_format_diagnostic_value(row.get('score_normalization'))}")
    print(f"{'Candidate Pool Size':<30}: {_format_diagnostic_value(row.get('candidate_pool_size'))}")
    print(f"{'Group-stratified Pool':<30}: {_format_diagnostic_value(row.get('group_stratified_candidate_pool'))}")
    for column_name, label in (
        ("ml_score", "ML Score"),
        ("ris_score", "RIS Score"),
        ("fair_ris_score", "Fair RIS Score"),
        ("fairness_bonus", "Fairness Bonus"),
        ("weak_group_bonus", "Weak Group Bonus"),
        ("protected_group_coverage_bonus", "Coverage Bonus"),
        ("community_diversity_bonus", "Community Bonus"),
        ("spread_proxy_score", "Spread Proxy Score"),
        ("combined_score", "Combined Score"),
    ):
        stat = stats.get(column_name, {}) if isinstance(stats.get(column_name), Mapping) else {}
        print(
            f"{label + ' Std':<30}: "
            f"{_format_diagnostic_value(stat.get('std') if stat else pd.NA)}"
        )
    print(f"{'Any Constant Components?':<30}: {'Yes' if constants else 'No'}")
    print(f"{'Constant Components':<30}: {constants if constants else 'None'}")
    print(f"{'RIS Active Verified':<30}: {_format_diagnostic_value(row.get('ris_active_verified', row.get('ris_verified')))}")
    print(f"{'Fair RIS Active Verified':<30}: {_format_diagnostic_value(row.get('fair_ris_active_verified', row.get('fair_ris_verified')))}")
    print("")


def print_optimizer_diagnostics(row: Mapping[str, object]) -> None:
    print("Optimizer Diagnostics")
    print("-" * 60)
    rows = [
        ("Initial Population Size", row.get("population_size", pd.NA)),
        ("Generations", row.get("generations", pd.NA)),
        ("Candidate Pool Size", row.get("candidate_pool_size", pd.NA)),
        ("Repair Attempts", row.get("repair_attempts", pd.NA)),
        ("Weak-group Repairs", row.get("weak_group_repairs", pd.NA)),
        ("Duplicate Repairs", row.get("duplicate_repairs", pd.NA)),
        ("Swap Attempts", row.get("swap_attempts", pd.NA)),
        ("Successful Swaps", row.get("successful_swaps", row.get("swap_accepted", pd.NA))),
        ("Rejected Fairness Drops", row.get("rejected_fairness_drops", row.get("swap_rejected_fairness_degradation", pd.NA))),
        ("Accepted F-score Swaps", row.get("swap_accepted_fscore_improvement", pd.NA)),
        ("Accepted Spread-safe Swaps", row.get("swap_accepted_spread_fairness_preserved", pd.NA)),
        ("Best Generation", row.get("best_generation", pd.NA)),
        ("Final Fitness", row.get("final_fitness", pd.NA)),
    ]
    for label, value in rows:
        print(f"{label:<30}: {_format_diagnostic_value(value)}")
    print("")


def print_stack_result_diagnostics(frame: pd.DataFrame, config: FIMPermutationRunConfig) -> None:
    for _, row in frame.iterrows():
        status = str(row.get("status", "")).strip().lower()
        if status != "ok":
            print("Method Failure")
            print("-" * 60)
            print(f"{'Method':<30}: {_format_diagnostic_value(row.get('stack_name'))}")
            print(f"{'Reason':<30}: {_format_diagnostic_value(row.get('skip_reason', row.get('skipped_reason')))}")
            print("")
            continue
        if bool(config.print_runtime_breakdown):
            print_runtime_breakdown(row)
        if bool(config.print_seed_diagnostics):
            print_final_seed_set_diagnostics(row)
        if bool(config.print_group_influence):
            print_protected_group_influence(row)
        if bool(config.print_score_diagnostics):
            print_candidate_score_diagnostics(row)
        if bool(config.print_optimizer_diagnostics):
            print_optimizer_diagnostics(row)


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
        f"search_estimator={_canonical_search_estimator(spec.spread_estimator_search)}",
        f"final_estimator={spec.spread_estimator_final}",
        f"fair_ris={_fair_ris_enabled(spec)}",
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
    score_weights = _resolved_candidate_score_weights(spec, config, protected_group_report, dataset)
    imbalance = _protected_group_imbalance(protected_group_report)
    total_group_support = max(1, int(sum(protected_group_report.group_sizes.values())))
    budget_per_group_estimate = {
        group_name: float(config.budget) * float(group_size) / float(total_group_support)
        for group_name, group_size in protected_group_report.group_sizes.items()
    }
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
        "search_estimator": _canonical_search_estimator(spec.spread_estimator_search),
        "final_estimator": spec.spread_estimator_final,
        "spread_estimator_search": _canonical_search_estimator(spec.spread_estimator_search),
        "spread_estimator_final": spec.spread_estimator_final,
        "search_spread_estimator": (
            "ris_guidance"
            if _canonical_search_estimator(spec.spread_estimator_search) == "fairness_aware_ris"
            else _canonical_search_estimator(spec.spread_estimator_search)
        ),
        "search_guidance_estimator": _search_guidance_estimator(spec),
        "final_spread_estimator": spec.spread_estimator_final,
        "method": method,
        "variant_type": _FIM_VARIANT_TYPES[spec.variant_family],
        "runtime_seconds": float(search_runtime_seconds + evaluation.runtime_seconds),
        "search_runtime_seconds": float(search_runtime_seconds),
        "time_dataset_loading": pd.NA,
        "time_preprocessing": pd.NA,
        "time_community_detection": pd.NA,
        "time_embedding": pd.NA,
        "time_ris": pd.NA,
        "time_candidate_scoring": pd.NA,
        "time_optimizer": pd.NA,
        "time_repair": pd.NA,
        "time_local_search": pd.NA,
        "time_final_mc": float(evaluation.runtime_seconds),
        "time_reporting": pd.NA,
        "runtime_breakdown_json": "",
        "scalability_mode": str(config.scalability_mode),
        "scalability_pass": pd.NA,
        "candidate_pool_size": pd.NA,
        "use_ris": _spec_uses_ris(spec),
        "use_fair_ris": bool(spec.use_fair_ris),
        "fair_ris_enabled": _fair_ris_enabled(spec),
        "ris_mode": _reported_ris_mode(config),
        "ris_num_rr_sets": int(config.ris_num_rr_sets),
        "effective_ris_num_rr_sets": _effective_ris_rr_sets(dataset, config, protected_group_report),
        "ris_reuse_rr_sets": bool(config.ris_reuse_rr_sets),
        "ris_score_nonzero_count": 0,
        "ris_score_std": 0.0,
        "fair_ris_score_nonzero_count": 0,
        "fair_ris_score_std": 0.0,
        "ris_verified": False,
        "fair_ris_verified": False,
        "ris_active_verified": False,
        "fair_ris_active_verified": False,
        "ris_verification_warnings": "",
        "force_ris_for_all_stacks": bool(config.force_ris_for_all_stacks),
        "require_ris": bool(config.require_ris),
        "mc_runs_search": int(config.mc_runs_search),
        "mc_runs_eval": int(config.mc_runs_eval),
        "population_size": int(config.population_size),
        "generations": int(config.generations),
        "key_enabled_modules": _key_enabled_modules(spec, resolved_clustering_input_mode),
        "use_community_features": bool(config.use_community_features_for_ml),
        "community_feature_mode": str(config.community_feature_mode),
        "allow_protected_features_in_ml": bool(config.allow_protected_features_in_ml),
        "ml_score_weight": score_weights["ml_score_weight"],
        "ris_score_weight": score_weights["ris_score_weight"],
        "fair_ris_score_weight": score_weights["fair_ris_score_weight"],
        "fairness_bonus_weight": float(config.fairness_bonus_weight),
        "weak_group_bonus_weight": score_weights["weak_group_bonus_weight"],
        "community_diversity_weight": score_weights["community_diversity_weight"],
        "protected_group_coverage_weight": score_weights["protected_group_coverage_weight"],
        "spread_proxy_weight": score_weights["spread_proxy_weight"],
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
        "protected_group_counts": dict(protected_group_report.group_sizes),
        "group_imbalance_ratio": float(imbalance["imbalance_ratio"]),
        "budget_per_group_estimate": budget_per_group_estimate,
        "group_influence_distribution": pd.NA,
        "normalized_group_influence_distribution": pd.NA,
        "weakest_protected_group": pd.NA,
        "strongest_protected_group": pd.NA,
        "negative_f_score_reason": pd.NA,
        "adaptive_fairness_multiplier": _adaptive_fairness_multiplier(config, protected_group_report),
        "large_imbalance_fairness_active": _large_imbalance_fairness_active(dataset, protected_group_report, config),
        "large_imbalance_fairness_mode": str(config.large_imbalance_fairness_mode),
        "score_normalization": _effective_score_normalization(dataset, protected_group_report, config),
        "candidate_pool_group_counts": pd.NA,
        "candidate_pool_group_quota": pd.NA,
        "candidate_pool_group_quota_shortfall": pd.NA,
        "group_stratified_candidate_pool": pd.NA,
        "budget_adequacy_warning": pd.NA,
        "seed_quota_per_protected_group": pd.NA,
        "initial_population_group_coverage_summary": pd.NA,
        "repair_attempts": pd.NA,
        "weak_group_repairs": pd.NA,
        "protected_group_seed_counts_before_repair": pd.NA,
        "protected_group_seed_counts_after_repair": pd.NA,
        "swap_attempts": pd.NA,
        "swap_accepted": pd.NA,
        "swap_rejected_fairness_degradation": pd.NA,
        "swap_accepted_fscore_improvement": pd.NA,
        "swap_accepted_mf_improvement": pd.NA,
        "swap_accepted_spread_fairness_preserved": pd.NA,
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
    row.update(
        _algorithm_composition_fields(
            spec,
            config,
            dataset,
            protected_group_report,
            resolved_clustering_input_mode,
        )
    )
    row.update(_spread_dict(evaluation, int(config.budget)))
    row.update(_fairness_coverage_fields(evaluation))
    row.update(_seed_diagnostic_fields(evaluation, protected_group_report, config))
    if extra_fields:
        row.update(dict(extra_fields))
    candidate_pool_raw = pd.to_numeric(pd.Series([row.get("candidate_pool_size", pd.NA)]), errors="coerce").iloc[0]
    row["scalability_pass"] = _scalability_pass(
        dataset,
        config,
        None if pd.isna(candidate_pool_raw) else int(candidate_pool_raw),
    )
    gate_failures = fairness_gate_failures(row, professor_priority_config_from_object(config))
    row["fairness_gate_status"] = "warn" if gate_failures and bool(config.warn_only_fairness_gates) else ("fail" if gate_failures else "pass")
    row["fairness_gate_failures"] = ";".join(gate_failures)
    return _finalize_diagnostic_row(row)


def _skipped_row(
    *,
    spec: FIMPermutationSpec,
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    config: FIMPermutationRunConfig,
    skip_reason: str,
) -> dict[str, object]:
    row = {column: pd.NA for column in permutation_summary_columns()}
    imbalance = _protected_group_imbalance(protected_group_report)
    total_group_support = max(1, int(sum(protected_group_report.group_sizes.values())))
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
            "search_estimator": _canonical_search_estimator(spec.spread_estimator_search),
            "final_estimator": spec.spread_estimator_final,
            "spread_estimator_search": _canonical_search_estimator(spec.spread_estimator_search),
            "spread_estimator_final": spec.spread_estimator_final,
            "search_spread_estimator": (
                "ris_guidance"
                if _canonical_search_estimator(spec.spread_estimator_search) == "fairness_aware_ris"
                else _canonical_search_estimator(spec.spread_estimator_search)
            ),
            "search_guidance_estimator": _search_guidance_estimator(spec),
            "final_spread_estimator": spec.spread_estimator_final,
            "mc_runs_search": int(config.mc_runs_search),
            "mc_runs_eval": int(config.mc_runs_eval),
            "scalability_mode": str(config.scalability_mode),
            "use_ris": _spec_uses_ris(spec),
            "use_fair_ris": bool(spec.use_fair_ris),
            "fair_ris_enabled": _fair_ris_enabled(spec),
            "ris_mode": _reported_ris_mode(config),
            "ris_num_rr_sets": int(config.ris_num_rr_sets),
            "effective_ris_num_rr_sets": _effective_ris_rr_sets(dataset, config, protected_group_report),
            "ris_reuse_rr_sets": bool(config.ris_reuse_rr_sets),
            "ris_score_nonzero_count": 0,
            "ris_score_std": 0.0,
            "fair_ris_score_nonzero_count": 0,
            "fair_ris_score_std": 0.0,
            "ris_verified": False,
            "fair_ris_verified": False,
            "ris_active_verified": False,
            "fair_ris_active_verified": False,
            "ris_verification_warnings": "",
            "force_ris_for_all_stacks": bool(config.force_ris_for_all_stacks),
            "require_ris": bool(config.require_ris),
            "fairness_gate_status": "fail",
            "fairness_gate_failures": "status",
            "protected_group_counts": dict(protected_group_report.group_sizes),
            "group_imbalance_ratio": float(imbalance["imbalance_ratio"]),
            "budget_per_group_estimate": {
                group_name: float(config.budget) * float(group_size) / float(total_group_support)
                for group_name, group_size in protected_group_report.group_sizes.items()
            },
            "adaptive_fairness_multiplier": _adaptive_fairness_multiplier(config, protected_group_report),
            "large_imbalance_fairness_active": _large_imbalance_fairness_active(dataset, protected_group_report, config),
            "large_imbalance_fairness_mode": str(config.large_imbalance_fairness_mode),
            "score_normalization": _effective_score_normalization(dataset, protected_group_report, config),
            "budget_adequacy_warning": _budget_adequacy_warning_text(dataset, protected_group_report, config),
            "key_enabled_modules": _key_enabled_modules(spec, spec.clustering_input_mode),
            "use_community_features": bool(config.use_community_features_for_ml),
            "community_feature_mode": str(config.community_feature_mode),
            "allow_protected_features_in_ml": bool(config.allow_protected_features_in_ml),
            "notes": spec.notes,
            "skip_reason": skip_reason,
            "skipped_reason": skip_reason,
        }
    )
    row.update(
        _algorithm_composition_fields(
            spec,
            config,
            dataset,
            protected_group_report,
            spec.clustering_input_mode,
        )
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
    protected_group_report: ProtectedGroupReport | None = None,
    score_normalization: str = "global",
) -> pd.DataFrame:
    normalized_ranking = _normalize_score_map_for_mode(
        ranking_scores or {},
        protected_group_report=protected_group_report,
        score_normalization=score_normalization,
        component_name="ml_score",
    )
    normalized_ris = _normalize_score_map_for_mode(
        ris_scores or {},
        protected_group_report=protected_group_report,
        score_normalization=score_normalization,
        component_name="ris_score",
    )
    normalized_fair_ris = _normalize_score_map_for_mode(
        fair_ris_scores or {},
        protected_group_report=protected_group_report,
        score_normalization=score_normalization,
        component_name="fair_ris_score",
    )
    normalized_weak_group_bonus = _normalize_score_map_for_mode(
        weak_group_bonus_scores or fairness_bonus_scores or {},
        protected_group_report=protected_group_report,
        score_normalization=score_normalization,
        component_name="weak_group_bonus",
    )
    normalized_diversity_bonus = _normalize_score_map_for_mode(
        diversity_bonus_scores or {},
        protected_group_report=protected_group_report,
        score_normalization=score_normalization,
        component_name="community_diversity_bonus",
    )
    normalized_protected_group_coverage = _normalize_score_map_for_mode(
        protected_group_coverage_scores or {},
        protected_group_report=protected_group_report,
        score_normalization=score_normalization,
        component_name="protected_group_coverage_bonus",
    )
    normalized_spread_proxy = _normalize_score_map_for_mode(
        spread_proxy_scores or {},
        protected_group_report=protected_group_report,
        score_normalization=score_normalization,
        component_name="spread_proxy_score",
    )
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
                "score_normalization",
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
                "score_normalization": str(score_normalization),
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
    protected_group_report: ProtectedGroupReport | None = None,
    score_normalization: str = "global",
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
        protected_group_report=protected_group_report,
        score_normalization=score_normalization,
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


def _normalize_score_map_for_mode(
    scores: Mapping[Any, float],
    *,
    protected_group_report: ProtectedGroupReport | None = None,
    score_normalization: str = "global",
    component_name: str = "score",
) -> dict[Any, float]:
    if not scores:
        return {}
    mode = str(score_normalization).strip().lower()
    if mode not in {"global", "per_group", "hybrid"} or protected_group_report is None:
        mode = "global"
    if mode == "global":
        return _normalize_score_map(scores)

    global_normalized = _normalize_score_map(scores)
    group_by_node = {
        node_id: group_name
        for group_name, node_ids in protected_group_report.protected_groups.items()
        for node_id in node_ids
    }
    per_group: dict[Any, float] = {}
    for group_name in sorted(protected_group_report.group_sizes, key=_sort_key):
        group_nodes = sorted(
            [
                node_id
                for node_id, mapped_group in group_by_node.items()
                if mapped_group == group_name and node_id in scores
            ],
            key=_sort_key,
        )
        if not group_nodes:
            continue
        normalized = safe_minmax_normalize(
            [float(scores[node_id]) for node_id in group_nodes],
            default=0.0,
            context=f"{component_name} per-group normalization for {group_name}",
        )
        for node_id, value in zip(group_nodes, normalized, strict=True):
            per_group[node_id] = float(value)

    for node_id in scores:
        per_group.setdefault(node_id, float(global_normalized.get(node_id, 0.0)))

    if mode == "hybrid":
        return {
            node_id: 0.5 * float(global_normalized.get(node_id, 0.0)) + 0.5 * float(per_group.get(node_id, 0.0))
            for node_id in sorted(scores, key=_sort_key)
        }
    return {
        node_id: float(per_group.get(node_id, 0.0))
        for node_id in sorted(scores, key=_sort_key)
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


def _seed_community_coverage_ratio(seed_set: Sequence[Any], community_result) -> float:
    covered_communities = {
        community_result.community_id_by_node[node_id]
        for node_id in seed_set
        if node_id in community_result.community_id_by_node
    }
    return float(len(covered_communities)) / float(max(1, int(community_result.stats.num_communities)))


def _fairness_coverage_fields(evaluation: SeedSetEvaluation) -> dict[str, object]:
    group_spread = dict(evaluation.fairness.group_spread)
    normalized_group_spread = dict(evaluation.fairness.normalized_group_spread)
    covered_groups = [group_name for group_name, value in group_spread.items() if float(value) > 0.0]
    total_groups = max(1, len(group_spread))
    fraction_groups_covered = float(len(covered_groups)) / float(total_groups)
    weakest_group = None
    strongest_group = None
    if normalized_group_spread:
        weakest_group = min(normalized_group_spread, key=lambda group_name: (float(normalized_group_spread[group_name]), _sort_key(group_name)))
        strongest_group = max(normalized_group_spread, key=lambda group_name: (float(normalized_group_spread[group_name]), _sort_key(group_name)))
    negative_reason = ""
    if float(evaluation.f_score) < 0.0:
        negative_reason = (
            "DCV exceeds MF after lambda weighting; weakest protected group has low normalized influence."
        )
    return {
        "zero_covered_groups_count": int(total_groups - len(covered_groups)),
        "fraction_groups_covered": fraction_groups_covered,
        "group_influence_distribution": group_spread,
        "normalized_group_influence_distribution": normalized_group_spread,
        "weakest_protected_group": "" if weakest_group is None else str(weakest_group),
        "strongest_protected_group": "" if strongest_group is None else str(strongest_group),
        "negative_f_score_reason": negative_reason,
    }


def _seed_diagnostic_fields(
    evaluation: SeedSetEvaluation,
    protected_group_report: ProtectedGroupReport,
    config: FIMPermutationRunConfig,
) -> dict[str, object]:
    seed_list = list(evaluation.seed_set)
    duplicate_count = len(seed_list) - len(set(seed_list))
    seed_group_counts = _seed_protected_group_coverage(seed_list, protected_group_report)
    seed_group_mapping = _parse_json_mapping(seed_group_counts)
    group_seed_values = [int(_safe_numeric(value) or 0) for value in seed_group_mapping.values()]
    group_spread = dict(evaluation.fairness.group_spread)
    normalized_group_spread = dict(evaluation.fairness.normalized_group_spread)
    weakest = ""
    strongest = ""
    if normalized_group_spread:
        weakest = str(min(normalized_group_spread, key=lambda group_name: (float(normalized_group_spread[group_name]), _sort_key(group_name))))
        strongest = str(max(normalized_group_spread, key=lambda group_name: (float(normalized_group_spread[group_name]), _sort_key(group_name))))
    return {
        "seed_count": int(len(seed_list)),
        "expected_seed_count": int(config.budget),
        "duplicate_seed_count": int(duplicate_count),
        "seed_group_counts": seed_group_counts,
        "smallest_group_seed_count": min(group_seed_values) if group_seed_values else pd.NA,
        "largest_group_seed_count": max(group_seed_values) if group_seed_values else pd.NA,
        "protected_group_influence_json": json.dumps(_json_ready_mapping(group_spread), sort_keys=True),
        "protected_group_normalized_influence_json": json.dumps(_json_ready_mapping(normalized_group_spread), sort_keys=True),
        "weakest_group": weakest,
        "strongest_group": strongest,
    }


def _row_runtime_breakdown_json(row: Mapping[str, object]) -> str:
    payload = {
        label.lower().replace(" / ", "_").replace(" ", "_").replace("+", "")
        : (None if _is_missing_value(value) else float(value))
        for label, value in _runtime_breakdown_from_row(row).items()
    }
    return json.dumps(payload, sort_keys=True)


def _optimizer_diagnostics_json(row: Mapping[str, object]) -> str:
    payload = {
        "initial_population_size": row.get("population_size", pd.NA),
        "generations": row.get("generations", pd.NA),
        "candidate_pool_size": row.get("candidate_pool_size", pd.NA),
        "repair_attempts": row.get("repair_attempts", pd.NA),
        "weak_group_repairs": row.get("weak_group_repairs", pd.NA),
        "duplicate_repairs": row.get("duplicate_repairs", pd.NA),
        "swap_attempts": row.get("swap_attempts", pd.NA),
        "successful_swaps": row.get("successful_swaps", row.get("swap_accepted", pd.NA)),
        "rejected_fairness_drops": row.get("rejected_fairness_drops", row.get("swap_rejected_fairness_degradation", pd.NA)),
        "accepted_fscore_swaps": row.get("swap_accepted_fscore_improvement", pd.NA),
        "accepted_spread_safe_swaps": row.get("swap_accepted_spread_fairness_preserved", pd.NA),
        "best_generation": row.get("best_generation", pd.NA),
        "final_fitness": row.get("final_fitness", pd.NA),
    }
    return json.dumps(
        {
            key: (None if _is_missing_value(value) else value)
            for key, value in payload.items()
        },
        sort_keys=True,
        default=str,
    )


def _finalize_diagnostic_row(row: dict[str, object]) -> dict[str, object]:
    row["time_final_mc"] = row.get("time_final_mc", row.get("final_eval_runtime_seconds", pd.NA))
    row["time_optimizer"] = row.get("time_optimizer", pd.NA)
    if _is_missing_value(row.get("time_optimizer")) and str(row.get("optimizer_mode", "")).strip().lower() == "hybrid_si_ea":
        row["time_optimizer"] = row.get("search_runtime_seconds", pd.NA)
    row["zero_covered_groups"] = row.get("zero_covered_groups_count", pd.NA)
    row["weakest_group"] = row.get("weakest_group", row.get("weakest_protected_group", pd.NA))
    row["strongest_group"] = row.get("strongest_group", row.get("strongest_protected_group", pd.NA))
    if _is_missing_value(row.get("seed_community_counts")):
        row["seed_community_counts"] = row.get("final_seed_count_per_community", pd.NA)
    community_counts = _parse_json_mapping(row.get("seed_community_counts", {}))
    if community_counts:
        positive = [float(_safe_numeric(value) or 0.0) for value in community_counts.values()]
        row["seed_communities_covered"] = int(sum(1 for value in positive if value > 0.0))
        seed_count = max(1, int(_safe_numeric(row.get("seed_count")) or _safe_numeric(row.get("budget")) or 1))
        row["largest_community_seed_fraction"] = float(max(positive, default=0.0) / float(seed_count))
    else:
        row["seed_communities_covered"] = row.get("seed_communities_covered", pd.NA)
        row["largest_community_seed_fraction"] = row.get("largest_community_seed_fraction", pd.NA)
    row["rejected_fairness_drops"] = row.get(
        "rejected_fairness_drops",
        row.get("swap_rejected_fairness_degradation", pd.NA),
    )
    row["duplicate_repairs"] = row.get("duplicate_repairs", pd.NA)
    row["runtime_breakdown_json"] = _row_runtime_breakdown_json(row)
    row["optimizer_diagnostics_json"] = _optimizer_diagnostics_json(row)
    return row


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
    history = getattr(optimization_result, "history", pd.DataFrame())
    best_generation: object = pd.NA
    if isinstance(history, pd.DataFrame) and not history.empty:
        score_column = "best_score" if "best_score" in history.columns else None
        generation_column = "generation" if "generation" in history.columns else None
        if score_column is not None and generation_column is not None:
            best_index = pd.to_numeric(history[score_column], errors="coerce").idxmax()
            if best_index in history.index:
                best_generation = history.at[best_index, generation_column]
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
        "repair_attempts": int(getattr(optimization_result, "repair_attempts", 0)),
        "weak_group_repairs": int(getattr(optimization_result, "weak_group_repairs", 0)),
        "protected_group_seed_counts_before_repair": getattr(optimization_result, "protected_group_seed_counts_before_repair", {}) or {},
        "protected_group_seed_counts_after_repair": getattr(optimization_result, "protected_group_seed_counts_after_repair", {}) or {},
        "swap_attempts": int(getattr(optimization_result, "swap_attempts", 0)),
        "swap_accepted": int(getattr(optimization_result, "swap_accepted", 0)),
        "swap_rejected_fairness_degradation": int(getattr(optimization_result, "swap_rejected_fairness_degradation", 0)),
        "swap_accepted_fscore_improvement": int(getattr(optimization_result, "swap_accepted_fscore_improvement", 0)),
        "swap_accepted_mf_improvement": int(getattr(optimization_result, "swap_accepted_mf_improvement", 0)),
        "swap_accepted_spread_fairness_preserved": int(getattr(optimization_result, "swap_accepted_spread_fairness_preserved", 0)),
        "best_generation": best_generation,
        "final_fitness": float(getattr(optimization_result, "best_score", 0.0)),
        "seed_quota_per_protected_group": getattr(optimization_result, "seed_quota_per_protected_group", {}) or {},
        "initial_population_group_coverage_summary": getattr(optimization_result, "initial_population_group_coverage_summary", {}) or {},
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
    timing_fields: dict[str, object] = {
        "time_preprocessing": pd.NA,
        "time_community_detection": pd.NA,
        "time_embedding": pd.NA,
        "time_ris": pd.NA,
        "time_candidate_scoring": pd.NA,
    }
    embedding_config = dict(spec.embedding_config)
    if spec.embedding_method not in {"", "none", "off"}:
        embedding_config.setdefault("embedding_dim", int(config.embedding_dim))
        if spec.embedding_method in {"graphsage", "gcn", "dgi", "vgae", "graphcl"}:
            embedding_config.setdefault("hidden_dim", int(config.gnn_hidden_dim))
        if spec.embedding_method == "graphcl":
            embedding_config.setdefault("projection_dim", int(config.embedding_dim))

    phase_start = perf_counter()
    community_result = _community_result(dataset, spec, config)
    timing_fields["time_community_detection"] = float(perf_counter() - phase_start)
    phase_start = perf_counter()
    embedding_artifact = prepare_embedding_frame(
        dataset=dataset,
        method_name=spec.embedding_method,
        output_dir=config.output_dir,
        random_seed=int(config.random_seed),
        method_config=embedding_config,
        allow_cache=bool(config.use_embedding_cache),
    )
    timing_fields["time_embedding"] = float(perf_counter() - phase_start)
    clustering_artifact = prepare_optional_clustering(
        dataset=dataset,
        method_name=spec.clustering_method,
        input_mode=spec.clustering_input_mode,
        embeddings=embedding_artifact.embedding_frame,
        config=_clustering_config(spec, config),
        output_dir=config.output_dir,
        random_seed=int(config.random_seed),
    )
    phase_start = perf_counter()
    base_feature_frame = build_ranking_feature_frame(
        dataset=dataset,
        protected_group_report=protected_group_report,
        community_result=community_result,
        embedding_frame=embedding_artifact.embedding_frame,
        clustering_result=clustering_artifact.clustering_result,
    )
    timing_fields["time_preprocessing"] = float(perf_counter() - phase_start)
    ris_artifact = None
    ris_cache_status = "disabled"
    effective_rr_sets = _effective_ris_rr_sets(dataset, config, protected_group_report)
    if _spec_uses_ris(spec):
        phase_start = perf_counter()
        _emit_ris_rr_set_adjustment_warning(spec=spec, config=config, effective_rr_sets=effective_rr_sets)
        cache_path = _ris_cache_path(
            config.output_dir,
            dataset.name,
            protected_group_report.protected_attribute,
            spec.name,
            effective_rr_sets,
            _ris_config_mode(config),
        )
        if bool(config.ris_cache and config.ris_reuse_rr_sets) and cache_path is not None and cache_path.exists():
            try:
                ris_artifact = pd.read_pickle(cache_path)
                ris_cache_status = "hit"
            except Exception:
                ris_artifact = None
                ris_cache_status = "miss"
        if ris_artifact is None:
            ris_cache_status = "miss" if bool(config.ris_cache and config.ris_reuse_rr_sets) else "disabled"
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
                    mode=_ris_config_mode(config),
                    reuse_rr_sets=bool(config.ris_reuse_rr_sets),
                ),
            )
            if bool(config.ris_cache and config.ris_reuse_rr_sets) and cache_path is not None:
                try:
                    pd.to_pickle(ris_artifact, cache_path)
                except Exception:
                    ris_cache_status = "miss_unwritten"
        timing_fields["time_ris"] = float(perf_counter() - phase_start)
    phase_start = perf_counter()
    feature_frame = build_ranking_feature_frame(
        dataset=dataset,
        protected_group_report=protected_group_report,
        community_result=community_result,
        embedding_frame=embedding_artifact.embedding_frame,
        clustering_result=clustering_artifact.clustering_result,
        ris_scores=None if ris_artifact is None else ris_artifact.global_scores,
        fair_ris_scores=(
            None
            if ris_artifact is None or not _fair_ris_enabled(spec)
            else ris_artifact.fair_scores
        ),
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
    score_weights = _resolved_candidate_score_weights(spec, config, protected_group_report, dataset)
    score_frame = _combined_guidance_score_frame(
        spec,
        ranking_artifact.training_result.predicted_scores,
        None if ris_artifact is None else ris_artifact.global_scores,
        None if ris_artifact is None or not _fair_ris_enabled(spec) else ris_artifact.fair_scores,
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
        protected_group_report=protected_group_report,
        score_normalization=_effective_score_normalization(dataset, protected_group_report, config),
    )
    ris_verification_fields = _verify_ris_score_table(
        spec=spec,
        config=config,
        score_frame=score_frame,
    )
    combined_scores = {
        row.node_id: float(row.combined_score)
        for row in score_frame[["node_id", "combined_score"]].itertuples(index=False)
    }
    candidate_nodes = list(ranking_artifact.training_result.candidate_nodes)
    candidate_pool_diagnostics: dict[str, object]
    if _effective_group_stratified_candidate_pool(dataset, protected_group_report, config):
        candidate_nodes, candidate_pool_diagnostics = _group_stratified_candidate_pool(
            score_frame=score_frame,
            protected_group_report=protected_group_report,
            config=config,
            target_size=_effective_candidate_pool_size(dataset, config),
        )
    else:
        candidate_pool_diagnostics = {
            "group_stratified_candidate_pool": False,
            "candidate_pool_group_counts": _candidate_pool_group_counts(candidate_nodes, protected_group_report),
            "candidate_pool_group_quota": pd.NA,
            "candidate_pool_group_quota_shortfall": {},
        }
    score_table_path = _write_combined_score_table(
        dataset=dataset,
        protected_group_report=protected_group_report,
        spec=spec,
        config=config,
        score_frame=score_frame,
    )
    timing_fields["time_candidate_scoring"] = float(perf_counter() - phase_start)
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
        "candidate_nodes": tuple(candidate_nodes),
        "candidate_pool_size": len(candidate_nodes),
        "candidate_pool_diagnostics": candidate_pool_diagnostics,
        "diagnostics_fields": diagnostics_fields,
        "ris_verification_fields": ris_verification_fields,
        "timing_fields": timing_fields,
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
        f"candidate_pool={shared.get('candidate_pool_size', len(ranking_artifact.training_result.candidate_nodes))}",
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
    candidate_pool_diagnostics = dict(shared.get("candidate_pool_diagnostics", {}) or {})
    ris_verification_fields = dict(shared.get("ris_verification_fields", {}) or {})
    timing_fields = dict(shared.get("timing_fields", {}) or {})
    return {
        "candidate_pool_size": shared.get("candidate_pool_size", pd.NA),
        "candidate_pool_group_counts": candidate_pool_diagnostics.get("candidate_pool_group_counts", pd.NA),
        "candidate_pool_group_quota": candidate_pool_diagnostics.get("candidate_pool_group_quota", pd.NA),
        "candidate_pool_group_quota_shortfall": candidate_pool_diagnostics.get("candidate_pool_group_quota_shortfall", pd.NA),
        "group_stratified_candidate_pool": candidate_pool_diagnostics.get("group_stratified_candidate_pool", pd.NA),
        "effective_ris_num_rr_sets": shared.get("effective_ris_num_rr_sets", pd.NA),
        "num_communities": getattr(quality_metrics, "num_communities", pd.NA),
        "community_modularity": getattr(quality_metrics, "modularity", pd.NA),
        "embedding_cache_status": shared.get("embedding_cache_status", ""),
        "community_cache_status": shared.get("community_cache_status", ""),
        "ris_cache_status": shared.get("ris_cache_status", ""),
        "score_cache_status": shared.get("score_cache_status", ""),
        **timing_fields,
        **ris_verification_fields,
    }


def _run_community_aware_fair_greedy(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
) -> pd.DataFrame:
    start = perf_counter()
    phase_start = perf_counter()
    community_result = _community_result(dataset, spec, config)
    time_community_detection = float(perf_counter() - phase_start)
    scoring_start = perf_counter()
    time_ris: object = pd.NA
    greedy_candidate_nodes = None
    greedy_candidate_scores = None
    score_frame = None
    score_table_path = ""
    ris_verification_fields: dict[str, object] = {}
    ris_note = ""
    candidate_pool_note = ""
    if bool(config.use_ris_greedy_approximation) or _spec_uses_ris(spec):
        feature_frame = build_ranking_feature_frame(
            dataset=dataset,
            protected_group_report=protected_group_report,
            community_result=community_result,
            use_community_features_for_ml=bool(config.use_community_features_for_ml),
            community_feature_mode=str(config.community_feature_mode),
        )
        effective_rr_sets = _effective_ris_rr_sets(dataset, config, protected_group_report)
        _emit_ris_rr_set_adjustment_warning(spec=spec, config=config, effective_rr_sets=effective_rr_sets)
        ris_artifact = prepare_ris_guidance(
            dataset=dataset,
            protected_group_report=protected_group_report,
            propagation_probability=float(config.propagation_probability),
            feature_frame=feature_frame,
            output_dir=config.output_dir,
            protected_attribute=protected_group_report.protected_attribute,
            stack_name=spec.name,
            config=RISConfig(
                num_rr_sets=effective_rr_sets,
                random_seed=int(config.random_seed),
                mode=_ris_config_mode(config),
                reuse_rr_sets=bool(config.ris_reuse_rr_sets),
            ),
        )
        time_ris = float(getattr(ris_artifact.ris_result, "runtime_seconds", perf_counter() - scoring_start))
        score_weights = _resolved_candidate_score_weights(spec, config, protected_group_report, dataset)
        structural_scores = compute_structural_node_scores(feature_frame)
        fairness_bonus_scores = _feature_score_map(feature_frame, "fraction_neighbors_in_undercovered_groups")
        diversity_bonus_scores = _feature_score_map(feature_frame, "neighboring_communities")
        protected_group_coverage_scores = _protected_group_coverage_bonus_scores(feature_frame, protected_group_report)
        score_frame = _combined_guidance_score_frame(
            spec,
            structural_scores,
            ris_artifact.global_scores,
            ris_artifact.fair_scores if _fair_ris_enabled(spec) else None,
            ml_score_weight=0.0,
            ris_score_weight=score_weights["ris_score_weight"],
            fair_ris_score_weight=score_weights["fair_ris_score_weight"],
            fairness_bonus_scores=fairness_bonus_scores,
            weak_group_bonus_scores=fairness_bonus_scores,
            diversity_bonus_scores=diversity_bonus_scores,
            protected_group_coverage_scores=protected_group_coverage_scores,
            spread_proxy_scores=structural_scores,
            fairness_bonus_weight=float(config.fairness_bonus_weight),
            weak_group_bonus_weight=score_weights["weak_group_bonus_weight"],
            diversity_bonus_weight=score_weights["community_diversity_weight"],
            protected_group_coverage_weight=score_weights["protected_group_coverage_weight"],
            spread_proxy_weight=score_weights["spread_proxy_weight"],
            protected_group_report=protected_group_report,
            score_normalization=_effective_score_normalization(dataset, protected_group_report, config),
        )
        greedy_candidate_scores = {
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
        ris_verification_fields = _verify_ris_score_table(
            spec=spec,
            config=config,
            score_frame=score_frame,
        )
        if _effective_group_stratified_candidate_pool(dataset, protected_group_report, config):
            greedy_candidate_nodes, _ = _group_stratified_candidate_pool(
                score_frame=score_frame,
                protected_group_report=protected_group_report,
                config=config,
                target_size=_effective_candidate_pool_size(dataset, config),
            )
        else:
            greedy_candidate_nodes = sorted(
                greedy_candidate_scores,
                key=lambda node_id: (-float(greedy_candidate_scores[node_id]), _sort_key(node_id)),
            )[: _effective_candidate_pool_size(dataset, config)]
        ris_note = f"; ris_greedy_approximation=true; ris_rr_sets={effective_rr_sets}"
    elif bool(config.require_ris):
        ris_verification_fields = _verify_ris_score_table(
            spec=spec,
            config=config,
            score_frame=score_frame,
        )
    elif str(config.scalability_mode) != "off":
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
        score_weights = _resolved_candidate_score_weights(spec, config, protected_group_report, dataset)
        score_frame = _combined_guidance_score_frame(
            spec,
            structural_scores,
            None,
            None,
            ml_score_weight=score_weights["ml_score_weight"],
            fairness_bonus_scores=fairness_bonus_scores,
            weak_group_bonus_scores=fairness_bonus_scores,
            diversity_bonus_scores=diversity_bonus_scores,
            protected_group_coverage_scores=protected_group_coverage_scores,
            spread_proxy_scores=structural_scores,
            fairness_bonus_weight=float(config.fairness_bonus_weight),
            weak_group_bonus_weight=score_weights["weak_group_bonus_weight"],
            diversity_bonus_weight=score_weights["community_diversity_weight"],
            protected_group_coverage_weight=score_weights["protected_group_coverage_weight"],
            spread_proxy_weight=score_weights["spread_proxy_weight"],
            protected_group_report=protected_group_report,
            score_normalization=_effective_score_normalization(dataset, protected_group_report, config),
        )
        greedy_candidate_scores = {
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
        target_size = _effective_candidate_pool_size(dataset, config)
        if _effective_group_stratified_candidate_pool(dataset, protected_group_report, config):
            greedy_candidate_nodes, _ = _group_stratified_candidate_pool(
                score_frame=score_frame,
                protected_group_report=protected_group_report,
                config=config,
                target_size=target_size,
            )
        else:
            greedy_candidate_nodes = sorted(
                greedy_candidate_scores,
                key=lambda node_id: (-float(greedy_candidate_scores[node_id]), _sort_key(node_id)),
            )[:target_size]
        candidate_pool_note = f"; baseline_candidate_pool={len(greedy_candidate_nodes)}"
    time_candidate_scoring = float(perf_counter() - scoring_start) if score_frame is not None else pd.NA
    diagnostics_fields = _emit_stack_diagnostics(
        dataset=dataset,
        protected_group_report=protected_group_report,
        spec=spec,
        config=config,
        community_result=community_result,
        score_frame=score_frame,
        score_table_path=score_table_path,
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
        candidate_nodes=greedy_candidate_nodes,
        candidate_scores=greedy_candidate_scores,
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
    objective_mode = (
        PROFESSOR_PRIORITY
        if _fairness_first_enabled(spec, config) or _normalize_ranking_policy(config.fair_greedy_objective) == PROFESSOR_PRIORITY
        else "f_score"
    )
    chosen_start = (
        greedy_seed_set
        if _rank_search_evaluation(objective_mode, greedy_eval) >= _rank_search_evaluation(objective_mode, initial_eval)
        else initial_seed_set
    )
    local_search_start = perf_counter()
    local_search_start = perf_counter()
    refined_seed_set, _ = _swap_local_search(
        dataset=dataset,
        protected_group_report=protected_group_report,
        initial_seed_set=chosen_start,
        diffusion_model=spec.diffusion_model,
        config=config,
        objective_mode=objective_mode,
    )
    time_local_search = float(perf_counter() - local_search_start)
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
        f"objective={objective_mode}; communities={quality.num_communities}; modularity={quality.modularity:.6f}"
    )
    if ris_note:
        notes = f"{notes}{ris_note}; final_eval=monte_carlo"
    if candidate_pool_note:
        notes = f"{notes}{candidate_pool_note}"
    baseline_candidate_pool_size = (
        len(greedy_candidate_nodes)
        if greedy_candidate_nodes is not None
        else dataset.graph.number_of_nodes()
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
                    "candidate_pool_size": baseline_candidate_pool_size,
                    "num_communities": quality.num_communities,
                    "community_modularity": quality.modularity,
                    "score_table_path": score_table_path,
                    "community_cache_status": community_result.metadata.get("community_cache_status", "disabled"),
                    "time_community_detection": time_community_detection,
                    "time_ris": time_ris,
                    "time_candidate_scoring": time_candidate_scoring,
                    "time_local_search": time_local_search,
                    "time_optimizer": float(max(0.0, search_runtime - time_community_detection - (0.0 if pd.isna(time_candidate_scoring) else float(time_candidate_scoring)) - time_local_search)),
                    "community_assignments_path": community_paths["community_assignments_path"],
                    "community_sizes_path": community_paths["community_sizes_path"],
                    **ris_verification_fields,
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
        ris_num_rr_sets=int(_effective_ris_rr_sets(dataset, config, protected_group_report)),
        ris_random_seed=int(config.random_seed),
        ris_mode=_ris_config_mode(config),
        ris_reuse_rr_sets=bool(config.ris_reuse_rr_sets),
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
    if bool(config.require_ris):
        raise RuntimeError(
            f"RIS verification failed for stack {spec.name}: delegated GNN/RIS runner does not expose "
            "a combined candidate score table with ris_score and fair_ris_score columns."
        )
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
            "search_estimator": _canonical_search_estimator(spec.spread_estimator_search),
            "final_estimator": spec.spread_estimator_final,
            "spread_estimator_search": _canonical_search_estimator(spec.spread_estimator_search),
            "spread_estimator_final": spec.spread_estimator_final,
            "search_spread_estimator": "ris_guidance",
            "search_guidance_estimator": "fair_ris_guidance",
            "final_spread_estimator": spec.spread_estimator_final,
            "scalability_mode": str(config.scalability_mode),
            "use_ris": _spec_uses_ris(spec),
            "use_fair_ris": bool(spec.use_fair_ris),
            "fair_ris_enabled": _fair_ris_enabled(spec),
            "ris_mode": _reported_ris_mode(config),
            "ris_num_rr_sets": int(config.ris_num_rr_sets),
            "effective_ris_num_rr_sets": int(_effective_ris_rr_sets(dataset, config, protected_group_report)),
            "ris_reuse_rr_sets": bool(config.ris_reuse_rr_sets),
            "ris_score_nonzero_count": pd.NA,
            "ris_score_std": pd.NA,
            "fair_ris_score_nonzero_count": pd.NA,
            "fair_ris_score_std": pd.NA,
            "ris_verified": bool(_spec_uses_ris(spec)),
            "fair_ris_verified": bool(_fair_ris_enabled(spec)),
            "ris_active_verified": bool(_spec_uses_ris(spec)),
            "fair_ris_active_verified": bool(_fair_ris_enabled(spec)),
            "ris_verification_warnings": "Delegated runner does not expose combined candidate score table.",
            "force_ris_for_all_stacks": bool(config.force_ris_for_all_stacks),
            "require_ris": bool(config.require_ris),
            "candidate_pool_size": pd.NA,
            "scalability_pass": True,
            "key_enabled_modules": _key_enabled_modules(spec, "none"),
            **_algorithm_composition_fields(spec, config, dataset, protected_group_report, "none"),
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
        candidate_nodes=shared.get("candidate_nodes", shared["ranking_artifact"].training_result.candidate_nodes),
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
        candidate_nodes=shared.get("candidate_nodes", shared["ranking_artifact"].training_result.candidate_nodes),
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
    time_local_search = float(perf_counter() - local_search_start)
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
                    "time_local_search": time_local_search,
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


def _hybrid_optimizer_config(
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
    protected_group_report: ProtectedGroupReport | None = None,
    dataset: LoadedDataset | None = None,
) -> HybridSIEAConfig:
    use_ml_initialization = True if config.use_ml_scores_in_initialization is None else bool(config.use_ml_scores_in_initialization)
    use_ml_mutation = True if config.use_ml_scores_in_mutation is None else bool(config.use_ml_scores_in_mutation)
    use_ml_crossover = True if config.use_ml_scores_in_crossover is None else bool(config.use_ml_scores_in_crossover)
    use_ml_repair = True if config.use_ml_scores_in_repair is None else bool(config.use_ml_scores_in_repair)
    use_ml_local_search = True if config.use_ml_scores_in_local_search is None else bool(config.use_ml_scores_in_local_search)
    protected_balance = bool(config.protected_group_balance_enabled)
    fairness_first = _fairness_first_enabled(spec, config)
    large_imbalance_active = _large_imbalance_fairness_active(dataset, protected_group_report, config)
    fair_guidance = bool((spec.use_fair_ris or fairness_first or large_imbalance_active) and protected_balance)
    score_weights = _resolved_candidate_score_weights(spec, config, protected_group_report, dataset)
    return HybridSIEAConfig(
        budget=int(config.budget),
        population_size=int(config.population_size),
        generations=int(config.generations),
        propagation_probability=float(config.propagation_probability),
        mc_runs=int(config.mc_runs_search),
        diffusion_model=spec.diffusion_model,
        lambda_weight=float(config.lambda_weight),
        random_seed=int(config.random_seed),
        fitness_policy="professor_priority" if fairness_first or large_imbalance_active or bool(config.use_professor_priority_fitness) else "f_score",
        fscore_weight=float(config.fscore_weight),
        mf_weight=float(config.mf_weight),
        dcv_weight=max(float(config.dcv_weight), float(score_weights["dcv_penalty_weight"])),
        group_coverage_weight=float(config.group_coverage_weight),
        scalability_weight=float(config.scalability_weight),
        spread_weight=float(config.spread_weight),
        runtime_penalty_weight=float(config.runtime_penalty_weight),
        runtime_weight=float(config.runtime_weight),
        fairness_tolerance_dcv=float(config.fairness_tolerance_dcv),
        fairness_tolerance_fscore_drop=float(config.fairness_tolerance_fscore_drop),
        use_fairness_first_swap_acceptance=bool(config.use_fairness_first_swap_acceptance or fairness_first or large_imbalance_active),
        swap_reject_spread_gain_if_fairness_collapses=bool(config.swap_reject_spread_gain_if_fairness_collapses),
        use_group_quota_repair=bool(config.use_group_quota_repair),
        min_seeds_per_protected_group=int(config.min_seeds_per_protected_group),
        quota_mode=str(config.quota_mode),
        quota_min_group_support=int(config.quota_min_group_support),
        use_protected_group_quota_initialization=bool(_effective_quota_initialization(dataset, protected_group_report, config)) if dataset is not None and protected_group_report is not None else bool(config.use_protected_group_quota_initialization),
        small_group_seed_fraction=float(config.small_group_seed_fraction),
        initialization_quota_mode=str(config.initialization_quota_mode),
        use_fairness_first_repair=bool(config.use_fairness_first_repair),
        weak_group_repair_rounds=int(config.weak_group_repair_rounds),
        majority_overconcentration_threshold=float(config.majority_overconcentration_threshold),
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
        fairness_first_init_slots=min(3 if fairness_first or large_imbalance_active else 2, int(config.budget)) if fair_guidance else 0,
        fairness_first_init_weight=1.0 if (fairness_first or large_imbalance_active) and fair_guidance else (0.75 if fair_guidance else 0.0),
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
        config=_hybrid_optimizer_config(spec, config, protected_group_report, dataset),
        candidate_nodes=shared.get("candidate_nodes", shared["ranking_artifact"].training_result.candidate_nodes),
        node_scores=structural_scores,
        ml_node_scores=shared["combined_guidance_scores"],
    )
    optimizer_start = perf_counter()
    optimization_result = optimizer.optimize()
    time_optimizer = float(perf_counter() - optimizer_start)
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
                    "time_optimizer": time_optimizer,
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
    score_weights = _resolved_candidate_score_weights(spec, config, protected_group_report, dataset)
    ris_artifact = None
    effective_rr_sets = _effective_ris_rr_sets(dataset, config, protected_group_report)
    if _spec_uses_ris(spec):
        _emit_ris_rr_set_adjustment_warning(spec=spec, config=config, effective_rr_sets=effective_rr_sets)
        ris_artifact = prepare_ris_guidance(
            dataset=dataset,
            protected_group_report=protected_group_report,
            propagation_probability=float(config.propagation_probability),
            feature_frame=feature_frame,
            output_dir=config.output_dir,
            protected_attribute=protected_group_report.protected_attribute,
            stack_name=spec.name,
            config=RISConfig(
                num_rr_sets=effective_rr_sets,
                random_seed=int(config.random_seed),
                mode=_ris_config_mode(config),
                reuse_rr_sets=bool(config.ris_reuse_rr_sets),
            ),
        )
    score_frame = _combined_guidance_score_frame(
        spec,
        structural_scores,
        None if ris_artifact is None else ris_artifact.global_scores,
        None if ris_artifact is None or not _fair_ris_enabled(spec) else ris_artifact.fair_scores,
        ml_score_weight=score_weights["ml_score_weight"],
        ris_score_weight=score_weights["ris_score_weight"],
        fair_ris_score_weight=score_weights["fair_ris_score_weight"],
        fairness_bonus_scores=fairness_bonus_scores,
        weak_group_bonus_scores=fairness_bonus_scores,
        diversity_bonus_scores=diversity_bonus_scores,
        protected_group_coverage_scores=protected_group_coverage_scores,
        spread_proxy_scores=structural_scores,
        fairness_bonus_weight=float(config.fairness_bonus_weight),
        weak_group_bonus_weight=score_weights["weak_group_bonus_weight"],
        diversity_bonus_weight=score_weights["community_diversity_weight"],
        protected_group_coverage_weight=score_weights["protected_group_coverage_weight"],
        spread_proxy_weight=score_weights["spread_proxy_weight"],
        protected_group_report=protected_group_report,
        score_normalization=_effective_score_normalization(dataset, protected_group_report, config),
    )
    ris_verification_fields = _verify_ris_score_table(
        spec=spec,
        config=config,
        score_frame=score_frame,
    )
    combined_scores = {
        row.node_id: float(row.combined_score)
        for row in score_frame[["node_id", "combined_score"]].itertuples(index=False)
    }
    candidate_nodes: Sequence[Any] | None = None
    candidate_pool_diagnostics: dict[str, object] = {
        "group_stratified_candidate_pool": False,
        "candidate_pool_group_counts": _candidate_pool_group_counts(list(combined_scores), protected_group_report),
        "candidate_pool_group_quota": pd.NA,
        "candidate_pool_group_quota_shortfall": {},
    }
    if _effective_group_stratified_candidate_pool(dataset, protected_group_report, config):
        candidate_nodes, candidate_pool_diagnostics = _group_stratified_candidate_pool(
            score_frame=score_frame,
            protected_group_report=protected_group_report,
            config=config,
            target_size=_effective_candidate_pool_size(dataset, config),
        )
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
        config=_hybrid_optimizer_config(spec, config, protected_group_report, dataset),
        candidate_nodes=candidate_nodes,
        node_scores=structural_scores,
        ml_node_scores=combined_scores,
    )
    optimizer_start = perf_counter()
    optimization_result = optimizer.optimize()
    time_optimizer = float(perf_counter() - optimizer_start)
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
                    "candidate_pool_size": len(candidate_nodes) if candidate_nodes is not None else dataset.graph.number_of_nodes(),
                    "candidate_pool_group_counts": candidate_pool_diagnostics.get("candidate_pool_group_counts", pd.NA),
                    "candidate_pool_group_quota": candidate_pool_diagnostics.get("candidate_pool_group_quota", pd.NA),
                    "candidate_pool_group_quota_shortfall": candidate_pool_diagnostics.get("candidate_pool_group_quota_shortfall", pd.NA),
                    "group_stratified_candidate_pool": candidate_pool_diagnostics.get("group_stratified_candidate_pool", pd.NA),
                    "effective_ris_num_rr_sets": effective_rr_sets,
                    **_optimizer_diagnostics_fields(
                        optimization_result,
                        protected_group_report,
                        community_result,
                    ),
                    "time_optimizer": time_optimizer,
                    "num_communities": quality.num_communities,
                    "community_modularity": quality.modularity,
                    "community_cache_status": community_result.metadata.get("community_cache_status", "disabled"),
                    "score_table_path": score_table_path,
                    **ris_verification_fields,
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
    return _policy_fairness_valid_mask(frame, professor_priority_config_from_object(config))


def _fairness_priority_order(frame: pd.DataFrame, *, config: FIMPermutationRunConfig | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return rank_frame_professor_priority(
        frame,
        professor_priority_config_from_object(config),
        gate_validity_enabled=True,
    )


def _recommend_stack_name(frame: pd.DataFrame, mask: pd.Series, *, config: FIMPermutationRunConfig | None = None) -> str | None:
    candidates = frame.loc[mask].copy()
    if candidates.empty:
        return None
    if config is not None and _normalize_ranking_policy(config.ranking_policy) == PROFESSOR_PRIORITY:
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
    strict_fairness_first = config is not None and _normalize_ranking_policy(config.ranking_policy) == PROFESSOR_PRIORITY
    no_fairness_gate_pass = strict_fairness_first and not bool(valid_fairness_mask.any())
    if not bool(valid_fairness_mask.any()) and not strict_fairness_first:
        valid_fairness_mask = ok_mask
    ranking_fairness_mask = valid_fairness_mask
    if strict_fairness_first and not bool(valid_fairness_mask.any()) and bool(ok_mask.any()):
        ranking_fairness_mask = ok_mask
    if strict_fairness_first:
        interpretable_mask &= ranking_fairness_mask
        ml_mask &= ranking_fairness_mask
        fairness_mask &= ranking_fairness_mask
    scalability_values = frame["scalability_pass"] if "scalability_pass" in frame.columns else pd.Series([True] * len(frame), index=frame.index)
    scalability_mask = scalability_values.fillna(True).map(
        lambda value: str(value).strip().lower() not in {"false", "0", "no"}
    )
    scalable_mask = ranking_fairness_mask & scalability_mask
    spread_candidates = frame.loc[ranking_fairness_mask].copy()
    runtime_candidates = _fairness_priority_order(frame.loc[ranking_fairness_mask].copy(), config=config) if bool(ranking_fairness_mask.any()) else pd.DataFrame()
    runtime_close_mask = ranking_fairness_mask
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
    final_professor = _recommend_stack_name(frame, ranking_fairness_mask, config=config)
    return {
        "best_interpretable_baseline": None if no_fairness_gate_pass else _recommend_stack_name(frame, interpretable_mask, config=config),
        "best_practical_ml_default": None if no_fairness_gate_pass else _recommend_stack_name(frame, ml_mask, config=config),
        "best_exploratory_fairness_heavy": None if no_fairness_gate_pass else _recommend_stack_name(frame, fairness_mask, config=config),
        "best_fairness_quality_method": final_professor,
        "best_scalable_method": None if no_fairness_gate_pass else _recommend_stack_name(frame, scalable_mask, config=config),
        "best_spread_method": None if no_fairness_gate_pass else best_spread_method,
        "best_runtime_method": None if no_fairness_gate_pass else best_runtime_method,
        "final_professor_priority_recommendation": final_professor,
    }


def _mapping_from_cell(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if pd.isna(value):
        return {}
    text = str(value).strip()
    if not text or text == "<NA>":
        return {}
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001 - report diagnostics should be best effort.
        return {}
    if isinstance(parsed, Mapping):
        return {str(key): item for key, item in parsed.items()}
    return {}


def _format_no_pass_fairness_diagnostics(frame: pd.DataFrame, config: FIMPermutationRunConfig) -> list[str]:
    ok_rows = frame[frame["status"].astype(str).str.lower().eq("ok")].copy()
    if ok_rows.empty:
        return []

    first_row = ok_rows.iloc[0]
    protected_group_counts = _mapping_from_cell(first_row.get("protected_group_counts", {}))
    budget_estimate = _mapping_from_cell(first_row.get("budget_per_group_estimate", {}))
    imbalance_ratio = pd.to_numeric(
        pd.Series([first_row.get("group_imbalance_ratio", pd.NA)]),
        errors="coerce",
    ).iloc[0]
    if pd.isna(imbalance_ratio) and protected_group_counts:
        sizes = [float(value) for value in protected_group_counts.values() if pd.notna(value) and float(value) > 0.0]
        imbalance_ratio = max(sizes) / min(sizes) if sizes else pd.NA

    lines = [
        "Fairness failure diagnostics",
        "-" * 72,
        "No method passed the professor-priority fairness gates.",
    ]
    if protected_group_counts:
        lines.append(f"protected_group_counts={json.dumps(protected_group_counts, sort_keys=True)}")
    if pd.notna(imbalance_ratio):
        lines.append(f"group_imbalance_ratio={float(imbalance_ratio):.4f}")
    if budget_estimate:
        lines.append(f"budget_per_group_estimate={json.dumps(budget_estimate, sort_keys=True)}")

    for _, row in ok_rows.iterrows():
        failures = str(row.get("fairness_gate_failures", "")).strip() or "unknown"
        influence = _mapping_from_cell(row.get("group_influence_distribution", {}))
        normalized_influence = _mapping_from_cell(row.get("normalized_group_influence_distribution", {}))
        zero_groups = [
            group_name
            for group_name, value in influence.items()
            if pd.notna(value) and float(value) <= 1e-12
        ]
        lines.append(
            "method_failure="
            f"{row.get('stack_name')}: reasons={failures}; "
            f"F-score={float(row.get('f_score')):.4f}; "
            f"MF={float(row.get('mf')):.4f}; "
            f"DCV={float(row.get('dcv')):.4f}; "
            f"scalability_pass={row.get('scalability_pass')}; "
            f"zero_covered_groups={row.get('zero_covered_groups_count')}"
        )
        if zero_groups:
            lines.append(f"zero_covered_group_names[{row.get('stack_name')}]={json.dumps(zero_groups, sort_keys=True)}")
        if influence:
            lines.append(f"group_influence_distribution[{row.get('stack_name')}]={json.dumps(influence, sort_keys=True)}")
        if normalized_influence:
            lines.append(f"normalized_group_influence_distribution[{row.get('stack_name')}]={json.dumps(normalized_influence, sort_keys=True)}")

    lines.extend(
        [
            "Recommended actions:",
            "- increase budget",
            "- increase weak_group_bonus_weight",
            "- increase fair_ris_score_weight",
            "- enable group quota repair",
            "- use support-aware fairness metrics",
            "- run budget sweep",
        ]
    )
    return lines


def format_fim_permutation_report(frame: pd.DataFrame, config: FIMPermutationRunConfig) -> str:
    """Format a compact terminal/report comparison table."""

    lines = [
        "FIM Algorithm-Stack Permutation Comparison",
        "-" * 72,
        f"Protected attribute: {config.protected_attribute}",
        f"Budget: {config.budget}",
        f"Ranking policy: {_normalize_ranking_policy(config.ranking_policy)}",
        (
            "Fairness gates: "
            f"min_f_score={config.min_f_score}, min_mf={config.min_mf}, max_dcv={config.max_dcv}, "
            f"min_fraction_groups_covered={config.min_fraction_groups_covered}, "
            f"scalability_required={config.scalability_required}, warn_only={config.warn_only_fairness_gates}"
        ),
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
    policy_warning = professor_priority_warning(ok_slice, professor_priority_config_from_object(config))
    if policy_warning:
        lines.extend([policy_warning, ""])
        lines.extend(_format_no_pass_fairness_diagnostics(sortable, config))
        lines.append("")
    budget_warnings = sorted(
        {
            str(value)
            for value in sortable.get("budget_adequacy_warning", pd.Series(dtype=object)).dropna().tolist()
            if str(value).strip()
        }
    )
    if budget_warnings:
        lines.extend(budget_warnings)
        lines.append("")
    if _normalize_ranking_policy(config.ranking_policy) == PROFESSOR_PRIORITY:
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
            lines.append(
                "   "
                f"fairness_diagnostics: seed_group_counts={row.get('final_protected_group_coverage')} | "
                f"influence={row.get('group_influence_distribution')} | "
                f"normalized_influence={row.get('normalized_group_influence_distribution')} | "
                f"weakest={row.get('weakest_protected_group')} | "
                f"strongest={row.get('strongest_protected_group')}"
            )
            lines.append(
                "   "
                f"runtime_breakdown: dataset={row.get('time_dataset_loading')} | "
                f"community={row.get('time_community_detection')} | "
                f"embedding={row.get('time_embedding')} | "
                f"ris={row.get('time_ris')} | "
                f"candidate_scoring={row.get('time_candidate_scoring')} | "
                f"optimizer={row.get('time_optimizer')} | "
                f"local_search={row.get('time_local_search')} | "
                f"final_mc={row.get('time_final_mc')}"
            )
            lines.append(
                "   "
                f"seed_diagnostics: seed_count={row.get('seed_count')}/{row.get('expected_seed_count')} | "
                f"duplicates={row.get('duplicate_seed_count')} | "
                f"seed_groups={row.get('seed_group_counts')} | "
                f"seed_communities={row.get('seed_community_counts')} | "
                f"communities_covered={row.get('seed_communities_covered')}"
            )
            lines.append(
                "   "
                f"score_diagnostics: component_stats={row.get('score_component_stats_json')} | "
                f"constant_components={row.get('constant_score_components')}"
            )
            negative_reason_value = row.get("negative_f_score_reason", "")
            negative_reason = "" if pd.isna(negative_reason_value) else str(negative_reason_value).strip()
            if negative_reason and negative_reason != "<NA>":
                lines.append(f"   negative_f_score_reason={negative_reason}")
        else:
            lines.append(f"   skip_reason={row.get('skip_reason', '')}")
        lines.append(
            "   "
            f"modules: diffusion={row.get('diffusion_model')} | "
            f"search_estimator={row.get('search_estimator', row.get('spread_estimator_search'))} | "
            f"final_estimator={row.get('final_estimator', row.get('spread_estimator_final'))} | "
            f"fair_ris={row.get('fair_ris_enabled')} | "
            f"ris_rr_sets={row.get('effective_ris_num_rr_sets')} | "
            f"community={row.get('community_method')} | "
            f"embedding={row.get('embedding_method')} | "
            f"clustering={row.get('clustering_method')} | "
            f"ranking={row.get('ranking_model')} | "
            f"optimizer={row.get('optimizer_mode')} | "
            f"debias={row.get('debias_mode')}"
        )
        lines.append(
            "   "
            f"algorithms: embedding={row.get('graph_embedding_algorithm')} | "
            f"ml_ranking={row.get('ml_ranking_algorithm')} | "
            f"community={row.get('community_detection_algorithm')} | "
            f"clustering={row.get('clustering_algorithm')} | "
            f"diffusion={row.get('diffusion_model_name')} | "
            f"search={row.get('search_time_spread_estimator')} | "
            f"optimizer={row.get('fair_influence_optimizer')} | "
            f"repair={row.get('repair_strategy')} | "
            f"local_refinement={row.get('local_refinement')} | "
            f"final={row.get('final_evaluator')} | "
            f"ranking_policy={row.get('ranking_policy_name')}"
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
                f"ris_verified={row.get('ris_verified', row.get('ris_active_verified'))} | "
                f"fair_ris_verified={row.get('fair_ris_verified', row.get('fair_ris_active_verified'))} | "
                f"ris_score_nonzero={row.get('ris_score_nonzero_count')} | "
                f"fair_ris_score_nonzero={row.get('fair_ris_score_nonzero_count')} | "
                f"weak_group={row.get('weak_group_bonus_weight')} | "
                f"community_diversity={row.get('community_diversity_weight')} | "
                f"protected_group_coverage={row.get('protected_group_coverage_weight')} | "
                f"spread_proxy={row.get('spread_proxy_weight')} | "
                f"large_imbalance_active={row.get('large_imbalance_fairness_active')} | "
                f"score_normalization={row.get('score_normalization')}"
            )
            lines.append(
                "   "
                f"RIS config: search_estimator={row.get('search_estimator', row.get('spread_estimator_search'))} | "
                f"final_estimator={row.get('final_estimator', row.get('spread_estimator_final', 'monte_carlo'))} | "
                f"fair_ris={row.get('fair_ris_enabled')} | "
                f"ris_rr_sets={row.get('effective_ris_num_rr_sets')} | "
                f"ris_verified={row.get('ris_verified', row.get('ris_active_verified'))} | "
                f"fair_ris_verified={row.get('fair_ris_verified', row.get('fair_ris_active_verified'))} | "
                f"force_ris_for_all_stacks={row.get('force_ris_for_all_stacks')} | "
                f"require_ris={row.get('require_ris')}"
            )
            lines.append(
                "   "
                f"candidate_pool_diagnostics: stratified={row.get('group_stratified_candidate_pool')} | "
                f"group_counts={row.get('candidate_pool_group_counts')} | "
                f"group_quota={row.get('candidate_pool_group_quota')} | "
                f"quota_shortfall={row.get('candidate_pool_group_quota_shortfall')}"
            )
            lines.append(
                "   "
                f"optimizer_diagnostics: initial_seed_source={row.get('initial_seed_source')} | "
                f"repaired_seed_sets={row.get('repaired_seed_sets')} | "
                f"repair_attempts={row.get('repair_attempts')} | "
                f"weak_group_repairs={row.get('weak_group_repairs')} | "
                f"successful_swaps={row.get('successful_swaps')} | "
                f"swap_attempts={row.get('swap_attempts')} | "
                f"swap_accepted={row.get('swap_accepted')} | "
                f"swap_rejected_fairness={row.get('swap_rejected_fairness_degradation')} | "
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


def _enrich_frame_runtime_fields(
    frame: pd.DataFrame,
    *,
    dataset_loading_seconds: float | None,
    preprocessing_seconds: float | None,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        if _is_missing_value(row.get("time_dataset_loading")) and dataset_loading_seconds is not None:
            row["time_dataset_loading"] = float(dataset_loading_seconds)
        if _is_missing_value(row.get("time_preprocessing")) and preprocessing_seconds is not None:
            row["time_preprocessing"] = float(preprocessing_seconds)
        if str(row.get("status", "")).strip().lower() == "ok":
            row = _finalize_diagnostic_row(row)
        records.append(row)
    return pd.DataFrame(records)


def run_fim_permutation_benchmark(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    config: FIMPermutationRunConfig,
    permutations: Iterable[str | FIMPermutationSpec] | None = None,
    *,
    dataset_loading_seconds: float | None = None,
    preprocessing_seconds: float | None = None,
) -> FIMPermutationBenchmarkResult:
    """Run named FIM stacks with shared final Monte Carlo evaluation."""

    requested = tuple(permutations or available_fim_permutations())
    rows: list[dict[str, object]] = []
    results_by_permutation: dict[str, pd.DataFrame] = {}

    if bool(config.print_experiment_header):
        print_experiment_setup_header(
            dataset,
            protected_group_report,
            config,
            requested,
            dataset_loading_seconds=dataset_loading_seconds,
            preprocessing_seconds=preprocessing_seconds,
        )
    if bool(config.print_budget_check):
        print_budget_adequacy_check(dataset, protected_group_report, config)

    for requested_value in requested:
        spec = _resolve_fim_permutation_spec(requested_value)
        if spec.spread_estimator_final != "monte_carlo":
            raise ValueError("All permutation specs must use spread_estimator_final='monte_carlo'.")
        runner = _RUNNERS[spec.runner_kind]
        try:
            if bool(config.print_stack_summary):
                print_stack_algorithm_summary(
                    spec,
                    config,
                    dataset=dataset,
                    protected_group_report=protected_group_report,
                )
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
        frame = _enrich_frame_runtime_fields(
            frame,
            dataset_loading_seconds=dataset_loading_seconds,
            preprocessing_seconds=preprocessing_seconds,
        )
        print_stack_result_diagnostics(frame, config)
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

    load_start = perf_counter()
    dataset = load_dataset(dataset_config)
    dataset_loading_seconds = float(perf_counter() - load_start)
    node_count = dataset.graph.number_of_nodes()
    if int(config.budget) > int(node_count):
        raise ValueError(f"Budget {config.budget} exceeds graph node count {node_count} for dataset '{dataset.name}'.")
    preprocessing_start = perf_counter()
    protected_group_report = verify_protected_groups(dataset, config.protected_attribute)
    preprocessing_seconds = float(perf_counter() - preprocessing_start)
    return run_fim_permutation_benchmark(
        dataset=dataset,
        protected_group_report=protected_group_report,
        config=config,
        permutations=permutations,
        dataset_loading_seconds=dataset_loading_seconds,
        preprocessing_seconds=preprocessing_seconds,
    )
