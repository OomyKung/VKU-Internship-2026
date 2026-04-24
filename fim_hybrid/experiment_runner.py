"""Experiment runner for fair comparison across FIM methods."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import json
from time import perf_counter

import pandas as pd

from .baselines import select_baseline_seed_set
from .clustering import get_clustering_method_spec
from .community_detection import CommunityQualityMetrics, compute_community_quality_metrics, detect_communities
from .config import DatasetConfig
from .data_loader import LoadedDataset, ProtectedGroupReport, load_dataset, verify_protected_groups
from .diffusion import DEFAULT_DIFFUSION_MODEL, validate_diffusion_model
from .evaluation import SeedSetEvaluation, evaluate_seed_set
from .feature_extraction import compute_node_features
from .gnn_training import require_gnn_dependencies
from .hybrid_optimizer import HybridOptimizationResult, HybridSIEAConfig, HybridSIEAOptimizer
from .label_generation import NodeUtilityLabelResult, generate_singleton_node_utility_labels
from .ml_training import RankingTrainingResult
from .node2vec_embeddings import Node2VecConfig, build_node2vec_cache_path
from .ris_guidance import RISConfig, RISGuidanceResult
from .stack_pipeline import prepare_ris_guidance as prepare_stack_ris_guidance
from .stack_pipeline import train_ranking_model as train_stack_ranking_model

_KEPT_ML_VARIANT_LABEL = "hybrid_siea_ml_two_tier_tuned_swap_local_search"
_KEPT_GNN_ML_VARIANT_LABEL = "hybrid_siea_ml_gnn_two_tier_tuned_swap_local_search"
_KEPT_GNN_NODE2VEC_ML_VARIANT_LABEL = "hybrid_siea_ml_gnn_node2vec_two_tier_tuned_swap_local_search"
_KEPT_RIS_ML_VARIANT_LABEL = "hybrid_siea_ml_ris_two_tier_tuned_swap_local_search"
_KEPT_GNN_RIS_ML_VARIANT_LABEL = "hybrid_siea_ml_gnn_ris_two_tier_tuned_swap_local_search"
_KEPT_GNN_RIS_NODE2VEC_ML_VARIANT_LABEL = "hybrid_siea_ml_gnn_ris_node2vec_two_tier_tuned_swap_local_search"
_REMOVED_ML_VARIANT_LABELS = {
    "hybrid_siea_ml_hard_filter",
    "hybrid_siea_ml_soft_bias",
    "hybrid_siea_ml_two_tier",
    "hybrid_siea_ml_two_tier_tuned",
    "hybrid_siea_ml_two_tier_tuned_fair_init",
    "hybrid_siea_ml_two_tier_tuned_weak_mutation",
    "hybrid_siea_ml_two_tier_tuned_fair_repair",
    "hybrid_siea_ml_two_tier_tuned_worst_group_local_search",
    "hybrid_siea_ml_two_tier_tuned_fairness_full",
    "hybrid_siea_ml_two_tier_tuned_marginal_gain",
    "hybrid_siea_ml_two_tier_tuned_urgency_weighted",
    "hybrid_siea_ml_two_tier_tuned_overlap_penalty",
    "hybrid_siea_ml_two_tier_tuned_refinement_full",
    "hybrid_siea_ml_two_tier_tuned_marginal_gain_optimized",
    "hybrid_siea_ml_two_tier_tuned_marginal_gain_balanced",
    "hybrid_siea_ml_two_tier_tuned_marginal_gain_fast",
    "hybrid_siea_ml_two_tier_tuned_swap_local_search_optimized",
    "hybrid_siea_ml_two_tier_tuned_swap_local_search_first_improvement",
    "hybrid_siea_ml_two_tier_tuned_swap_local_search_reduced_candidates",
    "hybrid_siea_ml_two_tier_tuned_fairness_full_balanced",
    "hybrid_siea_ml_two_tier_tuned_fairness_full_fast",
    "hybrid_siea_ml_two_tier_tuned_node2vec",
    "hybrid_siea_ml_two_tier_tuned_node2vec_diversity",
    "ml_topk",
    "ml_topk_node2vec",
}

_FAIRNESS_COMPARE_DEFAULTS: dict[str, object] = {
    "fairness_first_init_enabled": True,
    "fairness_first_init_slots": 2,
    "fairness_first_init_weight": 0.75,
    "weakest_group_k": 2,
    "weakest_group_mutation_weight": 0.45,
    "zero_group_bonus_weight": 0.35,
    "bridge_to_weak_group_weight": 0.25,
    "repair_fairness_weight": 0.50,
    "repair_bridge_weight": 0.30,
    "repair_centrality_weight": 0.20,
    "repair_diversity_weight": 0.15,
    "local_search_focus_mode": "worst_group",
    "local_search_bottom_k_groups": 3,
    "local_search_max_trials": 8,
}

_REFINEMENT_COMPARE_DEFAULTS: dict[str, object] = {
    "marginal_gain_scoring_enabled": True,
    "marginal_gain_delta_mf_weight": 0.55,
    "marginal_gain_delta_dcv_weight": 0.45,
    "marginal_gain_spread_weight": 0.15,
    "local_search_swap_trials": 10,
    "local_search_candidate_pool_size": 8,
    "local_search_delta_mf_weight": 0.60,
    "local_search_delta_dcv_weight": 0.45,
    "local_search_overlap_penalty_weight": 1.0,
    "urgency_weight_enabled": True,
    "urgency_exponent": 1.5,
    "weak_group_focus_weight": 0.35,
    "overlap_penalty_enabled": True,
    "same_community_penalty_weight": 0.25,
    "neighborhood_overlap_penalty_weight": 0.20,
}

_FINAL_EVAL_RANDOM_SEED_OFFSET = 1_000_000
_FINAL_RECHECK_RANDOM_SEED_OFFSET = 2_000_000


@dataclass(slots=True)
class ExperimentSettings:
    """Explicit experiment settings for method comparison."""

    protected_attribute: str
    budget: int
    diffusion_model: str = DEFAULT_DIFFUSION_MODEL
    spread_estimator: str = "auto"
    community_method: str = "leiden"
    community_input_mode: str = "auto"
    community_n_clusters: int | None = None
    community_min_cluster_size: int | None = None
    community_embedding_source: str = "auto"
    community_embedding_csv: Path | None = None
    community_method_config: dict[str, object] | None = None
    propagation_probability: float = 0.01
    mc_runs: int | None = 20
    mc_runs_search: int | None = None
    mc_runs_eval: int | None = None
    enable_final_recheck: bool = False
    final_recheck_mc_runs: int = 1000
    final_recheck_top_k: int = 0
    lambda_weight: float = 0.5
    population_size: int = 12
    generations: int = 10
    crossover_probability: float = 0.7
    mutation_probability: float = 0.2
    elite_fraction: float = 0.25
    leader_guidance_fraction: float = 0.34
    local_search_steps: int = 2
    random_seed: int = 42
    output_dir: Path | None = None
    derive_protected_groups: bool = False
    derived_group_method: str | None = None
    use_node2vec: bool = False
    node2vec_dimensions: int = 8
    node2vec_walk_length: int = 20
    node2vec_num_walks: int = 10
    node2vec_window: int = 5
    node2vec_p: float = 1.0
    node2vec_q: float = 1.0
    node2vec_scale_embeddings: bool = False
    node2vec_pca_components: int | None = None
    node2vec_integration_mode: str = "feature_concat"
    node2vec_diversity_weight: float = 0.15
    use_ml: bool = False
    ml_model_type: str = "random_forest"
    ml_backend: str = "tabular"
    ml_guidance_mode: str = "off"
    debias_mode: str = "off"
    ml_top_fraction: float | None = 0.25
    ml_top_n: int | None = None
    ml_max_nodes: int | None = None
    ml_singleton_runs: int = 15
    gnn_model_type: str = "graphsage"
    gnn_node2vec_mode: str = "off"
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 2
    gnn_dropout: float = 0.2
    gnn_learning_rate: float = 1e-3
    gnn_weight_decay: float = 5e-4
    gnn_epochs: int = 100
    ris_num_rr_sets: int = 256
    ris_random_seed: int | None = None
    ris_reuse_rr_sets: bool = True
    ris_mode: str = "global"
    graphsage_weight: float | None = None
    gnn_weight: float = 1.0
    ris_weight: float = 1.0
    fair_ris_weight: float = 0.0
    fairness_urgency_weight: float = 0.0
    diversity_weight: float = 0.0
    ml_primary_pool_ratio: float = 0.25
    ml_secondary_exploration_rate: float = 0.10
    ml_initialization_bias: float = 0.25
    ml_initialization_primary_rate: float = 0.90
    ml_mutation_primary_rate: float = 0.80
    ml_repair_primary_rate: float = 0.70
    ml_local_search_primary_rate: float = 0.60
    ml_mutation_bias_weight: float = 0.20
    ml_repair_bias_weight: float = 0.15
    ml_local_search_bias_weight: float = 0.25
    fairness_first_init_enabled: bool = False
    fairness_first_init_slots: int = 0
    fairness_first_init_weight: float = 0.0
    weakest_group_k: int = 1
    weakest_group_mutation_weight: float = 0.0
    zero_group_bonus_weight: float = 0.0
    bridge_to_weak_group_weight: float = 0.0
    repair_fairness_weight: float = 0.0
    repair_bridge_weight: float = 0.0
    repair_centrality_weight: float = 0.0
    repair_diversity_weight: float = 0.0
    local_search_focus_mode: str = "default"
    local_search_bottom_k_groups: int = 3
    local_search_max_trials: int = 0
    compare_fairness_variants: bool = False
    marginal_gain_scoring_enabled: bool = False
    marginal_gain_delta_mf_weight: float = 0.0
    marginal_gain_delta_dcv_weight: float = 0.0
    marginal_gain_spread_weight: float = 0.0
    local_search_swap_trials: int = 0
    local_search_candidate_pool_size: int = 0
    local_search_delta_mf_weight: float = 0.0
    local_search_delta_dcv_weight: float = 0.0
    local_search_overlap_penalty_weight: float = 0.0
    swap_candidate_pool_size: int = 0
    swap_prefilter_top_k: int = 0
    enable_swap_cache: bool = False
    local_search_failed_patience: int = 0
    local_search_first_improvement: bool = False
    full_eval_top_k: int = 0
    proxy_score_weights: dict[str, float] | None = None
    urgency_weight_enabled: bool = False
    urgency_exponent: float = 1.0
    weak_group_focus_weight: float = 0.0
    overlap_penalty_enabled: bool = False
    same_community_penalty_weight: float = 0.0
    neighborhood_overlap_penalty_weight: float = 0.0
    compare_refinement_variants: bool = False
    marginal_candidate_pool_size: int = 0
    mutation_candidate_pool_size: int = 0
    repair_candidate_pool_size: int = 0
    candidate_prefilter_top_k: int = 0
    enable_fitness_cache: bool = True
    enable_marginal_cache: bool = False
    cache_max_size: int = 0
    local_search_early_stop_patience: int = 0
    local_search_use_prefilter: bool = False
    local_search_elite_count: int = 0
    local_search_every_n_generations: int = 1
    use_staged_mc: bool = False
    mc_runs_fast: int = 0
    mc_runs_full: int = 0
    optimization_mode: str = "full"
    refinement_intensity: float = 1.0
    marginal_eval_fraction: float = 1.0
    compare_runtime_variants: bool = False
    compare_swap_runtime_variants: bool = False
    compare_scalability_variants: bool = False


def _resolved_mc_runs_search(settings: ExperimentSettings) -> int:
    """Resolve the search-time MC budget, preserving mc_runs as a compatibility alias."""

    mc_runs_search = settings.mc_runs_search if settings.mc_runs_search is not None else settings.mc_runs
    if mc_runs_search is None or int(mc_runs_search) < 1:
        raise ValueError("mc_runs_search must resolve to an integer >= 1.")
    return int(mc_runs_search)


def _resolved_mc_runs_eval(settings: ExperimentSettings) -> int:
    """Resolve the final evaluation MC budget."""

    mc_runs_eval = settings.mc_runs_eval if settings.mc_runs_eval is not None else _resolved_mc_runs_search(settings)
    if mc_runs_eval is None or int(mc_runs_eval) < 1:
        raise ValueError("mc_runs_eval must resolve to an integer >= 1.")
    return int(mc_runs_eval)


def _resolved_eval_random_seed(settings: ExperimentSettings) -> int:
    """Use a deterministic offset so final reporting is reproducible but independent from search-time streams."""

    return int(settings.random_seed) + _FINAL_EVAL_RANDOM_SEED_OFFSET


def _validate_registry_aliases(settings: ExperimentSettings) -> None:
    valid_spread_estimators = {"auto", "mc", "ris_guidance"}
    if settings.spread_estimator not in valid_spread_estimators:
        raise ValueError(
            f"spread_estimator must be one of {sorted(valid_spread_estimators)}."
        )
    valid_debias_modes = {"off", "fairness_first", "worst_group_boost", "repair_fairness"}
    if settings.debias_mode not in valid_debias_modes:
        raise ValueError(f"debias_mode must be one of {sorted(valid_debias_modes)}.")


def _resolved_final_recheck_mc_runs(settings: ExperimentSettings) -> int:
    """Resolve the optional high-MC recheck budget used only after final seed sets are selected."""

    if int(settings.final_recheck_mc_runs) < 1:
        raise ValueError("final_recheck_mc_runs must be an integer >= 1.")
    return int(settings.final_recheck_mc_runs)


def _resolved_final_recheck_random_seed(settings: ExperimentSettings) -> int:
    """Use a second deterministic offset for optional post-hoc high-MC rechecks."""

    return int(settings.random_seed) + _FINAL_RECHECK_RANDOM_SEED_OFFSET


def _runtime_columns(
    search_runtime_seconds: float,
    final_eval_runtime_seconds: float,
    mc_runs_search: int,
    mc_runs_eval: int,
) -> dict[str, float | int]:
    return {
        "search_runtime_seconds": float(search_runtime_seconds),
        "final_eval_runtime_seconds": float(final_eval_runtime_seconds),
        "runtime_seconds": float(search_runtime_seconds + final_eval_runtime_seconds),
        "mc_runs_search": int(mc_runs_search),
        "mc_runs_eval": int(mc_runs_eval),
        "mc_runs_search_used": int(mc_runs_search),
        "mc_runs_eval_used": int(mc_runs_eval),
    }


def _spread_columns(
    evaluation: SeedSetEvaluation,
    expected_budget: int,
) -> dict[str, float | int | bool]:
    seed_set_size = len(evaluation.seed_set)
    if seed_set_size != int(expected_budget):
        raise ValueError(
            "Final evaluated seed_set size does not match the configured budget. "
            f"Expected {expected_budget}, got {seed_set_size}."
        )

    total_spread = float(evaluation.total_spread_mean)
    if total_spread + 1e-9 < float(expected_budget):
        raise ValueError(
            "total_spread is smaller than the seed budget. "
            "The IC evaluation path is expected to count the seed nodes as activated."
        )
    return {
        "total_spread": total_spread,
        "extra_spread": float(total_spread - float(expected_budget)),
        "seed_set_size": seed_set_size,
        "spread_includes_seed_nodes": True,
    }


def _final_recheck_columns() -> dict[str, object]:
    return {
        "final_recheck_applied": False,
        "final_recheck_top_k_rank": pd.NA,
        "final_recheck_mc_runs_used": pd.NA,
        "final_recheck_random_seed": pd.NA,
        "final_recheck_total_spread": pd.NA,
        "final_recheck_extra_spread": pd.NA,
        "final_recheck_mf": pd.NA,
        "final_recheck_dcv": pd.NA,
        "final_recheck_f_score": pd.NA,
        "final_recheck_runtime_seconds": pd.NA,
    }


def _final_evaluate_seed_set(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    settings: ExperimentSettings,
    seed_set: tuple[object, ...],
) -> SeedSetEvaluation:
    """Run the authoritative final evaluation budget used for reported tables and CSV output."""

    return evaluate_seed_set(
        dataset=dataset,
        protected_group_report=protected_group_report,
        seed_set=seed_set,
        propagation_probability=settings.propagation_probability,
        mc_runs=_resolved_mc_runs_eval(settings),
        random_seed=_resolved_eval_random_seed(settings),
        lambda_weight=settings.lambda_weight,
        include_soft_mf=True,
        diffusion_model=settings.diffusion_model,
    )


def _dataset_output_dir(output_dir: Path | None, dataset_name: str) -> Path | None:
    if output_dir is None:
        return None

    return output_dir / dataset_name


def _protected_attribute_output_component(protected_attribute: str) -> str:
    sanitized = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in protected_attribute.strip()
    ).strip("._-")
    return sanitized or "protected_attribute"


def build_results_output_dir(
    output_dir: Path | None,
    dataset_name: str,
    protected_attribute: str,
) -> Path | None:
    dataset_dir = _dataset_output_dir(output_dir, dataset_name)
    if dataset_dir is None:
        return None

    return dataset_dir / _protected_attribute_output_component(protected_attribute)


def _ensure_results_output_dir(
    output_dir: Path | None,
    dataset_name: str,
    protected_attribute: str,
) -> Path | None:
    dataset_dir = _dataset_output_dir(output_dir, dataset_name)
    if dataset_dir is None:
        return None

    dataset_dir.mkdir(parents=True, exist_ok=True)
    attribute_dir = dataset_dir / _protected_attribute_output_component(protected_attribute)
    attribute_dir.mkdir(parents=True, exist_ok=True)
    return attribute_dir


def _history_path(
    output_dir: Path | None,
    dataset_name: str,
    protected_attribute: str,
    budget: int,
    community_method: str,
    label: str,
) -> Path | None:
    attribute_dir = _ensure_results_output_dir(output_dir, dataset_name, protected_attribute)
    if attribute_dir is None:
        return None

    path = attribute_dir / f"{dataset_name}_budget{budget}_{community_method}_{label}_history.csv"
    return path


def _community_columns(quality: CommunityQualityMetrics) -> dict[str, float | int]:
    return {
        "community_modularity": quality.modularity,
        "num_communities": quality.num_communities,
        "largest_community_size": quality.largest_community_size,
        "smallest_community_size": quality.smallest_community_size,
        "average_community_size": quality.average_community_size,
        "community_size_std": quality.community_size_std,
    }


def _community_metadata_columns(community_result) -> dict[str, object]:
    return {
        "community_category": getattr(community_result, "category", "graph_native"),
        "community_input_mode": getattr(community_result, "resolved_input_mode", "graph"),
        "community_requested_input_mode": getattr(community_result, "requested_input_mode", "graph"),
    }


def _load_community_embedding_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "node_id" not in frame.columns:
        raise ValueError(
            f"community_embedding_csv '{path}' must contain a node_id column."
        )
    return frame


def _resolve_community_clustering_inputs(
    dataset: LoadedDataset,
    settings: ExperimentSettings,
    community_method: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str, dict[str, object]]:
    method_spec = get_clustering_method_spec(community_method)
    method_config = dict(settings.community_method_config or {})
    if settings.community_n_clusters is not None:
        method_config.setdefault("n_clusters", int(settings.community_n_clusters))
    if settings.community_min_cluster_size is not None:
        method_config.setdefault("min_cluster_size", int(settings.community_min_cluster_size))

    if method_spec.category == "graph_native":
        return None, None, "graph", method_config

    from .embeddings.features import prepare_benchmark_features

    configured_source = str(settings.community_embedding_source).strip().lower()
    if configured_source not in {"auto", "csv", "feature"}:
        raise ValueError(
            "community_embedding_source must be one of ['auto', 'csv', 'feature']."
        )
    if str(settings.community_input_mode).strip().lower() == "graph":
        raise ValueError(
            f"community_input_mode='graph' is incompatible with embedding-space community method '{community_method}'."
        )

    embeddings: pd.DataFrame | None = None
    features: pd.DataFrame | None = None
    if configured_source in {"auto", "csv"} and settings.community_embedding_csv is not None:
        embedding_path = Path(settings.community_embedding_csv)
        if not embedding_path.is_file():
            raise FileNotFoundError(f"community_embedding_csv file not found: {embedding_path}")
        embeddings = _load_community_embedding_frame(embedding_path)
    if configured_source == "csv" and embeddings is None:
        raise ValueError(
            "community_embedding_source='csv' requires community_embedding_csv to be set."
        )
    if embeddings is None and configured_source in {"auto", "feature"}:
        prepared = prepare_benchmark_features(dataset, graph=dataset.graph, random_seed=settings.random_seed)
        features = prepared.feature_frame.copy()

    return embeddings, features, str(settings.community_input_mode).strip().lower(), method_config


def _fairness_diagnostic_columns(
    group_spread: dict[str, float],
    normalized_group_spread: dict[str, float],
) -> dict[str, float | int | str]:
    ordered_raw = sorted(group_spread.items(), key=lambda item: (float(item[1]), item[0]))
    ordered_normalized = sorted(
        normalized_group_spread.items(),
        key=lambda item: (float(item[1]), item[0]),
    )
    covered_groups = [
        group_name
        for group_name, spread in group_spread.items()
        if float(spread) > 1e-12
    ]
    bottom_raw_values = [float(value) for _, value in ordered_raw[: min(3, len(ordered_raw))]]
    weakest_groups = [group_name for group_name, _ in ordered_normalized[: min(3, len(ordered_normalized))]]
    return {
        "zero_covered_groups_count": len(group_spread) - len(covered_groups),
        "bottom_3_avg_group_spread": float(sum(bottom_raw_values) / len(bottom_raw_values)) if bottom_raw_values else 0.0,
        "fraction_groups_covered": float(len(covered_groups)) / float(len(group_spread)) if group_spread else 0.0,
        "weakest_groups_note": ",".join(weakest_groups),
    }


def _ml_columns(
    guidance_mode: str = "off",
    ml_backend: str = "none",
    gnn_model_type: str | None = None,
    ris_enabled: bool = False,
    ris_mode: str = "off",
    validation_spearman: float | None = None,
    validation_precision_at_budget: float | None = None,
) -> dict[str, object]:
    return {
        "ml_guidance_mode": guidance_mode,
        "guidance_mode": ml_backend,
        "ml_backend": ml_backend,
        "gnn_model_type": gnn_model_type,
        "graphsage_enabled": _graphsage_enabled_flag(gnn_model_type, ml_backend=ml_backend),
        "ris_enabled": bool(ris_enabled),
        "ris_mode": ris_mode,
        "ml_validation_spearman": validation_spearman,
        "ml_validation_precision_at_budget": validation_precision_at_budget,
    }


def _baseline_row(
    dataset: LoadedDataset,
    community_method: str,
    method: str,
    evaluation: SeedSetEvaluation,
    budget: int,
    search_runtime_seconds: float,
    quality: CommunityQualityMetrics,
    diffusion_model: str,
    mc_runs_search: int,
    mc_runs_eval: int,
    community_result=None,
) -> dict[str, object]:
    return {
        "dataset": dataset.name,
        "community_method": community_method,
        "clustering_method": "none",
        "clustering_input_mode": "none",
        "diffusion_model": diffusion_model,
        "ranking_model": "none",
        "embedding_method": "none",
        "search_spread_estimator": (
            "monte_carlo"
            if method in {"greedy", "fairness_weighted_greedy", "maximin_greedy"}
            else "not_used"
        ),
        "search_guidance_estimator": "none",
        "final_spread_estimator": "monte_carlo",
        "method": method,
        "variant_type": "baseline",
        "seed_set": json.dumps(list(evaluation.seed_set)),
        **_spread_columns(evaluation, expected_budget=budget),
        "mf": evaluation.fairness.mf,
        "dcv": evaluation.fairness.dcv,
        "f_score": evaluation.f_score,
        "candidate_pool_size": dataset.graph.number_of_nodes(),
        "optimization_mode": "full",
        "swarm_guidance": False,
        "crossover": False,
        "local_search": False,
        "community_aware_mutation": method == "community_round_robin",
        "node2vec_enabled": False,
        "node2vec_mode": "off",
        "note": "",
        **_runtime_columns(
            search_runtime_seconds=search_runtime_seconds,
            final_eval_runtime_seconds=evaluation.runtime_seconds,
            mc_runs_search=mc_runs_search,
            mc_runs_eval=mc_runs_eval,
        ),
        **_final_recheck_columns(),
        **_ml_columns(guidance_mode="off", ml_backend="none"),
        **_fairness_diagnostic_columns(
            evaluation.fairness.group_spread,
            evaluation.fairness.normalized_group_spread,
        ),
        **_community_columns(quality),
        **({} if community_result is None else _community_metadata_columns(community_result)),
    }


def _hybrid_row(
    dataset: LoadedDataset,
    community_method: str,
    label: str,
    variant_type: str,
    result: HybridOptimizationResult,
    evaluation: SeedSetEvaluation,
    search_runtime_seconds: float,
    config: HybridSIEAConfig,
    quality: CommunityQualityMetrics,
    diffusion_model: str,
    mc_runs_search: int,
    mc_runs_eval: int,
    note: str = "",
    node2vec_enabled: bool = False,
    node2vec_mode: str = "off",
    embedding_method: str = "none",
    clustering_method: str = "none",
    clustering_input_mode: str = "none",
    community_result=None,
) -> dict[str, object]:
    return {
        "dataset": dataset.name,
        "community_method": community_method,
        "clustering_method": clustering_method,
        "clustering_input_mode": clustering_input_mode,
        "diffusion_model": diffusion_model,
        "ranking_model": "none",
        "embedding_method": (
            embedding_method
            if embedding_method != "none"
            else ("node2vec" if node2vec_enabled else "none")
        ),
        "search_spread_estimator": "monte_carlo",
        "search_guidance_estimator": "none",
        "final_spread_estimator": "monte_carlo",
        "method": label,
        "variant_type": variant_type,
        "seed_set": json.dumps(list(evaluation.seed_set)),
        **_spread_columns(evaluation, expected_budget=config.budget),
        "mf": evaluation.fairness.mf,
        "dcv": evaluation.fairness.dcv,
        "f_score": evaluation.f_score,
        "candidate_pool_size": result.candidate_pool_size,
        "optimization_mode": config.optimization_mode,
        "swarm_guidance": not config.disable_swarm_guidance,
        "crossover": not config.disable_crossover,
        "local_search": not config.disable_local_search,
        "community_aware_mutation": not config.disable_community_aware_mutation,
        "node2vec_enabled": node2vec_enabled,
        "node2vec_mode": node2vec_mode,
        "note": note,
        **_runtime_columns(
            search_runtime_seconds=search_runtime_seconds,
            final_eval_runtime_seconds=evaluation.runtime_seconds,
            mc_runs_search=mc_runs_search,
            mc_runs_eval=mc_runs_eval,
        ),
        **_final_recheck_columns(),
        **_ml_columns(guidance_mode=config.ml_guidance_mode, ml_backend="none"),
        **_fairness_diagnostic_columns(
            evaluation.fairness.group_spread,
            evaluation.fairness.normalized_group_spread,
        ),
        **_community_columns(quality),
        **({} if community_result is None else _community_metadata_columns(community_result)),
    }


def _ml_row(
    dataset: LoadedDataset,
    community_method: str,
    label: str,
    variant_type: str,
    evaluation: SeedSetEvaluation,
    budget: int,
    runtime_seconds: float,
    quality: CommunityQualityMetrics,
    candidate_pool_size: int,
    ml_backend: str,
    gnn_model_type: str | None,
    ris_enabled: bool,
    ris_mode: str,
    validation_spearman: float,
    validation_precision_at_budget: float,
    guidance_mode: str,
    diffusion_model: str,
    note: str = "",
    node2vec_enabled: bool = False,
    node2vec_mode: str = "off",
    search_runtime_seconds: float | None = None,
    mc_runs_search: int | None = None,
    mc_runs_eval: int | None = None,
    community_result=None,
) -> dict[str, object]:
    resolved_search_runtime = runtime_seconds if search_runtime_seconds is None else search_runtime_seconds
    resolved_mc_runs_search = 0 if mc_runs_search is None else int(mc_runs_search)
    resolved_mc_runs_eval = 0 if mc_runs_eval is None else int(mc_runs_eval)
    return {
        "dataset": dataset.name,
        "community_method": community_method,
        "diffusion_model": diffusion_model,
        "ranking_model": "none" if gnn_model_type is None else gnn_model_type,
        "embedding_method": "node2vec" if node2vec_enabled else "none",
        "search_spread_estimator": "monte_carlo",
        "search_guidance_estimator": "ris_guidance" if ris_enabled else "none",
        "final_spread_estimator": "monte_carlo",
        "method": label,
        "variant_type": variant_type,
        "seed_set": json.dumps(list(evaluation.seed_set)),
        **_spread_columns(evaluation, expected_budget=budget),
        "mf": evaluation.fairness.mf,
        "dcv": evaluation.fairness.dcv,
        "f_score": evaluation.f_score,
        "candidate_pool_size": candidate_pool_size,
        "optimization_mode": "full",
        "swarm_guidance": False,
        "crossover": False,
        "local_search": False,
        "community_aware_mutation": False,
        "node2vec_enabled": node2vec_enabled,
        "node2vec_mode": node2vec_mode,
        "note": note,
        **_runtime_columns(
            search_runtime_seconds=resolved_search_runtime,
            final_eval_runtime_seconds=evaluation.runtime_seconds,
            mc_runs_search=resolved_mc_runs_search,
            mc_runs_eval=resolved_mc_runs_eval,
        ),
        **_final_recheck_columns(),
        **_ml_columns(
            guidance_mode=guidance_mode,
            ml_backend=ml_backend,
            gnn_model_type=gnn_model_type,
            ris_enabled=ris_enabled,
            ris_mode=ris_mode,
            validation_spearman=validation_spearman,
            validation_precision_at_budget=validation_precision_at_budget,
        ),
        **_fairness_diagnostic_columns(
            evaluation.fairness.group_spread,
            evaluation.fairness.normalized_group_spread,
        ),
        **_community_columns(quality),
        **({} if community_result is None else _community_metadata_columns(community_result)),
    }


def _generate_ml_labels(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    settings: ExperimentSettings,
) -> tuple[NodeUtilityLabelResult, float]:
    start = perf_counter()
    label_result = generate_singleton_node_utility_labels(
        dataset=dataset,
        protected_group_report=protected_group_report,
        propagation_probability=settings.propagation_probability,
        mc_runs=settings.ml_singleton_runs,
        diffusion_model=settings.diffusion_model,
        lambda_weight=settings.lambda_weight,
        random_seed=settings.random_seed,
    )
    return label_result, perf_counter() - start


def _resolve_ml_backends(settings: ExperimentSettings) -> tuple[str, ...]:
    valid_backends = {
        "tabular": ("tabular",),
        "gnn": ("gnn",),
        "ris": ("ris",),
        "gnn_ris": ("gnn_ris",),
        "both": ("tabular", "gnn"),
        "all": ("tabular", "gnn", "ris", "gnn_ris"),
    }
    if settings.ml_backend not in valid_backends:
        raise ValueError("ml_backend must be one of ['tabular', 'gnn', 'ris', 'gnn_ris', 'both', 'all'].")
    return valid_backends[settings.ml_backend]


def _resolve_gnn_node2vec_modes(settings: ExperimentSettings) -> tuple[str, ...]:
    valid_modes = {
        "off": ("off",),
        "input_concat": ("input_concat",),
        "compare": ("off", "input_concat"),
    }
    if settings.gnn_node2vec_mode not in valid_modes:
        raise ValueError("gnn_node2vec_mode must be one of ['off', 'input_concat', 'compare'].")
    return valid_modes[settings.gnn_node2vec_mode]


def _node2vec_config(settings: ExperimentSettings) -> Node2VecConfig:
    return Node2VecConfig(
        dimensions=int(settings.node2vec_dimensions),
        walk_length=int(settings.node2vec_walk_length),
        num_walks=int(settings.node2vec_num_walks),
        window=int(settings.node2vec_window),
        p=float(settings.node2vec_p),
        q=float(settings.node2vec_q),
        scale_embeddings=bool(settings.node2vec_scale_embeddings),
        pca_components=(
            None
            if settings.node2vec_pca_components is None
            else int(settings.node2vec_pca_components)
        ),
        random_seed=int(settings.random_seed),
    )


def _node2vec_cache_path(
    output_dir: Path | None,
    dataset_name: str,
    settings: ExperimentSettings,
) -> Path | None:
    return build_node2vec_cache_path(output_dir, dataset_name, _node2vec_config(settings))


def _compute_ml_feature_frame(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    community_result,
    settings: ExperimentSettings,
) -> tuple[dict[str, pd.DataFrame], float]:
    start = perf_counter()
    plain_frame = compute_node_features(
        dataset=dataset,
        protected_group_report=protected_group_report,
        community_result=community_result,
    )
    feature_frames = {"plain": plain_frame}
    if settings.use_ml and any(backend in {"gnn", "gnn_ris"} for backend in _resolve_ml_backends(settings)):
        if "input_concat" in _resolve_gnn_node2vec_modes(settings):
            feature_frames["input_concat"] = compute_node_features(
                dataset=dataset,
                protected_group_report=protected_group_report,
                community_result=community_result,
                node2vec_config=_node2vec_config(settings),
                node2vec_cache_path=_node2vec_cache_path(
                    output_dir=settings.output_dir,
                    dataset_name=dataset.name,
                    settings=settings,
                ),
            )
    return feature_frames, perf_counter() - start


def _minmax_normalize(values: pd.Series, *, column_name: str) -> pd.Series:
    numeric_values = values.astype(float)
    value_min = float(numeric_values.min())
    value_max = float(numeric_values.max())
    if value_max - value_min <= 1e-12:
        raise ValueError(f"{column_name} is degenerate and cannot be normalized for GNN labels.")
    return (numeric_values - value_min) / (value_max - value_min)


def _build_gnn_label_frame(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    community_result,
    settings: ExperimentSettings,
    label_result: NodeUtilityLabelResult,
) -> tuple[pd.DataFrame, float]:
    start = perf_counter()
    label_frame = label_result.label_frame.copy()
    if "soft_fair_norm" not in label_frame.columns:
        raise ValueError("label_frame must contain soft_fair_norm for GNN label construction.")
    if "weak_group_gain_norm" not in label_frame.columns:
        raise ValueError("label_frame must contain weak_group_gain_norm for GNN label construction.")

    proxy_config = _build_optimizer_config(
        settings,
        include_fairness_parameters=False,
        ml_guidance_mode="off",
        **_resolved_refinement_compare_settings(settings),
    )
    optimizer = HybridSIEAOptimizer(
        dataset=dataset,
        protected_group_report=protected_group_report,
        community_result=community_result,
        config=proxy_config,
    )
    marginal_proxy_scores = [
        optimizer._marginal_gain_proxy_score(  # noqa: SLF001
            node_id=node_id,
            seed_set=(),
            community_counts={},
        )
        for node_id in label_frame["node_id"]
    ]
    label_frame["marginal_proxy_score"] = pd.Series(marginal_proxy_scores, index=label_frame.index, dtype=float)
    label_frame["marginal_proxy_norm"] = _minmax_normalize(
        label_frame["marginal_proxy_score"],
        column_name="marginal_proxy_score",
    )
    # Keep the historical alias for compatibility, but the canonical supervised
    # target for FIM ranking remains label_score.
    label_frame["gnn_label_score"] = label_frame["label_score"].astype(float)
    label_variance = float(label_frame["gnn_label_score"].var(ddof=0))
    if label_variance <= 1e-12:
        raise ValueError("label_score / gnn_label_score is degenerate; adjust label construction or dataset settings.")
    return label_frame, perf_counter() - start


def _normalize_score_map(scores: dict[object, float]) -> dict[object, float]:
    if not scores:
        return {}
    values = pd.Series(scores, dtype=float)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return {node_id: 0.0 for node_id in scores}
    return {
        node_id: float((float(score) - minimum) / (maximum - minimum))
        for node_id, score in scores.items()
    }


def _effective_gnn_weight(settings: ExperimentSettings) -> float:
    if settings.graphsage_weight is not None:
        return float(settings.graphsage_weight)
    return float(settings.gnn_weight)


def _graphsage_enabled_flag(
    gnn_model_type: object,
    *,
    ml_backend: str,
) -> bool:
    return bool(
        ml_backend in {"gnn", "gnn_ris"}
        and not pd.isna(gnn_model_type)
        and str(gnn_model_type) == "graphsage"
    )


def _combine_weighted_score_maps(
    component_maps: list[tuple[dict[object, float], float]],
    node_ids: list[object],
) -> dict[object, float]:
    combined = {node_id: 0.0 for node_id in node_ids}
    active_components = 0
    for component_map, weight in component_maps:
        if weight <= 0.0:
            continue
        active_components += 1
        normalized_component = _normalize_score_map(component_map)
        for node_id in node_ids:
            combined[node_id] += float(weight) * float(normalized_component.get(node_id, 0.0))
    if active_components == 0:
        raise ValueError("At least one positive guidance component weight is required to build node scores.")
    return _normalize_score_map(combined)


def _static_fairness_urgency_scores(feature_frame: pd.DataFrame) -> dict[object, float]:
    required_columns = {
        "node_id",
        "fraction_neighbors_in_undercovered_groups",
        "inverse_group_size",
        "minority_group_indicator",
    }
    missing = required_columns.difference(feature_frame.columns)
    if missing:
        raise ValueError(f"feature_frame is missing fairness urgency columns: {sorted(missing)}.")
    frame = feature_frame.loc[:, list(required_columns)].copy()
    frame["fairness_prior"] = (
        0.50 * frame["fraction_neighbors_in_undercovered_groups"].astype(float)
        + 0.35 * frame["inverse_group_size"].astype(float)
        + 0.15 * frame["minority_group_indicator"].astype(float)
    )
    return _normalize_score_map(
        {
            row.node_id: float(row.fairness_prior)
            for row in frame.itertuples(index=False)
        }
    )


def _static_diversity_scores(feature_frame: pd.DataFrame) -> dict[object, float]:
    required_columns = {
        "node_id",
        "cross_community_degree",
        "neighboring_communities",
        "neighborhood_group_entropy",
        "clustering_coefficient",
    }
    missing = required_columns.difference(feature_frame.columns)
    if missing:
        raise ValueError(f"feature_frame is missing diversity columns: {sorted(missing)}.")
    frame = feature_frame.loc[:, list(required_columns)].copy()
    frame["diversity_prior"] = (
        0.35 * frame["cross_community_degree"].astype(float)
        + 0.30 * frame["neighboring_communities"].astype(float)
        + 0.25 * frame["neighborhood_group_entropy"].astype(float)
        + 0.10 * (1.0 - frame["clustering_coefficient"].astype(float))
    )
    return _normalize_score_map(
        {
            row.node_id: float(row.diversity_prior)
            for row in frame.itertuples(index=False)
        }
    )


def _ris_group_weights(
    feature_frame: pd.DataFrame,
    protected_group_report: ProtectedGroupReport,
) -> dict[str, float]:
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


def _ris_config(settings: ExperimentSettings) -> RISConfig:
    return RISConfig(
        num_rr_sets=int(settings.ris_num_rr_sets),
        random_seed=(
            int(settings.random_seed)
            if settings.ris_random_seed is None
            else int(settings.ris_random_seed)
        ),
        reuse_rr_sets=bool(settings.ris_reuse_rr_sets),
        mode=settings.ris_mode,
    )


def _prepare_ris_guidance(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    settings: ExperimentSettings,
    *,
    feature_frame: pd.DataFrame,
    stack_name: str = "experiment_runner_ris",
) -> tuple[RISGuidanceResult, float]:
    artifact = prepare_stack_ris_guidance(
        dataset=dataset,
        protected_group_report=protected_group_report,
        propagation_probability=settings.propagation_probability,
        feature_frame=feature_frame,
        output_dir=None,
        protected_attribute=None,
        stack_name=None,
        config=_ris_config(settings),
    )
    del stack_name
    return artifact.ris_result, float(artifact.ris_result.runtime_seconds)


def _select_ris_scores(
    ris_result: RISGuidanceResult,
    feature_frame: pd.DataFrame,
    protected_group_report: ProtectedGroupReport,
    settings: ExperimentSettings,
) -> dict[object, float]:
    if settings.ris_mode == "global":
        return dict(ris_result.global_node_scores)
    if settings.ris_mode == "weak_group_weighted":
        return ris_result.weighted_node_scores(
            _ris_group_weights(feature_frame, protected_group_report)
        )
    raise ValueError("ris_mode must be one of ['global', 'weak_group_weighted'].")


def _build_fair_ris_scores(
    ris_result: RISGuidanceResult,
    feature_frame: pd.DataFrame,
    protected_group_report: ProtectedGroupReport,
) -> dict[object, float]:
    return ris_result.weighted_node_scores(
        _ris_group_weights(feature_frame, protected_group_report)
    )


def _build_guidance_score_map(
    node_ids: list[object],
    settings: ExperimentSettings,
    *,
    gnn_scores: dict[object, float] | None = None,
    ris_scores: dict[object, float] | None = None,
    fair_ris_scores: dict[object, float] | None = None,
    fairness_scores: dict[object, float] | None = None,
    diversity_scores: dict[object, float] | None = None,
) -> dict[object, float]:
    component_maps: list[tuple[dict[object, float], float]] = []
    if gnn_scores is not None:
        component_maps.append((gnn_scores, _effective_gnn_weight(settings)))
    if ris_scores is not None:
        component_maps.append((ris_scores, float(settings.ris_weight)))
    if fair_ris_scores is not None:
        component_maps.append((fair_ris_scores, float(settings.fair_ris_weight)))
    if fairness_scores is not None:
        component_maps.append((fairness_scores, float(settings.fairness_urgency_weight)))
    if diversity_scores is not None:
        component_maps.append((diversity_scores, float(settings.diversity_weight)))
    return _combine_weighted_score_maps(component_maps, node_ids=node_ids)


def _prepare_tabular_ml_training(
    dataset: LoadedDataset,
    feature_frame: pd.DataFrame,
    settings: ExperimentSettings,
    label_result: NodeUtilityLabelResult,
) -> tuple[RankingTrainingResult, float]:
    artifact = train_stack_ranking_model(
        dataset,
        feature_frame=feature_frame,
        label_frame=label_result.label_frame,
        budget=settings.budget,
        model_type=settings.ml_model_type,
        target_column="label_score",
        protected_attribute=None,
        stack_name=None,
        output_dir=None,
        top_fraction=settings.ml_top_fraction,
        top_n=settings.ml_top_n,
        max_nodes=settings.ml_max_nodes,
        random_seed=settings.random_seed,
    )
    return artifact.training_result, float(artifact.training_result.runtime_seconds)


def _gnn_cache_path(
    output_dir: Path | None,
    dataset_name: str,
    protected_attribute: str,
    budget: int,
    community_method: str,
    settings: ExperimentSettings,
    *,
    node2vec_mode: str,
) -> Path | None:
    attribute_dir = _ensure_results_output_dir(output_dir, dataset_name, protected_attribute)
    if attribute_dir is None:
        return None

    cache_signature = {
        "gnn_node2vec_mode": node2vec_mode,
    }
    if node2vec_mode == "input_concat":
        cache_signature["node2vec_config"] = {
            "dimensions": int(settings.node2vec_dimensions),
            "walk_length": int(settings.node2vec_walk_length),
            "num_walks": int(settings.node2vec_num_walks),
            "window": int(settings.node2vec_window),
            "p": float(settings.node2vec_p),
            "q": float(settings.node2vec_q),
            "scale_embeddings": bool(settings.node2vec_scale_embeddings),
            "pca_components": settings.node2vec_pca_components,
            "random_seed": int(settings.random_seed),
        }
    cache_suffix = hashlib.sha256(
        json.dumps(cache_signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return (
        attribute_dir
        / f"{dataset_name}_budget{budget}_{community_method}_gnn_"
        f"{settings.gnn_model_type}_{node2vec_mode}_seed{settings.random_seed}_{cache_suffix}.pkl"
    )


def _prepare_gnn_training(
    dataset: LoadedDataset,
    feature_frame: pd.DataFrame,
    community_method: str,
    settings: ExperimentSettings,
    label_frame: pd.DataFrame,
    *,
    target_column: str,
    node2vec_mode: str,
) -> tuple[RankingTrainingResult, float]:
    artifact = train_stack_ranking_model(
        dataset,
        feature_frame=feature_frame,
        label_frame=label_frame,
        budget=settings.budget,
        model_type=settings.gnn_model_type,
        target_column=target_column,
        protected_attribute=None,
        stack_name=None,
        output_dir=None,
        hidden_dim=settings.gnn_hidden_dim,
        num_layers=settings.gnn_num_layers,
        dropout=settings.gnn_dropout,
        learning_rate=settings.gnn_learning_rate,
        weight_decay=settings.gnn_weight_decay,
        epochs=settings.gnn_epochs,
        top_fraction=settings.ml_top_fraction,
        top_n=settings.ml_top_n,
        max_nodes=settings.ml_max_nodes,
        random_seed=settings.random_seed,
        node2vec_mode=node2vec_mode,
        node2vec_config=(
            {
                "dimensions": int(settings.node2vec_dimensions),
                "walk_length": int(settings.node2vec_walk_length),
                "num_walks": int(settings.node2vec_num_walks),
                "window": int(settings.node2vec_window),
                "p": float(settings.node2vec_p),
                "q": float(settings.node2vec_q),
                "scale_embeddings": bool(settings.node2vec_scale_embeddings),
                "pca_components": settings.node2vec_pca_components,
                "random_seed": int(settings.random_seed),
            }
            if node2vec_mode == "input_concat"
            else None
        ),
        cache_path=_gnn_cache_path(
            output_dir=settings.output_dir,
            dataset_name=dataset.name,
            protected_attribute=settings.protected_attribute,
            budget=settings.budget,
            community_method=community_method,
            settings=settings,
            node2vec_mode=node2vec_mode,
        ),
        debias_mode="none",
        protected_attribute_column="protected_group",
    )
    del community_method
    return artifact.training_result, float(artifact.training_result.runtime_seconds)


def _selected_ml_variants(settings: ExperimentSettings) -> list[tuple[str, str, str, str, bool]]:
    if not settings.use_ml:
        return []

    selected: list[tuple[str, str, str, str, bool]] = []
    for backend in _resolve_ml_backends(settings):
        if backend == "tabular":
            selected.append((_KEPT_ML_VARIANT_LABEL, "two_tier", "tabular", "off", False))
        elif backend == "gnn":
            for node2vec_mode in _resolve_gnn_node2vec_modes(settings):
                label = (
                    _KEPT_GNN_NODE2VEC_ML_VARIANT_LABEL
                    if node2vec_mode == "input_concat"
                    else _KEPT_GNN_ML_VARIANT_LABEL
                )
                selected.append((label, "two_tier", "gnn", node2vec_mode, False))
        elif backend == "ris":
            selected.append((_KEPT_RIS_ML_VARIANT_LABEL, "two_tier", "ris", "off", True))
        elif backend == "gnn_ris":
            for node2vec_mode in _resolve_gnn_node2vec_modes(settings):
                label = (
                    _KEPT_GNN_RIS_NODE2VEC_ML_VARIANT_LABEL
                    if node2vec_mode == "input_concat"
                    else _KEPT_GNN_RIS_ML_VARIANT_LABEL
                )
                selected.append((label, "two_tier", "gnn_ris", node2vec_mode, True))
    return selected


def _resolved_kept_ml_overrides(settings: ExperimentSettings) -> dict[str, object]:
    fairness_defaults = _resolved_fairness_compare_settings(settings)
    refinement_defaults = _resolved_refinement_compare_settings(settings)
    return {
        **fairness_defaults,
        "local_search_swap_trials": int(refinement_defaults["local_search_swap_trials"]),
        "local_search_candidate_pool_size": int(refinement_defaults["local_search_candidate_pool_size"]),
        "local_search_delta_mf_weight": float(refinement_defaults["local_search_delta_mf_weight"]),
        "local_search_delta_dcv_weight": float(refinement_defaults["local_search_delta_dcv_weight"]),
        "local_search_overlap_penalty_weight": float(refinement_defaults["local_search_overlap_penalty_weight"]),
    }


def _resolved_fairness_compare_settings(settings: ExperimentSettings) -> dict[str, object]:
    resolved = dict(_FAIRNESS_COMPARE_DEFAULTS)
    resolved["fairness_first_init_enabled"] = (
        settings.fairness_first_init_enabled or bool(_FAIRNESS_COMPARE_DEFAULTS["fairness_first_init_enabled"])
    )
    resolved["fairness_first_init_slots"] = (
        settings.fairness_first_init_slots
        if settings.fairness_first_init_slots > 0
        else min(int(_FAIRNESS_COMPARE_DEFAULTS["fairness_first_init_slots"]), settings.budget)
    )
    for field_name in [
        "fairness_first_init_weight",
        "weakest_group_k",
        "weakest_group_mutation_weight",
        "zero_group_bonus_weight",
        "bridge_to_weak_group_weight",
        "repair_fairness_weight",
        "repair_bridge_weight",
        "repair_centrality_weight",
        "repair_diversity_weight",
        "local_search_bottom_k_groups",
        "local_search_max_trials",
    ]:
        current_value = getattr(settings, field_name)
        resolved[field_name] = current_value if current_value not in {0, 0.0} else _FAIRNESS_COMPARE_DEFAULTS[field_name]
    resolved["local_search_focus_mode"] = (
        settings.local_search_focus_mode
        if settings.local_search_focus_mode != "default"
        else str(_FAIRNESS_COMPARE_DEFAULTS["local_search_focus_mode"])
    )
    return resolved


def _resolved_refinement_compare_settings(settings: ExperimentSettings) -> dict[str, object]:
    resolved = dict(_REFINEMENT_COMPARE_DEFAULTS)
    for field_name in [
        "marginal_gain_delta_mf_weight",
        "marginal_gain_delta_dcv_weight",
        "marginal_gain_spread_weight",
        "local_search_swap_trials",
        "local_search_candidate_pool_size",
        "local_search_delta_mf_weight",
        "local_search_delta_dcv_weight",
        "local_search_overlap_penalty_weight",
        "urgency_exponent",
        "weak_group_focus_weight",
        "same_community_penalty_weight",
        "neighborhood_overlap_penalty_weight",
    ]:
        current_value = getattr(settings, field_name)
        resolved[field_name] = current_value if current_value not in {0, 0.0} else _REFINEMENT_COMPARE_DEFAULTS[field_name]
    resolved["marginal_gain_scoring_enabled"] = (
        settings.marginal_gain_scoring_enabled or bool(_REFINEMENT_COMPARE_DEFAULTS["marginal_gain_scoring_enabled"])
    )
    resolved["urgency_weight_enabled"] = (
        settings.urgency_weight_enabled or bool(_REFINEMENT_COMPARE_DEFAULTS["urgency_weight_enabled"])
    )
    resolved["overlap_penalty_enabled"] = (
        settings.overlap_penalty_enabled or bool(_REFINEMENT_COMPARE_DEFAULTS["overlap_penalty_enabled"])
    )
    return resolved


def _build_optimizer_config(
    settings: ExperimentSettings,
    include_fairness_parameters: bool = True,
    **overrides: object,
) -> HybridSIEAConfig:
    mc_runs_search = _resolved_mc_runs_search(settings)
    config = HybridSIEAConfig(
        budget=settings.budget,
        diffusion_model=settings.diffusion_model,
        population_size=settings.population_size,
        generations=settings.generations,
        crossover_probability=settings.crossover_probability,
        mutation_probability=settings.mutation_probability,
        elite_fraction=settings.elite_fraction,
        leader_guidance_fraction=settings.leader_guidance_fraction,
        propagation_probability=settings.propagation_probability,
        mc_runs=mc_runs_search,
        lambda_weight=settings.lambda_weight,
        random_seed=settings.random_seed,
        local_search_steps=settings.local_search_steps,
        ml_guidance_mode="off",
        ml_primary_pool_ratio=settings.ml_primary_pool_ratio,
        ml_secondary_exploration_rate=settings.ml_secondary_exploration_rate,
        ml_initialization_bias=settings.ml_initialization_bias,
        ml_initialization_primary_rate=settings.ml_initialization_primary_rate,
        ml_mutation_primary_rate=settings.ml_mutation_primary_rate,
        ml_repair_primary_rate=settings.ml_repair_primary_rate,
        ml_local_search_primary_rate=settings.ml_local_search_primary_rate,
        ml_mutation_bias_weight=settings.ml_mutation_bias_weight,
        ml_repair_bias_weight=settings.ml_repair_bias_weight,
        ml_local_search_bias_weight=settings.ml_local_search_bias_weight,
    )
    if include_fairness_parameters:
        config.fairness_first_init_enabled = settings.fairness_first_init_enabled
        config.fairness_first_init_slots = settings.fairness_first_init_slots
        config.fairness_first_init_weight = settings.fairness_first_init_weight
        config.weakest_group_k = settings.weakest_group_k
        config.weakest_group_mutation_weight = settings.weakest_group_mutation_weight
        config.zero_group_bonus_weight = settings.zero_group_bonus_weight
        config.bridge_to_weak_group_weight = settings.bridge_to_weak_group_weight
        config.repair_fairness_weight = settings.repair_fairness_weight
        config.repair_bridge_weight = settings.repair_bridge_weight
        config.repair_centrality_weight = settings.repair_centrality_weight
        config.repair_diversity_weight = settings.repair_diversity_weight
        config.local_search_focus_mode = settings.local_search_focus_mode
        config.local_search_bottom_k_groups = settings.local_search_bottom_k_groups
        config.local_search_max_trials = settings.local_search_max_trials
        config.marginal_gain_scoring_enabled = settings.marginal_gain_scoring_enabled
        config.marginal_gain_delta_mf_weight = settings.marginal_gain_delta_mf_weight
        config.marginal_gain_delta_dcv_weight = settings.marginal_gain_delta_dcv_weight
        config.marginal_gain_spread_weight = settings.marginal_gain_spread_weight
        config.local_search_swap_trials = settings.local_search_swap_trials
        config.local_search_candidate_pool_size = settings.local_search_candidate_pool_size
        config.local_search_delta_mf_weight = settings.local_search_delta_mf_weight
        config.local_search_delta_dcv_weight = settings.local_search_delta_dcv_weight
        config.local_search_overlap_penalty_weight = settings.local_search_overlap_penalty_weight
        config.swap_candidate_pool_size = settings.swap_candidate_pool_size
        config.swap_prefilter_top_k = settings.swap_prefilter_top_k
        config.enable_swap_cache = settings.enable_swap_cache
        config.local_search_failed_patience = settings.local_search_failed_patience
        config.local_search_first_improvement = settings.local_search_first_improvement
        config.full_eval_top_k = settings.full_eval_top_k
        config.proxy_score_weights = settings.proxy_score_weights
        config.urgency_weight_enabled = settings.urgency_weight_enabled
        config.urgency_exponent = settings.urgency_exponent
        config.weak_group_focus_weight = settings.weak_group_focus_weight
        config.overlap_penalty_enabled = settings.overlap_penalty_enabled
        config.same_community_penalty_weight = settings.same_community_penalty_weight
        config.neighborhood_overlap_penalty_weight = settings.neighborhood_overlap_penalty_weight
    config.marginal_candidate_pool_size = settings.marginal_candidate_pool_size
    config.mutation_candidate_pool_size = settings.mutation_candidate_pool_size
    config.repair_candidate_pool_size = settings.repair_candidate_pool_size
    config.candidate_prefilter_top_k = settings.candidate_prefilter_top_k
    config.enable_fitness_cache = settings.enable_fitness_cache
    config.enable_marginal_cache = settings.enable_marginal_cache
    config.cache_max_size = settings.cache_max_size
    config.local_search_early_stop_patience = settings.local_search_early_stop_patience
    config.local_search_use_prefilter = settings.local_search_use_prefilter
    config.local_search_elite_count = settings.local_search_elite_count
    config.local_search_every_n_generations = settings.local_search_every_n_generations
    config.use_staged_mc = settings.use_staged_mc
    config.mc_runs_fast = settings.mc_runs_fast
    config.mc_runs_full = settings.mc_runs_full
    config.optimization_mode = settings.optimization_mode
    config.refinement_intensity = settings.refinement_intensity
    config.marginal_eval_fraction = settings.marginal_eval_fraction
    for key, value in overrides.items():
        setattr(config, key, value)
    if config.mc_runs_full > config.mc_runs:
        raise ValueError("mc_runs_full cannot exceed mc_runs_search for search-time optimization.")
    return config


def _selected_final_recheck_indices(result_frame: pd.DataFrame, top_k: int) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    for _, community_frame in result_frame.groupby("community_method", sort=False):
        ordered = community_frame.sort_values(
            ["f_score", "runtime_seconds", "method"],
            ascending=[False, True, True],
        ).reset_index()
        if top_k > 0:
            ordered = ordered.head(top_k)
        for rank, row in enumerate(ordered.itertuples(index=False), start=1):
            selected.append((int(row.index), rank))
    return selected


def _apply_final_recheck(
    result_frame: pd.DataFrame,
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    settings: ExperimentSettings,
) -> pd.DataFrame:
    if result_frame.empty or not settings.enable_final_recheck:
        return result_frame

    recheck_mc_runs = _resolved_final_recheck_mc_runs(settings)
    recheck_seed = _resolved_final_recheck_random_seed(settings)
    recheck_top_k = max(0, int(settings.final_recheck_top_k))
    updated_frame = result_frame.copy()

    for row_index, rank in _selected_final_recheck_indices(updated_frame, recheck_top_k):
        seed_set = tuple(json.loads(str(updated_frame.at[row_index, "seed_set"])))
        recheck_evaluation = evaluate_seed_set(
            dataset=dataset,
            protected_group_report=protected_group_report,
            seed_set=seed_set,
            propagation_probability=settings.propagation_probability,
            mc_runs=recheck_mc_runs,
            random_seed=recheck_seed,
            lambda_weight=settings.lambda_weight,
            include_soft_mf=True,
            diffusion_model=settings.diffusion_model,
        )
        updated_frame.at[row_index, "final_recheck_applied"] = True
        updated_frame.at[row_index, "final_recheck_top_k_rank"] = rank
        updated_frame.at[row_index, "final_recheck_mc_runs_used"] = recheck_mc_runs
        updated_frame.at[row_index, "final_recheck_random_seed"] = recheck_seed
        updated_frame.at[row_index, "final_recheck_total_spread"] = float(recheck_evaluation.total_spread_mean)
        updated_frame.at[row_index, "final_recheck_extra_spread"] = float(
            recheck_evaluation.total_spread_mean - float(settings.budget)
        )
        updated_frame.at[row_index, "final_recheck_mf"] = float(recheck_evaluation.fairness.mf)
        updated_frame.at[row_index, "final_recheck_dcv"] = float(recheck_evaluation.fairness.dcv)
        updated_frame.at[row_index, "final_recheck_f_score"] = float(recheck_evaluation.f_score)
        updated_frame.at[row_index, "final_recheck_runtime_seconds"] = float(recheck_evaluation.runtime_seconds)

    return updated_frame


def run_loaded_experiment(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    settings: ExperimentSettings,
    community_methods: list[str] | None = None,
    baseline_methods: list[str] | None = None,
    selected_methods: list[str] | None = None,
    include_ablations: bool = True,
) -> pd.DataFrame:
    """Run one or more method comparisons on a preloaded dataset."""

    validate_diffusion_model(settings.diffusion_model)
    _validate_registry_aliases(settings)
    mc_runs_search = _resolved_mc_runs_search(settings)
    mc_runs_eval = _resolved_mc_runs_eval(settings)
    resolved_ml_backends = _resolve_ml_backends(settings) if settings.use_ml else ()
    if settings.use_node2vec:
        raise ValueError("Node2Vec ML variants were removed. use_node2vec is no longer supported.")
    if settings.gnn_node2vec_mode != "off" and not any(
        backend in {"gnn", "gnn_ris"}
        for backend in resolved_ml_backends
    ):
        raise ValueError("gnn_node2vec_mode requires a GNN-capable backend such as 'gnn', 'gnn_ris', 'both', or 'all'.")
    if settings.use_ml and settings.ml_guidance_mode not in {"off", "two_tier"}:
        raise ValueError(
            "Removed ML guidance mode requested. Only ml_guidance_mode='two_tier' is supported for ML runs."
        )
    if settings.use_ml and any(backend in {"gnn", "gnn_ris"} for backend in resolved_ml_backends):
        require_gnn_dependencies()
    if settings.use_ml and any(backend in {"ris", "gnn_ris"} for backend in resolved_ml_backends):
        _ris_config(settings)
    if settings.graphsage_weight is not None and float(settings.graphsage_weight) < 0.0:
        raise ValueError("graphsage_weight must be non-negative when provided.")
    if float(settings.gnn_weight) < 0.0:
        raise ValueError("gnn_weight must be non-negative.")
    if float(settings.ris_weight) < 0.0:
        raise ValueError("ris_weight must be non-negative.")
    if float(settings.fair_ris_weight) < 0.0:
        raise ValueError("fair_ris_weight must be non-negative.")
    if float(settings.fairness_urgency_weight) < 0.0:
        raise ValueError("fairness_urgency_weight must be non-negative.")
    if float(settings.diversity_weight) < 0.0:
        raise ValueError("diversity_weight must be non-negative.")
    if any(
        [
            settings.compare_fairness_variants,
            settings.compare_refinement_variants,
            settings.compare_runtime_variants,
            settings.compare_swap_runtime_variants,
            settings.compare_scalability_variants,
        ]
    ):
        raise ValueError(
            "Obsolete ML comparison families were removed. "
            "Use "
            f"{_KEPT_ML_VARIANT_LABEL}, {_KEPT_GNN_ML_VARIANT_LABEL}, "
            f"{_KEPT_GNN_NODE2VEC_ML_VARIANT_LABEL}, {_KEPT_RIS_ML_VARIANT_LABEL}, "
            f"{_KEPT_GNN_RIS_ML_VARIANT_LABEL}, and/or {_KEPT_GNN_RIS_NODE2VEC_ML_VARIANT_LABEL} "
            "as the supported ML variants."
        )

    methods = community_methods or [settings.community_method]
    baselines = baseline_methods or ["degree", "pagerank", "community_round_robin", "random"]
    selected_method_set = {str(method) for method in selected_methods} if selected_methods else None
    if selected_method_set is not None:
        removed_requested = sorted(_REMOVED_ML_VARIANT_LABELS.intersection(selected_method_set))
        if removed_requested:
            removed_display = ", ".join(removed_requested)
            raise ValueError(
                "Removed ML variants are no longer supported: "
                f"{removed_display}. Supported ML variants: {_KEPT_ML_VARIANT_LABEL}, "
                f"{_KEPT_GNN_ML_VARIANT_LABEL}, {_KEPT_GNN_NODE2VEC_ML_VARIANT_LABEL}, "
                f"{_KEPT_RIS_ML_VARIANT_LABEL}, {_KEPT_GNN_RIS_ML_VARIANT_LABEL}, "
                f"{_KEPT_GNN_RIS_NODE2VEC_ML_VARIANT_LABEL}."
            )

    def is_selected(label: str) -> bool:
        return selected_method_set is None or label in selected_method_set

    def any_ml_methods_selected() -> bool:
        if selected_method_set is None:
            return True
        return any(method.startswith("hybrid_siea_ml_") for method in selected_method_set)

    results: list[dict[str, object]] = []

    for community_method in methods:
        community_embeddings, community_features, community_input_mode, community_method_config = (
            _resolve_community_clustering_inputs(dataset, settings, community_method)
        )
        community_result = detect_communities(
            dataset.graph,
            method=community_method,
            seed=settings.random_seed,
            embeddings=community_embeddings,
            features=community_features,
            input_mode=community_input_mode,
            config=community_method_config,
        )
        quality = compute_community_quality_metrics(dataset.graph, community_result)

        for baseline_name in baselines:
            if not is_selected(baseline_name):
                continue
            baseline_search_start = perf_counter()
            baseline_seed_set = select_baseline_seed_set(
                dataset=dataset,
                method=baseline_name,
                budget=settings.budget,
                protected_group_report=protected_group_report,
                propagation_probability=settings.propagation_probability,
                mc_runs=mc_runs_search,
                lambda_weight=settings.lambda_weight,
                community_result=community_result,
                random_seed=settings.random_seed,
                diffusion_model=settings.diffusion_model,
            )
            baseline_search_runtime = perf_counter() - baseline_search_start
            baseline_evaluation = _final_evaluate_seed_set(
                dataset=dataset,
                protected_group_report=protected_group_report,
                settings=settings,
                seed_set=baseline_seed_set,
            )
            results.append(
                _baseline_row(
                    dataset=dataset,
                    community_method=community_method,
                    method=str(baseline_name).lower(),
                    evaluation=baseline_evaluation,
                    budget=settings.budget,
                    search_runtime_seconds=baseline_search_runtime,
                    quality=quality,
                    diffusion_model=settings.diffusion_model,
                    mc_runs_search=mc_runs_search,
                    mc_runs_eval=mc_runs_eval,
                    community_result=community_result,
                )
            )

        hybrid_variants: list[tuple[str, str, str, dict[str, object]]] = [
            (
                "cea_fim",
                "comparator",
                "CEA-style baseline: community EA without swarm guidance or local search.",
                {"disable_swarm_guidance": True, "disable_local_search": True},
            ),
            ("hybrid_siea", "proposed", "", {}),
        ]
        if include_ablations:
            hybrid_variants.extend(
                [
                    ("hybrid_no_swarm", "ablation", "Ablation: swarm guidance disabled.", {"disable_swarm_guidance": True}),
                    ("hybrid_no_crossover", "ablation", "Ablation: crossover disabled.", {"disable_crossover": True}),
                    ("hybrid_no_local_search", "ablation", "Ablation: local search disabled.", {"disable_local_search": True}),
                    (
                        "hybrid_no_community_mutation",
                        "ablation",
                        "Ablation: community-aware mutation bonus disabled.",
                        {"disable_community_aware_mutation": True},
                    ),
                ]
            )

        hybrid_variants = [variant for variant in hybrid_variants if is_selected(variant[0])]
        for label, variant_type, note, overrides in hybrid_variants:
            config = _build_optimizer_config(settings, **overrides)
            optimizer = HybridSIEAOptimizer(
                dataset=dataset,
                protected_group_report=protected_group_report,
                community_result=community_result,
                config=config,
            )
            result = optimizer.optimize()
            final_evaluation = _final_evaluate_seed_set(
                dataset=dataset,
                protected_group_report=protected_group_report,
                settings=settings,
                seed_set=result.best_seed_set,
            )
            history_path = _history_path(
                settings.output_dir,
                dataset.name,
                settings.protected_attribute,
                settings.budget,
                community_method,
                label,
            )
            if history_path is not None:
                result.history.to_csv(history_path, index=False)
                note = "; ".join(part for part in [note, f"history={history_path.name}"] if part)
            results.append(
                _hybrid_row(
                    dataset=dataset,
                    community_method=community_method,
                    label=label,
                    variant_type=variant_type,
                    result=result,
                    evaluation=final_evaluation,
                    search_runtime_seconds=result.runtime_seconds,
                    config=config,
                    quality=quality,
                    diffusion_model=settings.diffusion_model,
                    mc_runs_search=mc_runs_search,
                    mc_runs_eval=mc_runs_eval,
                    note=note,
                    community_result=community_result,
                )
            )

        selected_ml_variants = [
            variant
            for variant in _selected_ml_variants(settings)
            if is_selected(variant[0])
        ]
        if settings.use_ml and selected_ml_variants and any_ml_methods_selected():
            feature_frames, feature_runtime = _compute_ml_feature_frame(
                dataset=dataset,
                protected_group_report=protected_group_report,
                community_result=community_result,
                settings=settings,
            )
            plain_feature_frame = feature_frames["plain"]
            node_ids = plain_feature_frame["node_id"].tolist()
            fairness_scores = (
                _static_fairness_urgency_scores(plain_feature_frame)
                if settings.fairness_urgency_weight > 0.0
                else None
            )
            diversity_scores = (
                _static_diversity_scores(plain_feature_frame)
                if settings.diversity_weight > 0.0
                else None
            )

            label_result: NodeUtilityLabelResult | None = None
            label_runtime = 0.0
            if any(variant[2] in {"tabular", "gnn", "gnn_ris"} for variant in selected_ml_variants):
                label_result, label_runtime = _generate_ml_labels(
                    dataset=dataset,
                    protected_group_report=protected_group_report,
                    settings=settings,
                )
            gnn_label_frame: pd.DataFrame | None = None
            gnn_label_runtime = 0.0
            gnn_label_variance: float | None = None
            if any(variant[2] in {"gnn", "gnn_ris"} for variant in selected_ml_variants):
                if label_result is None:
                    raise ValueError("label_result was not prepared for the selected GNN variant.")
                gnn_label_frame, gnn_label_runtime = _build_gnn_label_frame(
                    dataset=dataset,
                    protected_group_report=protected_group_report,
                    community_result=community_result,
                    settings=settings,
                    label_result=label_result,
                )
                gnn_label_variance = float(gnn_label_frame["label_score"].var(ddof=0))
            training_results_by_variant: dict[tuple[str, str], tuple[RankingTrainingResult, float]] = {}
            for _, _, ml_backend, node2vec_mode, _ in selected_ml_variants:
                variant_key = (ml_backend, node2vec_mode)
                if variant_key in training_results_by_variant:
                    continue
                if ml_backend == "tabular":
                    if label_result is None:
                        raise ValueError("label_result was not prepared for the selected tabular variant.")
                    training_results_by_variant[variant_key] = _prepare_tabular_ml_training(
                        dataset=dataset,
                        feature_frame=plain_feature_frame,
                        settings=settings,
                        label_result=label_result,
                    )
                    continue
                if ml_backend not in {"gnn", "gnn_ris"}:
                    continue
                if gnn_label_frame is None:
                    raise ValueError("gnn_label_frame was not prepared for the selected GNN variant.")
                feature_key = "input_concat" if node2vec_mode == "input_concat" else "plain"
                training_results_by_variant[variant_key] = _prepare_gnn_training(
                    dataset=dataset,
                    feature_frame=feature_frames[feature_key],
                    community_method=community_method,
                    settings=settings,
                    label_frame=gnn_label_frame,
                    target_column="label_score",
                    node2vec_mode=node2vec_mode,
                )

            shared_ris_result: RISGuidanceResult | None = None
            shared_ris_runtime = 0.0
            if settings.ris_reuse_rr_sets and any(variant[4] for variant in selected_ml_variants):
                shared_ris_result, shared_ris_runtime = _prepare_ris_guidance(
                    dataset=dataset,
                    protected_group_report=protected_group_report,
                    settings=settings,
                    feature_frame=plain_feature_frame,
                )

            for label, guidance_mode, ml_backend, node2vec_mode, uses_ris in selected_ml_variants:
                guidance_scores: dict[object, float]
                backend_preparation_runtime = feature_runtime
                validation_spearman: float | object = pd.NA
                validation_precision_at_budget: float | object = pd.NA
                gnn_model_type: str | object = pd.NA
                guidance_note_parts = [
                    f"backend={ml_backend}",
                    (
                        f"weights=(graphsage={_effective_gnn_weight(settings):.3f}, "
                        f"ris={settings.ris_weight:.3f}, fair_ris={settings.fair_ris_weight:.3f}, "
                        f"fairness={settings.fairness_urgency_weight:.3f}, diversity={settings.diversity_weight:.3f})"
                    ),
                ]

                if ml_backend == "tabular":
                    training_result, backend_training_runtime = training_results_by_variant[(ml_backend, node2vec_mode)]
                    backend_preparation_runtime += label_runtime + backend_training_runtime
                    guidance_scores = dict(training_result.predicted_scores)
                    validation_spearman = training_result.validation_spearman
                    validation_precision_at_budget = training_result.validation_precision_at_budget
                    if label_result is not None:
                        guidance_note_parts.append(f"label_variance={label_result.label_variance:.6f}")
                    guidance_note_parts.append(f"model_type={settings.ml_model_type}")
                else:
                    base_gnn_scores: dict[object, float] | None = None
                    if ml_backend in {"gnn", "gnn_ris"}:
                        training_result, backend_training_runtime = training_results_by_variant[(ml_backend, node2vec_mode)]
                        backend_preparation_runtime += label_runtime + gnn_label_runtime + backend_training_runtime
                        base_gnn_scores = dict(training_result.predicted_scores)
                        validation_spearman = training_result.validation_spearman
                        validation_precision_at_budget = training_result.validation_precision_at_budget
                        gnn_model_type = settings.gnn_model_type
                        cache_status = "hit" if training_result.loaded_from_cache else "miss"
                        guidance_note_parts.extend(
                            [
                                "target=label_score",
                                (
                                    f"target_variance={float(gnn_label_variance):.6f}"
                                    if gnn_label_variance is not None
                                    else "target_variance=nan"
                                ),
                                f"gnn_model_type={settings.gnn_model_type}",
                                f"node2vec_mode={node2vec_mode}",
                                f"feature_shape={training_result.feature_matrix_shape}",
                                f"edge_index_shape={training_result.edge_index_shape}",
                                f"cache={cache_status}",
                            ]
                        )
                        if label_result is not None:
                            guidance_note_parts.append(f"label_variance={label_result.label_variance:.6f}")

                    ris_scores: dict[object, float] | None = None
                    fair_ris_scores: dict[object, float] | None = None
                    if uses_ris:
                        ris_result = shared_ris_result
                        ris_runtime = shared_ris_runtime
                        if ris_result is None:
                            ris_result, ris_runtime = _prepare_ris_guidance(
                                dataset=dataset,
                                protected_group_report=protected_group_report,
                                settings=settings,
                                feature_frame=plain_feature_frame,
                            )
                        backend_preparation_runtime += ris_runtime
                        global_ris_scores = dict(ris_result.global_node_scores)
                        weighted_ris_scores = _build_fair_ris_scores(
                            ris_result=ris_result,
                            feature_frame=plain_feature_frame,
                            protected_group_report=protected_group_report,
                        )
                        fair_ris_scores = weighted_ris_scores if settings.fair_ris_weight > 0.0 else None
                        if settings.fair_ris_weight > 0.0:
                            ris_scores = global_ris_scores
                        else:
                            ris_scores = (
                                weighted_ris_scores
                                if settings.ris_mode == "weak_group_weighted"
                                else global_ris_scores
                            )
                        guidance_note_parts.extend(
                            [
                                f"ris_num_rr_sets={settings.ris_num_rr_sets}",
                                f"ris_mode={settings.ris_mode}",
                                f"fair_ris_enabled={settings.fair_ris_weight > 0.0}",
                                f"ris_reuse_rr_sets={settings.ris_reuse_rr_sets}",
                            ]
                        )

                    guidance_scores = _build_guidance_score_map(
                        node_ids=node_ids,
                        settings=settings,
                        gnn_scores=base_gnn_scores,
                        ris_scores=ris_scores,
                        fair_ris_scores=fair_ris_scores,
                        fairness_scores=fairness_scores,
                        diversity_scores=diversity_scores,
                    )

                ml_config = _build_optimizer_config(
                    settings,
                    include_fairness_parameters=False,
                    ml_guidance_mode="two_tier",
                    **_resolved_kept_ml_overrides(settings),
                )
                optimizer_kwargs: dict[str, object] = {
                    "dataset": dataset,
                    "protected_group_report": protected_group_report,
                    "community_result": community_result,
                    "config": ml_config,
                    "ml_node_scores": guidance_scores,
                }
                pool_note = f"full_pool={dataset.graph.number_of_nodes()}; tier_policy=tuned"

                ml_optimizer = HybridSIEAOptimizer(**optimizer_kwargs)
                ml_result = ml_optimizer.optimize()
                ml_final_evaluation = _final_evaluate_seed_set(
                    dataset=dataset,
                    protected_group_report=protected_group_report,
                    settings=settings,
                    seed_set=ml_result.best_seed_set,
                )
                ml_history_path = _history_path(
                    settings.output_dir,
                    dataset.name,
                    settings.protected_attribute,
                    settings.budget,
                    community_method,
                    label,
                )
                ml_row = _hybrid_row(
                    dataset=dataset,
                    community_method=community_method,
                    label=label,
                    variant_type="ml_guided",
                    result=ml_result,
                    evaluation=ml_final_evaluation,
                    search_runtime_seconds=(
                        backend_preparation_runtime
                        + ml_result.runtime_seconds
                    ),
                    config=ml_config,
                    quality=quality,
                    diffusion_model=settings.diffusion_model,
                    mc_runs_search=mc_runs_search,
                    mc_runs_eval=mc_runs_eval,
                    note="; ".join([f"ML guidance mode={guidance_mode}", pool_note, *guidance_note_parts]),
                    node2vec_enabled=node2vec_mode == "input_concat" and ml_backend in {"gnn", "gnn_ris"},
                    node2vec_mode=node2vec_mode,
                    community_result=community_result,
                )
                if ml_history_path is not None:
                    ml_result.history.to_csv(ml_history_path, index=False)
                    ml_row["note"] = "; ".join(
                        part for part in [str(ml_row["note"]), f"history={ml_history_path.name}"] if part
                    )
                ml_row["ml_validation_spearman"] = validation_spearman
                ml_row["ml_validation_precision_at_budget"] = validation_precision_at_budget
                ml_row["ml_guidance_mode"] = "two_tier"
                ml_row["guidance_mode"] = ml_backend
                ml_row["ml_backend"] = ml_backend
                ml_row["gnn_model_type"] = gnn_model_type
                ml_row["graphsage_enabled"] = _graphsage_enabled_flag(gnn_model_type, ml_backend=ml_backend)
                ml_row["ris_enabled"] = bool(uses_ris)
                ml_row["ris_mode"] = settings.ris_mode if uses_ris else "off"
                ml_row["clustering_method"] = "none"
                ml_row["clustering_input_mode"] = "none"
                ml_row["embedding_method"] = (
                    settings.gnn_model_type
                    if ml_backend in {"gnn", "gnn_ris"}
                    else "none"
                )
                if ml_backend == "tabular":
                    ml_row["ranking_model"] = settings.ml_model_type
                elif ml_backend in {"gnn", "gnn_ris"}:
                    ml_row["ranking_model"] = gnn_model_type or settings.gnn_model_type
                elif ml_backend == "ris":
                    ml_row["ranking_model"] = "ris_guidance"
                else:
                    ml_row["ranking_model"] = "none"
                ml_row["search_spread_estimator"] = "monte_carlo"
                ml_row["search_guidance_estimator"] = "ris_guidance" if uses_ris else "none"
                ml_row["final_spread_estimator"] = "monte_carlo"
                results.append(ml_row)

    result_frame = pd.DataFrame(results)
    if selected_method_set is not None and result_frame.empty:
        requested = ", ".join(sorted(selected_method_set))
        raise ValueError(
            "No methods matched the selected filter. "
            f"Requested: {requested}. "
            f"Supported ML variants: {_KEPT_ML_VARIANT_LABEL}, "
            f"{_KEPT_GNN_ML_VARIANT_LABEL}, {_KEPT_GNN_NODE2VEC_ML_VARIANT_LABEL}, "
            f"{_KEPT_RIS_ML_VARIANT_LABEL}, {_KEPT_GNN_RIS_ML_VARIANT_LABEL}, "
            f"{_KEPT_GNN_RIS_NODE2VEC_ML_VARIANT_LABEL}."
        )
    if not result_frame.empty:
        result_frame["protected_attribute"] = settings.protected_attribute
        result_frame["requested_spread_estimator"] = settings.spread_estimator
        result_frame["debias_mode"] = settings.debias_mode
    result_frame["comparison_baseline_method"] = "hybrid_siea"
    result_frame["delta_f_score"] = pd.NA
    if not result_frame.empty:
        for community_method in result_frame["community_method"].dropna().unique():
            community_mask = result_frame["community_method"] == community_method
            baseline_method = "hybrid_siea"
            baseline_rows = result_frame.loc[community_mask & (result_frame["method"] == baseline_method)]
            if baseline_rows.empty:
                continue
            baseline_f_score = float(baseline_rows.iloc[0]["f_score"])
            result_frame.loc[community_mask, "comparison_baseline_method"] = baseline_method
            result_frame.loc[community_mask, "delta_f_score"] = (
                result_frame.loc[community_mask, "f_score"].astype(float) - baseline_f_score
            )
    result_frame = _apply_final_recheck(
        result_frame=result_frame,
        dataset=dataset,
        protected_group_report=protected_group_report,
        settings=settings,
    )
    attribute_dir = _ensure_results_output_dir(settings.output_dir, dataset.name, settings.protected_attribute)
    if attribute_dir is not None:
        output_path = attribute_dir / f"{dataset.name}_budget{settings.budget}_results.csv"
        result_frame.to_csv(output_path, index=False)
    return result_frame


def run_experiment(
    dataset_config: DatasetConfig,
    settings: ExperimentSettings,
    community_methods: list[str] | None = None,
    baseline_methods: list[str] | None = None,
    selected_methods: list[str] | None = None,
    include_ablations: bool = True,
) -> pd.DataFrame:
    """Load a dataset and run the comparison experiment."""

    dataset = load_dataset(dataset_config)
    derived_group_method = settings.derived_group_method or settings.community_method
    protected_group_report = verify_protected_groups(
        dataset,
        settings.protected_attribute,
        derive_protected_groups=settings.derive_protected_groups,
        derived_group_method=derived_group_method,
        random_seed=settings.random_seed,
    )
    return run_loaded_experiment(
        dataset=dataset,
        protected_group_report=protected_group_report,
        settings=settings,
        community_methods=community_methods,
        baseline_methods=baseline_methods,
        selected_methods=selected_methods,
        include_ablations=include_ablations,
    )
