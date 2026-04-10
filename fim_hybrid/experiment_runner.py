"""Experiment runner for fair comparison across FIM methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from time import perf_counter

import pandas as pd

from .baselines import BaselineResult, run_baseline
from .community_detection import CommunityQualityMetrics, compute_community_quality_metrics, detect_communities
from .config import DatasetConfig
from .data_loader import LoadedDataset, ProtectedGroupReport, load_dataset, verify_protected_groups
from .diffusion import DEFAULT_DIFFUSION_MODEL, validate_diffusion_model
from .evaluation import SeedSetEvaluation
from .feature_extraction import compute_node_features
from .hybrid_optimizer import HybridOptimizationResult, HybridSIEAConfig, HybridSIEAOptimizer
from .label_generation import NodeUtilityLabelResult, generate_singleton_node_utility_labels
from .ml_training import MLTrainingResult, train_node_utility_model
_KEPT_ML_VARIANT_LABEL = "hybrid_siea_ml_two_tier_tuned_swap_local_search"
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

_RUNTIME_COMPARE_DEFAULTS: dict[str, object] = {
    "enable_fitness_cache": True,
    "enable_marginal_cache": True,
    "cache_max_size": 4096,
    "marginal_candidate_pool_size": 24,
    "mutation_candidate_pool_size": 12,
    "local_search_max_trials": 8,
    "local_search_early_stop_patience": 4,
    "local_search_use_prefilter": True,
    "optimization_mode": "full",
    "refinement_intensity": 1.0,
    "marginal_eval_fraction": 1.0,
}

_SWAP_RUNTIME_COMPARE_DEFAULTS: dict[str, object] = {
    "enable_fitness_cache": True,
    "enable_swap_cache": True,
    "cache_max_size": 4096,
    "swap_candidate_pool_size": 6,
    "swap_prefilter_top_k": 12,
    "local_search_failed_patience": 4,
    "local_search_first_improvement": False,
    "full_eval_top_k": 4,
    "proxy_score_weights": {},
}

_SCALABILITY_COMPARE_DEFAULTS: dict[str, object] = {
    "enable_fitness_cache": True,
    "enable_swap_cache": True,
    "enable_marginal_cache": True,
    "cache_max_size": 8192,
    "local_search_use_prefilter": True,
    "use_staged_mc": True,
}


@dataclass(slots=True)
class ExperimentSettings:
    """Explicit experiment settings for method comparison."""

    protected_attribute: str
    budget: int
    diffusion_model: str = DEFAULT_DIFFUSION_MODEL
    community_method: str = "leiden"
    propagation_probability: float = 0.01
    mc_runs: int = 20
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
    ml_guidance_mode: str = "off"
    ml_top_fraction: float | None = 0.25
    ml_top_n: int | None = None
    ml_max_nodes: int | None = None
    ml_singleton_runs: int = 15
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
    validation_spearman: float | None = None,
    validation_precision_at_budget: float | None = None,
) -> dict[str, float | None]:
    return {
        "ml_guidance_mode": guidance_mode,
        "ml_validation_spearman": validation_spearman,
        "ml_validation_precision_at_budget": validation_precision_at_budget,
    }


def _baseline_row(
    dataset: LoadedDataset,
    community_method: str,
    result: BaselineResult,
    quality: CommunityQualityMetrics,
    diffusion_model: str,
) -> dict[str, object]:
    return {
        "dataset": dataset.name,
        "community_method": community_method,
        "diffusion_model": diffusion_model,
        "method": result.method,
        "variant_type": "baseline",
        "seed_set": json.dumps(list(result.seed_set)),
        "total_spread": result.total_spread_mean,
        "mf": result.mf,
        "dcv": result.dcv,
        "f_score": result.f_score,
        "runtime_seconds": result.runtime_seconds,
        "candidate_pool_size": dataset.graph.number_of_nodes(),
        "optimization_mode": "full",
        "swarm_guidance": False,
        "crossover": False,
        "local_search": False,
        "community_aware_mutation": result.method == "community_round_robin",
        "node2vec_enabled": False,
        "node2vec_mode": "off",
        "note": "",
        **_ml_columns(guidance_mode="off"),
        **_fairness_diagnostic_columns(result.group_spread, result.normalized_group_spread),
        **_community_columns(quality),
    }


def _hybrid_row(
    dataset: LoadedDataset,
    community_method: str,
    label: str,
    variant_type: str,
    result: HybridOptimizationResult,
    config: HybridSIEAConfig,
    quality: CommunityQualityMetrics,
    diffusion_model: str,
    note: str = "",
    node2vec_enabled: bool = False,
    node2vec_mode: str = "off",
) -> dict[str, object]:
    return {
        "dataset": dataset.name,
        "community_method": community_method,
        "diffusion_model": diffusion_model,
        "method": label,
        "variant_type": variant_type,
        "seed_set": json.dumps(list(result.best_seed_set)),
        "total_spread": result.best_spread,
        "mf": result.best_fairness.mf,
        "dcv": result.best_fairness.dcv,
        "f_score": result.best_score,
        "runtime_seconds": result.runtime_seconds,
        "candidate_pool_size": result.candidate_pool_size,
        "optimization_mode": config.optimization_mode,
        "swarm_guidance": not config.disable_swarm_guidance,
        "crossover": not config.disable_crossover,
        "local_search": not config.disable_local_search,
        "community_aware_mutation": not config.disable_community_aware_mutation,
        "node2vec_enabled": node2vec_enabled,
        "node2vec_mode": node2vec_mode,
        "note": note,
        **_ml_columns(guidance_mode=config.ml_guidance_mode),
        **_fairness_diagnostic_columns(
            result.best_fairness.group_spread,
            result.best_fairness.normalized_group_spread,
        ),
        **_community_columns(quality),
    }


def _ml_row(
    dataset: LoadedDataset,
    community_method: str,
    label: str,
    variant_type: str,
    evaluation: SeedSetEvaluation,
    runtime_seconds: float,
    quality: CommunityQualityMetrics,
    candidate_pool_size: int,
    validation_spearman: float,
    validation_precision_at_budget: float,
    guidance_mode: str,
    diffusion_model: str,
    note: str = "",
    node2vec_enabled: bool = False,
    node2vec_mode: str = "off",
) -> dict[str, object]:
    return {
        "dataset": dataset.name,
        "community_method": community_method,
        "diffusion_model": diffusion_model,
        "method": label,
        "variant_type": variant_type,
        "seed_set": json.dumps(list(evaluation.seed_set)),
        "total_spread": evaluation.total_spread_mean,
        "mf": evaluation.fairness.mf,
        "dcv": evaluation.fairness.dcv,
        "f_score": evaluation.f_score,
        "runtime_seconds": runtime_seconds,
        "candidate_pool_size": candidate_pool_size,
        "optimization_mode": "full",
        "swarm_guidance": False,
        "crossover": False,
        "local_search": False,
        "community_aware_mutation": False,
        "node2vec_enabled": node2vec_enabled,
        "node2vec_mode": node2vec_mode,
        "note": note,
        **_ml_columns(
            guidance_mode=guidance_mode,
            validation_spearman=validation_spearman,
            validation_precision_at_budget=validation_precision_at_budget,
        ),
        **_fairness_diagnostic_columns(
            evaluation.fairness.group_spread,
            evaluation.fairness.normalized_group_spread,
        ),
        **_community_columns(quality),
    }


def _build_node2vec_config(settings: ExperimentSettings) -> Node2VecConfig:
    return Node2VecConfig(
        dimensions=settings.node2vec_dimensions,
        walk_length=settings.node2vec_walk_length,
        num_walks=settings.node2vec_num_walks,
        window=settings.node2vec_window,
        p=settings.node2vec_p,
        q=settings.node2vec_q,
        scale_embeddings=settings.node2vec_scale_embeddings,
        pca_components=settings.node2vec_pca_components,
        random_seed=settings.random_seed,
    )


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


def _prepare_ml_training(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    community_result,
    settings: ExperimentSettings,
    label_result: NodeUtilityLabelResult,
) -> tuple[MLTrainingResult, float]:
    start = perf_counter()
    feature_frame = compute_node_features(
        dataset=dataset,
        protected_group_report=protected_group_report,
        community_result=community_result,
    )
    training_result = train_node_utility_model(
        feature_frame=feature_frame,
        label_frame=label_result.label_frame,
        budget=settings.budget,
        model_type=settings.ml_model_type,
        top_fraction=settings.ml_top_fraction,
        top_n=settings.ml_top_n,
        max_nodes=settings.ml_max_nodes,
        random_seed=settings.random_seed,
    )
    return training_result, perf_counter() - start


def _selected_ml_variants(settings: ExperimentSettings) -> list[tuple[str, str, bool]]:
    if not settings.use_ml:
        return []
    return [(_KEPT_ML_VARIANT_LABEL, "two_tier", False)]


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


def _selected_fairness_variants(settings: ExperimentSettings) -> list[tuple[str, str, dict[str, object]]]:
    fairness_defaults = _resolved_fairness_compare_settings(settings)
    return [
        (
            "hybrid_siea_ml_two_tier_tuned_fair_init",
            "Fairness-first initialization only.",
            {
                "fairness_first_init_enabled": bool(fairness_defaults["fairness_first_init_enabled"]),
                "fairness_first_init_slots": int(fairness_defaults["fairness_first_init_slots"]),
                "fairness_first_init_weight": float(fairness_defaults["fairness_first_init_weight"]),
            },
        ),
        (
            "hybrid_siea_ml_two_tier_tuned_weak_mutation",
            "Weakest-group-aware mutation only.",
            {
                "weakest_group_k": int(fairness_defaults["weakest_group_k"]),
                "weakest_group_mutation_weight": float(fairness_defaults["weakest_group_mutation_weight"]),
                "zero_group_bonus_weight": float(fairness_defaults["zero_group_bonus_weight"]),
                "bridge_to_weak_group_weight": float(fairness_defaults["bridge_to_weak_group_weight"]),
            },
        ),
        (
            "hybrid_siea_ml_two_tier_tuned_fair_repair",
            "Fairness-aware repair only.",
            {
                "weakest_group_k": int(fairness_defaults["weakest_group_k"]),
                "repair_fairness_weight": float(fairness_defaults["repair_fairness_weight"]),
                "repair_bridge_weight": float(fairness_defaults["repair_bridge_weight"]),
                "repair_centrality_weight": float(fairness_defaults["repair_centrality_weight"]),
                "repair_diversity_weight": float(fairness_defaults["repair_diversity_weight"]),
            },
        ),
        (
            "hybrid_siea_ml_two_tier_tuned_worst_group_local_search",
            "Worst-group-focused local search only.",
            {
                "weakest_group_k": int(fairness_defaults["weakest_group_k"]),
                "local_search_focus_mode": str(fairness_defaults["local_search_focus_mode"]),
                "local_search_bottom_k_groups": int(fairness_defaults["local_search_bottom_k_groups"]),
                "local_search_max_trials": int(fairness_defaults["local_search_max_trials"]),
                "weakest_group_mutation_weight": float(fairness_defaults["weakest_group_mutation_weight"]),
                "zero_group_bonus_weight": float(fairness_defaults["zero_group_bonus_weight"]),
                "bridge_to_weak_group_weight": float(fairness_defaults["bridge_to_weak_group_weight"]),
            },
        ),
        (
            "hybrid_siea_ml_two_tier_tuned_fairness_full",
            "Combined fairness-targeted initialization, mutation, repair, and local search.",
            fairness_defaults,
        ),
    ]


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


def _selected_refinement_variants(settings: ExperimentSettings) -> list[tuple[str, str, dict[str, object]]]:
    refinement_defaults = _resolved_refinement_compare_settings(settings)
    return [
        (
            "hybrid_siea_ml_two_tier_tuned_marginal_gain",
            "Current best method plus marginal fairness-gain candidate scoring.",
            {
                "marginal_gain_scoring_enabled": True,
                "marginal_gain_delta_mf_weight": float(refinement_defaults["marginal_gain_delta_mf_weight"]),
                "marginal_gain_delta_dcv_weight": float(refinement_defaults["marginal_gain_delta_dcv_weight"]),
                "marginal_gain_spread_weight": float(refinement_defaults["marginal_gain_spread_weight"]),
            },
        ),
        (
            "hybrid_siea_ml_two_tier_tuned_swap_local_search",
            "Current best method plus swap-focused local search refinement.",
            {
                "local_search_swap_trials": int(refinement_defaults["local_search_swap_trials"]),
                "local_search_candidate_pool_size": int(refinement_defaults["local_search_candidate_pool_size"]),
                "local_search_delta_mf_weight": float(refinement_defaults["local_search_delta_mf_weight"]),
                "local_search_delta_dcv_weight": float(refinement_defaults["local_search_delta_dcv_weight"]),
                "local_search_overlap_penalty_weight": float(refinement_defaults["local_search_overlap_penalty_weight"]),
            },
        ),
        (
            "hybrid_siea_ml_two_tier_tuned_urgency_weighted",
            "Current best method plus urgency-weighted marginal scoring toward weak groups.",
            {
                "marginal_gain_scoring_enabled": True,
                "marginal_gain_delta_mf_weight": float(refinement_defaults["marginal_gain_delta_mf_weight"]),
                "marginal_gain_delta_dcv_weight": float(refinement_defaults["marginal_gain_delta_dcv_weight"]),
                "marginal_gain_spread_weight": float(refinement_defaults["marginal_gain_spread_weight"]),
                "urgency_weight_enabled": True,
                "urgency_exponent": float(refinement_defaults["urgency_exponent"]),
                "weak_group_focus_weight": float(refinement_defaults["weak_group_focus_weight"]),
            },
        ),
        (
            "hybrid_siea_ml_two_tier_tuned_overlap_penalty",
            "Current best method plus overlap-aware candidate penalties.",
            {
                "overlap_penalty_enabled": True,
                "same_community_penalty_weight": float(refinement_defaults["same_community_penalty_weight"]),
                "neighborhood_overlap_penalty_weight": float(refinement_defaults["neighborhood_overlap_penalty_weight"]),
            },
        ),
        (
            "hybrid_siea_ml_two_tier_tuned_refinement_full",
            "Current best method plus marginal scoring, swap local search, urgency weighting, and overlap penalties.",
            refinement_defaults,
        ),
    ]


def _resolved_runtime_compare_settings(settings: ExperimentSettings) -> dict[str, object]:
    resolved = dict(_RUNTIME_COMPARE_DEFAULTS)
    for field_name in [
        "marginal_candidate_pool_size",
        "mutation_candidate_pool_size",
        "cache_max_size",
        "local_search_early_stop_patience",
        "refinement_intensity",
        "marginal_eval_fraction",
    ]:
        current_value = getattr(settings, field_name)
        resolved[field_name] = current_value if current_value not in {0, 0.0} else _RUNTIME_COMPARE_DEFAULTS[field_name]
    resolved["enable_fitness_cache"] = settings.enable_fitness_cache
    resolved["enable_marginal_cache"] = (
        settings.enable_marginal_cache or bool(_RUNTIME_COMPARE_DEFAULTS["enable_marginal_cache"])
    )
    resolved["local_search_use_prefilter"] = (
        settings.local_search_use_prefilter or bool(_RUNTIME_COMPARE_DEFAULTS["local_search_use_prefilter"])
    )
    resolved["optimization_mode"] = (
        settings.optimization_mode
        if settings.optimization_mode != "full"
        else str(_RUNTIME_COMPARE_DEFAULTS["optimization_mode"])
    )
    return resolved


def _selected_runtime_variants(settings: ExperimentSettings) -> list[tuple[str, str, dict[str, object]]]:
    runtime_defaults = _resolved_runtime_compare_settings(settings)
    return [
        (
            "hybrid_siea_ml_two_tier_tuned_marginal_gain",
            "Current best marginal-gain method without runtime pruning.",
            {},
        ),
        (
            "hybrid_siea_ml_two_tier_tuned_marginal_gain_optimized",
            "Runtime-optimized marginal-gain method with bounded candidate shortlists and caches.",
            runtime_defaults,
        ),
        (
            "hybrid_siea_ml_two_tier_tuned_marginal_gain_balanced",
            "Balanced mode: lighter refinement budgets with most fairness guidance preserved.",
            {
                "enable_fitness_cache": bool(runtime_defaults["enable_fitness_cache"]),
                "enable_marginal_cache": bool(runtime_defaults["enable_marginal_cache"]),
                "cache_max_size": int(runtime_defaults["cache_max_size"]),
                "local_search_use_prefilter": bool(runtime_defaults["local_search_use_prefilter"]),
                "optimization_mode": "balanced",
                "refinement_intensity": float(runtime_defaults["refinement_intensity"]),
                "marginal_eval_fraction": float(runtime_defaults["marginal_eval_fraction"]),
            },
        ),
        (
            "hybrid_siea_ml_two_tier_tuned_marginal_gain_fast",
            "Fast mode: stronger pruning and lower refinement intensity for lower runtime.",
            {
                "enable_fitness_cache": bool(runtime_defaults["enable_fitness_cache"]),
                "enable_marginal_cache": bool(runtime_defaults["enable_marginal_cache"]),
                "cache_max_size": int(runtime_defaults["cache_max_size"]),
                "local_search_use_prefilter": bool(runtime_defaults["local_search_use_prefilter"]),
                "optimization_mode": "fast",
                "refinement_intensity": float(runtime_defaults["refinement_intensity"]),
                "marginal_eval_fraction": float(runtime_defaults["marginal_eval_fraction"]),
            },
        ),
    ]


def _resolved_swap_runtime_compare_settings(settings: ExperimentSettings) -> dict[str, object]:
    resolved = dict(_SWAP_RUNTIME_COMPARE_DEFAULTS)
    for field_name in [
        "cache_max_size",
        "swap_candidate_pool_size",
        "swap_prefilter_top_k",
        "local_search_failed_patience",
        "full_eval_top_k",
    ]:
        current_value = getattr(settings, field_name)
        resolved[field_name] = current_value if current_value not in {0, 0.0} else _SWAP_RUNTIME_COMPARE_DEFAULTS[field_name]
    resolved["enable_fitness_cache"] = settings.enable_fitness_cache
    resolved["enable_swap_cache"] = (
        settings.enable_swap_cache or bool(_SWAP_RUNTIME_COMPARE_DEFAULTS["enable_swap_cache"])
    )
    resolved["local_search_first_improvement"] = settings.local_search_first_improvement
    resolved["proxy_score_weights"] = (
        settings.proxy_score_weights
        if settings.proxy_score_weights
        else dict(_SWAP_RUNTIME_COMPARE_DEFAULTS["proxy_score_weights"])
    )
    return resolved


def _selected_swap_runtime_variants(settings: ExperimentSettings) -> list[tuple[str, str, dict[str, object]]]:
    runtime_defaults = _resolved_swap_runtime_compare_settings(settings)
    return [
        (
            "hybrid_siea_ml_two_tier_tuned_swap_local_search",
            "Current swap-local-search method without extra runtime pruning.",
            {},
        ),
        (
            "hybrid_siea_ml_two_tier_tuned_swap_local_search_optimized",
            "Swap-local-search runtime optimization with shortlist pruning, swap caching, and top-k full evaluation.",
            runtime_defaults,
        ),
        (
            "hybrid_siea_ml_two_tier_tuned_swap_local_search_first_improvement",
            "Swap-local-search runtime optimization with first-improvement stopping.",
            {
                **runtime_defaults,
                "local_search_first_improvement": True,
            },
        ),
        (
            "hybrid_siea_ml_two_tier_tuned_swap_local_search_reduced_candidates",
            "Swap-local-search runtime optimization with a smaller candidate shortlist and top-k full evaluation budget.",
            {
                **runtime_defaults,
                "swap_candidate_pool_size": 4,
                "swap_prefilter_top_k": 8,
                "full_eval_top_k": 2,
                "local_search_failed_patience": 2,
                "proxy_score_weights": {
                    "delta_mf": 0.60,
                    "delta_dcv": 0.45,
                    "overlap": 1.0,
                    "ml": 0.25,
                    "bridge": 0.25,
                },
            },
        ),
    ]


def _resolved_scalability_compare_settings(settings: ExperimentSettings) -> dict[str, object]:
    fairness_defaults = _resolved_fairness_compare_settings(settings)
    resolved = {
        **fairness_defaults,
        **_SCALABILITY_COMPARE_DEFAULTS,
    }
    full_mc_runs = settings.mc_runs_full if settings.mc_runs_full > 0 else settings.mc_runs
    if full_mc_runs <= 1:
        fast_mc_runs = 1
    elif settings.mc_runs_fast > 0:
        fast_mc_runs = settings.mc_runs_fast
    else:
        fast_mc_runs = max(1, min(full_mc_runs - 1, int(math.ceil(full_mc_runs * 0.35))))

    resolved["enable_fitness_cache"] = settings.enable_fitness_cache
    resolved["enable_swap_cache"] = settings.enable_swap_cache or bool(_SCALABILITY_COMPARE_DEFAULTS["enable_swap_cache"])
    resolved["enable_marginal_cache"] = settings.enable_marginal_cache or bool(_SCALABILITY_COMPARE_DEFAULTS["enable_marginal_cache"])
    resolved["cache_max_size"] = settings.cache_max_size if settings.cache_max_size > 0 else int(_SCALABILITY_COMPARE_DEFAULTS["cache_max_size"])
    resolved["mutation_candidate_pool_size"] = (
        settings.mutation_candidate_pool_size
        if settings.mutation_candidate_pool_size > 0
        else max(12, settings.budget)
    )
    resolved["repair_candidate_pool_size"] = (
        settings.repair_candidate_pool_size
        if settings.repair_candidate_pool_size > 0
        else max(16, settings.budget * 2)
    )
    resolved["swap_candidate_pool_size"] = (
        settings.swap_candidate_pool_size
        if settings.swap_candidate_pool_size > 0
        else max(8, max(4, settings.budget // 2))
    )
    resolved["candidate_prefilter_top_k"] = (
        settings.candidate_prefilter_top_k
        if settings.candidate_prefilter_top_k > 0
        else max(24, settings.budget * 2)
    )
    resolved["local_search_elite_count"] = (
        settings.local_search_elite_count
        if settings.local_search_elite_count > 0
        else max(2, int(math.ceil(settings.population_size * 0.5)))
    )
    resolved["local_search_every_n_generations"] = max(1, settings.local_search_every_n_generations)
    resolved["local_search_use_prefilter"] = (
        settings.local_search_use_prefilter
        or bool(_SCALABILITY_COMPARE_DEFAULTS["local_search_use_prefilter"])
    )
    resolved["local_search_failed_patience"] = (
        settings.local_search_failed_patience
        if settings.local_search_failed_patience > 0
        else max(3, settings.local_search_max_trials if settings.local_search_max_trials > 0 else 4)
    )
    resolved["full_eval_top_k"] = (
        settings.full_eval_top_k
        if settings.full_eval_top_k > 0
        else max(3, min(max(4, settings.budget // 4), 8))
    )
    resolved["use_staged_mc"] = settings.use_staged_mc or bool(_SCALABILITY_COMPARE_DEFAULTS["use_staged_mc"])
    resolved["mc_runs_fast"] = fast_mc_runs
    resolved["mc_runs_full"] = full_mc_runs
    return resolved


def _selected_scalability_variants(settings: ExperimentSettings) -> list[tuple[str, str, dict[str, object]]]:
    fairness_defaults = _resolved_fairness_compare_settings(settings)
    scalability_defaults = _resolved_scalability_compare_settings(settings)
    fast_elite_count = max(1, int(math.ceil(int(scalability_defaults["local_search_elite_count"]) * 0.5)))
    fast_swap_pool = max(4, min(int(scalability_defaults["swap_candidate_pool_size"]), max(4, settings.budget // 3)))
    fast_prefilter_top_k = max(
        fast_swap_pool,
        min(int(scalability_defaults["candidate_prefilter_top_k"]), max(12, settings.budget)),
    )
    return [
        (
            "hybrid_siea_ml_two_tier_tuned_fairness_full",
            "Current full-quality fairness-focused method without extra scalability pruning.",
            fairness_defaults,
        ),
        (
            "hybrid_siea_ml_two_tier_tuned_fairness_full_balanced",
            "Balanced scalability mode: bounded shortlists, selective local search, caches, and staged MC screening.",
            {
                **scalability_defaults,
                "optimization_mode": "balanced",
            },
        ),
        (
            "hybrid_siea_ml_two_tier_tuned_fairness_full_fast",
            "Fast scalability mode: stronger shortlist reduction, periodic elite-only local search, and staged MC screening.",
            {
                **scalability_defaults,
                "optimization_mode": "fast",
                "local_search_elite_count": fast_elite_count,
                "local_search_every_n_generations": max(2, int(scalability_defaults["local_search_every_n_generations"])),
                "local_search_first_improvement": True,
                "swap_candidate_pool_size": fast_swap_pool,
                "candidate_prefilter_top_k": fast_prefilter_top_k,
            },
        ),
    ]


def _build_optimizer_config(
    settings: ExperimentSettings,
    include_fairness_parameters: bool = True,
    **overrides: object,
) -> HybridSIEAConfig:
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
        mc_runs=settings.mc_runs,
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
    return config


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
    if settings.use_node2vec:
        raise ValueError("Node2Vec ML variants were removed. use_node2vec is no longer supported.")
    if settings.use_ml and settings.ml_guidance_mode not in {"off", "two_tier"}:
        raise ValueError(
            "Removed ML guidance mode requested. Only ml_guidance_mode='two_tier' is supported for ML runs."
        )
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
            f"Use {_KEPT_ML_VARIANT_LABEL} as the single supported ML variant."
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
                f"{removed_display}. Supported ML variant: {_KEPT_ML_VARIANT_LABEL}."
            )

    def is_selected(label: str) -> bool:
        return selected_method_set is None or label in selected_method_set

    def any_ml_methods_selected() -> bool:
        if selected_method_set is None:
            return True
        return any(method.startswith("hybrid_siea_ml_") for method in selected_method_set)

    results: list[dict[str, object]] = []

    for community_method in methods:
        community_result = detect_communities(dataset.graph, method=community_method, seed=settings.random_seed)
        quality = compute_community_quality_metrics(dataset.graph, community_result)

        for baseline_name in baselines:
            if not is_selected(baseline_name):
                continue
            baseline_result = run_baseline(
                dataset=dataset,
                protected_group_report=protected_group_report,
                method=baseline_name,
                budget=settings.budget,
                propagation_probability=settings.propagation_probability,
                mc_runs=settings.mc_runs,
                diffusion_model=settings.diffusion_model,
                lambda_weight=settings.lambda_weight,
                community_result=community_result,
                random_seed=settings.random_seed,
            )
            results.append(_baseline_row(dataset, community_method, baseline_result, quality, settings.diffusion_model))

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
                    config=config,
                    quality=quality,
                    diffusion_model=settings.diffusion_model,
                    note=note,
                )
            )

        if settings.use_ml and any_ml_methods_selected():
            label_result, label_runtime = _generate_ml_labels(
                dataset=dataset,
                protected_group_report=protected_group_report,
                settings=settings,
            )
            training_result, base_ml_training_runtime = _prepare_ml_training(
                dataset=dataset,
                protected_group_report=protected_group_report,
                community_result=community_result,
                settings=settings,
                label_result=label_result,
            )
            ml_preparation_runtime = label_runtime + base_ml_training_runtime
            ml_metrics = (
                f"spearman={training_result.validation_spearman:.6f}; "
                f"precision_at_budget={training_result.validation_precision_at_budget:.6f}; "
                f"label_variance={label_result.label_variance:.6f}"
            )

            for label, guidance_mode, use_legacy_two_tier in _selected_ml_variants(settings):
                if not is_selected(label):
                    continue
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
                    "ml_node_scores": training_result.predicted_scores,
                }
                pool_note = f"full_pool={dataset.graph.number_of_nodes()}; tier_policy=tuned"

                ml_optimizer = HybridSIEAOptimizer(**optimizer_kwargs)
                ml_result = ml_optimizer.optimize()
                ml_history_path = _history_path(
                    settings.output_dir,
                    dataset.name,
                    settings.protected_attribute,
                    settings.budget,
                    community_method,
                    label,
                )
                ml_note = f"ML guidance mode={guidance_mode}; {pool_note}; {ml_metrics}"
                if ml_history_path is not None:
                    ml_result.history.to_csv(ml_history_path, index=False)
                    ml_note = "; ".join(part for part in [ml_note, f"history={ml_history_path.name}"] if part)

                ml_row = _hybrid_row(
                    dataset=dataset,
                    community_method=community_method,
                    label=label,
                    variant_type="ml_guided",
                    result=ml_result,
                    config=ml_config,
                    quality=quality,
                    diffusion_model=settings.diffusion_model,
                    note=ml_note,
                    node2vec_mode="off",
                )
                ml_row["runtime_seconds"] = ml_preparation_runtime + ml_result.runtime_seconds
                ml_row["ml_validation_spearman"] = training_result.validation_spearman
                ml_row["ml_validation_precision_at_budget"] = training_result.validation_precision_at_budget
                ml_row["ml_guidance_mode"] = "two_tier"
                results.append(ml_row)

            if settings.compare_fairness_variants:
                for label, fairness_note, fairness_overrides in _selected_fairness_variants(settings):
                    if not is_selected(label):
                        continue
                    fairness_config = _build_optimizer_config(
                        settings,
                        include_fairness_parameters=False,
                        ml_guidance_mode="two_tier",
                        **fairness_overrides,
                    )
                    fairness_optimizer = HybridSIEAOptimizer(
                        dataset=dataset,
                        protected_group_report=protected_group_report,
                        community_result=community_result,
                        config=fairness_config,
                        ml_node_scores=training_result.predicted_scores,
                    )
                    fairness_result = fairness_optimizer.optimize()
                    fairness_history_path = _history_path(
                        settings.output_dir,
                        dataset.name,
                        settings.protected_attribute,
                        settings.budget,
                        community_method,
                        label,
                    )
                    fairness_note_full = "; ".join(
                        part
                        for part in [
                            fairness_note,
                            "ML guidance mode=two_tier",
                            f"full_pool={dataset.graph.number_of_nodes()}",
                            f"spearman={training_result.validation_spearman:.6f}",
                            f"precision_at_budget={training_result.validation_precision_at_budget:.6f}",
                            f"label_variance={label_result.label_variance:.6f}",
                        ]
                        if part
                    )
                    if fairness_history_path is not None:
                        fairness_result.history.to_csv(fairness_history_path, index=False)
                        fairness_note_full = "; ".join(
                            part
                            for part in [fairness_note_full, f"history={fairness_history_path.name}"]
                            if part
                        )

                    fairness_row = _hybrid_row(
                        dataset=dataset,
                        community_method=community_method,
                        label=label,
                        variant_type="fairness_variant",
                        result=fairness_result,
                        config=fairness_config,
                        quality=quality,
                        diffusion_model=settings.diffusion_model,
                        note=fairness_note_full,
                        node2vec_mode="off",
                    )
                    fairness_row["runtime_seconds"] = ml_preparation_runtime + fairness_result.runtime_seconds
                    fairness_row["ml_validation_spearman"] = training_result.validation_spearman
                    fairness_row["ml_validation_precision_at_budget"] = training_result.validation_precision_at_budget
                    fairness_row["ml_guidance_mode"] = "two_tier"
                    results.append(fairness_row)

            if settings.compare_scalability_variants:
                existing_scalability_methods = {
                    str(row["method"])
                    for row in results
                    if row.get("community_method") == community_method
                }
                for label, scalability_note, scalability_overrides in _selected_scalability_variants(settings):
                    if not is_selected(label):
                        continue
                    if label in existing_scalability_methods:
                        continue
                    scalability_config = _build_optimizer_config(
                        settings,
                        include_fairness_parameters=False,
                        ml_guidance_mode="two_tier",
                        **scalability_overrides,
                    )
                    scalability_optimizer = HybridSIEAOptimizer(
                        dataset=dataset,
                        protected_group_report=protected_group_report,
                        community_result=community_result,
                        config=scalability_config,
                        ml_node_scores=training_result.predicted_scores,
                    )
                    scalability_result = scalability_optimizer.optimize()
                    scalability_history_path = _history_path(
                        settings.output_dir,
                        dataset.name,
                        settings.protected_attribute,
                        settings.budget,
                        community_method,
                        label,
                    )
                    scalability_note_full = "; ".join(
                        part
                        for part in [
                            scalability_note,
                            f"mode={scalability_config.optimization_mode}",
                            f"staged_mc={int(scalability_config.use_staged_mc)}",
                            f"full_pool={dataset.graph.number_of_nodes()}",
                            f"spearman={training_result.validation_spearman:.6f}",
                            f"precision_at_budget={training_result.validation_precision_at_budget:.6f}",
                            f"label_variance={label_result.label_variance:.6f}",
                        ]
                        if part
                    )
                    if scalability_history_path is not None:
                        scalability_result.history.to_csv(scalability_history_path, index=False)
                        scalability_note_full = "; ".join(
                            part
                            for part in [scalability_note_full, f"history={scalability_history_path.name}"]
                            if part
                        )

                    scalability_row = _hybrid_row(
                        dataset=dataset,
                        community_method=community_method,
                        label=label,
                        variant_type="scalability_baseline" if label == "hybrid_siea_ml_two_tier_tuned_fairness_full" else "scalability_variant",
                        result=scalability_result,
                        config=scalability_config,
                        quality=quality,
                        diffusion_model=settings.diffusion_model,
                        note=scalability_note_full,
                        node2vec_mode="off",
                    )
                    scalability_row["runtime_seconds"] = ml_preparation_runtime + scalability_result.runtime_seconds
                    scalability_row["ml_validation_spearman"] = training_result.validation_spearman
                    scalability_row["ml_validation_precision_at_budget"] = training_result.validation_precision_at_budget
                    scalability_row["ml_guidance_mode"] = "two_tier"
                    results.append(scalability_row)

            if settings.compare_refinement_variants:
                refinement_base_label = "hybrid_siea_ml_two_tier_tuned_fairness_full"
                refinement_base_note = "Current best fairness-focused method before the new refinement study."
                fairness_defaults = _resolved_fairness_compare_settings(settings)

                if not settings.compare_fairness_variants and is_selected(refinement_base_label):
                    refinement_base_config = _build_optimizer_config(
                        settings,
                        include_fairness_parameters=False,
                        ml_guidance_mode="two_tier",
                        **fairness_defaults,
                    )
                    refinement_base_optimizer = HybridSIEAOptimizer(
                        dataset=dataset,
                        protected_group_report=protected_group_report,
                        community_result=community_result,
                        config=refinement_base_config,
                        ml_node_scores=training_result.predicted_scores,
                    )
                    refinement_base_result = refinement_base_optimizer.optimize()
                    refinement_base_history_path = _history_path(
                        settings.output_dir,
                        dataset.name,
                        settings.protected_attribute,
                        settings.budget,
                        community_method,
                        refinement_base_label,
                    )
                    refinement_base_note_full = "; ".join(
                        part
                        for part in [
                            refinement_base_note,
                            "ML guidance mode=two_tier",
                            f"full_pool={dataset.graph.number_of_nodes()}",
                            f"spearman={training_result.validation_spearman:.6f}",
                            f"precision_at_budget={training_result.validation_precision_at_budget:.6f}",
                            f"label_variance={label_result.label_variance:.6f}",
                        ]
                        if part
                    )
                    if refinement_base_history_path is not None:
                        refinement_base_result.history.to_csv(refinement_base_history_path, index=False)
                        refinement_base_note_full = "; ".join(
                            part
                            for part in [refinement_base_note_full, f"history={refinement_base_history_path.name}"]
                            if part
                        )
                    refinement_base_row = _hybrid_row(
                        dataset=dataset,
                        community_method=community_method,
                        label=refinement_base_label,
                        variant_type="refinement_baseline",
                        result=refinement_base_result,
                        config=refinement_base_config,
                        quality=quality,
                        diffusion_model=settings.diffusion_model,
                        note=refinement_base_note_full,
                        node2vec_mode="off",
                    )
                    refinement_base_row["runtime_seconds"] = ml_preparation_runtime + refinement_base_result.runtime_seconds
                    refinement_base_row["ml_validation_spearman"] = training_result.validation_spearman
                    refinement_base_row["ml_validation_precision_at_budget"] = training_result.validation_precision_at_budget
                    refinement_base_row["ml_guidance_mode"] = "two_tier"
                    results.append(refinement_base_row)

                for label, refinement_note, refinement_overrides in _selected_refinement_variants(settings):
                    if not is_selected(label):
                        continue
                    refinement_config = _build_optimizer_config(
                        settings,
                        include_fairness_parameters=False,
                        ml_guidance_mode="two_tier",
                        **fairness_defaults,
                        **refinement_overrides,
                    )
                    refinement_optimizer = HybridSIEAOptimizer(
                        dataset=dataset,
                        protected_group_report=protected_group_report,
                        community_result=community_result,
                        config=refinement_config,
                        ml_node_scores=training_result.predicted_scores,
                    )
                    refinement_result = refinement_optimizer.optimize()
                    refinement_history_path = _history_path(
                        settings.output_dir,
                        dataset.name,
                        settings.protected_attribute,
                        settings.budget,
                        community_method,
                        label,
                    )
                    refinement_note_full = "; ".join(
                        part
                        for part in [
                            refinement_note,
                            "ML guidance mode=two_tier",
                            f"full_pool={dataset.graph.number_of_nodes()}",
                            f"spearman={training_result.validation_spearman:.6f}",
                            f"precision_at_budget={training_result.validation_precision_at_budget:.6f}",
                            f"label_variance={label_result.label_variance:.6f}",
                        ]
                        if part
                    )
                    if refinement_history_path is not None:
                        refinement_result.history.to_csv(refinement_history_path, index=False)
                        refinement_note_full = "; ".join(
                            part
                            for part in [refinement_note_full, f"history={refinement_history_path.name}"]
                            if part
                        )

                    refinement_row = _hybrid_row(
                        dataset=dataset,
                        community_method=community_method,
                        label=label,
                        variant_type="refinement_variant",
                        result=refinement_result,
                        config=refinement_config,
                        quality=quality,
                        diffusion_model=settings.diffusion_model,
                        note=refinement_note_full,
                        node2vec_mode="off",
                    )
                    refinement_row["runtime_seconds"] = ml_preparation_runtime + refinement_result.runtime_seconds
                    refinement_row["ml_validation_spearman"] = training_result.validation_spearman
                    refinement_row["ml_validation_precision_at_budget"] = training_result.validation_precision_at_budget
                    refinement_row["ml_guidance_mode"] = "two_tier"
                    results.append(refinement_row)

            if settings.compare_swap_runtime_variants:
                fairness_defaults = _resolved_fairness_compare_settings(settings)
                refinement_defaults = _resolved_refinement_compare_settings(settings)
                swap_runtime_base = {
                    "local_search_swap_trials": int(refinement_defaults["local_search_swap_trials"]),
                    "local_search_candidate_pool_size": int(refinement_defaults["local_search_candidate_pool_size"]),
                    "local_search_delta_mf_weight": float(refinement_defaults["local_search_delta_mf_weight"]),
                    "local_search_delta_dcv_weight": float(refinement_defaults["local_search_delta_dcv_weight"]),
                    "local_search_overlap_penalty_weight": float(refinement_defaults["local_search_overlap_penalty_weight"]),
                }
                existing_swap_methods = {
                    str(row["method"])
                    for row in results
                    if row.get("community_method") == community_method
                }

                for label, swap_note, swap_overrides in _selected_swap_runtime_variants(settings):
                    if not is_selected(label):
                        continue
                    if label in existing_swap_methods:
                        continue
                    swap_config = _build_optimizer_config(
                        settings,
                        include_fairness_parameters=False,
                        ml_guidance_mode="two_tier",
                        **fairness_defaults,
                        **swap_runtime_base,
                        **swap_overrides,
                    )
                    swap_optimizer = HybridSIEAOptimizer(
                        dataset=dataset,
                        protected_group_report=protected_group_report,
                        community_result=community_result,
                        config=swap_config,
                        ml_node_scores=training_result.predicted_scores,
                    )
                    swap_result = swap_optimizer.optimize()
                    swap_history_path = _history_path(
                        settings.output_dir,
                        dataset.name,
                        settings.protected_attribute,
                        settings.budget,
                        community_method,
                        label,
                    )
                    swap_note_full = "; ".join(
                        part
                        for part in [
                            swap_note,
                            "ML guidance mode=two_tier",
                            f"full_pool={dataset.graph.number_of_nodes()}",
                            f"spearman={training_result.validation_spearman:.6f}",
                            f"precision_at_budget={training_result.validation_precision_at_budget:.6f}",
                            f"label_variance={label_result.label_variance:.6f}",
                        ]
                        if part
                    )
                    if swap_history_path is not None:
                        swap_result.history.to_csv(swap_history_path, index=False)
                        swap_note_full = "; ".join(
                            part
                            for part in [swap_note_full, f"history={swap_history_path.name}"]
                            if part
                        )

                    swap_row = _hybrid_row(
                        dataset=dataset,
                        community_method=community_method,
                        label=label,
                        variant_type="swap_runtime_baseline" if label == "hybrid_siea_ml_two_tier_tuned_swap_local_search" else "swap_runtime_variant",
                        result=swap_result,
                        config=swap_config,
                        quality=quality,
                        diffusion_model=settings.diffusion_model,
                        note=swap_note_full,
                        node2vec_mode="off",
                    )
                    swap_row["runtime_seconds"] = ml_preparation_runtime + swap_result.runtime_seconds
                    swap_row["ml_validation_spearman"] = training_result.validation_spearman
                    swap_row["ml_validation_precision_at_budget"] = training_result.validation_precision_at_budget
                    swap_row["ml_guidance_mode"] = "two_tier"
                    results.append(swap_row)

            if settings.compare_runtime_variants:
                fairness_defaults = _resolved_fairness_compare_settings(settings)
                refinement_defaults = _resolved_refinement_compare_settings(settings)
                marginal_runtime_base = {
                    "marginal_gain_scoring_enabled": True,
                    "marginal_gain_delta_mf_weight": float(refinement_defaults["marginal_gain_delta_mf_weight"]),
                    "marginal_gain_delta_dcv_weight": float(refinement_defaults["marginal_gain_delta_dcv_weight"]),
                    "marginal_gain_spread_weight": float(refinement_defaults["marginal_gain_spread_weight"]),
                }
                existing_runtime_methods = {
                    str(row["method"])
                    for row in results
                    if row.get("community_method") == community_method
                }

                for label, runtime_note, runtime_overrides in _selected_runtime_variants(settings):
                    if not is_selected(label):
                        continue
                    if label in existing_runtime_methods:
                        continue
                    runtime_full_overrides = {
                        **fairness_defaults,
                        **marginal_runtime_base,
                        **runtime_overrides,
                    }
                    runtime_config = _build_optimizer_config(
                        settings,
                        include_fairness_parameters=False,
                        ml_guidance_mode="two_tier",
                        **runtime_full_overrides,
                    )
                    runtime_optimizer = HybridSIEAOptimizer(
                        dataset=dataset,
                        protected_group_report=protected_group_report,
                        community_result=community_result,
                        config=runtime_config,
                        ml_node_scores=training_result.predicted_scores,
                    )
                    runtime_result = runtime_optimizer.optimize()
                    runtime_history_path = _history_path(
                        settings.output_dir,
                        dataset.name,
                        settings.protected_attribute,
                        settings.budget,
                        community_method,
                        label,
                    )
                    runtime_note_full = "; ".join(
                        part
                        for part in [
                            runtime_note,
                            "ML guidance mode=two_tier",
                            f"full_pool={dataset.graph.number_of_nodes()}",
                            f"spearman={training_result.validation_spearman:.6f}",
                            f"precision_at_budget={training_result.validation_precision_at_budget:.6f}",
                            f"label_variance={label_result.label_variance:.6f}",
                        ]
                        if part
                    )
                    if runtime_history_path is not None:
                        runtime_result.history.to_csv(runtime_history_path, index=False)
                        runtime_note_full = "; ".join(
                            part
                            for part in [runtime_note_full, f"history={runtime_history_path.name}"]
                            if part
                        )

                    runtime_row = _hybrid_row(
                        dataset=dataset,
                        community_method=community_method,
                        label=label,
                        variant_type="runtime_baseline" if label == "hybrid_siea_ml_two_tier_tuned_marginal_gain" else "runtime_variant",
                        result=runtime_result,
                        config=runtime_config,
                        quality=quality,
                        diffusion_model=settings.diffusion_model,
                        note=runtime_note_full,
                        node2vec_mode="off",
                    )
                    runtime_row["runtime_seconds"] = ml_preparation_runtime + runtime_result.runtime_seconds
                    runtime_row["ml_validation_spearman"] = training_result.validation_spearman
                    runtime_row["ml_validation_precision_at_budget"] = training_result.validation_precision_at_budget
                    runtime_row["ml_guidance_mode"] = "two_tier"
                    results.append(runtime_row)

            if settings.use_node2vec:
                node2vec_config = _build_node2vec_config(settings)
                base_node2vec_metrics = (
                    f"node2vec_dims={node2vec_config.dimensions}; "
                    f"walk_length={node2vec_config.walk_length}; "
                    f"num_walks={node2vec_config.num_walks}; "
                    f"window={node2vec_config.window}; "
                    f"p={node2vec_config.p:g}; "
                    f"q={node2vec_config.q:g}; "
                    f"scale={int(node2vec_config.scale_embeddings)}; "
                    f"pca={node2vec_config.pca_components if node2vec_config.pca_components is not None else 'none'}; "
                    f"model={settings.ml_model_type}"
                )

                for node2vec_variant in _selected_node2vec_variants(settings):
                    if node2vec_variant == "feature_concat":
                        if not (
                            is_selected("ml_topk_node2vec") or is_selected("hybrid_siea_ml_two_tier_tuned_node2vec")
                        ):
                            continue
                        node2vec_training_result, node2vec_training_runtime = _prepare_ml_training(
                            dataset=dataset,
                            protected_group_report=protected_group_report,
                            community_result=community_result,
                            settings=settings,
                            label_result=label_result,
                            use_node2vec=True,
                        )
                        node2vec_preparation_runtime = label_runtime + node2vec_training_runtime
                        node2vec_metrics = (
                            f"spearman={node2vec_training_result.validation_spearman:.6f}; "
                            f"precision_at_budget={node2vec_training_result.validation_precision_at_budget:.6f}; "
                            f"label_variance={label_result.label_variance:.6f}; "
                            f"{base_node2vec_metrics}"
                        )

                        ml_topk_node2vec_evaluation = evaluate_seed_set(
                            dataset=dataset,
                            protected_group_report=protected_group_report,
                            seed_set=node2vec_training_result.ranked_nodes[: settings.budget],
                            propagation_probability=settings.propagation_probability,
                            mc_runs=settings.mc_runs,
                            diffusion_model=settings.diffusion_model,
                            random_seed=settings.random_seed,
                            lambda_weight=settings.lambda_weight,
                            include_soft_mf=True,
                        )
                        if is_selected("ml_topk_node2vec"):
                            results.append(
                                _ml_row(
                                    dataset=dataset,
                                    community_method=community_method,
                                    label="ml_topk_node2vec",
                                    variant_type="ml_baseline",
                                    evaluation=ml_topk_node2vec_evaluation,
                                    runtime_seconds=node2vec_preparation_runtime + ml_topk_node2vec_evaluation.runtime_seconds,
                                    quality=quality,
                                    candidate_pool_size=dataset.graph.number_of_nodes(),
                                    validation_spearman=node2vec_training_result.validation_spearman,
                                    validation_precision_at_budget=node2vec_training_result.validation_precision_at_budget,
                                    guidance_mode="off",
                                    diffusion_model=settings.diffusion_model,
                                    note=f"ML ranking baseline with Node2Vec features; {node2vec_metrics}",
                                    node2vec_enabled=True,
                                    node2vec_mode="feature_concat",
                                )
                            )

                        if is_selected("hybrid_siea_ml_two_tier_tuned_node2vec"):
                            node2vec_optimizer_config = _build_optimizer_config(settings, ml_guidance_mode="two_tier")
                            node2vec_optimizer = HybridSIEAOptimizer(
                                dataset=dataset,
                                protected_group_report=protected_group_report,
                                community_result=community_result,
                                config=node2vec_optimizer_config,
                                ml_node_scores=node2vec_training_result.predicted_scores,
                            )
                            node2vec_result = node2vec_optimizer.optimize()
                            node2vec_history_path = _history_path(
                                settings.output_dir,
                                dataset.name,
                                settings.protected_attribute,
                                settings.budget,
                                community_method,
                                "hybrid_siea_ml_two_tier_tuned_node2vec",
                            )
                            node2vec_note = (
                                "ML guidance mode=two_tier; full_pool="
                                f"{dataset.graph.number_of_nodes()}; tier_policy=tuned; {node2vec_metrics}"
                            )
                            if node2vec_history_path is not None:
                                node2vec_result.history.to_csv(node2vec_history_path, index=False)
                                node2vec_note = "; ".join(
                                    part
                                    for part in [node2vec_note, f"history={node2vec_history_path.name}"]
                                    if part
                                )

                            node2vec_row = _hybrid_row(
                                dataset=dataset,
                                community_method=community_method,
                                label="hybrid_siea_ml_two_tier_tuned_node2vec",
                                variant_type="ml_guided",
                                result=node2vec_result,
                                config=node2vec_optimizer_config,
                                quality=quality,
                                diffusion_model=settings.diffusion_model,
                                note=node2vec_note,
                                node2vec_enabled=True,
                                node2vec_mode="feature_concat",
                            )
                            node2vec_row["runtime_seconds"] = node2vec_preparation_runtime + node2vec_result.runtime_seconds
                            node2vec_row["ml_validation_spearman"] = node2vec_training_result.validation_spearman
                            node2vec_row["ml_validation_precision_at_budget"] = node2vec_training_result.validation_precision_at_budget
                            node2vec_row["ml_guidance_mode"] = "two_tier"
                            results.append(node2vec_row)

                    if node2vec_variant == "diversity_signal":
                        if not is_selected("hybrid_siea_ml_two_tier_tuned_node2vec_diversity"):
                            continue
                        cache_path = build_node2vec_cache_path(settings.output_dir, dataset.name, node2vec_config)
                        embedding_result = generate_node2vec_embeddings(
                            graph=dataset.graph,
                            config=node2vec_config,
                            cache_path=cache_path,
                        )
                        diversity_preparation_runtime = ml_preparation_runtime + embedding_result.runtime_seconds
                        node2vec_embeddings = {
                            row.node_id: [
                                float(getattr(row, column_name))
                                for column_name in embedding_result.embedding_frame.columns
                                if column_name.startswith("node2vec_")
                            ]
                            for row in embedding_result.embedding_frame.itertuples(index=False)
                        }
                        diversity_metrics = (
                            f"spearman={training_result.validation_spearman:.6f}; "
                            f"precision_at_budget={training_result.validation_precision_at_budget:.6f}; "
                            f"label_variance={label_result.label_variance:.6f}; "
                            f"{base_node2vec_metrics}; "
                            f"diversity_weight={settings.node2vec_diversity_weight:.6f}"
                        )
                        diversity_config = _build_optimizer_config(
                            settings,
                            ml_guidance_mode="two_tier",
                            node2vec_diversity_weight=settings.node2vec_diversity_weight,
                        )
                        diversity_optimizer = HybridSIEAOptimizer(
                            dataset=dataset,
                            protected_group_report=protected_group_report,
                            community_result=community_result,
                            config=diversity_config,
                            ml_node_scores=training_result.predicted_scores,
                            node2vec_embeddings=node2vec_embeddings,
                        )
                        diversity_result = diversity_optimizer.optimize()
                        diversity_history_path = _history_path(
                            settings.output_dir,
                            dataset.name,
                            settings.protected_attribute,
                            settings.budget,
                            community_method,
                            "hybrid_siea_ml_two_tier_tuned_node2vec_diversity",
                        )
                        diversity_note = (
                            "ML guidance mode=two_tier; full_pool="
                            f"{dataset.graph.number_of_nodes()}; tier_policy=tuned; {diversity_metrics}"
                        )
                        if diversity_history_path is not None:
                            diversity_result.history.to_csv(diversity_history_path, index=False)
                            diversity_note = "; ".join(
                                part
                                for part in [diversity_note, f"history={diversity_history_path.name}"]
                                if part
                            )

                        diversity_row = _hybrid_row(
                            dataset=dataset,
                            community_method=community_method,
                            label="hybrid_siea_ml_two_tier_tuned_node2vec_diversity",
                            variant_type="ml_guided",
                            result=diversity_result,
                            config=diversity_config,
                            quality=quality,
                            diffusion_model=settings.diffusion_model,
                            note=diversity_note,
                            node2vec_enabled=True,
                            node2vec_mode="diversity_signal",
                        )
                        diversity_row["runtime_seconds"] = diversity_preparation_runtime + diversity_result.runtime_seconds
                        diversity_row["ml_validation_spearman"] = training_result.validation_spearman
                        diversity_row["ml_validation_precision_at_budget"] = training_result.validation_precision_at_budget
                        diversity_row["ml_guidance_mode"] = "two_tier"
                        results.append(diversity_row)

    result_frame = pd.DataFrame(results)
    if selected_method_set is not None and result_frame.empty:
        requested = ", ".join(sorted(selected_method_set))
        raise ValueError(
            "No methods matched the selected filter. "
            f"Requested: {requested}. "
            f"Supported ML variant: {_KEPT_ML_VARIANT_LABEL}."
        )
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
    protected_group_report = verify_protected_groups(dataset, settings.protected_attribute)
    return run_loaded_experiment(
        dataset=dataset,
        protected_group_report=protected_group_report,
        settings=settings,
        community_methods=community_methods,
        baseline_methods=baseline_methods,
        selected_methods=selected_methods,
        include_ablations=include_ablations,
    )
