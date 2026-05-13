"""Phase 5 unified hybrid SI+EA optimizer for Fair Influence Maximization."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .clustering import ClusteringResult
from .community_detection import CommunityDetectionResult, get_community_stats, sample_community, sample_node_from_community
from .data_loader import LoadedDataset, ProtectedGroupReport
from .diffusion import DEFAULT_DIFFUSION_MODEL, validate_diffusion_model
from .evaluation import evaluate_seed_set
from .feature_extraction import compute_node_features, compute_structural_node_scores
from .fairness import FairnessMetrics
from .safe_math import safe_divide, safe_minmax_normalize
from .search_evaluator import SearchObjectiveEvaluator


@dataclass(slots=True)
class HybridSIEAConfig:
    """Explicit configuration for the unified hybrid SI+EA optimizer."""

    budget: int
    population_size: int = 12
    generations: int = 10
    crossover_probability: float = 0.7
    mutation_probability: float = 0.2
    elite_fraction: float = 0.25
    leader_guidance_fraction: float = 0.34
    propagation_probability: float = 0.01
    mc_runs: int = 30
    diffusion_model: str = DEFAULT_DIFFUSION_MODEL
    lambda_weight: float = 0.5
    random_seed: int = 42
    fitness_policy: str = "f_score"
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
    swap_reject_spread_gain_if_fairness_collapses: bool = False
    use_group_quota_repair: bool = True
    min_seeds_per_protected_group: int = 1
    quota_mode: str = "support_aware"
    quota_min_group_support: int = 5
    use_protected_group_quota_initialization: bool = False
    small_group_seed_fraction: float = 0.10
    initialization_quota_mode: str = "sqrt"
    use_fairness_first_repair: bool = False
    weak_group_repair_rounds: int = 0
    majority_overconcentration_threshold: float = 0.60
    local_search_steps: int = 2
    disable_swarm_guidance: bool = False
    disable_crossover: bool = False
    disable_local_search: bool = False
    disable_community_aware_mutation: bool = False
    use_ml_scores_in_crossover: bool = False
    debug_logging: bool = False
    ml_guidance_mode: str = "off"
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
    ml_use_legacy_two_tier: bool = False
    node2vec_diversity_weight: float = 0.0
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
    use_dcv_targeting: bool = False
    dcv_target_weight: float = 3.0
    parity_error_weight: float = 2.0
    over_served_penalty_weight: float = 2.0
    under_served_bonus_weight: float = 1.5
    parity_tolerance: float = 0.005
    use_over_served_group_penalty: bool = False
    use_dcv_first_swap_acceptance: bool = False
    dcv_improvement_epsilon: float = 0.0005
    mf_drop_tolerance: float = 0.001
    fscore_drop_tolerance: float = 0.001
    spread_safe_dcv_tolerance: float = 0.002
    use_dcv_parity_repair: bool = False
    dcv_parity_repair_rounds: int = 5
    dcv_parity_repair_candidate_limit: int = 100
    cluster_diversity_enabled: bool = False
    cluster_balance_enabled: bool = False
    cluster_repair_enabled: bool = False
    cluster_diversity_weight: float = 0.0
    max_seeds_per_cluster_fraction: float = 0.5
    use_dcv_minimization: bool = False
    dcv_target_mode: str = "mean"
    parity_error_improvement_epsilon: float = 0.0005
    # Shortfall DCV fields.
    shortfall_dcv_weight: float = 1.0
    ideal_influences: dict[str, float] | None = None
    # Overshoot cap — stop weak-group repair for a group once its seed count
    # reaches cap × proportional_seed_target.  0.0 = disabled (default).
    weak_group_overshoot_cap: float = 0.0
    # Large-group minimum quota — groups whose node-fraction ≥ large_group_size_threshold
    # receive at least ceil(budget × size_ratio × large_group_min_quota_ratio) seeds
    # in the repair loop, regardless of quota_mode.  0.0 = disabled (default).
    large_group_min_quota_ratio: float = 0.0
    large_group_size_threshold: float = 0.20


@dataclass(slots=True)
class CandidateEvaluation:
    """Evaluation bundle for one candidate seed set."""

    seed_set: tuple[Any, ...]
    total_spread_mean: float
    total_spread_std: float
    fairness: FairnessMetrics
    f_score: float
    runtime_seconds: float
    score: float


@dataclass(slots=True)
class WeakGroupContext:
    """Compact summary of currently weakest protected groups."""

    weak_groups: tuple[str, ...]
    zero_covered_groups: tuple[str, ...]


@dataclass(slots=True)
class HybridOptimizationResult:
    """Final optimizer output with history for reporting/debugging."""

    best_seed_set: tuple[Any, ...]
    best_score: float
    best_spread: float
    best_fairness: FairnessMetrics
    runtime_seconds: float
    candidate_pool_size: int
    history: pd.DataFrame
    repaired_seed_sets: int = 0
    successful_swaps: int = 0
    fitness_cache_hits: int = 0
    marginal_cache_hits: int = 0
    swap_cache_hits: int = 0
    repair_attempts: int = 0
    weak_group_repairs: int = 0
    seeds_redirected_by_overshoot_cap: int = 0
    protected_group_seed_counts_before_repair: dict[str, int] | None = None
    protected_group_seed_counts_after_repair: dict[str, int] | None = None
    swap_attempts: int = 0
    swap_accepted: int = 0
    swap_rejected_fairness_degradation: int = 0
    swap_accepted_fscore_improvement: int = 0
    swap_accepted_mf_improvement: int = 0
    swap_accepted_spread_fairness_preserved: int = 0
    swaps_accepted_dcv_improvement: int = 0
    swaps_rejected_dcv_worsening: int = 0
    swaps_accepted_parity_improvement: int = 0
    seed_quota_per_protected_group: dict[str, int] | None = None
    initial_population_group_coverage_summary: dict[str, int] | None = None
    parity_repair_attempts: int = 0
    parity_repair_successes: int = 0
    dcv_before_parity_repair: float | None = None
    dcv_after_parity_repair_estimated: float | None = None
    groups_rebalanced: tuple[str, ...] = ()
    optimizer_mode: str = "hybrid_si_ea"
    crossover_rate: float | None = None
    mutation_rate: float | None = None
    local_search_enabled: bool | None = None
    local_search_top_elites: float | None = None
    local_search_intensity: str = ""
    local_search_attempts: int = 0
    local_search_improvements: int = 0
    accepted_fscore_moves: int = 0
    accepted_mf_dcv_moves: int = 0
    accepted_spread_safe_moves: int = 0
    rejected_fairness_drops: int = 0
    search_mc_eval_calls: int = 0
    search_ris_eval_calls: int = 0
    search_fair_ris_eval_calls: int = 0
    time_ris_evaluation: float = 0.0
    time_mc_search_evaluation: float = 0.0
    time_search_objective_total: float = 0.0
    rr_sets_used: int = 0
    rr_sets_generated: int = 0
    duplicate_repairs: int = 0
    budget_repairs: int = 0
    community_repairs: int = 0
    diversity_score: float | None = None


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _normalize_seed_set(seed_set: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(sorted(set(seed_set), key=_sort_key))


def _normalize_score_map(
    scores: dict[Any, float],
    nodes: Sequence[Any],
) -> dict[Any, float]:
    if not nodes:
        return {}

    normalized = safe_minmax_normalize(
        [float(scores[node_id]) for node_id in nodes],
        default=0.0,
        context="hybrid optimizer score normalization",
    )
    return {
        node_id: float(score)
        for node_id, score in zip(nodes, normalized, strict=True)
    }


class HybridSIEAOptimizer:
    """Simple unified hybrid optimizer combining leader guidance and EA operators."""

    def __init__(
        self,
        dataset: LoadedDataset,
        protected_group_report: ProtectedGroupReport,
        community_result: CommunityDetectionResult,
        config: HybridSIEAConfig,
        candidate_nodes: Sequence[Any] | None = None,
        node_scores: dict[Any, float] | None = None,
        ml_node_scores: dict[Any, float] | None = None,
        node2vec_embeddings: dict[Any, Sequence[float]] | None = None,
        search_evaluator: SearchObjectiveEvaluator | None = None,
        clustering_result: ClusteringResult | None = None,
    ) -> None:
        self.dataset = dataset
        self.protected_group_report = protected_group_report
        self.community_result = community_result
        self.config = config
        self.search_evaluator = search_evaluator
        self.rng = np.random.default_rng(config.random_seed)
        self.evaluation_cache: OrderedDict[tuple[Any, ...], CandidateEvaluation] = OrderedDict()
        self.screening_evaluation_cache: OrderedDict[tuple[tuple[Any, ...], int], CandidateEvaluation] = OrderedDict()
        self.marginal_gain_cache: OrderedDict[tuple[Any, ...], tuple[float, float, float, float]] = OrderedDict()
        self.swap_evaluation_cache: OrderedDict[Any, CandidateEvaluation] = OrderedDict()
        self.fitness_cache_hits = 0
        self.marginal_cache_hits = 0
        self.swap_cache_hits = 0
        self.full_evaluation_calls = 0
        self.screening_evaluation_calls = 0
        self.last_screening_mc_runs = 0
        self.last_full_mc_runs = 0
        self.last_local_search_swap_evaluations = 0
        self.last_local_search_applied_count = 0
        self.repaired_seed_sets = 0
        self.repair_attempts = 0
        self.weak_group_repairs = 0
        self.seeds_redirected_by_overshoot_cap = 0
        self.last_repair_group_counts_before: dict[str, int] = {}
        self.last_repair_group_counts_after: dict[str, int] = {}
        self.swap_attempts = 0
        self.swap_accepted = 0
        self.swap_rejected_fairness_degradation = 0
        self.swap_accepted_fscore_improvement = 0
        self.swap_accepted_mf_improvement = 0
        self.swap_accepted_spread_fairness_preserved = 0
        self.swaps_accepted_dcv_improvement = 0
        self.swaps_rejected_dcv_worsening = 0
        self.swaps_accepted_parity_improvement = 0
        self.last_swap_acceptance_reason = ""
        self.last_swap_rejection_reason = ""
        self.parity_repair_attempts = 0
        self.parity_repair_successes = 0
        self.dcv_before_parity_repair: float | None = None
        self.dcv_after_parity_repair_estimated: float | None = None
        self.groups_rebalanced: set[str] = set()
        self.seed_quota_per_protected_group: dict[str, int] = {}
        self.initial_population_group_coverage_summary: dict[str, int] = {}

        self._validate_inputs()
        if self.config.ml_guidance_mode in {"soft_bias", "two_tier"} and ml_node_scores is None:
            raise ValueError("ml_node_scores is required when ml_guidance_mode uses soft ML guidance.")
        self.candidate_pool = self._build_candidate_pool(candidate_nodes)
        self.candidate_pool_set = set(self.candidate_pool)
        self.node_group_by_node = {
            node_id: group_name
            for group_name, node_ids in self.protected_group_report.protected_groups.items()
            for node_id in node_ids
        }
        self.group_names = tuple(
            sorted(self.protected_group_report.group_sizes, key=_sort_key)
        )
        self.structural_graph = (
            self.dataset.graph if not self.dataset.graph.is_directed() else self.dataset.graph.to_undirected()
        )
        self.node_neighbors = {
            node_id: set(self.structural_graph.neighbors(node_id))
            for node_id in self.structural_graph.nodes()
        }
        self.clustering_result = clustering_result
        self.cluster_id_by_node: dict[Any, int] = (
            clustering_result.cluster_id_by_node if clustering_result is not None else {}
        )
        self.available_communities = self._build_available_communities()
        self.available_community_stats = get_community_stats(self.available_communities)
        self.available_clusters: dict[int, tuple[Any, ...]] = self._build_available_clusters()
        self.max_unique_seed_sets = math.comb(len(self.candidate_pool), self.config.budget)
        self.node_scores = self._build_node_scores(node_scores)
        self.normalized_node_scores = _normalize_score_map(self.node_scores, self.candidate_pool)
        self.ml_node_scores = self._build_ml_node_scores(ml_node_scores)
        self.node2vec_embeddings = self._build_node2vec_embeddings(node2vec_embeddings)
        self.community_group_fraction = self._build_community_group_fraction()
        self.group_community_ids = self._build_group_community_ids()
        self.node_neighbor_group_fraction = self._build_node_neighbor_group_fraction()
        self.node_bridge_group_fraction = self._build_node_bridge_group_fraction()
        self.node_group_reach_profile = self._build_node_group_reach_profile()
        self.global_ranked_nodes = tuple(
            sorted(
                self.candidate_pool,
                key=lambda node_id: (-float(self.node_scores[node_id]), _sort_key(node_id)),
            )
        )
        self.ml_ranked_nodes = tuple(
            sorted(
                self.candidate_pool,
                key=lambda node_id: (-float(self.ml_node_scores[node_id]), _sort_key(node_id)),
            )
        )
        self.primary_ml_pool, self.secondary_ml_pool = self._build_ml_tiers()
        self.primary_ml_pool_set = set(self.primary_ml_pool)

    def _validate_inputs(self) -> None:
        valid_ml_modes = {"off", "hard_filter", "soft_bias", "two_tier"}
        valid_local_search_modes = {"default", "worst_group"}
        valid_optimization_modes = {"full", "balanced", "fast"}
        valid_fitness_policies = {"f_score", "fairness_first", "professor_priority"}
        valid_quota_modes = {"none", "at_least_one", "proportional", "support_aware"}
        valid_initialization_quota_modes = {"proportional", "sqrt", "uniform_min"}
        validate_diffusion_model(self.config.diffusion_model)
        if self.dataset.graph.number_of_nodes() == 0:
            raise ValueError("dataset.graph must contain at least one node.")
        if self.dataset.name != self.protected_group_report.dataset_name:
            raise ValueError("protected_group_report.dataset_name must match dataset.name.")
        if set(self.dataset.graph.nodes()) != set(self.community_result.community_id_by_node):
            raise ValueError("community_result must contain an assignment for every graph node.")
        if (
            not self.community_result.validation.every_node_assigned_exactly_once
            or not self.community_result.validation.mapping_matches_grouped_communities
            or self.community_result.validation.has_empty_communities
        ):
            raise ValueError("community_result validation must confirm a complete non-empty one-to-one assignment.")
        if self.config.budget < 1:
            raise ValueError("budget must be at least 1.")
        if self.config.population_size < 1:
            raise ValueError("population_size must be at least 1.")
        if self.config.generations < 1:
            raise ValueError("generations must be at least 1.")
        if self.config.local_search_steps < 0:
            raise ValueError("local_search_steps must be non-negative.")
        if self.config.mc_runs < 1 and self.search_evaluator is None:
            raise ValueError("mc_runs must be at least 1.")
        if not 0.0 <= self.config.crossover_probability <= 1.0:
            raise ValueError("crossover_probability must be between 0.0 and 1.0.")
        if not 0.0 <= self.config.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be between 0.0 and 1.0.")
        if not 0.0 < self.config.elite_fraction <= 1.0:
            raise ValueError("elite_fraction must be in the interval (0.0, 1.0].")
        if not 0.0 <= self.config.leader_guidance_fraction <= 1.0:
            raise ValueError("leader_guidance_fraction must be between 0.0 and 1.0.")
        if not 0.0 <= self.config.propagation_probability <= 1.0:
            raise ValueError("propagation_probability must be between 0.0 and 1.0.")
        if not 0.0 <= self.config.lambda_weight <= 1.0:
            raise ValueError("lambda_weight must be between 0.0 and 1.0.")
        if self.config.fitness_policy not in valid_fitness_policies:
            raise ValueError(f"fitness_policy must be one of {sorted(valid_fitness_policies)}.")
        if self.config.quota_mode not in valid_quota_modes:
            raise ValueError(f"quota_mode must be one of {sorted(valid_quota_modes)}.")
        if self.config.min_seeds_per_protected_group < 0:
            raise ValueError("min_seeds_per_protected_group must be non-negative.")
        if self.config.quota_min_group_support < 1:
            raise ValueError("quota_min_group_support must be at least 1.")
        if self.config.initialization_quota_mode not in valid_initialization_quota_modes:
            raise ValueError(f"initialization_quota_mode must be one of {sorted(valid_initialization_quota_modes)}.")
        if not 0.0 <= self.config.small_group_seed_fraction <= 1.0:
            raise ValueError("small_group_seed_fraction must be between 0.0 and 1.0.")
        if self.config.weak_group_repair_rounds < 0:
            raise ValueError("weak_group_repair_rounds must be non-negative.")
        if not 0.0 <= self.config.majority_overconcentration_threshold <= 1.0:
            raise ValueError("majority_overconcentration_threshold must be between 0.0 and 1.0.")
        for field_name in (
            "fscore_weight",
            "mf_weight",
            "dcv_weight",
            "group_coverage_weight",
            "scalability_weight",
            "spread_weight",
            "runtime_penalty_weight",
            "runtime_weight",
            "fairness_tolerance_dcv",
            "fairness_tolerance_fscore_drop",
            "dcv_target_weight",
            "parity_error_weight",
            "over_served_penalty_weight",
            "under_served_bonus_weight",
            "parity_tolerance",
            "dcv_improvement_epsilon",
            "mf_drop_tolerance",
            "fscore_drop_tolerance",
            "spread_safe_dcv_tolerance",
        ):
            if float(getattr(self.config, field_name)) < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.config.ml_guidance_mode not in valid_ml_modes:
            raise ValueError(f"ml_guidance_mode must be one of {sorted(valid_ml_modes)}.")
        if not 0.0 < self.config.ml_primary_pool_ratio <= 1.0:
            raise ValueError("ml_primary_pool_ratio must be in the interval (0.0, 1.0].")
        if not 0.0 <= self.config.ml_secondary_exploration_rate <= 1.0:
            raise ValueError("ml_secondary_exploration_rate must be between 0.0 and 1.0.")
        if not 0.0 <= self.config.ml_initialization_bias <= 1.0:
            raise ValueError("ml_initialization_bias must be between 0.0 and 1.0.")
        if not 0.0 <= self.config.ml_initialization_primary_rate <= 1.0:
            raise ValueError("ml_initialization_primary_rate must be between 0.0 and 1.0.")
        if not 0.0 <= self.config.ml_mutation_primary_rate <= 1.0:
            raise ValueError("ml_mutation_primary_rate must be between 0.0 and 1.0.")
        if not 0.0 <= self.config.ml_repair_primary_rate <= 1.0:
            raise ValueError("ml_repair_primary_rate must be between 0.0 and 1.0.")
        if not 0.0 <= self.config.ml_local_search_primary_rate <= 1.0:
            raise ValueError("ml_local_search_primary_rate must be between 0.0 and 1.0.")
        if self.config.ml_mutation_bias_weight < 0.0:
            raise ValueError("ml_mutation_bias_weight must be non-negative.")
        if self.config.ml_repair_bias_weight < 0.0:
            raise ValueError("ml_repair_bias_weight must be non-negative.")
        if self.config.ml_local_search_bias_weight < 0.0:
            raise ValueError("ml_local_search_bias_weight must be non-negative.")
        if self.config.node2vec_diversity_weight < 0.0:
            raise ValueError("node2vec_diversity_weight must be non-negative.")
        if self.config.fairness_first_init_slots < 0:
            raise ValueError("fairness_first_init_slots must be non-negative.")
        if self.config.fairness_first_init_slots > self.config.budget:
            raise ValueError("fairness_first_init_slots cannot exceed budget.")
        if self.config.fairness_first_init_weight < 0.0:
            raise ValueError("fairness_first_init_weight must be non-negative.")
        if self.config.weakest_group_k < 1:
            raise ValueError("weakest_group_k must be at least 1.")
        if self.config.weakest_group_mutation_weight < 0.0:
            raise ValueError("weakest_group_mutation_weight must be non-negative.")
        if self.config.zero_group_bonus_weight < 0.0:
            raise ValueError("zero_group_bonus_weight must be non-negative.")
        if self.config.bridge_to_weak_group_weight < 0.0:
            raise ValueError("bridge_to_weak_group_weight must be non-negative.")
        if self.config.repair_fairness_weight < 0.0:
            raise ValueError("repair_fairness_weight must be non-negative.")
        if self.config.repair_bridge_weight < 0.0:
            raise ValueError("repair_bridge_weight must be non-negative.")
        if self.config.repair_centrality_weight < 0.0:
            raise ValueError("repair_centrality_weight must be non-negative.")
        if self.config.repair_diversity_weight < 0.0:
            raise ValueError("repair_diversity_weight must be non-negative.")
        if self.config.local_search_focus_mode not in valid_local_search_modes:
            raise ValueError(f"local_search_focus_mode must be one of {sorted(valid_local_search_modes)}.")
        if self.config.local_search_bottom_k_groups < 1:
            raise ValueError("local_search_bottom_k_groups must be at least 1.")
        if self.config.local_search_max_trials < 0:
            raise ValueError("local_search_max_trials must be non-negative.")
        if self.config.marginal_gain_delta_mf_weight < 0.0:
            raise ValueError("marginal_gain_delta_mf_weight must be non-negative.")
        if self.config.marginal_gain_delta_dcv_weight < 0.0:
            raise ValueError("marginal_gain_delta_dcv_weight must be non-negative.")
        if self.config.marginal_gain_spread_weight < 0.0:
            raise ValueError("marginal_gain_spread_weight must be non-negative.")
        if self.config.local_search_swap_trials < 0:
            raise ValueError("local_search_swap_trials must be non-negative.")
        if self.config.local_search_candidate_pool_size < 0:
            raise ValueError("local_search_candidate_pool_size must be non-negative.")
        if self.config.local_search_delta_mf_weight < 0.0:
            raise ValueError("local_search_delta_mf_weight must be non-negative.")
        if self.config.local_search_delta_dcv_weight < 0.0:
            raise ValueError("local_search_delta_dcv_weight must be non-negative.")
        if self.config.local_search_overlap_penalty_weight < 0.0:
            raise ValueError("local_search_overlap_penalty_weight must be non-negative.")
        if self.config.swap_candidate_pool_size < 0:
            raise ValueError("swap_candidate_pool_size must be non-negative.")
        if self.config.swap_prefilter_top_k < 0:
            raise ValueError("swap_prefilter_top_k must be non-negative.")
        if self.config.local_search_failed_patience < 0:
            raise ValueError("local_search_failed_patience must be non-negative.")
        if self.config.full_eval_top_k < 0:
            raise ValueError("full_eval_top_k must be non-negative.")
        if self.config.proxy_score_weights is not None:
            valid_proxy_keys = {
                "delta_mf",
                "delta_dcv",
                "spread",
                "overlap",
                "ml",
                "bridge",
                "community_diversity",
                "diversity",
            }
            invalid_proxy_keys = sorted(
                key for key in self.config.proxy_score_weights if key not in valid_proxy_keys
            )
            if invalid_proxy_keys:
                raise ValueError(f"proxy_score_weights contains unsupported keys: {invalid_proxy_keys}.")
            for key, value in self.config.proxy_score_weights.items():
                if float(value) < 0.0:
                    raise ValueError(f"proxy_score_weights['{key}'] must be non-negative.")
        if self.config.urgency_exponent <= 0.0:
            raise ValueError("urgency_exponent must be positive.")
        if self.config.weak_group_focus_weight < 0.0:
            raise ValueError("weak_group_focus_weight must be non-negative.")
        if self.config.same_community_penalty_weight < 0.0:
            raise ValueError("same_community_penalty_weight must be non-negative.")
        if self.config.neighborhood_overlap_penalty_weight < 0.0:
            raise ValueError("neighborhood_overlap_penalty_weight must be non-negative.")
        if self.config.marginal_candidate_pool_size < 0:
            raise ValueError("marginal_candidate_pool_size must be non-negative.")
        if self.config.mutation_candidate_pool_size < 0:
            raise ValueError("mutation_candidate_pool_size must be non-negative.")
        if self.config.repair_candidate_pool_size < 0:
            raise ValueError("repair_candidate_pool_size must be non-negative.")
        if self.config.candidate_prefilter_top_k < 0:
            raise ValueError("candidate_prefilter_top_k must be non-negative.")
        if self.config.cache_max_size < 0:
            raise ValueError("cache_max_size must be non-negative.")
        if self.config.local_search_early_stop_patience < 0:
            raise ValueError("local_search_early_stop_patience must be non-negative.")
        if self.config.local_search_elite_count < 0:
            raise ValueError("local_search_elite_count must be non-negative.")
        if self.config.local_search_every_n_generations < 1:
            raise ValueError("local_search_every_n_generations must be at least 1.")
        if self.config.mc_runs_fast < 0:
            raise ValueError("mc_runs_fast must be non-negative.")
        if self.config.mc_runs_full < 0:
            raise ValueError("mc_runs_full must be non-negative.")
        if self.config.optimization_mode not in valid_optimization_modes:
            raise ValueError(f"optimization_mode must be one of {sorted(valid_optimization_modes)}.")
        if not 0.0 < self.config.refinement_intensity <= 1.0:
            raise ValueError("refinement_intensity must be in the interval (0.0, 1.0].")
        if not 0.0 < self.config.marginal_eval_fraction <= 1.0:
            raise ValueError("marginal_eval_fraction must be in the interval (0.0, 1.0].")
        if self.config.dcv_parity_repair_rounds < 0:
            raise ValueError("dcv_parity_repair_rounds must be non-negative.")
        if self.config.dcv_parity_repair_candidate_limit < 0:
            raise ValueError("dcv_parity_repair_candidate_limit must be non-negative.")
        if self.config.mc_runs_full > 0 and self.config.mc_runs_full < 1:
            raise ValueError("mc_runs_full must be at least 1 when provided.")
        if self.config.use_staged_mc:
            resolved_full_runs = self.config.mc_runs_full if self.config.mc_runs_full > 0 else self.config.mc_runs
            if resolved_full_runs < 1:
                raise ValueError("use_staged_mc requires a positive full MC budget.")
            if resolved_full_runs > self.config.mc_runs:
                raise ValueError("mc_runs_full cannot exceed the search-time mc_runs budget.")
            if self.config.mc_runs_fast <= 0:
                raise ValueError("use_staged_mc requires mc_runs_fast to be set to a positive value.")
            if self.config.mc_runs_fast >= resolved_full_runs:
                raise ValueError("mc_runs_fast must be smaller than the full MC budget when use_staged_mc is enabled.")

    def _build_candidate_pool(self, candidate_nodes: Sequence[Any] | None) -> tuple[Any, ...]:
        pool = self.dataset.graph.nodes() if candidate_nodes is None else candidate_nodes
        unique_nodes = _normalize_seed_set(pool)
        invalid_nodes = [node for node in unique_nodes if node not in self.dataset.graph]
        if invalid_nodes:
            raise ValueError(f"candidate_nodes contains graph-invalid nodes: {invalid_nodes[:5]}.")
        if len(unique_nodes) < self.config.budget:
            raise ValueError("candidate pool must contain at least budget unique nodes.")
        return unique_nodes

    def _build_node_scores(self, node_scores: dict[Any, float] | None) -> dict[Any, float]:
        if node_scores is None:
            feature_frame = compute_node_features(
                dataset=self.dataset,
                protected_group_report=self.protected_group_report,
                community_result=self.community_result,
            )
            node_scores = compute_structural_node_scores(feature_frame)

        missing_nodes = [node for node in self.candidate_pool if node not in node_scores]
        if missing_nodes:
            raise ValueError(f"node_scores is missing candidate nodes: {missing_nodes[:5]}.")

        return {
            node_id: float(node_scores[node_id])
            for node_id in self.candidate_pool
        }

    def _build_ml_node_scores(self, ml_node_scores: dict[Any, float] | None) -> dict[Any, float]:
        if ml_node_scores is None:
            return {node_id: 0.0 for node_id in self.candidate_pool}

        missing_nodes = [node for node in self.candidate_pool if node not in ml_node_scores]
        if missing_nodes:
            raise ValueError(f"ml_node_scores is missing candidate nodes: {missing_nodes[:5]}.")

        raw_scores = {
            node_id: float(ml_node_scores[node_id])
            for node_id in self.candidate_pool
        }
        return _normalize_score_map(raw_scores, self.candidate_pool)

    def _build_node2vec_embeddings(
        self,
        node2vec_embeddings: dict[Any, Sequence[float]] | None,
    ) -> dict[Any, np.ndarray]:
        if node2vec_embeddings is None:
            if self.config.node2vec_diversity_weight > 0.0:
                raise ValueError("node2vec_embeddings is required when node2vec_diversity_weight is positive.")
            return {}

        missing_nodes = [node for node in self.candidate_pool if node not in node2vec_embeddings]
        if missing_nodes:
            raise ValueError(f"node2vec_embeddings is missing candidate nodes: {missing_nodes[:5]}.")

        normalized_embeddings: dict[Any, np.ndarray] = {}
        for node_id in self.candidate_pool:
            vector = np.asarray(node2vec_embeddings[node_id], dtype=float)
            norm = float(np.linalg.norm(vector))
            if norm > 0.0:
                normalized_embeddings[node_id] = vector / norm
            else:
                normalized_embeddings[node_id] = vector
        return normalized_embeddings

    def _ml_guidance_enabled(self) -> bool:
        return self.config.ml_guidance_mode in {"soft_bias", "two_tier"} and any(
            score > 0.0 for score in self.ml_node_scores.values()
        )

    def _uses_two_tier_guidance(self) -> bool:
        return self.config.ml_guidance_mode == "two_tier" and self._ml_guidance_enabled()

    def _uses_legacy_two_tier_guidance(self) -> bool:
        return self._uses_two_tier_guidance() and self.config.ml_use_legacy_two_tier

    def _uses_tuned_two_tier_guidance(self) -> bool:
        return self._uses_two_tier_guidance() and not self.config.ml_use_legacy_two_tier

    def _node2vec_diversity_enabled(self) -> bool:
        return self.config.node2vec_diversity_weight > 0.0 and bool(self.node2vec_embeddings)

    def _node2vec_diversity_score(
        self,
        node_id: Any,
        reference_nodes: Sequence[Any] | set[Any] | None,
        weight: float | None = None,
    ) -> float:
        if not self._node2vec_diversity_enabled():
            return 0.0
        if reference_nodes is None:
            return 0.0

        effective_weight = self.config.node2vec_diversity_weight if weight is None else weight
        if effective_weight <= 0.0:
            return 0.0

        candidate_vector = self.node2vec_embeddings.get(node_id)
        if candidate_vector is None or candidate_vector.size == 0 or not np.any(candidate_vector):
            return 0.0

        novelty_scores: list[float] = []
        for reference_node in reference_nodes:
            if reference_node == node_id:
                continue
            reference_vector = self.node2vec_embeddings.get(reference_node)
            if reference_vector is None or reference_vector.size == 0 or not np.any(reference_vector):
                continue
            similarity = float(np.clip(np.dot(candidate_vector, reference_vector), -1.0, 1.0))
            novelty_scores.append((1.0 - similarity) / 2.0)

        if not novelty_scores:
            return 0.0
        return float(effective_weight * np.mean(novelty_scores))

    def _build_ml_tiers(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        if not self._ml_guidance_enabled():
            return (), ()

        primary_count = max(
            self.config.budget,
            int(math.ceil(len(self.candidate_pool) * self.config.ml_primary_pool_ratio)),
        )
        primary_count = min(primary_count, len(self.ml_ranked_nodes))
        primary_nodes = self.ml_ranked_nodes[:primary_count]
        secondary_nodes = self.ml_ranked_nodes[primary_count:]
        return tuple(primary_nodes), tuple(secondary_nodes)

    def _build_available_communities(self) -> dict[int, tuple[Any, ...]]:
        communities: dict[int, tuple[Any, ...]] = {}
        for community_id, community_nodes in self.community_result.communities.items():
            filtered_nodes = tuple(
                node for node in community_nodes
                if node in self.candidate_pool_set
            )
            if filtered_nodes:
                communities[community_id] = filtered_nodes

        if not communities:
            raise ValueError("No non-empty candidate communities are available for optimization.")
        return communities

    def _build_available_clusters(self) -> dict[int, tuple[Any, ...]]:
        if self.clustering_result is None:
            return {}
        clusters: dict[int, tuple[Any, ...]] = {}
        for cluster_id, cluster_nodes in self.clustering_result.clusters.items():
            filtered = tuple(n for n in cluster_nodes if n in self.candidate_pool_set)
            if filtered:
                clusters[cluster_id] = filtered
        return clusters

    def _cluster_seed_counts(self, seed_set: Sequence[Any]) -> dict[int, int]:
        counts: dict[int, int] = {}
        for node_id in seed_set:
            cid = self.cluster_id_by_node.get(node_id)
            if cid is not None:
                counts[cid] = counts.get(cid, 0) + 1
        return counts

    def _uncovered_cluster_ids(self, seed_set: Sequence[Any]) -> set[int]:
        covered = {
            self.cluster_id_by_node[node_id]
            for node_id in seed_set
            if node_id in self.cluster_id_by_node
        }
        return set(self.available_clusters.keys()) - covered

    def _build_community_group_fraction(self) -> dict[int, dict[str, float]]:
        fractions: dict[int, dict[str, float]] = {}
        for community_id, community_nodes in self.community_result.communities.items():
            group_counts = {group_name: 0 for group_name in self.group_names}
            for node_id in community_nodes:
                group_counts[self.node_group_by_node[node_id]] += 1
            community_size = max(1, len(community_nodes))
            fractions[community_id] = {
                group_name: safe_divide(
                    float(group_counts[group_name]),
                    float(community_size),
                    default=0.0,
                    context=f"optimizer community {community_id} group fraction",
                )
                for group_name in self.group_names
            }
        return fractions

    def _build_group_community_ids(self) -> dict[str, frozenset[int]]:
        community_ids = {
            group_name: set()
            for group_name in self.group_names
        }
        for node_id, group_name in self.node_group_by_node.items():
            community_ids[group_name].add(self.community_result.community_id_by_node[node_id])
        return {
            group_name: frozenset(ids)
            for group_name, ids in community_ids.items()
        }

    def _build_node_neighbor_group_fraction(self) -> dict[Any, dict[str, float]]:
        fractions: dict[Any, dict[str, float]] = {}
        for node_id, neighbors in self.node_neighbors.items():
            if not neighbors:
                fractions[node_id] = {group_name: 0.0 for group_name in self.group_names}
                continue
            neighbor_counts = {group_name: 0 for group_name in self.group_names}
            for neighbor_id in neighbors:
                neighbor_counts[self.node_group_by_node[neighbor_id]] += 1
            fractions[node_id] = {
                group_name: safe_divide(
                    float(neighbor_counts[group_name]),
                    float(len(neighbors)),
                    default=0.0,
                    context=f"optimizer neighbor group fraction for node {node_id}",
                )
                for group_name in self.group_names
            }
        return fractions

    def _build_node_bridge_group_fraction(self) -> dict[Any, dict[str, float]]:
        bridge_fractions: dict[Any, dict[str, float]] = {}
        for node_id, neighbors in self.node_neighbors.items():
            node_community = self.community_result.community_id_by_node[node_id]
            cross_neighbors = [
                neighbor_id
                for neighbor_id in neighbors
                if self.community_result.community_id_by_node[neighbor_id] != node_community
            ]
            if not cross_neighbors:
                bridge_fractions[node_id] = {group_name: 0.0 for group_name in self.group_names}
                continue

            bridge_scores = {group_name: 0 for group_name in self.group_names}
            for neighbor_id in cross_neighbors:
                neighbor_community = self.community_result.community_id_by_node[neighbor_id]
                for group_name in self.group_names:
                    if neighbor_community in self.group_community_ids[group_name]:
                        bridge_scores[group_name] += 1
            bridge_fractions[node_id] = {
                group_name: safe_divide(
                    float(bridge_scores[group_name]),
                    float(len(cross_neighbors)),
                    default=0.0,
                    context=f"optimizer bridge group fraction for node {node_id}",
                )
                for group_name in self.group_names
            }
        return bridge_fractions

    def _build_node_group_reach_profile(self) -> dict[Any, dict[str, float]]:
        profiles: dict[Any, dict[str, float]] = {}
        for node_id in self.candidate_pool:
            node_group = self.node_group_by_node[node_id]
            community_id = self.community_result.community_id_by_node[node_id]
            profiles[node_id] = {
                group_name: float(
                    (
                        (1.0 if node_group == group_name else 0.0)
                        + float(self.node_neighbor_group_fraction[node_id].get(group_name, 0.0))
                        + float(self.community_group_fraction[community_id].get(group_name, 0.0))
                    )
                    / 3.0
                )
                for group_name in self.group_names
            }
        return profiles

    def _seed_sort_key(self, seed_set: tuple[Any, ...]) -> tuple[tuple[str, str], ...]:
        return tuple(_sort_key(node) for node in seed_set)

    def _cache_limit(self) -> int | None:
        if self.config.cache_max_size <= 0:
            return None
        return self.config.cache_max_size

    def _cache_lookup(self, cache: OrderedDict[Any, Any], key: Any, counter_name: str) -> Any | None:
        if key not in cache:
            return None
        cache.move_to_end(key)
        setattr(self, counter_name, int(getattr(self, counter_name)) + 1)
        return cache[key]

    def _cache_store(self, cache: OrderedDict[Any, Any], key: Any, value: Any) -> None:
        cache[key] = value
        cache.move_to_end(key)
        cache_limit = self._cache_limit()
        if cache_limit is None:
            return
        while len(cache) > cache_limit:
            cache.popitem(last=False)

    def _candidate_rank_key(self, evaluation: CandidateEvaluation) -> tuple[float, float, float, float, float, tuple[tuple[str, str], ...]]:
        if self.config.fitness_policy in {"fairness_first", "professor_priority"}:
            return (
                evaluation.score,
                evaluation.f_score,
                evaluation.fairness.mf,
                -evaluation.fairness.dcv,
                evaluation.total_spread_mean,
                self._seed_sort_key(evaluation.seed_set),
            )
        return (
            evaluation.score,
            evaluation.total_spread_mean,
            evaluation.f_score,
            evaluation.fairness.mf,
            -evaluation.fairness.dcv,
            self._seed_sort_key(evaluation.seed_set),
        )

    def _fitness_score(self, evaluation_result: Any) -> float:
        if self.config.fitness_policy not in {"fairness_first", "professor_priority"}:
            base_score = float(evaluation_result.f_score)
            if self.config.use_dcv_targeting:
                base_score += self._dcv_targeting_objective_adjustment(evaluation_result)
            return base_score

        node_count = max(1, int(self.dataset.graph.number_of_nodes()))
        normalized_spread = safe_divide(
            float(evaluation_result.total_spread_mean),
            float(node_count),
            default=0.0,
            context="fairness-first normalized spread fitness",
        )
        normalized_runtime = min(1.0, max(0.0, float(evaluation_result.runtime_seconds)))
        group_values = dict(getattr(evaluation_result.fairness, "normalized_group_spread", {}) or {})
        if not group_values:
            group_values = dict(getattr(evaluation_result.fairness, "group_spread", {}) or {})
        fraction_groups_covered = safe_divide(
            sum(1 for value in group_values.values() if float(value) > 0.0),
            len(group_values),
            default=0.0,
            context="professor-priority group coverage fitness",
        )
        candidate_pool_size = len(getattr(self, "candidate_pool", ()) or ())
        scalability_score = 1.0 - min(
            1.0,
            safe_divide(
                candidate_pool_size,
                node_count,
                default=1.0,
                context="professor-priority scalability fitness",
            ),
        )
        group_coverage_weight = self.config.group_coverage_weight if self.config.fitness_policy == "professor_priority" else 0.0
        scalability_weight = self.config.scalability_weight if self.config.fitness_policy == "professor_priority" else 0.0
        runtime_weight = self.config.runtime_weight if self.config.fitness_policy == "professor_priority" else self.config.runtime_penalty_weight
        _dcv_sf = float(getattr(evaluation_result.fairness, "dcv_shortfall", evaluation_result.fairness.dcv))
        _sf_w = float(getattr(self.config, "shortfall_dcv_weight", 1.0))
        _tcr = float(getattr(evaluation_result.fairness, "target_coverage_ratio", 1.0))
        score = float(
            self.config.fscore_weight * float(evaluation_result.f_score)
            + self.config.mf_weight * float(evaluation_result.fairness.mf)
            - self.config.dcv_weight * _sf_w * _dcv_sf
            + group_coverage_weight * fraction_groups_covered
            + group_coverage_weight * _tcr
            + scalability_weight * scalability_score
            + self.config.spread_weight * normalized_spread
            - runtime_weight * normalized_runtime
        )
        if self.config.use_dcv_targeting:
            score += self._dcv_targeting_objective_adjustment(evaluation_result)
        return score

    def _dcv_targeting_objective_adjustment(self, evaluation_result: Any) -> float:
        fairness = evaluation_result.fairness
        normalized = {
            group_name: float(value)
            for group_name, value in dict(getattr(fairness, "normalized_group_spread", {}) or {}).items()
        }
        if self.config.dcv_target_mode == "median" and normalized:
            sorted_vals = sorted(normalized.values())
            n = len(sorted_vals)
            parity_target = (
                (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
                if n % 2 == 0
                else float(sorted_vals[n // 2])
            )
        else:
            parity_target = float(getattr(fairness, "parity_target", 0.0))
        tolerance = float(self.config.parity_tolerance)
        over_error = float(
            sum(max(0.0, value - parity_target - tolerance) for value in normalized.values())
        )
        under_error = float(
            sum(max(0.0, parity_target - tolerance - value) for value in normalized.values())
        )
        parity_error = float(getattr(fairness, "parity_abs_error", 0.0)) + float(
            getattr(fairness, "parity_squared_error", 0.0)
        )
        return float(
            -self.config.dcv_target_weight * float(fairness.dcv)
            - self.config.parity_error_weight * parity_error
            - self.config.over_served_penalty_weight * over_error
            - self.config.under_served_bonus_weight * under_error
        )

    def _mode_scale(self) -> float:
        mode_scale = {
            "full": 1.0,
            "balanced": 0.65,
            "fast": 0.40,
        }[self.config.optimization_mode]
        return float(np.clip(self.config.refinement_intensity * mode_scale, 0.05, 1.0))

    def _full_mc_runs(self) -> int:
        return int(self.config.mc_runs_full) if self.config.mc_runs_full > 0 else int(self.config.mc_runs)

    def _screening_mc_runs(self) -> int:
        full_runs = self._full_mc_runs()
        if not self.config.use_staged_mc:
            return full_runs
        if self.config.mc_runs_fast > 0:
            return int(self.config.mc_runs_fast)
        if full_runs <= 1:
            return full_runs
        derived_runs = int(math.ceil(full_runs * 0.35))
        return max(1, min(full_runs - 1, derived_runs))

    def _staged_mc_active(self) -> bool:
        return self.config.use_staged_mc and self._screening_mc_runs() < self._full_mc_runs()

    def _effective_marginal_eval_fraction(self) -> float:
        mode_fraction = {
            "full": 1.0,
            "balanced": 0.40,
            "fast": 0.20,
        }[self.config.optimization_mode]
        return float(np.clip(self.config.marginal_eval_fraction * mode_fraction, 0.02, 1.0))

    def _mode_candidate_cap(self, stage: str) -> int:
        if self.config.optimization_mode == "full":
            return 0
        if self.config.optimization_mode == "balanced":
            base_caps = {
                "marginal": max(24, self.config.budget * 6),
                "mutation": max(12, self.config.budget * 4),
                "repair": max(16, self.config.budget * 4),
                "local_search": max(6, self.config.budget * 2),
            }
        else:
            base_caps = {
                "marginal": max(16, self.config.budget * 4),
                "mutation": max(8, self.config.budget * 3),
                "repair": max(10, self.config.budget * 3),
                "local_search": max(4, self.config.budget + 1),
            }
        return max(1, int(math.ceil(base_caps[stage] * self._mode_scale())))

    def _candidate_pool_limit(self, total_candidates: int, stage: str) -> int:
        if total_candidates <= 0:
            return 0

        explicit_limit = {
            "marginal": self.config.marginal_candidate_pool_size,
            "mutation": self.config.mutation_candidate_pool_size,
            "repair": self.config.repair_candidate_pool_size,
            "local_search": self.config.local_search_candidate_pool_size,
        }[stage]
        if (
            stage == "local_search"
            and explicit_limit > 0
            and not self.config.local_search_use_prefilter
            and self.config.optimization_mode == "full"
        ):
            explicit_limit = 0
        if stage == "local_search" and explicit_limit <= 0 and self.config.local_search_use_prefilter:
            explicit_limit = max(4, self.config.budget * 2)
        limit = total_candidates
        if explicit_limit > 0:
            limit = min(limit, explicit_limit)

        should_apply_fraction = (
            self._marginal_gain_scoring_active()
            or stage != "marginal"
            or self.config.optimization_mode != "full"
        )
        if should_apply_fraction:
            fraction_limit = max(1, int(math.ceil(total_candidates * self._effective_marginal_eval_fraction())))
            limit = min(limit, fraction_limit)

        mode_cap = self._mode_candidate_cap(stage)
        if mode_cap > 0:
            limit = min(limit, mode_cap)

        return max(1, min(total_candidates, limit))

    def _candidate_prefilter_limit(self, total_candidates: int, candidate_limit: int, stage: str) -> int:
        if total_candidates <= 0:
            return 0
        minimum_limit = candidate_limit if 0 < candidate_limit < total_candidates else 0
        explicit_prefilter = self.config.candidate_prefilter_top_k
        if explicit_prefilter <= 0:
            if minimum_limit > 0:
                return max(1, min(total_candidates, minimum_limit))
            return total_candidates
        prefilter_limit = explicit_prefilter
        if self.config.optimization_mode != "full":
            prefilter_limit = max(1, int(math.ceil(prefilter_limit * self._mode_scale())))
        return max(1, min(total_candidates, max(minimum_limit, prefilter_limit)))

    def _effective_local_search_budget(self) -> tuple[int, int]:
        candidate_pool_size = (
            self.config.local_search_candidate_pool_size
            if self.config.local_search_candidate_pool_size > 0
            else max(4, self.config.budget * 2)
        )
        candidate_pool_size = max(1, int(math.ceil(candidate_pool_size * self._mode_scale())))

        explicit_max_trials = (
            self.config.local_search_max_trials
            if self.config.local_search_max_trials > 0
            else self.config.local_search_swap_trials
        )
        if explicit_max_trials > 0:
            max_trials = max(1, int(math.ceil(explicit_max_trials * self._mode_scale())))
        else:
            max_trials = max(4, candidate_pool_size)

        patience = (
            self.config.local_search_early_stop_patience
            if self.config.local_search_early_stop_patience > 0
            else max_trials
        )
        patience = min(max_trials, max(1, int(math.ceil(patience * self._mode_scale()))))
        return candidate_pool_size, max_trials if max_trials > 0 else 1, patience

    def _effective_local_search_elite_count(self) -> int:
        if self.config.local_search_elite_count > 0:
            return min(self.config.population_size, self.config.local_search_elite_count)
        if self.config.optimization_mode == "full":
            return self.config.population_size
        if self.config.optimization_mode == "balanced":
            return max(1, int(math.ceil(self.config.population_size * 0.5)))
        return max(1, int(math.ceil(self.config.population_size * 0.34)))

    def _should_run_local_search_generation(self, generation: int) -> bool:
        interval = max(1, self.config.local_search_every_n_generations)
        if generation == self.config.generations - 1:
            return True
        return generation % interval == 0

    def _effective_swap_stage_limits(self, candidate_pool_size: int) -> tuple[int, int, int]:
        swap_candidate_pool_size = (
            self.config.swap_candidate_pool_size
            if self.config.swap_candidate_pool_size > 0
            else candidate_pool_size
        )
        if self.config.optimization_mode != "full":
            swap_candidate_pool_size = max(1, int(math.ceil(swap_candidate_pool_size * self._mode_scale())))

        prefilter_top_k = (
            self.config.swap_prefilter_top_k
            if self.config.swap_prefilter_top_k > 0
            else max(swap_candidate_pool_size, candidate_pool_size)
        )
        if self.config.optimization_mode != "full":
            prefilter_top_k = max(1, int(math.ceil(prefilter_top_k * self._mode_scale())))
        prefilter_top_k = max(swap_candidate_pool_size, prefilter_top_k)

        full_eval_top_k = (
            self.config.full_eval_top_k
            if self.config.full_eval_top_k > 0
            else swap_candidate_pool_size
        )
        if self.config.optimization_mode != "full":
            full_eval_top_k = max(1, int(math.ceil(full_eval_top_k * self._mode_scale())))
        full_eval_top_k = max(1, min(full_eval_top_k, swap_candidate_pool_size))
        return swap_candidate_pool_size, prefilter_top_k, full_eval_top_k

    def _fairness_first_init_active(self) -> bool:
        return (
            self.config.fairness_first_init_enabled
            and self.config.fairness_first_init_slots > 0
            and self.config.fairness_first_init_weight > 0.0
        )

    def _weak_group_mutation_active(self) -> bool:
        return (
            self.config.weakest_group_mutation_weight > 0.0
            or self.config.zero_group_bonus_weight > 0.0
            or self.config.bridge_to_weak_group_weight > 0.0
        )

    def _fairness_repair_active(self) -> bool:
        return any(
            weight > 0.0
            for weight in (
                self.config.repair_fairness_weight,
                self.config.repair_bridge_weight,
                self.config.repair_centrality_weight,
                self.config.repair_diversity_weight,
            )
        )

    def _worst_group_local_search_active(self) -> bool:
        return self.config.local_search_focus_mode == "worst_group"

    def _marginal_gain_scoring_active(self) -> bool:
        return self.config.marginal_gain_scoring_enabled and any(
            weight > 0.0
            for weight in (
                self.config.marginal_gain_delta_mf_weight,
                self.config.marginal_gain_delta_dcv_weight,
                self.config.marginal_gain_spread_weight,
            )
        )

    def _urgency_weighting_active(self) -> bool:
        return self.config.urgency_weight_enabled or self.config.weak_group_focus_weight > 0.0

    def _overlap_penalty_active(self) -> bool:
        return self.config.overlap_penalty_enabled and any(
            weight > 0.0
            for weight in (
                self.config.same_community_penalty_weight,
                self.config.neighborhood_overlap_penalty_weight,
            )
        )

    def _swap_runtime_optimization_active(self) -> bool:
        return any(
            (
                self.config.swap_candidate_pool_size > 0,
                self.config.swap_prefilter_top_k > 0,
                self.config.local_search_first_improvement,
                self.config.full_eval_top_k > 0,
                bool(self.config.proxy_score_weights),
            )
        )

    def _selected_group_counts(self, seed_set: Sequence[Any]) -> dict[str, int]:
        counts = {
            group_name: 0
            for group_name in self.protected_group_report.group_sizes
        }
        for node_id in seed_set:
            counts[self.node_group_by_node[node_id]] += 1
        return counts

    def _selected_community_counts(self, seed_set: Sequence[Any]) -> dict[int, int]:
        counts: dict[int, int] = {}
        for node_id in seed_set:
            community_id = self.community_result.community_id_by_node[node_id]
            counts[community_id] = counts.get(community_id, 0) + 1
        return counts

    def _weak_group_context_from_group_counts(
        self,
        group_counts: dict[str, int],
        weakest_group_k: int | None = None,
    ) -> WeakGroupContext:
        bottom_k = max(1, weakest_group_k or self.config.weakest_group_k)
        ordered_groups = tuple(
            sorted(
                self.group_names,
                key=lambda group_name: (group_counts.get(group_name, 0), _sort_key(group_name)),
            )
        )
        zero_groups = tuple(
            group_name
            for group_name in ordered_groups
            if group_counts.get(group_name, 0) <= 0
        )
        weak_groups = tuple(dict.fromkeys((*zero_groups, *ordered_groups[:bottom_k])))
        return WeakGroupContext(weak_groups=weak_groups, zero_covered_groups=zero_groups)

    def _weak_group_context_from_fairness(
        self,
        fairness: FairnessMetrics,
        weakest_group_k: int | None = None,
    ) -> WeakGroupContext:
        bottom_k = max(1, weakest_group_k or self.config.weakest_group_k)
        ordered_groups = tuple(
            sorted(
                self.group_names,
                key=lambda group_name: (
                    float(fairness.normalized_group_spread.get(group_name, 0.0)),
                    _sort_key(group_name),
                ),
            )
        )
        zero_groups = tuple(
            group_name
            for group_name in ordered_groups
            if float(fairness.group_spread.get(group_name, 0.0)) <= 1e-12
        )
        weak_groups = tuple(dict.fromkeys((*zero_groups, *ordered_groups[:bottom_k])))
        return WeakGroupContext(weak_groups=weak_groups, zero_covered_groups=zero_groups)

    def _weak_group_context_for_seed_set(
        self,
        seed_set: Sequence[Any],
        evaluation: CandidateEvaluation | None = None,
        weakest_group_k: int | None = None,
    ) -> WeakGroupContext:
        if evaluation is None and len(seed_set) == self.config.budget and len(set(seed_set)) == len(seed_set):
            normalized_seed_set = tuple(sorted(seed_set, key=_sort_key))
            if all(node_id in self.candidate_pool_set for node_id in normalized_seed_set):
                evaluation = self._evaluate_seed_set(normalized_seed_set)

        if evaluation is not None:
            return self._weak_group_context_from_fairness(
                evaluation.fairness,
                weakest_group_k=weakest_group_k,
            )

        return self._weak_group_context_from_group_counts(
            self._selected_group_counts(seed_set),
            weakest_group_k=weakest_group_k,
        )

    def _group_support_score(
        self,
        node_id: Any,
        target_groups: Sequence[str],
    ) -> float:
        if not target_groups:
            return 0.0
        node_group = self.node_group_by_node[node_id]
        community_id = self.community_result.community_id_by_node[node_id]
        own_group_score = 1.0 if node_group in target_groups else 0.0
        neighbor_group_score = sum(
            float(self.node_neighbor_group_fraction[node_id].get(group_name, 0.0))
            for group_name in target_groups
        )
        community_group_score = sum(
            float(self.community_group_fraction[community_id].get(group_name, 0.0))
            for group_name in target_groups
        )
        return float(own_group_score + neighbor_group_score + community_group_score)

    def _bridge_to_group_score(
        self,
        node_id: Any,
        target_groups: Sequence[str],
    ) -> float:
        if not target_groups:
            return 0.0
        return float(
            sum(
                float(self.node_bridge_group_fraction[node_id].get(group_name, 0.0))
                for group_name in target_groups
            )
        )

    def _graph_diversity_score(
        self,
        node_id: Any,
        reference_nodes: Sequence[Any] | set[Any] | None,
    ) -> float:
        if not reference_nodes:
            return 0.0

        candidate_neighborhood = set(self.node_neighbors.get(node_id, set()))
        candidate_neighborhood.add(node_id)
        diversity_scores: list[float] = []
        for reference_node in reference_nodes:
            if reference_node == node_id:
                continue
            reference_neighborhood = set(self.node_neighbors.get(reference_node, set()))
            reference_neighborhood.add(reference_node)
            union = candidate_neighborhood | reference_neighborhood
            if not union:
                diversity_scores.append(0.0)
                continue
            overlap = candidate_neighborhood & reference_neighborhood
            diversity_scores.append(1.0 - (len(overlap) / len(union)))
        if not diversity_scores:
            return 0.0
        return float(np.mean(diversity_scores))

    def _normalized_group_coverage(
        self,
        seed_set: Sequence[Any],
        evaluation: CandidateEvaluation | None = None,
    ) -> dict[str, float]:
        if evaluation is not None:
            return {
                group_name: float(np.clip(evaluation.fairness.normalized_group_spread.get(group_name, 0.0), 0.0, 1.0))
                for group_name in self.group_names
            }

        group_counts = self._selected_group_counts(seed_set)
        return self._normalized_group_coverage_from_counts(group_counts)

    def _normalized_group_coverage_from_counts(
        self,
        group_counts: dict[str, int],
    ) -> dict[str, float]:
        return {
            group_name: float(
                np.clip(
                    safe_divide(
                        float(group_counts.get(group_name, 0)),
                        float(self.protected_group_report.group_sizes[group_name]),
                        default=0.0,
                        context=f"optimizer protected-group coverage for {group_name}",
                    ),
                    0.0,
                    1.0,
                )
            )
            for group_name in self.group_names
        }

    def _group_urgency_weights(
        self,
        seed_set: Sequence[Any],
        weak_group_context: WeakGroupContext | None = None,
        evaluation: CandidateEvaluation | None = None,
        coverage_levels: dict[str, float] | None = None,
    ) -> dict[str, float]:
        if coverage_levels is None:
            coverage_levels = self._normalized_group_coverage(seed_set, evaluation=evaluation)
        urgency_weights: dict[str, float] = {}
        for group_name in self.group_names:
            uncovered_fraction = max(0.0, 1.0 - float(coverage_levels[group_name]))
            if self.config.urgency_weight_enabled:
                weight = float(uncovered_fraction**self.config.urgency_exponent)
            else:
                weight = 1.0
            if weak_group_context is not None and group_name in weak_group_context.weak_groups:
                weight *= 1.0 + self.config.weak_group_focus_weight
            urgency_weights[group_name] = weight
        return urgency_weights

    def _parity_adjustment_active(self) -> bool:
        return bool(self.config.use_dcv_targeting or self.config.use_over_served_group_penalty)

    def _parity_candidate_adjustment(
        self,
        node_id: Any,
        coverage_levels: dict[str, float] | None,
    ) -> tuple[float, float, float]:
        if not self._parity_adjustment_active():
            return 0.0, 0.0, 0.0
        if coverage_levels is None:
            coverage_levels = {group_name: 0.0 for group_name in self.group_names}
        if not coverage_levels:
            return 0.0, 0.0, 0.0

        values = {group_name: float(coverage_levels.get(group_name, 0.0)) for group_name in self.group_names}
        parity_target = float(np.mean(list(values.values()))) if values else 0.0
        tolerance = float(self.config.parity_tolerance)
        reach_profile = self.node_group_reach_profile[node_id]
        under_bonus = 0.0
        over_penalty = 0.0
        for group_name, current_value in values.items():
            reach = float(reach_profile.get(group_name, 0.0))
            if current_value < parity_target - tolerance:
                under_bonus += reach * (parity_target - tolerance - current_value)
            elif current_value > parity_target + tolerance:
                over_penalty += reach * (current_value - parity_target - tolerance)
        weighted_bonus = float(self.config.under_served_bonus_weight * under_bonus)
        weighted_penalty = float(self.config.over_served_penalty_weight * over_penalty)
        return weighted_bonus, weighted_penalty, float(weighted_bonus - weighted_penalty)

    def _scoring_priors(
        self,
        reference_nodes: Sequence[Any] | set[Any] | None,
        weak_group_context: WeakGroupContext | None = None,
        reference_evaluation: CandidateEvaluation | None = None,
        require_proxy: bool = False,
    ) -> tuple[tuple[Any, ...], dict[str, float] | None, dict[str, float] | None]:
        reference_seed_set = _normalize_seed_set(reference_nodes or ())
        if not (require_proxy or self._urgency_weighting_active() or self._parity_adjustment_active()):
            return reference_seed_set, None, None

        coverage_levels = self._normalized_group_coverage(
            reference_seed_set,
            evaluation=reference_evaluation,
        )
        urgency_weights = self._group_urgency_weights(
            reference_seed_set,
            weak_group_context=weak_group_context,
            evaluation=reference_evaluation,
            coverage_levels=coverage_levels,
        )
        return reference_seed_set, coverage_levels, urgency_weights

    def _cheap_prefilter_score(
        self,
        node_id: Any,
        group_counts: dict[str, int],
        community_counts: dict[int, int],
        ml_bias_weight: float = 0.0,
        weak_group_context: WeakGroupContext | None = None,
        fairness_weight: float = 0.0,
        zero_bonus_weight: float = 0.0,
        bridge_weight: float = 0.0,
        centrality_weight: float = 0.0,
        coverage_levels: dict[str, float] | None = None,
        urgency_weights: dict[str, float] | None = None,
    ) -> float:
        score = float(self.node_scores[node_id])
        group_name = self.node_group_by_node[node_id]
        community_id = self.community_result.community_id_by_node[node_id]
        if group_counts.get(group_name, 0) == 0:
            score += 0.20
        if not self.config.disable_community_aware_mutation and community_counts.get(community_id, 0) == 0:
            score += 0.20
        score += self._ml_bias_score(node_id, ml_bias_weight)
        if weak_group_context is not None and fairness_weight > 0.0:
            score += float(fairness_weight * self._group_support_score(node_id, weak_group_context.weak_groups))
        if weak_group_context is not None and zero_bonus_weight > 0.0 and weak_group_context.zero_covered_groups:
            score += float(
                zero_bonus_weight * self._group_support_score(node_id, weak_group_context.zero_covered_groups)
            )
        if weak_group_context is not None and bridge_weight > 0.0:
            score += float(bridge_weight * self._bridge_to_group_score(node_id, weak_group_context.weak_groups))
        if centrality_weight > 0.0:
            score += float(centrality_weight * self.normalized_node_scores.get(node_id, 0.0))
        if coverage_levels is not None and urgency_weights is not None:
            reach_profile = self.node_group_reach_profile[node_id]
            target_groups = weak_group_context.weak_groups if weak_group_context is not None else self.group_names
            proxy_signal = float(
                np.mean(
                    [
                        max(0.0, min(1.0, coverage_levels[group_name] + reach_profile[group_name]) - coverage_levels[group_name])
                        * urgency_weights[group_name]
                        for group_name in target_groups
                    ]
                )
            ) if target_groups else 0.0
            score += proxy_signal
        _, _, parity_adjustment = self._parity_candidate_adjustment(node_id, coverage_levels)
        score += parity_adjustment
        return score

    def _prefilter_candidate_nodes(
        self,
        candidate_nodes: Sequence[Any],
        limit: int,
        group_counts: dict[str, int],
        community_counts: dict[int, int],
        ml_bias_weight: float = 0.0,
        weak_group_context: WeakGroupContext | None = None,
        fairness_weight: float = 0.0,
        zero_bonus_weight: float = 0.0,
        bridge_weight: float = 0.0,
        centrality_weight: float = 0.0,
        coverage_levels: dict[str, float] | None = None,
        urgency_weights: dict[str, float] | None = None,
    ) -> list[Any]:
        if limit <= 0 or limit >= len(candidate_nodes):
            return list(candidate_nodes)
        ranked = sorted(
            candidate_nodes,
            key=lambda node_id: (
                -self._cheap_prefilter_score(
                    node_id,
                    group_counts=group_counts,
                    community_counts=community_counts,
                    ml_bias_weight=ml_bias_weight,
                    weak_group_context=weak_group_context,
                    fairness_weight=fairness_weight,
                    zero_bonus_weight=zero_bonus_weight,
                    bridge_weight=bridge_weight,
                    centrality_weight=centrality_weight,
                    coverage_levels=coverage_levels,
                    urgency_weights=urgency_weights,
                ),
                _sort_key(node_id),
            ),
        )
        return ranked[:limit]

    def _resolved_proxy_score_weights(self) -> dict[str, float]:
        weights = {
            "delta_mf": float(self.config.local_search_delta_mf_weight),
            "delta_dcv": float(self.config.local_search_delta_dcv_weight),
            "spread": float(self.config.marginal_gain_spread_weight),
            "overlap": float(self.config.local_search_overlap_penalty_weight),
            "ml": float(self.config.ml_local_search_bias_weight if self._ml_guidance_enabled() else 0.0),
            "bridge": float(
                self.config.bridge_to_weak_group_weight if self._worst_group_local_search_active() else 0.0
            ),
            "community_diversity": 0.0,
            "diversity": 0.0,
        }
        if self.config.proxy_score_weights:
            for key, value in self.config.proxy_score_weights.items():
                weights[key] = float(value)
        return weights

    def _partial_counts_after_removal(
        self,
        seed_set: Sequence[Any],
        node_to_remove: Any,
    ) -> tuple[tuple[Any, ...], dict[str, int], dict[int, int]]:
        partial_seed_set = tuple(node_id for node_id in seed_set if node_id != node_to_remove)
        group_counts = self._selected_group_counts(partial_seed_set)
        community_counts = self._selected_community_counts(partial_seed_set)
        return partial_seed_set, group_counts, community_counts

    def _swap_proxy_score(
        self,
        node_id: Any,
        partial_seed_set: Sequence[Any],
        group_counts: dict[str, int],
        community_counts: dict[int, int],
        weak_group_context: WeakGroupContext | None,
        coverage_levels: dict[str, float],
        urgency_weights: dict[str, float],
        proxy_weights: dict[str, float],
    ) -> float:
        score = self._cheap_prefilter_score(
            node_id,
            group_counts=group_counts,
            community_counts=community_counts,
            ml_bias_weight=proxy_weights.get("ml", 0.0),
            weak_group_context=weak_group_context,
            fairness_weight=self.config.weakest_group_mutation_weight if self._worst_group_local_search_active() else 0.0,
            zero_bonus_weight=self.config.zero_group_bonus_weight if self._worst_group_local_search_active() else 0.0,
            bridge_weight=proxy_weights.get("bridge", 0.0),
            centrality_weight=0.0,
            coverage_levels=coverage_levels,
            urgency_weights=urgency_weights,
        )
        score += self._marginal_gain_proxy_score(
            node_id,
            seed_set=partial_seed_set,
            community_counts=community_counts,
            weak_group_context=weak_group_context,
            delta_mf_weight=proxy_weights.get("delta_mf", 0.0),
            delta_dcv_weight=proxy_weights.get("delta_dcv", 0.0),
            spread_weight=proxy_weights.get("spread", 0.0),
            overlap_weight=proxy_weights.get("overlap", 0.0),
            coverage_levels=coverage_levels,
            urgency_weights=urgency_weights,
        )
        if proxy_weights.get("diversity", 0.0) > 0.0:
            score += float(
                proxy_weights["diversity"] * self._graph_diversity_score(node_id, partial_seed_set)
            )
        if (
            proxy_weights.get("community_diversity", 0.0) > 0.0
            and not self.config.disable_community_aware_mutation
        ):
            community_id = self.community_result.community_id_by_node[node_id]
            if community_counts.get(community_id, 0) == 0:
                score += float(proxy_weights["community_diversity"])
        return score

    def _evaluate_seed_set_with_cache_info(
        self,
        seed_set: tuple[Any, ...],
        screening: bool = False,
    ) -> tuple[CandidateEvaluation, bool]:
        normalized_seed_set = self._validate_seed_set(tuple(sorted(seed_set, key=_sort_key)))
        mc_runs = self._screening_mc_runs() if screening and self._staged_mc_active() else self._full_mc_runs()
        cache_key: Any = normalized_seed_set
        target_cache: OrderedDict[Any, CandidateEvaluation] = self.evaluation_cache
        if mc_runs != self._full_mc_runs():
            cache_key = (normalized_seed_set, mc_runs)
            target_cache = self.screening_evaluation_cache
        if self.config.enable_fitness_cache:
            cached = self._cache_lookup(target_cache, cache_key, "fitness_cache_hits")
            if cached is not None:
                return cached, True

        if mc_runs == self._full_mc_runs():
            self.full_evaluation_calls += 1
            self.last_full_mc_runs = mc_runs
        else:
            self.screening_evaluation_calls += 1
            self.last_screening_mc_runs = mc_runs

        if self.search_evaluator is not None:
            evaluation_result = self.search_evaluator.evaluate_seed_set_evaluation(normalized_seed_set)
        else:
            evaluation_result = evaluate_seed_set(
                dataset=self.dataset,
                protected_group_report=self.protected_group_report,
                seed_set=normalized_seed_set,
                propagation_probability=self.config.propagation_probability,
                mc_runs=mc_runs,
                diffusion_model=self.config.diffusion_model,
                random_seed=self.config.random_seed,
                lambda_weight=self.config.lambda_weight,
                include_soft_mf=True,
            )
        evaluation = CandidateEvaluation(
            seed_set=evaluation_result.seed_set,
            total_spread_mean=evaluation_result.total_spread_mean,
            total_spread_std=evaluation_result.total_spread_std,
            fairness=evaluation_result.fairness,
            f_score=evaluation_result.f_score,
            runtime_seconds=evaluation_result.runtime_seconds,
            score=self._fitness_score(evaluation_result),
        )
        if self.config.enable_fitness_cache:
            self._cache_store(target_cache, cache_key, evaluation)
        return evaluation, False

    def _evaluate_swap_candidate(
        self,
        partial_seed_set: Sequence[Any],
        candidate_node: Any,
        screening: bool = False,
    ) -> tuple[CandidateEvaluation, bool]:
        partial_key = _normalize_seed_set(partial_seed_set)
        screening_mc_runs = self._screening_mc_runs() if screening and self._staged_mc_active() else self._full_mc_runs()
        cache_key: Any = (partial_key, candidate_node)
        if screening_mc_runs != self._full_mc_runs():
            cache_key = (partial_key, candidate_node, screening_mc_runs)
        if self.config.enable_swap_cache:
            cached = self._cache_lookup(self.swap_evaluation_cache, cache_key, "swap_cache_hits")
            if cached is not None:
                return cached, True

        candidate_seed_set = self._repair_seed_set(list(partial_key) + [candidate_node])
        evaluation, fitness_cache_hit = self._evaluate_seed_set_with_cache_info(candidate_seed_set, screening=screening)
        if self.config.enable_swap_cache:
            self._cache_store(self.swap_evaluation_cache, cache_key, evaluation)
        return evaluation, fitness_cache_hit

    def _same_community_penalty(
        self,
        node_id: Any,
        community_counts: dict[int, int],
    ) -> float:
        if not self._overlap_penalty_active():
            return 0.0
        community_id = self.community_result.community_id_by_node[node_id]
        return safe_divide(
            float(community_counts.get(community_id, 0)),
            float(self.config.budget),
            default=0.0,
            context="same-community overlap penalty",
        )

    def _neighborhood_overlap_penalty(
        self,
        node_id: Any,
        reference_nodes: Sequence[Any] | set[Any] | None,
    ) -> float:
        if not self._overlap_penalty_active() or not reference_nodes:
            return 0.0

        candidate_neighborhood = set(self.node_neighbors.get(node_id, set()))
        candidate_neighborhood.add(node_id)
        overlap_scores: list[float] = []
        for reference_node in reference_nodes:
            if reference_node == node_id:
                continue
            reference_neighborhood = set(self.node_neighbors.get(reference_node, set()))
            reference_neighborhood.add(reference_node)
            union = candidate_neighborhood | reference_neighborhood
            if not union:
                overlap_scores.append(0.0)
                continue
            overlap_scores.append(float(len(candidate_neighborhood & reference_neighborhood)) / float(len(union)))
        if not overlap_scores:
            return 0.0
        return float(np.mean(overlap_scores))

    def _combined_overlap_penalty(
        self,
        node_id: Any,
        community_counts: dict[int, int],
        reference_nodes: Sequence[Any] | set[Any] | None,
        extra_weight: float = 1.0,
    ) -> float:
        if not self._overlap_penalty_active():
            return 0.0

        same_community = self._same_community_penalty(node_id, community_counts)
        neighborhood_overlap = self._neighborhood_overlap_penalty(node_id, reference_nodes)
        weighted_penalty = (
            self.config.same_community_penalty_weight * same_community
            + self.config.neighborhood_overlap_penalty_weight * neighborhood_overlap
        )
        return float(max(0.0, extra_weight) * weighted_penalty)

    def _marginal_gain_proxy_components(
        self,
        node_id: Any,
        seed_set: Sequence[Any] | set[Any] | None,
        community_counts: dict[int, int],
        weak_group_context: WeakGroupContext | None = None,
        evaluation: CandidateEvaluation | None = None,
        coverage_levels: dict[str, float] | None = None,
        urgency_weights: dict[str, float] | None = None,
    ) -> tuple[float, float, float, float]:
        if seed_set is None:
            seed_set = ()

        seed_sequence = _normalize_seed_set(seed_set)
        if coverage_levels is None:
            coverage_levels = self._normalized_group_coverage(seed_sequence, evaluation=evaluation)
        if urgency_weights is None:
            urgency_weights = self._group_urgency_weights(
                seed_sequence,
                weak_group_context=weak_group_context,
                evaluation=evaluation,
                coverage_levels=coverage_levels,
            )
        weak_groups_key = tuple(weak_group_context.weak_groups) if weak_group_context is not None else ()
        coverage_key = tuple(
            int(round(float(coverage_levels[group_name]) * 1_000_000))
            for group_name in self.group_names
        )
        urgency_key = tuple(
            int(round(float(urgency_weights[group_name]) * 1_000_000))
            for group_name in self.group_names
        )
        cache_key = (
            seed_sequence,
            node_id,
            weak_groups_key,
            coverage_key,
            urgency_key,
        )
        if self.config.enable_marginal_cache:
            cached = self._cache_lookup(self.marginal_gain_cache, cache_key, "marginal_cache_hits")
            if cached is not None:
                return cached

        reach_profile = self.node_group_reach_profile[node_id]
        weak_groups = weak_group_context.weak_groups if weak_group_context is not None else self.group_names

        delta_mf_proxy = float(
            np.mean([
                max(0.0, min(1.0, coverage_levels[group_name] + reach_profile[group_name]) - coverage_levels[group_name])
                * urgency_weights[group_name]
                for group_name in weak_groups
            ])
        ) if weak_groups else 0.0
        dcv_reduction_proxy = float(
            np.mean([
                reach_profile[group_name] * urgency_weights[group_name]
                for group_name in self.group_names
            ])
        )
        spread_proxy = float(self.normalized_node_scores.get(node_id, 0.0))
        overlap_penalty = self._combined_overlap_penalty(
            node_id,
            community_counts,
            seed_sequence,
            extra_weight=1.0,
        )
        components = (delta_mf_proxy, dcv_reduction_proxy, spread_proxy, overlap_penalty)
        if self.config.enable_marginal_cache:
            self._cache_store(self.marginal_gain_cache, cache_key, components)
        return components

    def _marginal_gain_proxy_score(
        self,
        node_id: Any,
        seed_set: Sequence[Any] | set[Any] | None,
        community_counts: dict[int, int],
        weak_group_context: WeakGroupContext | None = None,
        evaluation: CandidateEvaluation | None = None,
        delta_mf_weight: float | None = None,
        delta_dcv_weight: float | None = None,
        spread_weight: float | None = None,
        overlap_weight: float = 1.0,
        coverage_levels: dict[str, float] | None = None,
        urgency_weights: dict[str, float] | None = None,
    ) -> float:
        effective_mf_weight = (
            self.config.marginal_gain_delta_mf_weight
            if delta_mf_weight is None
            else delta_mf_weight
        )
        effective_dcv_weight = (
            self.config.marginal_gain_delta_dcv_weight
            if delta_dcv_weight is None
            else delta_dcv_weight
        )
        effective_spread_weight = (
            self.config.marginal_gain_spread_weight
            if spread_weight is None
            else spread_weight
        )
        delta_mf_proxy, dcv_reduction_proxy, spread_proxy, overlap_penalty = self._marginal_gain_proxy_components(
            node_id,
            seed_set=seed_set,
            community_counts=community_counts,
            weak_group_context=weak_group_context,
            evaluation=evaluation,
            coverage_levels=coverage_levels,
            urgency_weights=urgency_weights,
        )
        return float(
            effective_mf_weight * delta_mf_proxy
            + effective_dcv_weight * dcv_reduction_proxy
            + effective_spread_weight * spread_proxy
            - (overlap_weight * overlap_penalty)
        )

    def _bottom_k_average_group_spread(
        self,
        fairness: FairnessMetrics,
        k: int = 3,
        normalized: bool = False,
    ) -> float:
        spread_map = fairness.normalized_group_spread if normalized else fairness.group_spread
        ordered_values = [
            float(value)
            for _, value in sorted(spread_map.items(), key=lambda item: (float(item[1]), _sort_key(item[0])))
        ]
        if not ordered_values:
            return 0.0
        window = ordered_values[: min(k, len(ordered_values))]
        return float(np.mean(window))

    def _fairness_diagnostics(self, fairness: FairnessMetrics) -> dict[str, float | int | str]:
        covered_groups = [
            group_name
            for group_name, spread in fairness.group_spread.items()
            if float(spread) > 1e-12
        ]
        weakest_groups = [
            group_name
            for group_name, _ in sorted(
                fairness.normalized_group_spread.items(),
                key=lambda item: (float(item[1]), _sort_key(item[0])),
            )[: min(3, len(fairness.normalized_group_spread))]
        ]
        return {
            "zero_covered_groups_count": len(fairness.group_spread) - len(covered_groups),
            "bottom_3_avg_group_spread": self._bottom_k_average_group_spread(fairness, k=3, normalized=False),
            "fraction_groups_covered": safe_divide(
                float(len(covered_groups)),
                float(len(fairness.group_spread)),
                default=0.0,
                context="optimizer fraction of protected groups covered",
            ),
            "weakest_groups_note": ",".join(weakest_groups),
            "parity_target": float(getattr(fairness, "parity_target", 0.0)),
            "parity_abs_error": float(getattr(fairness, "parity_abs_error", 0.0)),
            "parity_squared_error": float(getattr(fairness, "parity_squared_error", 0.0)),
            "over_served_groups": ",".join(str(group) for group in getattr(fairness, "over_served_groups", ())),
            "under_served_groups": ",".join(str(group) for group in getattr(fairness, "under_served_groups", ())),
        }

    def _local_search_rank_key(
        self,
        evaluation: CandidateEvaluation,
    ) -> tuple[float, ...] | tuple[float, float, tuple[tuple[str, str], ...]]:
        if self.config.fitness_policy in {"fairness_first", "professor_priority"}:
            return (
                evaluation.f_score,
                evaluation.fairness.mf,
                -evaluation.fairness.dcv,
                evaluation.total_spread_mean,
                evaluation.score,
            )
        if not self._worst_group_local_search_active():
            return self._candidate_rank_key(evaluation)
        return (
            evaluation.fairness.mf,
            -evaluation.fairness.dcv,
            evaluation.score,
            evaluation.total_spread_mean,
            self._bottom_k_average_group_spread(
                evaluation.fairness,
                k=self.config.local_search_bottom_k_groups,
                normalized=True,
            ),
        )

    def _fairness_first_swap_acceptance_reason(
        self,
        incumbent: CandidateEvaluation,
        candidate: CandidateEvaluation,
    ) -> str | None:
        eps = 1e-12
        if candidate.f_score > incumbent.f_score + eps:
            return "fscore"
        if (
            candidate.fairness.mf > incumbent.fairness.mf + eps
            and candidate.fairness.dcv <= incumbent.fairness.dcv + self.config.fairness_tolerance_dcv
        ):
            return "mf"
        if (
            candidate.total_spread_mean > incumbent.total_spread_mean + eps
            and candidate.f_score >= incumbent.f_score - self.config.fairness_tolerance_fscore_drop
            and candidate.fairness.dcv <= incumbent.fairness.dcv + self.config.fairness_tolerance_dcv
        ):
            if (
                bool(self.config.swap_reject_spread_gain_if_fairness_collapses)
                and float(candidate.f_score) < 0.0
                and float(candidate.f_score) < float(incumbent.f_score)
            ):
                return None
            return "spread"
        return None

    def _dcv_first_swap_acceptance_reason(
        self,
        incumbent: CandidateEvaluation,
        candidate: CandidateEvaluation,
    ) -> str | None:
        eps = 1e-12
        dcv_delta = float(candidate.fairness.dcv) - float(incumbent.fairness.dcv)
        mf_delta = float(candidate.fairness.mf) - float(incumbent.fairness.mf)
        fscore_delta = float(candidate.f_score) - float(incumbent.f_score)
        spread_delta = float(candidate.total_spread_mean) - float(incumbent.total_spread_mean)
        if (
            dcv_delta < -float(self.config.dcv_improvement_epsilon)
            and mf_delta >= -float(self.config.mf_drop_tolerance)
        ):
            return "dcv"
        if (
            fscore_delta > eps
            and dcv_delta <= float(self.config.spread_safe_dcv_tolerance)
        ):
            return "fscore"
        if (
            mf_delta > eps
            and dcv_delta <= float(self.config.spread_safe_dcv_tolerance)
        ):
            return "mf"
        if (
            spread_delta > eps
            and fscore_delta >= -float(self.config.fscore_drop_tolerance)
            and dcv_delta <= float(self.config.spread_safe_dcv_tolerance)
        ):
            return "spread"
        if self.config.use_dcv_minimization:
            parity_error_delta = float(
                getattr(candidate.fairness, "parity_abs_error", 0.0)
            ) - float(getattr(incumbent.fairness, "parity_abs_error", 0.0))
            if (
                parity_error_delta < -float(self.config.parity_error_improvement_epsilon)
                and mf_delta >= -float(self.config.mf_drop_tolerance)
            ):
                return "parity"
        return None

    def _record_swap_acceptance(self) -> None:
        self.swap_accepted += 1
        if self.last_swap_acceptance_reason == "dcv":
            self.swaps_accepted_dcv_improvement += 1
        elif self.last_swap_acceptance_reason == "fscore":
            self.swap_accepted_fscore_improvement += 1
        elif self.last_swap_acceptance_reason == "mf":
            self.swap_accepted_mf_improvement += 1
        elif self.last_swap_acceptance_reason == "spread":
            self.swap_accepted_spread_fairness_preserved += 1
        elif self.last_swap_acceptance_reason == "parity":
            self.swaps_accepted_parity_improvement += 1

    def _record_swap_rejection(
        self,
        candidate: CandidateEvaluation,
        incumbent: CandidateEvaluation,
    ) -> None:
        if self.config.use_fairness_first_swap_acceptance or self.config.use_dcv_first_swap_acceptance:
            self.swap_rejected_fairness_degradation += 1
        if (
            self.config.use_dcv_first_swap_acceptance
            and float(candidate.fairness.dcv)
            > float(incumbent.fairness.dcv) + float(self.config.spread_safe_dcv_tolerance)
        ):
            self.swaps_rejected_dcv_worsening += 1

    def _fairness_first_swap_is_acceptable(
        self,
        incumbent: CandidateEvaluation,
        candidate: CandidateEvaluation,
    ) -> bool:
        return self._fairness_first_swap_acceptance_reason(incumbent, candidate) is not None

    def _shortfall_first_swap_acceptance_reason(
        self,
        incumbent: CandidateEvaluation,
        candidate: CandidateEvaluation,
    ) -> str | None:
        """Accept if shortfall DCV decreases, target coverage improves, or spread
        gains while shortfall does not worsen beyond tolerance."""
        eps = 1e-12
        tol = float(getattr(self.config, "shortfall_dcv_worsen_tolerance",
                            getattr(self.config, "mf_drop_tolerance", 0.001)))
        sf_inc = float(getattr(incumbent.fairness, "dcv_shortfall", incumbent.fairness.dcv))
        sf_can = float(getattr(candidate.fairness, "dcv_shortfall", candidate.fairness.dcv))
        tcr_inc = float(getattr(incumbent.fairness, "target_coverage_ratio", 1.0))
        tcr_can = float(getattr(candidate.fairness, "target_coverage_ratio", 1.0))
        mf_delta = float(candidate.fairness.mf) - float(incumbent.fairness.mf)
        spread_delta = float(candidate.total_spread_mean) - float(incumbent.total_spread_mean)

        if sf_can < sf_inc - eps:
            return "shortfall_dcv"
        if tcr_can > tcr_inc + eps and sf_can <= sf_inc + tol:
            return "target_coverage"
        if mf_delta > eps and sf_can <= sf_inc + tol:
            return "mf"
        if spread_delta > eps and sf_can <= sf_inc + tol:
            return "spread"
        return None

    def _is_local_search_improvement(
        self,
        candidate: CandidateEvaluation,
        incumbent: CandidateEvaluation,
    ) -> bool:
        reason = self._shortfall_first_swap_acceptance_reason(incumbent, candidate)
        self.last_swap_acceptance_reason = "" if reason is None else reason
        return reason is not None
        if self.config.use_dcv_first_swap_acceptance:
            reason = self._dcv_first_swap_acceptance_reason(incumbent, candidate)
            self.last_swap_acceptance_reason = "" if reason is None else reason
            self.last_swap_rejection_reason = "dcv_worsening" if (
                reason is None
                and float(candidate.fairness.dcv)
                > float(incumbent.fairness.dcv) + float(self.config.spread_safe_dcv_tolerance)
            ) else ""
            return reason is not None
        if self.config.use_fairness_first_swap_acceptance:
            reason = self._fairness_first_swap_acceptance_reason(incumbent, candidate)
            self.last_swap_acceptance_reason = "" if reason is None else reason
            return reason is not None
        return self._local_search_rank_key(candidate) > self._local_search_rank_key(incumbent)

    def _confirm_local_search_improvement(
        self,
        current_full_evaluation: CandidateEvaluation,
        screening_candidates: Sequence[tuple[tuple[Any, ...], CandidateEvaluation]],
    ) -> tuple[tuple[Any, ...], CandidateEvaluation] | None:
        if not screening_candidates:
            return None

        unique_candidates: dict[tuple[Any, ...], CandidateEvaluation] = {}
        for candidate_seed_set, screening_evaluation in screening_candidates:
            unique_candidates[candidate_seed_set] = screening_evaluation

        ranked_candidates = sorted(
            unique_candidates.items(),
            key=lambda item: (
                self._local_search_rank_key(item[1]),
                self._seed_sort_key(item[0]),
            ),
            reverse=True,
        )
        for candidate_seed_set, _ in ranked_candidates:
            full_evaluation = self._evaluate_seed_set(candidate_seed_set, screening=False)
            if self._is_local_search_improvement(full_evaluation, current_full_evaluation):
                return candidate_seed_set, full_evaluation
            if self.config.local_search_first_improvement:
                break
        return None

    def _ml_bias_score(self, node_id: Any, weight: float) -> float:
        if weight <= 0.0:
            return 0.0
        return float(weight * self.ml_node_scores.get(node_id, 0.0))

    def _dynamic_candidate_score(
        self,
        node_id: Any,
        group_counts: dict[str, int],
        community_counts: dict[int, int],
        ml_bias_weight: float = 0.0,
        reference_nodes: Sequence[Any] | set[Any] | None = None,
        reference_evaluation: CandidateEvaluation | None = None,
        node2vec_diversity_weight: float | None = None,
        weak_group_context: WeakGroupContext | None = None,
        fairness_weight: float = 0.0,
        zero_bonus_weight: float = 0.0,
        bridge_weight: float = 0.0,
        centrality_weight: float = 0.0,
        diversity_weight: float = 0.0,
        local_search_delta_mf_weight: float = 0.0,
        local_search_delta_dcv_weight: float = 0.0,
        local_search_overlap_penalty_weight: float = 0.0,
        coverage_levels: dict[str, float] | None = None,
        urgency_weights: dict[str, float] | None = None,
    ) -> float:
        score = float(self.node_scores[node_id])
        group_name = self.node_group_by_node[node_id]
        community_id = self.community_result.community_id_by_node[node_id]
        if group_counts.get(group_name, 0) == 0:
            score += 0.20
        if not self.config.disable_community_aware_mutation and community_counts.get(community_id, 0) == 0:
            score += 0.20
        score += self._ml_bias_score(node_id, ml_bias_weight)
        score += self._node2vec_diversity_score(node_id, reference_nodes, node2vec_diversity_weight)
        if weak_group_context is not None and fairness_weight > 0.0:
            score += float(fairness_weight * self._group_support_score(node_id, weak_group_context.weak_groups))
        if weak_group_context is not None and zero_bonus_weight > 0.0 and weak_group_context.zero_covered_groups:
            score += float(
                zero_bonus_weight * self._group_support_score(node_id, weak_group_context.zero_covered_groups)
            )
        if weak_group_context is not None and bridge_weight > 0.0:
            score += float(bridge_weight * self._bridge_to_group_score(node_id, weak_group_context.weak_groups))
        if centrality_weight > 0.0:
            score += float(centrality_weight * self.normalized_node_scores.get(node_id, 0.0))
        if diversity_weight > 0.0:
            score += float(diversity_weight * self._graph_diversity_score(node_id, reference_nodes))
        if self._overlap_penalty_active() and not self._marginal_gain_scoring_active():
            score -= self._combined_overlap_penalty(
                node_id,
                community_counts,
                reference_nodes,
            )
        if self._marginal_gain_scoring_active():
            score += self._marginal_gain_proxy_score(
                node_id,
                seed_set=reference_nodes,
                community_counts=community_counts,
                weak_group_context=weak_group_context,
                evaluation=reference_evaluation,
                coverage_levels=coverage_levels,
                urgency_weights=urgency_weights,
            )
        if (
            local_search_delta_mf_weight > 0.0
            or local_search_delta_dcv_weight > 0.0
            or local_search_overlap_penalty_weight > 0.0
        ):
            score += self._marginal_gain_proxy_score(
                node_id,
                seed_set=reference_nodes,
                community_counts=community_counts,
                weak_group_context=weak_group_context,
                evaluation=reference_evaluation,
                delta_mf_weight=local_search_delta_mf_weight,
                delta_dcv_weight=local_search_delta_dcv_weight,
                spread_weight=0.0,
                overlap_weight=local_search_overlap_penalty_weight,
                coverage_levels=coverage_levels,
                urgency_weights=urgency_weights,
            )
        _, _, parity_adjustment = self._parity_candidate_adjustment(node_id, coverage_levels)
        score += parity_adjustment
        return score

    def _interleave_tier_rankings(
        self,
        primary_nodes: Sequence[Any],
        secondary_nodes: Sequence[Any],
    ) -> list[Any]:
        if not self._uses_two_tier_guidance() or not primary_nodes or not secondary_nodes:
            return list(primary_nodes) + list(secondary_nodes)

        interval = max(1, int(round(1.0 / max(self.config.ml_secondary_exploration_rate, 1e-9))))
        ordered: list[Any] = []
        primary_index = 0
        secondary_index = 0

        while primary_index < len(primary_nodes) or secondary_index < len(secondary_nodes):
            for _ in range(max(1, interval - 1)):
                if primary_index >= len(primary_nodes):
                    break
                ordered.append(primary_nodes[primary_index])
                primary_index += 1
            if secondary_index < len(secondary_nodes):
                ordered.append(secondary_nodes[secondary_index])
                secondary_index += 1
            elif primary_index < len(primary_nodes):
                ordered.append(primary_nodes[primary_index])
                primary_index += 1

        return ordered

    def _order_two_tier_rankings(
        self,
        primary_nodes: Sequence[Any],
        secondary_nodes: Sequence[Any],
        primary_rate: float,
    ) -> list[Any]:
        if not self._uses_two_tier_guidance() or not primary_nodes or not secondary_nodes:
            return list(primary_nodes) + list(secondary_nodes)
        if self._uses_legacy_two_tier_guidance():
            return self._interleave_tier_rankings(primary_nodes, secondary_nodes)

        prefer_primary = self.rng.random() < primary_rate
        if self.rng.random() < self.config.ml_secondary_exploration_rate:
            prefer_primary = False

        preferred = list(primary_nodes if prefer_primary else secondary_nodes)
        fallback = list(secondary_nodes if prefer_primary else primary_nodes)
        return preferred + fallback

    def _rank_candidate_nodes(
        self,
        candidate_nodes: Sequence[Any],
        group_counts: dict[str, int],
        community_counts: dict[int, int],
        ml_bias_weight: float = 0.0,
        primary_rate: float | None = None,
        reference_nodes: Sequence[Any] | set[Any] | None = None,
        reference_evaluation: CandidateEvaluation | None = None,
        node2vec_diversity_weight: float | None = None,
        weak_group_context: WeakGroupContext | None = None,
        fairness_weight: float = 0.0,
        zero_bonus_weight: float = 0.0,
        bridge_weight: float = 0.0,
        centrality_weight: float = 0.0,
        diversity_weight: float = 0.0,
        local_search_delta_mf_weight: float = 0.0,
        local_search_delta_dcv_weight: float = 0.0,
        local_search_overlap_penalty_weight: float = 0.0,
        candidate_limit: int | None = None,
        prefilter_top_k: int | None = None,
    ) -> list[Any]:
        require_proxy_priors = (
            self._marginal_gain_scoring_active()
            or local_search_delta_mf_weight > 0.0
            or local_search_delta_dcv_weight > 0.0
            or self._parity_adjustment_active()
        )
        reference_seed_set, coverage_levels, urgency_weights = self._scoring_priors(
            reference_nodes,
            weak_group_context=weak_group_context,
            reference_evaluation=reference_evaluation,
            require_proxy=require_proxy_priors,
        )
        ranked_input = list(candidate_nodes)
        effective_prefilter_limit = 0
        if prefilter_top_k is not None and prefilter_top_k > 0:
            effective_prefilter_limit = prefilter_top_k
        if candidate_limit is not None and candidate_limit > 0:
            effective_prefilter_limit = max(effective_prefilter_limit, candidate_limit)
        if 0 < effective_prefilter_limit < len(ranked_input):
            ranked_input = self._prefilter_candidate_nodes(
                ranked_input,
                limit=effective_prefilter_limit,
                group_counts=group_counts,
                community_counts=community_counts,
                ml_bias_weight=ml_bias_weight,
                weak_group_context=weak_group_context,
                fairness_weight=fairness_weight,
                zero_bonus_weight=zero_bonus_weight,
                bridge_weight=bridge_weight,
                centrality_weight=centrality_weight,
                coverage_levels=coverage_levels,
                urgency_weights=urgency_weights,
            )
        if self._uses_two_tier_guidance():
            primary_candidates = [
                node_id for node_id in ranked_input
                if node_id in self.primary_ml_pool_set
            ]
            secondary_candidates = [
                node_id for node_id in ranked_input
                if node_id not in self.primary_ml_pool_set
            ]
            primary_ranked = sorted(
                primary_candidates,
                key=lambda node_id: (
                    -self._dynamic_candidate_score(
                        node_id,
                        group_counts,
                        community_counts,
                        ml_bias_weight,
                        reference_nodes=reference_seed_set,
                        reference_evaluation=reference_evaluation,
                        node2vec_diversity_weight=node2vec_diversity_weight,
                        weak_group_context=weak_group_context,
                        fairness_weight=fairness_weight,
                        zero_bonus_weight=zero_bonus_weight,
                        bridge_weight=bridge_weight,
                        centrality_weight=centrality_weight,
                        diversity_weight=diversity_weight,
                        local_search_delta_mf_weight=local_search_delta_mf_weight,
                        local_search_delta_dcv_weight=local_search_delta_dcv_weight,
                        local_search_overlap_penalty_weight=local_search_overlap_penalty_weight,
                        coverage_levels=coverage_levels,
                        urgency_weights=urgency_weights,
                    ),
                    _sort_key(node_id),
                ),
            )
            secondary_ranked = sorted(
                secondary_candidates,
                key=lambda node_id: (
                    -self._dynamic_candidate_score(
                        node_id,
                        group_counts,
                        community_counts,
                        ml_bias_weight,
                        reference_nodes=reference_seed_set,
                        reference_evaluation=reference_evaluation,
                        node2vec_diversity_weight=node2vec_diversity_weight,
                        weak_group_context=weak_group_context,
                        fairness_weight=fairness_weight,
                        zero_bonus_weight=zero_bonus_weight,
                        bridge_weight=bridge_weight,
                        centrality_weight=centrality_weight,
                        diversity_weight=diversity_weight,
                        local_search_delta_mf_weight=local_search_delta_mf_weight,
                        local_search_delta_dcv_weight=local_search_delta_dcv_weight,
                        local_search_overlap_penalty_weight=local_search_overlap_penalty_weight,
                        coverage_levels=coverage_levels,
                        urgency_weights=urgency_weights,
                    ),
                    _sort_key(node_id),
                ),
            )
            effective_primary_rate = (
                1.0 - self.config.ml_secondary_exploration_rate
                if primary_rate is None
                else primary_rate
            )
            ordered = self._order_two_tier_rankings(
                primary_ranked,
                secondary_ranked,
                primary_rate=effective_primary_rate,
            )
            if candidate_limit is not None and 0 < candidate_limit < len(ordered):
                return ordered[:candidate_limit]
            return ordered

        ordered = sorted(
            ranked_input,
            key=lambda node_id: (
                -self._dynamic_candidate_score(
                    node_id,
                    group_counts,
                    community_counts,
                    ml_bias_weight,
                    reference_nodes=reference_seed_set,
                    reference_evaluation=reference_evaluation,
                    node2vec_diversity_weight=node2vec_diversity_weight,
                    weak_group_context=weak_group_context,
                    fairness_weight=fairness_weight,
                    zero_bonus_weight=zero_bonus_weight,
                    bridge_weight=bridge_weight,
                    centrality_weight=centrality_weight,
                    diversity_weight=diversity_weight,
                    local_search_delta_mf_weight=local_search_delta_mf_weight,
                    local_search_delta_dcv_weight=local_search_delta_dcv_weight,
                    local_search_overlap_penalty_weight=local_search_overlap_penalty_weight,
                    coverage_levels=coverage_levels,
                    urgency_weights=urgency_weights,
                ),
                _sort_key(node_id),
            ),
        )
        if candidate_limit is not None and 0 < candidate_limit < len(ordered):
            return ordered[:candidate_limit]
        return ordered

    def _rank_external_candidates(
        self,
        seed_set: Sequence[Any],
        ml_bias_weight: float = 0.0,
        primary_rate: float | None = None,
        node2vec_diversity_weight: float | None = None,
        weak_group_context: WeakGroupContext | None = None,
        reference_evaluation: CandidateEvaluation | None = None,
        fairness_weight: float = 0.0,
        zero_bonus_weight: float = 0.0,
        bridge_weight: float = 0.0,
        centrality_weight: float = 0.0,
        diversity_weight: float = 0.0,
        local_search_delta_mf_weight: float = 0.0,
        local_search_delta_dcv_weight: float = 0.0,
        local_search_overlap_penalty_weight: float = 0.0,
        stage: str = "marginal",
    ) -> list[Any]:
        selected_nodes = set(seed_set)
        group_counts = self._selected_group_counts(seed_set)
        community_counts = self._selected_community_counts(seed_set)
        available_nodes = [node_id for node_id in self.candidate_pool if node_id not in selected_nodes]
        raw_candidate_limit = self._candidate_pool_limit(len(available_nodes), stage=stage)
        prefilter_top_k = self._candidate_prefilter_limit(len(available_nodes), raw_candidate_limit, stage=stage)
        candidate_limit = (
            raw_candidate_limit
            if 0 < raw_candidate_limit < len(available_nodes)
            else None
        )
        return self._rank_candidate_nodes(
            available_nodes,
            group_counts=group_counts,
            community_counts=community_counts,
            ml_bias_weight=ml_bias_weight,
            primary_rate=primary_rate,
            reference_nodes=seed_set,
            reference_evaluation=reference_evaluation,
            node2vec_diversity_weight=node2vec_diversity_weight,
            weak_group_context=weak_group_context,
            fairness_weight=fairness_weight,
            zero_bonus_weight=zero_bonus_weight,
            bridge_weight=bridge_weight,
            centrality_weight=centrality_weight,
            diversity_weight=diversity_weight,
            local_search_delta_mf_weight=local_search_delta_mf_weight,
            local_search_delta_dcv_weight=local_search_delta_dcv_weight,
            local_search_overlap_penalty_weight=local_search_overlap_penalty_weight,
            candidate_limit=candidate_limit,
            prefilter_top_k=prefilter_top_k,
        )

    def _rank_fairness_first_candidates(
        self,
        seed_set: Sequence[Any],
        ml_bias_weight: float = 0.0,
        primary_rate: float | None = None,
    ) -> list[Any]:
        fairness_context = self._weak_group_context_for_seed_set(
            seed_set,
            weakest_group_k=max(1, self.config.fairness_first_init_slots),
        )
        return self._rank_external_candidates(
            seed_set,
            ml_bias_weight=ml_bias_weight,
            primary_rate=primary_rate,
            weak_group_context=fairness_context,
            fairness_weight=self.config.fairness_first_init_weight,
            stage="marginal",
        )

    def _rank_seed_nodes_for_replacement(
        self,
        seed_set: Sequence[Any],
        ml_bias_weight: float = 0.0,
        weak_group_context: WeakGroupContext | None = None,
    ) -> list[Any]:
        group_counts = self._selected_group_counts(seed_set)
        community_counts = self._selected_community_counts(seed_set)

        def replacement_priority(node_id: Any) -> tuple[float, tuple[str, str]]:
            score = float(self.node_scores[node_id])
            group_name = self.node_group_by_node[node_id]
            community_id = self.community_result.community_id_by_node[node_id]
            if group_counts.get(group_name, 0) <= 1:
                score += 0.20
            if not self.config.disable_community_aware_mutation and community_counts.get(community_id, 0) <= 1:
                score += 0.20
            score += self._ml_bias_score(node_id, ml_bias_weight)
            if weak_group_context is not None and self._worst_group_local_search_active():
                score -= self._group_support_score(node_id, weak_group_context.weak_groups)
                score -= self._bridge_to_group_score(node_id, weak_group_context.weak_groups)
            if self._overlap_penalty_active():
                score -= self._combined_overlap_penalty(
                    node_id,
                    community_counts,
                    [other_node for other_node in seed_set if other_node != node_id],
                )
            return (score, _sort_key(node_id))

        return sorted(seed_set, key=replacement_priority)

    def _sample_unused_node(
        self,
        community_nodes: Sequence[Any],
        used_nodes: set[Any],
        ml_bias_weight: float = 0.0,
        primary_rate: float | None = None,
        node2vec_diversity_weight: float | None = None,
        weak_group_context: WeakGroupContext | None = None,
        fairness_weight: float = 0.0,
        zero_bonus_weight: float = 0.0,
        bridge_weight: float = 0.0,
        centrality_weight: float = 0.0,
        diversity_weight: float = 0.0,
        candidate_limit: int | None = None,
        prefilter_top_k: int | None = None,
    ) -> Any | None:
        available_nodes = [node for node in community_nodes if node not in used_nodes]
        if not available_nodes:
            return None

        used_seed_set = tuple(sorted(used_nodes, key=_sort_key))
        raw_candidate_limit = (
            self._candidate_pool_limit(len(available_nodes), stage="repair")
            if candidate_limit is None
            else max(1, min(len(available_nodes), candidate_limit))
        )
        effective_candidate_limit = (
            raw_candidate_limit
            if 0 < raw_candidate_limit < len(available_nodes)
            else None
        )
        effective_prefilter_top_k = (
            self._candidate_prefilter_limit(len(available_nodes), raw_candidate_limit, stage="repair")
            if prefilter_top_k is None
            else max(
                1,
                min(
                    len(available_nodes),
                    max(
                        effective_candidate_limit if effective_candidate_limit is not None else 0,
                        prefilter_top_k,
                    ),
                ),
            )
        )

        ranked_nodes = self._rank_candidate_nodes(
            available_nodes,
            group_counts=self._selected_group_counts(used_seed_set),
            community_counts=self._selected_community_counts(used_seed_set),
            ml_bias_weight=ml_bias_weight,
            primary_rate=primary_rate,
            reference_nodes=used_nodes,
            node2vec_diversity_weight=node2vec_diversity_weight,
            weak_group_context=weak_group_context,
            fairness_weight=fairness_weight,
            zero_bonus_weight=zero_bonus_weight,
            bridge_weight=bridge_weight,
            centrality_weight=centrality_weight,
            diversity_weight=diversity_weight,
            candidate_limit=effective_candidate_limit,
            prefilter_top_k=effective_prefilter_top_k,
        )

        top_candidates = tuple(ranked_nodes[: min(3, len(ranked_nodes))])
        return sample_node_from_community(top_candidates, self.rng)

    def _sample_repair_node(
        self,
        used_nodes: set[Any],
        ml_bias_weight: float = 0.0,
        primary_rate: float | None = None,
        node2vec_diversity_weight: float | None = None,
        weak_group_context: WeakGroupContext | None = None,
        fairness_weight: float = 0.0,
        zero_bonus_weight: float = 0.0,
        bridge_weight: float = 0.0,
        centrality_weight: float = 0.0,
        diversity_weight: float = 0.0,
        candidate_limit: int | None = None,
        prefilter_top_k: int | None = None,
    ) -> Any | None:
        represented_communities = {
            self.community_result.community_id_by_node[node_id]
            for node_id in used_nodes
            if node_id in self.community_result.community_id_by_node
        }

        preferred = {
            community_id: community_nodes
            for community_id, community_nodes in self.available_communities.items()
            if community_id not in represented_communities
            and any(node not in used_nodes for node in community_nodes)
        }
        candidate_communities = preferred
        if not candidate_communities:
            candidate_communities = {
                community_id: community_nodes
                for community_id, community_nodes in self.available_communities.items()
                if any(node not in used_nodes for node in community_nodes)
            }
        if not candidate_communities:
            return None

        candidate_sizes = {
            community_id: sum(1 for node in community_nodes if node not in used_nodes)
            for community_id, community_nodes in candidate_communities.items()
        }
        community_id = sample_community(candidate_communities, candidate_sizes, self.rng)
        return self._sample_unused_node(
            candidate_communities[community_id],
            used_nodes,
            ml_bias_weight=ml_bias_weight,
            primary_rate=primary_rate,
            node2vec_diversity_weight=node2vec_diversity_weight,
            weak_group_context=weak_group_context,
            fairness_weight=fairness_weight,
            zero_bonus_weight=zero_bonus_weight,
            bridge_weight=bridge_weight,
            centrality_weight=centrality_weight,
            diversity_weight=diversity_weight,
            candidate_limit=candidate_limit,
            prefilter_top_k=prefilter_top_k,
        )

    def _apply_large_group_quota_overlay(self, targets: dict[str, int]) -> dict[str, int]:
        """Raise quota targets for groups that exceed large_group_size_threshold.

        Applied on top of whatever _quota_target_counts() returns (including {}).
        Non-large groups are reduced first when the total overshoots budget.
        """
        _large_ratio = float(getattr(self.config, "large_group_min_quota_ratio", 0.0))
        _large_threshold = float(getattr(self.config, "large_group_size_threshold", 0.20))
        if _large_ratio <= 0.0:
            return targets
        total_nodes = float(sum(int(sz) for sz in self.protected_group_report.group_sizes.values())) or 1.0
        result = dict(targets)
        for group_name in self.group_names:
            sz = float(self.protected_group_report.group_sizes.get(group_name, 0))
            size_ratio = sz / total_nodes
            if size_ratio >= _large_threshold:
                min_q = max(1, math.ceil(float(self.config.budget) * size_ratio * _large_ratio))
                result[group_name] = max(result.get(group_name, 0), min_q)
        # Clamp to budget: reduce non-large groups first, then any group.
        while sum(result.values()) > self.config.budget:
            non_large = [
                g for g, t in result.items()
                if t > 0 and float(self.protected_group_report.group_sizes.get(g, 0)) / total_nodes < _large_threshold
            ]
            reducible = non_large or [g for g, t in result.items() if t > 0]
            if not reducible:
                break
            g_r = min(reducible, key=lambda g: (result[g], _sort_key(g)))
            result[g_r] -= 1
            if result[g_r] <= 0:
                del result[g_r]
        return {g: t for g, t in result.items() if t > 0}

    def _quota_target_counts(self) -> dict[str, int]:
        if (
            not bool(self.config.use_group_quota_repair)
            or self.config.quota_mode == "none"
            or self.config.min_seeds_per_protected_group <= 0
        ):
            return {}

        minimum = max(0, int(self.config.min_seeds_per_protected_group))
        if minimum <= 0:
            return {}
        if self.config.quota_mode == "support_aware":
            eligible_groups = [
                group_name
                for group_name in self.group_names
                if int(self.protected_group_report.group_sizes[group_name]) >= int(self.config.quota_min_group_support)
            ]
        else:
            eligible_groups = [
                group_name
                for group_name in self.group_names
                if int(self.protected_group_report.group_sizes[group_name]) > 0
            ]
        if not eligible_groups:
            return {}

        if self.config.quota_mode in {"at_least_one", "support_aware"}:
            ordered_groups = sorted(
                eligible_groups,
                key=lambda group_name: (
                    int(self.protected_group_report.group_sizes[group_name]),
                    _sort_key(group_name),
                ),
            )
            max_groups = max(0, self.config.budget // minimum)
            return {
                group_name: minimum
                for group_name in ordered_groups[:max_groups]
            }

        total_support = float(sum(int(self.protected_group_report.group_sizes[group_name]) for group_name in eligible_groups))
        if total_support <= 0.0:
            return {}
        raw_targets = {
            group_name: float(self.config.budget) * float(self.protected_group_report.group_sizes[group_name]) / total_support
            for group_name in eligible_groups
        }
        targets = {
            group_name: int(math.floor(raw_target))
            for group_name, raw_target in raw_targets.items()
        }
        remaining = int(self.config.budget) - int(sum(targets.values()))
        for group_name, _ in sorted(
            raw_targets.items(),
            key=lambda item: (-(item[1] - math.floor(item[1])), _sort_key(item[0])),
        ):
            if remaining <= 0:
                break
            targets[group_name] = targets.get(group_name, 0) + 1
            remaining -= 1
        for group_name in sorted(eligible_groups, key=lambda name: (targets.get(name, 0), _sort_key(name))):
            if sum(targets.values()) >= self.config.budget:
                break
            targets[group_name] = max(targets.get(group_name, 0), minimum)
        while sum(targets.values()) > self.config.budget:
            reducible = [
                group_name
                for group_name, target in targets.items()
                if target > 0
            ]
            if not reducible:
                break
            group_to_reduce = max(
                reducible,
                key=lambda group_name: (
                    targets[group_name],
                    int(self.protected_group_report.group_sizes[group_name]),
                    _sort_key(group_name),
                ),
            )
            targets[group_to_reduce] -= 1
        return {group_name: target for group_name, target in targets.items() if target > 0}

    def _initialization_quota_target_counts(self) -> dict[str, int]:
        if not bool(self.config.use_protected_group_quota_initialization):
            return {}
        eligible_groups = [
            group_name
            for group_name in self.group_names
            if int(self.protected_group_report.group_sizes[group_name]) > 0
            and any(node_id in self.candidate_pool_set for node_id in self.protected_group_report.protected_groups.get(group_name, ()))
        ]
        if not eligible_groups:
            return {}

        budget = int(self.config.budget)
        mode = str(self.config.initialization_quota_mode)
        if mode == "uniform_min":
            targets = {group_name: 0 for group_name in eligible_groups}
            for group_name in sorted(
                eligible_groups,
                key=lambda name: (int(self.protected_group_report.group_sizes[name]), _sort_key(name)),
            ):
                if sum(targets.values()) >= budget:
                    break
                targets[group_name] = 1
        else:
            if mode == "sqrt":
                weights = {
                    group_name: math.sqrt(float(self.protected_group_report.group_sizes[group_name]))
                    for group_name in eligible_groups
                }
            else:
                weights = {
                    group_name: float(self.protected_group_report.group_sizes[group_name])
                    for group_name in eligible_groups
                }
            total_weight = sum(weights.values())
            if total_weight <= 0.0:
                return {}
            raw_targets = {
                group_name: float(budget) * weight / total_weight
                for group_name, weight in weights.items()
            }
            targets = {
                group_name: int(math.floor(raw_target))
                for group_name, raw_target in raw_targets.items()
            }
            remaining = budget - sum(targets.values())
            for group_name, _ in sorted(
                raw_targets.items(),
                key=lambda item: (-(item[1] - math.floor(item[1])), _sort_key(item[0])),
            ):
                if remaining <= 0:
                    break
                targets[group_name] = targets.get(group_name, 0) + 1
                remaining -= 1

        if budget >= len(eligible_groups):
            for group_name in eligible_groups:
                targets[group_name] = max(1, targets.get(group_name, 0))

        if eligible_groups and self.config.small_group_seed_fraction > 0.0:
            smallest_group = min(
                eligible_groups,
                key=lambda group_name: (int(self.protected_group_report.group_sizes[group_name]), _sort_key(group_name)),
            )
            min_small_group_quota = max(1, int(math.ceil(float(budget) * float(self.config.small_group_seed_fraction))))
            min_small_group_quota = min(min_small_group_quota, budget)
            targets[smallest_group] = max(targets.get(smallest_group, 0), min_small_group_quota)

        while sum(targets.values()) > budget:
            reducible = [
                group_name
                for group_name, target in targets.items()
                if target > (1 if budget >= len(eligible_groups) else 0)
            ]
            if not reducible:
                reducible = [group_name for group_name, target in targets.items() if target > 0]
            if not reducible:
                break
            group_to_reduce = max(
                reducible,
                key=lambda group_name: (
                    targets[group_name],
                    int(self.protected_group_report.group_sizes[group_name]),
                    _sort_key(group_name),
                ),
            )
            targets[group_to_reduce] -= 1

        return {group_name: int(target) for group_name, target in targets.items() if int(target) > 0}

    def _quota_initial_seed_nodes(
        self,
        current_nodes: Sequence[Any],
        use_ml_bias: bool,
    ) -> list[Any]:
        targets = self._initialization_quota_target_counts()
        self.seed_quota_per_protected_group = {str(group): int(target) for group, target in targets.items()}
        if not targets:
            return []
        seed_nodes = list(current_nodes)
        for group_name, target in sorted(
            targets.items(),
            key=lambda item: (int(self.protected_group_report.group_sizes[item[0]]), _sort_key(item[0])),
        ):
            while len(seed_nodes) < self.config.budget and self._selected_group_counts(seed_nodes).get(group_name, 0) < int(target):
                ranked_candidates = self._rank_quota_group_candidates(group_name, seed_nodes)
                if not ranked_candidates:
                    break
                seed_nodes.append(ranked_candidates[0])
        return seed_nodes

    def _summarize_initial_population_coverage(self, population: Sequence[tuple[Any, ...]]) -> dict[str, int]:
        summary = {str(group_name): 0 for group_name in self.group_names}
        for seed_set in population:
            group_counts = self._selected_group_counts(seed_set)
            for group_name, count in group_counts.items():
                if count > 0:
                    summary[str(group_name)] = summary.get(str(group_name), 0) + 1
        return summary

    def _rank_quota_group_candidates(
        self,
        group_name: str,
        current_nodes: Sequence[Any],
    ) -> list[Any]:
        used_nodes = set(current_nodes)
        group_nodes = [
            node_id
            for node_id in self.protected_group_report.protected_groups.get(group_name, ())
            if node_id in self.candidate_pool_set and node_id not in used_nodes
        ]
        if not group_nodes:
            return []
        group_counts = self._selected_group_counts(current_nodes)
        community_counts = self._selected_community_counts(current_nodes)
        weak_context = WeakGroupContext(
            weak_groups=(group_name,),
            zero_covered_groups=(group_name,) if group_counts.get(group_name, 0) <= 0 else (),
        )
        return self._rank_candidate_nodes(
            group_nodes,
            group_counts=group_counts,
            community_counts=community_counts,
            ml_bias_weight=self.config.ml_repair_bias_weight if self._ml_guidance_enabled() else 0.0,
            primary_rate=self.config.ml_repair_primary_rate if self._uses_tuned_two_tier_guidance() else None,
            reference_nodes=current_nodes,
            weak_group_context=weak_context,
            fairness_weight=max(1.0, float(self.config.repair_fairness_weight)),
            zero_bonus_weight=max(0.5, float(self.config.zero_group_bonus_weight)),
            bridge_weight=max(0.25, float(self.config.repair_bridge_weight)),
            centrality_weight=self.config.repair_centrality_weight,
            diversity_weight=self.config.repair_diversity_weight,
            candidate_limit=self._candidate_pool_limit(len(group_nodes), stage="repair"),
            prefilter_top_k=self._candidate_prefilter_limit(
                len(group_nodes),
                self._candidate_pool_limit(len(group_nodes), stage="repair"),
                stage="repair",
            ),
        )

    def _quota_removal_candidate(
        self,
        current_nodes: Sequence[Any],
        target_group: str,
        target_counts: dict[str, int],
        deficit_groups: set[str],
    ) -> Any | None:
        group_counts = self._selected_group_counts(current_nodes)
        replacement_order = self._rank_seed_nodes_for_replacement(
            current_nodes,
            ml_bias_weight=self.config.ml_repair_bias_weight if self._ml_guidance_enabled() else 0.0,
            weak_group_context=WeakGroupContext(weak_groups=tuple(sorted(deficit_groups)), zero_covered_groups=()),
        )
        for node_id in replacement_order:
            group_name = self.node_group_by_node[node_id]
            if group_name == target_group or group_name in deficit_groups:
                continue
            protected_target = int(target_counts.get(group_name, 0))
            if group_counts.get(group_name, 0) > max(protected_target, 0):
                return node_id
        for node_id in replacement_order:
            if self.node_group_by_node[node_id] != target_group:
                return node_id
        return None

    def _apply_group_quota_repair(self, cleaned: Sequence[Any]) -> list[Any]:
        target_counts = self._quota_target_counts()
        if not target_counts:
            return list(cleaned)

        repaired = list(cleaned[: self.config.budget])
        failed_groups: set[str] = set()
        max_attempts = max(1, self.config.budget * max(1, len(target_counts)))
        for _ in range(max_attempts):
            group_counts = self._selected_group_counts(repaired)
            deficit_groups = {
                group_name
                for group_name, target in target_counts.items()
                if group_counts.get(group_name, 0) < int(target) and group_name not in failed_groups
            }
            if not deficit_groups:
                break
            target_group = min(
                deficit_groups,
                key=lambda group_name: (
                    group_counts.get(group_name, 0) / max(1, int(target_counts[group_name])),
                    int(self.protected_group_report.group_sizes[group_name]),
                    _sort_key(group_name),
                ),
            )
            ranked_candidates = self._rank_quota_group_candidates(target_group, repaired)
            if not ranked_candidates:
                failed_groups.add(target_group)
                continue
            replacement = ranked_candidates[0]
            if len(repaired) >= self.config.budget:
                node_to_remove = self._quota_removal_candidate(
                    repaired,
                    target_group,
                    target_counts,
                    deficit_groups,
                )
                if node_to_remove is None:
                    failed_groups.add(target_group)
                    continue
                repaired = [node_id for node_id in repaired if node_id != node_to_remove]
            if replacement not in repaired:
                repaired.append(replacement)
        return repaired

    def _majority_overconcentrated_group(self, current_nodes: Sequence[Any]) -> str | None:
        if not current_nodes:
            return None
        group_counts = self._selected_group_counts(current_nodes)
        group_name, count = max(
            group_counts.items(),
            key=lambda item: (item[1], int(self.protected_group_report.group_sizes[item[0]]), _sort_key(item[0])),
        )
        if safe_divide(
            float(count),
            float(max(1, len(current_nodes))),
            default=0.0,
            context="majority overconcentration repair ratio",
        ) > float(self.config.majority_overconcentration_threshold):
            return group_name
        return None

    def _group_overshoot_cap_met(self, group_name: str, group_counts: dict[str, int]) -> bool:
        """Return True when group_name has accumulated >= cap × proportional_seed_target seeds.

        Uses seed count as a proxy for influence.  Disabled when weak_group_overshoot_cap <= 0.
        """
        cap = float(self.config.weak_group_overshoot_cap)
        if cap <= 0.0:
            return False
        total_nodes = float(sum(int(sz) for sz in self.protected_group_report.group_sizes.values()))
        if total_nodes <= 0.0:
            return False
        group_size = float(self.protected_group_report.group_sizes.get(group_name, 0))
        proportional_target = float(self.config.budget) * group_size / total_nodes
        if proportional_target <= 0.0:
            return False
        return float(group_counts.get(group_name, 0)) >= cap * proportional_target

    def _weak_group_repair_targets(self, current_nodes: Sequence[Any]) -> list[str]:
        group_counts = self._selected_group_counts(current_nodes)
        target_counts = self._apply_large_group_quota_overlay(self._quota_target_counts())
        deficit_groups: list[str] = []
        for group_name, target in target_counts.items():
            if group_counts.get(group_name, 0) < int(target):
                if self._group_overshoot_cap_met(group_name, group_counts):
                    # Cap prevents this repair — count as a redirected slot.
                    self.seeds_redirected_by_overshoot_cap += 1
                else:
                    deficit_groups.append(group_name)
        if deficit_groups:
            return sorted(
                deficit_groups,
                key=lambda group_name: (
                    group_counts.get(group_name, 0) / max(1, int(target_counts[group_name])),
                    int(self.protected_group_report.group_sizes[group_name]),
                    _sort_key(group_name),
                ),
            )
        return sorted(
            [g for g in self.group_names if not self._group_overshoot_cap_met(g, group_counts)],
            key=lambda group_name: (
                safe_divide(
                    float(group_counts.get(group_name, 0)),
                    float(max(1, int(self.protected_group_report.group_sizes[group_name]))),
                    default=0.0,
                    context=f"weak-group repair normalized seed count for {group_name}",
                ),
                group_counts.get(group_name, 0),
                int(self.protected_group_report.group_sizes[group_name]),
                _sort_key(group_name),
            ),
        )

    def _repair_removal_for_weak_group(
        self,
        current_nodes: Sequence[Any],
        target_group: str,
    ) -> Any | None:
        majority_group = self._majority_overconcentrated_group(current_nodes)
        group_counts = self._selected_group_counts(current_nodes)
        replacement_order = self._rank_seed_nodes_for_replacement(
            current_nodes,
            ml_bias_weight=self.config.ml_repair_bias_weight if self._ml_guidance_enabled() else 0.0,
            weak_group_context=WeakGroupContext(weak_groups=(target_group,), zero_covered_groups=()),
        )
        if majority_group is not None and majority_group != target_group:
            for node_id in replacement_order:
                if self.node_group_by_node[node_id] == majority_group and group_counts.get(majority_group, 0) > 1:
                    return node_id
        target_counts = self._quota_target_counts()
        return self._quota_removal_candidate(
            current_nodes,
            target_group,
            target_counts,
            {target_group},
        )

    def _apply_fairness_first_repair(self, cleaned: Sequence[Any]) -> list[Any]:
        if not bool(self.config.use_fairness_first_repair) or self.config.weak_group_repair_rounds <= 0:
            return list(cleaned)
        repaired = list(cleaned[: self.config.budget])
        for _ in range(int(self.config.weak_group_repair_rounds)):
            self.repair_attempts += 1
            changed = False
            for target_group in self._weak_group_repair_targets(repaired):
                ranked_candidates = self._rank_quota_group_candidates(target_group, repaired)
                if not ranked_candidates:
                    continue
                replacement = ranked_candidates[0]
                if replacement in repaired:
                    continue
                if len(repaired) < self.config.budget:
                    repaired.append(replacement)
                    changed = True
                else:
                    node_to_remove = self._repair_removal_for_weak_group(repaired, target_group)
                    if node_to_remove is None:
                        continue
                    repaired = [node_id for node_id in repaired if node_id != node_to_remove]
                    repaired.append(replacement)
                    changed = True
                if changed:
                    self.weak_group_repairs += 1
                    break
            if not changed:
                break
        return repaired

    def _repair_seed_set(
        self,
        proposed_nodes: Sequence[Any],
        use_node2vec_diversity: bool = True,
    ) -> tuple[Any, ...]:
        self.repaired_seed_sets += 1
        cleaned: list[Any] = []
        used_nodes: set[Any] = set()

        for node in proposed_nodes:
            if node not in self.candidate_pool_set or node in used_nodes:
                continue
            cleaned.append(node)
            used_nodes.add(node)
            if len(cleaned) == self.config.budget:
                break

        self.last_repair_group_counts_before = {
            str(group_name): int(count)
            for group_name, count in self._selected_group_counts(cleaned).items()
        }
        cleaned = self._apply_group_quota_repair(cleaned)
        cleaned = self._apply_fairness_first_repair(cleaned)
        used_nodes = set(cleaned)

        while len(cleaned) < self.config.budget:
            repair_candidate_limit = self._candidate_pool_limit(
                len(self.candidate_pool) - len(used_nodes),
                stage="repair",
            )
            repair_prefilter_top_k = self._candidate_prefilter_limit(
                len(self.candidate_pool) - len(used_nodes),
                repair_candidate_limit,
                stage="repair",
            )
            repair_context = (
                self._weak_group_context_for_seed_set(cleaned)
                if self._fairness_repair_active()
                else None
            )
            replacement = self._sample_repair_node(
                used_nodes,
                ml_bias_weight=self.config.ml_repair_bias_weight if self._ml_guidance_enabled() else 0.0,
                primary_rate=self.config.ml_repair_primary_rate if self._uses_tuned_two_tier_guidance() else None,
                node2vec_diversity_weight=self.config.node2vec_diversity_weight if use_node2vec_diversity else 0.0,
                weak_group_context=repair_context,
                fairness_weight=self.config.repair_fairness_weight,
                zero_bonus_weight=0.0,
                bridge_weight=self.config.repair_bridge_weight,
                centrality_weight=self.config.repair_centrality_weight,
                diversity_weight=self.config.repair_diversity_weight,
                candidate_limit=repair_candidate_limit,
                prefilter_top_k=repair_prefilter_top_k,
            )
            if replacement is None:
                break
            cleaned.append(replacement)
            used_nodes.add(replacement)

        if len(cleaned) < self.config.budget:
            refill_context = (
                self._weak_group_context_for_seed_set(cleaned)
                if self._fairness_repair_active() or bool(self.config.use_group_quota_repair)
                else None
            )
            ranked_refill = self._rank_external_candidates(
                cleaned,
                ml_bias_weight=self.config.ml_repair_bias_weight if self._ml_guidance_enabled() else 0.0,
                primary_rate=self.config.ml_repair_primary_rate if self._uses_tuned_two_tier_guidance() else None,
                node2vec_diversity_weight=self.config.node2vec_diversity_weight if use_node2vec_diversity else 0.0,
                weak_group_context=refill_context,
                fairness_weight=self.config.repair_fairness_weight,
                zero_bonus_weight=self.config.zero_group_bonus_weight,
                bridge_weight=self.config.repair_bridge_weight,
                centrality_weight=self.config.repair_centrality_weight,
                diversity_weight=self.config.repair_diversity_weight,
                stage="repair",
            )
            for node in ranked_refill:
                if node in used_nodes:
                    continue
                cleaned.append(node)
                used_nodes.add(node)
                if len(cleaned) == self.config.budget:
                    break

        if len(cleaned) < self.config.budget:
            for node in self.global_ranked_nodes:
                if node in used_nodes:
                    continue
                cleaned.append(node)
                used_nodes.add(node)
                if len(cleaned) == self.config.budget:
                    break

        if bool(self.config.cluster_repair_enabled) and self.available_clusters and len(cleaned) == self.config.budget:
            max_per_cluster = max(1, int(self.config.max_seeds_per_cluster_fraction * self.config.budget))
            cluster_counts = self._cluster_seed_counts(cleaned)
            for cluster_id, cnt in list(cluster_counts.items()):
                if cnt <= max_per_cluster:
                    continue
                excess = cnt - max_per_cluster
                uncovered = self._uncovered_cluster_ids(cleaned)
                if not uncovered:
                    break
                cleaned_list = list(cleaned)
                removed = 0
                cluster_members_in_seeds = [
                    n for n in cleaned_list
                    if self.cluster_id_by_node.get(n) == cluster_id
                ]
                cluster_members_scored = sorted(
                    cluster_members_in_seeds,
                    key=lambda n: (float(self.node_scores.get(n, 0.0)), _sort_key(n)),
                )
                for victim in cluster_members_scored:
                    if removed >= excess:
                        break
                    target_cluster_id = next(iter(sorted(uncovered)))
                    target_candidates = [
                        n for n in self.available_clusters.get(target_cluster_id, ())
                        if n not in set(cleaned_list)
                    ]
                    if not target_candidates:
                        uncovered.discard(target_cluster_id)
                        continue
                    replacement = sorted(
                        target_candidates,
                        key=lambda n: (-float(self.node_scores.get(n, 0.0)), _sort_key(n)),
                    )[0]
                    cleaned_list.remove(victim)
                    cleaned_list.append(replacement)
                    uncovered = self._uncovered_cluster_ids(cleaned_list)
                    removed += 1
                cleaned = cleaned_list

        self.last_repair_group_counts_after = {
            str(group_name): int(count)
            for group_name, count in self._selected_group_counts(cleaned).items()
        }
        return self._validate_seed_set(tuple(sorted(cleaned, key=_sort_key)))

    def _validate_seed_set(self, seed_set: tuple[Any, ...]) -> tuple[Any, ...]:
        if len(seed_set) != self.config.budget:
            raise RuntimeError(
                f"Invalid seed-set size {len(seed_set)} produced; expected {self.config.budget}."
            )
        if len(set(seed_set)) != len(seed_set):
            raise RuntimeError("Seed set contains duplicate nodes after repair.")
        invalid_nodes = [node for node in seed_set if node not in self.candidate_pool_set]
        if invalid_nodes:
            raise RuntimeError(f"Seed set contains invalid nodes after repair: {invalid_nodes[:5]}.")
        return seed_set

    def _population_retry_limit(self) -> int:
        return max(50, self.config.population_size * 10)

    def _initialize_individual(self, use_ml_bias: bool = False) -> tuple[Any, ...]:
        seed_nodes: list[Any] = []
        used_nodes: set[Any] = set()
        ml_bias_weight = self.config.ml_repair_bias_weight if use_ml_bias else 0.0
        primary_rate = (
            self.config.ml_initialization_primary_rate
            if use_ml_bias and self._uses_tuned_two_tier_guidance()
            else None
        )
        quota_nodes = self._quota_initial_seed_nodes(seed_nodes, use_ml_bias=use_ml_bias)
        for quota_node in quota_nodes:
            if quota_node in self.candidate_pool_set and quota_node not in used_nodes:
                seed_nodes.append(quota_node)
                used_nodes.add(quota_node)
            if len(seed_nodes) >= self.config.budget:
                break

        while len(seed_nodes) < self.config.budget:
            node_id: Any | None = None
            if self._fairness_first_init_active() and len(seed_nodes) < self.config.fairness_first_init_slots:
                fairness_ranked = self._rank_fairness_first_candidates(
                    seed_nodes,
                    ml_bias_weight=ml_bias_weight,
                    primary_rate=primary_rate,
                )
                if fairness_ranked:
                    init_choices = tuple(fairness_ranked[: min(3, len(fairness_ranked))])
                    node_id = sample_node_from_community(init_choices, self.rng)

            if node_id is None and bool(self.config.cluster_diversity_enabled) and self.available_clusters:
                uncovered = self._uncovered_cluster_ids(seed_nodes)
                if uncovered:
                    cluster_id = next(iter(sorted(uncovered)))
                    cluster_candidates = [
                        n for n in self.available_clusters[cluster_id]
                        if n not in used_nodes
                    ]
                    if cluster_candidates:
                        cluster_candidates_scored = sorted(
                            cluster_candidates,
                            key=lambda n: (-float(self.node_scores.get(n, 0.0)), _sort_key(n)),
                        )
                        node_id = cluster_candidates_scored[0]

            if node_id is None:
                node_id = self._sample_repair_node(
                    used_nodes,
                    ml_bias_weight=ml_bias_weight,
                    primary_rate=primary_rate,
                    node2vec_diversity_weight=0.0,
                )
            if node_id is None:
                break
            seed_nodes.append(node_id)
            used_nodes.add(node_id)

        return self._repair_seed_set(seed_nodes, use_node2vec_diversity=False)

    def _initialize_population(self) -> list[tuple[Any, ...]]:
        population: list[tuple[Any, ...]] = []
        seen: set[tuple[Any, ...]] = set()
        retry_count = 0

        strongest = self._repair_seed_set(
            self.global_ranked_nodes[: self.config.budget],
            use_node2vec_diversity=False,
        )
        population.append(strongest)
        seen.add(strongest)
        if self._ml_guidance_enabled() and self.config.ml_initialization_bias > 0.0:
            if self._uses_tuned_two_tier_guidance():
                ml_seed = self._initialize_individual(use_ml_bias=True)
            else:
                ml_seed = self._repair_seed_set(
                    self.ml_ranked_nodes[: self.config.budget],
                    use_node2vec_diversity=False,
                )
            if ml_seed not in seen and len(population) < self.config.population_size:
                population.append(ml_seed)
                seen.add(ml_seed)

        while len(population) < self.config.population_size:
            use_ml_bias = self._ml_guidance_enabled() and self.rng.random() < self.config.ml_initialization_bias
            candidate = self._initialize_individual(use_ml_bias=use_ml_bias)
            if candidate in seen and len(seen) < self.max_unique_seed_sets and retry_count < self._population_retry_limit():
                retry_count += 1
                continue
            population.append(candidate)
            seen.add(candidate)
            retry_count = 0

        self.initial_population_group_coverage_summary = self._summarize_initial_population_coverage(population)
        return population

    def _evaluate_seed_set(self, seed_set: tuple[Any, ...], screening: bool = False) -> CandidateEvaluation:
        evaluation, _ = self._evaluate_seed_set_with_cache_info(seed_set, screening=screening)
        return evaluation

    def _select_elites(self, evaluations: Sequence[CandidateEvaluation]) -> list[CandidateEvaluation]:
        elite_count = max(1, int(np.ceil(self.config.population_size * self.config.elite_fraction)))
        ranked = sorted(evaluations, key=self._candidate_rank_key, reverse=True)
        return ranked[:elite_count]

    def _apply_leader_guidance(
        self,
        individual: tuple[Any, ...],
        leader: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        if self.config.leader_guidance_fraction <= 0.0:
            return self._validate_seed_set(tuple(sorted(individual, key=_sort_key)))

        replace_count = min(
            self.config.budget,
            int(np.ceil(self.config.budget * self.config.leader_guidance_fraction)),
        )
        if replace_count == 0:
            return self._validate_seed_set(tuple(sorted(individual, key=_sort_key)))

        remove_indices = set(
            int(index)
            for index in self.rng.choice(
                np.arange(len(individual)),
                size=replace_count,
                replace=False,
            ).tolist()
        )
        guided_nodes = [
            node for index, node in enumerate(individual)
            if index not in remove_indices
        ]
        leader_nodes = [node for node in leader if node not in guided_nodes]
        guided_nodes.extend(leader_nodes[:replace_count])
        return self._repair_seed_set(guided_nodes)

    def _crossover(
        self,
        parent_a: tuple[Any, ...],
        parent_b: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        merged_nodes: list[Any] = []
        if self.config.use_ml_scores_in_crossover and self._ml_guidance_enabled():
            parent_union = sorted(
                set(parent_a) | set(parent_b),
                key=lambda node_id: (-float(self.ml_node_scores.get(node_id, 0.0)), _sort_key(node_id)),
            )
            merged_nodes.extend(parent_union[: max(1, self.config.budget // 3)])
        for node in parent_a:
            if self.rng.random() < 0.5:
                merged_nodes.append(node)
        for node in parent_b:
            if self.rng.random() < 0.5:
                merged_nodes.append(node)
        if not merged_nodes:
            merged_nodes.extend(parent_a[: max(1, self.config.budget // 2)])
            merged_nodes.extend(parent_b[: max(1, self.config.budget // 2)])
        if bool(self.config.cluster_diversity_enabled) and self.available_clusters and len(merged_nodes) > self.config.budget:
            seen_clusters: set[int] = set()
            cluster_diverse: list[Any] = []
            remainder: list[Any] = []
            for node in merged_nodes:
                cid = self.cluster_id_by_node.get(node)
                if cid is None or cid not in seen_clusters:
                    cluster_diverse.append(node)
                    if cid is not None:
                        seen_clusters.add(cid)
                else:
                    remainder.append(node)
            merged_nodes = cluster_diverse + remainder
        return self._repair_seed_set(merged_nodes)

    def _mutate(self, individual: tuple[Any, ...]) -> tuple[Any, ...]:
        mutated_nodes = list(individual)
        current_evaluation = self._evaluate_seed_set(individual)
        mutation_context = (
            self._weak_group_context_for_seed_set(individual, evaluation=current_evaluation)
            if self._weak_group_mutation_active()
            else None
        )
        for index, node_id in enumerate(list(mutated_nodes)):
            if self.rng.random() >= self.config.mutation_probability:
                continue

            remaining = [node for idx, node in enumerate(mutated_nodes) if idx != index]
            ranked_candidates = self._rank_external_candidates(
                remaining,
                ml_bias_weight=self.config.ml_mutation_bias_weight if self._ml_guidance_enabled() else 0.0,
                primary_rate=self.config.ml_mutation_primary_rate if self._uses_tuned_two_tier_guidance() else None,
                node2vec_diversity_weight=self.config.node2vec_diversity_weight,
                weak_group_context=mutation_context,
                fairness_weight=self.config.weakest_group_mutation_weight,
                zero_bonus_weight=self.config.zero_group_bonus_weight,
                bridge_weight=self.config.bridge_to_weak_group_weight,
                stage="mutation",
            )
            if not ranked_candidates:
                continue
            if bool(self.config.cluster_diversity_enabled) and self.available_clusters:
                uncovered = self._uncovered_cluster_ids(remaining)
                if uncovered:
                    cluster_preferred = [
                        n for n in ranked_candidates
                        if self.cluster_id_by_node.get(n) in uncovered
                    ]
                    if cluster_preferred:
                        ranked_candidates = cluster_preferred + [
                            n for n in ranked_candidates if n not in cluster_preferred
                        ]
            if self._uses_legacy_two_tier_guidance():
                secondary_ranked = [node for node in ranked_candidates if node not in self.primary_ml_pool_set]
                primary_ranked = [node for node in ranked_candidates if node in self.primary_ml_pool_set]
                chosen_pool = primary_ranked
                if secondary_ranked and self.rng.random() < self.config.ml_secondary_exploration_rate:
                    chosen_pool = secondary_ranked
                elif not chosen_pool:
                    chosen_pool = secondary_ranked
                mutation_choices = chosen_pool[: min(3, len(chosen_pool))]
                mutated_nodes[index] = sample_node_from_community(mutation_choices, self.rng)
                continue
            if self._ml_guidance_enabled():
                mutation_choices = ranked_candidates[: min(3, len(ranked_candidates))]
                mutated_nodes[index] = sample_node_from_community(mutation_choices, self.rng)
                continue
            mutated_nodes[index] = ranked_candidates[0]

        return self._repair_seed_set(mutated_nodes)

    def _local_search(self, individual: tuple[Any, ...]) -> tuple[Any, ...]:
        if self.config.disable_local_search or self.config.local_search_steps == 0:
            return self._validate_seed_set(tuple(sorted(individual, key=_sort_key)))

        current = self._validate_seed_set(tuple(sorted(individual, key=_sort_key)))
        current_full_evaluation = self._evaluate_seed_set(current, screening=False)
        current_evaluation = (
            self._evaluate_seed_set(current, screening=True)
            if self._staged_mc_active()
            else current_full_evaluation
        )
        self.last_local_search_swap_evaluations = 0

        for _ in range(self.config.local_search_steps):
            best_candidate = current
            best_evaluation = current_evaluation
            screening_improvements: list[tuple[tuple[Any, ...], CandidateEvaluation]] = []
            weak_group_context = (
                self._weak_group_context_for_seed_set(current, evaluation=current_evaluation)
                if self._worst_group_local_search_active()
                else None
            )
            replacement_nodes = self._rank_seed_nodes_for_replacement(
                current,
                ml_bias_weight=self.config.ml_local_search_bias_weight if self._ml_guidance_enabled() else 0.0,
                weak_group_context=weak_group_context,
            )
            if not self._swap_runtime_optimization_active():
                external_candidates = self._rank_external_candidates(
                    current,
                    ml_bias_weight=self.config.ml_local_search_bias_weight if self._ml_guidance_enabled() else 0.0,
                    primary_rate=self.config.ml_local_search_primary_rate if self._uses_tuned_two_tier_guidance() else None,
                    node2vec_diversity_weight=self.config.node2vec_diversity_weight,
                    weak_group_context=weak_group_context,
                    reference_evaluation=current_evaluation,
                    fairness_weight=self.config.weakest_group_mutation_weight if self._worst_group_local_search_active() else 0.0,
                    zero_bonus_weight=self.config.zero_group_bonus_weight if self._worst_group_local_search_active() else 0.0,
                    bridge_weight=self.config.bridge_to_weak_group_weight if self._worst_group_local_search_active() else 0.0,
                    local_search_delta_mf_weight=self.config.local_search_delta_mf_weight,
                    local_search_delta_dcv_weight=self.config.local_search_delta_dcv_weight,
                    local_search_overlap_penalty_weight=self.config.local_search_overlap_penalty_weight,
                    stage="local_search",
                )
                if bool(self.config.cluster_diversity_enabled) and self.available_clusters:
                    uncovered = self._uncovered_cluster_ids(current)
                    if uncovered:
                        cluster_preferred = [
                            n for n in external_candidates
                            if self.cluster_id_by_node.get(n) in uncovered
                        ]
                        other_candidates = [n for n in external_candidates if n not in cluster_preferred]
                        external_candidates = cluster_preferred + other_candidates
                candidate_pool_size, max_trials, early_stop_patience = self._effective_local_search_budget()
                failed_patience = (
                    self.config.local_search_failed_patience
                    if self.config.local_search_failed_patience > 0
                    else early_stop_patience
                )
                external_candidates = external_candidates[: candidate_pool_size]

                evaluated_trials = 0
                stale_trials = 0
                for node_to_remove in replacement_nodes:
                    partial = [node_id for node_id in current if node_id != node_to_remove]
                    for node_to_add in external_candidates:
                        if evaluated_trials >= max_trials or stale_trials >= failed_patience:
                            break
                        if self.config.enable_swap_cache:
                            evaluation, was_cached = self._evaluate_swap_candidate(
                                partial,
                                node_to_add,
                                screening=self._staged_mc_active(),
                            )
                            candidate = evaluation.seed_set
                        else:
                            candidate = self._repair_seed_set(partial + [node_to_add])
                            if candidate == current:
                                continue
                            evaluation = self._evaluate_seed_set(candidate, screening=self._staged_mc_active())
                            was_cached = False
                        if candidate == current:
                            continue
                        if not was_cached:
                            self.last_local_search_swap_evaluations += 1
                        evaluated_trials += 1
                        self.swap_attempts += 1
                        if self._is_local_search_improvement(evaluation, best_evaluation):
                            self._record_swap_acceptance()
                            best_candidate = candidate
                            best_evaluation = evaluation
                            screening_improvements.append((candidate, evaluation))
                            stale_trials = 0
                        else:
                            self._record_swap_rejection(evaluation, best_evaluation)
                            stale_trials += 1
                    if evaluated_trials >= max_trials or stale_trials >= failed_patience:
                        break

                if best_candidate == current:
                    break
                if self._staged_mc_active():
                    confirmed = self._confirm_local_search_improvement(
                        current_full_evaluation,
                        screening_improvements,
                    )
                    if confirmed is None:
                        break
                    current, current_full_evaluation = confirmed
                    current_evaluation = current_full_evaluation
                    continue
                current = best_candidate
                current_evaluation = best_evaluation
                current_full_evaluation = best_evaluation
                continue

            candidate_pool_size, max_trials, early_stop_patience = self._effective_local_search_budget()
            failed_patience = (
                self.config.local_search_failed_patience
                if self.config.local_search_failed_patience > 0
                else early_stop_patience
            )
            swap_candidate_pool_size, swap_prefilter_top_k, full_eval_top_k = self._effective_swap_stage_limits(
                candidate_pool_size
            )
            current_group_counts = self._selected_group_counts(current)
            current_community_counts = self._selected_community_counts(current)
            current_seed_set, current_coverage_levels, current_urgency_weights = self._scoring_priors(
                current,
                weak_group_context=weak_group_context,
                reference_evaluation=current_evaluation,
                require_proxy=True,
            )
            available_nodes = [node_id for node_id in self.candidate_pool if node_id not in current]
            prefiltered_candidates = self._prefilter_candidate_nodes(
                available_nodes,
                limit=swap_prefilter_top_k,
                group_counts=current_group_counts,
                community_counts=current_community_counts,
                ml_bias_weight=self.config.ml_local_search_bias_weight if self._ml_guidance_enabled() else 0.0,
                weak_group_context=weak_group_context,
                fairness_weight=self.config.weakest_group_mutation_weight if self._worst_group_local_search_active() else 0.0,
                zero_bonus_weight=self.config.zero_group_bonus_weight if self._worst_group_local_search_active() else 0.0,
                bridge_weight=self.config.bridge_to_weak_group_weight if self._worst_group_local_search_active() else 0.0,
                coverage_levels=current_coverage_levels,
                urgency_weights=current_urgency_weights,
            )
            external_candidates = self._rank_candidate_nodes(
                prefiltered_candidates,
                group_counts=current_group_counts,
                community_counts=current_community_counts,
                ml_bias_weight=self.config.ml_local_search_bias_weight if self._ml_guidance_enabled() else 0.0,
                primary_rate=self.config.ml_local_search_primary_rate if self._uses_tuned_two_tier_guidance() else None,
                reference_nodes=current_seed_set,
                reference_evaluation=current_evaluation,
                node2vec_diversity_weight=self.config.node2vec_diversity_weight,
                weak_group_context=weak_group_context,
                fairness_weight=self.config.weakest_group_mutation_weight if self._worst_group_local_search_active() else 0.0,
                zero_bonus_weight=self.config.zero_group_bonus_weight if self._worst_group_local_search_active() else 0.0,
                bridge_weight=self.config.bridge_to_weak_group_weight if self._worst_group_local_search_active() else 0.0,
                local_search_delta_mf_weight=self.config.local_search_delta_mf_weight,
                local_search_delta_dcv_weight=self.config.local_search_delta_dcv_weight,
                local_search_overlap_penalty_weight=self.config.local_search_overlap_penalty_weight,
                candidate_limit=swap_candidate_pool_size,
            )
            external_candidates = external_candidates[: swap_candidate_pool_size]
            proxy_weights = self._resolved_proxy_score_weights()

            evaluated_trials = 0
            stale_trials = 0
            accepted_first_improvement = False
            for node_to_remove in replacement_nodes:
                partial_seed_set, partial_group_counts, partial_community_counts = self._partial_counts_after_removal(
                    current,
                    node_to_remove,
                )
                partial_context = self._weak_group_context_from_group_counts(
                    partial_group_counts,
                    weakest_group_k=self.config.local_search_bottom_k_groups if self._worst_group_local_search_active() else None,
                )
                partial_coverage_levels = self._normalized_group_coverage_from_counts(partial_group_counts)
                partial_urgency_weights = self._group_urgency_weights(
                    partial_seed_set,
                    weak_group_context=partial_context,
                    coverage_levels=partial_coverage_levels,
                )
                swap_ranked_candidates = sorted(
                    external_candidates,
                    key=lambda node_to_add: (
                        -self._swap_proxy_score(
                            node_to_add,
                            partial_seed_set=partial_seed_set,
                            group_counts=partial_group_counts,
                            community_counts=partial_community_counts,
                            weak_group_context=partial_context,
                            coverage_levels=partial_coverage_levels,
                            urgency_weights=partial_urgency_weights,
                            proxy_weights=proxy_weights,
                        ),
                        _sort_key(node_to_add),
                    ),
                )
                for node_to_add in swap_ranked_candidates[:full_eval_top_k]:
                    if evaluated_trials >= max_trials or stale_trials >= failed_patience:
                        break
                    evaluation, was_cached = self._evaluate_swap_candidate(
                        partial_seed_set,
                        node_to_add,
                        screening=self._staged_mc_active(),
                    )
                    if evaluation.seed_set == current:
                        continue
                    if not was_cached:
                        self.last_local_search_swap_evaluations += 1
                    evaluated_trials += 1
                    self.swap_attempts += 1
                    if self._is_local_search_improvement(evaluation, best_evaluation):
                        self._record_swap_acceptance()
                        best_candidate = evaluation.seed_set
                        best_evaluation = evaluation
                        screening_improvements.append((evaluation.seed_set, evaluation))
                        stale_trials = 0
                        if self.config.local_search_first_improvement:
                            accepted_first_improvement = True
                            break
                    else:
                        self._record_swap_rejection(evaluation, best_evaluation)
                        stale_trials += 1
                if accepted_first_improvement:
                    break
                if evaluated_trials >= max_trials or stale_trials >= failed_patience:
                    break

            if best_candidate == current:
                break
            if self._staged_mc_active():
                confirmed = self._confirm_local_search_improvement(
                    current_full_evaluation,
                    screening_improvements,
                )
                if confirmed is None:
                    break
                current, current_full_evaluation = confirmed
                current_evaluation = current_full_evaluation
                continue
            current = best_candidate
            current_evaluation = best_evaluation
            current_full_evaluation = best_evaluation

        return current

    def _dcv_parity_repair_removal_order(
        self,
        seed_set: tuple[Any, ...],
        evaluation: CandidateEvaluation,
    ) -> list[Any]:
        over_groups = tuple(getattr(evaluation.fairness, "over_served_groups", ()) or ())
        group_counts = self._selected_group_counts(seed_set)
        over_group_set = set(over_groups)
        over_represented_groups = {
            group_name
            for group_name, count in group_counts.items()
            if count > max(1, int(math.ceil(float(self.config.budget) / float(max(1, len(self.group_names))))))
        }
        target_groups = over_group_set | over_represented_groups
        ranked = self._rank_seed_nodes_for_replacement(
            seed_set,
            ml_bias_weight=self.config.ml_repair_bias_weight if self._ml_guidance_enabled() else 0.0,
            weak_group_context=WeakGroupContext(
                weak_groups=tuple(getattr(evaluation.fairness, "under_served_groups", ()) or ()),
                zero_covered_groups=(),
            ),
        )
        preferred = [
            node_id
            for node_id in ranked
            if self.node_group_by_node.get(node_id) in target_groups
        ]
        return preferred + [node_id for node_id in ranked if node_id not in preferred]

    def _dcv_parity_repair(self, seed_set: tuple[Any, ...]) -> tuple[Any, ...]:
        if not bool(self.config.use_dcv_parity_repair) or self.config.dcv_parity_repair_rounds <= 0:
            return seed_set

        current = self._validate_seed_set(seed_set)
        current_eval = self._evaluate_seed_set(current, screening=False)
        self.dcv_before_parity_repair = float(current_eval.fairness.dcv)
        self.dcv_after_parity_repair_estimated = float(current_eval.fairness.dcv)
        candidate_limit = max(1, int(self.config.dcv_parity_repair_candidate_limit))

        for _ in range(int(self.config.dcv_parity_repair_rounds)):
            under_groups = tuple(getattr(current_eval.fairness, "under_served_groups", ()) or ())
            over_groups = tuple(getattr(current_eval.fairness, "over_served_groups", ()) or ())
            if not under_groups or not over_groups:
                break

            weak_context = WeakGroupContext(weak_groups=under_groups, zero_covered_groups=())
            add_candidates = self._rank_external_candidates(
                current,
                ml_bias_weight=self.config.ml_repair_bias_weight if self._ml_guidance_enabled() else 0.0,
                primary_rate=self.config.ml_repair_primary_rate if self._uses_tuned_two_tier_guidance() else None,
                weak_group_context=weak_context,
                reference_evaluation=current_eval,
                fairness_weight=max(float(self.config.under_served_bonus_weight), float(self.config.repair_fairness_weight)),
                zero_bonus_weight=self.config.zero_group_bonus_weight,
                bridge_weight=max(float(self.config.bridge_to_weak_group_weight), float(self.config.repair_bridge_weight)),
                centrality_weight=self.config.repair_centrality_weight,
                diversity_weight=self.config.repair_diversity_weight,
                stage="repair",
            )[:candidate_limit]
            if not add_candidates:
                break

            accepted: tuple[tuple[Any, ...], CandidateEvaluation] | None = None
            for node_to_remove in self._dcv_parity_repair_removal_order(current, current_eval):
                partial = [node_id for node_id in current if node_id != node_to_remove]
                for node_to_add in add_candidates:
                    if node_to_add in partial:
                        continue
                    self.parity_repair_attempts += 1
                    trial_seed_set = self._repair_seed_set([*partial, node_to_add])
                    if trial_seed_set == current:
                        continue
                    trial_eval = self._evaluate_seed_set(trial_seed_set, screening=False)
                    if (
                        float(trial_eval.fairness.dcv)
                        < float(current_eval.fairness.dcv) - float(self.config.dcv_improvement_epsilon)
                        and float(trial_eval.fairness.mf)
                        >= float(current_eval.fairness.mf) - float(self.config.mf_drop_tolerance)
                    ):
                        accepted = (trial_seed_set, trial_eval)
                        self.parity_repair_successes += 1
                        self.groups_rebalanced.update(str(group_name) for group_name in under_groups)
                        self.groups_rebalanced.update(str(group_name) for group_name in over_groups)
                        break
                if accepted is not None:
                    break

            if accepted is None:
                break
            current, current_eval = accepted
            self.dcv_after_parity_repair_estimated = float(current_eval.fairness.dcv)

        return current

    def _population_diversity(self, population: Sequence[tuple[Any, ...]]) -> float:
        if len(population) < 2:
            return 0.0

        distances: list[float] = []
        set_population = [set(individual) for individual in population]
        for left_index in range(len(set_population)):
            for right_index in range(left_index + 1, len(set_population)):
                union = set_population[left_index] | set_population[right_index]
                if not union:
                    distances.append(0.0)
                    continue
                overlap = set_population[left_index] & set_population[right_index]
                distances.append(1.0 - (len(overlap) / len(union)))
        return float(np.mean(distances)) if distances else 0.0

    def _survival_selection(
        self,
        population: Sequence[tuple[Any, ...]],
        offspring: Sequence[tuple[Any, ...]],
    ) -> list[CandidateEvaluation]:
        pool = list(population) + list(offspring)
        ranked_unique: dict[tuple[Any, ...], CandidateEvaluation] = {}
        evaluated_pool: list[CandidateEvaluation] = []
        for seed_set in pool:
            evaluation = self._evaluate_seed_set(seed_set)
            evaluated_pool.append(evaluation)
            ranked_unique[evaluation.seed_set] = evaluation

        ranked = sorted(ranked_unique.values(), key=self._candidate_rank_key, reverse=True)
        if len(ranked) >= self.config.population_size:
            return ranked[: self.config.population_size]

        ranked_with_duplicates = sorted(evaluated_pool, key=self._candidate_rank_key, reverse=True)
        return ranked_with_duplicates[: self.config.population_size]

    def optimize(self) -> HybridOptimizationResult:
        """Run the unified hybrid SI+EA optimizer end-to-end."""

        start = perf_counter()
        population = self._initialize_population()
        history_records: list[dict[str, float | int | str]] = []
        self.last_local_search_applied_count = 0

        for generation in range(self.config.generations):
            generation_full_eval_calls_before = self.full_evaluation_calls
            generation_screening_eval_calls_before = self.screening_evaluation_calls
            current_evaluations = [self._evaluate_seed_set(individual) for individual in population]
            elites = self._select_elites(current_evaluations)
            leader_pool = [evaluation.seed_set for evaluation in elites]
            ranked_current = sorted(current_evaluations, key=self._candidate_rank_key)
            weak_count = max(1, len(ranked_current) // 2)
            local_search_seed_sets = {
                evaluation.seed_set
                for evaluation in sorted(current_evaluations, key=self._candidate_rank_key, reverse=True)[
                    : self._effective_local_search_elite_count()
                ]
            }
            run_local_search_this_generation = self._should_run_local_search_generation(generation)

            offspring: list[tuple[Any, ...]] = []
            generation_local_search_applied_count = 0
            generation_local_search_swap_evaluations = 0
            for index, evaluation in enumerate(ranked_current):
                current = evaluation.seed_set
                leader = leader_pool[index % len(leader_pool)]
                child = current
                if not self.config.disable_swarm_guidance and index < weak_count:
                    child = self._apply_leader_guidance(current, leader)

                if (
                    not self.config.disable_crossover
                    and self.rng.random() < self.config.crossover_probability
                ):
                    mate = leader_pool[int(self.rng.integers(len(leader_pool)))]
                    child = self._crossover(child, mate)

                child = self._mutate(child)
                if run_local_search_this_generation and current in local_search_seed_sets:
                    child = self._local_search(child)
                    generation_local_search_applied_count += 1
                    generation_local_search_swap_evaluations += self.last_local_search_swap_evaluations
                offspring.append(self._validate_seed_set(child))

            next_evaluations = self._survival_selection(population, offspring)
            population = [evaluation.seed_set for evaluation in next_evaluations]
            for individual in population:
                self._validate_seed_set(individual)

            best_evaluation = max(next_evaluations, key=self._candidate_rank_key)
            average_score = float(np.mean([evaluation.score for evaluation in next_evaluations]))
            diversity = self._population_diversity(population)
            fairness_diagnostics = self._fairness_diagnostics(best_evaluation.fairness)
            self.last_local_search_applied_count = generation_local_search_applied_count
            history_records.append(
                {
                    "generation": generation,
                    "best_score": best_evaluation.score,
                    "best_f_score": best_evaluation.f_score,
                    "best_objective_score": best_evaluation.score,
                    "best_spread": best_evaluation.total_spread_mean,
                    "best_mf": best_evaluation.fairness.mf,
                    "best_dcv": best_evaluation.fairness.dcv,
                    "average_population_score": average_score,
                    "population_diversity": diversity,
                    "best_seed_set": str(list(best_evaluation.seed_set)),
                    "local_search_applied_count": generation_local_search_applied_count,
                    "local_search_swap_evaluations": generation_local_search_swap_evaluations,
                    "screening_evaluation_calls": self.screening_evaluation_calls - generation_screening_eval_calls_before,
                    "full_evaluation_calls": self.full_evaluation_calls - generation_full_eval_calls_before,
                    **fairness_diagnostics,
                }
            )

            if self.config.debug_logging:
                print(
                    "[hybrid_siea] "
                    f"gen={generation} "
                    f"best_score={best_evaluation.score:.6f} "
                    f"best_spread={best_evaluation.total_spread_mean:.6f} "
                    f"best_mf={best_evaluation.fairness.mf:.6f} "
                    f"best_dcv={best_evaluation.fairness.dcv:.6f} "
                    f"avg_score={average_score:.6f} "
                    f"diversity={diversity:.6f}"
                )

        final_evaluations = [self._evaluate_seed_set(individual) for individual in population]
        best_evaluation = max(final_evaluations, key=self._candidate_rank_key)
        repaired_seed_set = self._dcv_parity_repair(best_evaluation.seed_set)
        if repaired_seed_set != best_evaluation.seed_set:
            best_evaluation = self._evaluate_seed_set(repaired_seed_set, screening=False)
        runtime_seconds = perf_counter() - start
        search_verify = self.search_evaluator.verify() if self.search_evaluator is not None else {}

        return HybridOptimizationResult(
            best_seed_set=best_evaluation.seed_set,
            best_score=best_evaluation.score,
            best_spread=best_evaluation.total_spread_mean,
            best_fairness=best_evaluation.fairness,
            runtime_seconds=runtime_seconds,
            candidate_pool_size=len(self.candidate_pool),
            history=pd.DataFrame(history_records),
            repaired_seed_sets=int(self.repaired_seed_sets),
            successful_swaps=int(sum(record.get("local_search_applied_count", 0) for record in history_records)),
            fitness_cache_hits=int(self.fitness_cache_hits),
            marginal_cache_hits=int(self.marginal_cache_hits),
            swap_cache_hits=int(self.swap_cache_hits),
            repair_attempts=int(self.repair_attempts),
            weak_group_repairs=int(self.weak_group_repairs),
            seeds_redirected_by_overshoot_cap=int(self.seeds_redirected_by_overshoot_cap),
            protected_group_seed_counts_before_repair=dict(self.last_repair_group_counts_before),
            protected_group_seed_counts_after_repair=dict(self.last_repair_group_counts_after),
            swap_attempts=int(self.swap_attempts),
            swap_accepted=int(self.swap_accepted),
            swap_rejected_fairness_degradation=int(self.swap_rejected_fairness_degradation),
            swap_accepted_fscore_improvement=int(self.swap_accepted_fscore_improvement),
            swap_accepted_mf_improvement=int(self.swap_accepted_mf_improvement),
            swap_accepted_spread_fairness_preserved=int(self.swap_accepted_spread_fairness_preserved),
            swaps_accepted_dcv_improvement=int(self.swaps_accepted_dcv_improvement),
            swaps_rejected_dcv_worsening=int(self.swaps_rejected_dcv_worsening),
            swaps_accepted_parity_improvement=int(self.swaps_accepted_parity_improvement),
            seed_quota_per_protected_group=dict(self.seed_quota_per_protected_group),
            initial_population_group_coverage_summary=dict(self.initial_population_group_coverage_summary),
            parity_repair_attempts=int(self.parity_repair_attempts),
            parity_repair_successes=int(self.parity_repair_successes),
            dcv_before_parity_repair=self.dcv_before_parity_repair,
            dcv_after_parity_repair_estimated=self.dcv_after_parity_repair_estimated,
            groups_rebalanced=tuple(sorted(self.groups_rebalanced, key=_sort_key)),
            search_mc_eval_calls=int(search_verify.get("search_mc_eval_calls", 0)),
            search_ris_eval_calls=int(search_verify.get("search_ris_eval_calls", 0)),
            search_fair_ris_eval_calls=int(search_verify.get("search_fair_ris_eval_calls", 0)),
            time_ris_evaluation=float(search_verify.get("time_ris_evaluation", 0.0)),
            time_mc_search_evaluation=float(search_verify.get("time_mc_search_evaluation", 0.0)),
            time_search_objective_total=float(search_verify.get("time_search_objective_total", 0.0)),
            rr_sets_used=int(search_verify.get("rr_sets_used", 0)),
            rr_sets_generated=int(search_verify.get("rr_sets_generated", 0)),
        )
