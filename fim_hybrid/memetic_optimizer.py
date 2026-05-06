"""Fairness-first Memetic Algorithm optimizer for Fair Influence Maximization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .optimizer_core import (
    CandidateEvaluation,
    OptimizationResult,
    EAOptimizerConfig,
    EAOptimizerBase,
    WeakGroupContext,
    _sort_key,
)
from .safe_math import safe_divide


@dataclass(slots=True)
class MemeticConfig(EAOptimizerConfig):
    """Configuration for the fairness-first Memetic Algorithm."""

    memetic_initialization_mode: str = "fairness_guided"
    memetic_random_immigrant_rate: float = 0.10
    memetic_selection: str = "tournament"
    memetic_tournament_size: int = 3
    memetic_crossover: str = "fairness_preserving"
    memetic_mutation_strength: int = 0
    memetic_weak_group_mutation_bias: float = 0.70
    memetic_repair_enabled: bool = True
    memetic_repair_rounds: int = 2
    memetic_local_search_enabled: bool = True
    memetic_local_search_frequency: str = "every_generation"
    memetic_local_search_intensity: str = "light"
    memetic_local_search_top_elites: float = 0.25
    memetic_local_search_candidate_limit: int = 0
    memetic_elitism_rate: float = 0.10
    memetic_diversity_preservation: bool = True
    memetic_community_coverage_weight: float = 0.5


class MemeticOptimizer(EAOptimizerBase):
    """Memetic seed-set optimizer using fairness-first evolutionary refinement."""

    config: MemeticConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.local_search_attempts = 0
        self.local_search_improvements = 0
        self.memetic_duplicate_repairs = 0
        self.memetic_budget_repairs = 0
        self.memetic_community_repairs = 0
        self.final_diversity_score = 0.0

    def _validate_inputs(self) -> None:
        super()._validate_inputs()
        valid_initialization = {"fairness_guided", "score_guided", "random"}
        valid_selection = {"tournament", "rank", "roulette"}
        valid_crossover = {"fairness_preserving", "uniform"}
        valid_frequency = {"every_generation", "final_generation", "never"}
        valid_intensity = {"light", "medium", "heavy"}
        if self.config.memetic_initialization_mode not in valid_initialization:
            raise ValueError(f"memetic_initialization_mode must be one of {sorted(valid_initialization)}.")
        if self.config.memetic_selection not in valid_selection:
            raise ValueError(f"memetic_selection must be one of {sorted(valid_selection)}.")
        if self.config.memetic_crossover not in valid_crossover:
            raise ValueError(f"memetic_crossover must be one of {sorted(valid_crossover)}.")
        if self.config.memetic_local_search_frequency not in valid_frequency:
            raise ValueError(f"memetic_local_search_frequency must be one of {sorted(valid_frequency)}.")
        if self.config.memetic_local_search_intensity not in valid_intensity:
            raise ValueError(f"memetic_local_search_intensity must be one of {sorted(valid_intensity)}.")
        if self.config.memetic_tournament_size < 1:
            raise ValueError("memetic_tournament_size must be at least 1.")
        if self.config.memetic_mutation_strength < 0:
            raise ValueError("memetic_mutation_strength must be non-negative.")
        for field_name in (
            "memetic_random_immigrant_rate",
            "memetic_weak_group_mutation_bias",
            "memetic_elitism_rate",
            "memetic_local_search_top_elites",
        ):
            value = float(getattr(self.config, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
        if self.config.memetic_repair_rounds < 0:
            raise ValueError("memetic_repair_rounds must be non-negative.")
        if self.config.memetic_local_search_candidate_limit < 0:
            raise ValueError("memetic_local_search_candidate_limit must be non-negative.")
        if self.config.memetic_community_coverage_weight < 0.0:
            raise ValueError("memetic_community_coverage_weight must be non-negative.")

    def _community_coverage_fraction(self, seed_set: Sequence[Any]) -> float:
        if not self.available_communities:
            return 0.0
        covered = {
            self.community_result.community_id_by_node[node_id]
            for node_id in seed_set
            if node_id in self.community_result.community_id_by_node
        }
        return safe_divide(
            float(len(covered)),
            float(len(self.available_communities)),
            default=0.0,
            context="memetic community coverage fitness",
        )

    def _fitness_score(self, evaluation_result: Any) -> float:
        node_count = max(1, int(self.dataset.graph.number_of_nodes()))
        normalized_spread = safe_divide(
            float(evaluation_result.total_spread_mean),
            float(node_count),
            default=0.0,
            context="memetic normalized spread fitness",
        )
        group_values = dict(getattr(evaluation_result.fairness, "normalized_group_spread", {}) or {})
        if not group_values:
            group_values = dict(getattr(evaluation_result.fairness, "group_spread", {}) or {})
        fraction_groups_covered = safe_divide(
            float(sum(1 for value in group_values.values() if float(value) > 0.0)),
            float(len(group_values)),
            default=0.0,
            context="memetic protected-group coverage fitness",
        )
        community_coverage = self._community_coverage_fraction(getattr(evaluation_result, "seed_set", ()))
        dcv_value = (
            float(evaluation_result.fairness.dcv_shortfall)
            if str(self.config.primary_dcv_mode) == "shortfall"
            else float(evaluation_result.fairness.dcv)
        )
        score = float(
            self.config.fscore_weight * float(evaluation_result.f_score)
            + self.config.mf_weight * float(evaluation_result.fairness.mf)
            - self.config.dcv_weight * dcv_value
            + self.config.group_coverage_weight * fraction_groups_covered
            + self.config.memetic_community_coverage_weight * community_coverage
            + self.config.spread_weight * normalized_spread
        )
        if self.config.use_dcv_targeting:
            score += self._dcv_targeting_objective_adjustment(evaluation_result)
        return score

    def _combined_ranked_nodes(self) -> tuple[Any, ...]:
        return tuple(
            sorted(
                self.candidate_pool,
                key=lambda node_id: (
                    -float(self.ml_node_scores.get(node_id, 0.0)),
                    -float(self.node_scores.get(node_id, 0.0)),
                    _sort_key(node_id),
                ),
            )
        )

    def _top_unused_from_nodes(
        self,
        nodes: Sequence[Any],
        used_nodes: set[Any],
        *,
        randomized: bool,
    ) -> Any | None:
        available = [node_id for node_id in nodes if node_id not in used_nodes and node_id in self.candidate_pool_set]
        if not available:
            return None
        if not randomized:
            return available[0]
        window = available[: min(4, len(available))]
        return window[int(self.rng.integers(len(window)))]

    def _memetic_random_individual(self) -> tuple[Any, ...]:
        sample_size = min(int(self.config.budget), len(self.candidate_pool))
        sampled = self.rng.choice(np.asarray(self.candidate_pool, dtype=object), size=sample_size, replace=False).tolist()
        return self._repair_seed_set(sampled)

    def _memetic_initialize_individual(self, *, randomized: bool = True) -> tuple[Any, ...]:
        if self.config.memetic_initialization_mode == "random":
            return self._memetic_random_individual()
        if self.config.memetic_initialization_mode == "score_guided":
            ranked = list(self._combined_ranked_nodes())
            if randomized:
                top_window = ranked[: min(len(ranked), max(self.config.budget * 3, self.config.budget))]
                self.rng.shuffle(top_window)
                ranked = top_window + ranked[len(top_window):]
            return self._repair_seed_set(ranked[: self.config.budget])

        ranked_nodes = self._combined_ranked_nodes()
        used_nodes: set[Any] = set()
        seed_nodes: list[Any] = []

        quota_nodes = self._quota_initial_seed_nodes(seed_nodes, use_ml_bias=True)
        for node_id in quota_nodes:
            if node_id in self.candidate_pool_set and node_id not in used_nodes:
                seed_nodes.append(node_id)
                used_nodes.add(node_id)
            if len(seed_nodes) >= self.config.budget:
                break

        weak_groups = sorted(
            self.group_names,
            key=lambda group_name: (
                int(self.protected_group_report.group_sizes[group_name]),
                _sort_key(group_name),
            ),
        )
        if randomized and len(weak_groups) > 1:
            prefix = weak_groups[: min(len(weak_groups), max(1, len(weak_groups) // 2))]
            self.rng.shuffle(prefix)
            weak_groups = prefix + weak_groups[len(prefix):]
        for group_name in weak_groups:
            if len(seed_nodes) >= self.config.budget:
                break
            group_nodes = [node_id for node_id in ranked_nodes if self.node_group_by_node[node_id] == group_name]
            node_id = self._top_unused_from_nodes(group_nodes, used_nodes, randomized=randomized)
            if node_id is None:
                continue
            seed_nodes.append(node_id)
            used_nodes.add(node_id)

        community_ids = sorted(
            self.available_communities,
            key=lambda community_id: (
                len(self.available_communities[community_id]),
                int(community_id),
            ),
        )
        if randomized and len(community_ids) > 1:
            self.rng.shuffle(community_ids)
        for community_id in community_ids:
            if len(seed_nodes) >= self.config.budget:
                break
            community_nodes = [
                node_id
                for node_id in ranked_nodes
                if self.community_result.community_id_by_node[node_id] == community_id
            ]
            node_id = self._top_unused_from_nodes(community_nodes, used_nodes, randomized=randomized)
            if node_id is None:
                continue
            seed_nodes.append(node_id)
            used_nodes.add(node_id)

        for node_id in ranked_nodes:
            if len(seed_nodes) >= self.config.budget:
                break
            if node_id in used_nodes:
                continue
            seed_nodes.append(node_id)
            used_nodes.add(node_id)

        return self._repair_seed_set(seed_nodes)

    def _initialize_population(self) -> list[tuple[Any, ...]]:
        population: list[tuple[Any, ...]] = []
        seen: set[tuple[Any, ...]] = set()
        ranked = self._combined_ranked_nodes()
        strongest = self._repair_seed_set(ranked[: self.config.budget])
        population.append(strongest)
        seen.add(strongest)

        retry_count = 0
        while len(population) < self.config.population_size:
            if self.rng.random() < self.config.memetic_random_immigrant_rate:
                candidate = self._memetic_random_individual()
            else:
                candidate = self._memetic_initialize_individual(randomized=True)
            if candidate in seen and len(seen) < self.max_unique_seed_sets and retry_count < self._population_retry_limit():
                retry_count += 1
                continue
            population.append(candidate)
            seen.add(candidate)
            retry_count = 0
        self.initial_population_group_coverage_summary = self._summarize_initial_population_coverage(population)
        return population

    def _selection_weights(self, evaluations: Sequence[CandidateEvaluation]) -> np.ndarray:
        scores = np.asarray([float(evaluation.score) for evaluation in evaluations], dtype=float)
        if not np.isfinite(scores).all():
            return np.ones(len(evaluations), dtype=float) / max(1, len(evaluations))
        scores = scores - float(np.min(scores))
        scores = scores + 1e-9
        total = float(np.sum(scores))
        if total <= 0.0:
            return np.ones(len(evaluations), dtype=float) / max(1, len(evaluations))
        return scores / total

    def _select_parent(self, evaluations: Sequence[CandidateEvaluation]) -> CandidateEvaluation:
        if len(evaluations) == 1:
            return evaluations[0]
        selection = self.config.memetic_selection
        if selection == "rank":
            ranked = sorted(evaluations, key=self._candidate_rank_key, reverse=True)
            weights = np.asarray([len(ranked) - index for index in range(len(ranked))], dtype=float)
            weights = weights / float(np.sum(weights))
            return ranked[int(self.rng.choice(np.arange(len(ranked)), p=weights))]
        if selection == "roulette":
            weights = self._selection_weights(evaluations)
            return evaluations[int(self.rng.choice(np.arange(len(evaluations)), p=weights))]
        tournament_size = min(len(evaluations), max(1, int(self.config.memetic_tournament_size)))
        indices = self.rng.choice(np.arange(len(evaluations)), size=tournament_size, replace=False)
        contestants = [evaluations[int(index)] for index in indices]
        return max(contestants, key=self._candidate_rank_key)

    def _crossover(self, parent_a: tuple[Any, ...], parent_b: tuple[Any, ...]) -> tuple[Any, ...]:
        if self.config.memetic_crossover == "uniform":
            return super()._crossover(parent_a, parent_b)

        merged_nodes: list[Any] = []
        used_nodes: set[Any] = set()

        parent_intersection = sorted(
            set(parent_a) & set(parent_b),
            key=lambda node_id: (-float(self.ml_node_scores.get(node_id, 0.0)), _sort_key(node_id)),
        )
        for node_id in parent_intersection:
            merged_nodes.append(node_id)
            used_nodes.add(node_id)
            if len(merged_nodes) >= self.config.budget:
                return self._repair_seed_set(merged_nodes)

        parent_union = list(dict.fromkeys(list(parent_a) + list(parent_b)))
        union_group_counts = self._selected_group_counts(parent_union)
        weak_context = self._weak_group_context_from_group_counts(
            union_group_counts,
            weakest_group_k=min(3, len(self.group_names)),
        )
        weak_parent_nodes = sorted(
            parent_union,
            key=lambda node_id: (
                -self._group_support_score(node_id, weak_context.weak_groups),
                -float(self.ml_node_scores.get(node_id, 0.0)),
                _sort_key(node_id),
            ),
        )
        for node_id in weak_parent_nodes:
            if len(merged_nodes) >= self.config.budget:
                break
            if node_id in used_nodes:
                continue
            if self.node_group_by_node[node_id] not in weak_context.weak_groups:
                continue
            merged_nodes.append(node_id)
            used_nodes.add(node_id)

        community_counts = self._selected_community_counts(merged_nodes)
        for node_id in sorted(
            parent_union,
            key=lambda candidate: (
                self.community_result.community_id_by_node[candidate] in community_counts,
                -float(self.ml_node_scores.get(candidate, 0.0)),
                -float(self.node_scores.get(candidate, 0.0)),
                _sort_key(candidate),
            ),
        ):
            if len(merged_nodes) >= self.config.budget:
                break
            if node_id in used_nodes:
                continue
            merged_nodes.append(node_id)
            used_nodes.add(node_id)
            community_counts[self.community_result.community_id_by_node[node_id]] = (
                community_counts.get(self.community_result.community_id_by_node[node_id], 0) + 1
            )

        ranked_refill = self._rank_external_candidates(
            merged_nodes,
            ml_bias_weight=self.config.ml_repair_bias_weight if self._ml_guidance_enabled() else 0.0,
            weak_group_context=weak_context,
            fairness_weight=self.config.repair_fairness_weight,
            zero_bonus_weight=self.config.zero_group_bonus_weight,
            bridge_weight=self.config.repair_bridge_weight,
            stage="repair",
        )
        merged_nodes.extend(ranked_refill)
        return self._repair_seed_set(merged_nodes)

    def _mutation_strength(self) -> int:
        if self.config.memetic_mutation_strength > 0:
            return min(self.config.budget, int(self.config.memetic_mutation_strength))
        return max(1, int(math.ceil(float(self.config.budget) * 0.05)))

    def _mutate(self, individual: tuple[Any, ...]) -> tuple[Any, ...]:
        if self.rng.random() >= self.config.mutation_probability:
            return self._repair_seed_set(individual)

        mutated_nodes = list(individual)
        current_evaluation = self._evaluate_seed_set(individual)
        weak_context = self._weak_group_context_for_seed_set(individual, evaluation=current_evaluation)
        use_weak_bias = self.rng.random() < self.config.memetic_weak_group_mutation_bias
        replacement_nodes = self._rank_seed_nodes_for_replacement(
            mutated_nodes,
            ml_bias_weight=self.config.ml_mutation_bias_weight if self._ml_guidance_enabled() else 0.0,
            weak_group_context=weak_context if use_weak_bias else None,
        )

        for node_to_remove in replacement_nodes[: self._mutation_strength()]:
            retained = [node_id for node_id in mutated_nodes if node_id != node_to_remove]
            candidate_context = weak_context if use_weak_bias else None
            ranked_candidates = self._rank_external_candidates(
                retained,
                ml_bias_weight=self.config.ml_mutation_bias_weight if self._ml_guidance_enabled() else 0.0,
                primary_rate=self.config.ml_mutation_primary_rate if self._uses_tuned_two_tier_guidance() else None,
                node2vec_diversity_weight=self.config.node2vec_diversity_weight,
                weak_group_context=candidate_context,
                fairness_weight=self.config.weakest_group_mutation_weight if use_weak_bias else 0.0,
                zero_bonus_weight=self.config.zero_group_bonus_weight if use_weak_bias else 0.0,
                bridge_weight=self.config.bridge_to_weak_group_weight if use_weak_bias else 0.0,
                stage="mutation",
            )
            if not ranked_candidates:
                continue
            chosen = ranked_candidates[0]
            mutated_nodes = retained + [chosen]

        return self._repair_seed_set(mutated_nodes)

    def _repair_seed_set(
        self,
        proposed_nodes: Sequence[Any],
        use_node2vec_diversity: bool = True,
    ) -> tuple[Any, ...]:
        proposed_count = len(proposed_nodes)
        unique_valid_count = len({node for node in proposed_nodes if node in getattr(self, "candidate_pool_set", set())})
        if unique_valid_count < proposed_count:
            self.memetic_duplicate_repairs = getattr(self, "memetic_duplicate_repairs", 0) + 1
        if unique_valid_count != self.config.budget:
            self.memetic_budget_repairs = getattr(self, "memetic_budget_repairs", 0) + 1

        if not self.config.memetic_repair_enabled:
            return super()._repair_seed_set(proposed_nodes, use_node2vec_diversity=use_node2vec_diversity)

        repaired = tuple(proposed_nodes)
        rounds = max(1, int(self.config.memetic_repair_rounds))
        for _ in range(rounds):
            previous = repaired
            repaired = super()._repair_seed_set(repaired, use_node2vec_diversity=use_node2vec_diversity)
            community_before = len({
                self.community_result.community_id_by_node[node_id]
                for node_id in previous
                if node_id in self.community_result.community_id_by_node
            })
            community_after = len({
                self.community_result.community_id_by_node[node_id]
                for node_id in repaired
                if node_id in self.community_result.community_id_by_node
            })
            if community_after > community_before:
                self.memetic_community_repairs = getattr(self, "memetic_community_repairs", 0) + 1
            if repaired == previous:
                break
        return repaired

    def _should_apply_memetic_local_search(self, generation: int) -> bool:
        if not bool(self.config.memetic_local_search_enabled):
            return False
        frequency = self.config.memetic_local_search_frequency
        if frequency == "never":
            return False
        if frequency == "final_generation":
            return generation == self.config.generations - 1
        return True

    def _memetic_local_search_elite_count(self, population_size: int) -> int:
        return max(1, int(math.ceil(float(population_size) * float(self.config.memetic_local_search_top_elites))))

    def _run_memetic_local_search(self, individual: tuple[Any, ...]) -> tuple[Any, ...]:
        old_steps = self.config.local_search_steps
        old_disable = self.config.disable_local_search
        old_candidate_limit = self.config.local_search_candidate_pool_size
        old_swap_candidate_limit = self.config.swap_candidate_pool_size
        intensity_steps = {"light": 1, "medium": max(1, old_steps), "heavy": max(2, old_steps)}
        candidate_limit = (
            int(self.config.memetic_local_search_candidate_limit)
            if self.config.memetic_local_search_candidate_limit > 0
            else int(self.config.swap_candidate_pool_size)
        )
        self.config.local_search_steps = int(intensity_steps[self.config.memetic_local_search_intensity])
        self.config.disable_local_search = False
        self.config.local_search_candidate_pool_size = max(1, candidate_limit)
        self.config.swap_candidate_pool_size = max(1, candidate_limit)
        try:
            self.local_search_attempts += 1
            refined = super()._local_search(individual)
            if refined != individual:
                self.local_search_improvements += 1
            return refined
        finally:
            self.config.local_search_steps = old_steps
            self.config.disable_local_search = old_disable
            self.config.local_search_candidate_pool_size = old_candidate_limit
            self.config.swap_candidate_pool_size = old_swap_candidate_limit

    def _is_diverse_enough(self, candidate: tuple[Any, ...], survivors: Sequence[tuple[Any, ...]]) -> bool:
        if not survivors:
            return True
        candidate_set = set(candidate)
        distances: list[float] = []
        for survivor in survivors:
            survivor_set = set(survivor)
            union = candidate_set | survivor_set
            distances.append(1.0 - (len(candidate_set & survivor_set) / len(union)) if union else 0.0)
        return float(np.mean(distances)) >= 0.15

    def _survival_selection(
        self,
        population: Sequence[tuple[Any, ...]],
        offspring: Sequence[tuple[Any, ...]],
    ) -> list[CandidateEvaluation]:
        pool = list(population) + list(offspring)
        evaluated_unique: dict[tuple[Any, ...], CandidateEvaluation] = {}
        for seed_set in pool:
            evaluation = self._evaluate_seed_set(seed_set)
            evaluated_unique[evaluation.seed_set] = evaluation
        ranked = sorted(evaluated_unique.values(), key=self._candidate_rank_key, reverse=True)

        elite_count = max(1, int(math.ceil(float(self.config.population_size) * float(self.config.memetic_elitism_rate))))
        immigrant_count = int(math.floor(float(self.config.population_size) * float(self.config.memetic_random_immigrant_rate)))
        immigrant_count = min(max(0, immigrant_count), max(0, self.config.population_size - elite_count))
        target_before_immigrants = self.config.population_size - immigrant_count

        survivors: list[CandidateEvaluation] = ranked[:elite_count]
        survivor_seed_sets = [evaluation.seed_set for evaluation in survivors]
        for evaluation in ranked[elite_count:]:
            if len(survivors) >= target_before_immigrants:
                break
            if self.config.memetic_diversity_preservation and not self._is_diverse_enough(evaluation.seed_set, survivor_seed_sets):
                continue
            survivors.append(evaluation)
            survivor_seed_sets.append(evaluation.seed_set)
        for evaluation in ranked[elite_count:]:
            if len(survivors) >= target_before_immigrants:
                break
            if evaluation.seed_set in survivor_seed_sets:
                continue
            survivors.append(evaluation)
            survivor_seed_sets.append(evaluation.seed_set)

        while len(survivors) < self.config.population_size:
            immigrant = self._memetic_random_individual()
            evaluation = self._evaluate_seed_set(immigrant)
            survivors.append(evaluation)

        return sorted(survivors, key=self._candidate_rank_key, reverse=True)[: self.config.population_size]

    def optimize(self) -> OptimizationResult:
        """Run the fairness-first Memetic Algorithm end-to-end."""

        start = perf_counter()
        population = self._initialize_population()
        history_records: list[dict[str, float | int | str]] = []

        for generation in range(self.config.generations):
            generation_full_eval_calls_before = self.full_evaluation_calls
            generation_screening_eval_calls_before = self.screening_evaluation_calls
            swap_attempts_before = self.swap_attempts
            local_improvements_before = self.local_search_improvements

            current_evaluations = [self._evaluate_seed_set(individual) for individual in population]
            ranked_current = sorted(current_evaluations, key=self._candidate_rank_key, reverse=True)
            offspring: list[tuple[Any, ...]] = []
            while len(offspring) < self.config.population_size:
                parent_a = self._select_parent(ranked_current)
                parent_b = self._select_parent(ranked_current)
                child = parent_a.seed_set
                if self.rng.random() < self.config.crossover_probability:
                    child = self._crossover(parent_a.seed_set, parent_b.seed_set)
                child = self._mutate(child)
                if self.config.memetic_repair_enabled:
                    child = self._repair_seed_set(child)
                offspring.append(self._validate_seed_set(child))

            offspring_evaluations = [self._evaluate_seed_set(individual) for individual in offspring]
            if self._should_apply_memetic_local_search(generation):
                elite_count = self._memetic_local_search_elite_count(len(offspring_evaluations))
                elite_seed_sets = [
                    evaluation.seed_set
                    for evaluation in sorted(offspring_evaluations, key=self._candidate_rank_key, reverse=True)[:elite_count]
                ]
                refined_by_original: dict[tuple[Any, ...], tuple[Any, ...]] = {}
                for seed_set in elite_seed_sets:
                    refined_by_original[seed_set] = self._run_memetic_local_search(seed_set)
                offspring = [refined_by_original.get(seed_set, seed_set) for seed_set in offspring]

            next_evaluations = self._survival_selection(population, offspring)
            population = [evaluation.seed_set for evaluation in next_evaluations]
            for individual in population:
                self._validate_seed_set(individual)

            best_evaluation = max(next_evaluations, key=self._candidate_rank_key)
            average_score = float(np.mean([evaluation.score for evaluation in next_evaluations]))
            diversity = self._population_diversity(population)
            fairness_diagnostics = self._fairness_diagnostics(best_evaluation.fairness)
            history_records.append(
                {
                    "generation": generation,
                    "best_score": best_evaluation.score,
                    "best_fitness": best_evaluation.score,
                    "best_f_score": best_evaluation.f_score,
                    "best_objective_score": best_evaluation.score,
                    "best_spread": best_evaluation.total_spread_mean,
                    "best_mf": best_evaluation.fairness.mf,
                    "best_dcv": best_evaluation.fairness.dcv,
                    "average_population_score": average_score,
                    "population_diversity": diversity,
                    "best_seed_set": str(list(best_evaluation.seed_set)),
                    "local_search_applied_count": self.local_search_attempts,
                    "local_search_swap_evaluations": self.swap_attempts - swap_attempts_before,
                    "local_search_improvements": self.local_search_improvements - local_improvements_before,
                    "screening_evaluation_calls": self.screening_evaluation_calls - generation_screening_eval_calls_before,
                    "full_evaluation_calls": self.full_evaluation_calls - generation_full_eval_calls_before,
                    **fairness_diagnostics,
                }
            )

            if self.config.debug_logging:
                print(
                    "[memetic] "
                    f"gen={generation} "
                    f"best_fitness={best_evaluation.score:.6f} "
                    f"best_fscore={best_evaluation.f_score:.6f} "
                    f"best_mf={best_evaluation.fairness.mf:.6f} "
                    f"best_dcv={best_evaluation.fairness.dcv:.6f} "
                    f"diversity={diversity:.6f}"
                )

        final_evaluations = [self._evaluate_seed_set(individual) for individual in population]
        best_evaluation = max(final_evaluations, key=self._candidate_rank_key)
        repaired_seed_set = self._dcv_parity_repair(best_evaluation.seed_set)
        if repaired_seed_set != best_evaluation.seed_set:
            best_evaluation = self._evaluate_seed_set(repaired_seed_set, screening=False)
        runtime_seconds = perf_counter() - start
        self.final_diversity_score = self._population_diversity(population)
        search_verify = self.search_evaluator.verify() if self.search_evaluator is not None else {}

        return OptimizationResult(
            best_seed_set=best_evaluation.seed_set,
            best_score=best_evaluation.score,
            best_spread=best_evaluation.total_spread_mean,
            best_fairness=best_evaluation.fairness,
            runtime_seconds=runtime_seconds,
            candidate_pool_size=len(self.candidate_pool),
            history=pd.DataFrame(history_records),
            repaired_seed_sets=int(self.repaired_seed_sets),
            successful_swaps=int(self.swap_accepted),
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
            seed_quota_per_protected_group=dict(self.seed_quota_per_protected_group),
            initial_population_group_coverage_summary=dict(self.initial_population_group_coverage_summary),
            parity_repair_attempts=int(self.parity_repair_attempts),
            parity_repair_successes=int(self.parity_repair_successes),
            dcv_before_parity_repair=self.dcv_before_parity_repair,
            dcv_after_parity_repair_estimated=self.dcv_after_parity_repair_estimated,
            groups_rebalanced=tuple(sorted(self.groups_rebalanced, key=_sort_key)),
            optimizer_mode="ea_memetic",
            crossover_rate=float(self.config.crossover_probability),
            mutation_rate=float(self.config.mutation_probability),
            local_search_enabled=bool(self.config.memetic_local_search_enabled),
            local_search_top_elites=float(self.config.memetic_local_search_top_elites),
            local_search_intensity=str(self.config.memetic_local_search_intensity),
            local_search_attempts=int(self.swap_attempts),
            local_search_improvements=int(self.local_search_improvements),
            accepted_fscore_moves=int(self.swap_accepted_fscore_improvement),
            accepted_mf_dcv_moves=int(self.swap_accepted_mf_improvement),
            accepted_spread_safe_moves=int(self.swap_accepted_spread_fairness_preserved),
            rejected_fairness_drops=int(self.swap_rejected_fairness_degradation),
            search_mc_eval_calls=int(search_verify.get("search_mc_eval_calls", 0)),
            search_ris_eval_calls=int(search_verify.get("search_ris_eval_calls", 0)),
            search_fair_ris_eval_calls=int(search_verify.get("search_fair_ris_eval_calls", 0)),
            time_ris_evaluation=float(search_verify.get("time_ris_evaluation", 0.0)),
            time_mc_search_evaluation=float(search_verify.get("time_mc_search_evaluation", 0.0)),
            time_search_objective_total=float(search_verify.get("time_search_objective_total", 0.0)),
            rr_sets_used=int(search_verify.get("rr_sets_used", 0)),
            rr_sets_generated=int(search_verify.get("rr_sets_generated", 0)),
            duplicate_repairs=int(self.memetic_duplicate_repairs),
            budget_repairs=int(self.memetic_budget_repairs),
            community_repairs=int(self.memetic_community_repairs),
            diversity_score=float(self.final_diversity_score),
        )
