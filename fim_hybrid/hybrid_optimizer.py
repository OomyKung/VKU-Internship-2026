"""Phase 5 unified hybrid SI+EA optimizer for Fair Influence Maximization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .community_detection import CommunityDetectionResult, get_community_stats, sample_community, sample_node_from_community
from .data_loader import LoadedDataset, ProtectedGroupReport
from .evaluation import evaluate_seed_set
from .feature_extraction import compute_node_features, compute_structural_node_scores
from .fairness import FairnessMetrics


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
    lambda_weight: float = 0.5
    random_seed: int = 42
    local_search_steps: int = 2
    disable_swarm_guidance: bool = False
    disable_crossover: bool = False
    disable_local_search: bool = False
    disable_community_aware_mutation: bool = False
    debug_logging: bool = False


@dataclass(slots=True)
class CandidateEvaluation:
    """Evaluation bundle for one candidate seed set."""

    seed_set: tuple[Any, ...]
    total_spread_mean: float
    total_spread_std: float
    fairness: FairnessMetrics
    score: float


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


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _normalize_seed_set(seed_set: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(sorted(set(seed_set), key=_sort_key))


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
    ) -> None:
        self.dataset = dataset
        self.protected_group_report = protected_group_report
        self.community_result = community_result
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.evaluation_cache: dict[tuple[Any, ...], CandidateEvaluation] = {}

        self._validate_inputs()
        self.candidate_pool = self._build_candidate_pool(candidate_nodes)
        self.candidate_pool_set = set(self.candidate_pool)
        self.available_communities = self._build_available_communities()
        self.available_community_stats = get_community_stats(self.available_communities)
        self.max_unique_seed_sets = math.comb(len(self.candidate_pool), self.config.budget)
        self.node_group_by_node = {
            node_id: group_name
            for group_name, node_ids in self.protected_group_report.protected_groups.items()
            for node_id in node_ids
        }
        self.node_scores = self._build_node_scores(node_scores)
        self.global_ranked_nodes = tuple(
            sorted(
                self.candidate_pool,
                key=lambda node_id: (-float(self.node_scores[node_id]), _sort_key(node_id)),
            )
        )

    def _validate_inputs(self) -> None:
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
        if self.config.mc_runs < 1:
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

    def _seed_sort_key(self, seed_set: tuple[Any, ...]) -> tuple[tuple[str, str], ...]:
        return tuple(_sort_key(node) for node in seed_set)

    def _candidate_rank_key(self, evaluation: CandidateEvaluation) -> tuple[float, float, tuple[tuple[str, str], ...]]:
        return (
            evaluation.score,
            evaluation.total_spread_mean,
            self._seed_sort_key(evaluation.seed_set),
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

    def _dynamic_candidate_score(
        self,
        node_id: Any,
        group_counts: dict[str, int],
        community_counts: dict[int, int],
    ) -> float:
        score = float(self.node_scores[node_id])
        group_name = self.node_group_by_node[node_id]
        community_id = self.community_result.community_id_by_node[node_id]
        if group_counts.get(group_name, 0) == 0:
            score += 0.20
        if not self.config.disable_community_aware_mutation and community_counts.get(community_id, 0) == 0:
            score += 0.20
        return score

    def _rank_external_candidates(self, seed_set: Sequence[Any]) -> list[Any]:
        selected_nodes = set(seed_set)
        group_counts = self._selected_group_counts(seed_set)
        community_counts = self._selected_community_counts(seed_set)
        return sorted(
            (node_id for node_id in self.global_ranked_nodes if node_id not in selected_nodes),
            key=lambda node_id: (-self._dynamic_candidate_score(node_id, group_counts, community_counts), _sort_key(node_id)),
        )

    def _rank_seed_nodes_for_replacement(self, seed_set: Sequence[Any]) -> list[Any]:
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
            return (score, _sort_key(node_id))

        return sorted(seed_set, key=replacement_priority)

    def _sample_unused_node(
        self,
        community_nodes: Sequence[Any],
        used_nodes: set[Any],
    ) -> Any | None:
        available_nodes = tuple(
            sorted(
                (node for node in community_nodes if node not in used_nodes),
                key=lambda node_id: (-float(self.node_scores[node_id]), _sort_key(node_id)),
            )
        )
        if not available_nodes:
            return None
        top_candidates = available_nodes[: min(3, len(available_nodes))]
        return sample_node_from_community(top_candidates, self.rng)

    def _sample_repair_node(self, used_nodes: set[Any]) -> Any | None:
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
        return self._sample_unused_node(candidate_communities[community_id], used_nodes)

    def _repair_seed_set(self, proposed_nodes: Sequence[Any]) -> tuple[Any, ...]:
        cleaned: list[Any] = []
        used_nodes: set[Any] = set()

        for node in proposed_nodes:
            if node not in self.candidate_pool_set or node in used_nodes:
                continue
            cleaned.append(node)
            used_nodes.add(node)
            if len(cleaned) == self.config.budget:
                return self._validate_seed_set(tuple(sorted(cleaned, key=_sort_key)))

        while len(cleaned) < self.config.budget:
            replacement = self._sample_repair_node(used_nodes)
            if replacement is None:
                break
            cleaned.append(replacement)
            used_nodes.add(replacement)

        if len(cleaned) < self.config.budget:
            for node in self.global_ranked_nodes:
                if node in used_nodes:
                    continue
                cleaned.append(node)
                used_nodes.add(node)
                if len(cleaned) == self.config.budget:
                    break

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

    def _initialize_individual(self) -> tuple[Any, ...]:
        seed_nodes: list[Any] = []
        used_nodes: set[Any] = set()

        while len(seed_nodes) < self.config.budget:
            node_id = self._sample_repair_node(used_nodes)
            if node_id is None:
                break
            seed_nodes.append(node_id)
            used_nodes.add(node_id)

        return self._repair_seed_set(seed_nodes)

    def _initialize_population(self) -> list[tuple[Any, ...]]:
        population: list[tuple[Any, ...]] = []
        seen: set[tuple[Any, ...]] = set()
        retry_count = 0

        strongest = self._validate_seed_set(tuple(self.global_ranked_nodes[: self.config.budget]))
        population.append(strongest)
        seen.add(strongest)

        while len(population) < self.config.population_size:
            candidate = self._initialize_individual()
            if candidate in seen and len(seen) < self.max_unique_seed_sets and retry_count < self._population_retry_limit():
                retry_count += 1
                continue
            population.append(candidate)
            seen.add(candidate)
            retry_count = 0

        return population

    def _evaluate_seed_set(self, seed_set: tuple[Any, ...]) -> CandidateEvaluation:
        normalized_seed_set = self._validate_seed_set(tuple(sorted(seed_set, key=_sort_key)))
        if normalized_seed_set in self.evaluation_cache:
            return self.evaluation_cache[normalized_seed_set]

        evaluation_result = evaluate_seed_set(
            dataset=self.dataset,
            protected_group_report=self.protected_group_report,
            seed_set=normalized_seed_set,
            propagation_probability=self.config.propagation_probability,
            mc_runs=self.config.mc_runs,
            random_seed=self.config.random_seed,
            lambda_weight=self.config.lambda_weight,
            include_soft_mf=True,
        )
        evaluation = CandidateEvaluation(
            seed_set=evaluation_result.seed_set,
            total_spread_mean=evaluation_result.total_spread_mean,
            total_spread_std=evaluation_result.total_spread_std,
            fairness=evaluation_result.fairness,
            score=evaluation_result.f_score,
        )
        self.evaluation_cache[normalized_seed_set] = evaluation
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
        for node in parent_a:
            if self.rng.random() < 0.5:
                merged_nodes.append(node)
        for node in parent_b:
            if self.rng.random() < 0.5:
                merged_nodes.append(node)
        if not merged_nodes:
            merged_nodes.extend(parent_a[: max(1, self.config.budget // 2)])
            merged_nodes.extend(parent_b[: max(1, self.config.budget // 2)])
        return self._repair_seed_set(merged_nodes)

    def _mutate(self, individual: tuple[Any, ...]) -> tuple[Any, ...]:
        mutated_nodes = list(individual)
        for index, node_id in enumerate(list(mutated_nodes)):
            if self.rng.random() >= self.config.mutation_probability:
                continue

            remaining = [node for idx, node in enumerate(mutated_nodes) if idx != index]
            ranked_candidates = self._rank_external_candidates(remaining)
            if not ranked_candidates:
                continue
            mutated_nodes[index] = ranked_candidates[0]

        return self._repair_seed_set(mutated_nodes)

    def _local_search(self, individual: tuple[Any, ...]) -> tuple[Any, ...]:
        if self.config.disable_local_search or self.config.local_search_steps == 0:
            return self._validate_seed_set(tuple(sorted(individual, key=_sort_key)))

        current = self._validate_seed_set(tuple(sorted(individual, key=_sort_key)))
        current_evaluation = self._evaluate_seed_set(current)

        for _ in range(self.config.local_search_steps):
            best_candidate = current
            best_evaluation = current_evaluation
            replacement_nodes = self._rank_seed_nodes_for_replacement(current)[: min(2, len(current))]
            external_candidates = self._rank_external_candidates(current)[: max(4, self.config.budget * 2)]

            for node_to_remove in replacement_nodes:
                partial = [node_id for node_id in current if node_id != node_to_remove]
                for node_to_add in external_candidates:
                    candidate = self._repair_seed_set(partial + [node_to_add])
                    evaluation = self._evaluate_seed_set(candidate)
                    if self._candidate_rank_key(evaluation) > self._candidate_rank_key(best_evaluation):
                        best_candidate = candidate
                        best_evaluation = evaluation

            if best_candidate == current:
                break
            current = best_candidate
            current_evaluation = best_evaluation

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

        for generation in range(self.config.generations):
            current_evaluations = [self._evaluate_seed_set(individual) for individual in population]
            elites = self._select_elites(current_evaluations)
            leader_pool = [evaluation.seed_set for evaluation in elites]
            ranked_current = sorted(current_evaluations, key=self._candidate_rank_key)
            weak_count = max(1, len(ranked_current) // 2)

            offspring: list[tuple[Any, ...]] = []
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
                child = self._local_search(child)
                offspring.append(self._validate_seed_set(child))

            next_evaluations = self._survival_selection(population, offspring)
            population = [evaluation.seed_set for evaluation in next_evaluations]
            for individual in population:
                self._validate_seed_set(individual)

            best_evaluation = max(next_evaluations, key=self._candidate_rank_key)
            average_score = float(np.mean([evaluation.score for evaluation in next_evaluations]))
            diversity = self._population_diversity(population)
            history_records.append(
                {
                    "generation": generation,
                    "best_score": best_evaluation.score,
                    "best_spread": best_evaluation.total_spread_mean,
                    "best_mf": best_evaluation.fairness.mf,
                    "best_dcv": best_evaluation.fairness.dcv,
                    "average_population_score": average_score,
                    "population_diversity": diversity,
                    "best_seed_set": str(list(best_evaluation.seed_set)),
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
        runtime_seconds = perf_counter() - start

        return HybridOptimizationResult(
            best_seed_set=best_evaluation.seed_set,
            best_score=best_evaluation.score,
            best_spread=best_evaluation.total_spread_mean,
            best_fairness=best_evaluation.fairness,
            runtime_seconds=runtime_seconds,
            candidate_pool_size=len(self.candidate_pool),
            history=pd.DataFrame(history_records),
        )
