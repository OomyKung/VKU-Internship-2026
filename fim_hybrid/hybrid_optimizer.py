"""Unified hybrid Swarm Intelligence + Evolutionary optimizer for FIM."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

from .community_detection import CommunityDetectionResult
from .config import FairnessConfig, OptimizerConfig
from .diffusion import IndependentCascadeSimulator
from .fairness import FairnessMetrics, compute_group_sizes, evaluate_fairness


@dataclass(slots=True)
class CandidateEvaluation:
    """Evaluation result for a seed set."""

    # Keep the original seed set and its measured spread/fairness together so
    # the optimizer does not need to pass parallel arrays around.
    seed_set: tuple[object, ...]
    total_spread: float
    fairness: FairnessMetrics
    score: float


@dataclass(slots=True)
class HybridOptimizationResult:
    """Final optimization result."""

    # `history` stores per-generation diagnostics for later plotting/analysis.
    best_seed_set: list[object]
    best_score: float
    best_spread: float
    best_fairness: FairnessMetrics
    runtime_seconds: float
    candidate_pool_size: int
    history: pd.DataFrame


class HybridSIEAOptimizer:
    """Community-aware hybrid SI+EA optimizer for Fair Influence Maximization."""

    def __init__(
        self,
        graph: nx.Graph,
        community_result: CommunityDetectionResult,
        feature_frame: pd.DataFrame,
        simulator: IndependentCascadeSimulator,
        fairness_config: FairnessConfig,
        optimizer_config: OptimizerConfig,
        candidate_nodes: Optional[list[object]] = None,
        guidance_scores: Optional[pd.Series] = None,
        mc_runs: int = 100,
    ) -> None:
        self.graph = graph
        self.community_result = community_result
        self.feature_frame = feature_frame.copy()
        self.simulator = simulator
        self.fairness_config = fairness_config
        self.config = optimizer_config
        self.mc_runs = mc_runs
        self.rng = np.random.default_rng(optimizer_config.seed)
        self.guidance_scores = guidance_scores

        # Group sizes are reused in every fairness evaluation, so compute once.
        self.group_sizes = compute_group_sizes(graph, fairness_config.protected_attribute)
        self.candidate_pool = (
            list(candidate_nodes)
            if candidate_nodes is not None
            else self.feature_frame["node_id"].tolist()
        )
        # Preserve order while removing duplicates from the candidate pool.
        self.candidate_pool = list(dict.fromkeys(self.candidate_pool))
        if self.config.budget > len(self.candidate_pool):
            raise ValueError("Optimizer budget cannot exceed candidate pool size.")

        self.partition = community_result.partition
        # Precompute node priorities and community rankings for all later sampling,
        # repair, mutation, and swarm-move operations.
        self.node_priority = self._build_node_priority()
        self.global_ranked_nodes = sorted(
            self.candidate_pool,
            key=lambda node: self.node_priority[node],
            reverse=True,
        )
        self.community_ranked_nodes = self._build_community_rankings()
        self.evaluation_cache: dict[tuple[object, ...], CandidateEvaluation] = {}

    def _normalize_series(self, values: pd.Series) -> pd.Series:
        # Min-max normalization lets heterogeneous feature scales contribute in
        # one weighted priority score.
        minimum = float(values.min())
        maximum = float(values.max())
        if maximum - minimum < 1e-12:
            return pd.Series(1.0, index=values.index)
        return (values - minimum) / (maximum - minimum)

    def _build_node_priority(self) -> dict[object, float]:
        frame = self.feature_frame.set_index("node_id", drop=False).loc[self.candidate_pool].copy()

        # This weighted score is a simple research-friendly heuristic combining
        # structural influence proxies and community/fairness-aware features.
        base_score = (
            0.30 * self._normalize_series(frame["pagerank"]) +
            0.20 * self._normalize_series(frame["degree"]) +
            0.15 * self._normalize_series(frame["within_community_degree"]) +
            0.15 * self._normalize_series(frame["cross_community_degree"]) +
            0.10 * self._normalize_series(frame["community_size"]) +
            0.10 * self._normalize_series(frame["minority_neighbor_ratio"])
        )

        if self.guidance_scores is not None:
            # When ML guidance is available, blend it with the hand-crafted score
            # instead of replacing the graph-based heuristic completely.
            aligned = self.guidance_scores.reindex(frame.index).fillna(self.guidance_scores.min())
            guidance = self._normalize_series(aligned)
            base_score = 0.50 * base_score + 0.50 * guidance

        return base_score.to_dict()

    def _build_community_rankings(self) -> dict[int, list[object]]:
        # Store each community's candidate nodes ordered by priority.
        community_ranked_nodes: dict[int, list[object]] = {}
        for comm_idx, nodes in enumerate(self.community_result.communities):
            restricted_nodes = [node for node in nodes if node in self.candidate_pool]
            ranked = sorted(restricted_nodes, key=lambda node: self.node_priority[node], reverse=True)
            community_ranked_nodes[comm_idx] = ranked
        return community_ranked_nodes

    def _signature(self, seed_set: list[object] | tuple[object, ...]) -> tuple[object, ...]:
        # Canonicalize seed sets so caching and comparisons are reliable.
        return tuple(sorted(set(seed_set)))

    def _sample_from_sequence(self, nodes: list[object], top_window: int = 10) -> object:
        if not nodes:
            raise ValueError("Cannot sample from an empty node list.")

        # Sample from only the strongest part of the ranking to keep the search
        # focused while remaining stochastic.
        window = nodes[: min(len(nodes), top_window)]
        scores = np.asarray([self.node_priority[node] for node in window], dtype=float)
        if scores.sum() <= 0:
            scores = np.ones(len(window), dtype=float)
        probabilities = scores / scores.sum()
        choice_idx = int(self.rng.choice(len(window), p=probabilities))
        return window[choice_idx]

    def _available_node_from_community(
        self,
        community_idx: int,
        selected: set[object],
    ) -> Optional[object]:
        # Choose an unused node from a specific community.
        candidates = [node for node in self.community_ranked_nodes.get(community_idx, []) if node not in selected]
        if not candidates:
            return None
        return self._sample_from_sequence(candidates)

    def _available_global_node(self, selected: set[object]) -> Optional[object]:
        # Choose an unused node from the full candidate pool.
        candidates = [node for node in self.global_ranked_nodes if node not in selected]
        if not candidates:
            return None
        return self._sample_from_sequence(candidates)

    def _repair_seed_set(self, proposed_nodes: list[object]) -> tuple[object, ...]:
        # Repair is central to the unified optimizer: crossover, mutation, and
        # swarm moves can all propose invalid/duplicate/undersized seed sets.
        repaired: list[object] = []
        selected: set[object] = set()

        for node in proposed_nodes:
            # Keep only candidate-pool nodes and remove duplicates.
            if node not in self.candidate_pool or node in selected:
                continue
            repaired.append(node)
            selected.add(node)
            if len(repaired) == self.config.budget:
                return self._signature(repaired)

        represented_communities = {self.partition[node] for node in repaired}
        # First try to cover still-unrepresented communities to keep the search community-aware.
        unrepresented = [
            comm_idx
            for comm_idx in range(len(self.community_result.communities))
            if self.community_ranked_nodes.get(comm_idx) and comm_idx not in represented_communities
        ]

        while len(repaired) < self.config.budget and unrepresented:
            comm_idx = unrepresented.pop(0)
            candidate = self._available_node_from_community(comm_idx, selected)
            if candidate is None:
                continue
            repaired.append(candidate)
            selected.add(candidate)

        community_indices = list(self.community_ranked_nodes.keys())
        while len(repaired) < self.config.budget and community_indices:
            # Prefer communities that still have more unused candidate nodes available.
            weights = np.asarray(
                [
                    max(
                        1,
                        len([node for node in self.community_ranked_nodes[comm_idx] if node not in selected]),
                    )
                    for comm_idx in community_indices
                ],
                dtype=float,
            )
            if weights.sum() <= 0:
                break
            comm_idx = int(self.rng.choice(community_indices, p=weights / weights.sum()))
            candidate = self._available_node_from_community(comm_idx, selected)
            if candidate is None:
                community_indices = [idx for idx in community_indices if idx != comm_idx]
                continue
            repaired.append(candidate)
            selected.add(candidate)

        while len(repaired) < self.config.budget:
            # If community-aware filling still leaves gaps, fall back to the
            # strongest remaining global candidates.
            candidate = self._available_global_node(selected)
            if candidate is None:
                break
            repaired.append(candidate)
            selected.add(candidate)

        return self._signature(repaired)

    def _community_aware_seed_set(self) -> tuple[object, ...]:
        # This initializer seeds the population with candidates spread across
        # communities instead of collapsing onto one dense region of the graph.
        seed_nodes: list[object] = []
        selected: set[object] = set()

        community_order = list(self.community_ranked_nodes.keys())
        self.rng.shuffle(community_order)

        for comm_idx in community_order:
            candidate = self._available_node_from_community(comm_idx, selected)
            if candidate is None:
                continue
            seed_nodes.append(candidate)
            selected.add(candidate)
            if len(seed_nodes) == self.config.budget:
                break

        while len(seed_nodes) < self.config.budget:
            candidate = self._available_global_node(selected)
            if candidate is None:
                break
            seed_nodes.append(candidate)
            selected.add(candidate)

        return self._repair_seed_set(seed_nodes)

    def _initialize_population(self) -> list[tuple[object, ...]]:
        # Start with one strong deterministic candidate plus diverse stochastic ones.
        population: list[tuple[object, ...]] = []
        if self.global_ranked_nodes:
            population.append(self._signature(self.global_ranked_nodes[: self.config.budget]))

        while len(population) < self.config.population_size:
            population.append(self._community_aware_seed_set())

        return population

    def _crossover(
        self,
        parent_a: tuple[object, ...],
        parent_b: tuple[object, ...],
    ) -> tuple[object, ...]:
        # Uniform set-based crossover keeps the implementation simple and works
        # naturally for unordered seed sets.
        merged: list[object] = []
        for node in parent_a:
            if self.rng.random() < 0.5:
                merged.append(node)
        for node in parent_b:
            if self.rng.random() < 0.5:
                merged.append(node)
        if not merged:
            # Guarantee at least some inherited material from both parents.
            merged.extend(parent_a[: max(1, self.config.budget // 2)])
            merged.extend(parent_b[: max(1, self.config.budget // 2)])
        return self._repair_seed_set(merged)

    def _mutate(self, candidate: tuple[object, ...]) -> tuple[object, ...]:
        # Mutation tries to replace some nodes with alternatives from other
        # communities to preserve diversity and fairness coverage.
        mutated = list(candidate)
        for idx in range(len(mutated)):
            if self.rng.random() >= self.config.mutation_rate:
                continue
            community_idx = self.partition[mutated[idx]]
            selected = set(mutated)
            selected.remove(mutated[idx])

            alternative_communities = [idx_ for idx_ in self.community_ranked_nodes if idx_ != community_idx]
            self.rng.shuffle(alternative_communities)

            replacement: Optional[object] = None
            for alt_comm_idx in alternative_communities:
                replacement = self._available_node_from_community(alt_comm_idx, selected)
                if replacement is not None:
                    break
            if replacement is None:
                replacement = self._available_global_node(selected)
            if replacement is not None:
                mutated[idx] = replacement

        return self._repair_seed_set(mutated)

    def _swarm_move(
        self,
        current: tuple[object, ...],
        personal_best: tuple[object, ...],
        global_best: tuple[object, ...],
        elite_leader: tuple[object, ...],
    ) -> tuple[object, ...]:
        # Swarm movement combines four information sources:
        # current state, personal best, global best, and elite leader.
        seeds: list[object] = []

        if self.rng.random() >= self.config.restart_rate:
            seeds.extend([node for node in current if self.rng.random() < self.config.swarm_inertia_rate])
            seeds.extend([node for node in personal_best if self.rng.random() < self.config.swarm_cognitive_rate])
            seeds.extend([node for node in global_best if self.rng.random() < self.config.swarm_social_rate])
            seeds.extend([node for node in elite_leader if self.rng.random() < self.config.swarm_elite_rate])

        if not seeds and global_best:
            # Ensure the move does not become empty after stochastic filtering.
            seeds.append(global_best[0])

        return self._repair_seed_set(seeds)

    def _evaluate(self, seed_set: tuple[object, ...]) -> CandidateEvaluation:
        signature = self._signature(seed_set)
        if signature in self.evaluation_cache:
            # Cache Monte Carlo evaluations because they are the most expensive step.
            return self.evaluation_cache[signature]

        diffusion = self.simulator.simulate_many(
            signature,
            runs=self.mc_runs,
            protected_attribute=self.fairness_config.protected_attribute,
        )
        fairness = evaluate_fairness(
            group_spread=diffusion.group_spread_mean,
            group_sizes=self.group_sizes,
            lambda_weight=self.fairness_config.lambda_weight,
            target_mode=self.fairness_config.target_mode,
            total_spread=diffusion.total_spread_mean,
        )
        evaluation = CandidateEvaluation(
            seed_set=signature,
            total_spread=diffusion.total_spread_mean,
            fairness=fairness,
            score=fairness.combined_score,
        )
        self.evaluation_cache[signature] = evaluation
        return evaluation

    def _better(self, left: CandidateEvaluation, right: CandidateEvaluation) -> bool:
        # Compare by fairness-aware score first, then break ties on total spread.
        if left.score != right.score:
            return left.score > right.score
        return left.total_spread > right.total_spread

    def _elite_population(self, population: list[tuple[object, ...]], evaluations: list[CandidateEvaluation]) -> list[tuple[object, ...]]:
        # Elites serve two roles: parents for crossover and leaders for swarm guidance.
        elite_count = max(1, int(np.ceil(len(population) * self.config.elite_fraction)))
        ranked_indices = sorted(
            range(len(population)),
            key=lambda idx: (evaluations[idx].score, evaluations[idx].total_spread),
            reverse=True,
        )
        return [population[idx] for idx in ranked_indices[:elite_count]]

    def optimize(self) -> HybridOptimizationResult:
        """Run the unified hybrid SI+EA search."""

        start = perf_counter()
        # Initialize the current population and the best-known memories used by
        # the swarm component.
        population = self._initialize_population()
        current_evaluations = [self._evaluate(candidate) for candidate in population]
        personal_bests = population.copy()
        personal_best_evaluations = current_evaluations.copy()

        best_idx = max(range(len(personal_bests)), key=lambda idx: (personal_best_evaluations[idx].score, personal_best_evaluations[idx].total_spread))
        global_best = personal_bests[best_idx]
        global_best_evaluation = personal_best_evaluations[best_idx]

        history_records: list[dict[str, float | int]] = []
        stagnant_generations = 0

        for generation in range(self.config.generations):
            # Each generation produces two offspring streams:
            # one EA-style and one swarm-style.
            elites = self._elite_population(population, current_evaluations)
            evolutionary_population: list[tuple[object, ...]] = []
            swarm_population: list[tuple[object, ...]] = []

            for idx, current in enumerate(population):
                elite_leader = elites[idx % len(elites)]
                mate = elites[int(self.rng.integers(len(elites)))]

                if self.rng.random() < self.config.crossover_rate:
                    evolutionary_candidate = self._crossover(current, mate)
                else:
                    evolutionary_candidate = current
                # Mutation is applied after crossover so both offspring streams
                # retain local exploration ability.
                evolutionary_candidate = self._mutate(evolutionary_candidate)
                evolutionary_population.append(evolutionary_candidate)

                swarm_candidate = self._swarm_move(
                    current=current,
                    personal_best=personal_bests[idx],
                    global_best=global_best,
                    elite_leader=elite_leader,
                )
                swarm_candidate = self._mutate(swarm_candidate)
                swarm_population.append(swarm_candidate)

            # Evaluate both offspring streams with the same FIM objective.
            evolutionary_evaluations = [self._evaluate(candidate) for candidate in evolutionary_population]
            swarm_evaluations = [self._evaluate(candidate) for candidate in swarm_population]

            next_population: list[tuple[object, ...]] = []
            next_evaluations: list[CandidateEvaluation] = []
            for idx in range(self.config.population_size):
                # Survivor selection is integrated: each slot keeps the best of
                # current, EA offspring, swarm offspring, and personal memory.
                candidates = [
                    current_evaluations[idx],
                    evolutionary_evaluations[idx],
                    swarm_evaluations[idx],
                    personal_best_evaluations[idx],
                ]
                best_candidate = max(candidates, key=lambda item: (item.score, item.total_spread))
                next_population.append(best_candidate.seed_set)
                next_evaluations.append(best_candidate)

                if self._better(best_candidate, personal_best_evaluations[idx]):
                    personal_bests[idx] = best_candidate.seed_set
                    personal_best_evaluations[idx] = best_candidate

            population = next_population
            current_evaluations = next_evaluations

            # Update the global swarm leader from the current personal-best archive.
            previous_best = global_best_evaluation
            best_idx = max(
                range(len(personal_bests)),
                key=lambda idx: (personal_best_evaluations[idx].score, personal_best_evaluations[idx].total_spread),
            )
            global_best = personal_bests[best_idx]
            global_best_evaluation = personal_best_evaluations[best_idx]

            if self._better(global_best_evaluation, previous_best):
                stagnant_generations = 0
            else:
                stagnant_generations += 1

            # Store a compact history for later plots and ablation analysis.
            history_records.append(
                {
                    "generation": generation,
                    "best_score": global_best_evaluation.score,
                    "mean_score": float(np.mean([evaluation.score for evaluation in current_evaluations])),
                    "best_spread": global_best_evaluation.total_spread,
                    "best_mf": global_best_evaluation.fairness.mf,
                    "best_dcv": global_best_evaluation.fairness.dcv,
                }
            )

            if (
                self.config.convergence_patience is not None
                and stagnant_generations >= self.config.convergence_patience
            ):
                # Optional early stopping when the best score stops improving.
                break

        runtime_seconds = perf_counter() - start
        history = pd.DataFrame(history_records)
        return HybridOptimizationResult(
            best_seed_set=list(global_best),
            best_score=global_best_evaluation.score,
            best_spread=global_best_evaluation.total_spread,
            best_fairness=global_best_evaluation.fairness,
            runtime_seconds=runtime_seconds,
            candidate_pool_size=len(self.candidate_pool),
            history=history,
        )
