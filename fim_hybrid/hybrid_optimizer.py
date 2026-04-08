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

    _REQUIRED_FEATURE_COLUMNS = {
        "node_id",
        "degree",
        "pagerank",
        "community_size",
        "within_community_degree",
        "cross_community_degree",
        "minority_neighbor_ratio",
    }

    _RATE_FIELDS = (
        "crossover_rate",
        "mutation_rate",
        "elite_fraction",
        "swarm_inertia_rate",
        "swarm_cognitive_rate",
        "swarm_social_rate",
        "swarm_elite_rate",
        "restart_rate",
    )

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

        self._validate_inputs()

        # Group sizes are reused in every fairness evaluation, so compute once.
        self.group_sizes = compute_group_sizes(graph, fairness_config.protected_attribute)
        self.total_population = float(sum(self.group_sizes.values()))
        self.node_groups = {
            node: graph.nodes[node].get(fairness_config.protected_attribute, "__missing__")
            for node in graph.nodes()
        }
        self.seed_group_targets = self._build_seed_group_targets()
        raw_candidate_pool = (
            list(candidate_nodes)
            if candidate_nodes is not None
            else self.feature_frame["node_id"].tolist()
        )
        # Preserve order while removing duplicates from the candidate pool.
        self.candidate_pool = self._deduplicate_nodes(raw_candidate_pool)
        if not self.candidate_pool:
            raise ValueError("candidate_nodes must contain at least one usable node.")
        self.candidate_pool_set = set(self.candidate_pool)
        if self.config.budget > len(self.candidate_pool):
            raise ValueError("Optimizer budget cannot exceed candidate pool size.")

        feature_nodes = set(self.feature_frame["node_id"].tolist())
        missing_from_graph = [node for node in self.candidate_pool if node not in self.graph]
        if missing_from_graph:
            raise ValueError(
                f"candidate_nodes contains nodes missing from the graph: {missing_from_graph[:5]}"
            )
        missing_from_features = [node for node in self.candidate_pool if node not in feature_nodes]
        if missing_from_features:
            raise ValueError(
                "candidate_nodes contains nodes missing from feature_frame: "
                f"{missing_from_features[:5]}"
            )

        self.partition = community_result.partition
        missing_partition_nodes = [node for node in self.candidate_pool if node not in self.partition]
        if missing_partition_nodes:
            raise ValueError(
                "Community partition is missing candidate nodes: "
                f"{missing_partition_nodes[:5]}"
            )
        # Precompute node priorities and community rankings for all later sampling,
        # repair, mutation, and swarm-move operations.
        self.node_priority = self._build_node_priority()
        self.global_ranked_nodes = sorted(
            self.candidate_pool,
            key=lambda node: self.node_priority[node],
            reverse=True,
        )
        self.node_order = {node: idx for idx, node in enumerate(self.global_ranked_nodes)}
        self.community_ranked_nodes = self._build_community_rankings()
        self.evaluation_cache: dict[tuple[object, ...], CandidateEvaluation] = {}

    def _build_seed_group_targets(self) -> dict[object, float]:
        # These targets only guide local search operators. Final evaluation
        # still uses the diffusion-based fairness metrics.
        if self.fairness_config.target_mode == "uniform":
            per_group = self.config.budget / max(len(self.group_sizes), 1)
            return {group: per_group for group in self.group_sizes}
        if self.fairness_config.target_mode == "population_proportional":
            return {
                group: self.config.budget * (size / max(self.total_population, 1.0))
                for group, size in self.group_sizes.items()
            }
        raise ValueError(
            f"Unsupported target_mode '{self.fairness_config.target_mode}' for optimizer guidance."
        )

    def _selected_group_counts(self, selected: set[object]) -> dict[object, int]:
        counts = {group: 0 for group in self.group_sizes}
        for node in selected:
            counts[self.node_groups[node]] = counts.get(self.node_groups[node], 0) + 1
        return counts

    def _group_deficits(self, selected: set[object]) -> dict[object, float]:
        counts = self._selected_group_counts(selected)
        return {
            group: self.seed_group_targets.get(group, 0.0) - counts.get(group, 0)
            for group in self.group_sizes
        }

    def _fairness_priority_multiplier(self, node: object, selected: set[object] | None = None) -> float:
        if selected is None or self.config.fairness_repair_bias <= 0.0:
            return 1.0
        deficits = self._group_deficits(selected)
        return 1.0 + self.config.fairness_repair_bias * max(
            deficits.get(self.node_groups[node], 0.0),
            0.0,
        )

    def _population_diversity(self, population: list[tuple[object, ...]]) -> float:
        if len(population) < 2:
            return 0.0

        distances: list[float] = []
        seed_sets = [set(candidate) for candidate in population]
        for left_idx in range(len(seed_sets)):
            for right_idx in range(left_idx + 1, len(seed_sets)):
                union = seed_sets[left_idx] | seed_sets[right_idx]
                if not union:
                    distances.append(0.0)
                    continue
                overlap = seed_sets[left_idx] & seed_sets[right_idx]
                distances.append(1.0 - (len(overlap) / len(union)))
        return float(np.mean(distances)) if distances else 0.0

    def _seed_metadata(self, seed_set: tuple[object, ...]) -> dict[str, str]:
        communities = [self.partition[node] for node in seed_set]
        groups = [self.node_groups[node] for node in seed_set]
        group_counts = self._selected_group_counts(set(seed_set))
        labels = [f"{node}|g={self.node_groups[node]}|c={self.partition[node]}" for node in seed_set]
        return {
            "best_seed_nodes": str(list(seed_set)),
            "best_seed_groups": str(groups),
            "best_seed_communities": str(communities),
            "best_seed_group_counts": str(group_counts),
            "best_seed_labels": str(labels),
        }

    def _log_generation_diagnostics(
        self,
        generation: int,
        best_evaluation: CandidateEvaluation,
        population: list[tuple[object, ...]],
    ) -> None:
        if not self.config.debug_logging:
            return
        if generation % self.config.debug_frequency != 0:
            return

        metadata = self._seed_metadata(best_evaluation.seed_set)
        diversity = self._population_diversity(population)
        print(
            "[hybrid_siea] "
            f"gen={generation} "
            f"f={best_evaluation.score:.6f} "
            f"spread={best_evaluation.total_spread:.4f} "
            f"mf={best_evaluation.fairness.mf:.6f} "
            f"mf_to_ideal={best_evaluation.fairness.mf_to_ideal_ratio:.3f} "
            f"dcv={best_evaluation.fairness.dcv:.6f} "
            f"diversity={diversity:.3f}"
        )
        print(
            "[hybrid_siea] "
            f"best={metadata['best_seed_labels']} "
            f"group_counts={metadata['best_seed_group_counts']}"
        )

    def _validate_inputs(self) -> None:
        # Fail fast on malformed configurations so debugging stays local to the
        # optimizer instead of surfacing later as hard-to-trace runtime errors.
        if self.graph.number_of_nodes() == 0:
            raise ValueError("graph must contain at least one node.")
        if self.config.budget <= 0:
            raise ValueError("optimizer budget must be positive.")
        if self.config.population_size <= 0:
            raise ValueError("population_size must be positive.")
        if self.config.generations < 0:
            raise ValueError("generations must be non-negative.")
        if self.mc_runs <= 0:
            raise ValueError("mc_runs must be positive.")
        if self.config.debug_frequency <= 0:
            raise ValueError("debug_frequency must be positive.")
        if self.config.fairness_repair_bias < 0.0:
            raise ValueError("fairness_repair_bias must be non-negative.")
        if not 0.0 <= self.fairness_config.lambda_weight <= 1.0:
            raise ValueError("lambda_weight must be in the range [0, 1].")

        for field_name in self._RATE_FIELDS:
            value = float(getattr(self.config, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in the range [0, 1].")

        missing_columns = self._REQUIRED_FEATURE_COLUMNS.difference(self.feature_frame.columns)
        if missing_columns:
            raise ValueError(
                f"feature_frame is missing required columns: {sorted(missing_columns)}"
            )

        if self.feature_frame["node_id"].duplicated().any():
            raise ValueError("feature_frame contains duplicate node_id values.")

        if self.guidance_scores is not None and self.guidance_scores.empty:
            raise ValueError("guidance_scores cannot be empty when provided.")

        if not self.community_result.communities:
            raise ValueError("community_result must contain at least one community.")

    def _deduplicate_nodes(self, nodes: list[object] | tuple[object, ...]) -> list[object]:
        # Preserve the first occurrence so candidate order remains reproducible.
        unique_nodes: list[object] = []
        seen: set[object] = set()
        for node in nodes:
            if node in seen:
                continue
            seen.add(node)
            unique_nodes.append(node)
        return unique_nodes

    def _node_sort_key(self, node: object) -> tuple[int, str, str]:
        # Avoid relying on direct node comparability because custom datasets may
        # use heterogeneous or non-orderable node identifiers.
        return (
            self.node_order.get(node, len(self.node_order)),
            type(node).__name__,
            repr(node),
        )

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
        return tuple(sorted(self._deduplicate_nodes(list(seed_set)), key=self._node_sort_key))

    def _sample_from_sequence(
        self,
        nodes: list[object],
        top_window: int = 10,
        selected: set[object] | None = None,
    ) -> object:
        if not nodes:
            raise ValueError("Cannot sample from an empty node list.")

        # Sample from only the strongest part of the ranking to keep the search
        # focused while remaining stochastic.
        window = nodes[: min(len(nodes), top_window)]
        scores = np.asarray(
            [
                self.node_priority[node] * self._fairness_priority_multiplier(node, selected)
                for node in window
            ],
            dtype=float,
        )
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
        return self._sample_from_sequence(candidates, selected=selected)

    def _available_global_node(self, selected: set[object]) -> Optional[object]:
        # Choose an unused node from the full candidate pool.
        candidates = [node for node in self.global_ranked_nodes if node not in selected]
        if not candidates:
            return None
        return self._sample_from_sequence(candidates, selected=selected)

    def _community_search_order(
        self,
        selected: set[object],
        excluded_community: int | None = None,
    ) -> list[int]:
        # Prefer communities that are not yet represented in the seed set.
        represented = {self.partition[node] for node in selected}
        community_order = [
            comm_idx
            for comm_idx, nodes in self.community_ranked_nodes.items()
            if nodes and comm_idx != excluded_community
        ]
        self.rng.shuffle(community_order)
        return sorted(
            community_order,
            key=lambda comm_idx: (
                comm_idx in represented,
                -len([node for node in self.community_ranked_nodes[comm_idx] if node not in selected]),
            ),
        )

    def _repair_seed_set(self, proposed_nodes: list[object]) -> tuple[object, ...]:
        # Repair is central to the unified optimizer: crossover, mutation, and
        # swarm moves can all propose invalid/duplicate/undersized seed sets.
        repaired: list[object] = []
        selected: set[object] = set()

        for node in proposed_nodes:
            # Keep only candidate-pool nodes and remove duplicates.
            if node not in self.candidate_pool_set or node in selected:
                continue
            repaired.append(node)
            selected.add(node)
            if len(repaired) == self.config.budget:
                return self._validate_seed_set(self._signature(repaired))

        if not self.config.disable_community_repair:
            represented_communities = {self.partition[node] for node in repaired}
            # First try to cover still-unrepresented communities to keep the search community-aware.
            unrepresented = [
                comm_idx
                for comm_idx in range(len(self.community_result.communities))
                if self.community_ranked_nodes.get(comm_idx) and comm_idx not in represented_communities
            ]
            self.rng.shuffle(unrepresented)

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

        return self._validate_seed_set(self._signature(repaired))

    def _validate_seed_set(self, seed_set: tuple[object, ...]) -> tuple[object, ...]:
        # Keep every operator honest: all valid seed sets must be fixed-size,
        # duplicate-free, and limited to the candidate pool.
        if len(seed_set) != self.config.budget:
            raise RuntimeError(
                f"Invalid seed-set size {len(seed_set)} produced; expected {self.config.budget}."
            )
        if len(set(seed_set)) != len(seed_set):
            raise RuntimeError("Seed set contains duplicate nodes after repair.")
        invalid_nodes = [node for node in seed_set if node not in self.candidate_pool_set]
        if invalid_nodes:
            raise RuntimeError(f"Seed set contains invalid candidate nodes: {invalid_nodes[:5]}")
        return seed_set

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
        seen: set[tuple[object, ...]] = set()
        if self.global_ranked_nodes:
            strongest = self._validate_seed_set(
                self._signature(self.global_ranked_nodes[: self.config.budget])
            )
            population.append(strongest)
            seen.add(strongest)

        attempts = 0
        max_attempts = max(self.config.population_size * 10, 10)
        while len(population) < self.config.population_size:
            candidate = self._community_aware_seed_set()
            attempts += 1
            if candidate in seen and attempts < max_attempts:
                continue
            population.append(candidate)
            seen.add(candidate)
            attempts = 0

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

            replacement: Optional[object] = None
            for alt_comm_idx in self._community_search_order(
                selected=selected,
                excluded_community=community_idx,
            ):
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
        if self.rng.random() < self.config.restart_rate:
            # A true restart should inject a fresh community-aware candidate,
            # not a near-copy of the current global leader.
            return self._community_aware_seed_set()

        seeds: list[object] = []

        seeds.extend([node for node in current if self.rng.random() < self.config.swarm_inertia_rate])
        seeds.extend([node for node in personal_best if self.rng.random() < self.config.swarm_cognitive_rate])
        seeds.extend([node for node in global_best if self.rng.random() < self.config.swarm_social_rate])
        seeds.extend([node for node in elite_leader if self.rng.random() < self.config.swarm_elite_rate])

        if not seeds and global_best:
            # Ensure the move does not become empty after stochastic filtering.
            seeds.append(global_best[0])

        return self._repair_seed_set(seeds)

    def _evaluate(self, seed_set: tuple[object, ...]) -> CandidateEvaluation:
        signature = self._validate_seed_set(self._signature(seed_set))
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
            score_mode=self.fairness_config.score_mode,
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

                if not self.config.disable_crossover and self.rng.random() < self.config.crossover_rate:
                    evolutionary_candidate = self._crossover(current, mate)
                else:
                    evolutionary_candidate = current
                # Mutation is applied after crossover so both offspring streams
                # retain local exploration ability.
                evolutionary_candidate = self._mutate(evolutionary_candidate)
                evolutionary_population.append(evolutionary_candidate)

                if self.config.disable_swarm_guidance:
                    swarm_candidate = current
                else:
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
            metadata = self._seed_metadata(global_best_evaluation.seed_set)
            population_diversity = self._population_diversity(population)
            history_records.append(
                {
                    "generation": generation,
                    "best_score": global_best_evaluation.score,
                    "mean_score": float(np.mean([evaluation.score for evaluation in current_evaluations])),
                    "best_spread": global_best_evaluation.total_spread,
                    "best_mf": global_best_evaluation.fairness.mf,
                    "best_mf_to_ideal": global_best_evaluation.fairness.mf_to_ideal_ratio,
                    "best_dcv": global_best_evaluation.fairness.dcv,
                    "population_diversity": population_diversity,
                    **metadata,
                }
            )
            self._log_generation_diagnostics(generation, global_best_evaluation, population)

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
