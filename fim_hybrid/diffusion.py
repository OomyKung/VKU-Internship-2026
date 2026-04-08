"""Independent Cascade diffusion simulator."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable
import zlib

import networkx as nx
import numpy as np


@dataclass(slots=True)
class DiffusionResult:
    """Monte Carlo diffusion summary."""

    # Mean and standard deviation are kept because spread can vary materially
    # across Monte Carlo runs.
    seed_set: tuple[object, ...]
    total_spread_mean: float
    total_spread_std: float
    group_spread_mean: dict[object, float]
    group_spread_std: dict[object, float]
    runtime_seconds: float


class IndependentCascadeSimulator:
    """Repeated Monte Carlo simulator for the Independent Cascade model."""

    def __init__(
        self,
        graph: nx.Graph,
        propagation_probability: float | None = None,
        seed: int = 42,
    ) -> None:
        self.graph = graph
        self.seed = seed
        # Precompute adjacency in a diffusion-friendly format once at startup.
        self.out_neighbors = self._build_adjacency(propagation_probability)

    def _build_adjacency(
        self,
        propagation_probability: float | None,
    ) -> dict[object, list[tuple[object, float]]]:
        adjacency: dict[object, list[tuple[object, float]]] = {}
        for node in self.graph.nodes():
            adjacency[node] = []
            # For undirected graphs, NetworkX exposes neighbors; for directed
            # graphs, the IC process follows outgoing edges.
            neighbors = self.graph.successors(node) if self.graph.is_directed() else self.graph.neighbors(node)
            for neighbor in neighbors:
                edge_probability = self.graph[node][neighbor].get("p", propagation_probability or 0.01)
                adjacency[node].append((neighbor, float(edge_probability)))
        return adjacency

    def _rng_for_seed_set(self, seed_set: Iterable[object], runs: int) -> np.random.Generator:
        # Tie the RNG seed to the seed set so repeated evaluations of the same
        # candidate are reproducible across the project.
        signature = "|".join(map(str, sorted(seed_set)))
        checksum = zlib.adler32(signature.encode("utf-8"))
        return np.random.default_rng(self.seed + checksum + runs)

    def simulate_once(
        self,
        seed_set: Iterable[object],
        rng: np.random.Generator,
    ) -> set[object]:
        """Run one Independent Cascade simulation."""

        active = set(seed_set)
        # BFS-style frontier expansion mirrors the IC process generation by generation.
        frontier = deque(seed_set)

        while frontier:
            source = frontier.popleft()
            for target, probability in self.out_neighbors[source]:
                if target in active:
                    continue
                # A live activation succeeds with the edge probability.
                if rng.random() <= probability:
                    active.add(target)
                    frontier.append(target)

        return active

    def simulate_many(
        self,
        seed_set: Iterable[object],
        runs: int = 100,
        protected_attribute: str | None = None,
    ) -> DiffusionResult:
        """Run repeated Monte Carlo simulations and aggregate spread statistics."""

        normalized_seed_set = tuple(sorted(set(seed_set)))
        if not normalized_seed_set:
            raise ValueError("seed_set must contain at least one node.")

        rng = self._rng_for_seed_set(normalized_seed_set, runs)
        total_spreads: list[float] = []
        group_spreads: dict[object, list[float]] = {}
        start = perf_counter()

        for _ in range(runs):
            active_nodes = self.simulate_once(normalized_seed_set, rng)
            total_spreads.append(float(len(active_nodes)))

            if protected_attribute:
                # Collect per-group activated counts for fairness evaluation.
                per_group: dict[object, float] = {}
                for node in active_nodes:
                    group_value = self.graph.nodes[node].get(protected_attribute, "__missing__")
                    per_group[group_value] = per_group.get(group_value, 0.0) + 1.0
                for group_value in set(group_spreads).union(per_group):
                    group_spreads.setdefault(group_value, []).append(per_group.get(group_value, 0.0))

        # Return summary statistics rather than all raw runs to keep the interface simple.
        runtime_seconds = perf_counter() - start
        total_spread_mean = float(np.mean(total_spreads))
        total_spread_std = float(np.std(total_spreads))
        group_spread_mean = {group: float(np.mean(values)) for group, values in group_spreads.items()}
        group_spread_std = {group: float(np.std(values)) for group, values in group_spreads.items()}

        return DiffusionResult(
            seed_set=normalized_seed_set,
            total_spread_mean=total_spread_mean,
            total_spread_std=total_spread_std,
            group_spread_mean=group_spread_mean,
            group_spread_std=group_spread_std,
            runtime_seconds=runtime_seconds,
        )
