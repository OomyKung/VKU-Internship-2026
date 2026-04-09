"""Phase 2 Independent Cascade diffusion utilities."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable
import zlib

import networkx as nx
import numpy as np

from .data_loader import LoadedDataset, ProtectedGroupReport


@dataclass(slots=True)
class DiffusionResult:
    """Monte Carlo diffusion summary for one seed set."""

    seed_set: tuple[Any, ...]
    total_spread_mean: float
    total_spread_std: float
    group_spread_mean: dict[str, float]
    group_spread_std: dict[str, float]
    runtime_seconds: float


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _normalize_seed_set(seed_set: Iterable[Any]) -> tuple[Any, ...]:
    seed_values = list(seed_set)
    if not seed_values:
        raise ValueError("seed_set must contain at least one node.")

    unique_seeds = set(seed_values)
    if len(unique_seeds) != len(seed_values):
        raise ValueError("seed_set must not contain duplicate nodes.")

    return tuple(sorted(unique_seeds, key=_sort_key))


def _validate_probability(propagation_probability: float) -> None:
    if not 0.0 <= propagation_probability <= 1.0:
        raise ValueError("propagation_probability must be between 0.0 and 1.0.")


def _validate_seed_nodes(graph: nx.Graph, seed_set: tuple[Any, ...]) -> None:
    missing_nodes = [node for node in seed_set if node not in graph]
    if missing_nodes:
        preview = ", ".join(repr(node) for node in missing_nodes[:5])
        raise ValueError(f"seed_set contains nodes not present in the graph: {preview}.")


def _edge_probability(graph: nx.Graph, source: Any, target: Any, fallback_probability: float) -> float:
    probability = float(graph[source][target].get("p", fallback_probability))
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"Edge probability for ({source!r}, {target!r}) must be between 0.0 and 1.0."
        )
    return probability


def _run_independent_cascade(
    graph: nx.Graph,
    seed_set: tuple[Any, ...],
    propagation_probability: float,
    rng: np.random.Generator,
) -> set[Any]:
    active_nodes = set(seed_set)
    frontier = deque(seed_set)

    while frontier:
        source = frontier.popleft()
        neighbors = graph.successors(source) if graph.is_directed() else graph.neighbors(source)
        for target in neighbors:
            if target in active_nodes:
                continue

            probability = _edge_probability(graph, source, target, propagation_probability)
            if rng.random() <= probability:
                active_nodes.add(target)
                frontier.append(target)

    return active_nodes


def _node_to_group_mapping(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
) -> dict[Any, str]:
    if dataset.name != protected_group_report.dataset_name:
        raise ValueError(
            "protected_group_report.dataset_name must match dataset.name."
        )
    if dataset.graph.is_directed() != protected_group_report.is_directed:
        raise ValueError(
            "protected_group_report.is_directed must match the dataset graph."
        )

    node_to_group: dict[Any, str] = {}
    for group_name, node_ids in protected_group_report.protected_groups.items():
        for node_id in node_ids:
            node_to_group[node_id] = group_name

    graph_nodes = set(dataset.graph.nodes())
    report_nodes = set(node_to_group)
    if graph_nodes != report_nodes:
        raise ValueError(
            "protected_group_report nodes must match the dataset graph nodes exactly."
        )

    return node_to_group


def _rng_for_seed_set(seed_set: tuple[Any, ...], random_seed: int) -> np.random.Generator:
    signature = "|".join(f"{type(node).__name__}:{repr(node)}" for node in seed_set)
    checksum = zlib.adler32(signature.encode("utf-8"))
    return np.random.default_rng(random_seed + checksum)


def simulate_independent_cascade_once(
    graph: nx.Graph,
    seed_set: Iterable[Any],
    propagation_probability: float,
    rng: np.random.Generator,
) -> set[Any]:
    """Run one Independent Cascade simulation and return activated nodes."""

    _validate_probability(propagation_probability)
    normalized_seed_set = _normalize_seed_set(seed_set)
    _validate_seed_nodes(graph, normalized_seed_set)
    return _run_independent_cascade(graph, normalized_seed_set, propagation_probability, rng)


def simulate_independent_cascade(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    seed_set: Iterable[Any],
    propagation_probability: float = 0.01,
    mc_runs: int = 100,
    random_seed: int = 42,
) -> DiffusionResult:
    """Run Monte Carlo Independent Cascade simulations with per-group spread tracking."""

    _validate_probability(propagation_probability)
    if mc_runs < 1:
        raise ValueError("mc_runs must be at least 1.")

    normalized_seed_set = _normalize_seed_set(seed_set)
    _validate_seed_nodes(dataset.graph, normalized_seed_set)
    node_to_group = _node_to_group_mapping(dataset, protected_group_report)
    group_names = list(protected_group_report.protected_groups)

    rng = _rng_for_seed_set(normalized_seed_set, random_seed)
    total_spreads: list[float] = []
    group_spreads: dict[str, list[float]] = {group_name: [] for group_name in group_names}

    start = perf_counter()
    for _ in range(mc_runs):
        active_nodes = _run_independent_cascade(
            dataset.graph,
            normalized_seed_set,
            propagation_probability,
            rng,
        )
        total_spreads.append(float(len(active_nodes)))

        per_group_counts = {group_name: 0.0 for group_name in group_names}
        for node_id in active_nodes:
            per_group_counts[node_to_group[node_id]] += 1.0

        for group_name in group_names:
            group_spreads[group_name].append(per_group_counts[group_name])

    runtime_seconds = perf_counter() - start
    total_spread_array = np.asarray(total_spreads, dtype=float)
    group_spread_mean = {
        group_name: float(np.mean(values))
        for group_name, values in group_spreads.items()
    }
    group_spread_std = {
        group_name: float(np.std(values))
        for group_name, values in group_spreads.items()
    }

    return DiffusionResult(
        seed_set=normalized_seed_set,
        total_spread_mean=float(np.mean(total_spread_array)),
        total_spread_std=float(np.std(total_spread_array)),
        group_spread_mean=group_spread_mean,
        group_spread_std=group_spread_std,
        runtime_seconds=runtime_seconds,
    )
