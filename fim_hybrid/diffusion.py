"""Diffusion-model utilities for fair influence maximization."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable
import zlib

import networkx as nx
import numpy as np

from .data_loader import LoadedDataset, ProtectedGroupReport

DEFAULT_DIFFUSION_MODEL = "ic"
SUPPORTED_DIFFUSION_MODELS = (DEFAULT_DIFFUSION_MODEL, "lt", "wc")


@dataclass(slots=True)
class DiffusionResult:
    """Monte Carlo diffusion summary for one seed set."""

    seed_set: tuple[Any, ...]
    total_spread_mean: float
    total_spread_std: float
    group_spread_mean: dict[str, float]
    group_spread_std: dict[str, float]
    runtime_seconds: float


@dataclass(frozen=True, slots=True)
class DiffusionModelSpec:
    """Static metadata for one supported diffusion model."""

    name: str
    description: str
    search_supported: bool = True
    final_evaluation_supported: bool = True
    default: bool = False


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _diffusion_registry() -> dict[str, DiffusionModelSpec]:
    return {
        "ic": DiffusionModelSpec(
            name="ic",
            description="Independent Cascade with edge-specific or fallback activation probabilities.",
            default=True,
        ),
        "lt": DiffusionModelSpec(
            name="lt",
            description="Linear Threshold using edge weights or normalized incoming influence.",
        ),
        "wc": DiffusionModelSpec(
            name="wc",
            description="Weighted Cascade using inverse in-degree activation probabilities.",
        ),
    }


def available_diffusion_models() -> tuple[str, ...]:
    """Return the supported diffusion model names."""

    return tuple(_diffusion_registry())


def get_diffusion_model_spec(diffusion_model: str) -> DiffusionModelSpec:
    """Return static metadata for one diffusion model."""

    key = str(diffusion_model).strip().lower()
    registry = _diffusion_registry()
    if key not in registry:
        raise ValueError(f"Unknown diffusion model '{diffusion_model}'.")
    return registry[key]


def _normalize_seed_set(seed_set: Iterable[Any]) -> tuple[Any, ...]:
    seed_values = list(seed_set)
    if not seed_values:
        return ()

    unique_seeds = set(seed_values)
    if len(unique_seeds) != len(seed_values):
        raise ValueError("seed_set must not contain duplicate nodes.")

    return tuple(sorted(unique_seeds, key=_sort_key))


def validate_diffusion_model(diffusion_model: str = DEFAULT_DIFFUSION_MODEL) -> str:
    """Validate and normalize the requested diffusion model name."""

    normalized = str(diffusion_model).strip().lower()
    if normalized not in SUPPORTED_DIFFUSION_MODELS:
        supported = ", ".join(SUPPORTED_DIFFUSION_MODELS)
        raise ValueError(
            f"Unsupported diffusion_model '{diffusion_model}'. Supported models: {supported}."
        )
    return normalized


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


def _weighted_cascade_probability(graph: nx.Graph, source: Any, target: Any) -> float:
    del source
    if graph.is_directed():
        denominator = graph.in_degree(target)
    else:
        denominator = graph.degree(target)
    if denominator <= 0:
        return 0.0
    return float(1.0 / float(denominator))


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


def _run_weighted_cascade(
    graph: nx.Graph,
    seed_set: tuple[Any, ...],
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

            probability = _weighted_cascade_probability(graph, source, target)
            if rng.random() <= probability:
                active_nodes.add(target)
                frontier.append(target)

    return active_nodes


def _resolve_threshold(graph: nx.Graph, node_id: Any, fallback_threshold: float) -> float:
    threshold = float(graph.nodes[node_id].get("threshold", fallback_threshold))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Node threshold for {node_id!r} must be between 0.0 and 1.0.")
    return threshold


def _lt_neighbor_weight(graph: nx.Graph, source: Any, target: Any) -> float:
    weight = graph[source][target].get("weight")
    if weight is not None:
        resolved = float(weight)
        if resolved < 0.0:
            raise ValueError(f"Edge weight for ({source!r}, {target!r}) must be non-negative.")
        return resolved

    if graph.is_directed():
        predecessors = list(graph.predecessors(target))
        denominator = len(predecessors)
    else:
        denominator = graph.degree(target)
    if denominator <= 0:
        return 0.0
    return float(1.0 / float(denominator))


def _run_linear_threshold(
    graph: nx.Graph,
    seed_set: tuple[Any, ...],
    default_threshold: float,
    rng: np.random.Generator,
) -> set[Any]:
    thresholds = {
        node_id: _resolve_threshold(graph, node_id, default_threshold)
        for node_id in graph.nodes()
    }
    active_nodes = set(seed_set)
    inactive_nodes = set(graph.nodes()) - active_nodes
    if not inactive_nodes:
        return active_nodes

    tie_breakers = {
        node_id: float(rng.random())
        for node_id in graph.nodes()
    }
    changed = True
    while changed:
        changed = False
        next_frontier: list[Any] = []
        ordered_inactive = sorted(
            inactive_nodes,
            key=lambda node_id: (tie_breakers[node_id], _sort_key(node_id)),
        )
        for target in ordered_inactive:
            if graph.is_directed():
                neighbors = graph.predecessors(target)
            else:
                neighbors = graph.neighbors(target)
            influence = 0.0
            for source in neighbors:
                if source not in active_nodes:
                    continue
                influence += _lt_neighbor_weight(graph, source, target)
            if influence + 1e-12 >= thresholds[target]:
                next_frontier.append(target)

        if next_frontier:
            active_nodes.update(next_frontier)
            inactive_nodes.difference_update(next_frontier)
            changed = True

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


def simulate_linear_threshold_once(
    graph: nx.Graph,
    seed_set: Iterable[Any],
    propagation_probability: float,
    rng: np.random.Generator,
) -> set[Any]:
    """Run one Linear Threshold simulation and return activated nodes.

    The existing propagation_probability parameter is reused as the default
    node threshold when a node-specific ``threshold`` attribute is absent.
    """

    _validate_probability(propagation_probability)
    normalized_seed_set = _normalize_seed_set(seed_set)
    _validate_seed_nodes(graph, normalized_seed_set)
    return _run_linear_threshold(graph, normalized_seed_set, propagation_probability, rng)


def simulate_weighted_cascade_once(
    graph: nx.Graph,
    seed_set: Iterable[Any],
    rng: np.random.Generator,
) -> set[Any]:
    """Run one Weighted Cascade simulation and return activated nodes."""

    normalized_seed_set = _normalize_seed_set(seed_set)
    _validate_seed_nodes(graph, normalized_seed_set)
    return _run_weighted_cascade(graph, normalized_seed_set, rng)


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


def simulate_linear_threshold(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    seed_set: Iterable[Any],
    propagation_probability: float = 0.01,
    mc_runs: int = 100,
    random_seed: int = 42,
) -> DiffusionResult:
    """Run Monte Carlo Linear Threshold simulations with per-group spread tracking."""

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
        active_nodes = _run_linear_threshold(
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


def simulate_weighted_cascade(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    seed_set: Iterable[Any],
    propagation_probability: float = 0.01,
    mc_runs: int = 100,
    random_seed: int = 42,
) -> DiffusionResult:
    """Run Monte Carlo Weighted Cascade simulations with per-group spread tracking."""

    del propagation_probability
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
        active_nodes = _run_weighted_cascade(
            dataset.graph,
            normalized_seed_set,
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


def simulate_diffusion(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    seed_set: Iterable[Any],
    propagation_probability: float = 0.01,
    mc_runs: int = 100,
    random_seed: int = 42,
    diffusion_model: str = DEFAULT_DIFFUSION_MODEL,
) -> DiffusionResult:
    """Run the requested diffusion model through one shared dispatch surface."""

    resolved_model = validate_diffusion_model(diffusion_model)
    if resolved_model == "ic":
        return simulate_independent_cascade(
            dataset=dataset,
            protected_group_report=protected_group_report,
            seed_set=seed_set,
            propagation_probability=propagation_probability,
            mc_runs=mc_runs,
            random_seed=random_seed,
        )
    if resolved_model == "lt":
        return simulate_linear_threshold(
            dataset=dataset,
            protected_group_report=protected_group_report,
            seed_set=seed_set,
            propagation_probability=propagation_probability,
            mc_runs=mc_runs,
            random_seed=random_seed,
        )
    if resolved_model == "wc":
        return simulate_weighted_cascade(
            dataset=dataset,
            protected_group_report=protected_group_report,
            seed_set=seed_set,
            propagation_probability=propagation_probability,
            mc_runs=mc_runs,
            random_seed=random_seed,
        )
    raise ValueError(f"Unsupported diffusion_model '{diffusion_model}'.")
