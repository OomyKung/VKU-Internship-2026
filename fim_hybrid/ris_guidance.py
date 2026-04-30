"""Reverse Influence Sampling helpers for search-time candidate guidance."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable

import networkx as nx
import numpy as np

from .data_loader import LoadedDataset, ProtectedGroupReport
from .safe_math import safe_divide, safe_minmax_normalize


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _normalize_score_map(scores: dict[Any, float], nodes: Iterable[Any]) -> dict[Any, float]:
    ordered_nodes = tuple(nodes)
    if not ordered_nodes:
        return {}

    normalized = safe_minmax_normalize(
        [float(scores.get(node_id, 0.0)) for node_id in ordered_nodes],
        default=0.0,
        context="RIS score normalization",
    )
    return {
        node_id: float(score)
        for node_id, score in zip(ordered_nodes, normalized, strict=True)
    }


@dataclass(slots=True)
class RISConfig:
    """Configuration for deterministic Reverse Influence Sampling guidance."""

    num_rr_sets: int = 256
    random_seed: int = 42
    reuse_rr_sets: bool = True
    mode: str = "global"


@dataclass(slots=True)
class RISGuidanceResult:
    """Reusable RR-set statistics for node-level search-time guidance."""

    rr_sets: tuple[frozenset[Any], ...]
    rr_root_nodes: tuple[Any, ...]
    rr_root_groups: tuple[str, ...]
    global_node_scores: dict[Any, float]
    node_rr_counts: dict[Any, int]
    node_group_rr_counts: dict[Any, dict[str, int]]
    rr_set_counts_by_group: dict[str, int]
    runtime_seconds: float
    config: RISConfig

    def weighted_node_scores(self, group_weights: dict[str, float] | None = None) -> dict[Any, float]:
        """Return normalized node scores under optional protected-group RR weights."""

        if group_weights is None:
            group_weights = {
                group_name: 1.0
                for group_name in self.rr_set_counts_by_group
            }
        weighted_denominator = float(
            sum(
                float(group_weights.get(group_name, 0.0)) * float(rr_count)
                for group_name, rr_count in self.rr_set_counts_by_group.items()
            )
        )
        if weighted_denominator <= 0.0:
            return {node_id: 0.0 for node_id in self.node_rr_counts}

        raw_scores: dict[Any, float] = {}
        for node_id, group_counts in self.node_group_rr_counts.items():
            weighted_numerator = float(
                sum(
                    float(group_weights.get(group_name, 0.0)) * float(group_counts.get(group_name, 0))
                    for group_name in self.rr_set_counts_by_group
                )
            )
            raw_scores[node_id] = safe_divide(
                weighted_numerator,
                weighted_denominator,
                default=0.0,
                context="weighted RIS score",
            )
        return _normalize_score_map(raw_scores, self.node_rr_counts)


def _validate_config(config: RISConfig) -> None:
    if config.num_rr_sets < 1:
        raise ValueError("ris_num_rr_sets must be at least 1.")
    if config.mode not in {"global", "standard", "weak_group_weighted", "group_balanced"}:
        raise ValueError("ris_mode must be one of ['standard', 'global', 'weak_group_weighted', 'group_balanced'].")


def _reverse_reachable_set(
    reverse_graph: nx.Graph,
    root_node: Any,
    propagation_probability: float,
    rng: np.random.Generator,
) -> frozenset[Any]:
    rr_nodes = {root_node}
    frontier = [root_node]
    while frontier:
        current_node = frontier.pop()
        for predecessor in sorted(reverse_graph.neighbors(current_node), key=_sort_key):
            if predecessor in rr_nodes:
                continue
            if float(rng.random()) <= float(propagation_probability):
                rr_nodes.add(predecessor)
                frontier.append(predecessor)
    return frozenset(rr_nodes)


def generate_ris_guidance(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    propagation_probability: float,
    config: RISConfig,
) -> RISGuidanceResult:
    """Generate RR sets and reusable node-level coverage statistics for IC guidance."""

    _validate_config(config)
    if not 0.0 <= float(propagation_probability) <= 1.0:
        raise ValueError("propagation_probability must be between 0.0 and 1.0 for RIS generation.")
    if dataset.name != protected_group_report.dataset_name:
        raise ValueError("protected_group_report.dataset_name must match dataset.name.")

    start = perf_counter()
    graph = dataset.graph
    reverse_graph = graph.reverse(copy=False) if graph.is_directed() else graph
    ordered_nodes = tuple(sorted(graph.nodes(), key=_sort_key))
    if not ordered_nodes:
        raise ValueError("dataset.graph must contain at least one node for RIS guidance.")

    group_by_node = {
        node_id: group_name
        for group_name, node_ids in protected_group_report.protected_groups.items()
        for node_id in node_ids
    }
    missing_groups = [node_id for node_id in ordered_nodes if node_id not in group_by_node]
    if missing_groups:
        raise ValueError(f"Protected group mapping is missing graph nodes: {missing_groups[:5]}.")

    rng = np.random.default_rng(int(config.random_seed))
    rr_sets: list[frozenset[Any]] = []
    rr_root_nodes: list[Any] = []
    rr_root_groups: list[str] = []
    node_rr_counts = {node_id: 0 for node_id in ordered_nodes}
    node_group_rr_counts = {
        node_id: {group_name: 0 for group_name in protected_group_report.group_sizes}
        for node_id in ordered_nodes
    }
    rr_set_counts_by_group = {group_name: 0 for group_name in protected_group_report.group_sizes}

    for _ in range(int(config.num_rr_sets)):
        root_index = int(rng.integers(len(ordered_nodes)))
        root_node = ordered_nodes[root_index]
        root_group = group_by_node[root_node]
        rr_set = _reverse_reachable_set(
            reverse_graph=reverse_graph,
            root_node=root_node,
            propagation_probability=float(propagation_probability),
            rng=rng,
        )
        rr_sets.append(rr_set)
        rr_root_nodes.append(root_node)
        rr_root_groups.append(root_group)
        rr_set_counts_by_group[root_group] += 1
        for node_id in rr_set:
            node_rr_counts[node_id] += 1
            node_group_rr_counts[node_id][root_group] += 1

    global_scores = {
        node_id: safe_divide(
            float(node_rr_counts[node_id]),
            float(config.num_rr_sets),
            default=0.0,
            context="global RIS score",
        )
        for node_id in ordered_nodes
    }
    return RISGuidanceResult(
        rr_sets=tuple(rr_sets),
        rr_root_nodes=tuple(rr_root_nodes),
        rr_root_groups=tuple(rr_root_groups),
        global_node_scores=_normalize_score_map(global_scores, ordered_nodes),
        node_rr_counts=node_rr_counts,
        node_group_rr_counts=node_group_rr_counts,
        rr_set_counts_by_group=rr_set_counts_by_group,
        runtime_seconds=perf_counter() - start,
        config=config,
    )
