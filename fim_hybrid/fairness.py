"""Fairness metrics for Fair Influence Maximization."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(slots=True)
class FairnessMetrics:
    """Fairness-aware evaluation output."""

    # `mf` and `dcv` are kept separately so analyses can inspect the trade-off
    # instead of looking only at the combined score.
    mf: float
    dcv: float
    combined_score: float
    group_spread: dict[object, float]
    group_targets: dict[object, float]
    normalized_group_spread: dict[object, float]
    ideal_mf: float
    mf_to_ideal_ratio: float
    score_mode: str
    mf_component: float


def compute_group_sizes(graph: nx.Graph, protected_attribute: str) -> dict[object, int]:
    """Count graph nodes by protected attribute value."""

    # Missing values are treated as their own group so fairness accounting stays explicit.
    group_sizes: dict[object, int] = {}
    for _, attrs in graph.nodes(data=True):
        value = attrs.get(protected_attribute, "__missing__")
        group_sizes[value] = group_sizes.get(value, 0) + 1
    return group_sizes


def compute_group_targets(
    group_sizes: dict[object, int],
    total_spread: float,
    target_mode: str = "population_proportional",
) -> dict[object, float]:
    """Create fairness targets for each group."""

    # Targets express what "fair enough" means before we compare achieved spread.
    total_population = float(sum(group_sizes.values()))
    if total_population <= 0:
        raise ValueError("group_sizes must contain at least one node.")

    if target_mode == "population_proportional":
        # Larger groups are expected to receive proportionally larger spread.
        return {
            group: total_spread * (size / total_population)
            for group, size in group_sizes.items()
        }
    if target_mode == "uniform":
        # Uniform targets ask each group to receive the same expected spread.
        uniform_target = total_spread / max(len(group_sizes), 1)
        return {group: uniform_target for group in group_sizes}

    raise ValueError(
        f"Unsupported target_mode '{target_mode}'. Supported modes: population_proportional, uniform."
    )


def maximin_fairness(
    group_spread: dict[object, float],
    group_sizes: dict[object, int],
) -> tuple[float, dict[object, float]]:
    """Compute maximin fairness as the minimum normalized group spread."""

    # Normalize by group size so large groups do not dominate the fairness metric.
    normalized = {}
    for group, size in group_sizes.items():
        normalized[group] = float(group_spread.get(group, 0.0)) / max(float(size), 1.0)
    return min(normalized.values()), normalized


def diversity_constraint_violation(
    group_spread: dict[object, float],
    group_targets: dict[object, float],
) -> float:
    """Average relative shortfall from desired group targets."""

    # Only shortfalls count as violations; exceeding the target is not penalized.
    violations = []
    for group, target in group_targets.items():
        achieved = float(group_spread.get(group, 0.0))
        safe_target = max(float(target), 1e-9)
        violations.append(max(target - achieved, 0.0) / safe_target)
    return float(np.mean(violations)) if violations else 0.0


def evaluate_fairness(
    group_spread: dict[object, float],
    group_sizes: dict[object, int],
    lambda_weight: float,
    group_targets: dict[object, float] | None = None,
    target_mode: str = "population_proportional",
    total_spread: float | None = None,
    score_mode: str = "raw_mf",
) -> FairnessMetrics:
    """Compute MF, DCV, and the combined fairness-aware score."""

    # If no explicit total spread is given, infer it from the per-group spread totals.
    if total_spread is None:
        total_spread = float(sum(group_spread.values()))

    if group_targets is None:
        group_targets = compute_group_targets(
            group_sizes=group_sizes,
            total_spread=total_spread,
            target_mode=target_mode,
        )

    mf, normalized = maximin_fairness(group_spread, group_sizes)
    dcv = diversity_constraint_violation(group_spread, group_targets)
    total_population = float(sum(group_sizes.values()))
    # Under perfectly proportional spread, every group's normalized spread is
    # equal to total_spread / total_population. This is a useful ceiling-like
    # diagnostic for understanding why MF can look numerically small.
    ideal_mf = total_spread / max(total_population, 1.0)
    mf_to_ideal_ratio = mf / ideal_mf if ideal_mf > 1e-12 else 0.0
    if score_mode == "raw_mf":
        mf_component = mf
    elif score_mode == "normalized_mf":
        mf_component = mf_to_ideal_ratio
    else:
        raise ValueError(
            f"Unsupported score_mode '{score_mode}'. Supported modes: raw_mf, normalized_mf."
        )

    # This is the main optimization objective used by the hybrid algorithm.
    combined_score = lambda_weight * mf_component - (1.0 - lambda_weight) * dcv

    return FairnessMetrics(
        mf=mf,
        dcv=dcv,
        combined_score=combined_score,
        group_spread={group: float(group_spread.get(group, 0.0)) for group in group_sizes},
        group_targets=group_targets,
        normalized_group_spread=normalized,
        ideal_mf=ideal_mf,
        mf_to_ideal_ratio=mf_to_ideal_ratio,
        score_mode=score_mode,
        mf_component=mf_component,
    )
