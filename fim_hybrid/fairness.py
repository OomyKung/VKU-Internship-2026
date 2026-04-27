"""Phase 2 fairness metrics for protected-group spread diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .safe_math import safe_divide


@dataclass(slots=True)
class FairnessMetrics:
    """Fairness summary derived from mean per-group spread."""

    group_spread: dict[str, float]
    normalized_group_spread: dict[str, float]
    group_targets: dict[str, float]
    mf: float
    soft_mf: float | None
    dcv: float


def _normalize_group_spread_input(
    group_spread: dict[str, float],
    group_sizes: dict[str, int],
) -> dict[str, float]:
    if not group_sizes:
        raise ValueError("group_sizes must contain at least one protected group.")
    if any(size <= 0 for size in group_sizes.values()):
        raise ValueError("group_sizes must contain only positive group sizes.")

    unknown_groups = sorted(set(group_spread) - set(group_sizes))
    if unknown_groups:
        preview = ", ".join(repr(group_name) for group_name in unknown_groups[:5])
        raise ValueError(f"group_spread contains unknown groups: {preview}.")

    return {
        group_name: float(group_spread.get(group_name, 0.0))
        for group_name in group_sizes
    }


def compute_normalized_group_spread(
    group_spread: dict[str, float],
    group_sizes: dict[str, int],
) -> dict[str, float]:
    """Normalize mean group spread by protected-group size."""

    normalized_group_spread: dict[str, float] = {}
    for group_name, spread in _normalize_group_spread_input(group_spread, group_sizes).items():
        normalized_group_spread[group_name] = safe_divide(
            spread,
            float(group_sizes[group_name]),
            default=0.0,
            context=f"normalized spread for protected group {group_name}",
        )
    return normalized_group_spread


def compute_strict_mf(normalized_group_spread: dict[str, float]) -> float:
    """Return strict MF as the minimum normalized protected-group spread."""

    if not normalized_group_spread:
        raise ValueError("normalized_group_spread must contain at least one protected group.")
    return float(min(normalized_group_spread.values()))


def compute_soft_mf(normalized_group_spread: dict[str, float]) -> float:
    """Return diagnostic soft MF as the mean normalized protected-group spread."""

    if not normalized_group_spread:
        raise ValueError("normalized_group_spread must contain at least one protected group.")
    return float(np.mean(list(normalized_group_spread.values())))


def compute_dcv(
    group_spread: dict[str, float],
    group_sizes: dict[str, int],
    total_spread: float,
) -> tuple[float, dict[str, float]]:
    """Compute population-proportional DCV and the implied group targets."""

    if total_spread < 0.0:
        raise ValueError("total_spread must be non-negative.")

    completed_group_spread = _normalize_group_spread_input(group_spread, group_sizes)
    total_population = float(sum(group_sizes.values()))
    group_targets = {
        group_name: float(total_spread)
        * safe_divide(
            float(group_size),
            total_population,
            default=0.0,
            context=f"target quota for protected group {group_name}",
        )
        for group_name, group_size in group_sizes.items()
    }

    violations: list[float] = []
    for group_name, target in group_targets.items():
        achieved = completed_group_spread[group_name]
        safe_target = max(target, 1e-9)
        violations.append(
            safe_divide(
                max(target - achieved, 0.0),
                safe_target,
                default=0.0,
                context=f"DCV violation ratio for protected group {group_name}",
            )
        )

    return float(np.mean(violations)), group_targets


def evaluate_fairness(
    group_spread: dict[str, float],
    group_sizes: dict[str, int],
    total_spread: float | None = None,
    include_soft_mf: bool = True,
) -> FairnessMetrics:
    """Compute normalized group spread, strict MF, soft MF, and DCV."""

    normalized_group_spread = compute_normalized_group_spread(group_spread, group_sizes)
    completed_group_spread = {
        group_name: float(group_spread.get(group_name, 0.0))
        for group_name in group_sizes
    }

    if total_spread is None:
        total_spread = float(sum(completed_group_spread.values()))

    dcv, group_targets = compute_dcv(
        group_spread=completed_group_spread,
        group_sizes=group_sizes,
        total_spread=total_spread,
    )

    return FairnessMetrics(
        group_spread=completed_group_spread,
        normalized_group_spread=normalized_group_spread,
        group_targets=group_targets,
        mf=compute_strict_mf(normalized_group_spread),
        soft_mf=compute_soft_mf(normalized_group_spread) if include_soft_mf else None,
        dcv=dcv,
    )
