"""Phase 2 fairness metrics for protected-group spread diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    parity_target: float = 0.0
    parity_abs_error: float = 0.0
    parity_squared_error: float = 0.0
    over_served_groups: tuple[str, ...] = ()
    under_served_groups: tuple[str, ...] = ()
    # Shortfall DCV additions (Step 4–5 of the shortfall-DCV refactor).
    # dcv_shortfall: no-group-left-behind metric (primary when primary_dcv_mode=shortfall).
    # dcv_disparity: alias for original dcv — always populated for backward compat.
    dcv_shortfall: float = 0.0
    dcv_disparity: float = 0.0
    per_group_shortfall_violation: dict[str, float] = field(default_factory=dict)
    groups_below_target: tuple[str, ...] = ()
    groups_met_target: tuple[str, ...] = ()
    target_coverage_ratio: float = 1.0
    ideal_influences: dict[str, float] = field(default_factory=dict)


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


def compute_parity_diagnostics(
    normalized_group_spread: dict[str, float],
    parity_tolerance: float = 0.005,
) -> dict[str, float | tuple[str, ...]]:
    """Compute diagnostics for equal normalized influence across protected groups."""

    if not normalized_group_spread:
        raise ValueError("normalized_group_spread must contain at least one protected group.")
    tolerance = max(0.0, float(parity_tolerance))
    values = {
        group_name: float(value)
        for group_name, value in normalized_group_spread.items()
    }
    parity_target = float(np.mean(list(values.values())))
    return {
        "parity_target": parity_target,
        "parity_abs_error": float(sum(abs(value - parity_target) for value in values.values())),
        "parity_squared_error": float(sum((value - parity_target) ** 2 for value in values.values())),
        "over_served_groups": tuple(
            group_name
            for group_name, value in values.items()
            if value > parity_target + tolerance
        ),
        "under_served_groups": tuple(
            group_name
            for group_name, value in values.items()
            if value < parity_target - tolerance
        ),
    }


def compute_dcv_shortfall(
    group_influence: dict[str, float],
    ideal_influences: dict[str, float],
) -> tuple[float, dict[str, float], tuple[str, ...], tuple[str, ...], float]:
    """Shortfall DCV — no-group-left-behind metric.

    For each group: violation = max(0, (ideal - actual) / ideal).
    Groups that meet or exceed ideal contribute 0 violation.

    Returns:
        (dcv_shortfall, per_group_violation, groups_below, groups_met, target_coverage_ratio)
    """
    violations: dict[str, float] = {}
    groups_below: list[str] = []
    groups_met: list[str] = []
    all_groups = sorted(set(group_influence) | set(ideal_influences))
    for group_name in all_groups:
        ideal = float(ideal_influences.get(group_name, 0.0))
        actual = float(group_influence.get(group_name, 0.0))
        violation = max(0.0, (ideal - actual) / ideal) if ideal > 0.0 else 0.0
        violations[group_name] = violation
        if violation > 1e-9:
            groups_below.append(group_name)
        else:
            groups_met.append(group_name)
    total_groups = max(1, len(all_groups))
    dcv_shortfall = float(np.mean(list(violations.values()))) if violations else 0.0
    target_coverage_ratio = float(len(groups_met)) / float(total_groups)
    return dcv_shortfall, violations, tuple(groups_below), tuple(groups_met), target_coverage_ratio


def evaluate_fairness(
    group_spread: dict[str, float],
    group_sizes: dict[str, int],
    total_spread: float | None = None,
    include_soft_mf: bool = True,
    parity_tolerance: float = 0.005,
    ideal_influences: dict[str, float] | None = None,
) -> FairnessMetrics:
    """Compute normalized group spread, strict MF, soft MF, disparity DCV, and shortfall DCV.

    When ideal_influences is provided, dcv_shortfall is computed from per-group targets.
    When omitted, dcv_shortfall defaults to dcv (disparity-based) for backward compatibility.
    """
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
    parity = compute_parity_diagnostics(
        normalized_group_spread,
        parity_tolerance=parity_tolerance,
    )

    # Shortfall DCV — computed from ideal targets when available.
    if ideal_influences is not None and ideal_influences:
        (
            dcv_shortfall,
            per_group_shortfall_violation,
            groups_below_target,
            groups_met_target,
            target_coverage_ratio,
        ) = compute_dcv_shortfall(
            group_influence=completed_group_spread,
            ideal_influences=ideal_influences,
        )
        ideal_influences_used = dict(ideal_influences)
    else:
        dcv_shortfall = dcv
        per_group_shortfall_violation = {}
        groups_below_target = ()
        groups_met_target = ()
        target_coverage_ratio = 1.0
        ideal_influences_used: dict[str, float] = {}

    return FairnessMetrics(
        group_spread=completed_group_spread,
        normalized_group_spread=normalized_group_spread,
        group_targets=group_targets,
        mf=compute_strict_mf(normalized_group_spread),
        soft_mf=compute_soft_mf(normalized_group_spread) if include_soft_mf else None,
        dcv=dcv,
        parity_target=float(parity["parity_target"]),
        parity_abs_error=float(parity["parity_abs_error"]),
        parity_squared_error=float(parity["parity_squared_error"]),
        over_served_groups=tuple(parity["over_served_groups"]),
        under_served_groups=tuple(parity["under_served_groups"]),
        dcv_shortfall=dcv_shortfall,
        dcv_disparity=dcv,
        per_group_shortfall_violation=per_group_shortfall_violation,
        groups_below_target=groups_below_target,
        groups_met_target=groups_met_target,
        target_coverage_ratio=target_coverage_ratio,
        ideal_influences=ideal_influences_used,
    )
