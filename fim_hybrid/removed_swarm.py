"""Removed Swarm Intelligence option guards."""

from __future__ import annotations

REMOVED_SWARM_FEATURE_MESSAGE = (
    "Removed feature: Swarm Intelligence is no longer part of this framework. "
    "Use the EA+Memetic optimizer instead."
)
REMOVED_SWARM_OPTIMIZER_MESSAGE = "Swarm Intelligence has been removed. Use --optimizer ea_memetic instead."

REMOVED_SWARM_OPTIMIZERS = frozenset(
    {
        "hybrid_si_ea",
        "hybrid_siea",
        "swarm",
        "si_ea",
        "swarm_ea",
    }
)


def normalize_optimizer_name(value: object) -> str:
    """Normalize optimizer CLI/config values."""

    return str(value or "").strip().lower().replace("-", "_")


def ensure_active_optimizer(value: object) -> str:
    """Return the active optimizer name or raise the removed-feature error."""

    normalized = normalize_optimizer_name(value)
    if normalized in REMOVED_SWARM_OPTIMIZERS:
        raise ValueError(REMOVED_SWARM_OPTIMIZER_MESSAGE)
    if normalized != "ea_memetic":
        raise ValueError(f"Unsupported optimizer '{value}'. Use --optimizer ea_memetic.")
    return "ea_memetic"


def is_removed_swarm_stack_name(value: object) -> bool:
    """Detect stack names from the retired Swarm Intelligence framework."""

    normalized = str(value or "").strip().lower().replace("-", "_")
    return (
        normalized in REMOVED_SWARM_OPTIMIZERS
        or normalized.endswith("_siea")
        or "_siea_" in normalized
        or normalized.endswith("_si_ea")
        or "_si_ea_" in normalized
        or "swarm" in normalized
    )


def ensure_active_stack_name(value: object) -> str:
    """Return a stack name or raise the removed-feature message for retired stacks."""

    normalized = str(value or "").strip()
    if is_removed_swarm_stack_name(normalized):
        raise ValueError(REMOVED_SWARM_FEATURE_MESSAGE)
    return normalized
