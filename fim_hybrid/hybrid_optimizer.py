"""Compatibility guard for the removed Swarm Intelligence optimizer."""

from __future__ import annotations

from .removed_swarm import REMOVED_SWARM_FEATURE_MESSAGE


class RemovedSwarmOptimizerError(RuntimeError):
    """Raised when retired Swarm Intelligence optimizer code is requested."""


def _raise_removed_swarm(*_args: object, **_kwargs: object) -> None:
    raise RemovedSwarmOptimizerError(REMOVED_SWARM_FEATURE_MESSAGE)


HybridSIEAConfig = _raise_removed_swarm
HybridSIEAOptimizer = _raise_removed_swarm
HybridOptimizationResult = _raise_removed_swarm
