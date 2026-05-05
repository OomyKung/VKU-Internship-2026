"""Evolutionary Memetic optimizer for Fair Influence Maximization.

Combines EA global search with memetic local refinement using
professor-priority fitness (F-score > MF > DCV > spread > runtime).
Configured via dedicated --ea-* CLI flags, independent of the existing
memetic optimizer.  Functionally identical to MemeticOptimizer but
reports optimizer_mode='evolutionary_memetic' for separate tracking.
"""

from __future__ import annotations

from dataclasses import replace

from .hybrid_optimizer import HybridOptimizationResult
from .memetic_optimizer import MemeticOptimizer


class EvolutionaryMemeticOptimizer(MemeticOptimizer):
    """EA + Memetic optimizer using professor-priority fitness.

    Inherits all behaviour from MemeticOptimizer (population init,
    fairness-preserving crossover, fairness-aware mutation, repair,
    memetic local search on elites, diversity preservation).

    Configured from ea_* RunConfig fields so it can be tuned
    independently from the existing --memetic-* knobs, and stamps
    the result with optimizer_mode='evolutionary_memetic' so CSV/JSON
    reports distinguish the two variants.
    """

    def optimize(self) -> HybridOptimizationResult:
        result = super().optimize()
        return replace(result, optimizer_mode="evolutionary_memetic")
