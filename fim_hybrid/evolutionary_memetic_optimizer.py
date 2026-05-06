"""Evolutionary Memetic optimizer for Fair Influence Maximization.

Combines EA global search with memetic local refinement using
professor-priority fitness (F-score > MF > DCV > spread > runtime).
Configured via EA+Memetic runtime flags and reports optimizer_mode='ea_memetic'.
"""

from __future__ import annotations

from dataclasses import replace

from .optimizer_core import OptimizationResult
from .memetic_optimizer import MemeticOptimizer


class EvolutionaryMemeticOptimizer(MemeticOptimizer):
    """EA + Memetic optimizer using professor-priority fitness.

    Inherits all behaviour from MemeticOptimizer (population init,
    fairness-preserving crossover, fairness-aware mutation, repair,
    memetic local search on elites, diversity preservation).

    Configured from ea_* RunConfig fields so it can be tuned
    independently from the existing --memetic-* knobs, and stamps
    the result with optimizer_mode='ea_memetic' for CSV/JSON reporting.
    """

    def optimize(self) -> OptimizationResult:
        result = super().optimize()
        return replace(result, optimizer_mode="ea_memetic")
