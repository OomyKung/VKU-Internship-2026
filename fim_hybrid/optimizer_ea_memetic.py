"""Active EA+Memetic optimizer public entry point."""

from __future__ import annotations

from .evolutionary_memetic_optimizer import EvolutionaryMemeticOptimizer
from .memetic_optimizer import MemeticConfig, MemeticOptimizer
from .optimizer_core import CandidateEvaluation, EAOptimizerConfig, EAOptimizerBase, OptimizationResult

EAPlusMemeticOptimizer = EvolutionaryMemeticOptimizer

__all__ = [
    "CandidateEvaluation",
    "EAOptimizerBase",
    "EAOptimizerConfig",
    "EAPlusMemeticOptimizer",
    "EvolutionaryMemeticOptimizer",
    "MemeticConfig",
    "MemeticOptimizer",
    "OptimizationResult",
]
