"""Shared seed-set evaluation utilities for fair FIM comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable

from .data_loader import LoadedDataset, ProtectedGroupReport
from .diffusion import DEFAULT_DIFFUSION_MODEL, simulate_diffusion, validate_diffusion_model
from .fairness import FairnessMetrics, evaluate_fairness


@dataclass(slots=True)
class SeedSetEvaluation:
    """Unified spread, fairness, and F(S) evaluation for one seed set."""

    seed_set: tuple[Any, ...]
    total_spread_mean: float
    total_spread_std: float
    fairness: FairnessMetrics
    f_score: float
    runtime_seconds: float


def compute_f_score(mf: float, dcv: float, lambda_weight: float) -> float:
    """Compute F(S) = lambda * MF - (1 - lambda) * DCV."""

    if not 0.0 <= lambda_weight <= 1.0:
        raise ValueError("lambda_weight must be between 0.0 and 1.0.")
    # Higher MF is better, while higher DCV is worse.
    # `lambda_weight` controls the tradeoff:
    # - near 1.0: prioritize fairness floor (MF)
    # - near 0.0: prioritize reducing disparity cost (DCV)
    return float(lambda_weight * mf - (1.0 - lambda_weight) * dcv)


def evaluate_seed_set(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    seed_set: Iterable[Any],
    propagation_probability: float = 0.01,
    mc_runs: int = 100,
    random_seed: int = 42,
    lambda_weight: float = 0.5,
    include_soft_mf: bool = True,
    diffusion_model: str = DEFAULT_DIFFUSION_MODEL,
) -> SeedSetEvaluation:
    """Run the shared diffusion, fairness, and F(S) pipeline for one seed set."""

    start = perf_counter()
    validate_diffusion_model(diffusion_model)
    diffusion_result = simulate_diffusion(
        dataset=dataset,
        protected_group_report=protected_group_report,
        seed_set=seed_set,
        propagation_probability=propagation_probability,
        mc_runs=mc_runs,
        random_seed=random_seed,
        diffusion_model=diffusion_model,
    )
    fairness = evaluate_fairness(
        group_spread=diffusion_result.group_spread_mean,
        group_sizes=protected_group_report.group_sizes,
        total_spread=diffusion_result.total_spread_mean,
        include_soft_mf=include_soft_mf,
    )
    runtime_seconds = perf_counter() - start

    return SeedSetEvaluation(
        seed_set=diffusion_result.seed_set,
        total_spread_mean=diffusion_result.total_spread_mean,
        total_spread_std=diffusion_result.total_spread_std,
        fairness=fairness,
        f_score=compute_f_score(fairness.mf, fairness.dcv, lambda_weight),
        runtime_seconds=runtime_seconds,
    )
