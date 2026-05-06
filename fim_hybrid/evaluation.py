"""Shared seed-set evaluation utilities for fair FIM comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable

import numpy as np

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


def compute_f_score(
    mf: float,
    dcv: float,
    lambda_weight: float,
    *,
    dcv_shortfall: float | None = None,
    shortfall_dcv_weight: float = 1.0,
    disparity_dcv_weight: float = 0.25,
    fscore_mode: str = "disparity_primary",
) -> float:
    """Compute F(S).

    Default (fscore_mode='disparity_primary'): F(S) = lambda * MF - (1 - lambda) * DCV_disparity.
    Shortfall-primary mode: F(S) = MF - shortfall_dcv_weight * DCV_shortfall
                                      - disparity_dcv_weight * DCV_disparity.
    Backward-compatible: if dcv_shortfall is None the disparity formula is always used.
    """
    if not 0.0 <= lambda_weight <= 1.0:
        raise ValueError("lambda_weight must be between 0.0 and 1.0.")
    mode = str(fscore_mode or "disparity_primary").strip().lower()
    if dcv_shortfall is None or mode == "disparity_primary":
        # Original formula — backward compatible.
        return float(lambda_weight * mf - (1.0 - lambda_weight) * dcv)
    # shortfall_primary / combined: penalise shortfall strongly, disparity softly.
    return float(float(mf) - float(shortfall_dcv_weight) * float(dcv_shortfall) - float(disparity_dcv_weight) * float(dcv))


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
    ideal_influences: dict[str, float] | None = None,
) -> SeedSetEvaluation:
    """Run the shared diffusion, fairness, and F(S) pipeline for one seed set.

    When ideal_influences is provided, the returned FairnessMetrics will include
    dcv_shortfall, groups_below_target, groups_met_target, and target_coverage_ratio.
    The f_score field always uses the original lambda-weighted disparity formula so
    existing callers are unaffected.
    """
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
        ideal_influences=ideal_influences,
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


def compute_ideal_influences_proportional(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    budget: int,
    propagation_probability: float = 0.01,
    mc_runs: int = 20,
    random_seed: int = 42,
    diffusion_model: str = DEFAULT_DIFFUSION_MODEL,
) -> dict[str, float]:
    """Compute ideal influence per group via proportional-budget induced-subgraph IC.

    For each protected group g:
      k_g = ceil(budget * |g| / |V|)
      G_g = induced subgraph on g's nodes
      ideal_influence_g = expected IC spread inside G_g using top-degree k_g seeds

    Results are cacheable — call once per (dataset, attribute, budget) tuple.
    Falls back to min(|g|, k_g) when the group subgraph is trivially small.
    """
    rng = np.random.default_rng(int(random_seed))
    total_nodes = max(1, int(dataset.graph.number_of_nodes()))
    ideal_influences: dict[str, float] = {}

    for group_name, group_size in protected_group_report.group_sizes.items():
        k_g = max(1, math.ceil(int(budget) * int(group_size) / total_nodes))
        group_nodes = sorted(
            protected_group_report.protected_groups.get(group_name, set()),
            key=str,
        )

        if not group_nodes:
            ideal_influences[group_name] = float(min(group_size, k_g))
            continue

        subgraph = dataset.graph.subgraph(group_nodes)
        n_sg = subgraph.number_of_nodes()

        if n_sg == 0:
            ideal_influences[group_name] = float(k_g)
            continue

        if k_g >= n_sg:
            seeds = list(group_nodes)
        else:
            deg_fn = subgraph.out_degree if subgraph.is_directed() else subgraph.degree
            deg = dict(deg_fn())
            seeds = sorted(deg, key=lambda n: (-deg[n], str(n)))[:k_g]

        # Estimate IC spread within the induced subgraph.
        spreads: list[float] = []
        for _ in range(int(mc_runs)):
            reached = set(seeds)
            queue = list(seeds)
            while queue:
                node = queue.pop()
                for neighbor in subgraph.neighbors(node):
                    if neighbor not in reached and rng.random() < float(propagation_probability):
                        reached.add(neighbor)
                        queue.append(neighbor)
            spreads.append(float(len(reached)))

        ideal_influences[group_name] = float(np.mean(spreads))

    return ideal_influences


def compute_ideal_influences_feasible(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    budget: int,
    propagation_probability: float = 0.01,
    mc_runs: int = 20,
    random_seed: int = 42,
    diffusion_model: str = DEFAULT_DIFFUSION_MODEL,
    ceiling_factor: float = 0.95,
) -> dict[str, float]:
    """Two-pass feasibility-aware ideal influences.

    Pass 1 (proportional): same as compute_ideal_influences_proportional —
      k_g = ceil(budget * |g| / |V|), IC spread with top-degree k_g seeds.

    Pass 2 (ceiling): IC spread with min(budget, n_sg) seeds (full-budget upper bound).
      achievable_ceiling_g = IC(top-min(budget, n_sg) seeds in induced subgraph).

    Final: ideal_g = min(pass1_g, achievable_ceiling_g * ceiling_factor).

    This prevents setting targets that exceed what the group's subgraph can realistically
    absorb — e.g., sparse or disconnected groups where even the full budget stalls.
    """
    rng_pass1 = np.random.default_rng(int(random_seed))
    rng_pass2 = np.random.default_rng(int(random_seed) + 1)
    total_nodes = max(1, int(dataset.graph.number_of_nodes()))
    full_budget = max(1, int(budget))
    cf = float(ceiling_factor)
    ideal_influences: dict[str, float] = {}

    for group_name, group_size in protected_group_report.group_sizes.items():
        k_g = max(1, math.ceil(full_budget * int(group_size) / total_nodes))
        group_nodes = sorted(
            protected_group_report.protected_groups.get(group_name, set()),
            key=str,
        )

        if not group_nodes:
            ideal_influences[group_name] = float(min(group_size, k_g))
            continue

        subgraph = dataset.graph.subgraph(group_nodes)
        n_sg = subgraph.number_of_nodes()

        if n_sg == 0:
            ideal_influences[group_name] = float(k_g)
            continue

        deg_fn = subgraph.out_degree if subgraph.is_directed() else subgraph.degree
        deg = dict(deg_fn())
        sorted_nodes = sorted(deg, key=lambda n: (-deg[n], str(n)))

        def _ic_spread(seeds: list, rng: np.random.Generator) -> float:
            spreads: list[float] = []
            for _ in range(int(mc_runs)):
                reached = set(seeds)
                queue = list(seeds)
                while queue:
                    node = queue.pop()
                    for neighbor in subgraph.neighbors(node):
                        if neighbor not in reached and rng.random() < float(propagation_probability):
                            reached.add(neighbor)
                            queue.append(neighbor)
                spreads.append(float(len(reached)))
            return float(np.mean(spreads))

        # Pass 1: proportional budget seeds.
        seeds_p1 = sorted_nodes if k_g >= n_sg else sorted_nodes[:k_g]
        raw_ideal = _ic_spread(seeds_p1, rng_pass1)

        # Pass 2: full-budget seeds (achievable ceiling).
        k_ceil = min(full_budget, n_sg)
        seeds_p2 = sorted_nodes if k_ceil >= n_sg else sorted_nodes[:k_ceil]
        achievable_ceiling = _ic_spread(seeds_p2, rng_pass2)

        ideal_influences[group_name] = float(min(raw_ideal, achievable_ceiling * cf))

    return ideal_influences
