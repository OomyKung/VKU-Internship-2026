"""Experiment runner for fair comparison across FIM methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd

from .baselines import BaselineResult, run_baseline
from .community_detection import CommunityQualityMetrics, compute_community_quality_metrics, detect_communities
from .config import DatasetConfig
from .data_loader import LoadedDataset, ProtectedGroupReport, load_dataset, verify_protected_groups
from .hybrid_optimizer import HybridOptimizationResult, HybridSIEAConfig, HybridSIEAOptimizer


@dataclass(slots=True)
class ExperimentSettings:
    """Explicit experiment settings for method comparison."""

    protected_attribute: str
    budget: int
    community_method: str = "leiden"
    propagation_probability: float = 0.01
    mc_runs: int = 20
    lambda_weight: float = 0.5
    population_size: int = 12
    generations: int = 10
    crossover_probability: float = 0.7
    mutation_probability: float = 0.2
    elite_fraction: float = 0.25
    leader_guidance_fraction: float = 0.34
    local_search_steps: int = 2
    random_seed: int = 42
    output_dir: Path | None = None
    use_node2vec: bool = False


def _dataset_output_dir(output_dir: Path | None, dataset_name: str) -> Path | None:
    if output_dir is None:
        return None

    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def _history_path(
    output_dir: Path | None,
    dataset_name: str,
    budget: int,
    community_method: str,
    label: str,
) -> Path | None:
    dataset_dir = _dataset_output_dir(output_dir, dataset_name)
    if dataset_dir is None:
        return None

    path = dataset_dir / f"{dataset_name}_budget{budget}_{community_method}_{label}_history.csv"
    return path


def _community_columns(quality: CommunityQualityMetrics) -> dict[str, float | int]:
    return {
        "community_modularity": quality.modularity,
        "num_communities": quality.num_communities,
        "largest_community_size": quality.largest_community_size,
        "smallest_community_size": quality.smallest_community_size,
        "average_community_size": quality.average_community_size,
        "community_size_std": quality.community_size_std,
    }


def _baseline_row(
    dataset: LoadedDataset,
    community_method: str,
    result: BaselineResult,
    quality: CommunityQualityMetrics,
) -> dict[str, object]:
    return {
        "dataset": dataset.name,
        "community_method": community_method,
        "method": result.method,
        "variant_type": "baseline",
        "seed_set": json.dumps(list(result.seed_set)),
        "total_spread": result.total_spread_mean,
        "mf": result.mf,
        "dcv": result.dcv,
        "f_score": result.f_score,
        "runtime_seconds": result.runtime_seconds,
        "candidate_pool_size": dataset.graph.number_of_nodes(),
        "swarm_guidance": False,
        "crossover": False,
        "local_search": False,
        "community_aware_mutation": result.method == "community_round_robin",
        "node2vec_enabled": False,
        "note": "",
        **_community_columns(quality),
    }


def _hybrid_row(
    dataset: LoadedDataset,
    community_method: str,
    label: str,
    variant_type: str,
    result: HybridOptimizationResult,
    config: HybridSIEAConfig,
    quality: CommunityQualityMetrics,
    note: str = "",
) -> dict[str, object]:
    return {
        "dataset": dataset.name,
        "community_method": community_method,
        "method": label,
        "variant_type": variant_type,
        "seed_set": json.dumps(list(result.best_seed_set)),
        "total_spread": result.best_spread,
        "mf": result.best_fairness.mf,
        "dcv": result.best_fairness.dcv,
        "f_score": result.best_score,
        "runtime_seconds": result.runtime_seconds,
        "candidate_pool_size": result.candidate_pool_size,
        "swarm_guidance": not config.disable_swarm_guidance,
        "crossover": not config.disable_crossover,
        "local_search": not config.disable_local_search,
        "community_aware_mutation": not config.disable_community_aware_mutation,
        "node2vec_enabled": False,
        "note": note,
        **_community_columns(quality),
    }


def _build_optimizer_config(settings: ExperimentSettings, **overrides: object) -> HybridSIEAConfig:
    config = HybridSIEAConfig(
        budget=settings.budget,
        population_size=settings.population_size,
        generations=settings.generations,
        crossover_probability=settings.crossover_probability,
        mutation_probability=settings.mutation_probability,
        elite_fraction=settings.elite_fraction,
        leader_guidance_fraction=settings.leader_guidance_fraction,
        propagation_probability=settings.propagation_probability,
        mc_runs=settings.mc_runs,
        lambda_weight=settings.lambda_weight,
        random_seed=settings.random_seed,
        local_search_steps=settings.local_search_steps,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def run_loaded_experiment(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    settings: ExperimentSettings,
    community_methods: list[str] | None = None,
    baseline_methods: list[str] | None = None,
    include_ablations: bool = True,
) -> pd.DataFrame:
    """Run one or more method comparisons on a preloaded dataset."""

    if settings.use_node2vec:
        raise NotImplementedError("Optional Node2Vec guidance is not implemented in this rebuilt prototype.")

    methods = community_methods or [settings.community_method]
    baselines = baseline_methods or ["degree", "pagerank", "community_round_robin", "random"]
    results: list[dict[str, object]] = []

    for community_method in methods:
        community_result = detect_communities(dataset.graph, method=community_method, seed=settings.random_seed)
        quality = compute_community_quality_metrics(dataset.graph, community_result)

        for baseline_name in baselines:
            baseline_result = run_baseline(
                dataset=dataset,
                protected_group_report=protected_group_report,
                method=baseline_name,
                budget=settings.budget,
                propagation_probability=settings.propagation_probability,
                mc_runs=settings.mc_runs,
                lambda_weight=settings.lambda_weight,
                community_result=community_result,
                random_seed=settings.random_seed,
            )
            results.append(_baseline_row(dataset, community_method, baseline_result, quality))

        hybrid_variants: list[tuple[str, str, str, dict[str, object]]] = [
            (
                "cea_fim",
                "comparator",
                "CEA-style baseline: community EA without swarm guidance or local search.",
                {"disable_swarm_guidance": True, "disable_local_search": True},
            ),
            ("hybrid_siea", "proposed", "", {}),
        ]
        if include_ablations:
            hybrid_variants.extend(
                [
                    ("hybrid_no_swarm", "ablation", "Ablation: swarm guidance disabled.", {"disable_swarm_guidance": True}),
                    ("hybrid_no_crossover", "ablation", "Ablation: crossover disabled.", {"disable_crossover": True}),
                    ("hybrid_no_local_search", "ablation", "Ablation: local search disabled.", {"disable_local_search": True}),
                    (
                        "hybrid_no_community_mutation",
                        "ablation",
                        "Ablation: community-aware mutation bonus disabled.",
                        {"disable_community_aware_mutation": True},
                    ),
                ]
            )

        for label, variant_type, note, overrides in hybrid_variants:
            config = _build_optimizer_config(settings, **overrides)
            optimizer = HybridSIEAOptimizer(
                dataset=dataset,
                protected_group_report=protected_group_report,
                community_result=community_result,
                config=config,
            )
            result = optimizer.optimize()
            history_path = _history_path(settings.output_dir, dataset.name, settings.budget, community_method, label)
            if history_path is not None:
                result.history.to_csv(history_path, index=False)
                note = "; ".join(part for part in [note, f"history={history_path.name}"] if part)
            results.append(
                _hybrid_row(
                    dataset=dataset,
                    community_method=community_method,
                    label=label,
                    variant_type=variant_type,
                    result=result,
                    config=config,
                    quality=quality,
                    note=note,
                )
            )

    result_frame = pd.DataFrame(results)
    dataset_dir = _dataset_output_dir(settings.output_dir, dataset.name)
    if dataset_dir is not None:
        output_path = dataset_dir / f"{dataset.name}_budget{settings.budget}_results.csv"
        result_frame.to_csv(output_path, index=False)
    return result_frame


def run_experiment(
    dataset_config: DatasetConfig,
    settings: ExperimentSettings,
    community_methods: list[str] | None = None,
    baseline_methods: list[str] | None = None,
    include_ablations: bool = True,
) -> pd.DataFrame:
    """Load a dataset and run the comparison experiment."""

    dataset = load_dataset(dataset_config)
    protected_group_report = verify_protected_groups(dataset, settings.protected_attribute)
    return run_loaded_experiment(
        dataset=dataset,
        protected_group_report=protected_group_report,
        settings=settings,
        community_methods=community_methods,
        baseline_methods=baseline_methods,
        include_ablations=include_ablations,
    )
