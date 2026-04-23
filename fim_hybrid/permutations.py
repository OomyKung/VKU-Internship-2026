"""Named end-to-end FIM algorithm-stack permutations.

This module intentionally sits above the existing experiment runner.  It maps
research-facing permutation names onto the current diffusion, community,
baseline, RIS/GNN, and hybrid optimizer pieces without changing the baseline
experiment path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

import networkx as nx
import pandas as pd

from .baselines import select_baseline_seed_set
from .community_detection import CommunityDetectionResult, compute_community_quality_metrics, detect_communities
from .config import DatasetConfig
from .data_loader import LoadedDataset, ProtectedGroupReport, load_dataset, verify_protected_groups
from .diffusion import DEFAULT_DIFFUSION_MODEL, validate_diffusion_model
from .evaluation import SeedSetEvaluation, evaluate_seed_set
from .experiment_runner import ExperimentSettings, run_loaded_experiment

_FINAL_EVAL_RANDOM_SEED_OFFSET = 1_000_000
_GNN_RIS_METHOD = "hybrid_siea_ml_gnn_ris_two_tier_tuned_swap_local_search"


@dataclass(frozen=True, slots=True)
class FIMPermutationSpec:
    """Static metadata for a named FIM system design."""

    name: str
    description: str
    diffusion_model: str
    community_method: str
    initialization: str
    ranking_mode: str
    optimizer_mode: str
    refinement: str
    spread_estimator_search: str
    spread_estimator_final: str
    embedding_method: str = "none"
    clustering_method: str = "none"
    bias_control: str = "none"
    fairness_objective: str = "f_score"
    notes: str = ""


@dataclass(frozen=True, slots=True)
class FIMPermutationRunConfig:
    """Shared runtime controls for a permutation benchmark."""

    protected_attribute: str
    budget: int
    propagation_probability: float = 0.01
    mc_runs_search: int = 20
    mc_runs_eval: int = 100
    lambda_weight: float = 0.5
    random_seed: int = 42
    output_dir: Path | None = None
    continue_on_error: bool = True
    swap_candidate_pool_size: int = 24
    local_search_steps: int = 1
    permutation3_diffusion_model: str = DEFAULT_DIFFUSION_MODEL
    population_size: int = 8
    generations: int = 5
    gnn_epochs: int = 30
    gnn_hidden_dim: int = 32
    ris_num_rr_sets: int = 128


@dataclass(slots=True)
class FIMPermutationBenchmarkResult:
    """Benchmark output plus artifact paths."""

    summary_frame: pd.DataFrame
    results_by_permutation: dict[str, pd.DataFrame]
    comparison_csv_path: Path | None = None
    report_path: Path | None = None


def _permutation_registry() -> dict[str, FIMPermutationSpec]:
    return {
        "community_aware_fair_greedy": FIMPermutationSpec(
            name="community_aware_fair_greedy",
            description="Leiden communities, community round-robin initialization, fairness-weighted greedy ranking, and bounded swap local search.",
            diffusion_model="ic",
            community_method="leiden",
            initialization="community_round_robin",
            ranking_mode="fairness_weighted_greedy",
            optimizer_mode="local_search",
            refinement="swap_local_search",
            spread_estimator_search="monte_carlo",
            spread_estimator_final="monte_carlo",
            fairness_objective="f_score",
        ),
        "graphsage_fair_ris_hybrid": FIMPermutationSpec(
            name="graphsage_fair_ris_hybrid",
            description="GraphSAGE node scoring with fairness-aware RIS guidance and the existing hybrid SI+EA optimizer.",
            diffusion_model="ic",
            community_method="leiden",
            initialization="hybrid_population",
            ranking_mode="graphsage_plus_fair_ris",
            optimizer_mode="hybrid_si_ea",
            refinement="swap_local_search+repair",
            spread_estimator_search="fairness_aware_ris",
            spread_estimator_final="monte_carlo",
            embedding_method="graphsage",
            bias_control="worst_group_aware",
            fairness_objective="f_score",
        ),
        "infomap_graphcl_maximin": FIMPermutationSpec(
            name="infomap_graphcl_maximin",
            description="Infomap communities with maximin/worst-group seed selection and bounded local search; GraphCL/spectral are recorded as exploratory optional modules.",
            diffusion_model="ic",
            community_method="infomap",
            initialization="maximin_greedy",
            ranking_mode="maximin_greedy",
            optimizer_mode="local_search",
            refinement="swap_local_search",
            spread_estimator_search="monte_carlo",
            spread_estimator_final="monte_carlo",
            embedding_method="graphcl",
            clustering_method="spectral",
            bias_control="mild_adversarial_optional",
            fairness_objective="maximin",
            notes="GraphCL embedding and spectral clustering are optional exploratory modules; the runnable FIM path uses Infomap plus maximin/worst-group search.",
        ),
    }


def available_fim_permutations() -> tuple[str, ...]:
    """Return supported permutation names in deterministic order."""

    return tuple(_permutation_registry())


def get_fim_permutation_spec(name: str) -> FIMPermutationSpec:
    """Return metadata for one permutation."""

    key = str(name).strip().lower()
    registry = _permutation_registry()
    if key not in registry:
        supported = ", ".join(sorted(registry))
        raise ValueError(f"Unknown FIM permutation '{name}'. Supported permutations: {supported}.")
    return registry[key]


def fim_permutation_specs() -> tuple[FIMPermutationSpec, ...]:
    """Return all permutation specs."""

    return tuple(_permutation_registry().values())


def permutation_summary_columns() -> list[str]:
    """Return the normalized comparison columns written by this benchmark."""

    return [
        "permutation_name",
        "status",
        "dataset",
        "protected_attribute",
        "diffusion_model",
        "community_method",
        "embedding_method",
        "clustering_method",
        "optimizer_mode",
        "ranking_mode",
        "initialization",
        "refinement",
        "bias_control",
        "spread_estimator_search",
        "spread_estimator_final",
        "search_spread_estimator",
        "search_guidance_estimator",
        "final_spread_estimator",
        "method",
        "variant_type",
        "seed_set",
        "total_spread",
        "extra_spread",
        "mf",
        "dcv",
        "f_score",
        "runtime_seconds",
        "search_runtime_seconds",
        "final_eval_runtime_seconds",
        "mc_runs_search",
        "mc_runs_eval",
        "key_enabled_modules",
        "notes",
        "skipped_reason",
    ]


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _normalized_seed_set(seed_set: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(sorted(tuple(seed_set), key=_sort_key))


def _seed_tuple_key(seed_set: Iterable[Any]) -> tuple[tuple[str, str], ...]:
    return tuple(_sort_key(node) for node in _normalized_seed_set(seed_set))


def _resolved_eval_random_seed(config: FIMPermutationRunConfig) -> int:
    return int(config.random_seed) + _FINAL_EVAL_RANDOM_SEED_OFFSET


def _community_result(dataset: LoadedDataset, spec: FIMPermutationSpec, config: FIMPermutationRunConfig) -> CommunityDetectionResult:
    return detect_communities(
        dataset.graph,
        method=spec.community_method,
        seed=int(config.random_seed),
        input_mode="graph",
    )


def _final_evaluate(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    seed_set: Iterable[Any],
    diffusion_model: str,
    config: FIMPermutationRunConfig,
) -> SeedSetEvaluation:
    return evaluate_seed_set(
        dataset=dataset,
        protected_group_report=protected_group_report,
        seed_set=_normalized_seed_set(seed_set),
        propagation_probability=float(config.propagation_probability),
        mc_runs=int(config.mc_runs_eval),
        random_seed=_resolved_eval_random_seed(config),
        lambda_weight=float(config.lambda_weight),
        include_soft_mf=True,
        diffusion_model=diffusion_model,
    )


def _search_evaluate(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    seed_set: Iterable[Any],
    diffusion_model: str,
    config: FIMPermutationRunConfig,
    seed_offset: int,
) -> SeedSetEvaluation:
    return evaluate_seed_set(
        dataset=dataset,
        protected_group_report=protected_group_report,
        seed_set=_normalized_seed_set(seed_set),
        propagation_probability=float(config.propagation_probability),
        mc_runs=int(config.mc_runs_search),
        random_seed=int(config.random_seed) + int(seed_offset),
        lambda_weight=float(config.lambda_weight),
        include_soft_mf=True,
        diffusion_model=diffusion_model,
    )


def _spread_dict(evaluation: SeedSetEvaluation, budget: int) -> dict[str, object]:
    total_spread = float(evaluation.total_spread_mean)
    return {
        "seed_set": json.dumps(list(evaluation.seed_set)),
        "total_spread": total_spread,
        "extra_spread": float(total_spread - float(budget)),
        "mf": float(evaluation.fairness.mf),
        "dcv": float(evaluation.fairness.dcv),
        "f_score": float(evaluation.f_score),
        "final_eval_runtime_seconds": float(evaluation.runtime_seconds),
    }


def _candidate_pool(graph: nx.Graph, seed_set: tuple[Any, ...], max_size: int) -> list[Any]:
    excluded = set(seed_set)
    degree_view = graph.out_degree() if graph.is_directed() else graph.degree()
    degree_scores = {node: float(score) for node, score in degree_view if node not in excluded}
    candidates = sorted(degree_scores, key=lambda node: (-degree_scores[node], _sort_key(node)))
    if max_size > 0:
        candidates = candidates[: int(max_size)]
    return candidates


def _rank_search_evaluation(mode: str, evaluation: SeedSetEvaluation) -> tuple[float, float, float, float]:
    if mode == "maximin":
        return (
            float(evaluation.fairness.mf),
            float(evaluation.f_score),
            float(evaluation.total_spread_mean),
            -float(evaluation.fairness.dcv),
        )
    return (
        float(evaluation.f_score),
        float(evaluation.fairness.mf),
        float(evaluation.total_spread_mean),
        -float(evaluation.fairness.dcv),
    )


def _swap_local_search(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    initial_seed_set: tuple[Any, ...],
    diffusion_model: str,
    config: FIMPermutationRunConfig,
    objective_mode: str,
) -> tuple[tuple[Any, ...], SeedSetEvaluation]:
    current = _normalized_seed_set(initial_seed_set)
    current_eval = _search_evaluate(
        dataset,
        protected_group_report,
        current,
        diffusion_model,
        config,
        seed_offset=101,
    )
    cache: dict[tuple[Any, ...], SeedSetEvaluation] = {current: current_eval}

    for step_index in range(max(0, int(config.local_search_steps))):
        best_seed_set = current
        best_eval = current_eval
        best_key = _rank_search_evaluation(objective_mode, current_eval)
        candidates = _candidate_pool(dataset.graph, current, int(config.swap_candidate_pool_size))
        for removed_node in current:
            retained = [node for node in current if node != removed_node]
            for candidate_node in candidates:
                if candidate_node in retained:
                    continue
                trial = _normalized_seed_set([*retained, candidate_node])
                evaluation = cache.get(trial)
                if evaluation is None:
                    evaluation = _search_evaluate(
                        dataset,
                        protected_group_report,
                        trial,
                        diffusion_model,
                        config,
                        seed_offset=211 + step_index,
                    )
                    cache[trial] = evaluation
                rank_key = _rank_search_evaluation(objective_mode, evaluation)
                if rank_key > best_key or (
                    rank_key == best_key
                    and _seed_tuple_key(trial) < _seed_tuple_key(best_seed_set)
                ):
                    best_seed_set = trial
                    best_eval = evaluation
                    best_key = rank_key
        if best_seed_set == current:
            break
        current = best_seed_set
        current_eval = best_eval
    return current, current_eval


def _key_enabled_modules(spec: FIMPermutationSpec, diffusion_model: str) -> str:
    parts = [
        f"diffusion={diffusion_model}",
        f"community={spec.community_method}",
        f"ranking={spec.ranking_mode}",
        f"optimizer={spec.optimizer_mode}",
        f"refinement={spec.refinement}",
        f"search_estimator={spec.spread_estimator_search}",
        "final_estimator=monte_carlo",
    ]
    if spec.embedding_method != "none":
        parts.append(f"embedding={spec.embedding_method}")
    if spec.clustering_method != "none":
        parts.append(f"clustering={spec.clustering_method}")
    if spec.bias_control != "none":
        parts.append(f"bias_control={spec.bias_control}")
    return "; ".join(parts)


def _result_row(
    *,
    spec: FIMPermutationSpec,
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    config: FIMPermutationRunConfig,
    diffusion_model: str,
    method: str,
    variant_type: str,
    evaluation: SeedSetEvaluation,
    search_runtime_seconds: float,
    status: str = "ok",
    notes: str = "",
    skipped_reason: str = "",
) -> dict[str, object]:
    row = {
        "permutation_name": spec.name,
        "status": status,
        "dataset": dataset.name,
        "protected_attribute": protected_group_report.protected_attribute,
        "diffusion_model": diffusion_model,
        "community_method": spec.community_method,
        "embedding_method": spec.embedding_method,
        "clustering_method": spec.clustering_method,
        "optimizer_mode": spec.optimizer_mode,
        "ranking_mode": spec.ranking_mode,
        "initialization": spec.initialization,
        "refinement": spec.refinement,
        "bias_control": spec.bias_control,
        "spread_estimator_search": spec.spread_estimator_search,
        "spread_estimator_final": spec.spread_estimator_final,
        "search_spread_estimator": (
            "ris_guidance"
            if spec.spread_estimator_search == "fairness_aware_ris"
            else spec.spread_estimator_search
        ),
        "search_guidance_estimator": (
            "ris_guidance"
            if spec.spread_estimator_search == "fairness_aware_ris"
            else "none"
        ),
        "final_spread_estimator": spec.spread_estimator_final,
        "method": method,
        "variant_type": variant_type,
        "runtime_seconds": float(search_runtime_seconds + evaluation.runtime_seconds),
        "search_runtime_seconds": float(search_runtime_seconds),
        "mc_runs_search": int(config.mc_runs_search),
        "mc_runs_eval": int(config.mc_runs_eval),
        "key_enabled_modules": _key_enabled_modules(spec, diffusion_model),
        "notes": "; ".join(part for part in [spec.notes, notes] if part),
        "skipped_reason": skipped_reason,
    }
    row.update(_spread_dict(evaluation, int(config.budget)))
    return row


def _skipped_row(
    *,
    spec: FIMPermutationSpec,
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    config: FIMPermutationRunConfig,
    diffusion_model: str,
    skipped_reason: str,
) -> dict[str, object]:
    row = {
        column: pd.NA
        for column in permutation_summary_columns()
    }
    row.update(
        {
            "permutation_name": spec.name,
            "status": "skipped",
            "dataset": dataset.name,
            "protected_attribute": protected_group_report.protected_attribute,
            "diffusion_model": diffusion_model,
            "community_method": spec.community_method,
            "embedding_method": spec.embedding_method,
            "clustering_method": spec.clustering_method,
            "optimizer_mode": spec.optimizer_mode,
            "ranking_mode": spec.ranking_mode,
            "initialization": spec.initialization,
            "refinement": spec.refinement,
            "bias_control": spec.bias_control,
            "spread_estimator_search": spec.spread_estimator_search,
            "spread_estimator_final": spec.spread_estimator_final,
            "search_spread_estimator": (
                "ris_guidance"
                if spec.spread_estimator_search == "fairness_aware_ris"
                else spec.spread_estimator_search
            ),
            "search_guidance_estimator": (
                "ris_guidance"
                if spec.spread_estimator_search == "fairness_aware_ris"
                else "none"
            ),
            "final_spread_estimator": spec.spread_estimator_final,
            "mc_runs_search": int(config.mc_runs_search),
            "mc_runs_eval": int(config.mc_runs_eval),
            "key_enabled_modules": _key_enabled_modules(spec, diffusion_model),
            "notes": spec.notes,
            "skipped_reason": skipped_reason,
        }
    )
    return row


def _run_community_aware_fair_greedy(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
) -> pd.DataFrame:
    start = perf_counter()
    diffusion_model = spec.diffusion_model
    community_result = _community_result(dataset, spec, config)
    initial_seed_set = select_baseline_seed_set(
        dataset=dataset,
        method="community_round_robin",
        budget=int(config.budget),
        protected_group_report=protected_group_report,
        propagation_probability=float(config.propagation_probability),
        mc_runs=int(config.mc_runs_search),
        lambda_weight=float(config.lambda_weight),
        community_result=community_result,
        random_seed=int(config.random_seed),
        diffusion_model=diffusion_model,
    )
    greedy_seed_set = select_baseline_seed_set(
        dataset=dataset,
        method="fairness_weighted_greedy",
        budget=int(config.budget),
        protected_group_report=protected_group_report,
        propagation_probability=float(config.propagation_probability),
        mc_runs=int(config.mc_runs_search),
        lambda_weight=float(config.lambda_weight),
        community_result=community_result,
        random_seed=int(config.random_seed),
        diffusion_model=diffusion_model,
    )
    initial_eval = _search_evaluate(
        dataset,
        protected_group_report,
        initial_seed_set,
        diffusion_model,
        config,
        seed_offset=17,
    )
    greedy_eval = _search_evaluate(
        dataset,
        protected_group_report,
        greedy_seed_set,
        diffusion_model,
        config,
        seed_offset=19,
    )
    start_seed_set = (
        greedy_seed_set
        if _rank_search_evaluation("f_score", greedy_eval) >= _rank_search_evaluation("f_score", initial_eval)
        else initial_seed_set
    )
    refined_seed_set, _ = _swap_local_search(
        dataset=dataset,
        protected_group_report=protected_group_report,
        initial_seed_set=start_seed_set,
        diffusion_model=diffusion_model,
        config=config,
        objective_mode="f_score",
    )
    search_runtime = perf_counter() - start
    final_eval = _final_evaluate(dataset, protected_group_report, refined_seed_set, diffusion_model, config)
    quality = compute_community_quality_metrics(dataset.graph, community_result)
    notes = (
        f"initial_rr={list(initial_seed_set)}; greedy={list(greedy_seed_set)}; "
        f"communities={quality.num_communities}; modularity={quality.modularity:.6f}"
    )
    return pd.DataFrame(
        [
            _result_row(
                spec=spec,
                dataset=dataset,
                protected_group_report=protected_group_report,
                config=config,
                diffusion_model=diffusion_model,
                method="community_round_robin+fairness_weighted_greedy+swap_local_search",
                variant_type="interpretable_baseline",
                evaluation=final_eval,
                search_runtime_seconds=search_runtime,
                notes=notes,
            )
        ]
    )


def _run_graphsage_fair_ris_hybrid(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
) -> pd.DataFrame:
    settings = ExperimentSettings(
        protected_attribute=config.protected_attribute,
        budget=int(config.budget),
        diffusion_model=spec.diffusion_model,
        spread_estimator="ris_guidance",
        community_method=spec.community_method,
        propagation_probability=float(config.propagation_probability),
        mc_runs_search=int(config.mc_runs_search),
        mc_runs_eval=int(config.mc_runs_eval),
        lambda_weight=float(config.lambda_weight),
        population_size=int(config.population_size),
        generations=int(config.generations),
        local_search_steps=max(1, int(config.local_search_steps)),
        random_seed=int(config.random_seed),
        output_dir=config.output_dir,
        use_ml=True,
        ml_backend="gnn_ris",
        ml_guidance_mode="two_tier",
        debias_mode="worst_group_boost",
        gnn_model_type="graphsage",
        gnn_hidden_dim=int(config.gnn_hidden_dim),
        gnn_epochs=int(config.gnn_epochs),
        ris_num_rr_sets=int(config.ris_num_rr_sets),
        ris_random_seed=int(config.random_seed),
        ris_mode="weak_group_weighted",
        ris_reuse_rr_sets=True,
        gnn_weight=1.0,
        ris_weight=1.0,
        fair_ris_weight=0.50,
        fairness_urgency_weight=0.20,
        repair_fairness_weight=0.40,
        repair_bridge_weight=0.20,
        weakest_group_k=3,
        weakest_group_mutation_weight=0.35,
        zero_group_bonus_weight=0.25,
        bridge_to_weak_group_weight=0.15,
        local_search_focus_mode="worst_group",
        local_search_bottom_k_groups=3,
        local_search_swap_trials=max(1, int(config.local_search_steps)),
        local_search_candidate_pool_size=max(4, int(config.swap_candidate_pool_size)),
        swap_candidate_pool_size=max(4, int(config.swap_candidate_pool_size)),
    )
    frame = run_loaded_experiment(
        dataset=dataset,
        protected_group_report=protected_group_report,
        settings=settings,
        community_methods=[spec.community_method],
        baseline_methods=[],
        selected_methods=[_GNN_RIS_METHOD],
        include_ablations=False,
    )
    if frame.empty:
        raise RuntimeError(f"{spec.name} produced no result rows.")
    row = frame.iloc[0].to_dict()
    row.update(
        {
            "permutation_name": spec.name,
            "status": "ok",
            "embedding_method": spec.embedding_method,
            "clustering_method": spec.clustering_method,
            "optimizer_mode": spec.optimizer_mode,
            "ranking_mode": spec.ranking_mode,
            "initialization": spec.initialization,
            "refinement": spec.refinement,
            "bias_control": spec.bias_control,
            "spread_estimator_search": spec.spread_estimator_search,
            "spread_estimator_final": spec.spread_estimator_final,
            "search_spread_estimator": "ris_guidance",
            "search_guidance_estimator": "ris_guidance",
            "final_spread_estimator": spec.spread_estimator_final,
            "key_enabled_modules": _key_enabled_modules(spec, spec.diffusion_model),
            "notes": "; ".join(
                part
                for part in [
                    spec.notes,
                    str(row.get("note", "")),
                    "GraphSAGE ranking and fair RIS guidance delegated to existing gnn_ris hybrid path.",
                ]
                if part
            ),
            "skipped_reason": "",
        }
    )
    return pd.DataFrame([row])


def _run_infomap_graphcl_maximin(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    spec: FIMPermutationSpec,
    config: FIMPermutationRunConfig,
) -> pd.DataFrame:
    start = perf_counter()
    diffusion_model = validate_diffusion_model(config.permutation3_diffusion_model or spec.diffusion_model)
    community_result = _community_result(dataset, spec, config)
    initial_seed_set = select_baseline_seed_set(
        dataset=dataset,
        method="maximin_greedy",
        budget=int(config.budget),
        protected_group_report=protected_group_report,
        propagation_probability=float(config.propagation_probability),
        mc_runs=int(config.mc_runs_search),
        lambda_weight=float(config.lambda_weight),
        community_result=community_result,
        random_seed=int(config.random_seed),
        diffusion_model=diffusion_model,
    )
    refined_seed_set, _ = _swap_local_search(
        dataset=dataset,
        protected_group_report=protected_group_report,
        initial_seed_set=initial_seed_set,
        diffusion_model=diffusion_model,
        config=config,
        objective_mode="maximin",
    )
    search_runtime = perf_counter() - start
    final_eval = _final_evaluate(dataset, protected_group_report, refined_seed_set, diffusion_model, config)
    quality = compute_community_quality_metrics(dataset.graph, community_result)
    notes = (
        f"initial_maximin={list(initial_seed_set)}; communities={quality.num_communities}; "
        f"modularity={quality.modularity:.6f}; optional_graphcl_clustering=not_materialized_in_fim_path"
    )
    return pd.DataFrame(
        [
            _result_row(
                spec=replace(spec, diffusion_model=diffusion_model),
                dataset=dataset,
                protected_group_report=protected_group_report,
                config=config,
                diffusion_model=diffusion_model,
                method="infomap+maximin_greedy+swap_local_search",
                variant_type="exploratory_fairness",
                evaluation=final_eval,
                search_runtime_seconds=search_runtime,
                notes=notes,
            )
        ]
    )


_RUNNERS: Mapping[str, Callable[[LoadedDataset, ProtectedGroupReport, FIMPermutationSpec, FIMPermutationRunConfig], pd.DataFrame]] = {
    "community_aware_fair_greedy": _run_community_aware_fair_greedy,
    "graphsage_fair_ris_hybrid": _run_graphsage_fair_ris_hybrid,
    "infomap_graphcl_maximin": _run_infomap_graphcl_maximin,
}


def _normalize_summary_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in permutation_summary_columns():
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, permutation_summary_columns()].copy()


def _permutation_output_dir(output_dir: Path | None, dataset_name: str, protected_attribute: str) -> Path | None:
    if output_dir is None:
        return None
    safe_attribute = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in protected_attribute.strip()
    ).strip("._-") or "protected_attribute"
    path = Path(output_dir) / dataset_name / safe_attribute
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_fim_permutation_report(frame: pd.DataFrame, config: FIMPermutationRunConfig) -> str:
    """Format a compact terminal/report comparison table."""

    lines = [
        "FIM Algorithm-Stack Permutation Comparison",
        "-" * 72,
        f"Protected attribute: {config.protected_attribute}",
        f"Budget: {config.budget}",
        f"Final estimator: monte_carlo | mc_runs_eval={config.mc_runs_eval}",
        "",
    ]
    if frame.empty:
        lines.append("No permutation rows were produced.")
        return "\n".join(lines)

    sort_frame = frame.copy()
    for column in ["f_score", "mf", "total_spread", "runtime_seconds"]:
        sort_frame[column] = pd.to_numeric(sort_frame[column], errors="coerce")
    ok_frame = sort_frame[sort_frame["status"] == "ok"].sort_values(
        ["f_score", "mf", "total_spread", "runtime_seconds"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    skipped_frame = sort_frame[sort_frame["status"] != "ok"]
    ordered = pd.concat([ok_frame, skipped_frame], ignore_index=True)
    for index, row in ordered.iterrows():
        status = str(row.get("status", "unknown"))
        lines.append(f"{index + 1}. {row['permutation_name']} [{status}]")
        if status == "ok":
            lines.append(
                "   "
                f"spread={float(row['total_spread']):.4f} | "
                f"extra={float(row['extra_spread']):.4f} | "
                f"MF={float(row['mf']):.4f} | "
                f"DCV={float(row['dcv']):.4f} | "
                f"F-score={float(row['f_score']):.4f} | "
                f"runtime={float(row['runtime_seconds']):.3f}s"
            )
        else:
            lines.append(f"   skipped_reason={row.get('skipped_reason', '')}")
        lines.append(
            "   "
            f"modules: diffusion={row.get('diffusion_model')} | "
            f"community={row.get('community_method')} | "
            f"embedding={row.get('embedding_method')} | "
            f"clustering={row.get('clustering_method')} | "
            f"optimizer={row.get('optimizer_mode')} | "
            f"ranking={row.get('ranking_mode')}"
        )
        notes = str(row.get("notes", "")).strip()
        if notes and notes != "<NA>":
            lines.append(f"   notes={notes}")
        lines.append("")
    return "\n".join(lines).rstrip()


def run_fim_permutation_benchmark(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    config: FIMPermutationRunConfig,
    permutations: Iterable[str] | None = None,
) -> FIMPermutationBenchmarkResult:
    """Run named FIM permutations with shared final Monte Carlo evaluation."""

    requested = tuple(permutations or available_fim_permutations())
    rows: list[dict[str, object]] = []
    results_by_permutation: dict[str, pd.DataFrame] = {}

    for permutation_name in requested:
        spec = get_fim_permutation_spec(permutation_name)
        runner = _RUNNERS[spec.name]
        diffusion_model = (
            config.permutation3_diffusion_model
            if spec.name == "infomap_graphcl_maximin"
            else spec.diffusion_model
        )
        try:
            frame = runner(dataset, protected_group_report, spec, config)
        except Exception as exc:  # noqa: BLE001 - benchmark rows should skip cleanly.
            if not config.continue_on_error:
                raise
            frame = pd.DataFrame(
                [
                    _skipped_row(
                        spec=spec,
                        dataset=dataset,
                        protected_group_report=protected_group_report,
                        config=config,
                        diffusion_model=str(diffusion_model),
                        skipped_reason=f"{type(exc).__name__}: {exc}",
                    )
                ]
            )
        results_by_permutation[spec.name] = frame.copy()
        rows.extend(frame.to_dict(orient="records"))

    summary_frame = _normalize_summary_frame(rows)
    output_dir = _permutation_output_dir(config.output_dir, dataset.name, protected_group_report.protected_attribute)
    comparison_csv_path: Path | None = None
    report_path: Path | None = None
    if output_dir is not None:
        comparison_csv_path = output_dir / f"{dataset.name}_budget{config.budget}_permutation_comparison.csv"
        summary_frame.to_csv(comparison_csv_path, index=False)
        report_path = output_dir / f"{dataset.name}_budget{config.budget}_permutation_report.txt"
        report_path.write_text(format_fim_permutation_report(summary_frame, config), encoding="utf-8")

    return FIMPermutationBenchmarkResult(
        summary_frame=summary_frame,
        results_by_permutation=results_by_permutation,
        comparison_csv_path=comparison_csv_path,
        report_path=report_path,
    )


def run_fim_permutation_benchmark_from_config(
    dataset_config: DatasetConfig,
    config: FIMPermutationRunConfig,
    permutations: Iterable[str] | None = None,
) -> FIMPermutationBenchmarkResult:
    """Load a dataset and run the named permutation benchmark."""

    dataset = load_dataset(dataset_config)
    protected_group_report = verify_protected_groups(dataset, config.protected_attribute)
    return run_fim_permutation_benchmark(
        dataset=dataset,
        protected_group_report=protected_group_report,
        config=config,
        permutations=permutations,
    )
