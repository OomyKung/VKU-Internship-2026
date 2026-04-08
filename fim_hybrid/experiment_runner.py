"""Experiment runner for Fair Influence Maximization research prototypes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
import json

import matplotlib.pyplot as plt
import pandas as pd

from .baselines import community_round_robin, random_seed_set, top_k_by_feature
from .community_detection import detect_communities
from .config import ExperimentConfig
from .data_loader import load_dataset
from .diffusion import IndependentCascadeSimulator
from .fairness import compute_group_sizes, evaluate_fairness
from .feature_extraction import compute_node_features
from .hybrid_optimizer import HybridOptimizationResult, HybridSIEAOptimizer
from .label_generation import generate_singleton_labels
from .ml_training import predict_node_utilities, select_top_candidate_pool, train_node_ranker


def _optimizer_note(config: ExperimentConfig) -> str:
    notes: list[str] = []
    if config.optimizer.disable_swarm_guidance:
        notes.append("swarm_off")
    if config.optimizer.disable_crossover:
        notes.append("crossover_off")
    if config.optimizer.disable_community_repair:
        notes.append("community_repair_off")
    if config.optimizer.debug_logging:
        notes.append(f"debug_every_{config.optimizer.debug_frequency}")
    if config.optimizer.fairness_repair_bias > 0.0:
        notes.append(f"repair_bias={config.optimizer.fairness_repair_bias}")
    return ";".join(notes)


def _save_history_frame(
    history: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    budget: int,
    community_method: str,
    label: str,
) -> Path | None:
    if history.empty:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_method = community_method.replace(" ", "_")
    safe_label = label.replace(" ", "_")
    history_path = output_dir / f"{dataset_name}_budget{budget}_{safe_method}_{safe_label}_history.csv"
    history.to_csv(history_path, index=False)
    return history_path


def _evaluate_seed_set(
    seed_set: list[object],
    simulator: IndependentCascadeSimulator,
    group_sizes: dict[object, int],
    config: ExperimentConfig,
    runtime_seconds: float,
    label: str,
    community_method: str,
    candidate_pool_size: int | None = None,
    note: str | None = None,
) -> dict[str, object]:
    # Re-evaluate each final seed set with the same diffusion/fairness pipeline
    # so all methods are compared on one consistent scoring path.
    diffusion = simulator.simulate_many(
        seed_set,
        runs=config.diffusion.mc_runs,
        protected_attribute=config.fairness.protected_attribute,
        compute_std=False,
    )
    fairness = evaluate_fairness(
        group_spread=diffusion.group_spread_mean,
        group_sizes=group_sizes,
        lambda_weight=config.fairness.lambda_weight,
        target_mode=config.fairness.target_mode,
        total_spread=diffusion.total_spread_mean,
        score_mode=config.fairness.score_mode,
    )
    return {
        "dataset": config.dataset.name,
        "community_method": community_method,
        "method": label,
        "seed_set": json.dumps(list(seed_set)),
        "total_spread": diffusion.total_spread_mean,
        "mf": fairness.mf,
        "mf_component": fairness.mf_component,
        "ideal_mf": fairness.ideal_mf,
        "mf_to_ideal_ratio": fairness.mf_to_ideal_ratio,
        "dcv": fairness.dcv,
        "f_score": fairness.combined_score,
        "score_mode": fairness.score_mode,
        "runtime_seconds": runtime_seconds,
        "candidate_pool_size": candidate_pool_size if candidate_pool_size is not None else len(simulator.graph),
        "group_spread": json.dumps(fairness.group_spread),
        "group_targets": json.dumps(fairness.group_targets),
        "note": note or "",
    }


def _result_row_from_hybrid_result(
    hybrid_result: HybridOptimizationResult,
    config: ExperimentConfig,
    community_method: str,
    runtime_seconds: float,
    label: str,
    note: str | None = None,
) -> dict[str, object]:
    """Build a result row from the optimizer's final cached evaluation."""

    fairness = hybrid_result.best_fairness
    return {
        "dataset": config.dataset.name,
        "community_method": community_method,
        "method": label,
        "seed_set": json.dumps(hybrid_result.best_seed_set),
        "total_spread": hybrid_result.best_spread,
        "mf": fairness.mf,
        "mf_component": fairness.mf_component,
        "ideal_mf": fairness.ideal_mf,
        "mf_to_ideal_ratio": fairness.mf_to_ideal_ratio,
        "dcv": fairness.dcv,
        "f_score": fairness.combined_score,
        "score_mode": fairness.score_mode,
        "runtime_seconds": runtime_seconds,
        "candidate_pool_size": hybrid_result.candidate_pool_size,
        "group_spread": json.dumps(fairness.group_spread),
        "group_targets": json.dumps(fairness.group_targets),
        "note": note or "",
    }


def plot_experiment_results(results: pd.DataFrame, output_dir: Path, stem: str) -> None:
    """Create simple runtime and score comparison plots."""

    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the plotting layer intentionally simple so the research pipeline
    # works even without a larger experiment-management framework.
    for metric in ["f_score", "total_spread", "runtime_seconds"]:
        plt.figure(figsize=(10, 4))
        ordered = results.sort_values(metric, ascending=(metric == "runtime_seconds"))
        plt.bar(ordered["method"], ordered[metric])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(output_dir / f"{stem}_{metric}.png", dpi=200)
        plt.close()


def run_experiment(
    config: ExperimentConfig,
    community_methods: list[str] | None = None,
    baseline_methods: list[str] | None = None,
    enable_plots: bool = True,
    ml_max_nodes: int | None = None,
) -> pd.DataFrame:
    """Run a full FIM experiment and return a results table."""

    # Load the graph once, then reuse the same simulator across all methods.
    dataset = load_dataset(config.dataset)
    simulator = IndependentCascadeSimulator(
        graph=dataset.graph,
        propagation_probability=config.diffusion.propagation_probability,
        seed=config.diffusion.seed,
    )
    group_sizes = compute_group_sizes(dataset.graph, config.fairness.protected_attribute)

    methods = community_methods or [config.community.method]
    baselines = baseline_methods or ["degree", "pagerank", "community_round_robin", "random"]
    results: list[dict[str, object]] = []

    for community_method in methods:
        # Recompute communities and node features for each requested method so
        # baseline and hybrid comparisons stay method-specific.
        community_config = replace(config.community, method=community_method)
        community_result = detect_communities(dataset.graph, community_config)
        feature_frame = compute_node_features(
            dataset.graph,
            community_result,
            protected_attribute=config.fairness.protected_attribute,
        )

        for baseline_name in baselines:
            start = perf_counter()
            # Baselines are intentionally simple and interpretable.
            if baseline_name == "degree":
                seed_set = top_k_by_feature(feature_frame, "degree", config.optimizer.budget)
            elif baseline_name == "pagerank":
                seed_set = top_k_by_feature(feature_frame, "pagerank", config.optimizer.budget)
            elif baseline_name == "community_round_robin":
                seed_set = community_round_robin(
                    feature_frame,
                    community_result,
                    config.optimizer.budget,
                    ranking_feature="pagerank",
                )
            elif baseline_name == "random":
                seed_set = random_seed_set(
                    feature_frame["node_id"].tolist(),
                    config.optimizer.budget,
                    seed=config.random_seed,
                )
            else:
                raise ValueError(f"Unsupported baseline method '{baseline_name}'.")

            runtime_seconds = perf_counter() - start
            results.append(
                _evaluate_seed_set(
                    seed_set=seed_set,
                    simulator=simulator,
                    group_sizes=group_sizes,
                    config=config,
                    runtime_seconds=runtime_seconds,
                    label=baseline_name,
                    community_method=community_method,
                )
            )

        hybrid_start = perf_counter()
        # Run the unified SI+EA optimizer on the full candidate pool.
        optimizer = HybridSIEAOptimizer(
            graph=dataset.graph,
            community_result=community_result,
            feature_frame=feature_frame,
            simulator=simulator,
            fairness_config=config.fairness,
            optimizer_config=config.optimizer,
            mc_runs=config.diffusion.mc_runs,
        )
        hybrid_result = optimizer.optimize()
        hybrid_runtime = perf_counter() - hybrid_start
        history_path = _save_history_frame(
            history=hybrid_result.history,
            output_dir=config.output_dir,
            dataset_name=config.dataset.name,
            budget=config.optimizer.budget,
            community_method=community_method,
            label="hybrid_siea",
        )
        hybrid_note = _optimizer_note(config)
        if history_path is not None:
            hybrid_note = ";".join(part for part in [hybrid_note, f"history={history_path.name}"] if part)
        results.append(
            _result_row_from_hybrid_result(
                hybrid_result=hybrid_result,
                config=config,
                community_method=community_method,
                runtime_seconds=hybrid_runtime,
                label="hybrid_siea",
                note=hybrid_note,
            )
        )

        if config.ml.enabled:
            # The optional ML stage predicts promising nodes and then narrows the
            # hybrid optimizer's candidate pool.
            training_frame = generate_singleton_labels(
                feature_frame=feature_frame,
                simulator=simulator,
                protected_attribute=config.fairness.protected_attribute,
                lambda_weight=config.fairness.lambda_weight,
                mc_runs=config.ml.singleton_mc_runs,
                target_mode=config.fairness.target_mode,
                score_mode=config.fairness.score_mode,
                positive_fraction=config.ml.positive_fraction,
                max_nodes=ml_max_nodes,
            )
            ml_result = train_node_ranker(training_frame, config.ml)
            predicted_scores = predict_node_utilities(feature_frame, ml_result)
            candidate_pool = select_top_candidate_pool(
                feature_frame=feature_frame,
                predicted_scores=predicted_scores,
                top_fraction=config.ml.top_fraction,
            )

            ml_hybrid_start = perf_counter()
            ml_optimizer = HybridSIEAOptimizer(
                graph=dataset.graph,
                community_result=community_result,
                feature_frame=feature_frame,
                simulator=simulator,
                fairness_config=config.fairness,
                optimizer_config=config.optimizer,
                candidate_nodes=candidate_pool,
                guidance_scores=predicted_scores,
                mc_runs=config.diffusion.mc_runs,
            )
            ml_hybrid_result = ml_optimizer.optimize()
            ml_hybrid_runtime = perf_counter() - ml_hybrid_start
            ml_history_path = _save_history_frame(
                history=ml_hybrid_result.history,
                output_dir=config.output_dir,
                dataset_name=config.dataset.name,
                budget=config.optimizer.budget,
                community_method=community_method,
                label="ml_guided_hybrid_siea",
            )
            ml_note_parts = [
                _optimizer_note(config),
                f"model={ml_result.model_name};train_r2={ml_result.training_r2:.4f}",
            ]
            if ml_history_path is not None:
                ml_note_parts.append(f"history={ml_history_path.name}")
            results.append(
                _result_row_from_hybrid_result(
                    hybrid_result=ml_hybrid_result,
                    config=config,
                    community_method=community_method,
                    runtime_seconds=ml_hybrid_runtime,
                    label="ml_guided_hybrid_siea",
                    note=";".join(part for part in ml_note_parts if part),
                )
            )

    result_frame = pd.DataFrame(results)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{config.dataset.name}_budget{config.optimizer.budget}"
    # Save a CSV so results can be inspected without rerunning the experiment.
    result_frame.to_csv(config.output_dir / f"{stem}_results.csv", index=False)

    if enable_plots and not result_frame.empty:
        plot_experiment_results(result_frame, config.output_dir, stem)

    return result_frame
