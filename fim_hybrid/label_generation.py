"""Training label generation for ML-guided candidate filtering."""

from __future__ import annotations

from time import perf_counter

import pandas as pd

from .diffusion import IndependentCascadeSimulator
from .fairness import compute_group_sizes, evaluate_fairness


def generate_singleton_labels(
    feature_frame: pd.DataFrame,
    simulator: IndependentCascadeSimulator,
    protected_attribute: str,
    lambda_weight: float,
    mc_runs: int,
    target_mode: str = "population_proportional",
    positive_fraction: float = 0.2,
    max_nodes: int | None = None,
    ranking_feature: str = "pagerank",
) -> pd.DataFrame:
    """Generate simulation-based labels for singleton seed nodes."""

    if max_nodes is not None and ranking_feature in feature_frame.columns:
        # Limit label generation to a promising slice when the graph is large,
        # because Monte Carlo singleton scoring can become expensive.
        candidate_frame = feature_frame.sort_values(ranking_feature, ascending=False).head(max_nodes).copy()
    else:
        candidate_frame = feature_frame.copy()
    candidate_frame = candidate_frame.reset_index(drop=True)

    group_sizes = compute_group_sizes(simulator.graph, protected_attribute)
    records = []
    start = perf_counter()

    for node in candidate_frame["node_id"].tolist():
        # Evaluate each singleton node with the same fairness-aware objective used later.
        diffusion = simulator.simulate_many([node], runs=mc_runs, protected_attribute=protected_attribute)
        fairness = evaluate_fairness(
            group_spread=diffusion.group_spread_mean,
            group_sizes=group_sizes,
            lambda_weight=lambda_weight,
            target_mode=target_mode,
            total_spread=diffusion.total_spread_mean,
        )
        records.append(
            {
                "node_id": node,
                "singleton_spread": diffusion.total_spread_mean,
                "singleton_mf": fairness.mf,
                "singleton_dcv": fairness.dcv,
                "singleton_score": fairness.combined_score,
            }
        )

    label_frame = pd.DataFrame(records)
    if label_frame.empty:
        raise ValueError("Singleton label generation produced no training examples.")

    # Turn the continuous singleton score into a simple high-vs-not-high label.
    threshold = label_frame["singleton_score"].quantile(1.0 - positive_fraction)
    label_frame["ml_label"] = (label_frame["singleton_score"] >= threshold).astype(int)
    label_frame["label_generation_seconds"] = perf_counter() - start

    merged = candidate_frame.merge(label_frame, on="node_id", how="left")
    return merged.set_index("node_id", drop=False)
