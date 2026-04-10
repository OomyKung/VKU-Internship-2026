"""Fairness-aware singleton label generation for ML-guided FIM."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import LoadedDataset, ProtectedGroupReport
from .diffusion import DEFAULT_DIFFUSION_MODEL
from .evaluation import evaluate_seed_set


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _min_max_normalize(values: pd.Series) -> pd.Series:
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values.astype(float) - minimum) / (maximum - minimum)


@dataclass(slots=True)
class NodeUtilityLabelResult:
    """Node-level singleton utility labels and diagnostics for ML training."""

    label_frame: pd.DataFrame
    runtime_seconds: float
    label_variance: float
    spread_variance: float
    soft_fair_score_variance: float


def generate_singleton_node_utility_labels(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    propagation_probability: float = 0.01,
    mc_runs: int = 20,
    lambda_weight: float = 0.5,
    random_seed: int = 42,
    diffusion_model: str = DEFAULT_DIFFUSION_MODEL,
) -> NodeUtilityLabelResult:
    """Evaluate singleton seed sets and build a continuous fairness-aware label."""

    if dataset.name != protected_group_report.dataset_name:
        raise ValueError("protected_group_report.dataset_name must match dataset.name.")

    records: list[dict[str, float | int | str | Any]] = []
    start = perf_counter()
    for node_id in sorted(dataset.graph.nodes(), key=_sort_key):
        evaluation = evaluate_seed_set(
            dataset=dataset,
            protected_group_report=protected_group_report,
            seed_set=[node_id],
            propagation_probability=propagation_probability,
            mc_runs=mc_runs,
            random_seed=random_seed,
            lambda_weight=lambda_weight,
            include_soft_mf=True,
            diffusion_model=diffusion_model,
        )
        soft_mf = evaluation.fairness.soft_mf
        if soft_mf is None:
            raise RuntimeError("Singleton label generation requires soft_mf diagnostics to be enabled.")

        soft_fair_score = float(lambda_weight * soft_mf - (1.0 - lambda_weight) * evaluation.fairness.dcv)
        records.append(
            {
                "node_id": node_id,
                "singleton_total_spread": float(evaluation.total_spread_mean),
                "singleton_mf": float(evaluation.fairness.mf),
                "singleton_soft_mf": float(soft_mf),
                "singleton_dcv": float(evaluation.fairness.dcv),
                "singleton_soft_fair_score": soft_fair_score,
            }
        )

    label_frame = pd.DataFrame(records).set_index("node_id", drop=False)
    label_frame["spread_norm"] = _min_max_normalize(label_frame["singleton_total_spread"])
    label_frame["soft_fair_norm"] = _min_max_normalize(label_frame["singleton_soft_fair_score"])
    label_frame["label_score"] = 0.5 * label_frame["spread_norm"] + 0.5 * label_frame["soft_fair_norm"]

    runtime_seconds = perf_counter() - start
    label_variance = float(label_frame["label_score"].var(ddof=0))
    spread_variance = float(label_frame["singleton_total_spread"].var(ddof=0))
    soft_fair_score_variance = float(label_frame["singleton_soft_fair_score"].var(ddof=0))
    if label_variance <= 1e-12:
        raise ValueError(
            "ML-guided mode produced degenerate singleton labels; label_score variance is zero."
        )

    return NodeUtilityLabelResult(
        label_frame=label_frame,
        runtime_seconds=runtime_seconds,
        label_variance=label_variance,
        spread_variance=spread_variance,
        soft_fair_score_variance=soft_fair_score_variance,
    )
