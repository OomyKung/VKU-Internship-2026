"""Search-time objective evaluators for Fair Influence Maximization."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .data_loader import LoadedDataset, ProtectedGroupReport
from .diffusion import DEFAULT_DIFFUSION_MODEL
from .evaluation import SeedSetEvaluation, compute_f_score, evaluate_seed_set as evaluate_seed_set_mc
from .fairness import FairnessMetrics, evaluate_fairness
from .label_generation import NodeUtilityLabelResult
from .ris_guidance import RISGuidanceResult
from .safe_math import safe_divide, safe_minmax_normalize


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _normalized_seed_set(seed_set: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(sorted(set(seed_set), key=_sort_key))


def _min_max_normalize(values: pd.Series) -> pd.Series:
    normalized = safe_minmax_normalize(
        values.astype(float).tolist(),
        default=0.0,
        context=f"search label normalization '{values.name or 'series'}'",
    )
    return pd.Series(normalized, index=values.index, dtype=float)


def _weak_group_gain(normalized_group_spread: dict[Any, float], weakest_group_k: int = 2) -> float:
    if not normalized_group_spread:
        return 0.0
    values = sorted(float(value) for value in normalized_group_spread.values())
    return float(np.mean(values[: max(1, int(weakest_group_k))]))


def _target_attainment_summary(
    group_spread: dict[Any, float],
    group_targets: dict[Any, float],
) -> tuple[float, float]:
    attainment_scores: list[float] = []
    for group_name, target_value in group_targets.items():
        attained = safe_divide(
            float(group_spread.get(group_name, 0.0)),
            max(float(target_value), 1e-9),
            default=0.0,
            context=f"search target attainment for protected group {group_name}",
        )
        attainment_scores.append(float(np.clip(attained, 0.0, 1.0)))
    if not attainment_scores:
        return 0.0, 0.0
    return float(min(attainment_scores)), float(np.mean(attainment_scores))


@dataclass(slots=True)
class SearchObjectiveCounters:
    """Counters and timings for search objective calls."""

    search_mc_eval_calls: int = 0
    search_ris_eval_calls: int = 0
    search_fair_ris_eval_calls: int = 0
    time_ris_evaluation: float = 0.0
    time_mc_search_evaluation: float = 0.0
    rr_sets_used: int = 0
    zero_group_rr_warnings: tuple[str, ...] = ()


@dataclass(slots=True)
class SearchObjectiveEvaluator:
    """Evaluate search objectives with Monte Carlo, RIS, or Fair RIS backends."""

    dataset: LoadedDataset
    protected_group_report: ProtectedGroupReport
    propagation_probability: float = 0.01
    lambda_weight: float = 0.5
    random_seed: int = 42
    diffusion_model: str = DEFAULT_DIFFUSION_MODEL
    backend: str = "monte_carlo"
    mc_runs: int = 20
    ris_result: RISGuidanceResult | None = None
    enable_cache: bool = True
    ideal_influences: dict[str, float] | None = None
    _cache: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict, init=False)
    _zero_group_rr_warnings: set[str] = field(default_factory=set, init=False)
    counters: SearchObjectiveCounters = field(default_factory=SearchObjectiveCounters, init=False)

    def __post_init__(self) -> None:
        normalized_backend = str(self.backend or "monte_carlo").strip().lower()
        aliases = {"fair_ris": "fairness_aware_ris", "ris_guidance": "ris"}
        self.backend = aliases.get(normalized_backend, normalized_backend)
        if self.backend not in {"monte_carlo", "ris", "fairness_aware_ris"}:
            raise ValueError("search objective backend must be one of monte_carlo, ris, fairness_aware_ris.")
        if self.backend in {"ris", "fairness_aware_ris"} and self.ris_result is None:
            raise ValueError(f"{self.backend} search objective requires cached RR sets.")
        if self.backend == "monte_carlo" and int(self.mc_runs) < 1:
            raise ValueError("Monte Carlo search objective requires mc_runs_search >= 1.")
        if self.dataset.name != self.protected_group_report.dataset_name:
            raise ValueError("protected_group_report.dataset_name must match dataset.name.")

    @property
    def uses_ris(self) -> bool:
        return self.backend == "ris"

    @property
    def uses_fair_ris(self) -> bool:
        return self.backend == "fairness_aware_ris"

    def _evaluate_mc(self, seed_set: tuple[Any, ...]) -> dict[str, Any]:
        start = perf_counter()
        evaluation = evaluate_seed_set_mc(
            dataset=self.dataset,
            protected_group_report=self.protected_group_report,
            seed_set=seed_set,
            propagation_probability=float(self.propagation_probability),
            mc_runs=int(self.mc_runs),
            random_seed=int(self.random_seed),
            lambda_weight=float(self.lambda_weight),
            include_soft_mf=True,
            diffusion_model=self.diffusion_model,
            ideal_influences=self.ideal_influences,
        )
        elapsed = perf_counter() - start
        self.counters.search_mc_eval_calls += 1
        self.counters.time_mc_search_evaluation += elapsed
        return self._evaluation_to_payload(
            evaluation=evaluation,
            runtime_seconds=elapsed,
            rr_covered_count=0,
            group_rr_covered_counts={},
        )

    def _evaluate_ris(self, seed_set: tuple[Any, ...]) -> dict[str, Any]:
        if self.ris_result is None:
            raise ValueError("RIS search objective requires RR sets.")
        start = perf_counter()
        seed_nodes = set(seed_set)
        rr_sets = self.ris_result.rr_sets
        total_rr_sets = max(1, len(rr_sets))
        covered_flags = [bool(seed_nodes.intersection(rr_set)) for rr_set in rr_sets]
        covered_rr = int(sum(1 for covered in covered_flags if covered))
        group_rr_total = {
            group_name: int(self.ris_result.rr_set_counts_by_group.get(group_name, 0))
            for group_name in self.protected_group_report.group_sizes
        }
        group_rr_covered = {group_name: 0 for group_name in self.protected_group_report.group_sizes}
        for covered, group_name in zip(covered_flags, self.ris_result.rr_root_groups, strict=True):
            if covered:
                group_rr_covered[group_name] = group_rr_covered.get(group_name, 0) + 1

        approx_group_influence: dict[Any, float] = {}
        for group_name, group_size in self.protected_group_report.group_sizes.items():
            total_for_group = int(group_rr_total.get(group_name, 0))
            if total_for_group <= 0:
                self._zero_group_rr_warnings.add(str(group_name))
                approx_group_influence[group_name] = 0.0
                continue
            approx_group_influence[group_name] = float(group_size) * safe_divide(
                float(group_rr_covered.get(group_name, 0)),
                float(total_for_group),
                default=0.0,
                context=f"RIS group coverage for {group_name}",
            )

        global_spread = float(self.dataset.graph.number_of_nodes()) * safe_divide(
            float(covered_rr),
            float(total_rr_sets),
            default=0.0,
            context="RIS covered RR fraction",
        )
        group_spread_total = float(sum(approx_group_influence.values()))
        approx_total_spread = group_spread_total if self.uses_fair_ris else global_spread
        fairness = evaluate_fairness(
            group_spread=approx_group_influence,
            group_sizes=self.protected_group_report.group_sizes,
            total_spread=approx_total_spread,
            include_soft_mf=True,
            ideal_influences=self.ideal_influences,
        )
        f_score = compute_f_score(fairness.mf, fairness.dcv, float(self.lambda_weight))
        elapsed = perf_counter() - start
        if self.uses_fair_ris:
            self.counters.search_fair_ris_eval_calls += 1
        else:
            self.counters.search_ris_eval_calls += 1
        self.counters.time_ris_evaluation += elapsed
        self.counters.rr_sets_used = len(rr_sets)
        self.counters.zero_group_rr_warnings = tuple(sorted(self._zero_group_rr_warnings, key=_sort_key))
        return {
            "seed_set": seed_set,
            "total_spread_mean": approx_total_spread,
            "total_spread_std": 0.0,
            "fairness": fairness,
            "f_score": f_score,
            "runtime_seconds": elapsed,
            "approx_total_spread": approx_total_spread,
            "approx_extra_spread": float(approx_total_spread - len(seed_set)),
            "approx_group_influence": dict(fairness.group_spread),
            "approx_normalized_group_influence": dict(fairness.normalized_group_spread),
            "approx_MF": float(fairness.mf),
            "approx_DCV": float(fairness.dcv),
            "approx_DCV_shortfall": float(fairness.dcv_shortfall),
            "approx_target_coverage_ratio": float(fairness.target_coverage_ratio),
            "approx_F_score": float(f_score),
            "rr_covered_count": covered_rr,
            "group_rr_covered_counts": group_rr_covered,
            "group_rr_total_counts": group_rr_total,
            "backend": self.backend,
        }

    def _evaluation_to_payload(
        self,
        *,
        evaluation: SeedSetEvaluation,
        runtime_seconds: float,
        rr_covered_count: int,
        group_rr_covered_counts: dict[Any, int],
    ) -> dict[str, Any]:
        return {
            "seed_set": evaluation.seed_set,
            "total_spread_mean": float(evaluation.total_spread_mean),
            "total_spread_std": float(evaluation.total_spread_std),
            "fairness": evaluation.fairness,
            "f_score": float(evaluation.f_score),
            "runtime_seconds": float(runtime_seconds),
            "approx_total_spread": float(evaluation.total_spread_mean),
            "approx_extra_spread": float(evaluation.total_spread_mean - len(evaluation.seed_set)),
            "approx_group_influence": dict(evaluation.fairness.group_spread),
            "approx_normalized_group_influence": dict(evaluation.fairness.normalized_group_spread),
            "approx_MF": float(evaluation.fairness.mf),
            "approx_DCV": float(evaluation.fairness.dcv),
            "approx_DCV_shortfall": float(evaluation.fairness.dcv_shortfall),
            "approx_target_coverage_ratio": float(evaluation.fairness.target_coverage_ratio),
            "approx_F_score": float(evaluation.f_score),
            "rr_covered_count": int(rr_covered_count),
            "group_rr_covered_counts": dict(group_rr_covered_counts),
            "group_rr_total_counts": {},
            "backend": self.backend,
        }

    def evaluate_seed_set(self, seed_set: Iterable[Any]) -> dict[str, Any]:
        """Return approximate search metrics for one seed set."""

        normalized = _normalized_seed_set(seed_set)
        if self.enable_cache and normalized in self._cache:
            return self._cache[normalized]
        payload = self._evaluate_mc(normalized) if self.backend == "monte_carlo" else self._evaluate_ris(normalized)
        if self.enable_cache:
            self._cache[normalized] = payload
        return payload

    def evaluate_seed_set_evaluation(self, seed_set: Iterable[Any]) -> SeedSetEvaluation:
        payload = self.evaluate_seed_set(seed_set)
        return SeedSetEvaluation(
            seed_set=tuple(payload["seed_set"]),
            total_spread_mean=float(payload["total_spread_mean"]),
            total_spread_std=float(payload["total_spread_std"]),
            fairness=payload["fairness"],
            f_score=float(payload["f_score"]),
            runtime_seconds=float(payload["runtime_seconds"]),
        )

    def evaluate_many_seed_sets(self, seed_sets: Iterable[Iterable[Any]]) -> list[dict[str, Any]]:
        return [self.evaluate_seed_set(seed_set) for seed_set in seed_sets]

    def marginal_gain(self, seed_set: Iterable[Any], candidate: Any) -> dict[str, Any]:
        base_seed_set = _normalized_seed_set(seed_set)
        trial_seed_set = _normalized_seed_set([*base_seed_set, candidate])
        base = self.evaluate_seed_set(base_seed_set)
        trial = self.evaluate_seed_set(trial_seed_set)
        return {
            **trial,
            "marginal_total_spread": float(trial["approx_total_spread"]) - float(base["approx_total_spread"]),
            "marginal_f_score": float(trial["approx_F_score"]) - float(base["approx_F_score"]),
            "marginal_mf": float(trial["approx_MF"]) - float(base["approx_MF"]),
            "marginal_dcv_reduction": float(base["approx_DCV"]) - float(trial["approx_DCV"]),
            "marginal_shortfall_dcv_reduction": float(base.get("approx_DCV_shortfall", base["approx_DCV"])) - float(trial.get("approx_DCV_shortfall", trial["approx_DCV"])),
        }

    def verify(self) -> dict[str, Any]:
        rr_sets = len(self.ris_result.rr_sets) if self.ris_result is not None else 0
        return {
            "search_objective_backend": self.backend,
            "ris_active_verified": bool(self.backend in {"ris", "fairness_aware_ris"} and rr_sets > 0),
            "fair_ris_active_verified": bool(self.backend == "fairness_aware_ris" and rr_sets > 0),
            "search_mc_eval_calls": int(self.counters.search_mc_eval_calls),
            "search_ris_eval_calls": int(self.counters.search_ris_eval_calls),
            "search_fair_ris_eval_calls": int(self.counters.search_fair_ris_eval_calls),
            "time_ris_evaluation": float(self.counters.time_ris_evaluation),
            "time_mc_search_evaluation": float(self.counters.time_mc_search_evaluation),
            "time_search_objective_total": float(self.counters.time_ris_evaluation + self.counters.time_mc_search_evaluation),
            "rr_sets_used": int(rr_sets),
            "rr_sets_generated": int(rr_sets),
            "zero_group_rr_warnings": list(self.counters.zero_group_rr_warnings),
        }


def build_singleton_label_frame_from_search_evaluator(
    dataset: LoadedDataset,
    search_evaluator: SearchObjectiveEvaluator,
) -> NodeUtilityLabelResult:
    """Build ML singleton labels from the configured search objective."""

    records: list[dict[str, float | int | str | Any]] = []
    start = perf_counter()
    for node_id in sorted(dataset.graph.nodes(), key=_sort_key):
        payload = search_evaluator.evaluate_seed_set([node_id])
        fairness: FairnessMetrics = payload["fairness"]
        soft_mf = fairness.soft_mf
        if soft_mf is None:
            raise RuntimeError("Search singleton label generation requires soft_mf diagnostics.")
        soft_fair_score = float(search_evaluator.lambda_weight * soft_mf - (1.0 - search_evaluator.lambda_weight) * fairness.dcv)
        weak_group_gain = _weak_group_gain(fairness.normalized_group_spread)
        min_target_attainment, mean_target_attainment = _target_attainment_summary(
            fairness.group_spread,
            fairness.group_targets,
        )
        records.append(
            {
                "node_id": node_id,
                "singleton_total_spread": float(payload["approx_total_spread"]),
                "singleton_mf": float(fairness.mf),
                "singleton_soft_mf": float(soft_mf),
                "singleton_dcv": float(fairness.dcv),
                "singleton_soft_fair_score": soft_fair_score,
                "singleton_weak_group_gain": weak_group_gain,
                "singleton_min_target_attainment": min_target_attainment,
                "singleton_mean_target_attainment": mean_target_attainment,
            }
        )

    label_frame = pd.DataFrame(records).set_index("node_id", drop=False)
    label_frame["spread_norm"] = _min_max_normalize(label_frame["singleton_total_spread"])
    label_frame["soft_fair_norm"] = _min_max_normalize(label_frame["singleton_soft_fair_score"])
    label_frame["weak_group_gain_norm"] = _min_max_normalize(label_frame["singleton_weak_group_gain"])
    label_frame["target_attainment_norm"] = _min_max_normalize(label_frame["singleton_mean_target_attainment"])
    label_frame["label_score"] = (
        0.35 * label_frame["spread_norm"]
        + 0.30 * label_frame["soft_fair_norm"]
        + 0.20 * label_frame["weak_group_gain_norm"]
        + 0.15 * label_frame["target_attainment_norm"]
    )

    return NodeUtilityLabelResult(
        label_frame=label_frame,
        runtime_seconds=float(perf_counter() - start),
        label_variance=float(label_frame["label_score"].var(ddof=0)),
        spread_variance=float(label_frame["singleton_total_spread"].var(ddof=0)),
        soft_fair_score_variance=float(label_frame["singleton_soft_fair_score"].var(ddof=0)),
    )
