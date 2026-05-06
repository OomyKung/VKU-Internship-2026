"""Target-shortfall-first RIS memetic optimizer.

This module adds a focused method without changing the older optimizers. It
uses RIS for search-time marginal guidance and keeps the final benchmark
evaluation in the existing Monte Carlo path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from time import perf_counter
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np

from .data_loader import ProtectedGroupReport
from .fairness import compute_target_shortfall_diagnostics, evaluate_fairness
from .ris_guidance import RISGuidanceResult


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _normalized_seed_set(nodes: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(sorted(dict.fromkeys(nodes), key=_sort_key))


@dataclass(slots=True)
class TargetShortfallFitnessWeights:
    normalized_spread: float = 0.15
    target_coverage_ratio: float = 12.0
    mf: float = 3.0
    dcv_shortfall: float = 10.0
    dcv: float = 1.0
    f_score: float = 0.5
    total_shortfall: float = 0.0
    overcoverage_waste: float = 0.0


@dataclass(slots=True)
class TargetShortfallRepairConfig:
    budget: int
    population_size: int = 20
    generations: int = 12
    mutation_rate: float = 0.25
    crossover_rate: float = 0.90
    elite_fraction: float = 0.25
    tournament_size: int = 3
    mutation_strength: int | None = None
    local_search_steps: int = 2
    local_search_elite_fraction: float = 0.25
    repair_rounds: int = 12
    repair_candidate_limit: int = 400
    repair_aggressive: bool = True
    random_seed: int = 42
    propagation_probability: float = 0.01
    weights: TargetShortfallFitnessWeights = field(default_factory=TargetShortfallFitnessWeights)


@dataclass(slots=True)
class TargetShortfallProxyEvaluation:
    seed_set: tuple[Any, ...]
    total_spread: float
    normalized_spread: float
    group_influence: dict[str, float]
    normalized_group_influence: dict[str, float]
    mf: float
    dcv: float
    dcv_shortfall: float
    target_coverage_ratio: float
    f_score: float
    raw_gap_by_group: dict[str, float]
    shortfall_ratio_by_group: dict[str, float]
    overcoverage_ratio_by_group: dict[str, float]
    total_raw_shortfall: float
    total_shortfall: float
    overcoverage_waste: float
    fitness: float
    runtime_seconds: float = 0.0


@dataclass(slots=True)
class TargetShortfallRepairResult:
    seed_set: tuple[Any, ...]
    diagnostics: dict[str, object]
    best_generation: int
    final_fitness: float
    runtime_seconds: float


def _group_by_node(report: ProtectedGroupReport) -> dict[Any, str]:
    return {
        node_id: str(group_name)
        for group_name, nodes in report.protected_groups.items()
        for node_id in nodes
    }


def _seed_group_counts(seed_set: Sequence[Any], group_by_node: Mapping[Any, str]) -> dict[str, int]:
    return dict(Counter(str(group_by_node.get(node_id, "unknown")) for node_id in seed_set))


def _seed_spread_factor(graph: nx.Graph, propagation_probability: float) -> float:
    n_nodes = max(1, int(graph.number_of_nodes()))
    if graph.is_directed():
        avg_degree = float(graph.number_of_edges()) / float(n_nodes)
    else:
        avg_degree = 2.0 * float(graph.number_of_edges()) / float(n_nodes)
    return 1.0 + min(0.05, max(0.0, float(propagation_probability) * avg_degree))


def _target_gap_tuple(group_influence: Mapping[str, float], ideal_influences: Mapping[str, float]) -> tuple[float, float]:
    diagnostics = compute_target_shortfall_diagnostics(dict(group_influence), dict(ideal_influences))
    ratios = diagnostics["shortfall_ratio_by_group"]
    met = sum(1 for value in ratios.values() if float(value) <= 1e-9)
    total = max(1, len(ratios))
    return float(met) / float(total), float(diagnostics["total_shortfall"])


def _quota_score(
    quotas: Mapping[str, int],
    ideal_influences: Mapping[str, float],
    seed_factor: float,
) -> tuple[float, float, float, float]:
    group_influence = {
        group_name: float(quotas.get(group_name, 0)) * float(seed_factor)
        for group_name in ideal_influences
    }
    coverage, shortfall = _target_gap_tuple(group_influence, ideal_influences)
    diagnostics = compute_target_shortfall_diagnostics(group_influence, dict(ideal_influences))
    ratios = diagnostics["shortfall_ratio_by_group"]
    max_ratio = max((float(value) for value in ratios.values()), default=0.0)
    overcoverage = float(diagnostics["overcoverage_waste"])
    return -shortfall, -max_ratio, coverage, -overcoverage


def compute_target_shortfall_group_quotas(
    *,
    budget: int,
    group_sizes: Mapping[str, int],
    ideal_influences: Mapping[str, float],
    seed_factor: float,
) -> dict[str, int]:
    """Allocate exact-budget quotas that maximize proxy target coverage first."""

    group_names = [
        str(group_name)
        for group_name in sorted(group_sizes, key=_sort_key)
        if int(group_sizes[group_name]) > 0
    ]
    if not group_names or budget <= 0:
        return {}
    positive_ideal = {
        group_name: max(0.0, float(ideal_influences.get(group_name, 0.0)))
        for group_name in group_names
    }
    quotas = {group_name: 0 for group_name in group_names}
    if budget >= len(group_names):
        for group_name in group_names:
            if positive_ideal.get(group_name, 0.0) > 0.0:
                quotas[group_name] = 1

    required_to_meet = {
        group_name: max(1, int(math.ceil(float(positive_ideal[group_name]) / max(float(seed_factor), 1e-9))))
        for group_name in group_names
        if positive_ideal.get(group_name, 0.0) > 0.0
    }
    remaining = int(budget) - int(sum(quotas.values()))
    for group_name in sorted(
        required_to_meet,
        key=lambda name: (
            max(0, int(required_to_meet[name]) - int(quotas.get(name, 0))),
            int(group_sizes.get(name, 0)),
            _sort_key(name),
        ),
    ):
        needed = max(0, int(required_to_meet[group_name]) - int(quotas.get(group_name, 0)))
        if needed <= 0 or needed > remaining:
            continue
        quotas[group_name] += needed
        remaining -= needed

    while sum(quotas.values()) < int(budget):
        receiver = max(
            group_names,
            key=lambda group_name: (
                float(positive_ideal.get(group_name, 0.0))
                - float(quotas.get(group_name, 0)) * float(seed_factor),
                float(positive_ideal.get(group_name, 0.0)),
                int(group_sizes.get(group_name, 0)),
                _sort_key(group_name),
            ),
        )
        quotas[receiver] += 1
    while sum(quotas.values()) > int(budget):
        donor = min(
            [group_name for group_name in group_names if int(quotas.get(group_name, 0)) > 0],
            key=lambda group_name: (
                float(positive_ideal.get(group_name, 0.0))
                - float(max(0, int(quotas.get(group_name, 0)) - 1)) * float(seed_factor),
                int(group_sizes.get(group_name, 0)),
                _sort_key(group_name),
            ),
        )
        quotas[donor] -= 1
    return {group_name: int(quotas.get(group_name, 0)) for group_name in group_names}


def _covered_rr_flags(seed_set: set[Any], ris_result: RISGuidanceResult | None) -> list[bool]:
    if ris_result is None:
        return []
    return [bool(seed_set.intersection(rr_set)) for rr_set in ris_result.rr_sets]


def _rr_group_marginal_gain(
    *,
    candidate: Any,
    seed_set: set[Any],
    ris_result: RISGuidanceResult | None,
) -> dict[str, int]:
    if ris_result is None:
        return {}
    gains: dict[str, int] = {group_name: 0 for group_name in ris_result.rr_set_counts_by_group}
    for rr_set, root_group in zip(ris_result.rr_sets, ris_result.rr_root_groups, strict=True):
        if candidate not in rr_set:
            continue
        if seed_set.intersection(rr_set):
            continue
        gains[str(root_group)] = int(gains.get(str(root_group), 0)) + 1
    return gains


def _rr_unique_seed_gain(
    *,
    seed: Any,
    seed_set: set[Any],
    ris_result: RISGuidanceResult | None,
) -> dict[str, int]:
    if ris_result is None:
        return {}
    others = set(seed_set)
    others.discard(seed)
    gains: dict[str, int] = {group_name: 0 for group_name in ris_result.rr_set_counts_by_group}
    for rr_set, root_group in zip(ris_result.rr_sets, ris_result.rr_root_groups, strict=True):
        if seed not in rr_set or others.intersection(rr_set):
            continue
        gains[str(root_group)] = int(gains.get(str(root_group), 0)) + 1
    return gains


def _proxy_group_influence(
    *,
    graph: nx.Graph,
    seed_set: tuple[Any, ...],
    protected_group_report: ProtectedGroupReport,
    group_by_node: Mapping[Any, str],
    ris_result: RISGuidanceResult | None,
    propagation_probability: float,
) -> dict[str, float]:
    seed_factor = _seed_spread_factor(graph, propagation_probability)
    counts = _seed_group_counts(seed_set, group_by_node)
    seed_proxy = {
        str(group_name): float(counts.get(str(group_name), 0)) * seed_factor
        for group_name in protected_group_report.group_sizes
    }
    return seed_proxy


def _proxy_f_score(mf: float, dcv_shortfall: float, dcv_disparity: float) -> float:
    return float(float(mf) - float(dcv_shortfall) - 0.25 * float(dcv_disparity))


def _fitness(
    evaluation: TargetShortfallProxyEvaluation,
    weights: TargetShortfallFitnessWeights,
) -> float:
    return float(
        float(weights.target_coverage_ratio) * evaluation.target_coverage_ratio
        - float(weights.dcv_shortfall) * evaluation.dcv_shortfall
        + float(weights.mf) * evaluation.mf
        - float(weights.dcv) * evaluation.dcv
        + float(weights.f_score) * evaluation.f_score
        + float(weights.normalized_spread) * evaluation.normalized_spread
        - float(weights.total_shortfall) * evaluation.total_shortfall
        - float(weights.overcoverage_waste) * evaluation.overcoverage_waste
    )


def _evaluate_proxy(
    *,
    graph: nx.Graph,
    seed_set: Sequence[Any],
    protected_group_report: ProtectedGroupReport,
    group_by_node: Mapping[Any, str],
    ideal_influences: Mapping[str, float],
    ris_result: RISGuidanceResult | None,
    config: TargetShortfallRepairConfig,
) -> TargetShortfallProxyEvaluation:
    normalized = _normalized_seed_set(seed_set)
    group_influence = _proxy_group_influence(
        graph=graph,
        seed_set=normalized,
        protected_group_report=protected_group_report,
        group_by_node=group_by_node,
        ris_result=ris_result,
        propagation_probability=float(config.propagation_probability),
    )
    total_spread = float(sum(group_influence.values()))
    fairness = evaluate_fairness(
        group_spread=group_influence,
        group_sizes=protected_group_report.group_sizes,
        total_spread=total_spread,
        ideal_influences=dict(ideal_influences),
    )
    diagnostics = compute_target_shortfall_diagnostics(group_influence, dict(ideal_influences))
    shortfall_ratios = dict(diagnostics["shortfall_ratio_by_group"])
    f_score = _proxy_f_score(
        float(fairness.mf),
        float(fairness.dcv_shortfall),
        float(fairness.dcv),
    )
    evaluation = TargetShortfallProxyEvaluation(
        seed_set=normalized,
        total_spread=total_spread,
        normalized_spread=float(total_spread) / float(max(1, int(config.budget))),
        group_influence=dict(group_influence),
        normalized_group_influence=dict(fairness.normalized_group_spread),
        mf=float(fairness.mf),
        dcv=float(fairness.dcv),
        dcv_shortfall=float(fairness.dcv_shortfall),
        target_coverage_ratio=float(fairness.target_coverage_ratio),
        f_score=float(f_score),
        raw_gap_by_group=dict(diagnostics["raw_gap_by_group"]),
        shortfall_ratio_by_group=shortfall_ratios,
        overcoverage_ratio_by_group=dict(diagnostics["overcoverage_ratio_by_group"]),
        total_raw_shortfall=float(diagnostics["total_raw_shortfall"]),
        total_shortfall=float(diagnostics["total_shortfall"]),
        overcoverage_waste=float(diagnostics["overcoverage_waste"]),
        fitness=0.0,
        runtime_seconds=0.0,
    )
    evaluation.fitness = _fitness(evaluation, config.weights)
    return evaluation


def _priority_tuple(evaluation: TargetShortfallProxyEvaluation) -> tuple[float, float, float, float, float, float, float]:
    return (
        float(evaluation.target_coverage_ratio),
        -float(evaluation.dcv_shortfall),
        float(evaluation.mf),
        -float(evaluation.dcv),
        float(evaluation.f_score),
        float(evaluation.total_spread),
        -float(getattr(evaluation, "runtime_seconds", 0.0) or 0.0),
    )


def compare_shortfall_priority(a: TargetShortfallProxyEvaluation, b: TargetShortfallProxyEvaluation) -> int:
    """Comparator for target-coverage-first seed-set quality."""

    left = _priority_tuple(a)
    right = _priority_tuple(b)
    if left > right:
        return 1
    if left < right:
        return -1
    return 0


def _is_better(a: TargetShortfallProxyEvaluation, b: TargetShortfallProxyEvaluation) -> bool:
    return compare_shortfall_priority(a, b) > 0


def _groups_losing_target(
    current: TargetShortfallProxyEvaluation,
    trial: TargetShortfallProxyEvaluation,
    *,
    tolerance: float = 1e-9,
) -> tuple[str, ...]:
    lost = [
        group_name
        for group_name, current_ratio in current.shortfall_ratio_by_group.items()
        if float(current_ratio) <= tolerance
        and float(trial.shortfall_ratio_by_group.get(group_name, 0.0)) > tolerance
    ]
    return tuple(sorted(lost, key=_sort_key))


def _below_target_progress(
    current: TargetShortfallProxyEvaluation,
    trial: TargetShortfallProxyEvaluation,
    *,
    tolerance: float = 1e-9,
) -> bool:
    below_groups = [
        group_name
        for group_name, ratio in current.shortfall_ratio_by_group.items()
        if float(ratio) > tolerance
    ]
    if not below_groups:
        return False
    current_gap = sum(float(current.raw_gap_by_group.get(group_name, 0.0)) for group_name in below_groups)
    trial_gap = sum(float(trial.raw_gap_by_group.get(group_name, 0.0)) for group_name in below_groups)
    current_ratio = sum(float(current.shortfall_ratio_by_group.get(group_name, 0.0)) for group_name in below_groups)
    trial_ratio = sum(float(trial.shortfall_ratio_by_group.get(group_name, 0.0)) for group_name in below_groups)
    return bool(trial_gap < current_gap - tolerance or trial_ratio < current_ratio - tolerance)


def _target_first_acceptance_reason(
    current: TargetShortfallProxyEvaluation,
    trial: TargetShortfallProxyEvaluation,
    *,
    allow_spread_only: bool,
    dcv_worsen_tolerance: float = 1e-6,
) -> str | None:
    eps = 1e-12
    if trial.target_coverage_ratio < current.target_coverage_ratio - eps:
        return None
    if _groups_losing_target(current, trial):
        return None
    if trial.dcv_shortfall > current.dcv_shortfall + float(dcv_worsen_tolerance):
        return None
    if trial.target_coverage_ratio > current.target_coverage_ratio + eps:
        return "target_coverage"
    if trial.dcv_shortfall < current.dcv_shortfall - eps:
        return "shortfall"
    if _below_target_progress(current, trial):
        return "below_target_progress"
    if trial.f_score > current.f_score + eps and trial.dcv_shortfall <= current.dcv_shortfall + eps:
        return "f_score"
    if (
        allow_spread_only
        and trial.total_spread > current.total_spread + eps
        and abs(trial.target_coverage_ratio - current.target_coverage_ratio) <= eps
        and trial.dcv_shortfall <= current.dcv_shortfall + eps
    ):
        return "spread_only"
    return None


def _candidate_score(
    *,
    candidate: Any,
    seed_set: set[Any],
    current_eval: TargetShortfallProxyEvaluation,
    candidate_scores: Mapping[Any, float],
    group_by_node: Mapping[Any, str],
    ris_result: RISGuidanceResult | None,
    quotas: Mapping[str, int] | None = None,
) -> float:
    group_weights = {
        group_name: (float(value) if float(value) > 0.05 else 0.0)
        for group_name, value in current_eval.shortfall_ratio_by_group.items()
    }
    gains = _rr_group_marginal_gain(candidate=candidate, seed_set=seed_set, ris_result=ris_result)
    weighted_gain = 0.0
    total_gain = 0.0
    for group_name, gain in gains.items():
        rr_total = 1
        if ris_result is not None:
            rr_total = max(1, int(ris_result.rr_set_counts_by_group.get(group_name, 1)))
        normalized_gain = float(gain) / float(rr_total)
        weighted_gain += float(group_weights.get(group_name, 0.0)) * normalized_gain
        total_gain += normalized_gain
    own_group = str(group_by_node.get(candidate, ""))
    own_shortfall = float(group_weights.get(own_group, 0.0))
    current_counts = _seed_group_counts(tuple(seed_set), group_by_node)
    quota_gap = 0.0
    quota_over = 0.0
    if quotas is not None:
        quota = max(1, int(quotas.get(own_group, 0)))
        count = int(current_counts.get(own_group, 0))
        quota_gap = max(0.0, float(quota - count) / float(quota))
        quota_over = max(0.0, float(count - quota) / float(quota))
    over_penalty = 0.0
    if own_shortfall <= 1e-9:
        over_penalty += 0.30
    over_penalty += 1.25 * float(current_eval.overcoverage_ratio_by_group.get(own_group, 0.0))
    over_penalty += 0.50 * quota_over
    return float(
        5.0 * weighted_gain
        + 2.0 * own_shortfall
        + 0.75 * quota_gap
        + 0.05 * total_gain
        + 0.10 * float(candidate_scores.get(candidate, 0.0))
        - over_penalty
    )


def _seed_removal_score(
    *,
    seed: Any,
    seed_set: set[Any],
    current_eval: TargetShortfallProxyEvaluation,
    candidate_scores: Mapping[Any, float],
    group_by_node: Mapping[Any, str],
    ideal_influences: Mapping[str, float],
    ris_result: RISGuidanceResult | None,
    quotas: Mapping[str, int] | None = None,
) -> float:
    group_weights = {
        group_name: (float(value) if float(value) > 0.05 else 0.0)
        for group_name, value in current_eval.shortfall_ratio_by_group.items()
    }
    gains = _rr_unique_seed_gain(seed=seed, seed_set=seed_set, ris_result=ris_result)
    below_target_value = 0.0
    for group_name, gain in gains.items():
        rr_total = 1
        if ris_result is not None:
            rr_total = max(1, int(ris_result.rr_set_counts_by_group.get(group_name, 1)))
        below_target_value += float(group_weights.get(group_name, 0.0)) * float(gain) / float(rr_total)
    own_group = str(group_by_node.get(seed, ""))
    own_actual = float(current_eval.group_influence.get(own_group, 0.0))
    own_ideal = float(ideal_influences.get(own_group, 0.0))
    barely_met_penalty = 0.0
    if own_ideal > 0.0 and own_actual >= own_ideal and (own_actual - own_ideal) <= 0.5:
        barely_met_penalty = 5.0
    overcoverage_credit = float(current_eval.overcoverage_ratio_by_group.get(own_group, 0.0))
    quota_over_credit = 0.0
    if quotas is not None:
        counts = _seed_group_counts(tuple(seed_set), group_by_node)
        quota = max(1, int(quotas.get(own_group, 0)))
        quota_over_credit = max(0.0, float(int(counts.get(own_group, 0)) - quota) / float(quota))
    return float(
        3.0 * below_target_value
        + barely_met_penalty
        + 0.05 * float(candidate_scores.get(seed, 0.0))
        - 2.5 * overcoverage_credit
        - 1.0 * quota_over_credit
    )


def _rank_candidates(
    *,
    candidates: Sequence[Any],
    seed_set: set[Any],
    current_eval: TargetShortfallProxyEvaluation,
    candidate_scores: Mapping[Any, float],
    group_by_node: Mapping[Any, str],
    ris_result: RISGuidanceResult | None,
    limit: int,
    quotas: Mapping[str, int] | None = None,
) -> list[Any]:
    available = [node_id for node_id in candidates if node_id not in seed_set]
    ranked = sorted(
        available,
        key=lambda node_id: (
            -_candidate_score(
                candidate=node_id,
                seed_set=seed_set,
                current_eval=current_eval,
                candidate_scores=candidate_scores,
                group_by_node=group_by_node,
                ris_result=ris_result,
                quotas=quotas,
            ),
            _sort_key(node_id),
        ),
    )
    return ranked[: max(1, int(limit))]


def _rank_removals(
    *,
    seed_set: tuple[Any, ...],
    current_eval: TargetShortfallProxyEvaluation,
    candidate_scores: Mapping[Any, float],
    group_by_node: Mapping[Any, str],
    ideal_influences: Mapping[str, float],
    ris_result: RISGuidanceResult | None,
    quotas: Mapping[str, int] | None = None,
) -> list[Any]:
    seed_nodes = set(seed_set)
    return sorted(
        seed_set,
        key=lambda node_id: (
            _seed_removal_score(
                seed=node_id,
                seed_set=seed_nodes,
                current_eval=current_eval,
                candidate_scores=candidate_scores,
                group_by_node=group_by_node,
                ideal_influences=ideal_influences,
                ris_result=ris_result,
                quotas=quotas,
            ),
            _sort_key(node_id),
        ),
    )


def _enforce_quota_floor(
    *,
    seed_set: Sequence[Any],
    candidates: Sequence[Any],
    quotas: Mapping[str, int],
    candidate_scores: Mapping[Any, float],
    group_by_node: Mapping[Any, str],
    budget: int,
) -> tuple[Any, ...]:
    current = list(_normalized_seed_set(seed_set))
    selected = set(current)
    counts = _seed_group_counts(current, group_by_node)
    ranked_candidates_by_group: dict[str, list[Any]] = {}
    for node_id in sorted(candidates, key=lambda node: (-float(candidate_scores.get(node, 0.0)), _sort_key(node))):
        if node_id in selected:
            continue
        ranked_candidates_by_group.setdefault(str(group_by_node.get(node_id, "")), []).append(node_id)

    for group_name in sorted(quotas, key=_sort_key):
        required = int(quotas.get(group_name, 0))
        while int(counts.get(group_name, 0)) < required:
            additions = ranked_candidates_by_group.get(group_name, [])
            if not additions:
                break
            donor_candidates = [
                node_id
                for node_id in current
                if int(counts.get(str(group_by_node.get(node_id, "")), 0)) > int(quotas.get(str(group_by_node.get(node_id, "")), 0))
            ]
            if not donor_candidates:
                donor_candidates = list(current)
            donor = min(donor_candidates, key=lambda node: (float(candidate_scores.get(node, 0.0)), _sort_key(node)))
            addition = additions.pop(0)
            donor_group = str(group_by_node.get(donor, ""))
            current = [node_id for node_id in current if node_id != donor]
            selected.discard(donor)
            counts[donor_group] = int(counts.get(donor_group, 0)) - 1
            current.append(addition)
            selected.add(addition)
            counts[group_name] = int(counts.get(group_name, 0)) + 1

    if len(current) != int(budget):
        current = list(_repair_budget(current, candidates, candidate_scores, int(budget)))
    return _normalized_seed_set(current)


def _accept_repair_swap(
    current: TargetShortfallProxyEvaluation,
    trial: TargetShortfallProxyEvaluation,
    *,
    aggressive: bool = True,
) -> bool:
    if len(trial.seed_set) != len(current.seed_set):
        return False
    if trial.target_coverage_ratio + 1e-12 < current.target_coverage_ratio:
        return False
    if _groups_losing_target(current, trial):
        return False
    if trial.dcv_shortfall > current.dcv_shortfall + 1e-6:
        return False
    reason = _target_first_acceptance_reason(
        current,
        trial,
        allow_spread_only=False,
        dcv_worsen_tolerance=1e-6,
    )
    if reason is not None:
        return True
    if bool(aggressive) and trial.total_shortfall < current.total_shortfall - 1e-9:
        return True
    if bool(aggressive) and trial.dcv < current.dcv - 1e-6 and trial.total_spread >= current.total_spread - 0.5:
        return True
    return False


def _repair_seed_set(
    *,
    graph: nx.Graph,
    seed_set: Sequence[Any],
    candidates: Sequence[Any],
    protected_group_report: ProtectedGroupReport,
    group_by_node: Mapping[Any, str],
    ideal_influences: Mapping[str, float],
    candidate_scores: Mapping[Any, float],
    ris_result: RISGuidanceResult | None,
    config: TargetShortfallRepairConfig,
    diagnostics: dict[str, int],
    quotas: Mapping[str, int] | None = None,
) -> tuple[Any, ...]:
    current = _normalized_seed_set(seed_set)
    if len(current) != int(config.budget):
        current = _repair_budget(current, candidates, candidate_scores, int(config.budget))
    if quotas is not None:
        current = _enforce_quota_floor(
            seed_set=current,
            candidates=candidates,
            quotas=quotas,
            candidate_scores=candidate_scores,
            group_by_node=group_by_node,
            budget=int(config.budget),
        )
    rounds = max(1, int(config.repair_rounds))
    candidate_limit = max(1, int(config.repair_candidate_limit))
    for _ in range(rounds):
        current_eval = _evaluate_proxy(
            graph=graph,
            seed_set=current,
            protected_group_report=protected_group_report,
            group_by_node=group_by_node,
            ideal_influences=ideal_influences,
            ris_result=ris_result,
            config=config,
        )
        if current_eval.total_shortfall <= 1e-12:
            break
        seed_nodes = set(current)
        additions = _rank_candidates(
            candidates=candidates,
            seed_set=seed_nodes,
            current_eval=current_eval,
            candidate_scores=candidate_scores,
            group_by_node=group_by_node,
            ris_result=ris_result,
            limit=candidate_limit,
            quotas=quotas,
        )
        removals = _rank_removals(
            seed_set=current,
            current_eval=current_eval,
            candidate_scores=candidate_scores,
            group_by_node=group_by_node,
            ideal_influences=ideal_influences,
            ris_result=ris_result,
            quotas=quotas,
        )
        current_counts = _seed_group_counts(current, group_by_node)
        accepted: tuple[Any, ...] | None = None
        accepted_eval: TargetShortfallProxyEvaluation | None = None
        for node_to_remove in removals:
            removed_group = str(group_by_node.get(node_to_remove, ""))
            if quotas is not None and int(current_counts.get(removed_group, 0)) <= int(quotas.get(removed_group, 0)):
                if float(current_eval.overcoverage_ratio_by_group.get(removed_group, 0.0)) < 0.05:
                    continue
            retained = [node_id for node_id in current if node_id != node_to_remove]
            for node_to_add in additions:
                if node_to_add in retained:
                    continue
                diagnostics["repair_attempts"] = int(diagnostics.get("repair_attempts", 0)) + 1
                trial = _normalized_seed_set([*retained, node_to_add])
                if len(trial) != int(config.budget):
                    continue
                trial_eval = _evaluate_proxy(
                    graph=graph,
                    seed_set=trial,
                    protected_group_report=protected_group_report,
                    group_by_node=group_by_node,
                    ideal_influences=ideal_influences,
                    ris_result=ris_result,
                    config=config,
                )
                if trial_eval.target_coverage_ratio < current_eval.target_coverage_ratio - 1e-12 or _groups_losing_target(current_eval, trial_eval):
                    diagnostics["swaps_rejected_due_to_target_loss"] = int(diagnostics.get("swaps_rejected_due_to_target_loss", 0)) + 1
                    continue
                if _accept_repair_swap(current_eval, trial_eval, aggressive=bool(config.repair_aggressive)):
                    reason = _target_first_acceptance_reason(
                        current_eval,
                        trial_eval,
                        allow_spread_only=False,
                        dcv_worsen_tolerance=1e-6,
                    )
                    accepted = trial
                    accepted_eval = trial_eval
                    diagnostics["repair_successes"] = int(diagnostics.get("repair_successes", 0)) + 1
                    if trial_eval.target_coverage_ratio > current_eval.target_coverage_ratio + 1e-12:
                        diagnostics["repairs_improved_target_coverage"] = int(diagnostics.get("repairs_improved_target_coverage", 0)) + 1
                    if trial_eval.dcv_shortfall < current_eval.dcv_shortfall - 1e-9:
                        diagnostics["repairs_reduced_total_shortfall"] = int(diagnostics.get("repairs_reduced_total_shortfall", 0)) + 1
                    if reason == "target_coverage":
                        diagnostics["local_swaps_accepted_target_coverage"] = int(diagnostics.get("local_swaps_accepted_target_coverage", 0)) + 1
                    elif reason == "shortfall":
                        diagnostics["local_swaps_accepted_shortfall"] = int(diagnostics.get("local_swaps_accepted_shortfall", 0)) + 1
                    elif reason == "below_target_progress":
                        diagnostics["local_swaps_accepted_below_target_progress"] = int(diagnostics.get("local_swaps_accepted_below_target_progress", 0)) + 1
                    elif trial_eval.dcv_shortfall < current_eval.dcv_shortfall - 1e-9:
                        diagnostics["local_swaps_accepted_shortfall"] = int(diagnostics.get("local_swaps_accepted_shortfall", 0)) + 1
                    if float(current_eval.overcoverage_ratio_by_group.get(removed_group, 0.0)) > 0.05:
                        diagnostics["seeds_removed_from_overcovered_groups"] = int(diagnostics.get("seeds_removed_from_overcovered_groups", 0)) + 1
                    break
            if accepted is not None:
                break
        if accepted is None or accepted_eval is None:
            break
        current = accepted
    return current


def _repair_budget(
    seed_set: Sequence[Any],
    candidates: Sequence[Any],
    candidate_scores: Mapping[Any, float],
    budget: int,
) -> tuple[Any, ...]:
    unique = list(_normalized_seed_set(seed_set))
    if len(unique) > budget:
        unique = sorted(unique, key=lambda node_id: (-float(candidate_scores.get(node_id, 0.0)), _sort_key(node_id)))[:budget]
    selected = set(unique)
    for node_id in sorted(candidates, key=lambda node_id: (-float(candidate_scores.get(node_id, 0.0)), _sort_key(node_id))):
        if len(unique) >= budget:
            break
        if node_id not in selected:
            unique.append(node_id)
            selected.add(node_id)
    return _normalized_seed_set(unique)


def _build_individual_from_quotas(
    *,
    quotas: Mapping[str, int],
    candidates_by_group: Mapping[str, list[Any]],
    global_candidates: Sequence[Any],
    candidate_scores: Mapping[Any, float],
    budget: int,
    community_by_node: Mapping[Any, Any] | None = None,
    rng: np.random.Generator | None = None,
    randomize: bool = False,
) -> tuple[Any, ...]:
    selected: list[Any] = []
    selected_set: set[Any] = set()
    for group_name in sorted(quotas, key=_sort_key):
        group_candidates = list(candidates_by_group.get(group_name, ()))
        if randomize and rng is not None and len(group_candidates) > int(quotas[group_name]):
            top_window = group_candidates[: max(int(quotas[group_name]) * 3, int(quotas[group_name]), 1)]
            rng.shuffle(top_window)
            group_candidates = top_window + group_candidates[len(top_window):]
        selected_communities: set[Any] = set()
        group_selected = 0
        for node_id in group_candidates:
            if group_selected >= int(quotas[group_name]):
                break
            community_id = None if community_by_node is None else community_by_node.get(node_id)
            if community_id is not None and community_id in selected_communities:
                continue
            if node_id in selected_set:
                continue
            selected.append(node_id)
            selected_set.add(node_id)
            selected_communities.add(community_id)
            group_selected += 1
        if group_selected >= int(quotas[group_name]):
            continue
        for node_id in group_candidates:
            if group_selected >= int(quotas[group_name]):
                break
            if node_id in selected_set:
                continue
            selected.append(node_id)
            selected_set.add(node_id)
            group_selected += 1
    for node_id in sorted(global_candidates, key=lambda n: (-float(candidate_scores.get(n, 0.0)), _sort_key(n))):
        if len(selected) >= int(budget):
            break
        if node_id not in selected_set:
            selected.append(node_id)
            selected_set.add(node_id)
    return _normalized_seed_set(selected[:budget])


def _initial_population(
    *,
    graph: nx.Graph,
    candidates: Sequence[Any],
    protected_group_report: ProtectedGroupReport,
    group_by_node: Mapping[Any, str],
    quotas: Mapping[str, int],
    ideal_influences: Mapping[str, float],
    candidate_scores: Mapping[Any, float],
    ris_result: RISGuidanceResult | None,
    config: TargetShortfallRepairConfig,
    diagnostics: dict[str, int],
    rng: np.random.Generator,
) -> list[tuple[Any, ...]]:
    candidates_by_group: dict[str, list[Any]] = {str(group_name): [] for group_name in protected_group_report.group_sizes}
    community_by_node = {}
    try:
        for index, nodes in enumerate(nx.community.greedy_modularity_communities(graph.to_undirected() if graph.is_directed() else graph)):
            for node_id in nodes:
                community_by_node[node_id] = index
    except Exception:
        community_by_node = {}
    for node_id in candidates:
        group_name = str(group_by_node.get(node_id, ""))
        if group_name in candidates_by_group:
            candidates_by_group[group_name].append(node_id)
    for group_name, nodes in candidates_by_group.items():
        nodes.sort(key=lambda node_id: (-float(candidate_scores.get(node_id, 0.0)), _sort_key(node_id)))
    population: list[tuple[Any, ...]] = []
    base = _build_individual_from_quotas(
        quotas=quotas,
        candidates_by_group=candidates_by_group,
        global_candidates=candidates,
        candidate_scores=candidate_scores,
        budget=int(config.budget),
        community_by_node=community_by_node,
    )
    diagnostics["initial_population_strategies"] = [
        "quota_balanced",
        "shortfall_target_heavy",
        "combined_score_topk",
        "random_diverse",
    ]
    population.append(
        _repair_seed_set(
            graph=graph,
            seed_set=base,
            candidates=candidates,
            protected_group_report=protected_group_report,
            group_by_node=group_by_node,
            ideal_influences=ideal_influences,
            candidate_scores=candidate_scores,
            ris_result=ris_result,
            config=config,
            diagnostics=diagnostics,
            quotas=quotas,
        )
    )
    ideal_ranked_groups = sorted(
        quotas,
        key=lambda group_name: (
            -float(ideal_influences.get(group_name, 0.0)),
            -int(protected_group_report.group_sizes.get(group_name, 0)),
            _sort_key(group_name),
        ),
    )
    heavy_quotas = dict(quotas)
    for receiver in ideal_ranked_groups[: max(1, min(2, len(ideal_ranked_groups)))]:
        donor_candidates = [
            group_name
            for group_name in reversed(ideal_ranked_groups)
            if group_name != receiver and int(heavy_quotas.get(group_name, 0)) > 1
        ]
        if not donor_candidates:
            continue
        donor = donor_candidates[0]
        heavy_quotas[donor] -= 1
        heavy_quotas[receiver] = int(heavy_quotas.get(receiver, 0)) + 1
    for strategy_name, seed_nodes, strategy_quotas in (
        (
            "shortfall_target_heavy",
            _build_individual_from_quotas(
                quotas=heavy_quotas,
                candidates_by_group=candidates_by_group,
                global_candidates=candidates,
                candidate_scores=candidate_scores,
                budget=int(config.budget),
                community_by_node=community_by_node,
                rng=rng,
                randomize=True,
            ),
            heavy_quotas,
        ),
        (
            "combined_score_topk",
            _normalized_seed_set(
                sorted(candidates, key=lambda node: (-float(candidate_scores.get(node, 0.0)), _sort_key(node)))[: int(config.budget)]
            ),
            quotas,
        ),
    ):
        if len(population) >= int(config.population_size):
            break
        individual = _repair_seed_set(
            graph=graph,
            seed_set=seed_nodes,
            candidates=candidates,
            protected_group_report=protected_group_report,
            group_by_node=group_by_node,
            ideal_influences=ideal_influences,
            candidate_scores=candidate_scores,
            ris_result=ris_result,
            config=config,
            diagnostics=diagnostics,
            quotas=strategy_quotas,
        )
        if individual not in population:
            population.append(individual)
    while len(population) < int(config.population_size):
        trial_quotas = dict(quotas)
        if len(trial_quotas) > 1:
            donor_choices = [group for group, quota in trial_quotas.items() if int(quota) > 1]
            receiver_choices = list(trial_quotas)
            if donor_choices and receiver_choices:
                donor = str(rng.choice(donor_choices))
                receiver = str(rng.choice(receiver_choices))
                if donor != receiver:
                    trial_quotas[donor] -= 1
                    trial_quotas[receiver] += 1
        individual = _build_individual_from_quotas(
            quotas=trial_quotas,
            candidates_by_group=candidates_by_group,
            global_candidates=candidates,
            candidate_scores=candidate_scores,
            budget=int(config.budget),
            community_by_node=community_by_node,
            rng=rng,
            randomize=True,
        )
        if individual in population:
            sampled = rng.choice(list(candidates), size=min(int(config.budget), len(candidates)), replace=False)
            individual = _normalized_seed_set(list(sampled))
        individual = _repair_seed_set(
            graph=graph,
            seed_set=individual,
            candidates=candidates,
            protected_group_report=protected_group_report,
            group_by_node=group_by_node,
            ideal_influences=ideal_influences,
            candidate_scores=candidate_scores,
            ris_result=ris_result,
            config=config,
            diagnostics=diagnostics,
            quotas=trial_quotas,
        )
        population.append(individual)
    return population


def _select_parent(
    population: Sequence[tuple[Any, ...]],
    evaluations: Mapping[tuple[Any, ...], TargetShortfallProxyEvaluation],
    config: TargetShortfallRepairConfig,
    rng: np.random.Generator,
) -> tuple[Any, ...]:
    size = min(max(1, int(config.tournament_size)), len(population))
    indices = rng.choice(len(population), size=size, replace=False)
    contenders = [population[int(index)] for index in indices]
    return max(contenders, key=lambda individual: _priority_tuple(evaluations[individual]))


def _crossover(
    *,
    parent_a: tuple[Any, ...],
    parent_b: tuple[Any, ...],
    current_eval: TargetShortfallProxyEvaluation,
    candidates: Sequence[Any],
    quotas: Mapping[str, int],
    candidate_scores: Mapping[Any, float],
    group_by_node: Mapping[Any, str],
    config: TargetShortfallRepairConfig,
    rng: np.random.Generator,
) -> tuple[Any, ...]:
    merged = list(dict.fromkeys([*parent_a, *parent_b]))
    rng.shuffle(merged)
    selected: list[Any] = []
    selected_set: set[Any] = set()
    counts: dict[str, int] = {group_name: 0 for group_name in quotas}
    ranked_merged = sorted(
        merged,
        key=lambda node_id: (
            -float(current_eval.shortfall_ratio_by_group.get(str(group_by_node.get(node_id, "")), 0.0)),
            -float(candidate_scores.get(node_id, 0.0)),
            _sort_key(node_id),
        ),
    )
    for node_id in ranked_merged:
        group_name = str(group_by_node.get(node_id, ""))
        quota = int(quotas.get(group_name, 0))
        allowance = quota + (0 if current_eval.shortfall_ratio_by_group.get(group_name, 0.0) <= 1e-9 else 1)
        if counts.get(group_name, 0) >= max(1, allowance):
            continue
        selected.append(node_id)
        selected_set.add(node_id)
        counts[group_name] = int(counts.get(group_name, 0)) + 1
        if len(selected) >= int(config.budget):
            break
    for node_id in sorted(candidates, key=lambda n: (-float(candidate_scores.get(n, 0.0)), _sort_key(n))):
        if len(selected) >= int(config.budget):
            break
        if node_id not in selected_set:
            selected.append(node_id)
            selected_set.add(node_id)
    return _normalized_seed_set(selected)


def _mutate(
    *,
    graph: nx.Graph,
    individual: tuple[Any, ...],
    candidates: Sequence[Any],
    protected_group_report: ProtectedGroupReport,
    group_by_node: Mapping[Any, str],
    ideal_influences: Mapping[str, float],
    candidate_scores: Mapping[Any, float],
    ris_result: RISGuidanceResult | None,
    config: TargetShortfallRepairConfig,
    diagnostics: dict[str, int],
    quotas: Mapping[str, int] | None = None,
) -> tuple[Any, ...]:
    current_eval = _evaluate_proxy(
        graph=graph,
        seed_set=individual,
        protected_group_report=protected_group_report,
        group_by_node=group_by_node,
        ideal_influences=ideal_influences,
        ris_result=ris_result,
        config=config,
    )
    strength = int(config.mutation_strength or max(1, math.ceil(float(config.budget) * 0.05)))
    current = list(individual)
    for _ in range(strength):
        if not current:
            break
        removals = _rank_removals(
            seed_set=_normalized_seed_set(current),
            current_eval=current_eval,
            candidate_scores=candidate_scores,
            group_by_node=group_by_node,
            ideal_influences=ideal_influences,
            ris_result=ris_result,
            quotas=quotas,
        )
        node_to_remove = None
        for removal_candidate in removals:
            node_to_remove = removal_candidate
            break
        if node_to_remove is None:
            break
        current = [node_id for node_id in current if node_id != node_to_remove]
        partial_set = set(current)
        additions = _rank_candidates(
            candidates=candidates,
            seed_set=partial_set,
            current_eval=current_eval,
            candidate_scores=candidate_scores,
            group_by_node=group_by_node,
            ris_result=ris_result,
            limit=max(10, int(config.repair_candidate_limit)),
            quotas=quotas,
        )
        if additions:
            node_to_add = additions[0]
            current.append(node_to_add)
            if current_eval.shortfall_ratio_by_group.get(str(group_by_node.get(node_to_add, "")), 0.0) > 1e-9:
                diagnostics["mutations_to_below_target_groups"] = int(diagnostics.get("mutations_to_below_target_groups", 0)) + 1
            if current_eval.shortfall_ratio_by_group.get(str(group_by_node.get(node_to_remove, "")), 0.0) <= 1e-9:
                diagnostics["mutations_removed_from_met_groups"] = int(diagnostics.get("mutations_removed_from_met_groups", 0)) + 1
            if current_eval.overcoverage_ratio_by_group.get(str(group_by_node.get(node_to_remove, "")), 0.0) > 0.05:
                diagnostics["mutations_from_overcovered_groups"] = int(diagnostics.get("mutations_from_overcovered_groups", 0)) + 1
    mutated = _normalized_seed_set(current)
    after_eval = _evaluate_proxy(
        graph=graph,
        seed_set=mutated,
        protected_group_report=protected_group_report,
        group_by_node=group_by_node,
        ideal_influences=ideal_influences,
        ris_result=ris_result,
        config=config,
    )
    if (
        _target_first_acceptance_reason(current_eval, after_eval, allow_spread_only=False) is not None
        or _is_better(after_eval, current_eval)
        or (
            after_eval.target_coverage_ratio >= current_eval.target_coverage_ratio
            and after_eval.dcv_shortfall <= current_eval.dcv_shortfall + 1e-6
        )
    ):
        if after_eval.target_coverage_ratio > current_eval.target_coverage_ratio:
            diagnostics["mutations_improved_target_coverage"] = int(diagnostics.get("mutations_improved_target_coverage", 0)) + 1
        return mutated
    diagnostics["mutations_rejected_target_loss"] = int(diagnostics.get("mutations_rejected_target_loss", 0)) + 1
    return individual


def _local_search(
    *,
    graph: nx.Graph,
    individual: tuple[Any, ...],
    candidates: Sequence[Any],
    protected_group_report: ProtectedGroupReport,
    group_by_node: Mapping[Any, str],
    ideal_influences: Mapping[str, float],
    candidate_scores: Mapping[Any, float],
    ris_result: RISGuidanceResult | None,
    config: TargetShortfallRepairConfig,
    diagnostics: dict[str, int],
    quotas: Mapping[str, int] | None = None,
) -> tuple[Any, ...]:
    current = individual
    for _ in range(max(0, int(config.local_search_steps))):
        current_eval = _evaluate_proxy(
            graph=graph,
            seed_set=current,
            protected_group_report=protected_group_report,
            group_by_node=group_by_node,
            ideal_influences=ideal_influences,
            ris_result=ris_result,
            config=config,
        )
        removals = _rank_removals(
            seed_set=current,
            current_eval=current_eval,
            candidate_scores=candidate_scores,
            group_by_node=group_by_node,
            ideal_influences=ideal_influences,
            ris_result=ris_result,
            quotas=quotas,
        )[: max(1, min(10, len(current)))]
        additions = _rank_candidates(
            candidates=candidates,
            seed_set=set(current),
            current_eval=current_eval,
            candidate_scores=candidate_scores,
            group_by_node=group_by_node,
            ris_result=ris_result,
            limit=max(1, int(config.repair_candidate_limit)),
            quotas=quotas,
        )
        accepted: tuple[Any, ...] | None = None
        accepted_eval: TargetShortfallProxyEvaluation | None = None
        current_counts = _seed_group_counts(current, group_by_node)
        for node_to_remove in removals:
            removed_group = str(group_by_node.get(node_to_remove, ""))
            if quotas is not None and int(current_counts.get(removed_group, 0)) <= int(quotas.get(removed_group, 0)):
                if float(current_eval.overcoverage_ratio_by_group.get(removed_group, 0.0)) < 0.05:
                    continue
            retained = [node_id for node_id in current if node_id != node_to_remove]
            for node_to_add in additions:
                if node_to_add in retained:
                    continue
                diagnostics["local_search_attempts"] = int(diagnostics.get("local_search_attempts", 0)) + 1
                trial = _normalized_seed_set([*retained, node_to_add])
                if len(trial) != int(config.budget):
                    continue
                trial_eval = _evaluate_proxy(
                    graph=graph,
                    seed_set=trial,
                    protected_group_report=protected_group_report,
                    group_by_node=group_by_node,
                    ideal_influences=ideal_influences,
                    ris_result=ris_result,
                    config=config,
                )
                if trial_eval.target_coverage_ratio < current_eval.target_coverage_ratio - 1e-12:
                    diagnostics["local_swaps_rejected_target_loss"] = int(diagnostics.get("local_swaps_rejected_target_loss", 0)) + 1
                    diagnostics["swaps_rejected_due_to_target_loss"] = int(diagnostics.get("swaps_rejected_due_to_target_loss", 0)) + 1
                    continue
                if _groups_losing_target(current_eval, trial_eval):
                    diagnostics["local_swaps_rejected_target_loss"] = int(diagnostics.get("local_swaps_rejected_target_loss", 0)) + 1
                    diagnostics["swaps_rejected_due_to_target_loss"] = int(diagnostics.get("swaps_rejected_due_to_target_loss", 0)) + 1
                    continue
                if trial_eval.dcv_shortfall > current_eval.dcv_shortfall + 1e-6:
                    diagnostics["local_swaps_rejected_dcv_worsening"] = int(diagnostics.get("local_swaps_rejected_dcv_worsening", 0)) + 1
                    continue
                reason = _target_first_acceptance_reason(
                    current_eval,
                    trial_eval,
                    allow_spread_only=True,
                    dcv_worsen_tolerance=1e-6,
                )
                if reason == "target_coverage":
                    diagnostics["local_swaps_accepted_target_coverage"] = int(diagnostics.get("local_swaps_accepted_target_coverage", 0)) + 1
                    accepted = trial
                    accepted_eval = trial_eval
                    break
                if reason == "shortfall":
                    diagnostics["local_swaps_accepted_shortfall"] = int(diagnostics.get("local_swaps_accepted_shortfall", 0)) + 1
                    accepted = trial
                    accepted_eval = trial_eval
                    break
                if reason == "below_target_progress":
                    diagnostics["local_swaps_accepted_below_target_progress"] = int(diagnostics.get("local_swaps_accepted_below_target_progress", 0)) + 1
                    accepted = trial
                    accepted_eval = trial_eval
                    break
                if reason == "f_score":
                    diagnostics["local_swaps_accepted_fscore"] = int(diagnostics.get("local_swaps_accepted_fscore", 0)) + 1
                    accepted = trial
                    accepted_eval = trial_eval
                    break
                if reason == "spread_only":
                    diagnostics["local_swaps_accepted_spread_only"] = int(diagnostics.get("local_swaps_accepted_spread_only", 0)) + 1
                    accepted = trial
                    accepted_eval = trial_eval
                    break
            if accepted is not None:
                break
        if accepted is None or accepted_eval is None:
            break
        diagnostics["local_search_improvements"] = int(diagnostics.get("local_search_improvements", 0)) + 1
        current = accepted
    return current


def run_target_shortfall_repair_memetic_ris(
    *,
    graph: nx.Graph,
    protected_group_report: ProtectedGroupReport,
    ideal_influences: Mapping[str, float],
    candidate_nodes: Sequence[Any],
    candidate_scores: Mapping[Any, float],
    ris_result: RISGuidanceResult | None,
    config: TargetShortfallRepairConfig,
) -> TargetShortfallRepairResult:
    """Run the target-shortfall-first memetic search."""

    start = perf_counter()
    if int(config.budget) < 1:
        raise ValueError("budget must be positive.")
    if len(candidate_nodes) < int(config.budget):
        raise ValueError("candidate_nodes must contain at least budget unique nodes.")
    group_by_node = _group_by_node(protected_group_report)
    candidates = _normalized_seed_set(candidate_nodes)
    seed_factor = _seed_spread_factor(graph, float(config.propagation_probability))
    quotas = compute_target_shortfall_group_quotas(
        budget=int(config.budget),
        group_sizes={str(k): int(v) for k, v in protected_group_report.group_sizes.items()},
        ideal_influences={str(k): float(v) for k, v in ideal_influences.items()},
        seed_factor=seed_factor,
    )
    diagnostics: dict[str, int | float | str | dict[str, int] | list[str]] = {
        "target_shortfall_method": "target_shortfall_repair_memetic_ris",
        "group_seed_quotas": dict(quotas),
        "seed_spread_factor": float(seed_factor),
        "shortfall_repair_rounds": int(config.repair_rounds),
        "shortfall_repair_candidate_limit": int(config.repair_candidate_limit),
        "shortfall_repair_aggressive": bool(config.repair_aggressive),
        "repair_attempts": 0,
        "repair_successes": 0,
        "repairs_improved_target_coverage": 0,
        "repairs_reduced_total_shortfall": 0,
        "mutations_to_below_target_groups": 0,
        "mutations_removed_from_met_groups": 0,
        "mutations_from_overcovered_groups": 0,
        "mutations_improved_target_coverage": 0,
        "mutations_rejected_target_loss": 0,
        "children_repaired_for_shortfall": 0,
        "children_repaired_for_target_coverage": 0,
        "crossover_children_improved_target_coverage": 0,
        "crossover_children_improved_dcv_shortfall": 0,
        "local_search_attempts": 0,
        "local_search_improvements": 0,
        "local_swaps_accepted_target_coverage": 0,
        "local_swaps_accepted_shortfall": 0,
        "local_swaps_accepted_below_target_progress": 0,
        "local_swaps_accepted_fscore": 0,
        "local_swaps_accepted_spread_only": 0,
        "local_swaps_rejected_target_loss": 0,
        "local_swaps_rejected_dcv_worsening": 0,
        "seeds_removed_from_overcovered_groups": 0,
        "swaps_rejected_due_to_target_loss": 0,
    }
    rng = np.random.default_rng(int(config.random_seed))
    population = _initial_population(
        graph=graph,
        candidates=candidates,
        protected_group_report=protected_group_report,
        group_by_node=group_by_node,
        quotas=quotas,
        ideal_influences=ideal_influences,
        candidate_scores=candidate_scores,
        ris_result=ris_result,
        config=config,
        diagnostics=diagnostics,
        rng=rng,
    )
    eval_cache: dict[tuple[Any, ...], TargetShortfallProxyEvaluation] = {}

    def evaluate(individual: tuple[Any, ...]) -> TargetShortfallProxyEvaluation:
        if individual not in eval_cache:
            eval_cache[individual] = _evaluate_proxy(
                graph=graph,
                seed_set=individual,
                protected_group_report=protected_group_report,
                group_by_node=group_by_node,
                ideal_influences=ideal_influences,
                ris_result=ris_result,
                config=config,
            )
        return eval_cache[individual]

    initial_evals = [evaluate(individual) for individual in population]
    best_initial = max(initial_evals, key=_priority_tuple)
    diagnostics["initial_quota_individual_seed_counts_by_group"] = _seed_group_counts(population[0], group_by_node)
    diagnostics["best_initial_target_coverage"] = float(best_initial.target_coverage_ratio)
    diagnostics["best_initial_dcv_shortfall"] = float(best_initial.dcv_shortfall)
    diagnostics["best_initial_total_shortfall"] = float(best_initial.total_shortfall)
    diagnostics["best_initial_f_score"] = float(best_initial.f_score)
    diagnostics["best_initial_fitness"] = float(best_initial.fitness)
    diagnostics["best_initial_seed_counts_by_group"] = _seed_group_counts(best_initial.seed_set, group_by_node)

    best = best_initial
    best_generation = 0
    elite_count = max(1, int(math.ceil(float(config.population_size) * float(config.elite_fraction))))
    local_elite_count = max(1, int(math.ceil(float(config.population_size) * float(config.local_search_elite_fraction))))

    for generation in range(1, int(config.generations) + 1):
        evaluations = {individual: evaluate(individual) for individual in population}
        ordered = sorted(population, key=lambda individual: _priority_tuple(evaluations[individual]), reverse=True)
        if _priority_tuple(evaluations[ordered[0]]) > _priority_tuple(best):
            best = evaluations[ordered[0]]
            best_generation = generation
        new_population = list(ordered[:elite_count])
        while len(new_population) < int(config.population_size):
            parent_a = _select_parent(ordered, evaluations, config, rng)
            parent_b = _select_parent(ordered, evaluations, config, rng)
            parent_eval = evaluations[parent_a]
            if float(rng.random()) <= float(config.crossover_rate):
                child = _crossover(
                    parent_a=parent_a,
                    parent_b=parent_b,
                    current_eval=parent_eval,
                    candidates=candidates,
                    quotas=quotas,
                    candidate_scores=candidate_scores,
                    group_by_node=group_by_node,
                    config=config,
                    rng=rng,
                )
                before_repair = _evaluate_proxy(
                    graph=graph,
                    seed_set=child,
                    protected_group_report=protected_group_report,
                    group_by_node=group_by_node,
                    ideal_influences=ideal_influences,
                    ris_result=ris_result,
                    config=config,
                )
                repaired = _repair_seed_set(
                    graph=graph,
                    seed_set=child,
                    candidates=candidates,
                    protected_group_report=protected_group_report,
                    group_by_node=group_by_node,
                    ideal_influences=ideal_influences,
                    candidate_scores=candidate_scores,
                    ris_result=ris_result,
                    config=config,
                    diagnostics=diagnostics,
                    quotas=quotas,
                )
                after_repair = _evaluate_proxy(
                    graph=graph,
                    seed_set=repaired,
                    protected_group_report=protected_group_report,
                    group_by_node=group_by_node,
                    ideal_influences=ideal_influences,
                    ris_result=ris_result,
                    config=config,
                )
                if repaired != child:
                    diagnostics["children_repaired_for_shortfall"] = int(diagnostics.get("children_repaired_for_shortfall", 0)) + 1
                if after_repair.target_coverage_ratio > before_repair.target_coverage_ratio:
                    diagnostics["crossover_children_improved_target_coverage"] = int(diagnostics.get("crossover_children_improved_target_coverage", 0)) + 1
                    diagnostics["children_repaired_for_target_coverage"] = int(diagnostics.get("children_repaired_for_target_coverage", 0)) + 1
                if after_repair.dcv_shortfall < before_repair.dcv_shortfall - 1e-12:
                    diagnostics["crossover_children_improved_dcv_shortfall"] = int(diagnostics.get("crossover_children_improved_dcv_shortfall", 0)) + 1
                child = repaired
            else:
                child = parent_a
            if float(rng.random()) <= float(config.mutation_rate):
                child = _mutate(
                    graph=graph,
                    individual=child,
                    candidates=candidates,
                    protected_group_report=protected_group_report,
                    group_by_node=group_by_node,
                    ideal_influences=ideal_influences,
                    candidate_scores=candidate_scores,
                    ris_result=ris_result,
                    config=config,
                    diagnostics=diagnostics,
                    quotas=quotas,
                )
                child = _repair_seed_set(
                    graph=graph,
                    seed_set=child,
                    candidates=candidates,
                    protected_group_report=protected_group_report,
                    group_by_node=group_by_node,
                    ideal_influences=ideal_influences,
                    candidate_scores=candidate_scores,
                    ris_result=ris_result,
                    config=config,
                    diagnostics=diagnostics,
                    quotas=quotas,
                )
            new_population.append(child)
        refined_population = []
        interim_evals = {individual: evaluate(individual) for individual in new_population}
        for index, individual in enumerate(sorted(new_population, key=lambda ind: _priority_tuple(interim_evals[ind]), reverse=True)):
            if index < local_elite_count and int(config.local_search_steps) > 0:
                individual = _local_search(
                    graph=graph,
                    individual=individual,
                    candidates=candidates,
                    protected_group_report=protected_group_report,
                    group_by_node=group_by_node,
                    ideal_influences=ideal_influences,
                    candidate_scores=candidate_scores,
                    ris_result=ris_result,
                    config=config,
                    diagnostics=diagnostics,
                    quotas=quotas,
                )
                individual = _repair_seed_set(
                    graph=graph,
                    seed_set=individual,
                    candidates=candidates,
                    protected_group_report=protected_group_report,
                    group_by_node=group_by_node,
                    ideal_influences=ideal_influences,
                    candidate_scores=candidate_scores,
                    ris_result=ris_result,
                    config=config,
                    diagnostics=diagnostics,
                    quotas=quotas,
                )
            refined_population.append(individual)
        population = refined_population[: int(config.population_size)]

    final_evaluations = [evaluate(individual) for individual in population]
    final_best = max([best, *final_evaluations], key=_priority_tuple)
    if _priority_tuple(final_best) > _priority_tuple(best):
        best = final_best
        best_generation = int(config.generations)

    final_seed_set = _repair_seed_set(
        graph=graph,
        seed_set=best.seed_set,
        candidates=candidates,
        protected_group_report=protected_group_report,
        group_by_node=group_by_node,
        ideal_influences=ideal_influences,
        candidate_scores=candidate_scores,
        ris_result=ris_result,
        config=config,
        diagnostics=diagnostics,
        quotas=quotas,
    )
    final_eval = _evaluate_proxy(
        graph=graph,
        seed_set=final_seed_set,
        protected_group_report=protected_group_report,
        group_by_node=group_by_node,
        ideal_influences=ideal_influences,
        ris_result=ris_result,
        config=config,
    )
    if _priority_tuple(final_eval) > _priority_tuple(best):
        best = final_eval

    diagnostics["final_proxy_target_coverage"] = float(best.target_coverage_ratio)
    diagnostics["final_proxy_dcv_shortfall"] = float(best.dcv_shortfall)
    diagnostics["final_proxy_f_score"] = float(best.f_score)
    diagnostics["final_proxy_total_shortfall"] = float(best.total_shortfall)
    diagnostics["final_proxy_overcoverage_waste"] = float(best.overcoverage_waste)
    diagnostics["overcoverage_by_group"] = dict(best.overcoverage_ratio_by_group)
    diagnostics["final_proxy_group_influence"] = dict(best.group_influence)
    diagnostics["final_proxy_shortfall_ratio_by_group"] = dict(best.shortfall_ratio_by_group)
    diagnostics["final_proxy_seed_counts_by_group"] = _seed_group_counts(best.seed_set, group_by_node)
    diagnostics["fitness_weights"] = {
        "normalized_spread": float(config.weights.normalized_spread),
        "target_coverage_ratio": float(config.weights.target_coverage_ratio),
        "mf": float(config.weights.mf),
        "dcv_shortfall": float(config.weights.dcv_shortfall),
        "total_shortfall": float(config.weights.total_shortfall),
        "dcv": float(config.weights.dcv),
        "f_score": float(config.weights.f_score),
        "overcoverage_waste": float(config.weights.overcoverage_waste),
    }
    return TargetShortfallRepairResult(
        seed_set=best.seed_set,
        diagnostics=dict(diagnostics),
        best_generation=int(best_generation),
        final_fitness=float(best.fitness),
        runtime_seconds=float(perf_counter() - start),
    )
