"""Phase 3 baseline methods for Fair Influence Maximization."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Sequence

import networkx as nx
import numpy as np

from .community_detection import CommunityDetectionResult
from .data_loader import LoadedDataset, ProtectedGroupReport
from .diffusion import DEFAULT_DIFFUSION_MODEL
from .evaluation import evaluate_seed_set


@dataclass(slots=True)
class BaselineResult:
    """End-to-end baseline evaluation output."""

    method: str
    seed_set: tuple[Any, ...]
    total_spread_mean: float
    total_spread_std: float
    mf: float
    soft_mf: float | None
    dcv: float
    f_score: float
    runtime_seconds: float
    group_spread: dict[str, float]
    normalized_group_spread: dict[str, float]


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _validate_budget(graph: nx.Graph, budget: int) -> None:
    if budget < 1:
        raise ValueError("budget must be at least 1.")
    if budget > graph.number_of_nodes():
        raise ValueError("budget cannot exceed the number of graph nodes.")


def _rank_nodes(scores: dict[Any, float], budget: int) -> list[Any]:
    ranked_nodes = sorted(
        scores,
        key=lambda node: (-float(scores[node]), _sort_key(node)),
    )
    return ranked_nodes[:budget]


def select_random_seed_set(
    graph: nx.Graph,
    budget: int,
    random_seed: int = 42,
) -> list[Any]:
    """Select a uniformly random seed set without replacement."""

    _validate_budget(graph, budget)
    ordered_nodes = sorted(graph.nodes(), key=_sort_key)
    rng = np.random.default_rng(random_seed)
    chosen = rng.choice(np.asarray(ordered_nodes, dtype=object), size=budget, replace=False)
    return list(chosen.tolist())


def select_degree_seed_set(graph: nx.Graph, budget: int) -> list[Any]:
    """Select top-degree nodes; use out-degree for directed graphs."""

    _validate_budget(graph, budget)
    degree_view = graph.out_degree() if graph.is_directed() else graph.degree()
    degree_scores = {node: float(score) for node, score in degree_view}
    return _rank_nodes(degree_scores, budget)


def select_pagerank_seed_set(graph: nx.Graph, budget: int) -> list[Any]:
    """Select top PageRank nodes."""

    _validate_budget(graph, budget)
    pagerank_scores = nx.pagerank(graph)
    return _rank_nodes(pagerank_scores, budget)


def _detect_round_robin_communities(graph: nx.Graph, random_seed: int) -> list[list[Any]]:
    work_graph = graph if not graph.is_directed() else graph.to_undirected()
    communities = nx.community.louvain_communities(work_graph, seed=random_seed)
    normalized = [sorted(list(community), key=_sort_key) for community in communities if community]
    return sorted(normalized, key=lambda community: _sort_key(community[0]))


def select_community_round_robin_seed_set(
    graph: nx.Graph,
    budget: int,
    random_seed: int = 42,
    community_result: CommunityDetectionResult | None = None,
) -> list[Any]:
    """Select seeds by alternating across Louvain communities using PageRank ranking."""

    _validate_budget(graph, budget)
    pagerank_scores = nx.pagerank(graph)
    if community_result is None:
        communities = _detect_round_robin_communities(graph, random_seed)
    else:
        communities = [
            list(community_result.communities[community_id])
            for community_id in sorted(community_result.communities)
            if community_result.communities[community_id]
        ]

    per_community_rankings: list[list[Any]] = []
    for community_nodes in communities:
        ranked_nodes = sorted(
            community_nodes,
            key=lambda node: (-float(pagerank_scores[node]), _sort_key(node)),
        )
        per_community_rankings.append(ranked_nodes)

    seed_set: list[Any] = []
    while len(seed_set) < budget:
        added_any = False
        for ranked_nodes in per_community_rankings:
            if not ranked_nodes:
                continue
            candidate = ranked_nodes.pop(0)
            if candidate in seed_set:
                continue
            seed_set.append(candidate)
            added_any = True
            if len(seed_set) == budget:
                break
        if not added_any:
            break

    if len(seed_set) != budget:
        raise RuntimeError("community_round_robin could not select the requested budget.")
    return seed_set


_BASELINE_SELECTORS: dict[str, Callable[[nx.Graph, int, int], list[Any]]] = {
    "random": select_random_seed_set,
    "degree": lambda graph, budget, random_seed: select_degree_seed_set(graph, budget),
    "pagerank": lambda graph, budget, random_seed: select_pagerank_seed_set(graph, budget),
    "community_round_robin": select_community_round_robin_seed_set,
}


def select_baseline_seed_set(
    dataset: LoadedDataset,
    method: str,
    budget: int,
    community_result: CommunityDetectionResult | None = None,
    random_seed: int = 42,
) -> tuple[Any, ...]:
    """Select a baseline seed set without running diffusion or fairness evaluation."""

    method_key = method.lower()
    if method_key not in _BASELINE_SELECTORS:
        supported = ", ".join(sorted(_BASELINE_SELECTORS))
        raise ValueError(f"Unsupported baseline method '{method}'. Supported methods: {supported}.")

    selector = _BASELINE_SELECTORS[method_key]
    if method_key == "community_round_robin":
        selected_seeds = select_community_round_robin_seed_set(
            dataset.graph,
            budget,
            random_seed=random_seed,
            community_result=community_result,
        )
    else:
        selected_seeds = selector(dataset.graph, budget, random_seed)

    return tuple(sorted(selected_seeds, key=_sort_key))


def run_baseline(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    method: str,
    budget: int,
    propagation_probability: float = 0.01,
    mc_runs: int = 100,
    lambda_weight: float = 0.5,
    community_result: CommunityDetectionResult | None = None,
    random_seed: int = 42,
    diffusion_model: str = DEFAULT_DIFFUSION_MODEL,
) -> BaselineResult:
    """Select a seed set with one baseline and evaluate it with Phase 2 metrics."""

    start = perf_counter()
    method_key = method.lower()
    selected_seeds = select_baseline_seed_set(
        dataset=dataset,
        method=method_key,
        budget=budget,
        community_result=community_result,
        random_seed=random_seed,
    )
    evaluation = evaluate_seed_set(
        dataset=dataset,
        protected_group_report=protected_group_report,
        seed_set=selected_seeds,
        propagation_probability=propagation_probability,
        mc_runs=mc_runs,
        random_seed=random_seed,
        lambda_weight=lambda_weight,
        include_soft_mf=True,
        diffusion_model=diffusion_model,
    )
    runtime_seconds = perf_counter() - start

    return BaselineResult(
        method=method_key,
        seed_set=evaluation.seed_set,
        total_spread_mean=evaluation.total_spread_mean,
        total_spread_std=evaluation.total_spread_std,
        mf=evaluation.fairness.mf,
        soft_mf=evaluation.fairness.soft_mf,
        dcv=evaluation.fairness.dcv,
        f_score=evaluation.f_score,
        runtime_seconds=runtime_seconds,
        group_spread=evaluation.fairness.group_spread,
        normalized_group_spread=evaluation.fairness.normalized_group_spread,
    )


def run_baselines(
    dataset: LoadedDataset,
    protected_group_report: ProtectedGroupReport,
    methods: Sequence[str],
    budget: int,
    propagation_probability: float = 0.01,
    mc_runs: int = 100,
    lambda_weight: float = 0.5,
    community_result: CommunityDetectionResult | None = None,
    random_seed: int = 42,
    diffusion_model: str = DEFAULT_DIFFUSION_MODEL,
) -> list[BaselineResult]:
    """Run multiple baselines end-to-end using the Phase 2 evaluation stack."""

    return [
        run_baseline(
            dataset=dataset,
            protected_group_report=protected_group_report,
            method=method,
            budget=budget,
            propagation_probability=propagation_probability,
            mc_runs=mc_runs,
            lambda_weight=lambda_weight,
            community_result=community_result,
            random_seed=random_seed,
            diffusion_model=diffusion_model,
        )
        for method in methods
    ]
