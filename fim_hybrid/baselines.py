"""Baseline seed selection methods for FIM experiments."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .community_detection import CommunityDetectionResult


def top_k_by_feature(feature_frame: pd.DataFrame, feature_name: str, budget: int) -> list[object]:
    """Select the top-k nodes by a numeric feature."""

    if feature_name not in feature_frame.columns:
        raise KeyError(f"Feature '{feature_name}' is not present in the feature frame.")

    # This gives simple ranking baselines such as degree or PageRank.
    ranked = feature_frame.sort_values(feature_name, ascending=False)
    return ranked.head(budget)["node_id"].tolist()


def random_seed_set(nodes: Sequence[object], budget: int, seed: int = 42) -> list[object]:
    """Uniform random seed-set baseline."""

    if budget > len(nodes):
        raise ValueError("budget cannot exceed the number of available nodes.")

    rng = np.random.default_rng(seed)
    # Random is useful as a sanity-check lower baseline.
    chosen = rng.choice(np.asarray(list(nodes), dtype=object), size=budget, replace=False)
    return list(chosen.tolist())


def community_round_robin(
    feature_frame: pd.DataFrame,
    community_result: CommunityDetectionResult,
    budget: int,
    ranking_feature: str = "pagerank",
) -> list[object]:
    """Pick nodes in round-robin order across communities using a ranking feature."""

    if ranking_feature not in feature_frame.columns:
        raise KeyError(f"Feature '{ranking_feature}' is not present in the feature frame.")

    # Rank nodes inside each community first.
    per_community = {}
    for comm_idx, nodes in enumerate(community_result.communities):
        community_frame = feature_frame.loc[feature_frame["node_id"].isin(nodes)]
        community_frame = community_frame.sort_values(ranking_feature, ascending=False)
        per_community[comm_idx] = community_frame["node_id"].tolist()

    seed_set: list[object] = []
    # Then pick one node from each community in turn to encourage coverage.
    while len(seed_set) < budget:
        added = False
        for comm_idx in range(len(community_result.communities)):
            if not per_community[comm_idx]:
                continue
            candidate = per_community[comm_idx].pop(0)
            if candidate in seed_set:
                continue
            seed_set.append(candidate)
            added = True
            if len(seed_set) == budget:
                break
        if not added:
            break

    return seed_set
