"""CLI for Phase 4 community-layer verification."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.community_detection import (  # noqa: E402
    detect_communities,
    format_community_debug_report,
    get_community_stats,
    sample_community,
    sample_node_from_community,
    score_communities_by_size,
)
from fim_hybrid.config import DatasetConfig  # noqa: E402
from fim_hybrid.data_loader import load_dataset, resolve_builtin_dataset  # noqa: E402


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4 community-layer verification.")
    parser.add_argument("--dataset", default="graph_spa_500_0", help="Built-in dataset name.")
    parser.add_argument("--edge-path", default=None, help="Path to a simple edge-list file.")
    parser.add_argument("--pickle-path", default=None, help="Path to a pickled NetworkX graph.")
    parser.add_argument("--attribute-path", default=None, help="Path to a node-attribute CSV file.")
    parser.add_argument("--delimiter", default=None, help="Optional edge-list delimiter override.")
    parser.add_argument("--undirected", action="store_true", help="Load explicit edge lists as undirected graphs.")
    parser.add_argument("--node-id-column", default="node_id", help="Node id column name in the attribute CSV.")
    parser.add_argument("--method", default="louvain", choices=["louvain", "leiden"], help="Community detection method.")
    parser.add_argument("--resolution", type=float, default=1.0, help="Community resolution parameter.")
    parser.add_argument("--weight-attribute", default=None, help="Optional edge-weight attribute.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--community-count",
        type=int,
        default=3,
        help="Number of communities to choose for the helper demo.",
    )
    parser.add_argument(
        "--nodes-per-community",
        type=int,
        default=1,
        help="Number of candidate nodes to sample from each chosen community.",
    )
    return parser.parse_args()


def build_dataset_config(args: argparse.Namespace) -> DatasetConfig:
    if args.edge_path or args.pickle_path:
        return DatasetConfig(
            name=args.dataset,
            edge_path=_resolve_repo_path(args.edge_path),
            pickle_path=_resolve_repo_path(args.pickle_path),
            attribute_path=_resolve_repo_path(args.attribute_path),
            directed=not args.undirected,
            delimiter=args.delimiter,
            node_id_column=args.node_id_column,
        )

    config = resolve_builtin_dataset(args.dataset, ROOT)
    if args.attribute_path is not None:
        config.attribute_path = _resolve_repo_path(args.attribute_path)
        config.node_id_column = args.node_id_column
    return config


def main() -> None:
    args = parse_args()
    dataset = load_dataset(build_dataset_config(args))
    result = detect_communities(
        graph=dataset.graph,
        method=args.method,
        seed=args.random_seed,
        resolution=args.resolution,
        weight_attribute=args.weight_attribute,
    )

    print(f"dataset: {dataset.name}")
    print(format_community_debug_report(result))

    stats = get_community_stats(result.communities)
    scores = score_communities_by_size(result.communities)
    rng = np.random.default_rng(args.random_seed)
    sample_count = min(args.community_count, stats.num_communities)

    sampled_pairs: list[tuple[int, list[object]]] = []
    for _ in range(sample_count):
        community_id = sample_community(result.communities, stats.community_sizes, rng)
        node_ids = [
            sample_node_from_community(result.communities[community_id], rng)
            for _ in range(args.nodes_per_community)
        ]
        sampled_pairs.append((community_id, node_ids))

    print(f"community_scores: {scores}")
    print("sampled_community_node_pairs:")
    for community_id, node_ids in sampled_pairs:
        print(f"  - community {community_id}: {node_ids}")


if __name__ == "__main__":
    main()
