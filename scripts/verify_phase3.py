"""CLI for Phase 3 baseline verification."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.baselines import run_baselines  # noqa: E402
from fim_hybrid.config import DatasetConfig  # noqa: E402
from fim_hybrid.data_loader import load_dataset, resolve_builtin_dataset, verify_protected_groups  # noqa: E402


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 baseline verification.")
    parser.add_argument("--dataset", default="graph_spa_500_0", help="Built-in dataset name.")
    parser.add_argument("--protected-attribute", required=True, help="Protected attribute to verify.")
    parser.add_argument("--edge-path", default=None, help="Path to a simple edge-list file.")
    parser.add_argument("--pickle-path", default=None, help="Path to a pickled NetworkX graph.")
    parser.add_argument("--attribute-path", default=None, help="Path to a node-attribute CSV file.")
    parser.add_argument("--delimiter", default=None, help="Optional edge-list delimiter override.")
    parser.add_argument("--undirected", action="store_true", help="Load explicit edge lists as undirected graphs.")
    parser.add_argument("--node-id-column", default="node_id", help="Node id column name in the attribute CSV.")
    parser.add_argument("--budget", type=int, required=True, help="Seed budget.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["random", "degree", "pagerank", "community_round_robin"],
        help="Baseline methods to run.",
    )
    parser.add_argument("--propagation-prob", type=float, default=0.01, help="IC edge propagation probability.")
    parser.add_argument("--mc-runs", type=int, default=100, help="Monte Carlo run count.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
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
    protected_group_report = verify_protected_groups(dataset, args.protected_attribute)
    results = run_baselines(
        dataset=dataset,
        protected_group_report=protected_group_report,
        methods=args.methods,
        budget=args.budget,
        propagation_probability=args.propagation_prob,
        mc_runs=args.mc_runs,
        random_seed=args.random_seed,
    )

    frame = pd.DataFrame(
        [
            {
                "method": result.method,
                "seed_set": list(result.seed_set),
                "total_spread_mean": result.total_spread_mean,
                "total_spread_std": result.total_spread_std,
                "mf": result.mf,
                "soft_mf": result.soft_mf,
                "dcv": result.dcv,
                "runtime_seconds": result.runtime_seconds,
            }
            for result in results
        ]
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
