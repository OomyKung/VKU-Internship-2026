"""CLI for Phase 2 diffusion and fairness verification."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.config import DatasetConfig  # noqa: E402
from fim_hybrid.data_loader import load_dataset, resolve_builtin_dataset, verify_protected_groups  # noqa: E402
from fim_hybrid.diffusion import simulate_independent_cascade  # noqa: E402
from fim_hybrid.fairness import evaluate_fairness  # noqa: E402


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _graph_node_type(graph) -> type[Any]:
    return type(next(iter(graph.nodes())))


def _coerce_cli_seed(value: str, graph) -> Any:
    node_type = _graph_node_type(graph)
    try:
        return node_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not coerce seed value '{value}' to graph node type {node_type.__name__}."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 diffusion and fairness verification.")
    parser.add_argument("--dataset", default="graph_spa_500_0", help="Built-in dataset name.")
    parser.add_argument("--protected-attribute", required=True, help="Protected attribute to verify.")
    parser.add_argument("--edge-path", default=None, help="Path to a simple edge-list file.")
    parser.add_argument("--pickle-path", default=None, help="Path to a pickled NetworkX graph.")
    parser.add_argument("--attribute-path", default=None, help="Path to a node-attribute CSV file.")
    parser.add_argument("--delimiter", default=None, help="Optional edge-list delimiter override.")
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="Load explicit edge lists as undirected graphs.",
    )
    parser.add_argument(
        "--node-id-column",
        default="node_id",
        help="Node id column name in the attribute CSV.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        required=True,
        help="Seed node ids to evaluate.",
    )
    parser.add_argument("--propagation-prob", type=float, default=0.01, help="IC edge propagation probability.")
    parser.add_argument("--mc-runs", type=int, default=100, help="Monte Carlo run count.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducible simulation.")
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
    seeds = [_coerce_cli_seed(value, dataset.graph) for value in args.seeds]

    diffusion_result = simulate_independent_cascade(
        dataset=dataset,
        protected_group_report=protected_group_report,
        seed_set=seeds,
        propagation_probability=args.propagation_prob,
        mc_runs=args.mc_runs,
        random_seed=args.random_seed,
    )
    fairness_metrics = evaluate_fairness(
        group_spread=diffusion_result.group_spread_mean,
        group_sizes=protected_group_report.group_sizes,
        total_spread=diffusion_result.total_spread_mean,
        include_soft_mf=True,
    )

    print(f"dataset: {dataset.name}")
    print(f"protected_attribute: {protected_group_report.protected_attribute}")
    print(f"seed_set: {list(diffusion_result.seed_set)}")
    print(f"total_spread_mean: {diffusion_result.total_spread_mean:.6f}")
    print(f"total_spread_std: {diffusion_result.total_spread_std:.6f}")
    print("group_spread_mean:")
    for group_name, spread in diffusion_result.group_spread_mean.items():
        print(f"  - {group_name}: {spread:.6f}")
    print("group_spread_std:")
    for group_name, spread in diffusion_result.group_spread_std.items():
        print(f"  - {group_name}: {spread:.6f}")
    print("normalized_group_spread:")
    for group_name, spread in fairness_metrics.normalized_group_spread.items():
        print(f"  - {group_name}: {spread:.6f}")
    print(f"strict_mf: {fairness_metrics.mf:.6f}")
    print(f"soft_mf: {fairness_metrics.soft_mf:.6f}")
    print(f"dcv: {fairness_metrics.dcv:.6f}")


if __name__ == "__main__":
    main()
