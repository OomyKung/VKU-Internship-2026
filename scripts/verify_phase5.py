"""CLI for Phase 5 hybrid SI+EA optimizer verification."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.community_detection import detect_communities  # noqa: E402
from fim_hybrid.config import DatasetConfig  # noqa: E402
from fim_hybrid.data_loader import load_dataset, resolve_builtin_dataset, verify_protected_groups  # noqa: E402
from fim_hybrid.hybrid_optimizer import HybridSIEAConfig, HybridSIEAOptimizer  # noqa: E402


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 5 hybrid SI+EA optimizer verification.")
    parser.add_argument("--dataset", default="graph_spa_500_0", help="Built-in dataset name.")
    parser.add_argument("--protected-attribute", required=True, help="Protected attribute for fairness evaluation.")
    parser.add_argument("--edge-path", default=None, help="Path to a simple edge-list file.")
    parser.add_argument("--pickle-path", default=None, help="Path to a pickled NetworkX graph.")
    parser.add_argument("--attribute-path", default=None, help="Path to a node-attribute CSV file.")
    parser.add_argument("--delimiter", default=None, help="Optional edge-list delimiter override.")
    parser.add_argument("--undirected", action="store_true", help="Load explicit edge lists as undirected graphs.")
    parser.add_argument("--node-id-column", default="node_id", help="Node id column name in the attribute CSV.")
    parser.add_argument("--community-method", default="louvain", choices=["louvain", "leiden"], help="Community detection method.")
    parser.add_argument("--budget", type=int, required=True, help="Fixed seed-set size.")
    parser.add_argument("--population-size", type=int, default=8, help="Population size.")
    parser.add_argument("--generations", type=int, default=5, help="Number of generations.")
    parser.add_argument("--crossover-probability", type=float, default=0.7, help="Crossover probability.")
    parser.add_argument("--mutation-probability", type=float, default=0.2, help="Per-node mutation probability.")
    parser.add_argument("--elite-fraction", type=float, default=0.25, help="Elite fraction.")
    parser.add_argument("--leader-guidance-fraction", type=float, default=0.34, help="Leader replacement fraction.")
    parser.add_argument("--propagation-prob", type=float, default=0.01, help="IC edge propagation probability.")
    parser.add_argument("--mc-runs", type=int, default=20, help="Monte Carlo run count.")
    parser.add_argument("--lambda-weight", type=float, default=0.5, help="Score weight: lambda*MF - (1-lambda)*DCV.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--debug", action="store_true", help="Print per-generation debug logging.")
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
    community_result = detect_communities(
        graph=dataset.graph,
        method=args.community_method,
        seed=args.random_seed,
    )
    optimizer = HybridSIEAOptimizer(
        dataset=dataset,
        protected_group_report=protected_group_report,
        community_result=community_result,
        config=HybridSIEAConfig(
            budget=args.budget,
            population_size=args.population_size,
            generations=args.generations,
            crossover_probability=args.crossover_probability,
            mutation_probability=args.mutation_probability,
            elite_fraction=args.elite_fraction,
            leader_guidance_fraction=args.leader_guidance_fraction,
            propagation_probability=args.propagation_prob,
            mc_runs=args.mc_runs,
            lambda_weight=args.lambda_weight,
            random_seed=args.random_seed,
            debug_logging=args.debug,
        ),
    )
    result = optimizer.optimize()

    print(f"dataset: {dataset.name}")
    print(f"best_seed_set: {list(result.best_seed_set)}")
    print(f"best_score: {result.best_score:.6f}")
    print(f"best_spread: {result.best_spread:.6f}")
    print(f"best_mf: {result.best_fairness.mf:.6f}")
    print(f"best_dcv: {result.best_fairness.dcv:.6f}")
    print("history:")
    print(result.history.to_string(index=False))


if __name__ == "__main__":
    main()
