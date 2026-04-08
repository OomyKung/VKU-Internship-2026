"""CLI entry point for running FIM experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add the repository root to sys.path so the script can import the local package
# when it is run directly from PowerShell.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.config import (  # noqa: E402
    CommunityConfig,
    DatasetConfig,
    DiffusionConfig,
    ExperimentConfig,
    FairnessConfig,
    MLConfig,
    OptimizerConfig,
)
from fim_hybrid.data_loader import resolve_builtin_dataset  # noqa: E402
from fim_hybrid.experiment_runner import run_experiment  # noqa: E402


def _resolve_repo_path(path_value: str | None) -> Path | None:
    """Resolve CLI paths relative to the repository root when needed."""

    if path_value is None:
        return None

    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args() -> argparse.Namespace:
    # Keep the CLI flat and explicit so single-command local runs are easy.
    parser = argparse.ArgumentParser(description="Run Fair Influence Maximization experiments.")
    parser.add_argument("--dataset", default="twitter", help="Built-in dataset name.")
    parser.add_argument("--edge-path", default=None, help="Path to an edge-list file.")
    parser.add_argument("--attribute-path", default=None, help="Path to a node-attribute CSV file.")
    parser.add_argument("--pickle-path", default=None, help="Path to a pickled NetworkX graph.")
    parser.add_argument("--protected-attribute", default="color", help="Protected attribute for fairness metrics.")
    parser.add_argument("--budget", type=int, default=10, help="Seed budget.")
    parser.add_argument("--community-methods", nargs="+", default=["leiden"], help="Community detection methods to compare.")
    parser.add_argument(
        "--baseline-methods",
        nargs="+",
        default=["degree", "pagerank", "community_round_robin", "random"],
        help="Baseline seed selection methods to compare.",
    )
    parser.add_argument("--propagation-prob", type=float, default=0.01, help="IC edge propagation probability.")
    parser.add_argument("--mc-runs", type=int, default=100, help="Monte Carlo diffusion runs.")
    parser.add_argument("--population-size", type=int, default=20, help="Hybrid optimizer population size.")
    parser.add_argument("--generations", type=int, default=50, help="Hybrid optimizer generations.")
    parser.add_argument("--lambda-weight", type=float, default=0.5, help="Lambda in F(S) = lambda*MF - (1-lambda)*DCV.")
    parser.add_argument("--optimizer-seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--ml", action="store_true", help="Enable ML-guided candidate restriction.")
    parser.add_argument("--ml-top-fraction", type=float, default=0.3, help="Top candidate fraction for ML-guided search.")
    parser.add_argument("--ml-singleton-runs", type=int, default=30, help="Monte Carlo runs per singleton label.")
    parser.add_argument("--ml-max-nodes", type=int, default=None, help="Limit nodes used for label generation.")
    parser.add_argument("--output-dir", default="results", help="Output directory.")
    parser.add_argument("--no-plots", action="store_true", help="Disable result plots.")
    return parser.parse_args()


def build_dataset_config(args: argparse.Namespace) -> DatasetConfig:
    # Use explicit file paths when provided; otherwise resolve one of the built-in datasets.
    if args.edge_path or args.pickle_path:
        return DatasetConfig(
            name=args.dataset,
            edge_path=_resolve_repo_path(args.edge_path),
            attribute_path=_resolve_repo_path(args.attribute_path),
            pickle_path=_resolve_repo_path(args.pickle_path),
            propagation_probability=args.propagation_prob,
        )

    dataset_config = resolve_builtin_dataset(args.dataset, ROOT)
    dataset_config.propagation_probability = args.propagation_prob
    return dataset_config


def main() -> None:
    args = parse_args()
    output_dir = _resolve_repo_path(args.output_dir) or (ROOT / "results")

    # Translate CLI arguments into the dataclass-based experiment configuration
    # used throughout the rest of the package.
    experiment_config = ExperimentConfig(
        dataset=build_dataset_config(args),
        community=CommunityConfig(method=args.community_methods[0], seed=args.optimizer_seed),
        diffusion=DiffusionConfig(
            propagation_probability=args.propagation_prob,
            mc_runs=args.mc_runs,
            seed=args.optimizer_seed,
        ),
        fairness=FairnessConfig(
            protected_attribute=args.protected_attribute,
            lambda_weight=args.lambda_weight,
        ),
        optimizer=OptimizerConfig(
            budget=args.budget,
            population_size=args.population_size,
            generations=args.generations,
            seed=args.optimizer_seed,
        ),
        ml=MLConfig(
            enabled=args.ml,
            top_fraction=args.ml_top_fraction,
            singleton_mc_runs=args.ml_singleton_runs,
            seed=args.optimizer_seed,
        ),
        output_dir=output_dir,
        random_seed=args.optimizer_seed,
    )

    results = run_experiment(
        config=experiment_config,
        community_methods=args.community_methods,
        baseline_methods=args.baseline_methods,
        enable_plots=not args.no_plots,
        ml_max_nodes=args.ml_max_nodes,
    )
    # Print a compact summary table while full results are saved to CSV.
    print(results[["dataset", "community_method", "method", "total_spread", "mf", "dcv", "f_score", "runtime_seconds"]])


if __name__ == "__main__":
    main()
