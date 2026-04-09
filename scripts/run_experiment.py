"""CLI for fair FIM method comparison experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.data_loader import resolve_builtin_dataset  # noqa: E402
from fim_hybrid.experiment_runner import ExperimentSettings, run_experiment  # noqa: E402


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comparable Fair Influence Maximization experiments.")
    parser.add_argument("--dataset", default="graph_spa_500_0", help="Built-in dataset name.")
    parser.add_argument("--protected-attribute", required=True, help="Protected attribute for fairness metrics.")
    parser.add_argument("--community-methods", nargs="+", default=["leiden"], help="Community detection methods to compare.")
    parser.add_argument(
        "--baseline-methods",
        nargs="+",
        default=["degree", "pagerank", "community_round_robin", "random"],
        help="Baseline methods to compare.",
    )
    parser.add_argument("--budget", type=int, required=True, help="Fixed seed budget.")
    parser.add_argument("--propagation-prob", type=float, default=0.01, help="Independent Cascade propagation probability.")
    parser.add_argument("--mc-runs", type=int, default=20, help="Monte Carlo run count.")
    parser.add_argument("--lambda-weight", type=float, default=0.5, help="Lambda in F(S) = lambda*MF - (1-lambda)*DCV.")
    parser.add_argument("--population-size", type=int, default=12, help="Hybrid population size.")
    parser.add_argument("--generations", type=int, default=10, help="Hybrid generation count.")
    parser.add_argument("--crossover-probability", type=float, default=0.7, help="Hybrid crossover probability.")
    parser.add_argument("--mutation-probability", type=float, default=0.2, help="Hybrid mutation probability.")
    parser.add_argument("--elite-fraction", type=float, default=0.25, help="Hybrid elite fraction.")
    parser.add_argument("--leader-guidance-fraction", type=float, default=0.34, help="Swarm-style leader replacement fraction.")
    parser.add_argument("--local-search-steps", type=int, default=2, help="Local search refinement steps per offspring.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", default="results", help="Directory for CSV outputs.")
    parser.add_argument("--no-ablations", action="store_true", help="Skip optimizer ablation variants.")
    parser.add_argument("--use-node2vec", action="store_true", help="Request optional Node2Vec guidance.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = resolve_builtin_dataset(args.dataset, ROOT)
    output_dir = _resolve_repo_path(args.output_dir)
    settings = ExperimentSettings(
        protected_attribute=args.protected_attribute,
        budget=args.budget,
        community_method=args.community_methods[0],
        propagation_probability=args.propagation_prob,
        mc_runs=args.mc_runs,
        lambda_weight=args.lambda_weight,
        population_size=args.population_size,
        generations=args.generations,
        crossover_probability=args.crossover_probability,
        mutation_probability=args.mutation_probability,
        elite_fraction=args.elite_fraction,
        leader_guidance_fraction=args.leader_guidance_fraction,
        local_search_steps=args.local_search_steps,
        random_seed=args.random_seed,
        output_dir=output_dir,
        use_node2vec=args.use_node2vec,
    )
    result_frame = run_experiment(
        dataset_config=dataset_config,
        settings=settings,
        community_methods=args.community_methods,
        baseline_methods=args.baseline_methods,
        include_ablations=not args.no_ablations,
    )
    columns = [
        "dataset",
        "community_method",
        "method",
        "variant_type",
        "total_spread",
        "mf",
        "dcv",
        "f_score",
        "runtime_seconds",
        "community_modularity",
    ]
    print(result_frame[columns].sort_values(["community_method", "f_score"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
