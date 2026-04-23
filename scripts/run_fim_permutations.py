"""Run named end-to-end FIM algorithm-stack permutations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.diffusion import DEFAULT_DIFFUSION_MODEL, SUPPORTED_DIFFUSION_MODELS  # noqa: E402
from fim_hybrid.permutations import (  # noqa: E402
    FIMPermutationRunConfig,
    available_fim_permutations,
    format_fim_permutation_report,
    run_fim_permutation_benchmark_from_config,
)
from scripts.run_experiment import build_dataset_config  # noqa: E402


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare named FIM algorithm-stack permutations with a shared final Monte Carlo evaluator."
    )
    parser.add_argument(
        "--dataset",
        default="graph_spa_500_0",
        help="Built-in dataset name, supported custom dataset path, or dataset stem under networks/.",
    )
    parser.add_argument("--graph-path", default=None, help="Path to an external graph file.")
    parser.add_argument("--attributes-path", "--attribute-path", dest="attributes_path", default=None)
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "pickle", "pkl", "txt", "csv"],
        default="auto",
        help="External graph format.",
    )
    parser.add_argument("--dataset-config", default=None, help="Optional JSON dataset config file.")
    parser.add_argument(
        "--directed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Treat external edge-list datasets as directed.",
    )
    parser.add_argument("--source-col", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--node-id-col", default=None)
    parser.add_argument("--protected-attribute", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument(
        "--permutations",
        nargs="+",
        default=list(available_fim_permutations()),
        choices=list(available_fim_permutations()),
        help="Permutation names to run.",
    )
    parser.add_argument(
        "--propagation-prob",
        type=float,
        default=0.01,
        help="Propagation probability for IC/WC and default threshold for LT.",
    )
    parser.add_argument("--mc-runs-search", type=int, default=20)
    parser.add_argument("--mc-runs-eval", type=int, default=100)
    parser.add_argument("--lambda-weight", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert permutation failures into skipped rows instead of aborting the whole benchmark.",
    )
    parser.add_argument(
        "--permutation3-diffusion-model",
        choices=list(SUPPORTED_DIFFUSION_MODELS),
        default=DEFAULT_DIFFUSION_MODEL,
        help="Optional IC/LT/WC comparison mode for infomap_graphcl_maximin.",
    )
    parser.add_argument(
        "--swap-candidate-pool-size",
        type=int,
        default=24,
        help="Bounded candidate pool for explicit swap-local-search permutation glue.",
    )
    parser.add_argument("--local-search-steps", type=int, default=1)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--gnn-epochs", type=int, default=30)
    parser.add_argument("--gnn-hidden-dim", type=int, default=32)
    parser.add_argument("--ris-num-rr-sets", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = build_dataset_config(args)
    run_config = FIMPermutationRunConfig(
        protected_attribute=args.protected_attribute,
        budget=args.budget,
        propagation_probability=args.propagation_prob,
        mc_runs_search=args.mc_runs_search,
        mc_runs_eval=args.mc_runs_eval,
        lambda_weight=args.lambda_weight,
        random_seed=args.random_seed,
        output_dir=_resolve_repo_path(args.output_dir),
        continue_on_error=args.continue_on_error,
        swap_candidate_pool_size=args.swap_candidate_pool_size,
        local_search_steps=args.local_search_steps,
        permutation3_diffusion_model=args.permutation3_diffusion_model,
        population_size=args.population_size,
        generations=args.generations,
        gnn_epochs=args.gnn_epochs,
        gnn_hidden_dim=args.gnn_hidden_dim,
        ris_num_rr_sets=args.ris_num_rr_sets,
    )
    result = run_fim_permutation_benchmark_from_config(
        dataset_config=dataset_config,
        config=run_config,
        permutations=args.permutations,
    )
    report = format_fim_permutation_report(result.summary_frame, run_config)
    print(report)
    if result.comparison_csv_path is not None:
        print(f"\nSaved comparison CSV: {result.comparison_csv_path}")
    if result.report_path is not None:
        print(f"Saved report: {result.report_path}")


if __name__ == "__main__":
    main()
