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


_LEGACY_PERMUTATION_ALIASES = [
    "graphsage_fair_ris_hybrid",
    "infomap_graphcl_maximin",
]


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args() -> argparse.Namespace:
    permutation_choices = list(available_fim_permutations()) + _LEGACY_PERMUTATION_ALIASES
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
        choices=permutation_choices,
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
        "--alternate-diffusion-model",
        "--permutation3-diffusion-model",
        dest="alternate_diffusion_model",
        choices=list(SUPPORTED_DIFFUSION_MODELS),
        default=DEFAULT_DIFFUSION_MODEL,
        help="Optional IC/LT/WC comparison mode for alternate stack runs.",
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
    parser.add_argument("--gnn-num-layers", type=int, default=2)
    parser.add_argument("--gnn-dropout", type=float, default=0.2)
    parser.add_argument("--gnn-learning-rate", type=float, default=1e-3)
    parser.add_argument("--gnn-weight-decay", type=float, default=5e-4)
    parser.add_argument("--ris-num-rr-sets", type=int, default=128)
    parser.add_argument("--ranking-top-fraction", type=float, default=0.5)
    parser.add_argument("--ranking-top-n", type=int, default=None)
    parser.add_argument("--ranking-max-nodes", type=int, default=None)
    parser.add_argument("--clustering-n-clusters", type=int, default=None)
    parser.add_argument("--clustering-min-cluster-size", type=int, default=None)
    parser.add_argument("--adaptive-fairness-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--imbalance-threshold-medium", type=float, default=5.0)
    parser.add_argument("--imbalance-threshold-high", type=float, default=10.0)
    parser.add_argument("--adaptive-fairness-multiplier-medium", type=float, default=1.5)
    parser.add_argument("--adaptive-fairness-multiplier-high", type=float, default=2.0)
    parser.add_argument("--large-imbalance-fairness-mode", choices=["auto", "off", "force"], default="off")
    parser.add_argument("--large-imbalance-threshold", type=float, default=5.0)
    parser.add_argument("--large-graph-threshold", type=int, default=1000)
    parser.add_argument("--use-group-stratified-candidate-pool", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-group-candidate-floor", type=int, default=20)
    parser.add_argument("--group-candidate-multiplier", type=float, default=3.0)
    parser.add_argument("--use-protected-group-quota-initialization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--small-group-seed-fraction", type=float, default=0.10)
    parser.add_argument("--initialization-quota-mode", choices=["proportional", "sqrt", "uniform_min"], default="sqrt")
    parser.add_argument("--score-normalization", choices=["global", "per_group", "hybrid"], default="global")
    parser.add_argument("--large-imbalance-ml-score-weight", type=float, default=0.4)
    parser.add_argument("--large-imbalance-ris-score-weight", type=float, default=0.5)
    parser.add_argument("--large-imbalance-fair-ris-score-weight", type=float, default=2.0)
    parser.add_argument("--large-imbalance-weak-group-bonus-weight", type=float, default=2.0)
    parser.add_argument("--large-imbalance-protected-group-coverage-weight", type=float, default=2.0)
    parser.add_argument("--large-imbalance-community-diversity-weight", type=float, default=0.8)
    parser.add_argument("--large-imbalance-spread-proxy-weight", type=float, default=0.2)
    parser.add_argument("--use-group-quota-repair", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-seeds-per-protected-group", type=int, default=1)
    parser.add_argument("--quota-mode", choices=["none", "at_least_one", "proportional", "support_aware"], default="support_aware")
    parser.add_argument("--quota-min-group-support", type=int, default=5)
    parser.add_argument("--use-fairness-first-repair", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--weak-group-repair-rounds", type=int, default=0)
    parser.add_argument("--majority-overconcentration-threshold", type=float, default=0.60)
    parser.add_argument("--use-fairness-first-swap-acceptance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fairness-tolerance-fscore-drop", type=float, default=0.001)
    parser.add_argument("--fairness-tolerance-dcv", type=float, default=0.005)
    parser.add_argument("--swap-reject-spread-gain-if-fairness-collapses", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-budget-node-ratio-warning", type=float, default=0.02)
    parser.add_argument("--min-seeds-per-group-warning", type=int, default=5)
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
        alternate_diffusion_model=args.alternate_diffusion_model,
        permutation3_diffusion_model=args.alternate_diffusion_model,
        population_size=args.population_size,
        generations=args.generations,
        gnn_epochs=args.gnn_epochs,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_num_layers=args.gnn_num_layers,
        gnn_dropout=args.gnn_dropout,
        gnn_learning_rate=args.gnn_learning_rate,
        gnn_weight_decay=args.gnn_weight_decay,
        ris_num_rr_sets=args.ris_num_rr_sets,
        ranking_top_fraction=args.ranking_top_fraction,
        ranking_top_n=args.ranking_top_n,
        ranking_max_nodes=args.ranking_max_nodes,
        clustering_n_clusters=args.clustering_n_clusters,
        clustering_min_cluster_size=args.clustering_min_cluster_size,
        adaptive_fairness_weights=bool(args.adaptive_fairness_weights),
        imbalance_threshold_medium=float(args.imbalance_threshold_medium),
        imbalance_threshold_high=float(args.imbalance_threshold_high),
        adaptive_fairness_multiplier_medium=float(args.adaptive_fairness_multiplier_medium),
        adaptive_fairness_multiplier_high=float(args.adaptive_fairness_multiplier_high),
        large_imbalance_fairness_mode=str(args.large_imbalance_fairness_mode),
        large_imbalance_threshold=float(args.large_imbalance_threshold),
        large_graph_threshold=int(args.large_graph_threshold),
        use_group_stratified_candidate_pool=args.use_group_stratified_candidate_pool,
        min_group_candidate_floor=int(args.min_group_candidate_floor),
        group_candidate_multiplier=float(args.group_candidate_multiplier),
        use_protected_group_quota_initialization=args.use_protected_group_quota_initialization,
        small_group_seed_fraction=float(args.small_group_seed_fraction),
        initialization_quota_mode=str(args.initialization_quota_mode),
        score_normalization=str(args.score_normalization),
        large_imbalance_ml_score_weight=float(args.large_imbalance_ml_score_weight),
        large_imbalance_ris_score_weight=float(args.large_imbalance_ris_score_weight),
        large_imbalance_fair_ris_score_weight=float(args.large_imbalance_fair_ris_score_weight),
        large_imbalance_weak_group_bonus_weight=float(args.large_imbalance_weak_group_bonus_weight),
        large_imbalance_protected_group_coverage_weight=float(args.large_imbalance_protected_group_coverage_weight),
        large_imbalance_community_diversity_weight=float(args.large_imbalance_community_diversity_weight),
        large_imbalance_spread_proxy_weight=float(args.large_imbalance_spread_proxy_weight),
        use_group_quota_repair=bool(args.use_group_quota_repair),
        min_seeds_per_protected_group=int(args.min_seeds_per_protected_group),
        quota_mode=str(args.quota_mode),
        quota_min_group_support=int(args.quota_min_group_support),
        use_fairness_first_repair=bool(args.use_fairness_first_repair),
        weak_group_repair_rounds=int(args.weak_group_repair_rounds),
        majority_overconcentration_threshold=float(args.majority_overconcentration_threshold),
        use_fairness_first_swap_acceptance=bool(args.use_fairness_first_swap_acceptance),
        fairness_tolerance_fscore_drop=float(args.fairness_tolerance_fscore_drop),
        fairness_tolerance_dcv=float(args.fairness_tolerance_dcv),
        swap_reject_spread_gain_if_fairness_collapses=bool(args.swap_reject_spread_gain_if_fairness_collapses),
        min_budget_node_ratio_warning=float(args.min_budget_node_ratio_warning),
        min_seeds_per_group_warning=int(args.min_seeds_per_group_warning),
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
