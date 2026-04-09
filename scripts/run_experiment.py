"""CLI for fair FIM method comparison experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

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


def _rule(character: str = "=") -> str:
    width = max(80, min(120, shutil.get_terminal_size((100, 20)).columns))
    return character * width


def _format_float(value: object, digits: int = 6) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _format_runtime_seconds(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.3f}s"


def _format_int(value: object) -> str:
    if pd.isna(value):
        return "-"
    return str(int(value))


def _display_ml_mode(method: str, ml_guidance_mode: object) -> str:
    if method == "ml_topk":
        return "rank_only"
    if method == "ml_topk_node2vec":
        return "rank_only"
    if pd.isna(ml_guidance_mode) or ml_guidance_mode == "off":
        return "-"
    return str(ml_guidance_mode)


def _build_ranked_results_table(result_frame: pd.DataFrame) -> str:
    ordered = result_frame.sort_values(["f_score", "runtime_seconds"], ascending=[False, True]).reset_index(drop=True)
    zero_cov = ordered.get("zero_covered_groups_count", pd.Series([float("nan")] * len(ordered)))
    fraction_covered = ordered.get("fraction_groups_covered", pd.Series([float("nan")] * len(ordered)))
    bottom_3 = ordered.get("bottom_3_avg_group_spread", pd.Series([float("nan")] * len(ordered)))
    delta_f = ordered.get("delta_f_score", pd.Series([float("nan")] * len(ordered)))
    display = pd.DataFrame(
        {
            "Rank": range(1, len(ordered) + 1),
            "Method": ordered["method"],
            "Kind": ordered["variant_type"],
            "F-score": ordered["f_score"].map(_format_float),
            "Delta F": delta_f.map(lambda value: "-" if pd.isna(value) else f"{float(value):+0.6f}"),
            "Spread": ordered["total_spread"].map(_format_float),
            "MF": ordered["mf"].map(_format_float),
            "DCV": ordered["dcv"].map(_format_float),
            "Runtime": ordered["runtime_seconds"].map(_format_runtime_seconds),
            "Pool": ordered["candidate_pool_size"].map(_format_int),
            "ZeroCov": zero_cov.map(_format_int),
            "FracCov": fraction_covered.map(lambda value: _format_float(value, digits=3)),
            "Bottom3": bottom_3.map(lambda value: _format_float(value, digits=3)),
            "N2V": ordered["node2vec_mode"],
            "ML Mode": [
                _display_ml_mode(method, ml_mode)
                for method, ml_mode in zip(ordered["method"], ordered["ml_guidance_mode"], strict=False)
            ],
            "ML Spearman": ordered["ml_validation_spearman"].map(_format_float),
            "ML P@k": ordered["ml_validation_precision_at_budget"].map(_format_float),
        }
    )
    return display.to_string(index=False)


def _build_highlight_lines(result_frame: pd.DataFrame) -> list[str]:
    ranked = result_frame.sort_values(["f_score", "runtime_seconds"], ascending=[False, True]).reset_index(drop=True)
    best_row = ranked.iloc[0]
    fastest_row = result_frame.sort_values(["runtime_seconds", "f_score"], ascending=[True, False]).iloc[0]
    lines = [
        f"Best F-score: {best_row['method']} ({_format_float(best_row['f_score'])})",
        f"Fastest run : {fastest_row['method']} ({_format_runtime_seconds(fastest_row['runtime_seconds'])})",
    ]

    ml_rows = ranked[ranked["method"].astype(str).str.startswith("hybrid_siea_ml_")]
    if not ml_rows.empty:
        best_ml_row = ml_rows.iloc[0]
        lines.append(
            "Best ML run: "
            f"{best_ml_row['method']} "
            f"(F-score={_format_float(best_ml_row['f_score'])}, runtime={_format_runtime_seconds(best_ml_row['runtime_seconds'])})"
        )

    return lines


def _build_hybrid_delta_table(result_frame: pd.DataFrame) -> str | None:
    baseline = result_frame[result_frame["method"] == "hybrid_siea"]
    if baseline.empty:
        return None

    baseline_row = baseline.iloc[0]
    method_names = result_frame["method"].astype(str)
    comparison_rows = result_frame[
        method_names.eq("hybrid_siea") | method_names.str.startswith("hybrid_siea_ml_")
    ].copy()
    if comparison_rows.empty:
        return None

    comparison_rows["delta_f_score"] = comparison_rows["f_score"] - float(baseline_row["f_score"])
    comparison_rows["delta_runtime_seconds"] = comparison_rows["runtime_seconds"] - float(baseline_row["runtime_seconds"])
    comparison_rows = comparison_rows.sort_values(
        ["delta_f_score", "runtime_seconds", "method"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    display = pd.DataFrame(
        {
            "Method": comparison_rows["method"],
            "F-score": comparison_rows["f_score"].map(_format_float),
            "Delta F": comparison_rows["delta_f_score"].map(lambda value: f"{float(value):+0.6f}"),
            "Runtime": comparison_rows["runtime_seconds"].map(_format_runtime_seconds),
            "Delta T": comparison_rows["delta_runtime_seconds"].map(lambda value: f"{float(value):+0.3f}s"),
            "Pool": comparison_rows["candidate_pool_size"].map(_format_int),
            "ZeroCov": comparison_rows.get("zero_covered_groups_count", pd.Series([float("nan")] * len(comparison_rows))).map(_format_int),
            "FracCov": comparison_rows.get("fraction_groups_covered", pd.Series([float("nan")] * len(comparison_rows))).map(lambda value: _format_float(value, digits=3)),
            "N2V": comparison_rows["node2vec_mode"],
            "ML Mode": [
                _display_ml_mode(method, ml_mode)
                for method, ml_mode in zip(comparison_rows["method"], comparison_rows["ml_guidance_mode"], strict=False)
            ],
        }
    )
    return display.to_string(index=False)


def format_results_report(result_frame: pd.DataFrame, settings: ExperimentSettings) -> str:
    """Render a compact human-readable terminal report for experiment results."""

    if result_frame.empty:
        return "No experiment results were produced."

    dataset_names = ", ".join(sorted(str(value) for value in result_frame["dataset"].dropna().unique()))
    community_methods = ", ".join(sorted(str(value) for value in result_frame["community_method"].dropna().unique()))
    lines = [
        _rule("="),
        "Fair Influence Maximization Experiment Summary",
        _rule("="),
        f"Dataset: {dataset_names}",
        f"Protected attribute: {settings.protected_attribute}",
        f"Budget: {settings.budget} | MC runs: {settings.mc_runs} | Random seed: {settings.random_seed}",
        f"Community methods: {community_methods}",
        (
            f"ML: enabled | requested mode={settings.ml_guidance_mode} | "
            f"singleton label runs={settings.ml_singleton_runs}"
            if settings.use_ml
            else "ML: disabled"
        ),
    ]
    if settings.use_node2vec:
        lines.append(
            "Node2Vec: enabled | "
            f"dims={settings.node2vec_dimensions} | "
            f"walk_length={settings.node2vec_walk_length} | "
            f"num_walks={settings.node2vec_num_walks} | "
            f"window={settings.node2vec_window} | "
            f"p={settings.node2vec_p:g} | "
            f"q={settings.node2vec_q:g} | "
            f"scale={int(settings.node2vec_scale_embeddings)} | "
            f"pca={settings.node2vec_pca_components if settings.node2vec_pca_components is not None else 'none'} | "
            f"mode={settings.node2vec_integration_mode}"
        )
    else:
        lines.append("Node2Vec: disabled")

    for community_method, community_frame in result_frame.groupby("community_method", sort=True):
        lines.extend(
            [
                "",
                _rule("-"),
                f"Community: {community_method}",
                _rule("-"),
                _build_ranked_results_table(community_frame),
                "",
                "Highlights",
            ]
        )
        lines.extend(f"  {line}" for line in _build_highlight_lines(community_frame))

        delta_table = _build_hybrid_delta_table(community_frame)
        if delta_table is not None:
            lines.extend(
                [
                    "",
                    "Delta vs hybrid_siea",
                    delta_table,
                ]
            )

    if settings.output_dir is not None and len(result_frame["dataset"].dropna().unique()) == 1:
        dataset_name = str(result_frame["dataset"].dropna().unique()[0])
        result_path = settings.output_dir / dataset_name / f"{dataset_name}_budget{settings.budget}_results.csv"
        lines.extend(["", f"Saved CSV: {result_path}"])

    return "\n".join(lines)


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
    parser.add_argument("--ml", action="store_true", help="Enable ML-guided candidate selection.")
    parser.add_argument(
        "--ml-guidance-mode",
        default="off",
        choices=["off", "hard_filter", "soft_bias", "two_tier"],
        help="Guidance mode to run. With --ml and mode=off, the runner compares hard_filter, soft_bias, legacy two_tier, and tuned two_tier.",
    )
    parser.add_argument("--ml-top-fraction", type=float, default=0.25, help="Fraction of ranked nodes kept for ML-guided optimization.")
    parser.add_argument("--ml-top-n", type=int, default=None, help="Override ML candidate pool size with a fixed top-N cutoff.")
    parser.add_argument("--ml-max-nodes", type=int, default=None, help="Optional hard cap after ML ranking is applied.")
    parser.add_argument("--ml-singleton-runs", type=int, default=15, help="Monte Carlo runs for singleton label generation.")
    parser.add_argument("--ml-primary-pool-ratio", type=float, default=0.25, help="Top-ranked ML ratio used as the primary pool in soft guidance modes.")
    parser.add_argument("--ml-secondary-exploration-rate", type=float, default=0.10, help="Exploration rate for the secondary ML pool in two-tier guidance.")
    parser.add_argument("--ml-initialization-bias", type=float, default=0.25, help="Fraction of initialized individuals that receive ML-biased construction.")
    parser.add_argument("--ml-initialization-primary-rate", type=float, default=0.90, help="Primary-pool preference used by tuned two-tier initialization.")
    parser.add_argument("--ml-mutation-primary-rate", type=float, default=0.80, help="Primary-pool preference used by tuned two-tier mutation.")
    parser.add_argument("--ml-repair-primary-rate", type=float, default=0.70, help="Primary-pool preference used by tuned two-tier repair.")
    parser.add_argument("--ml-local-search-primary-rate", type=float, default=0.60, help="Primary-pool preference used by tuned two-tier local search.")
    parser.add_argument("--ml-mutation-bias-weight", type=float, default=0.20, help="Weight of the ML prior inside mutation candidate scoring.")
    parser.add_argument("--ml-repair-bias-weight", type=float, default=0.15, help="Weight of the ML prior inside repair and initialization sampling.")
    parser.add_argument("--ml-local-search-bias-weight", type=float, default=0.25, help="Weight of the ML prior inside local-search replacement ranking.")
    parser.add_argument("--fairness-first-init-enabled", action="store_true", help="Enable fairness-first initialization for hybrid optimizer runs.")
    parser.add_argument("--fairness-first-init-slots", type=int, default=0, help="Number of early seed positions reserved for fairness-first initialization.")
    parser.add_argument("--fairness-first-init-weight", type=float, default=0.0, help="Weight of weak-group support during fairness-first initialization.")
    parser.add_argument("--weakest-group-k", type=int, default=1, help="Number of weakest groups targeted by mutation, repair, and worst-group local search.")
    parser.add_argument("--weakest-group-mutation-weight", type=float, default=0.0, help="Weight of weak-group support inside mutation candidate ranking.")
    parser.add_argument("--zero-group-bonus-weight", type=float, default=0.0, help="Extra mutation and local-search bonus for zero-covered groups.")
    parser.add_argument("--bridge-to-weak-group-weight", type=float, default=0.0, help="Weight of bridge-to-weak-group structure inside mutation, repair, and worst-group local search.")
    parser.add_argument("--repair-fairness-weight", type=float, default=0.0, help="Weight of weak-group support inside repair scoring.")
    parser.add_argument("--repair-bridge-weight", type=float, default=0.0, help="Weight of bridge-to-weak-group structure inside repair scoring.")
    parser.add_argument("--repair-centrality-weight", type=float, default=0.0, help="Weight of centrality inside repair scoring.")
    parser.add_argument("--repair-diversity-weight", type=float, default=0.0, help="Weight of graph-structural diversity inside repair scoring.")
    parser.add_argument("--local-search-focus-mode", choices=["default", "worst_group"], default="default", help="How local search ranks candidate one-node replacement moves.")
    parser.add_argument("--local-search-bottom-k-groups", type=int, default=3, help="Bottom-k groups used for worst-group local-search tie diagnostics.")
    parser.add_argument("--local-search-max-trials", type=int, default=0, help="Optional cap on external candidates considered during each local-search step.")
    parser.add_argument("--compare-fairness-variants", action="store_true", help="Add current-best, init-only, mutation-only, repair-only, local-search-only, and full combined fairness variants for the tuned two-tier ML method.")
    parser.add_argument("--marginal-gain-scoring-enabled", action="store_true", help="Enable approximate fairness-aware marginal gain scoring in candidate ranking.")
    parser.add_argument("--marginal-gain-delta-mf-weight", type=float, default=0.0, help="Weight of proxy MF improvement inside marginal candidate scoring.")
    parser.add_argument("--marginal-gain-delta-dcv-weight", type=float, default=0.0, help="Weight of proxy DCV reduction inside marginal candidate scoring.")
    parser.add_argument("--marginal-gain-spread-weight", type=float, default=0.0, help="Weight of spread proxy inside marginal candidate scoring.")
    parser.add_argument("--local-search-swap-trials", type=int, default=0, help="Maximum number of non-improving swap trials before local search stops early.")
    parser.add_argument("--local-search-candidate-pool-size", type=int, default=0, help="Number of high-priority external candidates considered per local-search step.")
    parser.add_argument("--local-search-delta-mf-weight", type=float, default=0.0, help="Extra proxy MF-improvement weight used only for local-search candidate ranking.")
    parser.add_argument("--local-search-delta-dcv-weight", type=float, default=0.0, help="Extra proxy DCV-reduction weight used only for local-search candidate ranking.")
    parser.add_argument("--local-search-overlap-penalty-weight", type=float, default=0.0, help="Extra overlap-penalty multiplier used only for local-search candidate ranking.")
    parser.add_argument("--urgency-weight-enabled", action="store_true", help="Weight candidate benefit more strongly toward currently weak protected groups.")
    parser.add_argument("--urgency-exponent", type=float, default=1.0, help="Exponent applied to group urgency = 1 - coverage.")
    parser.add_argument("--weak-group-focus-weight", type=float, default=0.0, help="Extra multiplier applied to currently weakest groups inside urgency weighting.")
    parser.add_argument("--overlap-penalty-enabled", action="store_true", help="Penalize candidate nodes that overlap too much with the current seed set.")
    parser.add_argument("--same-community-penalty-weight", type=float, default=0.0, help="Penalty weight for adding nodes to already-represented communities.")
    parser.add_argument("--neighborhood-overlap-penalty-weight", type=float, default=0.0, help="Penalty weight for neighborhood overlap with the current seed set.")
    parser.add_argument("--compare-refinement-variants", action="store_true", help="Compare the current best fairness-full method against marginal-gain, swap-local-search, urgency, overlap, and full combined refinement variants.")
    parser.add_argument("--marginal-candidate-pool-size", type=int, default=0, help="Optional shortlist size before expensive marginal-gain scoring.")
    parser.add_argument("--mutation-candidate-pool-size", type=int, default=0, help="Optional shortlist size before mutation candidate ranking.")
    parser.add_argument("--enable-fitness-cache", action=argparse.BooleanOptionalAction, default=True, help="Enable seed-set evaluation caching.")
    parser.add_argument("--enable-marginal-cache", action="store_true", help="Enable bounded caching for marginal-gain proxy components.")
    parser.add_argument("--cache-max-size", type=int, default=0, help="Optional max size for fitness and marginal caches.")
    parser.add_argument("--local-search-early-stop-patience", type=int, default=0, help="Stop local search early after this many non-improving evaluated swaps.")
    parser.add_argument("--local-search-use-prefilter", action="store_true", help="Prefilter local-search external candidates before full ranking.")
    parser.add_argument("--optimization-mode", choices=["full", "balanced", "fast"], default="full", help="Runtime/quality trade-off mode for refinement-heavy variants.")
    parser.add_argument("--refinement-intensity", type=float, default=1.0, help="Global multiplier for runtime-sensitive refinement budgets.")
    parser.add_argument("--marginal-eval-fraction", type=float, default=1.0, help="Fraction of external candidates kept before expensive marginal ranking.")
    parser.add_argument("--compare-runtime-variants", action="store_true", help="Compare the current marginal-gain method against optimized, balanced, and fast runtime-aware variants.")
    parser.add_argument("--swap-candidate-pool-size", type=int, default=0, help="Optional candidate shortlist size used inside swap local search before full evaluation.")
    parser.add_argument("--swap-prefilter-top-k", type=int, default=0, help="Cheap-proxy prefilter size for swap local search.")
    parser.add_argument("--enable-swap-cache", action="store_true", help="Enable swap-evaluation caching in local search.")
    parser.add_argument("--local-search-failed-patience", type=int, default=0, help="Stop swap local search after this many non-improving fully evaluated swaps.")
    parser.add_argument("--local-search-first-improvement", action="store_true", help="Accept the first improving local-search swap instead of scanning for the best move.")
    parser.add_argument("--full-eval-top-k", type=int, default=0, help="Number of top proxy-ranked swaps to fully evaluate per removal candidate.")
    parser.add_argument("--proxy-score-weights", default="{}", help="JSON object overriding cheap swap-proxy weights, e.g. '{\"delta_mf\":0.6,\"delta_dcv\":0.45}'.")
    parser.add_argument("--compare-swap-runtime-variants", action="store_true", help="Compare the current swap-local-search method against optimized, first-improvement, and reduced-candidate runtime variants.")
    parser.add_argument("--use-node2vec", action="store_true", help="Request optional Node2Vec guidance.")
    parser.add_argument("--node2vec-dimensions", type=int, default=8, help="Node2Vec embedding dimensionality.")
    parser.add_argument("--node2vec-walk-length", type=int, default=20, help="Node2Vec random walk length.")
    parser.add_argument("--node2vec-num-walks", type=int, default=10, help="Node2Vec walks sampled per node.")
    parser.add_argument("--node2vec-window", type=int, default=5, help="Context window used to build Node2Vec co-occurrences.")
    parser.add_argument("--node2vec-p", type=float, default=1.0, help="Node2Vec return parameter p.")
    parser.add_argument("--node2vec-q", type=float, default=1.0, help="Node2Vec in-out parameter q.")
    parser.add_argument("--node2vec-scale-embeddings", action="store_true", help="Scale Node2Vec embeddings before optional PCA and concatenation.")
    parser.add_argument("--node2vec-pca-components", type=int, default=None, help="Optional PCA dimension for Node2Vec features.")
    parser.add_argument(
        "--node2vec-integration-mode",
        choices=["feature_concat", "diversity_signal", "both"],
        default="feature_concat",
        help="How Node2Vec is used when enabled.",
    )
    parser.add_argument("--node2vec-diversity-weight", type=float, default=0.15, help="Novelty bonus weight when Node2Vec is used as an optimizer diversity signal.")
    parser.add_argument(
        "--ml-model-type",
        choices=["random_forest", "xgboost"],
        default="random_forest",
        help="Tabular ML model used for candidate ranking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = resolve_builtin_dataset(args.dataset, ROOT)
    output_dir = _resolve_repo_path(args.output_dir)
    proxy_score_weights = json.loads(args.proxy_score_weights)
    if not isinstance(proxy_score_weights, dict):
        raise ValueError("--proxy-score-weights must parse to a JSON object.")
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
        node2vec_dimensions=args.node2vec_dimensions,
        node2vec_walk_length=args.node2vec_walk_length,
        node2vec_num_walks=args.node2vec_num_walks,
        node2vec_window=args.node2vec_window,
        node2vec_p=args.node2vec_p,
        node2vec_q=args.node2vec_q,
        node2vec_scale_embeddings=args.node2vec_scale_embeddings,
        node2vec_pca_components=args.node2vec_pca_components,
        node2vec_integration_mode=args.node2vec_integration_mode,
        node2vec_diversity_weight=args.node2vec_diversity_weight,
        use_ml=args.ml,
        ml_model_type=args.ml_model_type,
        ml_guidance_mode=args.ml_guidance_mode,
        ml_top_fraction=args.ml_top_fraction,
        ml_top_n=args.ml_top_n,
        ml_max_nodes=args.ml_max_nodes,
        ml_singleton_runs=args.ml_singleton_runs,
        ml_primary_pool_ratio=args.ml_primary_pool_ratio,
        ml_secondary_exploration_rate=args.ml_secondary_exploration_rate,
        ml_initialization_bias=args.ml_initialization_bias,
        ml_initialization_primary_rate=args.ml_initialization_primary_rate,
        ml_mutation_primary_rate=args.ml_mutation_primary_rate,
        ml_repair_primary_rate=args.ml_repair_primary_rate,
        ml_local_search_primary_rate=args.ml_local_search_primary_rate,
        ml_mutation_bias_weight=args.ml_mutation_bias_weight,
        ml_repair_bias_weight=args.ml_repair_bias_weight,
        ml_local_search_bias_weight=args.ml_local_search_bias_weight,
        fairness_first_init_enabled=args.fairness_first_init_enabled,
        fairness_first_init_slots=args.fairness_first_init_slots,
        fairness_first_init_weight=args.fairness_first_init_weight,
        weakest_group_k=args.weakest_group_k,
        weakest_group_mutation_weight=args.weakest_group_mutation_weight,
        zero_group_bonus_weight=args.zero_group_bonus_weight,
        bridge_to_weak_group_weight=args.bridge_to_weak_group_weight,
        repair_fairness_weight=args.repair_fairness_weight,
        repair_bridge_weight=args.repair_bridge_weight,
        repair_centrality_weight=args.repair_centrality_weight,
        repair_diversity_weight=args.repair_diversity_weight,
        local_search_focus_mode=args.local_search_focus_mode,
        local_search_bottom_k_groups=args.local_search_bottom_k_groups,
        local_search_max_trials=args.local_search_max_trials,
        compare_fairness_variants=args.compare_fairness_variants,
        marginal_gain_scoring_enabled=args.marginal_gain_scoring_enabled,
        marginal_gain_delta_mf_weight=args.marginal_gain_delta_mf_weight,
        marginal_gain_delta_dcv_weight=args.marginal_gain_delta_dcv_weight,
        marginal_gain_spread_weight=args.marginal_gain_spread_weight,
        local_search_swap_trials=args.local_search_swap_trials,
        local_search_candidate_pool_size=args.local_search_candidate_pool_size,
        local_search_delta_mf_weight=args.local_search_delta_mf_weight,
        local_search_delta_dcv_weight=args.local_search_delta_dcv_weight,
        local_search_overlap_penalty_weight=args.local_search_overlap_penalty_weight,
        urgency_weight_enabled=args.urgency_weight_enabled,
        urgency_exponent=args.urgency_exponent,
        weak_group_focus_weight=args.weak_group_focus_weight,
        overlap_penalty_enabled=args.overlap_penalty_enabled,
        same_community_penalty_weight=args.same_community_penalty_weight,
        neighborhood_overlap_penalty_weight=args.neighborhood_overlap_penalty_weight,
        compare_refinement_variants=args.compare_refinement_variants,
        marginal_candidate_pool_size=args.marginal_candidate_pool_size,
        mutation_candidate_pool_size=args.mutation_candidate_pool_size,
        enable_fitness_cache=args.enable_fitness_cache,
        enable_marginal_cache=args.enable_marginal_cache,
        cache_max_size=args.cache_max_size,
        local_search_early_stop_patience=args.local_search_early_stop_patience,
        local_search_use_prefilter=args.local_search_use_prefilter,
        optimization_mode=args.optimization_mode,
        refinement_intensity=args.refinement_intensity,
        marginal_eval_fraction=args.marginal_eval_fraction,
        compare_runtime_variants=args.compare_runtime_variants,
        swap_candidate_pool_size=args.swap_candidate_pool_size,
        swap_prefilter_top_k=args.swap_prefilter_top_k,
        enable_swap_cache=args.enable_swap_cache,
        local_search_failed_patience=args.local_search_failed_patience,
        local_search_first_improvement=args.local_search_first_improvement,
        full_eval_top_k=args.full_eval_top_k,
        proxy_score_weights=proxy_score_weights,
        compare_swap_runtime_variants=args.compare_swap_runtime_variants,
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
        "delta_f_score",
        "runtime_seconds",
        "candidate_pool_size",
        "zero_covered_groups_count",
        "bottom_3_avg_group_spread",
        "fraction_groups_covered",
        "weakest_groups_note",
        "node2vec_enabled",
        "node2vec_mode",
        "ml_guidance_mode",
        "ml_validation_spearman",
        "ml_validation_precision_at_budget",
        "community_modularity",
    ]
    print(format_results_report(result_frame[columns], settings))


if __name__ == "__main__":
    main()
