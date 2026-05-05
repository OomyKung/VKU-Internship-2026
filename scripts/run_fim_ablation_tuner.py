"""Automatic ablation and tuning system for FIM experiments.

Runs multiple pre-defined configurations for the same dataset/budget/stack
combination and selects the best using professor-priority ranking:
  1. fairness quality first (F-score, MF, DCV)
  2. scalability second
  3. spread third
  4. runtime last

Usage example:
  python scripts/run_fim_ablation_tuner.py \\
    --dataset graph_spa_500_0 --protected-attribute ethnicity --budget 40 \\
    --ml-stacks community_aware_fair_greedy graphsage_community_siea \\
                node2vec_xgboost_community_siea gcn_community_siea \\
    --random-seed 42 --mc-runs-search 20 --mc-runs-eval 300 \\
    --population-size 16 --generations 10 \\
    --ablation-configs all \\
    --output-dir results/ablation_tuner_ethnicity --save-json
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.permutations import FIMPermutationRunConfig  # noqa: E402
from fim_hybrid.priority_policy import normalize_ranking_policy  # noqa: E402
from scripts.evaluate_fim_results import InsightThresholds  # noqa: E402
from scripts.run_experiment import build_dataset_config  # noqa: E402
from scripts.run_ml_fim_benchmark import (  # noqa: E402
    MLFIMBenchmarkResult,
    resolve_ml_benchmark_specs,
    run_ml_fim_benchmark,
)


# ── Ablation config definitions ───────────────────────────────────────────────

_ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "baseline_fair_ris": {
        "description": "Standard Fair RIS with global score normalization",
        "overrides": {
            "clustering_method": "none",
            "score_normalization": "global",
            "fair_ris_score_weight": 2.0,
            "weak_group_bonus_weight": 1.2,
            "protected_group_coverage_weight": 1.0,
            "diversity_bonus_weight": 0.4,
        },
    },
    "soft_clustering": {
        "description": "KMeans embedding clustering with soft diversity bonus (no hard repair)",
        "overrides": {
            "clustering_method": "kmeans",
            "use_clustering_features": True,
            "use_cluster_diversity_bonus": True,
            "cluster_diversity_weight": 0.1,
            "cluster_repair_enabled": False,
        },
    },
    "group_stratified_hybrid": {
        "description": "Group-stratified candidate pool with hybrid score normalization",
        "overrides": {
            "use_group_stratified_candidate_pool": True,
            "score_normalization": "hybrid",
            "fair_ris_score_weight": 2.0,
            "weak_group_bonus_weight": 1.2,
        },
    },
    "aggressive_fairness": {
        "description": "High fairness weights with per-group normalization (may over-correct)",
        "overrides": {
            "use_group_stratified_candidate_pool": True,
            "score_normalization": "per_group",
            "fair_ris_score_weight": 4.0,
            "weak_group_bonus_weight": 4.0,
            "protected_group_coverage_weight": 4.0,
        },
    },
    "adaptive_fairness": {
        "description": "DCV-targeting with parity repair and hybrid normalization",
        "overrides": {
            "use_group_stratified_candidate_pool": True,
            "score_normalization": "hybrid",
            "use_dcv_parity_repair": True,
            "use_dcv_targeting": True,
            "use_fairness_first_repair": True,
        },
    },
    "gentle_fairness_decorrelated": {
        "description": (
            "Reduced, decorrelated fairness weights with coverage-normalization fix (#1) "
            "and RIS-parity-weighted weak-group bonus (#2)"
        ),
        "overrides": {
            "use_group_stratified_candidate_pool": True,
            "score_normalization": "hybrid",
            "fair_ris_score_weight": 2.0,
            "weak_group_bonus_weight": 1.5,
            "protected_group_coverage_weight": 1.5,
            "diversity_bonus_weight": 0.6,
            "use_ris_parity_weighted_weak_bonus": True,
        },
    },
}

_SAFE_CONFIGS = frozenset({"baseline_fair_ris", "soft_clustering", "group_stratified_hybrid"})
_AGGRESSIVE_CONFIGS = frozenset({"aggressive_fairness", "adaptive_fairness"})
_CONFIG_ORDER = list(_ABLATION_CONFIGS)


def _select_ablation_configs(tokens: list[str]) -> list[str]:
    normalized = [str(t).strip().lower() for t in tokens]
    if not normalized or "all" in normalized:
        return list(_CONFIG_ORDER)
    if "safe" in normalized:
        return [c for c in _CONFIG_ORDER if c in _SAFE_CONFIGS]
    if "aggressive" in normalized:
        return [c for c in _CONFIG_ORDER if c in _AGGRESSIVE_CONFIGS]
    # "custom" keyword: strip it, treat remaining tokens as config names
    config_tokens = [t for t in normalized if t != "custom"]
    if not config_tokens:
        return list(_CONFIG_ORDER)
    result: list[str] = []
    for token in config_tokens:
        if token not in _ABLATION_CONFIGS:
            supported = list(_ABLATION_CONFIGS) + ["all", "safe", "aggressive", "custom"]
            raise ValueError(f"Unknown ablation config '{token}'. Supported: {supported}")
        if token not in result:
            result.append(token)
    return result


# ── Silent-mode RunConfig print flag overrides ────────────────────────────────

_SILENT_FLAGS: dict[str, bool] = {
    "print_experiment_header": False,
    "print_budget_check": False,
    "print_stack_summary": False,
    "print_runtime_breakdown": False,
    "print_seed_diagnostics": False,
    "print_group_influence": False,
    "print_score_diagnostics": False,
    "print_optimizer_diagnostics": False,
    "print_delta_vs_baseline": False,
    "print_decision_trace": False,
    "print_collapse_explanations": False,
}


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class AblationConfigResult:
    config_name: str
    description: str
    status: str  # "ok" | "empty" | "failed"
    error_message: str
    elapsed_seconds: float
    all_rows: pd.DataFrame  # raw per-stack rows with ablation_config column
    best_row: pd.Series | None  # professor-priority #1 row for this config
    benchmark_result: MLFIMBenchmarkResult | None


# ── Utility helpers ───────────────────────────────────────────────────────────

def _safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _row_float(row: pd.Series | None, col: str) -> float:
    if row is None or col not in row:
        return float("nan")
    return _safe_float(row[col])


def _is_nan(v: float) -> bool:
    return v != v


def _gate_status(row: pd.Series | None, thresholds: InsightThresholds) -> str:
    if row is None:
        return "FAIL"
    f = _row_float(row, "f_score")
    mf = _row_float(row, "mf")
    dcv = _row_float(row, "dcv")
    if _is_nan(f) or _is_nan(mf) or _is_nan(dcv):
        return "unknown"
    if (
        f >= float(thresholds.min_f_score)
        and mf >= float(thresholds.mf_collapse_threshold)
        and dcv <= float(thresholds.dcv_collapse_threshold)
    ):
        return "PASS"
    return "FAIL"


def _fmt(value: object, spec: str = ".4f") -> str:
    try:
        return format(float(value), spec)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"


# ── Per-config runner ─────────────────────────────────────────────────────────

def run_ablation_config(
    config_name: str,
    overrides: dict[str, Any],
    description: str,
    *,
    base_config: FIMPermutationRunConfig,
    specs: list,
    dataset_config: Any,
    output_dir: Path,
    insight_thresholds: InsightThresholds,
    ranking_policy: str,
    verbose: bool,
    save_json: bool,
) -> AblationConfigResult:
    config_dir = output_dir / config_name
    extra_flags: dict[str, Any] = {} if verbose else _SILENT_FLAGS  # type: ignore[assignment]
    run_config = replace(base_config, output_dir=config_dir, **overrides, **extra_flags)
    t0 = time.monotonic()
    try:
        bm: MLFIMBenchmarkResult = run_ml_fim_benchmark(
            dataset_config=dataset_config,
            specs=specs,
            protected_attributes=[base_config.protected_attribute],
            budgets=[base_config.budget],
            seeds=[base_config.random_seed],
            base_run_config=run_config,
            output_dir=config_dir,
            report_name=config_name,
            insight_thresholds=insight_thresholds,
            ranking_policy=ranking_policy,
            save_json=save_json,
            baseline_inclusion_status="ablation",
        )
        elapsed = time.monotonic() - t0
        raw = bm.raw_comparison_frame.copy()
        raw["ablation_config"] = config_name
        ranked = (
            bm.evaluation_result.ranked_frame
            if bm.evaluation_result is not None
            else pd.DataFrame()
        )
        best_row: pd.Series | None = ranked.iloc[0] if not ranked.empty else None
        if verbose:
            print(bm.report_text.rstrip())
        return AblationConfigResult(
            config_name=config_name,
            description=description,
            status="ok" if best_row is not None else "empty",
            error_message="",
            elapsed_seconds=elapsed,
            all_rows=raw,
            best_row=best_row,
            benchmark_result=bm,
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        tb = traceback.format_exc()
        return AblationConfigResult(
            config_name=config_name,
            description=description,
            status="failed",
            error_message=f"{type(exc).__name__}: {exc}\n{tb}",
            elapsed_seconds=elapsed,
            all_rows=pd.DataFrame(),
            best_row=None,
            benchmark_result=None,
        )


# ── Professor-priority ranking across configs ─────────────────────────────────

def _rank_results(
    results: list[AblationConfigResult],
    thresholds: InsightThresholds,
) -> list[AblationConfigResult]:
    def sort_key(r: AblationConfigResult) -> tuple:
        gate = _gate_status(r.best_row, thresholds)
        gate_flag = 0 if gate == "PASS" else (1 if gate == "unknown" else 2)
        f = _row_float(r.best_row, "f_score")
        mf = _row_float(r.best_row, "mf")
        dcv = _row_float(r.best_row, "dcv")
        spread = _row_float(r.best_row, "total_spread")
        rt = _row_float(r.best_row, "runtime_seconds")
        return (
            gate_flag,
            -(f if not _is_nan(f) else -999.0),
            -(mf if not _is_nan(mf) else -999.0),
            dcv if not _is_nan(dcv) else 999.0,
            -(spread if not _is_nan(spread) else 0.0),
            rt if not _is_nan(rt) else 999.0,
        )
    return sorted(results, key=sort_key)


# ── Output builders ───────────────────────────────────────────────────────────

def _norm_abbrev(config_name: str) -> str:
    raw = str(_ABLATION_CONFIGS.get(config_name, {}).get("overrides", {}).get("score_normalization", "global"))
    return "per_grp" if raw == "per_group" else raw[:7]


def _comparison_table(
    results: list[AblationConfigResult],
    thresholds: InsightThresholds,
) -> str:
    widths = (28, 8, 30, 9, 8, 8, 9, 9, 6)
    headers = ("Config", "Norm", "Best Method", "F-score", "MF", "DCV", "Spread", "Runtime", "Gate")
    sep = "-" * sum(widths)
    lines = [
        "".join(h.ljust(w) for h, w in zip(headers, widths)),
        sep,
    ]
    for r in results:
        row = r.best_row
        gate = _gate_status(row, thresholds) if r.status == "ok" else "FAIL"
        method = str(row["stack_name"]) if (row is not None and "stack_name" in row) else "n/a"
        cells = (
            r.config_name,
            _norm_abbrev(r.config_name),
            method[:29],
            _fmt(_row_float(row, "f_score")),
            _fmt(_row_float(row, "mf")),
            _fmt(_row_float(row, "dcv")),
            _fmt(_row_float(row, "total_spread"), ".1f"),
            _fmt(_row_float(row, "runtime_seconds"), ".1f") + "s",
            gate,
        )
        lines.append("".join(c.ljust(w) for c, w in zip(cells, widths)))
    return "\n".join(lines)


def _build_report(
    all_results: list[AblationConfigResult],
    ranked: list[AblationConfigResult],
    *,
    dataset: str,
    protected_attribute: str,
    budget: int,
    stacks: list[str],
    thresholds: InsightThresholds,
) -> str:
    best = ranked[0] if ranked else None
    worst = ranked[-1] if len(ranked) > 1 else None
    sep = "-" * 60
    lines: list[str] = [
        "",
        "Ablation Tuning Summary",
        sep,
        f"Dataset                : {dataset}",
        f"Protected Attribute    : {protected_attribute}",
        f"Budget                 : {budget}",
        f"Stacks Tested          : {', '.join(stacks)}",
        f"Configs Tested         : {len(all_results)}",
        sep,
    ]

    any_passed = any(_gate_status(r.best_row, thresholds) == "PASS" for r in ranked if r.status == "ok")

    if best is not None and best.status == "ok" and best.best_row is not None:
        row = best.best_row
        gate = _gate_status(row, thresholds)
        lines += [
            f"Best Overall Config    : {best.config_name}",
            f"Best Method            : {row.get('stack_name', 'n/a')}",
            f"Best F-score           : {_fmt(_row_float(row, 'f_score'))}",
            f"Best MF                : {_fmt(_row_float(row, 'mf'))}",
            f"Best DCV               : {_fmt(_row_float(row, 'dcv'))}",
            f"Best Spread            : {_fmt(_row_float(row, 'total_spread'), '.2f')}",
            f"Runtime                : {_fmt(_row_float(row, 'runtime_seconds'), '.1f')}s",
        ]
        if not any_passed:
            lines.append(
                "WARNING: No config passed all fairness gates; "
                "showing least-bad result under professor-priority ranking."
            )
    else:
        lines.append("Best Overall Config    : n/a  (all configs failed or produced no results)")

    lines.append(sep)

    if worst is not None and worst is not best:
        if worst.status == "failed":
            worst_reason = worst.error_message.split("\n")[0][:80]
        else:
            worst_reason = (
                f"Gate={_gate_status(worst.best_row, thresholds)}, "
                f"F={_fmt(_row_float(worst.best_row, 'f_score'))}, "
                f"MF={_fmt(_row_float(worst.best_row, 'mf'))}, "
                f"DCV={_fmt(_row_float(worst.best_row, 'dcv'))}"
            )
        lines += [
            f"Worst Config           : {worst.config_name}",
            f"Reason                 : {worst_reason}",
            sep,
        ]

    if best is not None and best.status == "ok":
        best_overrides = _ABLATION_CONFIGS.get(best.config_name, {}).get("overrides", {})
        lines += [
            "Recommendation:",
            f"  Use config '{best.config_name}' for final evaluation.",
            f"  Description: {best.description}",
        ]
        if best_overrides:
            lines.append("  Active overrides:")
            for k, v in best_overrides.items():
                lines.append(f"    {k:<38} = {v}")
    else:
        lines += [
            "Recommendation:",
            "  All configs failed. Check dataset, stack list, and fairness thresholds.",
        ]

    lines += [sep, "", "Comparison Table", _comparison_table(all_results, thresholds), ""]

    # ── Per-config analysis: which modules helped or hurt ─────────────────────
    analysis_lines: list[str] = ["Config Analysis", "-" * 60]
    for r in all_results:
        gate = _gate_status(r.best_row, thresholds) if r.status == "ok" and r.best_row is not None else "FAIL"
        status_tag = f"[{gate}]" if r.status == "ok" else f"[{r.status.upper()}]"
        analysis_lines.append(f"{r.config_name}  {status_tag}")
        cfg_overrides = _ABLATION_CONFIGS.get(r.config_name, {}).get("overrides", {})
        if cfg_overrides:
            kv_parts = [f"{k}={v}" for k, v in cfg_overrides.items()]
            analysis_lines.append("  " + "  ".join(kv_parts))
        if r.best_row is not None:
            analysis_lines.append(
                f"  F={_fmt(_row_float(r.best_row, 'f_score'))}"
                f"  MF={_fmt(_row_float(r.best_row, 'mf'))}"
                f"  DCV={_fmt(_row_float(r.best_row, 'dcv'))}"
                f"  Spread={_fmt(_row_float(r.best_row, 'total_spread'), '.2f')}"
                f"  Method={r.best_row.get('stack_name', 'n/a')}"
            )
        elif r.status == "failed":
            analysis_lines.append(f"  Error: {r.error_message.split(chr(10))[0][:80]}")
        analysis_lines.append("")
    lines += analysis_lines
    return "\n".join(lines)


# ── Final confirmatory rerun ──────────────────────────────────────────────────

def run_final_rerun(
    best_result: AblationConfigResult,
    best_overrides: dict[str, Any],
    *,
    base_config: FIMPermutationRunConfig,
    specs: list,
    dataset_config: Any,
    output_dir: Path,
    insight_thresholds: InsightThresholds,
    ranking_policy: str,
    mc_runs_final: int,
) -> None:
    if best_result.best_row is None:
        print("Final rerun skipped: no best row found.")
        return
    best_stack = str(best_result.best_row.get("stack_name", ""))
    rerun_specs = [s for s in specs if s.name == best_stack] or list(specs)
    rerun_config = replace(
        base_config,
        output_dir=output_dir / "final_rerun",
        mc_runs_eval=mc_runs_final,
        **best_overrides,
    )
    print(f"\n{'=' * 60}")
    print(f"Final confirmatory rerun: {best_stack}")
    print(f"Config: {best_result.config_name}  |  mc_runs_eval={mc_runs_final}")
    print(f"{'=' * 60}\n")
    bm = run_ml_fim_benchmark(
        dataset_config=dataset_config,
        specs=rerun_specs,
        protected_attributes=[base_config.protected_attribute],
        budgets=[base_config.budget],
        seeds=[base_config.random_seed],
        base_run_config=rerun_config,
        output_dir=output_dir / "final_rerun",
        report_name="final_rerun",
        insight_thresholds=insight_thresholds,
        ranking_policy=ranking_policy,
        save_json=True,
        baseline_inclusion_status="final_rerun",
    )
    print(bm.report_text.rstrip())
    if bm.comparison_csv_path is not None:
        print(f"\nSaved final rerun: {bm.comparison_csv_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset
    parser.add_argument("--dataset", default="graph_spa_500_0")
    parser.add_argument("--graph-path", default=None)
    parser.add_argument(
        "--attributes-path", "--attribute-path", dest="attributes_path", default=None
    )
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "pickle", "pkl", "txt", "csv"],
        default="auto",
    )
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--directed", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--source-col", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--node-id-col", default=None)

    # Core experiment
    parser.add_argument("--protected-attribute", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument(
        "--ml-stacks",
        nargs="+",
        default=[
            "community_aware_fair_greedy",
            "graphsage_community_siea",
            "node2vec_xgboost_community_siea",
            "gcn_community_siea",
        ],
        help="FIM stack names to run under each ablation config.",
    )

    # Computation
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--mc-runs-search", type=int, default=20)
    parser.add_argument("--mc-runs-eval", type=int, default=300)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=5)

    # Output
    parser.add_argument("--output-dir", default="results/ablation_tuner")
    parser.add_argument("--save-json", action="store_true", default=False)
    parser.add_argument(
        "--verbose-evaluation-report",
        action="store_true",
        default=False,
        help="Print per-stack diagnostics and evaluation report for each config.",
    )

    # Ablation control
    parser.add_argument(
        "--ablation-configs",
        nargs="+",
        default=["all"],
        metavar="CONFIG",
        help=(
            "Which ablation configs to run. Options: "
            "all (default), safe, aggressive, custom, or specific names: "
            + ", ".join(_ABLATION_CONFIGS)
        ),
    )
    parser.add_argument(
        "--final-rerun-best",
        action="store_true",
        default=False,
        help=(
            "After selecting the best config, rerun only the best method "
            "with stronger final evaluation (mc_runs_eval=--final-rerun-mc-runs)."
        ),
    )
    parser.add_argument(
        "--final-rerun-mc-runs",
        type=int,
        default=1000,
        help="MC runs for the final confirmatory rerun (default: 1000).",
    )

    # Spread estimators
    parser.add_argument(
        "--spread-estimator-search",
        default="fairness_aware_ris",
        choices=["auto", "monte_carlo", "ris", "fairness_aware_ris"],
    )
    parser.add_argument("--spread-estimator-final", default="monte_carlo")
    parser.add_argument("--use-fair-ris", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--force-ris-for-all-stacks", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--require-ris", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-ris", action=argparse.BooleanOptionalAction, default=False)

    # Fairness gates
    parser.add_argument("--min-f-score", type=float, default=0.0)
    parser.add_argument("--min-mf", type=float, default=0.0001)
    parser.add_argument("--max-dcv", type=float, default=0.25)
    parser.add_argument("--min-fraction-groups-covered", type=float, default=0.80)
    parser.add_argument("--fairness-close-threshold", type=float, default=0.003)
    parser.add_argument(
        "--scalability-required", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--runtime-tiebreak-only", action=argparse.BooleanOptionalAction, default=True
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = build_dataset_config(args)
    selected_configs = _select_ablation_configs(args.ablation_configs)

    print(f"\nAblation Tuner")
    print(f"  Dataset    : {args.dataset}  |  Attribute: {args.protected_attribute}  |  Budget: {args.budget}")
    print(f"  Configs    : {len(selected_configs)}  →  {', '.join(selected_configs)}")
    print(f"  Stacks     : {', '.join(args.ml_stacks)}")
    print()

    specs = resolve_ml_benchmark_specs(
        ml_stacks=args.ml_stacks,
        include_baseline=False,
        embedding_methods=None,
        ranking_models=None,
        community_method=None,
        clustering_method=None,  # each config overrides this via RunConfig
        spread_estimator_search=args.spread_estimator_search,
        spread_estimator_final=args.spread_estimator_final,
        optimizer_mode=None,
        use_ris=bool(args.use_ris),
        use_fair_ris=bool(args.use_fair_ris),
        force_ris_for_all_stacks=bool(args.force_ris_for_all_stacks),
    )

    ranking_policy = normalize_ranking_policy("professor_priority")

    thresholds = InsightThresholds(
        close_threshold=float(args.fairness_close_threshold),
        dcv_collapse_threshold=float(args.max_dcv),
        mf_collapse_threshold=float(args.min_mf),
        min_f_score=float(args.min_f_score),
        min_fraction_groups_covered=float(args.min_fraction_groups_covered),
        scalability_required=bool(args.scalability_required),
        runtime_tiebreak_only=bool(args.runtime_tiebreak_only),
        warn_only_fairness_gates=False,
    )

    base_config = FIMPermutationRunConfig(
        protected_attribute=args.protected_attribute,
        budget=args.budget,
        mc_runs_search=args.mc_runs_search,
        mc_runs_eval=args.mc_runs_eval,
        random_seed=args.random_seed,
        population_size=args.population_size,
        generations=args.generations,
        use_ris=bool(args.use_ris),
        use_fair_ris=bool(args.use_fair_ris),
        force_ris_for_all_stacks=bool(args.force_ris_for_all_stacks),
        require_ris=bool(args.require_ris),
        ranking_policy=ranking_policy,
        min_f_score=float(args.min_f_score),
        min_mf=float(args.min_mf),
        max_dcv=float(args.max_dcv),
        min_fraction_groups_covered=float(args.min_fraction_groups_covered),
        fairness_close_threshold=float(args.fairness_close_threshold),
        scalability_required=bool(args.scalability_required),
        runtime_tiebreak_only=bool(args.runtime_tiebreak_only),
        output_dir=output_dir,
    )

    # ── Run each ablation config ──────────────────────────────────────────────
    all_results: list[AblationConfigResult] = []
    for i, name in enumerate(selected_configs, 1):
        cfg = _ABLATION_CONFIGS[name]
        print(f"[{i}/{len(selected_configs)}] {name}  —  {cfg['description']}")
        result = run_ablation_config(
            config_name=name,
            overrides=cfg["overrides"],
            description=cfg["description"],
            base_config=base_config,
            specs=specs,
            dataset_config=dataset_config,
            output_dir=output_dir,
            insight_thresholds=thresholds,
            ranking_policy=ranking_policy,
            verbose=bool(args.verbose_evaluation_report),
            save_json=bool(args.save_json),
        )
        all_results.append(result)

        if result.status == "ok" and result.best_row is not None:
            row = result.best_row
            print(
                f"  Done in {result.elapsed_seconds:.1f}s"
                f"  |  Best: {row.get('stack_name', 'n/a')}"
                f"  |  F={_fmt(_row_float(row, 'f_score'))}"
                f"  MF={_fmt(_row_float(row, 'mf'))}"
                f"  DCV={_fmt(_row_float(row, 'dcv'))}"
                f"  Gate={_gate_status(row, thresholds)}"
            )
        elif result.status == "failed":
            short_err = result.error_message.split("\n")[0][:100]
            print(f"  FAILED in {result.elapsed_seconds:.1f}s  |  {short_err}")
        else:
            print(f"  Empty result in {result.elapsed_seconds:.1f}s (no successful stack rows)")

    # ── Rank and report ───────────────────────────────────────────────────────
    ranked = _rank_results(all_results, thresholds)
    report = _build_report(
        all_results=all_results,
        ranked=ranked,
        dataset=args.dataset,
        protected_attribute=args.protected_attribute,
        budget=args.budget,
        stacks=args.ml_stacks,
        thresholds=thresholds,
    )
    print(report)

    # ── Write output files ────────────────────────────────────────────────────
    ok_with_rows = [r for r in all_results if not r.all_rows.empty]
    if ok_with_rows:
        all_rows_df = pd.concat([r.all_rows for r in ok_with_rows], ignore_index=True)
        all_rows_df.to_csv(output_dir / "ablation_results.csv", index=False)
        print(f"Saved ablation_results.csv    : {output_dir / 'ablation_results.csv'}")
    else:
        print("No successful results to save to ablation_results.csv")

    best_rows_data: list[dict[str, object]] = []
    for rank_idx, r in enumerate(ranked, 1):
        if r.best_row is not None:
            entry: dict[str, object] = {
                str(k): (v if isinstance(v, (int, float, bool, type(None))) else str(v))
                for k, v in r.best_row.items()
            }
            entry["ablation_config"] = r.config_name
            entry["ablation_rank"] = rank_idx
            entry["gate_status"] = _gate_status(r.best_row, thresholds)
            best_rows_data.append(entry)
    if best_rows_data:
        pd.DataFrame(best_rows_data).to_csv(output_dir / "ablation_ranked.csv", index=False)
        print(f"Saved ablation_ranked.csv     : {output_dir / 'ablation_ranked.csv'}")

    best = ranked[0] if ranked else None
    summary: dict[str, object] = {
        "dataset": args.dataset,
        "protected_attribute": args.protected_attribute,
        "budget": args.budget,
        "stacks_tested": args.ml_stacks,
        "configs_tested": selected_configs,
        "best_config": best.config_name if best else None,
        "best_method": (
            str(best.best_row.get("stack_name"))
            if best is not None and best.best_row is not None
            else None
        ),
        "all_configs": {
            r.config_name: {
                "description": _ABLATION_CONFIGS.get(r.config_name, {}).get("description", ""),
                "overrides": {
                    str(k): v
                    for k, v in _ABLATION_CONFIGS.get(r.config_name, {}).get("overrides", {}).items()
                },
                "status": r.status,
                "error": r.error_message.split("\n")[0][:200] if r.error_message else "",
                "elapsed_seconds": round(r.elapsed_seconds, 2),
                "best_method": (
                    str(r.best_row.get("stack_name")) if r.best_row is not None else None
                ),
                "f_score": _row_float(r.best_row, "f_score"),
                "mf": _row_float(r.best_row, "mf"),
                "dcv": _row_float(r.best_row, "dcv"),
                "total_spread": _row_float(r.best_row, "total_spread"),
                "runtime_seconds": _row_float(r.best_row, "runtime_seconds"),
                "gate_status": _gate_status(r.best_row, thresholds),
            }
            for r in all_results
        },
    }
    json_path = output_dir / "ablation_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Saved ablation_summary.json   : {json_path}")

    report_path = output_dir / "ablation_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved ablation_report.txt     : {report_path}")

    # ── Optional final confirmatory rerun ─────────────────────────────────────
    if args.final_rerun_best and best is not None and best.status == "ok":
        run_final_rerun(
            best_result=best,
            best_overrides=_ABLATION_CONFIGS[best.config_name]["overrides"],
            base_config=base_config,
            specs=specs,
            dataset_config=dataset_config,
            output_dir=output_dir,
            insight_thresholds=thresholds,
            ranking_policy=ranking_policy,
            mc_runs_final=args.final_rerun_mc_runs,
        )
    elif args.final_rerun_best:
        print("\nFinal rerun skipped: no successful config result available.")


if __name__ == "__main__":
    main()
