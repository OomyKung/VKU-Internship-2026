"""Run one FIM stack selected from benchmark fairness-runtime trade-offs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.permutations import (  # noqa: E402
    FIMPermutationRunConfig,
    format_fim_permutation_report,
    permutation_summary_columns,
    run_fim_permutation_benchmark_from_config,
)
from fim_hybrid.data_loader import load_dataset  # noqa: E402
from fim_hybrid.selection.tradeoff_selector import (  # noqa: E402
    SelectedStackConfig,
    TradeoffSelectionConfig,
    load_selected_stack_config,
    pipeline_config_for_stack,
    select_stack_from_benchmark_csv,
)
from scripts.run_experiment import build_dataset_config  # noqa: E402
from scripts.run_ml_fim_benchmark import resolve_ml_benchmark_specs  # noqa: E402


DEFAULT_REPORT_NAME = "selected_fim_stack"
_SUCCESS_STATUSES = {"ok", "success", "successful", "succeeded", "complete", "completed"}


@dataclass(slots=True)
class SelectionBundle:
    initial: SelectedStackConfig
    fallback: SelectedStackConfig | None = None
    valid_frame: pd.DataFrame | None = None


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _likely_benchmark_csvs(search_dir: Path) -> list[Path]:
    if not search_dir.is_dir():
        return []
    patterns = ("*aggregate*.csv", "*ranked*.csv", "*comparison*.csv")
    candidates: dict[Path, None] = {}
    for pattern in patterns:
        for path in search_dir.rglob(pattern):
            if path.is_file():
                candidates[path.resolve()] = None
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def _nearby_results_csvs(limit: int = 20) -> list[Path]:
    results_dir = ROOT / "results"
    if not results_dir.is_dir():
        return []
    likely = _likely_benchmark_csvs(results_dir)
    if likely:
        return likely[:limit]
    return sorted(
        (path.resolve() for path in results_dir.rglob("*.csv") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:limit]


def _format_candidate_list(paths: Sequence[Path]) -> str:
    if not paths:
        return "  (none)"
    lines = []
    for path in paths:
        lines.append(f"  - {_display_path(path)}")
    return "\n".join(lines)


def _example_command_for_csv(args: argparse.Namespace, csv_path: Path | None = None) -> str:
    path_text = _display_path(csv_path) if csv_path is not None else "results\\ml_benchmark_sweep\\<aggregate_or_ranked_csv>.csv"
    return (
        "python scripts\\run_selected_fim_stack.py "
        f"--dataset {args.dataset} "
        f"--protected-attribute {args.protected_attribute} "
        f"--budget {int(args.budget)} "
        f"--auto-select-stack-from {path_text} "
        "--selection-policy fairness_runtime_tradeoff "
        "--validate-selected-stack "
        f"--output-dir {args.output_dir}"
    )


def _resolve_benchmark_csv_input(path_value: str, args: argparse.Namespace) -> Path:
    requested = _resolve_repo_path(path_value)
    if requested is None:
        raise ValueError("--auto-select-stack-from could not be resolved.")

    if requested.is_file():
        return requested

    search_dir = requested if requested.is_dir() else requested.parent
    candidates = _likely_benchmark_csvs(search_dir) if search_dir.exists() else []
    if candidates:
        print("Benchmark CSV candidates found:")
        print(_format_candidate_list(candidates))
        chosen = candidates[0]
        reason = "requested directory" if requested.is_dir() else "requested file was not found; searched its parent directory"
        print(f"Using newest benchmark CSV from {reason}: {_display_path(chosen)}")
        return chosen

    nearby = _nearby_results_csvs()
    lines = [
        "Benchmark CSV could not be resolved.",
        f"requested_path={requested}",
        f"requested_path_exists={requested.exists()}",
        f"searched_directory={search_dir if search_dir.exists() else 'n/a'}",
        "",
        "Nearby likely CSV files found under results/:",
        _format_candidate_list(nearby),
        "",
        "Example corrected command:",
        _example_command_for_csv(args, nearby[0] if nearby else None),
    ]
    raise FileNotFoundError("\n".join(lines))


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001 - best-effort JSON conversion.
            return value
    return value


def _selection_config_from_args(args: argparse.Namespace) -> TradeoffSelectionConfig:
    close_threshold = args.fairness_close_threshold
    if close_threshold is None:
        close_threshold = args.selection_close_fscore_threshold
    return TradeoffSelectionConfig(
        close_fscore_threshold=float(close_threshold),
        min_f_score=float(args.min_f_score),
        min_mf=float(args.min_mf),
        max_dcv=float(args.max_dcv),
        min_fraction_groups_covered=float(args.min_fraction_groups_covered),
        scalability_required=bool(args.scalability_required),
        runtime_tiebreak_only=bool(args.runtime_tiebreak_only),
        warn_only_fairness_gates=bool(args.warn_only_fairness_gates),
        runtime_priority_when_close=bool(args.selection_runtime_priority_when_close),
        max_dcv_delta_vs_best=float(args.selection_max_dcv_delta_vs_best),
        min_mf_ratio_vs_best=float(args.selection_min_mf_ratio_vs_best),
        selection_policy=str(args.selection_policy),
        quality_runtime_lambda=float(args.quality_runtime_lambda),
        fallback_stack=args.fallback_stack,
    )


def _validate_selected_config(selected: SelectedStackConfig, args: argparse.Namespace) -> None:
    if str(selected.protected_attribute) != str(args.protected_attribute):
        raise ValueError(
            f"Selected config protected_attribute={selected.protected_attribute!r} does not match "
            f"CLI protected_attribute={args.protected_attribute!r}."
        )
    if int(selected.budget) != int(args.budget):
        raise ValueError(
            f"Selected config budget={selected.budget} does not match CLI budget={int(args.budget)}."
        )
    if selected.pipeline.spread_estimator_final != "monte_carlo":
        raise ValueError("Selected stack configs must use spread_estimator_final='monte_carlo'.")
    pipeline_config_for_stack(selected.selected_stack)


def _select_or_load_config(args: argparse.Namespace) -> SelectionBundle:
    if args.use_selected_stack_config and args.auto_select_stack_from:
        raise ValueError("Use either --auto-select-stack-from or --use-selected-stack-config, not both.")
    if args.use_selected_stack_config:
        selected = load_selected_stack_config(_resolve_repo_path(args.use_selected_stack_config))
        bundle = SelectionBundle(initial=selected)
    elif args.auto_select_stack_from:
        benchmark_csv_path = _resolve_benchmark_csv_input(args.auto_select_stack_from, args)
        result = select_stack_from_benchmark_csv(
            benchmark_csv_path,
            protected_attribute=args.protected_attribute,
            budget=int(args.budget),
            config=_selection_config_from_args(args),
        )
        selected = result.selected_config
        bundle = SelectionBundle(
            initial=selected,
            fallback=result.fallback_config,
            valid_frame=result.valid_frame.copy(),
        )
    else:
        raise ValueError("Provide --auto-select-stack-from or --use-selected-stack-config.")

    _validate_selected_config(selected, args)
    if bundle.fallback is not None:
        _validate_selected_config(bundle.fallback, args)
    return bundle


def _compact_benchmark_row(row: dict[str, object]) -> dict[str, object]:
    keys = [
        "stack_name",
        "status",
        "protected_attribute",
        "budget",
        "f_score",
        "mf",
        "dcv",
        "total_spread",
        "runtime_seconds",
    ]
    return {key: row[key] for key in keys if key in row}


def _selected_specs(selected: SelectedStackConfig):
    specs = resolve_ml_benchmark_specs(
        ml_stacks=[selected.selected_stack],
        include_baseline=False,
        embedding_methods=None,
        ranking_models=None,
        community_method=None,
        clustering_method=None,
        spread_estimator_search=None,
        spread_estimator_final="monte_carlo",
        optimizer_mode=None,
    )
    if len(specs) != 1:
        raise RuntimeError(f"Expected exactly one selected stack spec, got {len(specs)}.")
    return specs


def _format_selected_report(
    *,
    selected: SelectedStackConfig,
    initial_selected: SelectedStackConfig,
    fallback_selected: SelectedStackConfig | None,
    validation_summary: dict[str, object],
    result_frame: pd.DataFrame,
    permutation_report: str,
    report_name: str,
) -> str:
    row = result_frame.iloc[0] if not result_frame.empty else pd.Series(dtype=object)
    initial_row = initial_selected.benchmark_row
    fallback_row = {} if fallback_selected is None else fallback_selected.benchmark_row
    fallback_name = "n/a" if fallback_selected is None else fallback_selected.selected_stack
    candidate_summary = validation_summary.get("candidate", {})
    fallback_summary = validation_summary.get("fallback", {})
    lines = [
        "Selected FIM Stack Report",
        "=" * 72,
        "",
        "Selection stage",
        "-" * 72,
        f"report={report_name}",
        f"selected_stack_from_benchmark={initial_selected.selected_stack}",
        f"benchmark_f_score={_format_number(initial_row.get('f_score'))}",
        f"benchmark_mf={_format_number(initial_row.get('mf'))}",
        f"benchmark_dcv={_format_number(initial_row.get('dcv'))}",
        f"benchmark_runtime={_format_number(initial_row.get('runtime_seconds'), digits=3)}",
        f"benchmark_quality_runtime_score={_format_number(initial_row.get('quality_runtime_score'))}",
        f"selection_policy={initial_selected.selection_policy}",
        f"selection_reason={initial_selected.reason}",
        "",
        "Fallback",
        "-" * 72,
        f"fallback_stack={fallback_name}",
        f"fallback_benchmark_f_score={_format_number(fallback_row.get('f_score'))}",
        f"fallback_benchmark_mf={_format_number(fallback_row.get('mf'))}",
        f"fallback_benchmark_dcv={_format_number(fallback_row.get('dcv'))}",
        f"fallback_benchmark_runtime={_format_number(fallback_row.get('runtime_seconds'), digits=3)}",
        f"fallback_reason={'n/a' if fallback_selected is None else fallback_selected.reason}",
        "",
        "Validation stage",
        "-" * 72,
        f"candidate_stack={validation_summary.get('candidate_stack', initial_selected.selected_stack)}",
        f"fallback_stack={validation_summary.get('fallback_stack', fallback_name)}",
        f"validation_seeds={validation_summary.get('validation_seeds', [])}",
        f"candidate_mean_f_score={_format_number(candidate_summary.get('mean_f_score') if isinstance(candidate_summary, dict) else None)}",
        f"candidate_std_f_score={_format_number(candidate_summary.get('std_f_score') if isinstance(candidate_summary, dict) else None)}",
        f"candidate_mean_mf={_format_number(candidate_summary.get('mean_mf') if isinstance(candidate_summary, dict) else None)}",
        f"candidate_mean_dcv={_format_number(candidate_summary.get('mean_dcv') if isinstance(candidate_summary, dict) else None)}",
        f"candidate_mean_spread={_format_number(candidate_summary.get('mean_total_spread') if isinstance(candidate_summary, dict) else None)}",
        f"candidate_mean_runtime={_format_number(candidate_summary.get('mean_runtime_seconds') if isinstance(candidate_summary, dict) else None, digits=3)}",
        f"fallback_mean_f_score={_format_number(fallback_summary.get('mean_f_score') if isinstance(fallback_summary, dict) else None)}",
        f"fallback_std_f_score={_format_number(fallback_summary.get('std_f_score') if isinstance(fallback_summary, dict) else None)}",
        f"fallback_mean_mf={_format_number(fallback_summary.get('mean_mf') if isinstance(fallback_summary, dict) else None)}",
        f"fallback_mean_dcv={_format_number(fallback_summary.get('mean_dcv') if isinstance(fallback_summary, dict) else None)}",
        f"fallback_mean_spread={_format_number(fallback_summary.get('mean_total_spread') if isinstance(fallback_summary, dict) else None)}",
        f"fallback_mean_runtime={_format_number(fallback_summary.get('mean_runtime_seconds') if isinstance(fallback_summary, dict) else None, digits=3)}",
        f"f_score_gap={_format_number(validation_summary.get('f_score_gap'))}",
        f"dcv_delta={_format_number(validation_summary.get('dcv_delta'))}",
        f"mf_ratio={_format_number(validation_summary.get('mf_ratio'))}",
        f"runtime_speedup={_format_number(validation_summary.get('runtime_speedup'))}",
        f"accepted={validation_summary.get('accepted')}",
        f"final_selected_stack={selected.selected_stack}",
        f"final_reason={validation_summary.get('final_reason', selected.reason)}",
        "",
        "Selected pipeline",
        "-" * 72,
        f"decision_status={selected.decision_status}",
        f"community_method={selected.pipeline.community_method}",
        f"embedding_method={selected.pipeline.embedding_method}",
        f"ranking_model={selected.pipeline.ranking_model}",
        f"optimizer_mode={selected.pipeline.optimizer_mode}",
        f"spread_estimator_search={selected.pipeline.spread_estimator_search}",
        f"spread_estimator_final={selected.pipeline.spread_estimator_final}",
        f"use_repair={selected.pipeline.use_repair}",
        f"use_swap_local_search={selected.pipeline.use_swap_local_search}",
        "",
        "Benchmark row used for selection",
        "-" * 72,
        json.dumps(_json_ready(initial_selected.benchmark_row), indent=2),
        "",
        "Final result",
        "-" * 72,
        f"final_selected_stack={selected.selected_stack}",
        f"total_spread={row.get('total_spread', 'n/a')}",
        f"extra_spread={row.get('extra_spread', 'n/a')}",
        f"MF={row.get('mf', 'n/a')}",
        f"DCV={row.get('dcv', 'n/a')}",
        f"F-score={row.get('f_score', 'n/a')}",
        f"runtime_seconds={row.get('runtime_seconds', 'n/a')}",
        f"final_spread_estimator={row.get('final_spread_estimator', row.get('spread_estimator_final', 'n/a'))}",
        f"final_validation_warning={validation_summary.get('final_validation_warning')}",
        f"final_validation_warning_reasons={validation_summary.get('final_validation_warning_reasons', [])}",
        "",
        "Stack execution report",
        "-" * 72,
        permutation_report.rstrip(),
    ]
    return "\n".join(lines).rstrip() + "\n"


def _ensure_report_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill optional permutation-report columns for sparse mocked or failed rows."""

    normalized = frame.copy()
    for column in permutation_summary_columns():
        if column not in normalized.columns:
            normalized[column] = pd.NA
    return normalized


def _is_success_status(value: object) -> bool:
    return str(value).strip().lower() in _SUCCESS_STATUSES


def _resolve_validation_seeds(args: argparse.Namespace) -> list[int]:
    if args.validation_seeds:
        try:
            seeds = [int(value) for value in args.validation_seeds]
        except (TypeError, ValueError) as exc:
            raise ValueError("--validation-seeds must contain valid integers.") from exc
    else:
        runs = max(1, int(args.validation_runs))
        seeds = [int(args.random_seed) + offset for offset in range(runs)]
    if not seeds:
        raise ValueError("At least one validation seed is required.")
    return seeds


def _validate_budget_against_dataset(dataset_config, budget: int) -> None:
    dataset = load_dataset(dataset_config)
    node_count = int(dataset.graph.number_of_nodes())
    if int(budget) > node_count:
        raise ValueError(f"Budget {int(budget)} exceeds graph node count {node_count}.")


def _build_run_config(
    args: argparse.Namespace,
    selected: SelectedStackConfig,
    *,
    output_dir: Path,
    random_seed: int,
    mc_runs_search: int,
    mc_runs_eval: int,
    continue_on_error: bool,
) -> FIMPermutationRunConfig:
    return FIMPermutationRunConfig(
        protected_attribute=args.protected_attribute,
        budget=int(args.budget),
        propagation_probability=float(args.propagation_prob),
        mc_runs_search=int(mc_runs_search),
        mc_runs_eval=int(mc_runs_eval),
        lambda_weight=float(args.lambda_weight),
        random_seed=int(random_seed),
        output_dir=output_dir,
        continue_on_error=bool(continue_on_error),
        swap_candidate_pool_size=int(args.swap_candidate_pool_size),
        local_search_steps=int(args.local_search_steps),
        population_size=int(args.population_size),
        generations=int(args.generations),
        gnn_epochs=int(args.gnn_epochs),
        gnn_hidden_dim=int(args.gnn_hidden_dim),
        gnn_num_layers=int(args.gnn_num_layers),
        gnn_dropout=float(args.gnn_dropout),
        gnn_learning_rate=float(args.gnn_learning_rate),
        gnn_weight_decay=float(args.gnn_weight_decay),
        ris_num_rr_sets=int(args.ris_num_rr_sets),
        embedding_dim=int(args.embedding_dim),
        use_embedding_cache=bool(args.use_embedding_cache),
        use_score_cache=bool(args.use_score_cache),
        use_community_features_for_ml=bool(args.use_community_features_for_ml),
        community_feature_mode=str(args.community_feature_mode),
        use_ml_scores_in_initialization=bool(selected.pipeline.use_ml_scores_in_initialization),
        use_ml_scores_in_mutation=bool(selected.pipeline.use_ml_scores_in_mutation),
        use_ml_scores_in_repair=bool(selected.pipeline.use_ml_scores_in_repair),
        use_ml_scores_in_local_search=bool(selected.pipeline.use_ml_scores_in_local_search),
        ml_score_weight=float(selected.pipeline.ml_score_weight),
        ris_score_weight=float(selected.pipeline.ris_score_weight),
        fair_ris_score_weight=float(selected.pipeline.fair_ris_score_weight),
        fairness_bonus_weight=float(selected.pipeline.fairness_bonus_weight),
        weak_group_bonus_weight=float(selected.pipeline.weak_group_bonus_weight),
        diversity_bonus_weight=float(selected.pipeline.diversity_bonus_weight),
        protected_group_coverage_weight=float(selected.pipeline.protected_group_coverage_weight),
        dcv_penalty_weight=float(selected.pipeline.dcv_penalty_weight),
        ranking_policy=str(selected.selection_policy),
        min_f_score=float(args.min_f_score),
        min_mf=float(args.min_mf),
        max_dcv=float(args.max_dcv),
        min_fraction_groups_covered=float(args.min_fraction_groups_covered),
        fairness_close_threshold=float(args.fairness_close_threshold or args.selection_close_fscore_threshold),
        runtime_tiebreak_only=bool(args.runtime_tiebreak_only),
    )


def _run_stack_once(
    *,
    dataset_config,
    args: argparse.Namespace,
    selected: SelectedStackConfig,
    output_dir: Path,
    random_seed: int,
    mc_runs_search: int,
    mc_runs_eval: int,
    continue_on_error: bool,
) -> tuple[pd.DataFrame, FIMPermutationRunConfig]:
    run_config = _build_run_config(
        args,
        selected,
        output_dir=output_dir,
        random_seed=random_seed,
        mc_runs_search=mc_runs_search,
        mc_runs_eval=mc_runs_eval,
        continue_on_error=continue_on_error,
    )
    benchmark_result = run_fim_permutation_benchmark_from_config(
        dataset_config=dataset_config,
        config=run_config,
        permutations=_selected_specs(selected),
    )
    return _ensure_report_columns(benchmark_result.summary_frame), run_config


def _validation_result_row(
    *,
    role: str,
    selected: SelectedStackConfig,
    seed: int,
    frame: pd.DataFrame | None,
    error: Exception | None,
    args: argparse.Namespace,
    mc_runs_search: int,
    mc_runs_eval: int,
) -> dict[str, object]:
    if error is not None or frame is None or frame.empty:
        return {
            "validation_role": role,
            "dataset": args.dataset,
            "protected_attribute": args.protected_attribute,
            "budget": int(args.budget),
            "random_seed": int(seed),
            "stack_name": selected.selected_stack,
            "status": "failed",
            "skip_reason": "" if error is None else f"{type(error).__name__}: {error}",
            "total_spread": pd.NA,
            "extra_spread": pd.NA,
            "mf": pd.NA,
            "dcv": pd.NA,
            "f_score": pd.NA,
            "runtime_seconds": pd.NA,
            "final_spread_estimator": "monte_carlo",
            "mc_runs_search": int(mc_runs_search),
            "mc_runs_eval": int(mc_runs_eval),
        }

    row = frame.iloc[0]
    return {
        "validation_role": role,
        "dataset": row.get("dataset", args.dataset),
        "protected_attribute": row.get("protected_attribute", args.protected_attribute),
        "budget": int(args.budget),
        "random_seed": int(seed),
        "stack_name": selected.selected_stack,
        "status": row.get("status", "unknown"),
        "skip_reason": row.get("skip_reason", row.get("skipped_reason", "")),
        "total_spread": row.get("total_spread", pd.NA),
        "extra_spread": row.get("extra_spread", pd.NA),
        "mf": row.get("mf", pd.NA),
        "dcv": row.get("dcv", pd.NA),
        "f_score": row.get("f_score", pd.NA),
        "runtime_seconds": row.get("runtime_seconds", pd.NA),
        "final_spread_estimator": row.get("final_spread_estimator", row.get("spread_estimator_final", "monte_carlo")),
        "mc_runs_search": int(mc_runs_search),
        "mc_runs_eval": int(mc_runs_eval),
    }


def _metric_summary(rows: pd.DataFrame) -> dict[str, object]:
    success = rows[rows["status"].map(_is_success_status)].copy()
    summary: dict[str, object] = {"successful_runs": int(len(success)), "failed_runs": int(len(rows) - len(success))}
    for metric in ["f_score", "mf", "dcv", "total_spread", "extra_spread", "runtime_seconds"]:
        values = pd.to_numeric(success.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
        summary[f"mean_{metric}"] = None if values.empty else float(values.mean())
        summary[f"std_{metric}"] = None if len(values) <= 1 else float(values.std(ddof=1))
    return summary


def _format_number(value: object, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _summarize_validation(
    validation_frame: pd.DataFrame,
    *,
    candidate: SelectedStackConfig,
    fallback: SelectedStackConfig,
    seeds: list[int],
    args: argparse.Namespace,
) -> dict[str, object]:
    candidate_rows = validation_frame[validation_frame["validation_role"].eq("candidate")]
    fallback_rows = validation_frame[validation_frame["validation_role"].eq("fallback")]
    candidate_summary = _metric_summary(candidate_rows)
    fallback_summary = _metric_summary(fallback_rows)

    if int(fallback_summary["successful_runs"]) <= 0:
        raise RuntimeError(f"All fallback validation runs failed for stack '{fallback.selected_stack}'.")

    candidate_success = int(candidate_summary["successful_runs"]) > 0
    fallback_f = fallback_summary["mean_f_score"]
    fallback_dcv = fallback_summary["mean_dcv"]
    fallback_runtime = fallback_summary["mean_runtime_seconds"]
    candidate_f = candidate_summary["mean_f_score"]
    candidate_dcv = candidate_summary["mean_dcv"]
    fallback_mf = fallback_summary["mean_mf"]
    candidate_mf = candidate_summary["mean_mf"]
    candidate_runtime = candidate_summary["mean_runtime_seconds"]

    if candidate.selected_stack == fallback.selected_stack:
        accepted = False
        final_selected_stack = candidate.selected_stack
        final_reason = (
            "rejected_after_validation: selected-stack validation is experimental and self-comparison "
            "cannot prove a trade-off; use --fallback-stack with an independent stack or run the direct "
            "community+ML+SI+EA pipeline instead"
        )
    elif not candidate_success:
        accepted = False
        final_selected_stack = fallback.selected_stack
        final_reason = f"rejected_after_validation: all candidate validation runs failed; fallback_selected={fallback.selected_stack}"
    else:
        f_score_gap = float(candidate_f) - float(fallback_f)
        dcv_delta = float(candidate_dcv) - float(fallback_dcv)
        fscore_close = float(candidate_f) >= float(fallback_f) - float(args.validation_close_fscore_threshold)
        dcv_ok = float(candidate_dcv) <= float(fallback_dcv) + float(args.max_validation_dcv_delta)
        mf_ratio = (
            1.0
            if fallback_mf is None or float(fallback_mf) == 0.0
            else float(candidate_mf) / float(fallback_mf)
        )
        mf_ok = mf_ratio >= float(args.min_mf_ratio_vs_fallback)
        accepted = bool(fscore_close and dcv_ok and mf_ok)
        if accepted:
            final_selected_stack = candidate.selected_stack
            final_reason = (
                "accepted_after_validation: candidate met F-score, DCV, and MF validation gates"
            )
        else:
            final_selected_stack = fallback.selected_stack
            reasons = []
            if not fscore_close:
                reasons.append("F-score dropped beyond validation threshold")
            if not dcv_ok:
                reasons.append("DCV increased beyond validation threshold")
            if not mf_ok:
                reasons.append("MF ratio dropped below validation threshold")
            final_reason = "rejected_after_validation: " + "; ".join(reasons) + f"; fallback_selected={fallback.selected_stack}"

    f_score_gap = None if candidate_f is None or fallback_f is None else float(candidate_f) - float(fallback_f)
    dcv_delta = None if candidate_dcv is None or fallback_dcv is None else float(candidate_dcv) - float(fallback_dcv)
    mf_ratio = (
        None
        if candidate_mf is None or fallback_mf is None
        else (1.0 if float(fallback_mf) == 0.0 else float(candidate_mf) / float(fallback_mf))
    )
    runtime_speedup = (
        None
        if candidate_runtime in {None, 0} or fallback_runtime is None
        else float(fallback_runtime) / float(candidate_runtime)
    )
    return {
        "candidate_stack": candidate.selected_stack,
        "fallback_stack": fallback.selected_stack,
        "validation_seeds": seeds,
        "accepted": bool(accepted),
        "final_selected_stack": final_selected_stack,
        "final_reason": final_reason,
        "validation_close_fscore_threshold": float(args.validation_close_fscore_threshold),
        "max_validation_dcv_delta": float(args.max_validation_dcv_delta),
        "min_mf_ratio_vs_fallback": float(args.min_mf_ratio_vs_fallback),
        "f_score_gap": f_score_gap,
        "dcv_delta": dcv_delta,
        "mf_ratio": mf_ratio,
        "runtime_speedup": runtime_speedup,
        "candidate": candidate_summary,
        "fallback": fallback_summary,
    }


def _cached_validation_row(
    existing: pd.DataFrame,
    *,
    role: str,
    stack_name: str,
    seed: int,
    args: argparse.Namespace,
    mc_runs_search: int,
    mc_runs_eval: int,
) -> dict[str, object] | None:
    if existing.empty:
        return None
    mask = (
        existing.get("validation_role", pd.Series(dtype=object)).astype(str).eq(role)
        & existing.get("stack_name", pd.Series(dtype=object)).astype(str).eq(str(stack_name))
        & pd.to_numeric(existing.get("random_seed", pd.Series(dtype=object)), errors="coerce").eq(int(seed))
        & pd.to_numeric(existing.get("budget", pd.Series(dtype=object)), errors="coerce").eq(int(args.budget))
        & existing.get("protected_attribute", pd.Series(dtype=object)).astype(str).eq(str(args.protected_attribute))
        & pd.to_numeric(existing.get("mc_runs_search", pd.Series(dtype=object)), errors="coerce").eq(int(mc_runs_search))
        & pd.to_numeric(existing.get("mc_runs_eval", pd.Series(dtype=object)), errors="coerce").eq(int(mc_runs_eval))
    )
    matched = existing[mask]
    if matched.empty:
        return None
    return matched.iloc[0].to_dict()


def _run_validation(
    *,
    dataset_config,
    args: argparse.Namespace,
    candidate: SelectedStackConfig,
    fallback: SelectedStackConfig,
    output_dir: Path,
    validation_csv_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    seeds = _resolve_validation_seeds(args)
    mc_runs_search = int(args.validation_mc_runs_search or args.mc_runs_search)
    mc_runs_eval = int(args.validation_mc_runs_eval or args.mc_runs_eval)
    existing = pd.read_csv(validation_csv_path) if validation_csv_path.is_file() else pd.DataFrame()
    rows: list[dict[str, object]] = []

    for seed in seeds:
        role_specs = [("candidate", candidate)]
        if fallback.selected_stack == candidate.selected_stack:
            role_specs.append(("fallback", candidate))
        else:
            role_specs.append(("fallback", fallback))

        seed_run_cache: dict[str, tuple[pd.DataFrame | None, Exception | None]] = {}
        for role, selected in role_specs:
            cached = _cached_validation_row(
                existing,
                role=role,
                stack_name=selected.selected_stack,
                seed=seed,
                args=args,
                mc_runs_search=mc_runs_search,
                mc_runs_eval=mc_runs_eval,
            )
            if cached is not None:
                rows.append(cached)
                continue
            cache_key = f"{selected.selected_stack}:{seed}"
            if cache_key not in seed_run_cache:
                try:
                    frame, _ = _run_stack_once(
                        dataset_config=dataset_config,
                        args=args,
                        selected=selected,
                        output_dir=output_dir / "validation" / selected.selected_stack / f"seed_{seed}",
                        random_seed=int(seed),
                        mc_runs_search=mc_runs_search,
                        mc_runs_eval=mc_runs_eval,
                        continue_on_error=True,
                    )
                    seed_run_cache[cache_key] = (frame, None)
                except Exception as exc:  # noqa: BLE001 - validation should continue across seeds.
                    seed_run_cache[cache_key] = (None, exc)
            frame, error = seed_run_cache[cache_key]
            rows.append(
                _validation_result_row(
                    role=role,
                    selected=selected,
                    seed=seed,
                    frame=frame,
                    error=error,
                    args=args,
                    mc_runs_search=mc_runs_search,
                    mc_runs_eval=mc_runs_eval,
                )
            )

    validation_frame = pd.DataFrame(rows)
    validation_frame.to_csv(validation_csv_path, index=False)
    summary = _summarize_validation(
        validation_frame,
        candidate=candidate,
        fallback=fallback,
        seeds=seeds,
        args=args,
    )
    return validation_frame, summary


def _has_independent_valid_stack(valid_frame: pd.DataFrame | None, candidate_stack: str) -> bool:
    if valid_frame is None or valid_frame.empty or "stack_name" not in valid_frame.columns:
        return False
    return bool(valid_frame[~valid_frame["stack_name"].astype(str).eq(str(candidate_stack))]["stack_name"].nunique() > 0)


def _final_config_from_validation(
    *,
    candidate: SelectedStackConfig,
    fallback: SelectedStackConfig | None,
    validation_summary: dict[str, object],
) -> SelectedStackConfig:
    final_stack = str(validation_summary["final_selected_stack"])
    if final_stack == candidate.selected_stack:
        return replace(
            candidate,
            reason=str(validation_summary["final_reason"]),
            decision_status="accepted_after_validation",
        )
    if fallback is None:
        raise RuntimeError("Validation requested a fallback, but no fallback config is available.")
    return replace(
        fallback,
        reason=str(validation_summary["final_reason"]),
        decision_status="fallback_selected",
    )


def _add_final_degradation_check(
    validation_summary: dict[str, object],
    *,
    final_selected: SelectedStackConfig,
    result_frame: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    updated = dict(validation_summary)
    row = result_frame.iloc[0] if not result_frame.empty else pd.Series(dtype=object)
    benchmark_row = final_selected.benchmark_row
    final_f = pd.to_numeric(pd.Series([row.get("f_score", pd.NA)]), errors="coerce").iloc[0]
    final_mf = pd.to_numeric(pd.Series([row.get("mf", pd.NA)]), errors="coerce").iloc[0]
    final_dcv = pd.to_numeric(pd.Series([row.get("dcv", pd.NA)]), errors="coerce").iloc[0]
    final_runtime = pd.to_numeric(pd.Series([row.get("runtime_seconds", pd.NA)]), errors="coerce").iloc[0]
    benchmark_f = pd.to_numeric(pd.Series([benchmark_row.get("f_score", pd.NA)]), errors="coerce").iloc[0]
    benchmark_dcv = pd.to_numeric(pd.Series([benchmark_row.get("dcv", pd.NA)]), errors="coerce").iloc[0]

    warning_reasons: list[str] = []
    if not pd.isna(final_f) and not pd.isna(benchmark_f):
        if float(final_f) < float(benchmark_f) * float(args.min_final_vs_benchmark_ratio):
            warning_reasons.append("final F-score dropped below benchmark ratio threshold")
    if not pd.isna(final_dcv) and not pd.isna(benchmark_dcv):
        if float(final_dcv) > float(benchmark_dcv) + float(args.max_final_dcv_delta):
            warning_reasons.append("final DCV increased beyond benchmark delta threshold")

    updated["final_validation_warning"] = bool(warning_reasons)
    updated["final_validation_warning_reasons"] = warning_reasons
    updated["final_benchmark_stack"] = final_selected.selected_stack
    updated["final_benchmark_f_score"] = None if pd.isna(benchmark_f) else float(benchmark_f)
    updated["final_benchmark_dcv"] = None if pd.isna(benchmark_dcv) else float(benchmark_dcv)
    updated["final_f_score"] = None if pd.isna(final_f) else float(final_f)
    updated["final_mf"] = None if pd.isna(final_mf) else float(final_mf)
    updated["final_dcv"] = None if pd.isna(final_dcv) else float(final_dcv)
    updated["final_runtime"] = None if pd.isna(final_runtime) else float(final_runtime)
    updated["min_final_vs_benchmark_ratio"] = float(args.min_final_vs_benchmark_ratio)
    updated["max_final_dcv_delta"] = float(args.max_final_dcv_delta)
    return updated


def run_selected_fim_stack(args: argparse.Namespace) -> dict[str, Path | SelectedStackConfig | pd.DataFrame | str]:
    selection = _select_or_load_config(args)
    initial_selected = selection.initial
    output_dir = _resolve_repo_path(args.output_dir)
    if output_dir is None:
        raise ValueError("--output-dir could not be resolved.")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_config = build_dataset_config(args)
    _validate_budget_against_dataset(dataset_config, int(args.budget))

    initial_config_path = output_dir / "selected_stack_initial.json"
    initial_selected.to_json(initial_config_path)
    selected_config_path = (
        _resolve_repo_path(args.export_selected_stack_config)
        if args.export_selected_stack_config
        else output_dir / f"{args.report_name}_selected_stack_config.json"
    )
    initial_selected.to_json(selected_config_path)

    validation_csv_path = output_dir / "selected_stack_validation.csv"
    validation_summary_path = output_dir / "selected_stack_validation_summary.json"
    validation_frame = pd.DataFrame()
    validation_summary: dict[str, object] = {
        "accepted": None,
        "final_selected_stack": initial_selected.selected_stack,
        "final_reason": "selected_from_benchmark_not_validated",
        "candidate_stack": initial_selected.selected_stack,
        "fallback_stack": None,
        "validation_seeds": [],
    }
    final_selected = initial_selected
    if bool(args.validate_selected_stack):
        if selection.fallback is None:
            raise ValueError("--validate-selected-stack requires --auto-select-stack-from so a fairness-first fallback can be identified.")
        if str(args.fallback_policy) != "fairness_first":
            raise ValueError("Only fallback_policy='fairness_first' is supported.")
        if (
            selection.fallback.selected_stack == initial_selected.selected_stack
            and _has_independent_valid_stack(selection.valid_frame, initial_selected.selected_stack)
        ):
            raise RuntimeError(
                "Validation fallback equals candidate even though independent valid benchmark stacks exist. "
                "This would be a self-comparison; provide --fallback-stack or inspect the benchmark CSV."
            )
        if selection.fallback.selected_stack == initial_selected.selected_stack:
            print("No independent fallback stack available; validation is self-comparison only")
        validation_frame, validation_summary = _run_validation(
            dataset_config=dataset_config,
            args=args,
            candidate=initial_selected,
            fallback=selection.fallback,
            output_dir=output_dir,
            validation_csv_path=validation_csv_path,
        )
        final_selected = _final_config_from_validation(
            candidate=initial_selected,
            fallback=selection.fallback,
            validation_summary=validation_summary,
        )
    else:
        validation_frame.to_csv(validation_csv_path, index=False)

    validation_summary_path.write_text(json.dumps(_json_ready(validation_summary), indent=2), encoding="utf-8")

    final_config_path = output_dir / "selected_stack_final.json"
    final_selected.to_json(final_config_path)
    final_selected.to_json(selected_config_path)

    result_frame, run_config = _run_stack_once(
        dataset_config=dataset_config,
        args=args,
        selected=final_selected,
        output_dir=output_dir / "final",
        random_seed=int(args.random_seed),
        mc_runs_search=int(args.mc_runs_search),
        mc_runs_eval=int(args.mc_runs_eval),
        continue_on_error=bool(args.continue_on_error),
    )
    validation_summary = _add_final_degradation_check(
        validation_summary,
        final_selected=final_selected,
        result_frame=result_frame,
        args=args,
    )
    validation_summary_path.write_text(json.dumps(_json_ready(validation_summary), indent=2), encoding="utf-8")
    result_csv_path = output_dir / f"{args.report_name}_selected_result.csv"
    result_frame.to_csv(result_csv_path, index=False)
    permutation_report = format_fim_permutation_report(result_frame, run_config)
    report_text = _format_selected_report(
        selected=final_selected,
        initial_selected=initial_selected,
        fallback_selected=selection.fallback,
        validation_summary=validation_summary,
        result_frame=result_frame,
        permutation_report=permutation_report,
        report_name=str(args.report_name),
    )
    report_path = output_dir / f"{args.report_name}_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    fixed_report_path = output_dir / "selected_stack_report.txt"
    fixed_report_path.write_text(report_text, encoding="utf-8")

    json_path = None
    if args.save_json:
        json_path = output_dir / f"{args.report_name}_summary.json"
        payload = {
            "selected_stack_initial": initial_selected.to_dict(),
            "selected_stack_final": final_selected.to_dict(),
            "validation_summary": validation_summary,
            "result_rows": result_frame.to_dict(orient="records"),
            "output_files": {
                "selected_stack_initial": str(initial_config_path),
                "selected_stack_validation_csv": str(validation_csv_path),
                "selected_stack_validation_summary": str(validation_summary_path),
                "selected_stack_final": str(final_config_path),
                "selected_stack_config": str(selected_config_path),
                "selected_result_csv": str(result_csv_path),
                "report_md": str(report_path),
                "selected_stack_report_txt": str(fixed_report_path),
            },
        }
        json_path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")

    return {
        "selected": final_selected,
        "initial_selected": initial_selected,
        "fallback_selected": selection.fallback,
        "validation_frame": validation_frame,
        "validation_summary": validation_summary,
        "result_frame": result_frame,
        "report_text": report_text,
        "initial_config_path": initial_config_path,
        "validation_csv_path": validation_csv_path,
        "validation_summary_path": validation_summary_path,
        "final_config_path": final_config_path,
        "selected_config_path": selected_config_path,
        "result_csv_path": result_csv_path,
        "report_path": report_path,
        "fixed_report_path": fixed_report_path,
        "json_path": json_path,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="graph_spa_500_0")
    parser.add_argument("--graph-path", default=None)
    parser.add_argument("--attributes-path", "--attribute-path", dest="attributes_path", default=None)
    parser.add_argument("--dataset-format", choices=["auto", "pickle", "pkl", "txt", "csv"], default="auto")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--directed", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--source-col", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--node-id-col", default=None)
    parser.add_argument("--protected-attribute", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--auto-select-stack-from", default=None)
    parser.add_argument("--selection-policy", choices=["fairness_runtime_tradeoff", "quality_runtime", "fairness_first_priority", "professor_priority"], default="fairness_runtime_tradeoff")
    parser.add_argument("--selection-close-fscore-threshold", "--close-fscore-threshold", dest="selection_close_fscore_threshold", type=float, default=0.003)
    parser.add_argument("--selection-max-dcv-delta-vs-best", type=float, default=0.01)
    parser.add_argument("--selection-min-mf-ratio-vs-best", type=float, default=0.95)
    parser.add_argument("--fallback-stack", default=None)
    parser.add_argument("--min-f-score", type=float, default=0.0)
    parser.add_argument("--min-mf", type=float, default=0.0001)
    parser.add_argument("--max-dcv", type=float, default=0.25)
    parser.add_argument("--min-fraction-groups-covered", type=float, default=0.80)
    parser.add_argument("--fairness-close-threshold", type=float, default=None)
    parser.add_argument("--scalability-required", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runtime-tiebreak-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warn-only-fairness-gates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--selection-runtime-priority-when-close", "--runtime-priority-when-close", dest="selection_runtime_priority_when_close", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quality-runtime-lambda", type=float, default=0.0)
    parser.add_argument(
        "--validate-selected-stack",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Experimental selected-stack rerun validation. Disabled by default and self-comparison is never accepted.",
    )
    parser.add_argument("--validation-seeds", nargs="+", default=None)
    parser.add_argument("--validation-runs", type=int, default=3)
    parser.add_argument("--validation-mc-runs-eval", type=int, default=None)
    parser.add_argument("--validation-mc-runs-search", type=int, default=None)
    parser.add_argument("--validation-close-fscore-threshold", type=float, default=0.003)
    parser.add_argument("--min-validation-fscore-ratio", type=float, default=0.85)
    parser.add_argument("--max-validation-dcv-delta", type=float, default=0.010)
    parser.add_argument("--min-mf-ratio-vs-fallback", type=float, default=0.95)
    parser.add_argument("--min-final-vs-benchmark-ratio", type=float, default=0.85)
    parser.add_argument("--max-final-dcv-delta", type=float, default=0.015)
    parser.add_argument("--fallback-policy", choices=["fairness_first"], default="fairness_first")
    parser.add_argument("--export-selected-stack-config", default=None)
    parser.add_argument("--use-selected-stack-config", default=None)
    parser.add_argument("--propagation-prob", type=float, default=0.01)
    parser.add_argument("--mc-runs-search", type=int, default=20)
    parser.add_argument("--mc-runs-eval", type=int, default=100)
    parser.add_argument("--lambda-weight", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--gnn-epochs", type=int, default=30)
    parser.add_argument("--gnn-hidden-dim", type=int, default=32)
    parser.add_argument("--gnn-num-layers", type=int, default=2)
    parser.add_argument("--gnn-dropout", type=float, default=0.2)
    parser.add_argument("--gnn-learning-rate", type=float, default=1e-3)
    parser.add_argument("--gnn-weight-decay", type=float, default=5e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--ris-num-rr-sets", type=int, default=128)
    parser.add_argument("--swap-candidate-pool-size", type=int, default=24)
    parser.add_argument("--local-search-steps", type=int, default=1)
    parser.add_argument("--use-community-features-for-ml", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--community-feature-mode", choices=["none", "basic", "full"], default="basic")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-embedding-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-score-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="results/selected_stack_test")
    parser.add_argument("--report-name", default=DEFAULT_REPORT_NAME)
    parser.add_argument("--save-json", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_selected_fim_stack(args)
    selected: SelectedStackConfig = result["selected"]  # type: ignore[assignment]
    initial_selected: SelectedStackConfig = result["initial_selected"]  # type: ignore[assignment]
    validation_summary: dict[str, object] = result["validation_summary"]  # type: ignore[assignment]
    fallback_selected: SelectedStackConfig | None = result["fallback_selected"]  # type: ignore[assignment]
    frame: pd.DataFrame = result["result_frame"]  # type: ignore[assignment]
    row = frame.iloc[0] if not frame.empty else pd.Series(dtype=object)
    fallback_row = {} if fallback_selected is None else fallback_selected.benchmark_row

    print("Selected FIM stack")
    print(f"selected_stack_from_benchmark={initial_selected.selected_stack}")
    print(f"benchmark_row={json.dumps(_json_ready(_compact_benchmark_row(initial_selected.benchmark_row)), sort_keys=True)}")
    print(f"selection_reason={initial_selected.reason}")
    print(f"fallback_stack={'n/a' if fallback_selected is None else fallback_selected.selected_stack}")
    print(f"fallback_benchmark_row={json.dumps(_json_ready(_compact_benchmark_row(fallback_row)), sort_keys=True)}")
    print(f"validation_accepted={validation_summary.get('accepted')}")
    print(f"final_selected_stack={selected.selected_stack}")
    print(f"final_reason={validation_summary.get('final_reason', selected.reason)}")
    print(f"community_method={selected.pipeline.community_method}")
    print(f"embedding_method={selected.pipeline.embedding_method}")
    print(f"ranking_model={selected.pipeline.ranking_model}")
    print(f"optimizer_mode={selected.pipeline.optimizer_mode}")
    print(f"spread_estimator_search={selected.pipeline.spread_estimator_search}")
    print(f"spread_estimator_final={selected.pipeline.spread_estimator_final}")
    if validation_summary.get("candidate") is not None:
        print(
            "validation_summary="
            + json.dumps(
                _json_ready(
                    {
                        "candidate_stack": validation_summary.get("candidate_stack"),
                        "fallback_stack": validation_summary.get("fallback_stack"),
                        "f_score_gap": validation_summary.get("f_score_gap"),
                        "dcv_delta": validation_summary.get("dcv_delta"),
                        "mf_ratio": validation_summary.get("mf_ratio"),
                        "runtime_speedup": validation_summary.get("runtime_speedup"),
                    }
                ),
                sort_keys=True,
            )
        )
    print("")
    print("Final result")
    print(f"final_selected_stack={selected.selected_stack}")
    print(f"total_spread={row.get('total_spread', 'n/a')}")
    print(f"extra_spread={row.get('extra_spread', 'n/a')}")
    print(f"MF={row.get('mf', 'n/a')}")
    print(f"DCV={row.get('dcv', 'n/a')}")
    print(f"F-score={row.get('f_score', 'n/a')}")
    print(f"runtime_seconds={row.get('runtime_seconds', 'n/a')}")
    print(f"final_validation_warning={validation_summary.get('final_validation_warning')}")
    print(f"final_validation_warning_reasons={validation_summary.get('final_validation_warning_reasons', [])}")
    print("")
    print(f"Saved initial selected stack config: {result['initial_config_path']}")
    print(f"Saved validation CSV: {result['validation_csv_path']}")
    print(f"Saved validation summary: {result['validation_summary_path']}")
    print(f"Saved final selected stack config: {result['final_config_path']}")
    print(f"Saved selected result CSV: {result['result_csv_path']}")
    print(f"Saved report: {result['fixed_report_path']}")
    if result["json_path"] is not None:
        print(f"Saved JSON summary: {result['json_path']}")


if __name__ == "__main__":
    main()
