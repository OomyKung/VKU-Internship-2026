"""Run multi-budget, multi-seed ML-FIM benchmark sweeps and aggregate results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.data_loader import LoadedDataset, load_dataset, verify_protected_groups  # noqa: E402
from fim_hybrid.permutations import (  # noqa: E402
    FIMPermutationRunConfig,
    FIMPermutationSpec,
    permutation_summary_columns,
    run_fim_permutation_benchmark,
)
from scripts.evaluate_fim_results import (  # noqa: E402
    InsightThresholds,
    build_evaluation_report,
    build_json_summary,
    evaluate_result_frame,
)
from scripts.run_experiment import build_dataset_config  # noqa: E402
from scripts.run_ml_fim_benchmark import resolve_ml_benchmark_specs  # noqa: E402


DEFAULT_SWEEP_STACKS = (
    "graphsage_fair_ris_hybrid",
    "gcn_fair_ris_hybrid",
    "node2vec_xgboost",
    "deepwalk_mlp",
    "line_fast_ml",
)
DEFAULT_REPORT_NAME = "ml_fim_sweep"
DEFAULT_CLOSE_RESULT_THRESHOLD = 0.005
RANK_GROUP_COLUMNS = ("dataset", "protected_attribute", "budget")
RUN_GROUP_COLUMNS = ("dataset", "protected_attribute", "budget", "random_seed")
AGGREGATE_GROUP_COLUMNS = ("dataset", "protected_attribute", "budget", "stack_name")
METRIC_COLUMNS = ("f_score", "mf", "dcv", "total_spread", "extra_spread", "runtime_seconds")


@dataclass(slots=True)
class MLFIMSweepResult:
    raw_frame: pd.DataFrame
    aggregate_frame: pd.DataFrame
    ranked_aggregate_frame: pd.DataFrame
    evaluation_result: object
    report_text: str
    insights: list[str]
    recommendations: dict[str, str | None]
    warnings: list[str]
    raw_csv_path: Path
    aggregate_csv_path: Path
    ranked_aggregate_csv_path: Path
    report_path: Path
    json_path: Path | None = None


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _safe_path_token(value: object) -> str:
    text = str(value).strip()
    safe = "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in text)
    return safe.strip("._-") or "value"


def _budget_from_report_name(report_name: str) -> int | None:
    match = re.search(r"(?:^|_)budget(\d+)(?:_|$)", str(report_name), flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"(?:^|_)b(\d+)(?:_|$)", str(report_name), flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def _ordered_unique(values: Iterable[object]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        ordered.append(text)
        seen.add(text)
    return ordered


def resolve_protected_attributes(
    protected_attribute: str | None,
    protected_attributes: Sequence[str] | None,
) -> list[str]:
    values: list[object] = []
    if protected_attribute is not None:
        values.append(protected_attribute)
    if protected_attributes:
        values.extend(protected_attributes)
    resolved = _ordered_unique(values)
    if not resolved:
        raise ValueError("Provide --protected-attribute or --protected-attributes.")
    return resolved


def resolve_budget_values(budget: int | None, budgets: Sequence[int] | None) -> list[int]:
    values: list[int] = []
    if budget is not None:
        values.append(int(budget))
    if budgets:
        values.extend(int(value) for value in budgets)
    resolved: list[int] = []
    for value in values:
        if value not in resolved:
            resolved.append(value)
    if not resolved:
        raise ValueError("Provide --budget or --budgets.")
    return resolved


def resolve_seed_values(random_seed: int | None, random_seeds: Sequence[int] | None) -> list[int]:
    values: list[int] = []
    if random_seeds:
        values.extend(int(value) for value in random_seeds)
    elif random_seed is not None:
        values.append(int(random_seed))
    if not values:
        values.append(42)
    resolved: list[int] = []
    for value in values:
        if value not in resolved:
            resolved.append(value)
    return resolved


def _first_non_null(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def _coalesce_columns(frame: pd.DataFrame, primary: str, aliases: Sequence[str]) -> None:
    for alias in aliases:
        if alias not in frame.columns:
            continue
        if primary not in frame.columns:
            frame[primary] = frame[alias]
        else:
            frame[primary] = frame[primary].where(frame[primary].notna(), frame[alias])


def _add_metric_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    _coalesce_columns(result, "stack_name", ("permutation_name", "method"))
    _coalesce_columns(result, "skip_reason", ("skipped_reason",))
    _coalesce_columns(result, "total_spread", ("spread",))
    _coalesce_columns(result, "extra_spread", ("extra",))
    _coalesce_columns(result, "mf", ("MF",))
    _coalesce_columns(result, "dcv", ("DCV",))
    _coalesce_columns(result, "f_score", ("F-score", "fscore"))
    _coalesce_columns(result, "runtime_seconds", ("runtime",))
    result["spread"] = result.get("total_spread", pd.Series(pd.NA, index=result.index))
    result["extra"] = result.get("extra_spread", pd.Series(pd.NA, index=result.index))
    result["MF"] = result.get("mf", pd.Series(pd.NA, index=result.index))
    result["DCV"] = result.get("dcv", pd.Series(pd.NA, index=result.index))
    result["F-score"] = result.get("f_score", pd.Series(pd.NA, index=result.index))
    result["runtime"] = result.get("runtime_seconds", pd.Series(pd.NA, index=result.index))
    return result


def _ordered_raw_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "dataset",
        "protected_attribute",
        "budget",
        "random_seed",
        "stack_name",
        "status",
        "skip_reason",
        "spread",
        "total_spread",
        "extra",
        "extra_spread",
        "MF",
        "mf",
        "DCV",
        "dcv",
        "F-score",
        "f_score",
        "runtime",
        "runtime_seconds",
    ]
    ordered = [column for column in preferred if column in frame.columns]
    extras = [column for column in frame.columns if column not in ordered]
    return ordered + extras


def _standardize_raw_frame(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    protected_attribute: str,
    budget: int,
    random_seed: int,
) -> pd.DataFrame:
    standardized = _add_metric_aliases(frame)
    standardized["dataset"] = standardized.get("dataset", dataset_name)
    standardized["protected_attribute"] = standardized.get("protected_attribute", protected_attribute)
    standardized["budget"] = int(budget)
    standardized["random_seed"] = int(random_seed)
    standardized["status"] = standardized.get("status", "ok").fillna("ok").astype(str)
    if "skip_reason" not in standardized.columns:
        standardized["skip_reason"] = pd.NA
    if "skipped_reason" not in standardized.columns:
        standardized["skipped_reason"] = standardized["skip_reason"]
    for metric in METRIC_COLUMNS:
        if metric in standardized.columns:
            standardized[metric] = pd.to_numeric(standardized[metric], errors="coerce")
    return standardized.loc[:, _ordered_raw_columns(standardized)].copy()


def _skipped_rows_for_specs(
    *,
    dataset_name: str,
    protected_attribute: str,
    budget: int,
    random_seed: int,
    specs: Sequence[FIMPermutationSpec],
    skip_reason: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs:
        row = {column: pd.NA for column in permutation_summary_columns()}
        row.update(
            {
                "stack_name": spec.name,
                "permutation_name": spec.name,
                "status": "skipped",
                "dataset": dataset_name,
                "protected_attribute": protected_attribute,
                "budget": int(budget),
                "random_seed": int(random_seed),
                "diffusion_model": spec.diffusion_model,
                "community_method": spec.community_method,
                "community_input_mode": spec.community_input_mode,
                "clustering_method": spec.clustering_method,
                "clustering_input_mode": spec.clustering_input_mode,
                "embedding_method": spec.embedding_method,
                "ranking_model": spec.ranking_model,
                "optimizer_mode": spec.optimizer_mode,
                "debias_mode": spec.debias_mode,
                "fairness_objective": spec.fairness_objective,
                "spread_estimator_search": spec.spread_estimator_search,
                "spread_estimator_final": spec.spread_estimator_final,
                "final_spread_estimator": spec.spread_estimator_final,
                "variant_type": {
                    "baseline": "interpretable_baseline",
                    "ml": "ml_guided",
                    "fairness": "exploratory_fairness",
                    "clustering": "clustering_enhanced",
                }.get(spec.variant_family, spec.variant_family),
                "notes": spec.notes,
                "skip_reason": skip_reason,
                "skipped_reason": skip_reason,
            }
        )
        rows.append(row)
    return _standardize_raw_frame(
        pd.DataFrame(rows),
        dataset_name=dataset_name,
        protected_attribute=protected_attribute,
        budget=int(budget),
        random_seed=int(random_seed),
    )


def _run_output_dir(base_output_dir: Path, *, seed: int, budget: int, protected_attribute: str) -> Path:
    return (
        base_output_dir
        / "runs"
        / f"seed_{int(seed)}"
        / f"budget_{int(budget)}"
        / _safe_path_token(protected_attribute)
    )


def _status_is_ok(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().eq("ok")


def _f_score_winner(run_frame: pd.DataFrame) -> str | None:
    ok_rows = run_frame[_status_is_ok(run_frame["status"])].copy()
    ok_rows = ok_rows.dropna(subset=["f_score"])
    if ok_rows.empty:
        return None
    ordered = ok_rows.sort_values(
        ["f_score", "mf", "dcv", "total_spread", "runtime_seconds", "stack_name"],
        ascending=[False, False, True, False, True, True],
        na_position="last",
        kind="mergesort",
    )
    return str(ordered.iloc[0]["stack_name"])


def _runtime_winner(run_frame: pd.DataFrame) -> str | None:
    ok_rows = run_frame[_status_is_ok(run_frame["status"])].copy()
    ok_rows = ok_rows.dropna(subset=["runtime_seconds"])
    if ok_rows.empty:
        return None
    ordered = ok_rows.sort_values(
        ["runtime_seconds", "f_score", "mf", "dcv", "total_spread", "stack_name"],
        ascending=[True, False, False, True, False, True],
        na_position="last",
        kind="mergesort",
    )
    return str(ordered.iloc[0]["stack_name"])


def _win_count_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_key, run_frame in raw_frame.groupby(list(RUN_GROUP_COLUMNS), dropna=False, sort=False):
        context = dict(zip(RUN_GROUP_COLUMNS, run_key, strict=True))
        f_winner = _f_score_winner(run_frame)
        runtime_winner = _runtime_winner(run_frame)
        if f_winner is not None:
            rows.append({**context, "stack_name": f_winner, "win_type": "f_score"})
        if runtime_winner is not None:
            rows.append({**context, "stack_name": runtime_winner, "win_type": "runtime"})
    if not rows:
        return pd.DataFrame(columns=[*RANK_GROUP_COLUMNS, "stack_name", "win_count_by_f_score", "win_count_by_runtime"])
    wins = pd.DataFrame(rows)
    pivot = (
        wins.pivot_table(
            index=[*RANK_GROUP_COLUMNS, "stack_name"],
            columns="win_type",
            values="random_seed",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename(columns={"f_score": "win_count_by_f_score", "runtime": "win_count_by_runtime"})
    )
    for column in ("win_count_by_f_score", "win_count_by_runtime"):
        if column not in pivot.columns:
            pivot[column] = 0
    return pivot[[*RANK_GROUP_COLUMNS, "stack_name", "win_count_by_f_score", "win_count_by_runtime"]]


def build_sweep_aggregate(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw per-run rows into the requested ML-FIM sweep schema."""

    if raw_frame.empty:
        return pd.DataFrame()
    frame = _add_metric_aliases(raw_frame)
    for metric in METRIC_COLUMNS:
        if metric in frame.columns:
            frame[metric] = pd.to_numeric(frame[metric], errors="coerce")

    descriptor_columns = [
        column
        for column in (
            "variant_type",
            "embedding_method",
            "ranking_model",
            "community_method",
            "clustering_method",
            "optimizer_mode",
            "debias_mode",
            "spread_estimator_search",
            "spread_estimator_final",
            "key_enabled_modules",
        )
        if column in frame.columns
    ]

    grouped_all = frame.groupby(list(AGGREGATE_GROUP_COLUMNS), dropna=False, sort=False)
    count_frame = grouped_all["status"].agg(
        number_of_successful_runs=lambda value: int(_status_is_ok(value).sum()),
        number_of_failed_or_skipped_runs=lambda value: int((~_status_is_ok(value)).sum()),
    ).reset_index()
    if descriptor_columns:
        descriptors = grouped_all[descriptor_columns].agg(_first_non_null).reset_index()
        aggregate = count_frame.merge(descriptors, on=list(AGGREGATE_GROUP_COLUMNS), how="left")
    else:
        aggregate = count_frame

    ok_rows = frame[_status_is_ok(frame["status"])].copy()
    if not ok_rows.empty:
        metric_frame = (
            ok_rows.groupby(list(AGGREGATE_GROUP_COLUMNS), dropna=False, sort=False)[list(METRIC_COLUMNS)]
            .agg(["mean", "std"])
            .reset_index()
        )
        metric_frame.columns = [
            column
            if isinstance(column, str)
            else (column[0] if len(column) > 1 and column[1] == "" else f"{column[1]}_{column[0]}")
            for column in metric_frame.columns
        ]
        rename_map = {
            "mean_f_score": "mean_f_score",
            "std_f_score": "std_f_score",
            "mean_mf": "mean_mf",
            "std_mf": "std_mf",
            "mean_dcv": "mean_dcv",
            "std_dcv": "std_dcv",
            "mean_total_spread": "mean_spread",
            "std_total_spread": "std_spread",
            "mean_extra_spread": "mean_extra",
            "std_extra_spread": "std_extra",
            "mean_runtime_seconds": "mean_runtime",
            "std_runtime_seconds": "std_runtime",
        }
        metric_frame = metric_frame.rename(columns=rename_map)
        aggregate = aggregate.merge(metric_frame, on=list(AGGREGATE_GROUP_COLUMNS), how="left")

    for column in (
        "mean_f_score",
        "std_f_score",
        "mean_mf",
        "std_mf",
        "mean_dcv",
        "std_dcv",
        "mean_spread",
        "std_spread",
        "mean_extra",
        "std_extra",
        "mean_runtime",
        "std_runtime",
    ):
        if column not in aggregate.columns:
            aggregate[column] = pd.NA
    std_columns = [column for column in aggregate.columns if column.startswith("std_")]
    aggregate.loc[aggregate["number_of_successful_runs"].astype(int) > 0, std_columns] = (
        aggregate.loc[aggregate["number_of_successful_runs"].astype(int) > 0, std_columns].fillna(0.0)
    )

    wins = _win_count_frame(frame)
    aggregate = aggregate.merge(wins, on=[*RANK_GROUP_COLUMNS, "stack_name"], how="left")
    aggregate["win_count_by_f_score"] = aggregate["win_count_by_f_score"].fillna(0).astype(int)
    aggregate["win_count_by_runtime"] = aggregate["win_count_by_runtime"].fillna(0).astype(int)
    aggregate["status"] = "skipped"
    aggregate.loc[
        (aggregate["number_of_successful_runs"].astype(int) > 0)
        & (aggregate["number_of_failed_or_skipped_runs"].astype(int) == 0),
        "status",
    ] = "ok"
    aggregate.loc[
        (aggregate["number_of_successful_runs"].astype(int) > 0)
        & (aggregate["number_of_failed_or_skipped_runs"].astype(int) > 0),
        "status",
    ] = "partial"

    ordered_columns = [
        "dataset",
        "protected_attribute",
        "budget",
        "stack_name",
        "status",
        "mean_f_score",
        "std_f_score",
        "mean_mf",
        "std_mf",
        "mean_dcv",
        "std_dcv",
        "mean_spread",
        "std_spread",
        "mean_extra",
        "std_extra",
        "mean_runtime",
        "std_runtime",
        "win_count_by_f_score",
        "win_count_by_runtime",
        "number_of_successful_runs",
        "number_of_failed_or_skipped_runs",
    ]
    ordered = [column for column in ordered_columns if column in aggregate.columns]
    extras = [column for column in aggregate.columns if column not in ordered]
    return aggregate.loc[:, ordered + extras].copy()


def rank_sweep_aggregate(aggregate_frame: pd.DataFrame) -> pd.DataFrame:
    """Rank each dataset/attribute/budget aggregate group by the default FIM ordering."""

    if aggregate_frame.empty:
        return aggregate_frame.copy()
    ranked_groups: list[pd.DataFrame] = []
    for _, group in aggregate_frame.groupby(list(RANK_GROUP_COLUMNS), dropna=False, sort=False):
        sortable = group.copy()
        sortable["_has_success"] = sortable["number_of_successful_runs"].astype(int) > 0
        ordered = sortable.sort_values(
            [
                "_has_success",
                "mean_f_score",
                "mean_mf",
                "mean_dcv",
                "mean_spread",
                "mean_runtime",
                "stack_name",
            ],
            ascending=[False, False, False, True, False, True, True],
            na_position="last",
            kind="mergesort",
        ).drop(columns=["_has_success"])
        ordered.insert(3, "rank", range(1, len(ordered) + 1))
        ranked_groups.append(ordered)
    return pd.concat(ranked_groups, ignore_index=True)


def _is_baseline_row(row: pd.Series) -> bool:
    variant = str(row.get("variant_type", "")).strip().lower()
    embedding = str(row.get("embedding_method", "")).strip().lower()
    stack_name = str(row.get("stack_name", "")).strip().lower()
    return variant == "interpretable_baseline" or embedding in {"", "none", "nan", "<na>"} or "community_aware" in stack_name


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights.astype(float) > 0)
    if not bool(valid.any()):
        return float("nan")
    return float((values[valid].astype(float) * weights[valid].astype(float)).sum() / weights[valid].astype(float).sum())


def _stack_rollup(aggregate_frame: pd.DataFrame) -> pd.DataFrame:
    success_rows = aggregate_frame[aggregate_frame["number_of_successful_runs"].astype(int) > 0].copy()
    if success_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for stack_name, stack_frame in success_rows.groupby("stack_name", dropna=False, sort=False):
        weights = stack_frame["number_of_successful_runs"].astype(float)
        first_row = stack_frame.iloc[0]
        rows.append(
            {
                "stack_name": stack_name,
                "variant_type": first_row.get("variant_type", pd.NA),
                "embedding_method": first_row.get("embedding_method", pd.NA),
                "mean_f_score": _weighted_mean(stack_frame["mean_f_score"], weights),
                "mean_mf": _weighted_mean(stack_frame["mean_mf"], weights),
                "mean_dcv": _weighted_mean(stack_frame["mean_dcv"], weights),
                "mean_spread": _weighted_mean(stack_frame["mean_spread"], weights),
                "mean_runtime": _weighted_mean(stack_frame["mean_runtime"], weights),
                "win_count_by_f_score": int(stack_frame["win_count_by_f_score"].sum()),
                "win_count_by_runtime": int(stack_frame["win_count_by_runtime"].sum()),
                "number_of_successful_runs": int(stack_frame["number_of_successful_runs"].sum()),
                "number_of_failed_or_skipped_runs": int(stack_frame["number_of_failed_or_skipped_runs"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _sort_rollup(frame: pd.DataFrame, columns: Sequence[str], ascending: Sequence[bool]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.sort_values(
        list(columns) + ["stack_name"],
        ascending=list(ascending) + [True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _stack_name_at(frame: pd.DataFrame, index: int = 0) -> str | None:
    if frame.empty or len(frame) <= index:
        return None
    return str(frame.iloc[index]["stack_name"])


def build_sweep_insights(
    aggregate_frame: pd.DataFrame,
    ranked_aggregate_frame: pd.DataFrame,
    *,
    thresholds: InsightThresholds,
) -> tuple[list[str], dict[str, str | None]]:
    if aggregate_frame.empty or ranked_aggregate_frame.empty:
        return ["No aggregate rows were available for insight generation."], {
            "best_overall_stack": None,
            "best_ml_only_stack": None,
            "best_fairness_first_stack": None,
            "best_spread_first_stack": None,
            "best_runtime_first_stack": None,
            "best_quality_runtime_tradeoff": None,
            "final_recommendation": None,
        }

    rollup = _stack_rollup(aggregate_frame)
    if rollup.empty:
        return ["No successful runs were available for insight generation."], {
            "best_overall_stack": None,
            "best_ml_only_stack": None,
            "best_fairness_first_stack": None,
            "best_spread_first_stack": None,
            "best_runtime_first_stack": None,
            "best_quality_runtime_tradeoff": None,
            "final_recommendation": None,
        }

    overall = _sort_rollup(
        rollup,
        ["mean_f_score", "mean_mf", "mean_dcv", "mean_spread", "mean_runtime"],
        [False, False, True, False, True],
    )
    ml_only = overall[~overall.apply(_is_baseline_row, axis=1)].copy()
    fairness = _sort_rollup(rollup, ["mean_mf", "mean_dcv", "mean_f_score", "mean_spread", "mean_runtime"], [False, True, False, False, True])
    spread = _sort_rollup(rollup, ["mean_spread", "mean_f_score", "mean_mf", "mean_dcv", "mean_runtime"], [False, False, False, True, True])
    runtime = _sort_rollup(rollup, ["mean_runtime", "mean_f_score", "mean_mf", "mean_dcv", "mean_spread"], [True, False, False, True, False])

    best_overall = _stack_name_at(overall)
    best_ml = _stack_name_at(ml_only)
    best_fairness = _stack_name_at(fairness)
    best_spread = _stack_name_at(spread)
    best_runtime = _stack_name_at(runtime)

    tradeoff = best_overall
    if best_overall is not None and not runtime.empty:
        best_row = overall.iloc[0]
        for _, row in runtime.iterrows():
            if str(row["stack_name"]) == best_overall:
                continue
            if pd.isna(row.get("mean_f_score")) or pd.isna(best_row.get("mean_f_score")):
                continue
            f_gap = float(best_row["mean_f_score"]) - float(row["mean_f_score"])
            runtime_ratio = (
                float(row["mean_runtime"]) / float(best_row["mean_runtime"])
                if pd.notna(row.get("mean_runtime")) and pd.notna(best_row.get("mean_runtime")) and float(best_row["mean_runtime"]) > 0
                else 1.0
            )
            if f_gap <= max(float(thresholds.close_threshold), DEFAULT_CLOSE_RESULT_THRESHOLD) and runtime_ratio <= 0.8:
                tradeoff = str(row["stack_name"])
                break

    f_win_totals = rollup.sort_values(["win_count_by_f_score", "mean_f_score"], ascending=[False, False], kind="mergesort")
    runtime_win_totals = rollup.sort_values(["win_count_by_runtime", "mean_runtime"], ascending=[False, True], kind="mergesort")
    insights = [
        f"Best overall stack: {best_overall or 'n/a'}",
        f"Best ML-only stack: {best_ml or 'n/a'}",
        f"Best fairness-first stack: {best_fairness or 'n/a'}",
        f"Best spread-first stack: {best_spread or 'n/a'}",
        f"Best runtime-first stack: {best_runtime or 'n/a'}",
        f"Best quality-runtime trade-off: {tradeoff or 'n/a'}",
    ]
    if not f_win_totals.empty:
        insights.append(
            f"{f_win_totals.iloc[0]['stack_name']} wins most often by F-score "
            f"({int(f_win_totals.iloc[0]['win_count_by_f_score'])} run group win(s))."
        )
    if not runtime_win_totals.empty:
        insights.append(
            f"{runtime_win_totals.iloc[0]['stack_name']} wins most often by runtime "
            f"({int(runtime_win_totals.iloc[0]['win_count_by_runtime'])} run group win(s))."
        )

    if len(overall) >= 2 and pd.notna(overall.iloc[0].get("mean_f_score")) and pd.notna(overall.iloc[1].get("mean_f_score")):
        gap = abs(float(overall.iloc[0]["mean_f_score"]) - float(overall.iloc[1]["mean_f_score"]))
        if gap < DEFAULT_CLOSE_RESULT_THRESHOLD:
            insights.append(f"Close result; not decisive. Top two mean F-scores differ by {gap:.4f}.")

    baseline_rows = rollup[rollup.apply(_is_baseline_row, axis=1)].copy()
    graphsage_rows = rollup[rollup["stack_name"].astype(str).eq("graphsage_fair_ris_hybrid")]
    if not baseline_rows.empty and not graphsage_rows.empty:
        baseline = _sort_rollup(baseline_rows, ["mean_f_score", "mean_mf", "mean_dcv", "mean_runtime"], [False, False, True, True]).iloc[0]
        graphsage = graphsage_rows.iloc[0]
        if pd.notna(baseline.get("mean_f_score")) and pd.notna(graphsage.get("mean_f_score")):
            gap = float(baseline["mean_f_score"]) - float(graphsage["mean_f_score"])
            if gap <= DEFAULT_CLOSE_RESULT_THRESHOLD:
                insights.append("GraphSAGE+RIS is consistently close to the best baseline on mean F-score.")
        if (
            pd.notna(baseline.get("mean_runtime"))
            and pd.notna(graphsage.get("mean_runtime"))
            and float(baseline["mean_runtime"]) >= 1.5 * float(graphsage["mean_runtime"])
            and float(baseline.get("mean_f_score", float("-inf"))) >= float(graphsage.get("mean_f_score", float("-inf")))
        ):
            insights.append("Baseline is best quality but slow relative to GraphSAGE+RIS.")

    collapse_rows = rollup[
        (pd.to_numeric(rollup["mean_mf"], errors="coerce").fillna(float("inf")) <= float(thresholds.mf_collapse_threshold))
        | (pd.to_numeric(rollup["mean_dcv"], errors="coerce").fillna(float("-inf")) >= float(thresholds.dcv_collapse_threshold))
        | (pd.to_numeric(rollup["mean_f_score"], errors="coerce").fillna(0.0) < 0.0)
    ]
    if not collapse_rows.empty:
        shallow = collapse_rows[
            collapse_rows["stack_name"].astype(str).isin({"node2vec_xgboost", "deepwalk_mlp", "line_fast_ml"})
        ]
        if not shallow.empty:
            insights.append("Shallow ML fairness collapse warning: " + ", ".join(shallow["stack_name"].astype(str).tolist()))
        else:
            insights.append("Fairness collapse warning: " + ", ".join(collapse_rows["stack_name"].astype(str).tolist()))

    winners_by_budget = ranked_aggregate_frame[
        (ranked_aggregate_frame["rank"].astype(int) == 1)
        & (ranked_aggregate_frame["number_of_successful_runs"].astype(int) > 0)
    ][["budget", "stack_name"]]
    if winners_by_budget["stack_name"].nunique(dropna=True) > 1:
        pairs = [f"budget {int(row.budget)} -> {row.stack_name}" for row in winners_by_budget.itertuples(index=False)]
        insights.append("Budget size changes the winner: " + "; ".join(pairs))
        budgets = sorted(pd.to_numeric(winners_by_budget["budget"], errors="coerce").dropna().astype(int).unique())
        if budgets:
            midpoint = budgets[len(budgets) // 2]
            for stack_name, stack_wins in winners_by_budget.groupby("stack_name", sort=False):
                win_budgets = sorted(pd.to_numeric(stack_wins["budget"], errors="coerce").dropna().astype(int).unique())
                if win_budgets and min(win_budgets) > midpoint:
                    insights.append(f"{stack_name} wins only at larger budgets in this sweep.")

    top_rank_std = ranked_aggregate_frame[
        (ranked_aggregate_frame["rank"].astype(int) == 1)
        & (pd.to_numeric(ranked_aggregate_frame.get("std_f_score"), errors="coerce") > float(thresholds.close_threshold))
    ]
    if not top_rank_std.empty:
        insights.append("Some winner differences are seed-sensitive based on F-score standard deviation.")
    else:
        insights.append("Winner differences look stable under the configured seeds.")

    if best_overall is not None and tradeoff is not None and best_overall != tradeoff:
        insights.append(f"{best_overall} is best quality but slow; {tradeoff} is the best practical ML stack.")

    recommendations = {
        "best_overall_stack": best_overall,
        "best_ml_only_stack": best_ml,
        "best_fairness_first_stack": best_fairness,
        "best_spread_first_stack": best_spread,
        "best_runtime_first_stack": best_runtime,
        "best_quality_runtime_tradeoff": tradeoff,
        "final_recommendation": tradeoff or best_overall,
    }
    return list(dict.fromkeys(insights)), recommendations


def _format_float(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def format_aggregate_table(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    if frame.empty:
        return "No aggregate rows were produced."
    lines = [
        "dataset          protected_attribute  budget  rank  stack_name                    mean_f_score  mean_mf  mean_dcv  mean_spread  mean_runtime  f_wins  runtime_wins  ok  failed"
    ]
    for row in frame.head(max_rows).itertuples(index=False):
        lines.append(
            f"{str(row.dataset):<16} "
            f"{str(row.protected_attribute):<20} "
            f"{int(row.budget):<7} "
            f"{int(row.rank):<5} "
            f"{str(row.stack_name):<29} "
            f"{_format_float(row.mean_f_score):<13} "
            f"{_format_float(row.mean_mf):<8} "
            f"{_format_float(row.mean_dcv):<8} "
            f"{_format_float(row.mean_spread, digits=3):<12} "
            f"{_format_float(row.mean_runtime, digits=3):<13} "
            f"{int(row.win_count_by_f_score):<7} "
            f"{int(row.win_count_by_runtime):<13} "
            f"{int(row.number_of_successful_runs):<3} "
            f"{int(row.number_of_failed_or_skipped_runs)}"
        )
    if len(frame) > max_rows:
        lines.append(f"... {len(frame) - max_rows} more row(s)")
    return "\n".join(lines)


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
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


def build_sweep_report(
    *,
    dataset_name: str,
    protected_attributes: Sequence[str],
    budgets: Sequence[int],
    seeds: Sequence[int],
    stack_names: Sequence[str],
    raw_frame: pd.DataFrame,
    aggregate_frame: pd.DataFrame,
    ranked_aggregate_frame: pd.DataFrame,
    evaluation_result: object,
    insights: Sequence[str],
    recommendations: dict[str, str | None],
    warnings: Sequence[str],
    report_name: str,
) -> str:
    lines = [
        "# ML-FIM Sweep Report",
        "",
        f"- Report: {report_name}",
        f"- Dataset: {dataset_name}",
        f"- Protected attributes: {', '.join(protected_attributes)}",
        f"- Budgets: {', '.join(str(value) for value in budgets)}",
        f"- Random seeds: {', '.join(str(value) for value in seeds)}",
        f"- Stacks: {', '.join(stack_names)}",
        f"- Raw rows: {len(raw_frame)}",
        "",
    ]
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    lines.extend(["## Aggregate Ranking", "", "```text", format_aggregate_table(ranked_aggregate_frame), "```", ""])
    lines.extend(["## Instant Insights", ""])
    lines.extend(f"- {line}" for line in insights)
    lines.append("")
    lines.extend(["## Final Recommendation", ""])
    for key, value in recommendations.items():
        lines.append(f"- {key}={value or 'n/a'}")
    lines.append("")
    lines.extend(["## Existing Evaluation Report", ""])
    lines.append(
        build_evaluation_report(
            evaluation_result,
            input_paths=["ml_fim_sweep_raw_runs"],
            rank_by="fim_default",
            report_name=report_name,
        ).rstrip()
    )
    return "\n".join(lines).rstrip() + "\n"


def _report_name_warnings(report_name: str, budgets: Sequence[int]) -> list[str]:
    budget_in_name = _budget_from_report_name(report_name)
    if budget_in_name is None:
        return []
    actual_budgets = {int(value) for value in budgets}
    if actual_budgets == {budget_in_name}:
        return []
    return [
        f"report_name contains budget {budget_in_name}, but actual sweep budgets are {sorted(actual_budgets)}."
    ]


def run_ml_fim_sweep(
    *,
    dataset_config: object,
    specs: Sequence[FIMPermutationSpec],
    protected_attributes: Sequence[str],
    budgets: Sequence[int],
    seeds: Sequence[int],
    base_run_config: FIMPermutationRunConfig,
    output_dir: Path,
    report_name: str,
    insight_thresholds: InsightThresholds,
    save_json: bool = False,
    print_progress: bool = True,
) -> MLFIMSweepResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset: LoadedDataset = load_dataset(dataset_config)
    node_count = int(dataset.graph.number_of_nodes())
    warnings = _report_name_warnings(report_name, budgets)
    raw_frames: list[pd.DataFrame] = []
    total_runs = len(protected_attributes) * len(budgets) * len(seeds)
    run_index = 0

    if print_progress:
        print("ML-FIM sweep configuration")
        print(f"dataset={dataset.name} nodes={node_count} edges={dataset.graph.number_of_edges()}")
        print(f"protected_attributes={list(protected_attributes)}")
        print(f"budgets={list(budgets)}")
        print(f"random_seeds={list(seeds)}")
        print(f"stacks={[spec.name for spec in specs]}")
        for warning in warnings:
            print(f"WARNING: {warning}")
        print("")

    for protected_attribute in protected_attributes:
        try:
            protected_group_report = verify_protected_groups(dataset, protected_attribute)
            attribute_error: str | None = None
        except Exception as exc:  # noqa: BLE001 - sweep should preserve skipped rows.
            protected_group_report = None
            attribute_error = f"protected_attribute_validation_failed: {type(exc).__name__}: {exc}"

        for budget in budgets:
            for seed in seeds:
                run_index += 1
                prefix = f"[{run_index}/{total_runs}] attr={protected_attribute} budget={budget} seed={seed}"
                if attribute_error is not None:
                    if print_progress:
                        print(f"{prefix} skipped: {attribute_error}")
                    raw_frames.append(
                        _skipped_rows_for_specs(
                            dataset_name=dataset.name,
                            protected_attribute=protected_attribute,
                            budget=int(budget),
                            random_seed=int(seed),
                            specs=specs,
                            skip_reason=attribute_error,
                        )
                    )
                    continue
                if int(budget) < 1:
                    skip_reason = f"budget {budget} is invalid; budget must be at least 1."
                    if print_progress:
                        print(f"{prefix} skipped: {skip_reason}")
                    raw_frames.append(
                        _skipped_rows_for_specs(
                            dataset_name=dataset.name,
                            protected_attribute=protected_attribute,
                            budget=int(budget),
                            random_seed=int(seed),
                            specs=specs,
                            skip_reason=skip_reason,
                        )
                    )
                    continue
                if int(budget) > node_count:
                    skip_reason = f"budget {budget} exceeds graph node count {node_count}."
                    if print_progress:
                        print(f"{prefix} skipped: {skip_reason}")
                    raw_frames.append(
                        _skipped_rows_for_specs(
                            dataset_name=dataset.name,
                            protected_attribute=protected_attribute,
                            budget=int(budget),
                            random_seed=int(seed),
                            specs=specs,
                            skip_reason=skip_reason,
                        )
                    )
                    continue

                run_output_dir = _run_output_dir(
                    output_dir,
                    seed=int(seed),
                    budget=int(budget),
                    protected_attribute=protected_attribute,
                )
                run_config = replace(
                    base_run_config,
                    protected_attribute=protected_attribute,
                    budget=int(budget),
                    random_seed=int(seed),
                    output_dir=run_output_dir,
                )
                if print_progress:
                    print(f"{prefix} running")
                try:
                    benchmark_result = run_fim_permutation_benchmark(
                        dataset=dataset,
                        protected_group_report=protected_group_report,
                        config=run_config,
                        permutations=specs,
                    )
                    frame = _standardize_raw_frame(
                        benchmark_result.summary_frame,
                        dataset_name=dataset.name,
                        protected_attribute=protected_attribute,
                        budget=int(budget),
                        random_seed=int(seed),
                    )
                    raw_frames.append(frame)
                    if print_progress:
                        ok_count = int(_status_is_ok(frame["status"]).sum())
                        skipped_count = int(len(frame) - ok_count)
                        print(f"{prefix} completed: ok={ok_count} skipped={skipped_count}")
                except Exception as exc:  # noqa: BLE001 - optional full-run skip for sweep resilience.
                    if not bool(base_run_config.continue_on_error):
                        raise
                    skip_reason = f"run_failed: {type(exc).__name__}: {exc}"
                    if print_progress:
                        print(f"{prefix} skipped: {skip_reason}")
                    raw_frames.append(
                        _skipped_rows_for_specs(
                            dataset_name=dataset.name,
                            protected_attribute=protected_attribute,
                            budget=int(budget),
                            random_seed=int(seed),
                            specs=specs,
                            skip_reason=skip_reason,
                        )
                    )

    if raw_frames:
        raw_frame = pd.concat(raw_frames, ignore_index=True)
    else:
        raw_frame = pd.DataFrame()
    aggregate_frame = build_sweep_aggregate(raw_frame)
    ranked_aggregate_frame = rank_sweep_aggregate(aggregate_frame)
    evaluation_result = evaluate_result_frame(
        raw_frame,
        rank_by="fim_default",
        group_by=list(RANK_GROUP_COLUMNS),
        thresholds=insight_thresholds,
    )
    insights, recommendations = build_sweep_insights(
        aggregate_frame,
        ranked_aggregate_frame,
        thresholds=insight_thresholds,
    )
    report_text = build_sweep_report(
        dataset_name=dataset.name,
        protected_attributes=protected_attributes,
        budgets=budgets,
        seeds=seeds,
        stack_names=[spec.name for spec in specs],
        raw_frame=raw_frame,
        aggregate_frame=aggregate_frame,
        ranked_aggregate_frame=ranked_aggregate_frame,
        evaluation_result=evaluation_result,
        insights=insights,
        recommendations=recommendations,
        warnings=warnings,
        report_name=report_name,
    )

    raw_csv_path = output_dir / f"{report_name}_raw_runs.csv"
    aggregate_csv_path = output_dir / f"{report_name}_aggregate.csv"
    ranked_aggregate_csv_path = output_dir / f"{report_name}_ranked_aggregate.csv"
    report_path = output_dir / f"{report_name}_report.md"
    raw_frame.to_csv(raw_csv_path, index=False)
    aggregate_frame.to_csv(aggregate_csv_path, index=False)
    ranked_aggregate_frame.to_csv(ranked_aggregate_csv_path, index=False)
    report_path.write_text(report_text, encoding="utf-8")

    json_path = None
    if save_json:
        json_path = output_dir / f"{report_name}_summary.json"
        payload = {
            "configuration": {
                "dataset": dataset.name,
                "node_count": node_count,
                "protected_attributes": list(protected_attributes),
                "budgets": [int(value) for value in budgets],
                "random_seeds": [int(value) for value in seeds],
                "stacks": [spec.name for spec in specs],
            },
            "warnings": list(warnings),
            "insights": list(insights),
            "recommendations": dict(recommendations),
            "output_files": {
                "raw_csv": str(raw_csv_path),
                "aggregate_csv": str(aggregate_csv_path),
                "ranked_aggregate_csv": str(ranked_aggregate_csv_path),
                "report_md": str(report_path),
            },
            "evaluation_summary": build_json_summary(evaluation_result),
            "top_aggregate_rows": ranked_aggregate_frame.head(20).to_dict(orient="records"),
        }
        json_path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")

    return MLFIMSweepResult(
        raw_frame=raw_frame,
        aggregate_frame=aggregate_frame,
        ranked_aggregate_frame=ranked_aggregate_frame,
        evaluation_result=evaluation_result,
        report_text=report_text,
        insights=list(insights),
        recommendations=dict(recommendations),
        warnings=list(warnings),
        raw_csv_path=raw_csv_path,
        aggregate_csv_path=aggregate_csv_path,
        ranked_aggregate_csv_path=ranked_aggregate_csv_path,
        report_path=report_path,
        json_path=json_path,
    )


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
    parser.add_argument("--protected-attribute", default=None)
    parser.add_argument("--protected-attributes", nargs="+", default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--budgets", nargs="+", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--ml-stacks",
        nargs="+",
        default=list(DEFAULT_SWEEP_STACKS),
        help="Named ML stacks to sweep. Defaults to the five requested ML stacks.",
    )
    parser.add_argument("--embedding-methods", nargs="+", default=None)
    parser.add_argument("--ranking-models", nargs="+", default=None)
    parser.add_argument("--community-method", default=None)
    parser.add_argument("--clustering-method", default=None)
    parser.add_argument("--spread-estimator-search", default=None)
    parser.add_argument("--spread-estimator-final", default="monte_carlo")
    parser.add_argument("--optimizer-mode", default=None)
    parser.add_argument("--propagation-prob", type=float, default=0.01)
    parser.add_argument("--mc-runs-search", type=int, default=20)
    parser.add_argument("--mc-runs-eval", type=int, default=100)
    parser.add_argument("--lambda-weight", type=float, default=0.5)
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
    parser.add_argument("--ranking-top-fraction", type=float, default=0.5)
    parser.add_argument("--ranking-top-n", type=int, default=None)
    parser.add_argument("--ranking-max-nodes", type=int, default=None)
    parser.add_argument("--clustering-n-clusters", type=int, default=None)
    parser.add_argument("--clustering-min-cluster-size", type=int, default=None)
    parser.add_argument("--output-dir", default="results/ml_benchmark_sweep")
    parser.add_argument("--report-name", default=DEFAULT_REPORT_NAME)
    parser.add_argument("--close-threshold", type=float, default=0.002)
    parser.add_argument("--dcv-collapse-threshold", type=float, default=0.25)
    parser.add_argument("--mf-collapse-threshold", type=float, default=0.001)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-baseline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-embedding-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-score-cache", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--save-json", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_config = build_dataset_config(args)
    protected_attributes = resolve_protected_attributes(args.protected_attribute, args.protected_attributes)
    budgets = resolve_budget_values(args.budget, args.budgets)
    seeds = resolve_seed_values(args.random_seed, args.random_seeds)
    specs = resolve_ml_benchmark_specs(
        ml_stacks=args.ml_stacks,
        include_baseline=bool(args.include_baseline),
        embedding_methods=args.embedding_methods,
        ranking_models=args.ranking_models,
        community_method=args.community_method,
        clustering_method=args.clustering_method,
        spread_estimator_search=args.spread_estimator_search,
        spread_estimator_final=args.spread_estimator_final,
        optimizer_mode=args.optimizer_mode,
    )
    output_dir = _resolve_repo_path(args.output_dir)
    if output_dir is None:
        raise ValueError("--output-dir could not be resolved.")
    base_run_config = FIMPermutationRunConfig(
        protected_attribute=protected_attributes[0],
        budget=int(budgets[0]),
        propagation_probability=float(args.propagation_prob),
        mc_runs_search=int(args.mc_runs_search),
        mc_runs_eval=int(args.mc_runs_eval),
        lambda_weight=float(args.lambda_weight),
        random_seed=int(seeds[0]),
        output_dir=output_dir,
        continue_on_error=bool(args.continue_on_error),
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
        ranking_top_fraction=args.ranking_top_fraction,
        ranking_top_n=args.ranking_top_n,
        ranking_max_nodes=args.ranking_max_nodes,
        clustering_n_clusters=args.clustering_n_clusters,
        clustering_min_cluster_size=args.clustering_min_cluster_size,
        embedding_dim=int(args.embedding_dim),
        use_embedding_cache=bool(args.use_embedding_cache),
        use_score_cache=bool(args.use_score_cache),
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
    thresholds = InsightThresholds(
        close_threshold=float(args.close_threshold),
        dcv_collapse_threshold=float(args.dcv_collapse_threshold),
        mf_collapse_threshold=float(args.mf_collapse_threshold),
    )
    result = run_ml_fim_sweep(
        dataset_config=dataset_config,
        specs=specs,
        protected_attributes=protected_attributes,
        budgets=budgets,
        seeds=seeds,
        base_run_config=base_run_config,
        output_dir=output_dir,
        report_name=str(args.report_name),
        insight_thresholds=thresholds,
        save_json=bool(args.save_json),
        print_progress=True,
    )

    print("")
    print("Final aggregate ranking table")
    print(format_aggregate_table(result.ranked_aggregate_frame))
    print("")
    print("Instant insights")
    for line in result.insights:
        print(f"- {line}")
    print("")
    print("Final recommendation")
    for key, value in result.recommendations.items():
        print(f"- {key}={value or 'n/a'}")
    print("")
    print(f"Saved raw per-run CSV: {result.raw_csv_path}")
    print(f"Saved aggregate CSV: {result.aggregate_csv_path}")
    print(f"Saved ranked aggregate CSV: {result.ranked_aggregate_csv_path}")
    print(f"Saved report: {result.report_path}")
    if result.json_path is not None:
        print(f"Saved JSON summary: {result.json_path}")


if __name__ == "__main__":
    main()
