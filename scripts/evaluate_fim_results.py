"""Evaluate existing FIM result outputs and generate ranked insight reports."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Iterable, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.priority_policy import (  # noqa: E402
    PROFESSOR_PRIORITY,
    ProfessorPriorityConfig,
    fairness_gate_notes,
    fairness_valid_mask as policy_fairness_valid_mask,
    is_professor_priority,
    normalize_ranking_policy,
    professor_priority_warning,
    rank_frame_professor_priority,
)


DEFAULT_GLOB_PATTERNS = ("*_permutation_comparison.csv",)
DEFAULT_GROUP_BY = ("dataset", "protected_attribute", "budget")
DEFAULT_CLOSE_THRESHOLD = 0.003
DEFAULT_DCV_COLLAPSE_THRESHOLD = 0.25
DEFAULT_MF_COLLAPSE_THRESHOLD = 0.001
DEFAULT_MIN_F_SCORE = 0.0
DEFAULT_MIN_FRACTION_GROUPS_COVERED = 0.80
DEFAULT_REPORT_NAME = "fim_result_evaluation"
DEFAULT_REFERENCE_SELECTION = "winner"
STANDARD_COLUMNS = [
    "stack_name",
    "dataset",
    "protected_attribute",
    "budget",
    "random_seed",
    "status",
    "diffusion_model",
    "community_method",
    "embedding_method",
    "clustering_method",
    "ranking_model",
    "optimizer_mode",
    "debias_mode",
    "variant_type",
    "key_enabled_modules",
    "notes",
    "skip_reason",
    "spread_estimator_search",
    "spread_estimator_final",
    "use_ris",
    "use_fair_ris",
    "fair_ris_enabled",
    "ris_mode",
    "ris_num_rr_sets",
    "effective_ris_num_rr_sets",
    "ris_reuse_rr_sets",
    "ris_score_nonzero_count",
    "ris_score_std",
    "fair_ris_score_nonzero_count",
    "fair_ris_score_std",
    "ris_active_verified",
    "fair_ris_active_verified",
    "ris_verification_warnings",
    "total_spread",
    "extra_spread",
    "mf",
    "dcv",
    "f_score",
    "runtime_seconds",
    "zero_covered_groups_count",
    "fraction_groups_covered",
    "scalability_pass",
    "source_file",
    "source_path",
]
METRIC_COLUMNS = [
    "total_spread",
    "extra_spread",
    "mf",
    "dcv",
    "f_score",
    "runtime_seconds",
    "zero_covered_groups_count",
    "fraction_groups_covered",
    "scalability_pass",
]
LOWER_IS_BETTER_METRICS = {"dcv", "runtime_seconds", "zero_covered_groups_count"}
BASELINE_RANKING_MODELS = {"none", "greedy", "fairness_weighted_greedy", "maximin_greedy"}


@dataclass(frozen=True, slots=True)
class RankingCriterion:
    column: str
    ascending: bool


@dataclass(frozen=True, slots=True)
class RankingSpec:
    name: str
    criteria: tuple[RankingCriterion, ...]
    primary_metric: str


@dataclass(frozen=True, slots=True)
class InsightThresholds:
    close_threshold: float = DEFAULT_CLOSE_THRESHOLD
    dcv_collapse_threshold: float = DEFAULT_DCV_COLLAPSE_THRESHOLD
    mf_collapse_threshold: float = DEFAULT_MF_COLLAPSE_THRESHOLD
    min_f_score: float = DEFAULT_MIN_F_SCORE
    min_fraction_groups_covered: float = DEFAULT_MIN_FRACTION_GROUPS_COVERED
    scalability_required: bool = True
    runtime_tiebreak_only: bool = True
    warn_only_fairness_gates: bool = False


def _professor_config_from_thresholds(thresholds: InsightThresholds) -> ProfessorPriorityConfig:
    return ProfessorPriorityConfig(
        fairness_close_threshold=float(thresholds.close_threshold),
        scalability_required=bool(thresholds.scalability_required),
        runtime_tiebreak_only=bool(thresholds.runtime_tiebreak_only),
        min_f_score=float(thresholds.min_f_score),
        min_mf=float(thresholds.mf_collapse_threshold),
        max_dcv=float(thresholds.dcv_collapse_threshold),
        min_fraction_groups_covered=float(thresholds.min_fraction_groups_covered),
        warn_only_fairness_gates=bool(thresholds.warn_only_fairness_gates),
    )


@dataclass(slots=True)
class GroupEvaluation:
    context: dict[str, object]
    ranked_frame: pd.DataFrame
    skipped_frame: pd.DataFrame
    warnings: list[str]
    insight_lines: list[str]
    method_notes: dict[str, str]
    recommendations: dict[str, str | None]
    primary_metric: str
    primary_metric_direction: str
    closeness_label: str
    pairwise_delta_frame: pd.DataFrame


@dataclass(slots=True)
class EvaluationResult:
    normalized_frame: pd.DataFrame
    ranked_frame: pd.DataFrame
    pairwise_delta_frame: pd.DataFrame
    group_results: list[GroupEvaluation]
    warnings: list[str]
    ranking_spec: RankingSpec


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


def _canonical_token(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _format_float(value: object, digits: int = 4) -> str:
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


def _trim_text(value: object, width: int) -> str:
    text = str(value) if not pd.isna(value) else "-"
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."


def _first_non_null(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def _non_empty_strings(values: Iterable[object]) -> list[str]:
    items: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "<na>"}:
            continue
        items.append(text)
    return items


def _coalesce_duplicate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if not frame.columns.duplicated().any():
        return frame
    merged_columns: dict[str, pd.Series] = {}
    for column_name in dict.fromkeys(frame.columns):
        duplicate_frame = frame.loc[:, frame.columns == column_name]
        if duplicate_frame.shape[1] == 1:
            merged_columns[column_name] = duplicate_frame.iloc[:, 0]
            continue
        merged_columns[column_name] = duplicate_frame.bfill(axis=1).iloc[:, 0]
    return pd.DataFrame(merged_columns)


def _ordered_columns(frame: pd.DataFrame) -> list[str]:
    ordered = [column for column in STANDARD_COLUMNS if column in frame.columns]
    extras = sorted(column for column in frame.columns if column not in ordered)
    return ordered + extras


def _column_alias_map() -> dict[str, str]:
    return {
        "stack_name": "stack_name",
        "permutation_name": "stack_name",
        "permutation": "stack_name",
        "method_name": "stack_name",
        "method": "method",
        "dataset": "dataset",
        "protected_attribute": "protected_attribute",
        "protectedattribute": "protected_attribute",
        "budget": "budget",
        "seed": "random_seed",
        "random_seed": "random_seed",
        "status": "status",
        "diffusion_model": "diffusion_model",
        "community_method": "community_method",
        "embedding_method": "embedding_method",
        "clustering_method": "clustering_method",
        "ranking_model": "ranking_model",
        "ranking_mode": "ranking_model",
        "ranking": "ranking_model",
        "optimizer_mode": "optimizer_mode",
        "debias_mode": "debias_mode",
        "bias_control": "debias_mode",
        "variant_type": "variant_type",
        "key_enabled_modules": "key_enabled_modules",
        "notes": "notes",
        "note": "notes",
        "skip_reason": "skip_reason",
        "skipped_reason": "skip_reason",
        "spread_estimator_search": "spread_estimator_search",
        "search_spread_estimator": "spread_estimator_search",
        "spread_estimator_final": "spread_estimator_final",
        "final_spread_estimator": "spread_estimator_final",
        "total_spread": "total_spread",
        "spread": "total_spread",
        "extra_spread": "extra_spread",
        "extra": "extra_spread",
        "mf": "mf",
        "dcv": "dcv",
        "f_score": "f_score",
        "fscore": "f_score",
        "f_score_value": "f_score",
        "runtime_seconds": "runtime_seconds",
        "runtime": "runtime_seconds",
        "zero_covered_groups": "zero_covered_groups_count",
        "zero_covered_groups_count": "zero_covered_groups_count",
        "fraction_groups_covered": "fraction_groups_covered",
        "protected_fraction_groups_covered": "fraction_groups_covered",
        "scalability_pass": "scalability_pass",
        "source_file": "source_file",
        "source_path": "source_path",
    }


def _metric_alias_map() -> dict[str, str]:
    return {
        "spread": "total_spread",
        "total_spread": "total_spread",
        "extra": "extra_spread",
        "extra_spread": "extra_spread",
        "mf": "mf",
        "dcv": "dcv",
        "f_score": "f_score",
        "fscore": "f_score",
        "f": "f_score",
        "runtime": "runtime_seconds",
        "runtime_seconds": "runtime_seconds",
        "zero_covered_groups": "zero_covered_groups_count",
        "zero_covered_groups_count": "zero_covered_groups_count",
        "fraction_groups_covered": "fraction_groups_covered",
        "scalability_pass": "scalability_pass",
    }


def _ranking_presets() -> dict[str, RankingSpec]:
    return {
        "fim_default": RankingSpec(
            name="fim_default",
            criteria=(
                RankingCriterion("f_score", False),
                RankingCriterion("mf", False),
                RankingCriterion("dcv", True),
                RankingCriterion("total_spread", False),
                RankingCriterion("runtime_seconds", True),
            ),
            primary_metric="f_score",
        ),
        "spread_first": RankingSpec(
            name="spread_first",
            criteria=(
                RankingCriterion("total_spread", False),
                RankingCriterion("extra_spread", False),
                RankingCriterion("mf", False),
                RankingCriterion("dcv", True),
                RankingCriterion("runtime_seconds", True),
            ),
            primary_metric="total_spread",
        ),
        "fairness_first": RankingSpec(
            name="fairness_first",
            criteria=(
                RankingCriterion("mf", False),
                RankingCriterion("dcv", True),
                RankingCriterion("f_score", False),
                RankingCriterion("total_spread", False),
                RankingCriterion("runtime_seconds", True),
            ),
            primary_metric="mf",
        ),
        PROFESSOR_PRIORITY: RankingSpec(
            name=PROFESSOR_PRIORITY,
            criteria=(
                RankingCriterion("f_score", False),
                RankingCriterion("mf", False),
                RankingCriterion("dcv", True),
                RankingCriterion("scalability_pass", False),
                RankingCriterion("total_spread", False),
                RankingCriterion("extra_spread", False),
                RankingCriterion("runtime_seconds", True),
            ),
            primary_metric="f_score",
        ),
        "fairness_first_priority": RankingSpec(
            name=PROFESSOR_PRIORITY,
            criteria=(
                RankingCriterion("f_score", False),
                RankingCriterion("mf", False),
                RankingCriterion("dcv", True),
                RankingCriterion("scalability_pass", False),
                RankingCriterion("total_spread", False),
                RankingCriterion("extra_spread", False),
                RankingCriterion("runtime_seconds", True),
            ),
            primary_metric="f_score",
        ),
        "runtime_first": RankingSpec(
            name="runtime_first",
            criteria=(
                RankingCriterion("runtime_seconds", True),
                RankingCriterion("f_score", False),
                RankingCriterion("mf", False),
                RankingCriterion("dcv", True),
                RankingCriterion("total_spread", False),
            ),
            primary_metric="runtime_seconds",
        ),
    }


def _ranking_spec(rank_by: str, custom_rank: str | None = None) -> RankingSpec:
    rank_by = normalize_ranking_policy(rank_by)
    if rank_by != "custom":
        try:
            return _ranking_presets()[rank_by]
        except KeyError as exc:
            raise ValueError(f"Unsupported ranking mode '{rank_by}'.") from exc

    if not custom_rank or not custom_rank.strip():
        raise ValueError("--custom-rank is required when --rank-by custom is used.")
    criteria: list[RankingCriterion] = []
    alias_map = _metric_alias_map()
    for raw_item in custom_rank.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid custom rank token '{item}'. Expected METRIC:asc|desc.")
        metric_name, direction = item.split(":", 1)
        canonical_metric = alias_map.get(_canonical_token(metric_name))
        if canonical_metric is None:
            raise ValueError(f"Unknown custom rank metric '{metric_name}'.")
        direction_token = _canonical_token(direction)
        if direction_token not in {"asc", "desc"}:
            raise ValueError(f"Invalid custom rank direction '{direction}'.")
        criteria.append(RankingCriterion(canonical_metric, direction_token == "asc"))
    if not criteria:
        raise ValueError("Custom rank must include at least one metric.")
    return RankingSpec(name="custom", criteria=tuple(criteria), primary_metric=criteria[0].column)


def _budget_from_name(name: str) -> int | None:
    match = re.search(r"_budget(\d+)", name, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _seed_from_name(name: str) -> str | None:
    match = re.search(r"seed[-_](.+)$", name.strip(), re.IGNORECASE)
    if not match:
        return None
    return match.group(1)


def _infer_seed_from_path(path: Path) -> str | None:
    current: Path | None = path
    while current is not None:
        seed = _seed_from_name(current.name)
        if seed is not None:
            return seed
        current = current.parent if current.parent != current else None
    return None


def _infer_dataset_from_path(path: Path) -> str | None:
    match = re.match(r"(?P<dataset>.+?)_budget\d+", path.stem, re.IGNORECASE)
    if match:
        return match.group("dataset")
    return None


def _infer_protected_attribute_from_path(path: Path) -> str | None:
    for part in path.parts:
        if part.lower().startswith("attr_"):
            return part[5:]
    parent_name = path.parent.name
    if parent_name and parent_name.lower() not in {"results", "perm_eval"}:
        return parent_name
    return None


def _parse_scalar_segments(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for segment in line.split("|"):
        if "=" not in segment:
            continue
        key, value = segment.split("=", 1)
        values[_canonical_token(key)] = value.strip()
    return values


def _parse_permutation_report(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    if "FIM Algorithm-Stack Permutation Comparison" not in text:
        raise ValueError(
            f"Unsupported text report format in {path}. "
            "Use a CSV comparison file or the current permutation report layout."
        )

    rows: list[dict[str, object]] = []
    current_row: dict[str, object] | None = None
    protected_attribute: str | None = None
    budget: int | None = None
    final_estimator_fields: dict[str, str] = {}

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Protected attribute:"):
            protected_attribute = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("Budget:"):
            budget_text = stripped.split(":", 1)[1].strip()
            budget = int(budget_text)
            continue
        if stripped.startswith("Final estimator:"):
            final_estimator_fields = _parse_scalar_segments(stripped.split(":", 1)[1].strip())
            continue
        row_match = re.match(r"^\d+\.\s+(.+?)\s+\[(.+?)\]$", stripped)
        if row_match:
            if current_row is not None:
                rows.append(current_row)
            current_row = {
                "stack_name": row_match.group(1).strip(),
                "status": row_match.group(2).strip(),
            }
            continue
        if current_row is None:
            continue
        if stripped.startswith("modules:"):
            modules = _parse_scalar_segments(stripped.split(":", 1)[1].strip())
            current_row["diffusion_model"] = modules.get("diffusion", pd.NA)
            current_row["community_method"] = modules.get("community", pd.NA)
            current_row["embedding_method"] = modules.get("embedding", pd.NA)
            current_row["clustering_method"] = modules.get("clustering", pd.NA)
            current_row["ranking_model"] = modules.get("ranking", pd.NA)
            current_row["optimizer_mode"] = modules.get("optimizer", pd.NA)
            current_row["debias_mode"] = modules.get("debias", pd.NA)
            current_row["spread_estimator_search"] = modules.get("search_estimator", pd.NA)
            current_row["spread_estimator_final"] = modules.get("final_estimator", pd.NA)
            current_row["fair_ris_enabled"] = modules.get("fair_ris", pd.NA)
            current_row["effective_ris_num_rr_sets"] = modules.get("ris_rr_sets", pd.NA)
            current_row["key_enabled_modules"] = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("notes="):
            current_row["notes"] = stripped.split("=", 1)[1].strip()
            continue
        if stripped.startswith("skip_reason="):
            current_row["skip_reason"] = stripped.split("=", 1)[1].strip()
            continue

        metrics = _parse_scalar_segments(stripped)
        if "spread" in metrics:
            current_row["total_spread"] = metrics.get("spread")
            current_row["extra_spread"] = metrics.get("extra")
            current_row["mf"] = metrics.get("mf")
            current_row["dcv"] = metrics.get("dcv")
            current_row["f_score"] = metrics.get("f_score")
            runtime_value = metrics.get("runtime")
            if runtime_value is not None:
                current_row["runtime_seconds"] = runtime_value[:-1] if runtime_value.endswith("s") else runtime_value

    if current_row is not None:
        rows.append(current_row)
    if not rows:
        raise ValueError(f"No method rows were parsed from {path}.")

    frame = pd.DataFrame(rows)
    if protected_attribute is not None:
        frame["protected_attribute"] = protected_attribute
    if budget is not None:
        frame["budget"] = budget
    if final_estimator_fields:
        frame["spread_estimator_final"] = final_estimator_fields.get("final_estimator", "monte_carlo")
        mc_runs_eval = final_estimator_fields.get("mc_runs_eval")
        if mc_runs_eval is not None:
            frame["mc_runs_eval"] = mc_runs_eval
    return frame


def _collect_input_files(
    input_paths: Sequence[Path],
    *,
    glob_patterns: Sequence[str],
    recursive: bool,
) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for input_path in input_paths:
        if input_path.is_file():
            resolved = input_path.resolve()
            if resolved not in seen:
                files.append(resolved)
                seen.add(resolved)
            continue
        if not input_path.is_dir():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        for pattern in glob_patterns:
            matches = input_path.rglob(pattern) if recursive else input_path.glob(pattern)
            for match in sorted(matches):
                resolved = match.resolve()
                if resolved not in seen:
                    files.append(resolved)
                    seen.add(resolved)
    if not files:
        raise FileNotFoundError("No input result files were found for the requested paths/patterns.")
    return files


def _load_result_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".txt":
        return _parse_permutation_report(path)
    raise ValueError(f"Unsupported result file type '{path.suffix}' for {path}.")


def normalize_result_frame(frame: pd.DataFrame, *, source_path: Path) -> pd.DataFrame:
    """Normalize a result frame from any supported FIM output schema."""

    if frame.empty:
        raise ValueError(f"Input file '{source_path}' contains no rows.")

    alias_map = _column_alias_map()
    renamed = frame.copy()
    renamed.columns = [alias_map.get(_canonical_token(column_name), _canonical_token(column_name)) for column_name in renamed.columns]
    renamed = _coalesce_duplicate_columns(renamed)

    if "stack_name" not in renamed.columns and "method" in renamed.columns:
        renamed["stack_name"] = renamed["method"]
    if "status" not in renamed.columns:
        renamed["status"] = "ok"
    if "notes" not in renamed.columns:
        renamed["notes"] = pd.NA
    if "skip_reason" not in renamed.columns:
        renamed["skip_reason"] = pd.NA
    if "source_path" not in renamed.columns:
        renamed["source_path"] = str(source_path)
    else:
        renamed["source_path"] = renamed["source_path"].fillna(str(source_path))
    if "source_file" not in renamed.columns:
        renamed["source_file"] = source_path.name
    else:
        renamed["source_file"] = renamed["source_file"].fillna(source_path.name)

    inferred_dataset = _infer_dataset_from_path(source_path)
    inferred_budget = _budget_from_name(source_path.name)
    inferred_seed = _infer_seed_from_path(source_path.parent)
    inferred_protected_attribute = _infer_protected_attribute_from_path(source_path)

    if "dataset" not in renamed.columns:
        renamed["dataset"] = inferred_dataset
    else:
        renamed["dataset"] = renamed["dataset"].fillna(inferred_dataset)
    if "budget" not in renamed.columns:
        renamed["budget"] = inferred_budget
    else:
        renamed["budget"] = renamed["budget"].fillna(inferred_budget)
    if "random_seed" not in renamed.columns:
        renamed["random_seed"] = inferred_seed
    else:
        renamed["random_seed"] = renamed["random_seed"].fillna(inferred_seed)
    if "protected_attribute" not in renamed.columns:
        renamed["protected_attribute"] = inferred_protected_attribute
    else:
        renamed["protected_attribute"] = renamed["protected_attribute"].fillna(inferred_protected_attribute)

    if "stack_name" not in renamed.columns:
        raise ValueError(
            f"Input file '{source_path}' is missing a method identifier column. "
            "Expected one of: stack_name, permutation_name, method."
        )

    for metric_column in METRIC_COLUMNS:
        if metric_column in renamed.columns:
            if metric_column == "scalability_pass":
                renamed[metric_column] = renamed[metric_column].map(
                    lambda value: 0.0 if str(value).strip().lower() in {"false", "0", "no"} else 1.0
                )
            else:
                renamed[metric_column] = pd.to_numeric(renamed[metric_column], errors="coerce")
    if "budget" in renamed.columns:
        renamed["budget"] = pd.to_numeric(renamed["budget"], errors="coerce")
    if "random_seed" in renamed.columns:
        renamed["random_seed"] = pd.to_numeric(renamed["random_seed"], errors="coerce")

    renamed["status"] = renamed["status"].fillna("ok").astype(str)
    skip_reason = renamed["skip_reason"].astype("string")
    renamed.loc[
        renamed["status"].str.lower().eq("ok") & skip_reason.fillna("").str.strip().ne(""),
        "status",
    ] = "skipped"

    available_metrics = [column for column in METRIC_COLUMNS if column in renamed.columns and renamed[column].notna().any()]
    if not available_metrics:
        raise ValueError(
            f"Input file '{source_path}' does not contain any supported metrics. "
            "Expected one of: total_spread/spread, extra_spread/extra, mf, dcv, f_score, runtime_seconds/runtime."
        )

    if "key_enabled_modules" not in renamed.columns:
        module_parts: list[str] = []
        for column_name, display_name in (
            ("diffusion_model", "diffusion"),
            ("community_method", "community"),
            ("embedding_method", "embedding"),
            ("clustering_method", "clustering"),
            ("ranking_model", "ranking"),
            ("optimizer_mode", "optimizer"),
            ("debias_mode", "debias"),
        ):
            if column_name not in renamed.columns:
                continue
            values = renamed[column_name]
            if values.notna().any():
                module_parts.append(f"{display_name}=" + str(_first_non_null(values)))
        renamed["key_enabled_modules"] = "; ".join(module_parts) if module_parts else pd.NA

    for column_name in STANDARD_COLUMNS:
        if column_name not in renamed.columns:
            renamed[column_name] = pd.NA
    return renamed.loc[:, _ordered_columns(renamed)].copy()


def load_evaluation_inputs(
    input_paths: Sequence[Path | str],
    *,
    glob_patterns: Sequence[str] = DEFAULT_GLOB_PATTERNS,
    recursive: bool = True,
) -> pd.DataFrame:
    """Load and normalize one or more result files/directories."""

    resolved_paths = [Path(path) for path in input_paths]
    files = _collect_input_files(resolved_paths, glob_patterns=glob_patterns, recursive=recursive)
    frames = [normalize_result_frame(_load_result_file(path), source_path=path) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    return combined.loc[:, _ordered_columns(combined)].copy()


def _normalize_filter_values(values: Sequence[str] | None) -> set[str] | None:
    if not values:
        return None
    return {str(value).strip() for value in values if str(value).strip()}


def filter_result_frame(
    frame: pd.DataFrame,
    *,
    datasets: Sequence[str] | None = None,
    protected_attributes: Sequence[str] | None = None,
    budgets: Sequence[int] | None = None,
    permutations: Sequence[str] | None = None,
    random_seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Apply optional filters to a normalized result frame."""

    filtered = frame.copy()
    dataset_filter = _normalize_filter_values(datasets)
    if dataset_filter:
        if "dataset" not in filtered.columns or not filtered["dataset"].notna().any():
            raise ValueError("A dataset filter was requested, but no dataset column is available in the input data.")
        filtered = filtered[filtered["dataset"].astype(str).isin(dataset_filter)]
    protected_filter = _normalize_filter_values(protected_attributes)
    if protected_filter:
        if "protected_attribute" not in filtered.columns or not filtered["protected_attribute"].notna().any():
            raise ValueError(
                "A protected-attribute filter was requested, but no protected_attribute column is available in the input data."
            )
        filtered = filtered[filtered["protected_attribute"].astype(str).isin(protected_filter)]
    if budgets:
        if "budget" not in filtered.columns or not filtered["budget"].notna().any():
            raise ValueError("A budget filter was requested, but no budget column is available in the input data.")
        budget_values = {int(value) for value in budgets}
        filtered = filtered[pd.to_numeric(filtered["budget"], errors="coerce").isin(budget_values)]
    permutation_filter = _normalize_filter_values(permutations)
    if permutation_filter:
        filtered = filtered[filtered["stack_name"].astype(str).isin(permutation_filter)]
    if random_seeds:
        if "random_seed" not in filtered.columns or not filtered["random_seed"].notna().any():
            raise ValueError("A random-seed filter was requested, but no random_seed column is available in the input data.")
        seed_values = {int(value) for value in random_seeds}
        filtered = filtered[pd.to_numeric(filtered["random_seed"], errors="coerce").isin(seed_values)]
    if filtered.empty:
        raise ValueError("No result rows remain after applying the requested filters.")
    return filtered.reset_index(drop=True)


def _resolve_group_columns(frame: pd.DataFrame, requested: Sequence[str] | None) -> list[str]:
    if requested:
        columns = [column for column in requested if column in frame.columns and frame[column].notna().any()]
        if not columns:
            raise ValueError("None of the requested --group-by columns are available in the input data.")
        return columns
    return [column for column in DEFAULT_GROUP_BY if column in frame.columns and frame[column].notna().any()]


def _available_ranking_criteria(frame: pd.DataFrame, spec: RankingSpec) -> tuple[list[RankingCriterion], list[str]]:
    warnings: list[str] = []
    criteria: list[RankingCriterion] = []
    for criterion in spec.criteria:
        if criterion.column in frame.columns and frame[criterion.column].notna().any():
            criteria.append(criterion)
        else:
            warnings.append(
                f"Ranking metric '{criterion.column}' is unavailable or empty and was skipped for ranking mode '{spec.name}'."
            )
    if not criteria:
        raise ValueError(
            f"No usable ranking metrics are available for ranking mode '{spec.name}'."
        )
    return criteria, warnings


def _best_effort_ranking_criteria(frame: pd.DataFrame, spec: RankingSpec) -> tuple[list[RankingCriterion], list[str]]:
    try:
        return _available_ranking_criteria(frame, spec)
    except ValueError:
        return [], [
            f"Ranking mode '{spec.name}' could not be applied in this group because none of its metrics were available."
        ]


def _sort_frame(frame: pd.DataFrame, criteria: Sequence[RankingCriterion]) -> pd.DataFrame:
    sort_columns = [criterion.column for criterion in criteria] + ["stack_name"]
    ascending = [criterion.ascending for criterion in criteria] + [True]
    return frame.sort_values(sort_columns, ascending=ascending, na_position="last", kind="mergesort").reset_index(drop=True)


def _select_best_row(frame: pd.DataFrame, criteria: Sequence[RankingCriterion]) -> pd.Series | None:
    if frame.empty:
        return None
    ordered = _sort_frame(frame, criteria)
    if ordered.empty:
        return None
    return ordered.iloc[0]


def _row_identity(row: pd.Series | None) -> str | None:
    if row is None or row.empty:
        return None
    return str(row.get("stack_name"))


def _is_interpretable_baseline(row: pd.Series) -> bool:
    variant_token = _canonical_token(row.get("variant_type", ""))
    if variant_token in {"baseline", "interpretable_baseline"}:
        return True
    embedding_token = _canonical_token(row.get("embedding_method", "none"))
    ranking_token = _canonical_token(row.get("ranking_model", "none"))
    stack_token = _canonical_token(row.get("stack_name", ""))
    if embedding_token in {"", "na", "none"} and ranking_token in {_canonical_token(value) for value in BASELINE_RANKING_MODELS}:
        return True
    return any(
        marker in stack_token
        for marker in ("community_aware", "fair_greedy", "cea_fim", "degree", "pagerank")
    )


def _is_exploratory_comparator(row: pd.Series) -> bool:
    variant_token = _canonical_token(row.get("variant_type", ""))
    if variant_token in {"exploratory", "exploratory_fairness", "fairness", "clustering_enhanced"}:
        return True
    combined = " ".join(
        [
            str(row.get("stack_name", "")),
            str(row.get("embedding_method", "")),
            str(row.get("clustering_method", "")),
            str(row.get("community_method", "")),
            str(row.get("ranking_model", "")),
        ]
    ).lower()
    return any(token in combined for token in ("graphcl", "infomap", "maximin", "spectral", "dbscan", "hdbscan"))


def _primary_metric_gap(frame: pd.DataFrame, criterion: RankingCriterion) -> float | None:
    if len(frame) < 2 or criterion.column not in frame.columns:
        return None
    first_value = frame.iloc[0].get(criterion.column)
    second_value = frame.iloc[1].get(criterion.column)
    if pd.isna(first_value) or pd.isna(second_value):
        return None
    if criterion.ascending:
        return abs(float(second_value) - float(first_value))
    return abs(float(first_value) - float(second_value))


def _closeness_label(gap: float | None, close_threshold: float) -> str:
    if gap is None:
        return "insufficient_data"
    if gap <= close_threshold:
        return "close"
    if gap > (3.0 * close_threshold):
        return "decisive"
    return "moderate"


def _aggregated_method_frame(group_frame: pd.DataFrame, group_columns: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    success_rows = group_frame[group_frame["status"].astype(str).str.lower() == "ok"].copy()
    skipped_rows = group_frame[group_frame["status"].astype(str).str.lower() != "ok"].copy()
    if success_rows.empty:
        return pd.DataFrame(columns=["stack_name"]), skipped_rows

    metric_columns = [column for column in METRIC_COLUMNS if column in success_rows.columns and success_rows[column].notna().any()]
    if not metric_columns:
        return pd.DataFrame(columns=["stack_name"]), skipped_rows
    descriptor_columns = [
        column
        for column in success_rows.columns
        if column not in metric_columns and column not in group_columns and column != "stack_name"
    ]

    if descriptor_columns:
        descriptor_frame = (
            success_rows.groupby("stack_name", dropna=False, sort=False)[descriptor_columns]
            .agg(_first_non_null)
            .reset_index()
        )
    else:
        descriptor_frame = (
            success_rows.groupby("stack_name", dropna=False, sort=False)
            .size()
            .reset_index(name="_row_count")
            .drop(columns=["_row_count"])
        )
    metric_frame = (
        success_rows.groupby("stack_name", dropna=False, sort=False)[metric_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    metric_frame.columns = [
        "stack_name"
        if column == ("stack_name", "")
        else f"{column[0]}_{column[1]}"
        for column in metric_frame.columns
    ]
    for metric_column in metric_columns:
        metric_frame[metric_column] = metric_frame[f"{metric_column}_mean"]

    count_rows: list[dict[str, object]] = []
    for stack_name, stack_frame in group_frame.groupby("stack_name", dropna=False, sort=False):
        ok_count = int(stack_frame["status"].astype(str).str.lower().eq("ok").sum())
        skipped_count = int(len(stack_frame) - ok_count)
        skip_reasons = sorted(set(_non_empty_strings(stack_frame.get("skip_reason", pd.Series(dtype=object)))))
        count_rows.append(
            {
                "stack_name": stack_name,
                "run_count": int(len(stack_frame)),
                "ok_run_count": ok_count,
                "skipped_run_count": skipped_count,
                "status": "ok" if skipped_count == 0 else "partial",
                "skip_reason_summary": " | ".join(skip_reasons) if skip_reasons else pd.NA,
            }
        )
    count_frame = pd.DataFrame(count_rows)

    aggregated = descriptor_frame.merge(metric_frame, on="stack_name", how="left", validate="one_to_one")
    aggregated = aggregated.merge(count_frame, on="stack_name", how="left", validate="one_to_one")
    for column_name in group_columns:
        if column_name in group_frame.columns:
            aggregated[column_name] = _first_non_null(group_frame[column_name])
    return aggregated, skipped_rows


def _pairwise_delta_frame(
    ranked_frame: pd.DataFrame,
    *,
    group_context: dict[str, object],
    reference_name: str | None,
) -> pd.DataFrame:
    if ranked_frame.empty or reference_name is None:
        return pd.DataFrame()
    reference_rows = ranked_frame[ranked_frame["stack_name"].astype(str) == str(reference_name)]
    if reference_rows.empty:
        return pd.DataFrame()
    reference_row = reference_rows.iloc[0]
    rows: list[dict[str, object]] = []
    for _, row in ranked_frame.iterrows():
        delta_row = {
            **group_context,
            "reference_stack_name": reference_name,
            "stack_name": row["stack_name"],
        }
        for metric in METRIC_COLUMNS:
            if metric in ranked_frame.columns:
                delta_row[f"delta_{metric}"] = (
                    float(row[metric]) - float(reference_row[metric])
                    if pd.notna(row[metric]) and pd.notna(reference_row[metric])
                    else pd.NA
                )
        rows.append(delta_row)
    return pd.DataFrame(rows)


def _method_note(
    row: pd.Series,
    *,
    winner_name: str | None,
    spread_name: str | None,
    fairness_name: str | None,
    fastest_name: str | None,
    practical_name: str | None,
    baseline_name: str | None,
    exploratory_name: str | None,
    thresholds: InsightThresholds,
) -> str:
    notes: list[str] = []
    stack_name = str(row.get("stack_name"))
    if stack_name == winner_name:
        notes.append("Best overall fairness-adjusted result")
    if stack_name == spread_name and stack_name != winner_name:
        notes.append("Best spread-oriented method, but not best fairness-adjusted method")
    if stack_name == fairness_name and stack_name != winner_name:
        notes.append("Best fairness-first method")
    if stack_name == fastest_name:
        notes.append("Fastest evaluated method")
    if stack_name == practical_name and stack_name != winner_name:
        notes.append("Best practical high-speed option")
    if stack_name == baseline_name:
        notes.append("Best interpretable baseline")
    if stack_name == exploratory_name:
        notes.append("Exploratory comparator")
    if pd.notna(row.get("mf")) and float(row["mf"]) <= thresholds.mf_collapse_threshold:
        notes.append("Fairness collapse warning: MF near zero")
    if pd.notna(row.get("dcv")) and float(row["dcv"]) >= thresholds.dcv_collapse_threshold:
        notes.append("Fairness collapse warning: DCV extremely high")
    if pd.notna(row.get("f_score")) and float(row["f_score"]) < thresholds.min_f_score:
        notes.append("F-score below fairness threshold; not recommended as default")
    if pd.notna(row.get("fraction_groups_covered")) and float(row["fraction_groups_covered"]) < thresholds.min_fraction_groups_covered:
        notes.append("Fairness collapse warning: protected-group coverage below threshold")
    if pd.notna(row.get("spread_estimator_search")):
        ris_bits = [
            f"search_estimator={row.get('spread_estimator_search')}",
            f"final_estimator={row.get('spread_estimator_final', 'monte_carlo')}",
        ]
        if pd.notna(row.get("fair_ris_enabled")):
            ris_bits.append(f"fair_ris={row.get('fair_ris_enabled')}")
        if pd.notna(row.get("effective_ris_num_rr_sets")):
            ris_bits.append(f"ris_rr_sets={row.get('effective_ris_num_rr_sets')}")
        if pd.notna(row.get("ris_active_verified")):
            ris_bits.append(f"ris_verified={row.get('ris_active_verified')}")
        if pd.notna(row.get("fair_ris_active_verified")):
            ris_bits.append(f"fair_ris_verified={row.get('fair_ris_active_verified')}")
        notes.append("RIS config: " + " | ".join(str(part) for part in ris_bits))
    ris_warning = str(row.get("ris_verification_warnings", "") or "").strip()
    if ris_warning and ris_warning != "<NA>":
        notes.append("RIS warning: " + ris_warning)
    if not notes:
        notes.append("Competitive comparison method")
    return "; ".join(dict.fromkeys(notes))


def _group_context(group_columns: Sequence[str], group_key: object) -> dict[str, object]:
    if not group_columns:
        return {}
    if len(group_columns) == 1:
        return {group_columns[0]: group_key}
    return {column_name: group_key[index] for index, column_name in enumerate(group_columns)}


def _fairness_valid_mask(frame: pd.DataFrame, thresholds: InsightThresholds) -> pd.Series:
    return policy_fairness_valid_mask(frame, _professor_config_from_thresholds(thresholds))


def _fairness_first_ranked_frame(
    aggregated: pd.DataFrame,
    ranking_criteria: Sequence[RankingCriterion],
    thresholds: InsightThresholds,
) -> pd.DataFrame:
    policy_config = _professor_config_from_thresholds(thresholds)
    ranked = rank_frame_professor_priority(aggregated.copy(), policy_config)
    rejected_mask = ~_fairness_valid_mask(ranked, thresholds)
    ranked["fairness_rejected"] = rejected_mask
    ranked["fairness_gate_failures"] = fairness_gate_notes(ranked, policy_config)
    return ranked


def evaluate_result_frame(
    frame: pd.DataFrame,
    *,
    rank_by: str = "fim_default",
    custom_rank: str | None = None,
    group_by: Sequence[str] | None = None,
    thresholds: InsightThresholds = InsightThresholds(),
    reference_selection: str = DEFAULT_REFERENCE_SELECTION,
) -> EvaluationResult:
    """Aggregate, rank, and summarize a normalized result frame."""

    if frame.empty:
        raise ValueError("No result rows are available for evaluation.")

    ranking_spec = _ranking_spec(rank_by, custom_rank=custom_rank)
    priority_ranking = is_professor_priority(ranking_spec.name)
    group_columns = _resolve_group_columns(frame, group_by)
    overall_warnings: list[str] = []
    missing_metrics = [column for column in METRIC_COLUMNS if column not in frame.columns or not frame[column].notna().any()]
    if missing_metrics:
        overall_warnings.append(
            "Expected metrics missing or empty: " + ", ".join(missing_metrics)
        )

    if group_columns:
        grouped_iterable = frame.groupby(group_columns, dropna=False, sort=False)
    else:
        grouped_iterable = [((), frame)]

    group_results: list[GroupEvaluation] = []
    ranked_frames: list[pd.DataFrame] = []
    pairwise_frames: list[pd.DataFrame] = []

    spread_spec = _ranking_presets()["spread_first"]
    fairness_spec = _ranking_presets()[PROFESSOR_PRIORITY if priority_ranking else "fairness_first"]
    runtime_spec = _ranking_presets()["runtime_first"]

    for group_key, group_frame in grouped_iterable:
        context = _group_context(group_columns, group_key)
        aggregated, skipped_frame = _aggregated_method_frame(group_frame, group_columns)
        group_warnings: list[str] = []
        if aggregated.empty:
            group_warnings.append("No successful methods were available for ranking in this group.")
            group_results.append(
                GroupEvaluation(
                    context=context,
                    ranked_frame=pd.DataFrame(),
                    skipped_frame=skipped_frame.copy(),
                    warnings=group_warnings,
                    insight_lines=["No successful methods were available for evaluation."],
                    method_notes={},
                    recommendations={
                        "best_overall_method": None,
                        "best_fairness_first_method": None,
                        "best_spread_first_method": None,
                        "best_runtime_first_method": None,
                        "best_interpretable_baseline": None,
                        "best_exploratory_comparator": None,
                        "best_practical_choice": None,
                        "best_fairness_quality_method": None,
                        "best_scalable_method": None,
                        "best_spread_method": None,
                        "best_runtime_method": None,
                        "final_professor_priority_recommendation": None,
                    },
                    primary_metric=ranking_spec.primary_metric,
                    primary_metric_direction="desc",
                    closeness_label="insufficient_data",
                    pairwise_delta_frame=pd.DataFrame(),
                )
            )
            continue

        ranking_criteria, ranking_warnings = _available_ranking_criteria(aggregated, ranking_spec)
        group_warnings.extend(ranking_warnings)
        if priority_ranking:
            ranked = _fairness_first_ranked_frame(aggregated.copy(), ranking_criteria, thresholds)
            rejected_count = int(pd.to_numeric(ranked.get("fairness_rejected", False), errors="coerce").fillna(0).sum()) if "fairness_rejected" in ranked.columns else 0
            if rejected_count:
                group_warnings.append(f"{rejected_count} method(s) were placed after fairness-collapse filters.")
            all_failed_warning = professor_priority_warning(ranked, _professor_config_from_thresholds(thresholds))
            if all_failed_warning:
                group_warnings.append(all_failed_warning)
        else:
            ranked = _sort_frame(aggregated.copy(), ranking_criteria)
        ranked["rank"] = range(1, len(ranked) + 1)

        spread_criteria, spread_warnings = _best_effort_ranking_criteria(ranked, spread_spec)
        fairness_criteria, fairness_warnings = _best_effort_ranking_criteria(ranked, fairness_spec)
        runtime_criteria, runtime_warnings = _best_effort_ranking_criteria(ranked, runtime_spec)
        group_warnings.extend(spread_warnings)
        group_warnings.extend(fairness_warnings)
        group_warnings.extend(runtime_warnings)

        winner_row = ranked.iloc[0] if priority_ranking and not ranked.empty else _select_best_row(ranked, ranking_criteria)
        spread_row = _select_best_row(ranked, spread_criteria) if spread_criteria else None
        fairness_row = ranked.iloc[0] if priority_ranking and not ranked.empty else (_select_best_row(ranked, fairness_criteria) if fairness_criteria else None)
        fastest_row = _select_best_row(ranked, runtime_criteria) if runtime_criteria else None

        winner_name = _row_identity(winner_row)
        spread_name = _row_identity(spread_row)
        fairness_name = _row_identity(fairness_row)
        fastest_name = _row_identity(fastest_row)
        if fairness_row is not None:
            fairness_f = pd.to_numeric(pd.Series([fairness_row.get("f_score", pd.NA)]), errors="coerce").iloc[0]
            fairness_mf = pd.to_numeric(pd.Series([fairness_row.get("mf", pd.NA)]), errors="coerce").iloc[0]
            fairness_dcv = pd.to_numeric(pd.Series([fairness_row.get("dcv", pd.NA)]), errors="coerce").iloc[0]
            if pd.notna(fairness_f) and pd.notna(fairness_mf) and pd.notna(fairness_dcv):
                ranked_f = pd.to_numeric(ranked["f_score"], errors="coerce") if "f_score" in ranked.columns else pd.Series(pd.NA, index=ranked.index)
                ranked_mf = pd.to_numeric(ranked["mf"], errors="coerce") if "mf" in ranked.columns else pd.Series(pd.NA, index=ranked.index)
                ranked_dcv = pd.to_numeric(ranked["dcv"], errors="coerce") if "dcv" in ranked.columns else pd.Series(pd.NA, index=ranked.index)
                dominating_rows = ranked[
                    (ranked_f > float(fairness_f))
                    & (ranked_mf > float(fairness_mf))
                    & (ranked_dcv < float(fairness_dcv))
                ]
                if not dominating_rows.empty:
                    fairness_row = dominating_rows.iloc[0]
                    fairness_name = _row_identity(fairness_row)
        fairness_valid_ranked_mask = _fairness_valid_mask(ranked, thresholds)
        if priority_ranking and not bool(fairness_valid_ranked_mask.any()):
            group_warnings.append("No method passed professor-priority fairness gates; recommending the least-bad method with warning.")

        primary_criterion = ranking_criteria[0]
        primary_gap = _primary_metric_gap(ranked, primary_criterion)
        closeness_label = _closeness_label(primary_gap, thresholds.close_threshold)

        practical_name = winner_name
        if fastest_row is not None and winner_row is not None and _row_identity(fastest_row) != winner_name:
            faster_enough = (
                pd.notna(fastest_row.get("runtime_seconds"))
                and pd.notna(winner_row.get("runtime_seconds"))
                and float(fastest_row["runtime_seconds"]) <= (0.8 * float(winner_row["runtime_seconds"]))
            )
            close_on_f_score = (
                "f_score" in ranked.columns
                and pd.notna(fastest_row.get("f_score"))
                and pd.notna(winner_row.get("f_score"))
                and abs(float(winner_row["f_score"]) - float(fastest_row["f_score"])) <= thresholds.close_threshold
            )
            if faster_enough and close_on_f_score:
                practical_name = _row_identity(fastest_row)

        baseline_rows = ranked[ranked.apply(_is_interpretable_baseline, axis=1)]
        exploratory_rows = ranked[ranked.apply(_is_exploratory_comparator, axis=1)]
        baseline_row = baseline_rows.iloc[0] if not baseline_rows.empty else None
        exploratory_row = exploratory_rows.iloc[0] if not exploratory_rows.empty else None
        baseline_name = _row_identity(baseline_row)
        exploratory_name = _row_identity(exploratory_row)

        insight_lines: list[str] = []
        if winner_name is not None:
            insight_lines.append(f"Overall winner: {winner_name}")
        if spread_name is not None:
            insight_lines.append(f"Best spread: {spread_name}")
        if fairness_name is not None:
            insight_lines.append(f"Best fairness profile: {fairness_name}")
        if fastest_name is not None:
            insight_lines.append(f"Fastest method: {fastest_name}")

        if primary_gap is not None:
            insight_lines.append(
                f"Result closeness: {closeness_label} overall result; "
                f"top {primary_criterion.column} gap is {primary_gap:.4f}"
            )

        if spread_name is not None and winner_name is not None and spread_name != winner_name:
            winner_f = winner_row.get("f_score") if winner_row is not None else pd.NA
            spread_f = spread_row.get("f_score") if spread_row is not None else pd.NA
            if pd.notna(winner_f) and pd.notna(spread_f) and float(spread_f) < float(winner_f):
                insight_lines.append(
                    f"Trade-off: {spread_name} delivers the best spread, but {winner_name} keeps the stronger fairness-adjusted score."
                )

        if fastest_name is not None and winner_name is not None and fastest_name != winner_name and practical_name == fastest_name:
            insight_lines.append(
                f"Trade-off: {fastest_name} is the best practical efficiency-oriented alternative."
            )

        collapse_mask = pd.Series([False] * len(ranked), index=ranked.index)
        if "mf" in ranked.columns:
            collapse_mask |= pd.to_numeric(ranked["mf"], errors="coerce").fillna(float("inf")) <= thresholds.mf_collapse_threshold
        if "dcv" in ranked.columns:
            collapse_mask |= pd.to_numeric(ranked["dcv"], errors="coerce").fillna(float("-inf")) >= thresholds.dcv_collapse_threshold
        if "f_score" in ranked.columns:
            collapse_mask |= pd.to_numeric(ranked["f_score"], errors="coerce").fillna(0.0) < thresholds.min_f_score
        if "fraction_groups_covered" in ranked.columns:
            collapse_mask |= (
                pd.to_numeric(ranked["fraction_groups_covered"], errors="coerce").fillna(1.0)
                < thresholds.min_fraction_groups_covered
            )
        if "zero_covered_groups_count" in ranked.columns:
            collapse_mask |= pd.to_numeric(ranked["zero_covered_groups_count"], errors="coerce").fillna(0.0) > 0.0
        collapse_rows = ranked[collapse_mask]
        if not collapse_rows.empty:
            insight_lines.append(
                "Warning: "
                + ", ".join(collapse_rows["stack_name"].astype(str))
                + " shows a fairness collapse signal."
            )

        if practical_name is not None:
            insight_lines.append(f"Best practical choice: {practical_name}")
        if baseline_name is not None:
            insight_lines.append(f"Best interpretable baseline: {baseline_name}")
        if exploratory_name is not None:
            insight_lines.append(f"Best exploratory comparator: {exploratory_name}")
        if priority_ranking and winner_name is not None:
            insight_lines.append(f"Professor-priority recommendation: {winner_name}")

        method_notes = {
            str(row["stack_name"]): _method_note(
                row,
                winner_name=winner_name,
                spread_name=spread_name,
                fairness_name=fairness_name,
                fastest_name=fastest_name,
                practical_name=practical_name,
                baseline_name=baseline_name,
                exploratory_name=exploratory_name,
                thresholds=thresholds,
            )
            for _, row in ranked.iterrows()
        }
        ranked["evaluation_note"] = ranked["stack_name"].map(method_notes)

        reference_name = winner_name if reference_selection == "winner" else winner_name
        pairwise_frame = _pairwise_delta_frame(ranked, group_context=context, reference_name=reference_name)

        scalable_rows = ranked
        if "scalability_pass" in ranked.columns:
            scalability_mask = ranked["scalability_pass"].fillna(True).map(
                lambda value: str(value).strip().lower() not in {"false", "0", "no"}
            )
            scalable_rows = ranked[scalability_mask]
            if scalable_rows.empty:
                scalable_rows = ranked
        scalable_name = str(scalable_rows.iloc[0]["stack_name"]) if not scalable_rows.empty else None
        professor_priority_name = winner_name
        recommendations = {
            "best_overall_method": winner_name,
            "best_fairness_first_method": fairness_name,
            "best_spread_first_method": spread_name,
            "best_runtime_first_method": fastest_name,
            "best_interpretable_baseline": baseline_name,
            "best_exploratory_comparator": exploratory_name,
            "best_practical_choice": practical_name,
            "best_fairness_quality_method": professor_priority_name if priority_ranking else fairness_name,
            "best_scalable_method": scalable_name,
            "best_spread_method": spread_name,
            "best_runtime_method": fastest_name,
            "final_professor_priority_recommendation": professor_priority_name if priority_ranking else practical_name,
        }

        for key, value in context.items():
            ranked[key] = value
        pairwise_frames.append(pairwise_frame)
        ranked_frames.append(ranked)
        group_results.append(
            GroupEvaluation(
                context=context,
                ranked_frame=ranked,
                skipped_frame=skipped_frame.copy(),
                warnings=group_warnings,
                insight_lines=insight_lines,
                method_notes=method_notes,
                recommendations=recommendations,
                primary_metric=primary_criterion.column,
                primary_metric_direction="asc" if primary_criterion.ascending else "desc",
                closeness_label=closeness_label,
                pairwise_delta_frame=pairwise_frame,
            )
        )
        overall_warnings.extend(group_warnings)

    combined_ranked = pd.concat(ranked_frames, ignore_index=True) if ranked_frames else pd.DataFrame()
    combined_pairwise = pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames else pd.DataFrame()
    return EvaluationResult(
        normalized_frame=frame.loc[:, _ordered_columns(frame)].copy(),
        ranked_frame=combined_ranked,
        pairwise_delta_frame=combined_pairwise,
        group_results=group_results,
        warnings=list(dict.fromkeys(overall_warnings)),
        ranking_spec=ranking_spec,
    )


def _format_context(context: dict[str, object]) -> str:
    if not context:
        return "all loaded rows"
    parts = [f"{key}={value}" for key, value in context.items()]
    return " | ".join(parts)


def _format_ranked_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No successful methods to rank."
    include_runs = "run_count" in frame.columns and pd.to_numeric(frame["run_count"], errors="coerce").max() > 1
    header = (
        "Rank  Method                         "
        + ("n   " if include_runs else "")
        + "F-score   MF      DCV     Spread   Extra   Runtime"
    )
    lines = [header]
    for _, row in frame.iterrows():
        line = (
            f"{int(row['rank']):<5} "
            f"{_trim_text(row['stack_name'], 30):<30} "
        )
        if include_runs:
            line += f"{_format_int(row.get('run_count')):<3} "
        line += (
            f"{_format_float(row.get('f_score')):<8} "
            f"{_format_float(row.get('mf')):<7} "
            f"{_format_float(row.get('dcv')):<7} "
            f"{_format_float(row.get('total_spread'), digits=3):<8} "
            f"{_format_float(row.get('extra_spread'), digits=3):<7} "
            f"{_format_runtime_seconds(row.get('runtime_seconds'))}"
        )
        lines.append(line.rstrip())
    return "\n".join(lines)


def build_evaluation_report(
    result: EvaluationResult,
    *,
    input_paths: Sequence[str | Path],
    rank_by: str,
    report_name: str,
) -> str:
    """Render a compact human-readable evaluation report."""

    input_labels = ", ".join(str(path) for path in input_paths)
    lines = [
        "FIM Result Evaluation",
        _rule(),
        f"Report: {report_name}",
        f"Inputs: {input_labels}",
        f"Rows loaded: {len(result.normalized_frame)}",
        f"Ranking mode: {rank_by}",
        "",
    ]

    if result.warnings:
        lines.append("Global Warnings")
        lines.append("-" * 72)
        for warning in result.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    if len(result.group_results) > 1:
        win_counter = Counter(
            group.recommendations["best_overall_method"]
            for group in result.group_results
            if group.recommendations["best_overall_method"]
        )
        lines.append("Sweep Summary")
        lines.append("-" * 72)
        if win_counter:
            for stack_name, win_count in win_counter.most_common():
                lines.append(f"- {stack_name}: {win_count} overall win(s)")
        if not result.ranked_frame.empty and "f_score_std" in result.ranked_frame.columns:
            stability = (
                result.ranked_frame.dropna(subset=["f_score_std"])
                .groupby("stack_name", dropna=False)["f_score_std"]
                .mean()
                .sort_values()
            )
            if not stability.empty:
                lines.append(f"- Most stable by mean F-score std: {stability.index[0]}")
        lines.append("")

    for index, group in enumerate(result.group_results, start=1):
        lines.append(f"Group {index}: {_format_context(group.context)}")
        lines.append("-" * 72)
        aggregated_runs = (
            not group.ranked_frame.empty
            and "run_count" in group.ranked_frame.columns
            and pd.to_numeric(group.ranked_frame["run_count"], errors="coerce").max() > 1
        )
        lines.append(
            f"Rows: {len(group.ranked_frame)} methods | ranking={result.ranking_spec.name} | "
            f"aggregated_runs={'yes' if aggregated_runs else 'no'}"
        )
        lines.append("")
        lines.append(_format_ranked_table(group.ranked_frame))
        lines.append("")
        lines.append("Instant Insights")
        for line in group.insight_lines:
            lines.append(f"- {line}")
        lines.append("")
        lines.append("Per-Method Notes")
        if group.method_notes:
            for stack_name, note in group.method_notes.items():
                lines.append(f"- {stack_name}: {note}")
        else:
            lines.append("- No per-method notes were generated.")
        lines.append("")
        if group.warnings:
            lines.append("Warnings")
            for warning in group.warnings:
                lines.append(f"- {warning}")
            lines.append("")
        if not group.skipped_frame.empty:
            lines.append("Skipped / Partial Rows")
            for _, row in group.skipped_frame.iterrows():
                lines.append(
                    f"- {row.get('stack_name', row.get('method', 'unknown'))}: "
                    f"{row.get('skip_reason', 'no reason provided')}"
                )
            lines.append("")
        lines.append("Recommendation")
        for key, value in group.recommendations.items():
            lines.append(f"- {key}={value or 'n/a'}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _json_ready_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def build_json_summary(result: EvaluationResult) -> dict[str, object]:
    """Build a compact JSON-ready summary for downstream automation."""

    groups: list[dict[str, object]] = []
    for group in result.group_results:
        top_rows = group.ranked_frame.head(3) if not group.ranked_frame.empty else pd.DataFrame()
        groups.append(
            {
                "context": {key: _json_ready_value(value) for key, value in group.context.items()},
                "primary_metric": group.primary_metric,
                "primary_metric_direction": group.primary_metric_direction,
                "closeness_label": group.closeness_label,
                "insight_lines": list(group.insight_lines),
                "warnings": list(group.warnings),
                "recommendations": {key: _json_ready_value(value) for key, value in group.recommendations.items()},
                "top_rows": [
                    {key: _json_ready_value(value) for key, value in row.items()}
                    for row in top_rows.to_dict(orient="records")
                ],
            }
        )
    return {
        "warnings": list(result.warnings),
        "ranking_spec": {
            "name": result.ranking_spec.name,
            "primary_metric": result.ranking_spec.primary_metric,
            "criteria": [asdict(criterion) for criterion in result.ranking_spec.criteria],
        },
        "groups": groups,
    }


def save_evaluation_outputs(
    result: EvaluationResult,
    *,
    output_dir: Path | str,
    report_name: str,
    input_paths: Sequence[str | Path],
    rank_by: str,
    save_json: bool = False,
) -> dict[str, Path]:
    """Persist normalized/ranked/report outputs for one evaluation run."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    normalized_path = output_path / f"{report_name}_normalized.csv"
    ranked_path = output_path / f"{report_name}_ranked.csv"
    report_path = output_path / f"{report_name}_report.txt"
    paths = {
        "normalized_csv": normalized_path,
        "ranked_csv": ranked_path,
        "report_txt": report_path,
    }

    result.normalized_frame.to_csv(normalized_path, index=False)
    result.ranked_frame.to_csv(ranked_path, index=False)
    report_path.write_text(
        build_evaluation_report(result, input_paths=input_paths, rank_by=rank_by, report_name=report_name),
        encoding="utf-8",
    )

    if not result.pairwise_delta_frame.empty:
        pairwise_path = output_path / f"{report_name}_pairwise_deltas.csv"
        result.pairwise_delta_frame.to_csv(pairwise_path, index=False)
        paths["pairwise_delta_csv"] = pairwise_path

    if save_json:
        json_path = output_path / f"{report_name}_summary.json"
        json_path.write_text(json.dumps(build_json_summary(result), indent=2), encoding="utf-8")
        paths["summary_json"] = json_path

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        nargs="+",
        required=True,
        help="One or more result files or directories to evaluate.",
    )
    parser.add_argument(
        "--glob-pattern",
        nargs="+",
        default=list(DEFAULT_GLOB_PATTERNS),
        help="Filename patterns used when an input path is a directory.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively search directories for matching result files.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for normalized/ranked/report outputs.")
    parser.add_argument("--report-name", default=DEFAULT_REPORT_NAME, help="Base filename for saved outputs.")
    parser.add_argument("--dataset", nargs="+", default=None, help="Optional dataset filter.")
    parser.add_argument("--protected-attribute", nargs="+", default=None, help="Optional protected attribute filter.")
    parser.add_argument("--budget", nargs="+", type=int, default=None, help="Optional budget filter.")
    parser.add_argument("--permutation", nargs="+", default=None, help="Optional method/stack filter.")
    parser.add_argument("--random-seed", nargs="+", type=int, default=None, help="Optional random-seed filter.")
    parser.add_argument(
        "--group-by",
        nargs="+",
        choices=["dataset", "protected_attribute", "budget", "random_seed"],
        default=None,
        help="Columns used to define comparison groups before aggregation.",
    )
    parser.add_argument(
        "--rank-by",
        choices=["fim_default", "spread_first", "fairness_first", "fairness_first_priority", "professor_priority", "runtime_first", "custom"],
        default="fim_default",
        help="Ranking preset used for consistent method ordering.",
    )
    parser.add_argument(
        "--ranking-policy",
        choices=["fim_default", "spread_first", "fairness_first", "fairness_first_priority", "professor_priority", "runtime_first", "custom"],
        default=None,
        help="Alias for --rank-by used by FIM stack CLIs.",
    )
    parser.add_argument(
        "--custom-rank",
        default=None,
        help="Custom ranking spec such as 'F-score:desc,MF:desc,DCV:asc,spread:desc,runtime:asc'.",
    )
    parser.add_argument("--close-threshold", type=float, default=DEFAULT_CLOSE_THRESHOLD)
    parser.add_argument("--fairness-close-threshold", type=float, default=None)
    parser.add_argument("--dcv-collapse-threshold", type=float, default=DEFAULT_DCV_COLLAPSE_THRESHOLD)
    parser.add_argument("--mf-collapse-threshold", type=float, default=DEFAULT_MF_COLLAPSE_THRESHOLD)
    parser.add_argument("--min-f-score", type=float, default=DEFAULT_MIN_F_SCORE)
    parser.add_argument("--min-mf", type=float, default=None)
    parser.add_argument("--max-dcv", type=float, default=None)
    parser.add_argument("--min-fraction-groups-covered", type=float, default=DEFAULT_MIN_FRACTION_GROUPS_COVERED)
    parser.add_argument("--scalability-required", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runtime-tiebreak-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warn-only-fairness-gates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-json", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank_by = normalize_ranking_policy(str(args.ranking_policy or args.rank_by))
    input_paths = [path if Path(path).is_absolute() else ROOT / path for path in args.input_path]
    loaded = load_evaluation_inputs(
        input_paths,
        glob_patterns=args.glob_pattern,
        recursive=bool(args.recursive),
    )
    filtered = filter_result_frame(
        loaded,
        datasets=args.dataset,
        protected_attributes=args.protected_attribute,
        budgets=args.budget,
        permutations=args.permutation,
        random_seeds=args.random_seed,
    )
    result = evaluate_result_frame(
        filtered,
        rank_by=rank_by,
        custom_rank=args.custom_rank,
        group_by=args.group_by,
        thresholds=InsightThresholds(
            close_threshold=float(args.fairness_close_threshold if args.fairness_close_threshold is not None else args.close_threshold),
            dcv_collapse_threshold=float(args.max_dcv if args.max_dcv is not None else args.dcv_collapse_threshold),
            mf_collapse_threshold=float(args.min_mf if args.min_mf is not None else args.mf_collapse_threshold),
            min_f_score=float(args.min_f_score),
            min_fraction_groups_covered=float(args.min_fraction_groups_covered),
            scalability_required=bool(args.scalability_required),
            runtime_tiebreak_only=bool(args.runtime_tiebreak_only),
            warn_only_fairness_gates=bool(args.warn_only_fairness_gates),
        ),
    )

    output_dir = _resolve_repo_path(args.output_dir)
    if output_dir is None:
        first_input = input_paths[0]
        output_dir = (first_input.parent if first_input.is_file() else first_input) / "evaluation"

    saved_paths = save_evaluation_outputs(
        result,
        output_dir=output_dir,
        report_name=str(args.report_name),
        input_paths=input_paths,
        rank_by=rank_by,
        save_json=bool(args.save_json),
    )
    print(build_evaluation_report(result, input_paths=input_paths, rank_by=rank_by, report_name=str(args.report_name)).rstrip())
    print("")
    print(f"Saved normalized CSV: {saved_paths['normalized_csv']}")
    print(f"Saved ranked CSV: {saved_paths['ranked_csv']}")
    print(f"Saved report: {saved_paths['report_txt']}")
    if "pairwise_delta_csv" in saved_paths:
        print(f"Saved pairwise deltas: {saved_paths['pairwise_delta_csv']}")
    if "summary_json" in saved_paths:
        print(f"Saved JSON summary: {saved_paths['summary_json']}")


if __name__ == "__main__":
    main()
