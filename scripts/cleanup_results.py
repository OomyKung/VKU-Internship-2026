"""Safely preview and delete old Fair Influence Maximization result folders.

Examples:
  Preview old experiment cleanup:
    python scripts/cleanup_results.py --results-root results --preset old-experiments

  Permanently delete old experiment folders:
    python scripts/cleanup_results.py --results-root results --preset old-experiments --apply --delete --confirm-delete

  Preview failed runs only:
    python scripts/cleanup_results.py --results-root results --mode old-failed

  Delete failed runs only:
    python scripts/cleanup_results.py --results-root results --mode old-failed --apply --delete --confirm-delete

  Keep only useful current folders:
    python scripts/cleanup_results.py --results-root results --mode keep-list --keep-pattern "^cache$" --keep-pattern "^all_results_summary$" --keep-pattern "^scalability_" --keep-pattern "^twitter_balanced_seed_" --keep-pattern "^graph_spa_region_" --keep-pattern "^synth2_mf_lift_fast" --keep-pattern "^rice_" --keep-pattern "^target_alpha_08_full$" --keep-pattern "^ablation_full_stack$" --keep-pattern "^ablation_no_mf_lift$" --keep-pattern "^ablation_no_community_diversity$" --keep-pattern "^ablation_no_fair_ris$" --keep-pattern "^ablation_ris_only$"

  Apply keep-list deletion:
    python scripts/cleanup_results.py --results-root results --mode keep-list --keep-pattern "^cache$" --keep-pattern "^all_results_summary$" --keep-pattern "^scalability_" --keep-pattern "^twitter_balanced_seed_" --keep-pattern "^graph_spa_region_" --keep-pattern "^synth2_mf_lift_fast" --keep-pattern "^rice_" --keep-pattern "^target_alpha_08_full$" --keep-pattern "^ablation_full_stack$" --keep-pattern "^ablation_no_mf_lift$" --keep-pattern "^ablation_no_community_diversity$" --keep-pattern "^ablation_no_fair_ris$" --keep-pattern "^ablation_ris_only$" --apply --delete --confirm-delete

  After cleanup, regenerate summary:
    python scripts/summarize_scalability_results.py --results-root results --output-dir results/all_results_summary --print-table --save-csv --save-json --save-markdown --table-mode compact
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


RESULT_MARKERS = (
    "result.csv",
    "summary.json",
    "fim_stack_summary.json",
    "report.txt",
    "selected_seed_set.csv",
    "candidate_score_table.csv",
)

STRUCTURED_FILES = ("result.csv", "summary.json", "fim_stack_summary.json")

PROTECTED_NAMES = {
    "cache",
    "all_results_summary",
    "scalability_summary",
    "final_scalability_summary",
    "twitter_multiseed_summary",
    "results_cleanup_report",
}

OLD_EXPERIMENT_DELETE_PATTERNS = (
    r"^community_siea",
    r"^clustering_",
    r"^dcv_targeting_",
    r"^baseline_pool_smoke$",
    r"^capability_validation$",
    r"^config_b_fixes$",
    r"^ea_memetic_ethnicity$",
    r"^ablation_tuner_ethnicity$",
    r"^ablation_no_graphsage$",
    r"^old_",
    r"^debug_",
    r"^smoke_",
    r"^test_",
)

OLD_EXPERIMENT_KEEP_PATTERNS = (
    r"^scalability_",
    r"^twitter_balanced_seed_",
    r"^graph_spa_region_",
    r"^synth2_mf_lift_fast$",
    r"^rice_",
    r"^target_alpha_08_full$",
    r"^ablation_full_stack$",
    r"^ablation_no_mf_lift$",
    r"^ablation_no_community_diversity$",
    r"^ablation_no_fair_ris$",
    r"^ablation_ris_only$",
    r"^cache$",
    r".*summary.*",
    r".*report.*",
)

FIELD_ALIASES = {
    "dataset": ("dataset", "dataset_name", "Dataset"),
    "protected_attribute": ("protected_attribute", "protected_attr", "attribute", "Attr", "Protected attribute"),
    "budget": ("budget", "Budget"),
    "target_alpha": ("target_alpha", "Target Alpha", "alpha"),
    "random_seed": ("random_seed", "seed", "Random Seed"),
    "f_score": ("f_score", "F-score", "fscore", "F score"),
    "mf": ("mf", "MF"),
    "dcv_shortfall": ("dcv_shortfall", "DCV_shortfall", "DCV Shortfall"),
    "dcv_disparity": ("dcv_disparity", "DCV_disparity", "DCV Disparity"),
    "target_coverage_ratio": ("target_coverage_ratio", "Target Coverage Ratio", "target_coverage", "coverage"),
    "runtime_seconds": ("runtime_seconds", "runtime", "Runtime", "Runtime(s)"),
    "spread": ("total_spread", "spread", "Spread"),
    "verdict": ("verdict", "Verdict", "status", "fairness_gate_status"),
}

NUMERIC_FIELDS = {
    "budget",
    "target_alpha",
    "random_seed",
    "f_score",
    "mf",
    "dcv_shortfall",
    "dcv_disparity",
    "target_coverage_ratio",
    "runtime_seconds",
    "spread",
}

FALLBACK_DATASETS = ("twitter", "rice_subset", "synth2", "synth3", "graph_spa_500_0")
FALLBACK_ATTRS = ("color", "ethnicity", "region", "gender", "age")


@dataclass
class CleanupRecord:
    folder_name: str
    folder_path: str
    dataset: object = None
    protected_attribute: object = None
    budget: object = None
    target_alpha: object = None
    random_seed: object = None
    f_score: object = None
    mf: object = None
    dcv_shortfall: object = None
    dcv_disparity: object = None
    target_coverage_ratio: object = None
    runtime_seconds: object = None
    spread: object = None
    verdict: object = None
    last_modified_time: str = ""
    folder_size_mb: float = 0.0
    is_result_folder: bool = False
    status: str = "UNKNOWN"
    action: str = "KEEP"
    reason: str = ""
    deleted: bool = False
    error: str = ""
    protected: bool = False
    matched_pattern: str = ""
    metadata_source: str = "folder_name"
    extra: dict[str, object] = field(default_factory=dict)


def _normalize_key(key: object) -> str:
    text = str(key).strip().replace("\ufeff", "")
    text = text.replace("@", "_at_")
    text = re.sub(r"[\s\-/()]+", "_", text)
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _clean_scalar(value: object) -> object:
    if _is_missing(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _parse_scalar(value: object) -> object:
    value = _clean_scalar(value)
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a", "na", "<na>"}:
        return None
    lowered = text.lower()
    if lowered in {"true", "yes", "pass", "passed", "ok"}:
        return True
    if lowered in {"false", "no", "fail", "failed"}:
        return False
    numeric_text = text[:-1] if lowered.endswith("s") and re.fullmatch(r"[-+]?\d+(\.\d+)?s", lowered) else text
    try:
        if re.fullmatch(r"[-+]?\d+", numeric_text):
            return int(numeric_text)
        if re.fullmatch(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", numeric_text):
            return float(numeric_text)
    except ValueError:
        return text
    return text


def _coerce_numeric(value: object) -> float | None:
    value = _clean_scalar(value)
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        parsed = _parse_scalar(value)
        if parsed is None or isinstance(parsed, bool):
            return None
        try:
            numeric = float(parsed)
        except (TypeError, ValueError):
            return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _safe_compile(patterns: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    compiled = []
    for pattern in patterns:
        try:
            compiled.append((pattern, re.compile(pattern, flags=re.IGNORECASE)))
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern '{pattern}': {exc}") from exc
    return compiled


def _match_any(name: str, compiled: list[tuple[str, re.Pattern[str]]]) -> str | None:
    for pattern, regex in compiled:
        if regex.search(name):
            return pattern
    return None


def _read_csv_record(path: Path) -> dict[str, object]:
    try:
        frame = pd.read_csv(path, nrows=1)
    except Exception:
        return {}
    if frame.empty:
        return {}
    row = frame.astype(object).where(pd.notna(frame), None).iloc[0].to_dict()
    return dict(row)


def _read_json_record(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                return row
        return {}
    if isinstance(payload, dict):
        for key in ("records", "runs", "individual_runs", "summary"):
            value = payload.get(key)
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        return row
        return payload
    return {}


def _parse_report_record(path: Path) -> dict[str, object]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return {}
    alias_lookup = {
        _normalize_key(alias): canonical
        for canonical, aliases in FIELD_ALIASES.items()
        for alias in aliases
    }
    parsed: dict[str, object] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            label, value = line.split(":", 1)
            canonical = alias_lookup.get(_normalize_key(label))
            if canonical:
                parsed[canonical] = _parse_scalar(value)
        for match in re.finditer(r"([A-Za-z][A-Za-z0-9_ @/\-()]+)=([^|;,]+)", line):
            canonical = alias_lookup.get(_normalize_key(match.group(1)))
            if canonical and canonical not in parsed:
                parsed[canonical] = _parse_scalar(match.group(2))
    return parsed


def _first_present(record_by_key: dict[str, object], aliases: tuple[str, ...]) -> object:
    for alias in aliases:
        key = _normalize_key(alias)
        if key in record_by_key and not _is_missing(record_by_key[key]):
            return record_by_key[key]
    return None


def _normalize_metadata(raw: dict[str, object]) -> dict[str, object]:
    by_key = {_normalize_key(key): value for key, value in raw.items()}
    normalized: dict[str, object] = {}
    for field_name, aliases in FIELD_ALIASES.items():
        normalized[field_name] = _first_present(by_key, aliases)
    for field_name in NUMERIC_FIELDS:
        normalized[field_name] = _coerce_numeric(normalized.get(field_name))
    return normalized


def _folder_name_metadata(folder: Path) -> dict[str, object]:
    name = folder.name.lower()
    metadata: dict[str, object] = {}
    for dataset in FALLBACK_DATASETS:
        if dataset.lower() in name:
            metadata["dataset"] = dataset
            break
    for attr in FALLBACK_ATTRS:
        if re.search(rf"(^|[_\-]){re.escape(attr)}($|[_\-])", name):
            metadata["protected_attribute"] = attr
            break
    seed_match = re.search(r"(?:^|[_\-])seed[_\-]?(\d+)(?:$|[_\-])", name)
    if seed_match:
        metadata["random_seed"] = float(seed_match.group(1))
    budget_match = re.search(r"(?:^|[_\-])(?:budget|b)[_\-]?(\d+)(?:$|[_\-])", name)
    if budget_match:
        metadata["budget"] = float(budget_match.group(1))
    alpha_match = re.search(r"(?:target[_\-]?alpha|alpha)[_\-]?(\d+(?:\.\d+)?)", name)
    if alpha_match:
        value = alpha_match.group(1)
        metadata["target_alpha"] = float(f"0.{value}") if value.isdigit() and len(value) > 1 else float(value)
    return metadata


def _load_metadata(folder: Path) -> tuple[dict[str, object], str]:
    merged: dict[str, object] = {}
    source = "folder_name"
    for filename in STRUCTURED_FILES:
        path = folder / filename
        if not path.exists():
            continue
        raw = _read_csv_record(path) if path.suffix == ".csv" else _read_json_record(path)
        normalized = _normalize_metadata(raw)
        if any(not _is_missing(value) for value in normalized.values()):
            merged.update({key: value for key, value in normalized.items() if not _is_missing(value)})
            source = filename
            break
    report_path = folder / "report.txt"
    if report_path.exists():
        report_metadata = _normalize_metadata(_parse_report_record(report_path))
        for key, value in report_metadata.items():
            if key not in merged or _is_missing(merged.get(key)):
                merged[key] = value
        if source == "folder_name" and any(not _is_missing(value) for value in report_metadata.values()):
            source = "report.txt"
    fallback = _folder_name_metadata(folder)
    for key, value in fallback.items():
        if key not in merged or _is_missing(merged.get(key)):
            merged[key] = value
    return merged, source


def _folder_size_and_mtime(folder: Path) -> tuple[float, float]:
    total_size = 0
    latest_mtime = folder.stat().st_mtime
    for root, dirs, files in os.walk(folder):
        dirs[:] = [dirname for dirname in dirs if not (Path(root) / dirname).is_symlink()]
        for filename in files:
            path = Path(root) / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            total_size += int(stat.st_size)
            latest_mtime = max(latest_mtime, float(stat.st_mtime))
    return total_size / (1024.0 * 1024.0), latest_mtime


def _is_result_folder(folder: Path) -> bool:
    return any((folder / filename).exists() for filename in RESULT_MARKERS)


def _is_protected_folder(folder: Path, *, ignore_cache: bool, ignore_summary_folders: bool) -> bool:
    name = folder.name.lower()
    if name in PROTECTED_NAMES:
        return True
    if ignore_cache and name == "cache":
        return True
    if ignore_summary_folders and ("summary" in name or "report" in name):
        return True
    return False


def discover_folders(results_root: Path, *, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(path for path in results_root.rglob("*") if path.is_dir())
    return sorted(path for path in results_root.iterdir() if path.is_dir())


def determine_status(record: CleanupRecord) -> str:
    dcv = _coerce_numeric(record.dcv_shortfall)
    coverage = _coerce_numeric(record.target_coverage_ratio)
    f_score = _coerce_numeric(record.f_score)
    if dcv is None or coverage is None:
        return "UNKNOWN"
    if dcv <= 1e-9 and coverage >= 1.0:
        return "SUCCESS"
    if dcv > 1e-9 or coverage < 1.0 or (f_score is not None and f_score < 0.0):
        return "FAILED"
    return "UNKNOWN"


def build_records(args: argparse.Namespace) -> list[CleanupRecord]:
    results_root = Path(args.results_root).resolve()
    records: list[CleanupRecord] = []
    for folder in discover_folders(results_root, recursive=bool(args.recursive)):
        try:
            size_mb, mtime = _folder_size_and_mtime(folder)
        except OSError:
            size_mb, mtime = 0.0, folder.stat().st_mtime
        metadata, source = _load_metadata(folder)
        record = CleanupRecord(
            folder_name=folder.name,
            folder_path=str(folder),
            last_modified_time=datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
            folder_size_mb=round(float(size_mb), 3),
            is_result_folder=_is_result_folder(folder),
            protected=_is_protected_folder(
                folder,
                ignore_cache=bool(args.ignore_cache),
                ignore_summary_folders=bool(args.ignore_summary_folders),
            ),
            metadata_source=source,
        )
        for key, value in metadata.items():
            if hasattr(record, key):
                setattr(record, key, value)
        record.status = determine_status(record)
        record.verdict = record.verdict or record.status
        records.append(record)
    return records


def _normalize_filter_values(values: list[str] | None) -> set[str]:
    return {str(value).strip().lower() for value in values or [] if str(value).strip()}


def _passes_metadata_filters(record: CleanupRecord, args: argparse.Namespace) -> tuple[bool, str]:
    include_datasets = _normalize_filter_values(args.include_dataset)
    exclude_datasets = _normalize_filter_values(args.exclude_dataset)
    include_attrs = _normalize_filter_values(args.include_protected_attribute)
    dataset = "" if record.dataset is None else str(record.dataset).strip().lower()
    attr = "" if record.protected_attribute is None else str(record.protected_attribute).strip().lower()
    if include_datasets and dataset not in include_datasets:
        return False, "dataset not included"
    if exclude_datasets and dataset in exclude_datasets:
        return False, "dataset excluded"
    if include_attrs and attr not in include_attrs:
        return False, "protected attribute not included"
    return True, ""


def _status_allowed(record: CleanupRecord, args: argparse.Namespace) -> tuple[bool, str]:
    if record.status == "UNKNOWN" and not bool(args.include_unknown):
        return False, "unknown metrics require --include-unknown"
    if bool(args.only_failed) and record.status != "FAILED":
        return False, "not failed"
    if bool(args.only_successful) and record.status != "SUCCESS":
        return False, "not successful"
    return True, ""


def _set_keep(record: CleanupRecord, action: str, reason: str) -> None:
    record.action = action
    record.reason = reason


def _set_delete(record: CleanupRecord, reason: str, matched_pattern: str = "") -> None:
    record.action = "DELETE"
    record.reason = reason
    record.matched_pattern = matched_pattern


def _apply_common_guards(record: CleanupRecord, args: argparse.Namespace) -> bool:
    if record.protected and not bool(args.force_include_protected):
        _set_keep(record, "PROTECTED_KEEP", "protected folder")
        return False
    explicit_keep_match = _match_any(record.folder_name, _safe_compile(list(args.keep_pattern or [])))
    if explicit_keep_match:
        _set_keep(record, "KEEP", f"matched keep pattern {explicit_keep_match}")
        return False
    passes_filter, filter_reason = _passes_metadata_filters(record, args)
    if not passes_filter:
        _set_keep(record, "KEEP", filter_reason)
        return False
    return True


def _apply_status_guards(record: CleanupRecord, args: argparse.Namespace) -> bool:
    allowed, reason = _status_allowed(record, args)
    if not allowed:
        action = "UNKNOWN_KEEP" if record.status == "UNKNOWN" else "KEEP"
        _set_keep(record, action, reason)
        return False
    return True


def plan_pattern_mode(records: list[CleanupRecord], args: argparse.Namespace) -> None:
    delete_patterns = list(args.delete_pattern or [])
    keep_patterns = list(args.keep_pattern or [])
    if args.preset == "old-experiments":
        delete_patterns.extend(OLD_EXPERIMENT_DELETE_PATTERNS)
        keep_patterns.extend(OLD_EXPERIMENT_KEEP_PATTERNS)
    delete_regexes = _safe_compile(delete_patterns)
    keep_regexes = _safe_compile(keep_patterns)
    for record in records:
        if not _apply_common_guards(record, args):
            continue
        keep_match = _match_any(record.folder_name, keep_regexes)
        if keep_match:
            _set_keep(record, "KEEP", f"matched keep pattern {keep_match}")
            continue
        delete_match = _match_any(record.folder_name, delete_regexes)
        if not delete_match:
            _set_keep(record, "KEEP", "no delete pattern matched")
            continue
        if not _apply_status_guards(record, args):
            record.matched_pattern = delete_match
            continue
        _set_delete(record, f"matched delete pattern {delete_match}", delete_match)


def plan_keep_list_mode(records: list[CleanupRecord], args: argparse.Namespace) -> None:
    keep_regexes = _safe_compile(list(args.keep_pattern or []))
    for record in records:
        if not _apply_common_guards(record, args):
            continue
        keep_match = _match_any(record.folder_name, keep_regexes)
        if keep_match:
            _set_keep(record, "KEEP", f"matched keep pattern {keep_match}")
            continue
        if not _apply_status_guards(record, args):
            continue
        _set_delete(record, "not matched by keep-list")


def plan_old_failed_mode(records: list[CleanupRecord], args: argparse.Namespace) -> None:
    for record in records:
        if not _apply_common_guards(record, args):
            continue
        if record.status == "FAILED":
            _set_delete(record, "failed fairness quality")
        elif record.status == "UNKNOWN":
            _set_keep(record, "UNKNOWN_KEEP", "unknown metrics require --include-unknown")
        else:
            _set_keep(record, "KEEP", "not failed")


def _latest_group_key(record: CleanupRecord) -> tuple[object, object, object, object, object]:
    return (
        record.dataset,
        record.protected_attribute,
        record.budget,
        record.target_alpha,
        record.random_seed,
    )


def plan_latest_successful_mode(records: list[CleanupRecord], args: argparse.Namespace) -> None:
    eligible_records = [record for record in records if _apply_common_guards(record, args)]
    grouped: dict[tuple[object, object, object, object, object], list[CleanupRecord]] = {}
    for record in eligible_records:
        if record.status == "SUCCESS":
            grouped.setdefault(_latest_group_key(record), []).append(record)
        elif record.status == "FAILED" and bool(args.also_delete_failed):
            _set_delete(record, "also delete failed")
        elif record.status == "UNKNOWN" and bool(args.also_delete_failed) and bool(args.include_unknown):
            _set_delete(record, "also delete unknown")
        elif record.status == "UNKNOWN":
            _set_keep(record, "UNKNOWN_KEEP", "unknown metrics require --include-unknown")
        else:
            _set_keep(record, "KEEP", "kept failed run")

    keep_latest = max(0, int(args.keep_latest))
    for group_records in grouped.values():
        sorted_records = sorted(group_records, key=lambda item: item.last_modified_time, reverse=True)
        for index, record in enumerate(sorted_records):
            if index < keep_latest:
                _set_keep(record, "KEEP", f"latest successful kept ({index + 1}/{keep_latest})")
            else:
                _set_delete(record, f"older successful beyond latest {keep_latest}")


def plan_actions(records: list[CleanupRecord], args: argparse.Namespace) -> None:
    for record in records:
        _set_keep(record, "KEEP", "default keep")
    if args.mode in {"pattern", "interactive-preview"}:
        plan_pattern_mode(records, args)
    elif args.mode == "keep-list":
        plan_keep_list_mode(records, args)
    elif args.mode == "old-failed":
        plan_old_failed_mode(records, args)
    elif args.mode == "latest-successful":
        plan_latest_successful_mode(records, args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


def _format_int(value: object) -> str:
    numeric = _coerce_numeric(value)
    return "" if numeric is None else str(int(round(numeric)))


def _format_float(value: object, digits: int) -> str:
    numeric = _coerce_numeric(value)
    return "" if numeric is None else f"{numeric:.{digits}f}"


def _format_rate(value: object) -> str:
    numeric = _coerce_numeric(value)
    return "" if numeric is None else f"{numeric * 100.0:.1f}%"


def _truncate(value: object, width: int) -> str:
    text = str(value or "")
    if len(text) <= width:
        return text
    return text[: max(1, width - 1)] + "~"


def _text_table(headers: list[str], rows: list[list[str]], align: list[str]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    lines = []
    lines.append("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    lines.append("-" * len(lines[-1]))
    for row in rows:
        parts = []
        for index, cell in enumerate(row):
            text = str(cell)
            parts.append(text.rjust(widths[index]) if align[index] == "right" else text.ljust(widths[index]))
        lines.append("  ".join(parts))
    return "\n".join(lines)


def print_plan_table(records: list[CleanupRecord]) -> None:
    headers = [
        "Action",
        "Status",
        "Dataset",
        "Attr",
        "Budget",
        "Seed",
        "F-score",
        "MF",
        "DCV_s",
        "Coverage",
        "Runtime",
        "SizeMB",
        "Modified",
        "Folder",
    ]
    align = ["left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left"]
    rows = []
    for record in records:
        rows.append(
            [
                record.action,
                record.status,
                _truncate(record.dataset, 12),
                _truncate(record.protected_attribute, 8),
                _format_int(record.budget),
                _format_int(record.random_seed),
                _format_float(record.f_score, 4),
                _format_float(record.mf, 4),
                _format_float(record.dcv_shortfall, 4),
                _format_rate(record.target_coverage_ratio),
                _format_float(record.runtime_seconds, 1),
                _format_float(record.folder_size_mb, 1),
                record.last_modified_time[:10],
                _truncate(record.folder_name, 32),
            ]
        )
    print(_text_table(headers, rows, align) if rows else "No folders scanned.")


def _jsonable(value: object) -> object:
    value = _clean_scalar(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def record_to_dict(record: CleanupRecord) -> dict[str, object]:
    return {
        "folder_name": record.folder_name,
        "folder_path": record.folder_path,
        "dataset": _jsonable(record.dataset),
        "protected_attribute": _jsonable(record.protected_attribute),
        "budget": _jsonable(record.budget),
        "target_alpha": _jsonable(record.target_alpha),
        "random_seed": _jsonable(record.random_seed),
        "f_score": _jsonable(record.f_score),
        "mf": _jsonable(record.mf),
        "dcv_shortfall": _jsonable(record.dcv_shortfall),
        "dcv_disparity": _jsonable(record.dcv_disparity),
        "target_coverage_ratio": _jsonable(record.target_coverage_ratio),
        "runtime_seconds": _jsonable(record.runtime_seconds),
        "spread": _jsonable(record.spread),
        "verdict": _jsonable(record.verdict),
        "last_modified_time": record.last_modified_time,
        "folder_size_mb": record.folder_size_mb,
        "is_result_folder": record.is_result_folder,
        "status": record.status,
        "action": record.action,
        "reason": record.reason,
        "protected": record.protected,
        "matched_pattern": record.matched_pattern,
        "metadata_source": record.metadata_source,
        "deleted": record.deleted,
        "error": record.error,
    }


def save_reports(records: list[CleanupRecord], output_dir: Path, summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [record_to_dict(record) for record in records]
    csv_path = output_dir / "cleanup_plan.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["folder_name"])
        writer.writeheader()
        writer.writerows(rows)
    payload = {"summary": summary, "folders": rows}
    (output_dir / "cleanup_plan.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    lines = ["Cleanup Plan", "-" * 60]
    for key, value in summary.items():
        lines.append(f"{key}: {value}")
    lines.extend(["", "Folders"])
    for row in rows:
        lines.append(
            f"{row['action']:<14} {row['status']:<8} {row['folder_size_mb']:>8.3f} MB  "
            f"{row['folder_name']}  [{row['reason']}]"
        )
    (output_dir / "cleanup_plan.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _path_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _has_selected_ancestor(record: CleanupRecord, delete_paths: set[Path]) -> bool:
    path = Path(record.folder_path).resolve()
    for parent in path.parents:
        if parent in delete_paths:
            return True
    return False


def delete_selected(records: list[CleanupRecord], args: argparse.Namespace) -> None:
    results_root = Path(args.results_root).resolve()
    selected_paths = {Path(record.folder_path).resolve() for record in records if record.action == "DELETE"}
    for record in records:
        if record.action != "DELETE":
            continue
        folder = Path(record.folder_path).resolve()
        if _has_selected_ancestor(record, selected_paths):
            record.action = "KEEP"
            record.reason = "kept because parent folder is selected for deletion"
            continue
        if not folder.exists():
            record.error = "path does not exist"
            continue
        if folder == results_root:
            record.error = "refusing to delete results root"
            continue
        if not _path_inside(folder, results_root):
            record.error = "path is outside results root"
            continue
        if folder.parent == results_root.parent:
            record.error = "refusing to delete parent directory"
            continue
        if record.protected and not bool(args.force_include_protected):
            record.error = "refusing to delete protected folder"
            continue
        try:
            shutil.rmtree(folder)
            record.deleted = True
        except Exception as exc:
            record.error = str(exc)


def build_summary(records: list[CleanupRecord], args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    selected = [record for record in records if record.action == "DELETE"]
    protected = [record for record in records if record.action == "PROTECTED_KEEP"]
    kept = [record for record in records if record.action != "DELETE"]
    return {
        "Results root": str(Path(args.results_root).resolve()),
        "Folders scanned": len(records),
        "Protected kept": len(protected),
        "Folders kept": len(kept),
        "Folders selected for delete": len(selected),
        "Total size selected": f"{sum(float(record.folder_size_mb or 0.0) for record in selected):.3f} MB",
        "Dry run": not bool(args.apply),
        "Apply": bool(args.apply),
        "Delete": bool(args.delete),
        "Output report": str(output_dir),
    }


def print_summary(summary: dict[str, object]) -> None:
    print("\nCleanup Summary")
    print("-" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")


def deletion_allowed(args: argparse.Namespace) -> tuple[bool, str]:
    if not bool(args.apply):
        return False, "Dry run only. Nothing was deleted."
    if not (bool(args.delete) and bool(args.confirm_delete)):
        return False, "Refusing to delete. Permanent deletion requires --apply --delete --confirm-delete."
    return True, ""


def maybe_confirm_interactive(args: argparse.Namespace, records: list[CleanupRecord]) -> bool:
    if args.mode != "interactive-preview":
        return True
    if not (bool(args.apply) and bool(args.delete) and bool(args.confirm_delete)):
        return False
    if bool(args.yes):
        return True
    selected = [record for record in records if record.action == "DELETE"]
    print(f"\nInteractive delete preview: {len(selected)} folder(s) selected.")
    response = input("Type DELETE to permanently delete selected folders: ").strip()
    return response == "DELETE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--apply", action="store_true", default=False)
    parser.add_argument("--delete", action="store_true", default=False)
    parser.add_argument("--confirm-delete", action="store_true", default=False)
    parser.add_argument(
        "--mode",
        choices=["pattern", "keep-list", "old-failed", "latest-successful", "interactive-preview"],
        default="pattern",
    )
    parser.add_argument("--delete-pattern", action="append", default=[])
    parser.add_argument("--keep-pattern", action="append", default=[])
    parser.add_argument("--include-dataset", action="append", default=[])
    parser.add_argument("--exclude-dataset", action="append", default=[])
    parser.add_argument("--include-protected-attribute", action="append", default=[])
    parser.add_argument("--only-failed", action="store_true", default=False)
    parser.add_argument("--only-successful", action="store_true", default=False)
    parser.add_argument("--keep-latest", type=int, default=1)
    parser.add_argument("--ignore-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-summary-folders", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="results_cleanup_report")
    parser.add_argument("--yes", action="store_true", default=False)
    parser.add_argument("--force-include-protected", action="store_true", default=False)
    parser.add_argument("--recursive", action="store_true", default=False)
    parser.add_argument("--include-unknown", action="store_true", default=False)
    parser.add_argument("--also-delete-failed", action="store_true", default=False)
    parser.add_argument("--preset", choices=["old-experiments"], default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")
    if not results_root.is_dir():
        raise NotADirectoryError(f"Results root is not a directory: {results_root}")

    records = build_records(args)
    plan_actions(records, args)
    print_plan_table(records)

    allowed, refusal_message = deletion_allowed(args)
    if allowed and not maybe_confirm_interactive(args, records):
        allowed = False
        refusal_message = "Interactive confirmation failed. Nothing was deleted."
    if allowed:
        delete_selected(records, args)
    elif refusal_message:
        print(f"\n{refusal_message}")

    summary = build_summary(records, args, output_dir)
    print_summary(summary)
    save_reports(records, output_dir, summary)
    print(f"\nSaved cleanup report to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
