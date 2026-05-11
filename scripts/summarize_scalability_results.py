"""Aggregate saved FIM scalability runs into terminal, CSV, JSON, and Markdown reports."""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:  # Best-effort graph-size enrichment; result metrics remain loaded from saved files.
    from fim_hybrid.data_loader import load_dataset, resolve_dataset_config
except Exception:  # pragma: no cover - keeps report fallback usable in damaged environments.
    load_dataset = None
    resolve_dataset_config = None


RESULT_FILE_ORDER = ("result.csv", "summary.json", "fim_stack_summary.json", "report.txt")
STRUCTURED_FILE_ORDER = ("result.csv", "summary.json", "fim_stack_summary.json")
PM = "\u00b1"

SCHEMA_FIELDS = [
    "dataset",
    "protected_attribute",
    "nodes",
    "edges",
    "budget",
    "budget_ratio",
    "target_alpha",
    "rr_sets",
    "mc_runs_eval",
    "population_size",
    "generations",
    "random_seed",
    "spread",
    "mf",
    "dcv_shortfall",
    "dcv_disparity",
    "primary_dcv",
    "target_coverage_ratio",
    "f_score",
    "runtime_seconds",
    "graphsage_spearman",
    "graphsage_precision_at_budget",
    "fairness_gate_status",
    "scalability_pass",
    "output_dir",
]

SETTING_FIELDS = [
    "stack_name",
    "fim_stack",
    "pipeline_mode",
    "permutation_name",
    "embedding_method",
    "community_method",
    "optimizer_mode",
    "diffusion_model",
    "ris_mode",
    "candidate_pool_size",
    "graphsage_training_target",
    "graphsage_hidden_dim",
    "graphsage_embedding_dim",
    "graphsage_epochs_run",
    "graphsage_epochs",
    "use_mf_lift",
    "mf_lift_rounds",
    "mf_lift_candidate_limit",
    "shortfall_repair_rounds",
    "shortfall_repair_candidate_limit",
    "primary_dcv_mode",
    "fscore_mode",
]

GROUP_SETTING_FIELDS = [
    "embedding_method",
    "community_method",
    "optimizer_mode",
    "diffusion_model",
    "ris_mode",
    "rr_sets",
    "mc_runs_eval",
    "population_size",
    "generations",
    "candidate_pool_size",
    "graphsage_training_target",
    "graphsage_hidden_dim",
    "graphsage_embedding_dim",
    "graphsage_epochs_run",
    "graphsage_epochs",
    "use_mf_lift",
    "mf_lift_rounds",
    "mf_lift_candidate_limit",
    "shortfall_repair_rounds",
    "shortfall_repair_candidate_limit",
    "primary_dcv_mode",
    "fscore_mode",
]

FIELD_ALIASES = {
    "dataset": ("dataset", "dataset_name", "Dataset"),
    "protected_attribute": (
        "protected_attribute",
        "protected_attr",
        "protected",
        "attribute",
        "Attr",
        "Protected attribute",
    ),
    "nodes": ("nodes", "node_count", "num_nodes", "n_nodes", "graph_nodes", "Nodes"),
    "edges": ("edges", "edge_count", "num_edges", "n_edges", "graph_edges", "Edges"),
    "budget": ("budget", "Budget"),
    "budget_ratio": ("budget_ratio", "budget_node_ratio", "budget_to_node_ratio"),
    "target_alpha": ("target_alpha", "Target Alpha", "alpha"),
    "rr_sets": (
        "effective_ris_num_rr_sets",
        "rr_sets_used",
        "rr_sets_generated",
        "ris_num_rr_sets",
        "rr_sets",
        "RIS RR Sets",
        "ris_rr_sets",
    ),
    "mc_runs_eval": ("mc_runs_eval", "mc_runs", "MC Runs Eval", "Final MC Runs"),
    "population_size": ("population_size", "initial_population_size", "Population Size"),
    "generations": ("generations", "Generations"),
    "random_seed": ("random_seed", "seed", "Random Seed"),
    "spread": ("total_spread", "spread", "Spread"),
    "mf": ("mf", "MF"),
    "dcv_shortfall": ("dcv_shortfall", "DCV_shortfall", "DCV Shortfall"),
    "dcv_disparity": ("dcv_disparity", "DCV_disparity", "DCV Disparity"),
    "primary_dcv": ("primary_dcv", "dcv", "DCV"),
    "target_coverage_ratio": (
        "target_coverage_ratio",
        "Target Coverage Ratio",
        "coverage",
        "target_coverage",
    ),
    "f_score": ("f_score", "F-score", "fscore", "F score"),
    "runtime_seconds": ("runtime_seconds", "runtime", "Runtime", "Runtime(s)"),
    "graphsage_spearman": ("graphsage_spearman", "Spearman"),
    "graphsage_precision_at_budget": (
        "graphsage_precision_at_budget",
        "Precision@Budget",
        "graphsage_precision_budget",
    ),
    "fairness_gate_status": ("fairness_gate_status", "fairness_gate", "Fairness Gate Status"),
    "scalability_pass": ("scalability_pass", "Scalability Pass"),
}

SETTING_ALIASES = {
    "stack_name": ("stack_name",),
    "fim_stack": ("fim_stack",),
    "pipeline_mode": ("pipeline_mode",),
    "permutation_name": ("permutation_name",),
    "embedding_method": ("embedding_method", "graph_embedding_algorithm"),
    "community_method": ("community_method", "community_detection_algorithm"),
    "optimizer_mode": ("optimizer_mode", "fair_influence_optimizer"),
    "diffusion_model": ("diffusion_model", "diffusion_model_name"),
    "ris_mode": ("ris_mode", "fair_ris_mode"),
    "candidate_pool_size": ("candidate_pool_size", "candidate_pool", "max_candidate_pool_size"),
    "graphsage_training_target": ("graphsage_training_target", "Training Target"),
    "graphsage_hidden_dim": ("graphsage_hidden_dim", "Hidden Dim", "graphsage_hidden"),
    "graphsage_embedding_dim": ("graphsage_embedding_dim", "Embedding Dim", "graphsage_embedding"),
    "graphsage_epochs_run": ("graphsage_epochs_run", "Epochs Run"),
    "graphsage_epochs": ("graphsage_epochs", "graphsage_epochs_requested"),
    "use_mf_lift": ("use_mf_lift", "mf_lift_enabled", "MF-Lift Enabled"),
    "mf_lift_rounds": ("mf_lift_rounds", "MF-Lift Rounds"),
    "mf_lift_candidate_limit": ("mf_lift_candidate_limit",),
    "shortfall_repair_rounds": ("shortfall_repair_rounds",),
    "shortfall_repair_candidate_limit": ("shortfall_repair_candidate_limit",),
    "primary_dcv_mode": ("primary_dcv_mode",),
    "fscore_mode": ("fscore_mode",),
}

NUMERIC_FIELDS = {
    "nodes",
    "edges",
    "budget",
    "budget_ratio",
    "target_alpha",
    "rr_sets",
    "mc_runs_eval",
    "population_size",
    "generations",
    "random_seed",
    "spread",
    "mf",
    "dcv_shortfall",
    "dcv_disparity",
    "primary_dcv",
    "target_coverage_ratio",
    "f_score",
    "runtime_seconds",
    "graphsage_spearman",
    "graphsage_precision_at_budget",
    "candidate_pool_size",
    "graphsage_hidden_dim",
    "graphsage_embedding_dim",
    "graphsage_epochs_run",
    "graphsage_epochs",
    "mf_lift_rounds",
    "mf_lift_candidate_limit",
    "shortfall_repair_rounds",
    "shortfall_repair_candidate_limit",
}

BOOL_FIELDS = {"scalability_pass", "use_mf_lift"}


def _warn(message: str, *, verbose: bool) -> None:
    if verbose:
        print(f"Warning: {message}", file=sys.stderr)


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
    if lowered in {"true", "yes", "pass", "passed"}:
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


def _coerce_bool(value: object) -> bool | None:
    value = _clean_scalar(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(float(value)):
            return None
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "pass", "passed", "ok"}:
        return True
    if text in {"false", "0", "no", "n", "fail", "failed"}:
        return False
    return None


def _first_present(record_by_key: dict[str, object], aliases: tuple[str, ...]) -> object:
    for alias in aliases:
        key = _normalize_key(alias)
        if key in record_by_key and not _is_missing(record_by_key[key]):
            return record_by_key[key]
    return None


def _records_from_csv(path: Path, *, verbose: bool) -> list[dict[str, object]]:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        _warn(f"Could not read CSV {path}: {exc}", verbose=verbose)
        return []
    if frame.empty:
        return []
    frame = frame.astype(object).where(pd.notna(frame), None)
    return [dict(row) for row in frame.to_dict(orient="records")]


def _records_from_json(path: Path, *, verbose: bool) -> list[dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _warn(f"Could not read JSON {path}: {exc}", verbose=verbose)
        return []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("records", "runs", "individual_runs", "summary"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [payload]
    return []


def _parse_report(path: Path, *, verbose: bool) -> dict[str, object]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        _warn(f"Could not read report {path}: {exc}", verbose=verbose)
        return {}

    parsed: dict[str, object] = {}
    alias_lookup = {
        _normalize_key(alias): canonical
        for canonical, aliases in {**FIELD_ALIASES, **SETTING_ALIASES}.items()
        for alias in aliases
    }
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


def _merge_sources(source_rows: list[tuple[str, list[dict[str, object]]]]) -> list[dict[str, object]]:
    if not source_rows:
        return []
    source_name, base_rows = source_rows[0]
    merged_rows = [dict(row) for row in base_rows]
    for row in merged_rows:
        row["_source_file"] = source_name
        row["_source_priority"] = 1

    for priority, (name, rows) in enumerate(source_rows[1:], start=2):
        for index, merged in enumerate(merged_rows):
            fallback = rows[index] if len(rows) == len(merged_rows) else (rows[0] if len(rows) == 1 else {})
            if not fallback:
                continue
            for key, value in fallback.items():
                if key not in merged or _is_missing(merged[key]):
                    merged[key] = value
            merged.setdefault("_fallback_sources", [])
            if isinstance(merged["_fallback_sources"], list):
                merged["_fallback_sources"].append(name)
    return merged_rows


def _load_run_directory(run_dir: Path, *, verbose: bool) -> list[dict[str, object]]:
    sources: list[tuple[str, list[dict[str, object]]]] = []
    for filename in STRUCTURED_FILE_ORDER:
        path = run_dir / filename
        if not path.exists():
            continue
        rows = _records_from_csv(path, verbose=verbose) if path.suffix == ".csv" else _records_from_json(path, verbose=verbose)
        if rows:
            sources.append((filename, rows))

    report_path = run_dir / "report.txt"
    report_record = _parse_report(report_path, verbose=verbose) if report_path.exists() else {}
    rows = _merge_sources(sources)
    if rows:
        if report_record:
            for row in rows:
                for key, value in report_record.items():
                    if key not in row or _is_missing(row[key]):
                        row[key] = value
        return rows
    if report_record:
        report_record["_source_file"] = "report.txt"
        report_record["_source_priority"] = len(RESULT_FILE_ORDER)
        return [report_record]
    return []


def _matches_patterns(path: Path, root: Path, patterns: list[str], *, default: bool) -> bool:
    if not patterns:
        return default
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.as_posix()
    rel_lower = rel.lower()
    name_lower = path.name.lower()
    for pattern in patterns:
        pattern_lower = pattern.lower()
        if pattern_lower in rel_lower or pattern_lower in name_lower:
            return True
        if fnmatch.fnmatch(rel_lower, pattern_lower) or fnmatch.fnmatch(name_lower, pattern_lower):
            return True
    return False


def discover_run_directories(
    results_root: Path,
    *,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> list[Path]:
    run_dirs: set[Path] = set()
    for filename in RESULT_FILE_ORDER:
        for path in results_root.rglob(filename):
            if path.is_file():
                run_dirs.add(path.parent)
    filtered = []
    for run_dir in sorted(run_dirs):
        if not _matches_patterns(run_dir, results_root, include_patterns, default=True):
            continue
        if _matches_patterns(run_dir, results_root, exclude_patterns, default=False):
            continue
        filtered.append(run_dir)
    return filtered


def _infer_seed_from_path(path: Path) -> int | None:
    text = path.as_posix()
    matches = re.findall(r"(?:^|[/_\-])seed[_\-]?(\d+)(?:$|[/_\-])", text, flags=re.IGNORECASE)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def _dataset_graph_size(dataset_name: object, cache: dict[str, tuple[int | None, int | None]]) -> tuple[int | None, int | None]:
    if _is_missing(dataset_name):
        return None, None
    name = str(dataset_name)
    if name in cache:
        return cache[name]
    if load_dataset is None or resolve_dataset_config is None:
        cache[name] = (None, None)
        return cache[name]
    try:
        config = resolve_dataset_config(name, base_dir=ROOT)
        dataset = load_dataset(config)
        cache[name] = (int(dataset.graph.number_of_nodes()), int(dataset.graph.number_of_edges()))
    except Exception:
        cache[name] = (None, None)
    return cache[name]


def normalize_record(
    raw: dict[str, object],
    run_dir: Path,
    graph_size_cache: dict[str, tuple[int | None, int | None]],
    *,
    verbose: bool,
) -> dict[str, object]:
    raw_by_key: dict[str, object] = {}
    for key, value in raw.items():
        raw_by_key[_normalize_key(key)] = value

    row: dict[str, object] = {field: None for field in SCHEMA_FIELDS}
    for field, aliases in FIELD_ALIASES.items():
        row[field] = _first_present(raw_by_key, aliases)
    for field, aliases in SETTING_ALIASES.items():
        row[field] = _first_present(raw_by_key, aliases)

    row["output_dir"] = str(run_dir)
    if row.get("random_seed") is None:
        row["random_seed"] = _infer_seed_from_path(run_dir)

    dataset_name = row.get("dataset")
    if row.get("nodes") is None or row.get("edges") is None:
        nodes, edges = _dataset_graph_size(dataset_name, graph_size_cache)
        row["nodes"] = row.get("nodes") if row.get("nodes") is not None else nodes
        row["edges"] = row.get("edges") if row.get("edges") is not None else edges

    for field in NUMERIC_FIELDS:
        if field in row:
            row[field] = _coerce_numeric(row[field])
    for field in BOOL_FIELDS:
        if field in row:
            row[field] = _coerce_bool(row[field])

    if row.get("budget_ratio") is None and row.get("budget") is not None and row.get("nodes"):
        row["budget_ratio"] = float(row["budget"]) / float(row["nodes"])

    for field in SCHEMA_FIELDS:
        if field != "output_dir" and row.get(field) is None:
            _warn(f"{run_dir}: missing field '{field}'", verbose=verbose)

    row["dcv_shortfall_zero"] = bool(row.get("dcv_shortfall") is not None and float(row["dcv_shortfall"]) <= 1e-9)
    row["target_coverage_pass"] = bool(
        row.get("target_coverage_ratio") is not None and float(row["target_coverage_ratio"]) >= 1.0
    )
    row["fairness_quality_pass"] = bool(row["dcv_shortfall_zero"] and row["target_coverage_pass"])
    row["quality"] = "PASS" if row["fairness_quality_pass"] else "FAIL"
    row["source_file"] = raw.get("_source_file")
    return row


def load_individual_runs(run_dirs: list[Path], *, verbose: bool) -> pd.DataFrame:
    graph_size_cache: dict[str, tuple[int | None, int | None]] = {}
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        raw_rows = _load_run_directory(run_dir, verbose=verbose)
        if not raw_rows:
            _warn(f"No loadable result rows found in {run_dir}", verbose=verbose)
            continue
        rows.extend(normalize_record(raw, run_dir, graph_size_cache, verbose=verbose) for raw in raw_rows)
    columns = SCHEMA_FIELDS + SETTING_FIELDS + [
        "dcv_shortfall_zero",
        "target_coverage_pass",
        "fairness_quality_pass",
        "quality",
        "source_file",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[columns]


def _parse_group_by(value: str) -> list[str]:
    fields = [_normalize_key(part) for part in re.split(r"[, ]+", value) if part.strip()]
    return fields or ["dataset"]


def _first_non_null(series: pd.Series) -> object:
    for value in series:
        if not _is_missing(value):
            return value
    return None


def _mean_std(series: pd.Series) -> tuple[float | None, float | None]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None, None
    mean_value = float(numeric.mean())
    std_value = float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
    if math.isnan(std_value):
        std_value = 0.0
    return mean_value, std_value


def _rate_from_bool(series: pd.Series) -> float | None:
    if series.empty:
        return None
    values = series.map(lambda value: bool(value) if value is not None else False)
    return float(values.mean())


def _status_pass_rate(series: pd.Series) -> float | None:
    non_null = [value for value in series.tolist() if not _is_missing(value)]
    if not non_null:
        return None
    passed = 0
    for value in non_null:
        text = str(value).strip().lower()
        if text in {"pass", "passed", "ok", "true", "1"}:
            passed += 1
    return float(passed) / float(len(non_null))


def _bool_pass_rate(series: pd.Series) -> float | None:
    non_null = [value for value in series.tolist() if not _is_missing(value)]
    if not non_null:
        return None
    parsed = [_coerce_bool(value) for value in non_null]
    parsed = [value for value in parsed if value is not None]
    if not parsed:
        return None
    return float(sum(1 for value in parsed if value)) / float(len(parsed))


def _size_class(nodes: object) -> str:
    numeric = _coerce_numeric(nodes)
    if numeric is None:
        return "unknown"
    if numeric <= 1000:
        return "small"
    if numeric <= 5000:
        return "medium"
    return "large"


def _verdict(row: dict[str, object]) -> str:
    dcv_rate = row.get("dcv_shortfall_zero_rate")
    coverage_rate = row.get("target_coverage_pass_rate")
    runtime = row.get("mean_runtime_seconds")
    size_class = row.get("size_class") or "unknown"
    if dcv_rate is None or coverage_rate is None:
        return "NEEDS_WORK"
    fairness_100 = float(dcv_rate) >= 1.0 and float(coverage_rate) >= 1.0
    fairness_80 = float(dcv_rate) >= 0.8 and float(coverage_rate) >= 0.8
    if not fairness_80:
        return "NEEDS_WORK"

    runtime_value = _coerce_numeric(runtime)
    if fairness_100:
        if size_class == "small":
            return "EXCELLENT" if runtime_value is None or runtime_value <= 60.0 else "GOOD"
        if size_class == "medium":
            return "EXCELLENT" if runtime_value is None or runtime_value <= 300.0 else "GOOD"
        if size_class == "large":
            if runtime_value is None or runtime_value <= 300.0:
                return "EXCELLENT"
            return "GOOD" if runtime_value <= 600.0 else "NEEDS_WORK"
        return "EXCELLENT" if runtime_value is None or runtime_value <= 300.0 else "GOOD"
    return "GOOD"


def aggregate_runs(individual: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    if individual.empty:
        return pd.DataFrame()

    group_key = list(
        dict.fromkeys(
            ["dataset", *group_by, "protected_attribute", "budget", "target_alpha", *GROUP_SETTING_FIELDS]
        )
    )
    for field in group_key:
        if field not in individual.columns:
            individual[field] = None

    rows: list[dict[str, object]] = []
    for keys, group in individual.groupby(group_key, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = {field: (None if _is_missing(value) else value) for field, value in zip(group_key, keys, strict=True)}
        for field in ("nodes", "edges", "budget_ratio", "rr_sets", "mc_runs_eval", "population_size", "generations"):
            record[field] = _first_non_null(group[field]) if field in group.columns else None
        record["run_count"] = int(len(group))

        metric_pairs = {
            "f_score": ("mean_f_score", "std_f_score"),
            "mf": ("mean_mf", "std_mf"),
            "dcv_shortfall": ("mean_dcv_shortfall", "std_dcv_shortfall"),
            "dcv_disparity": ("mean_dcv_disparity", "std_dcv_disparity"),
            "spread": ("mean_spread", "std_spread"),
            "runtime_seconds": ("mean_runtime_seconds", "std_runtime_seconds"),
        }
        for source, (mean_name, std_name) in metric_pairs.items():
            mean_value, std_value = _mean_std(group[source]) if source in group.columns else (None, None)
            record[mean_name] = mean_value
            record[std_name] = std_value

        record["target_coverage_pass_rate"] = _rate_from_bool(group["target_coverage_pass"])
        record["dcv_shortfall_zero_rate"] = _rate_from_bool(group["dcv_shortfall_zero"])
        record["fairness_quality_pass_rate"] = _rate_from_bool(group["fairness_quality_pass"])
        record["fairness_gate_pass_rate"] = (
            _status_pass_rate(group["fairness_gate_status"]) if "fairness_gate_status" in group.columns else None
        )
        record["scalability_pass_rate"] = (
            _bool_pass_rate(group["scalability_pass"]) if "scalability_pass" in group.columns else None
        )
        record["size_class"] = _size_class(record.get("nodes"))
        record["verdict"] = _verdict(record)
        rows.append(record)

    output = pd.DataFrame(rows)
    preferred = [
        "dataset",
        "protected_attribute",
        "nodes",
        "edges",
        "budget",
        "budget_ratio",
        "target_alpha",
        "rr_sets",
        "mc_runs_eval",
        "population_size",
        "generations",
        "run_count",
        "mean_f_score",
        "std_f_score",
        "mean_mf",
        "std_mf",
        "mean_dcv_shortfall",
        "std_dcv_shortfall",
        "mean_dcv_disparity",
        "std_dcv_disparity",
        "mean_spread",
        "std_spread",
        "mean_runtime_seconds",
        "std_runtime_seconds",
        "target_coverage_pass_rate",
        "dcv_shortfall_zero_rate",
        "fairness_gate_pass_rate",
        "scalability_pass_rate",
        "fairness_quality_pass_rate",
        "size_class",
        "verdict",
    ]
    for column in preferred:
        if column not in output.columns:
            output[column] = None
    extras = [column for column in output.columns if column not in preferred]
    return output[preferred + extras]


def _sort_frame(frame: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    sort_column = _normalize_key(sort_by)
    if sort_column not in frame.columns:
        sort_column = "nodes" if "nodes" in frame.columns else frame.columns[0]
    sorted_frame = frame.copy()
    helper = f"__sort_{sort_column}"
    numeric = pd.to_numeric(sorted_frame[sort_column], errors="coerce")
    if numeric.notna().any():
        sorted_frame[helper] = numeric
        by = [helper]
        ascending = [True]
    else:
        sorted_frame[helper] = sorted_frame[sort_column].astype(str)
        by = [helper]
        ascending = [True]
    for tie in ("dataset", "protected_attribute", "budget", "random_seed"):
        if tie in sorted_frame.columns and tie not in by:
            by.append(tie)
            ascending.append(True)
    return sorted_frame.sort_values(by, ascending=ascending, na_position="last", kind="mergesort").drop(columns=[helper])


def _fmt_int(value: object) -> str:
    numeric = _coerce_numeric(value)
    return "" if numeric is None else str(int(round(numeric)))


def _fmt_float(value: object, digits: int) -> str:
    numeric = _coerce_numeric(value)
    return "" if numeric is None else f"{numeric:.{digits}f}"


def _fmt_rate(value: object) -> str:
    numeric = _coerce_numeric(value)
    return "" if numeric is None else f"{numeric * 100.0:.1f}%"


def _fmt_mean_std(mean_value: object, std_value: object, digits: int) -> str:
    mean_numeric = _coerce_numeric(mean_value)
    std_numeric = _coerce_numeric(std_value)
    if mean_numeric is None:
        return ""
    if std_numeric is None:
        std_numeric = 0.0
    return f"{mean_numeric:.{digits}f}{PM}{std_numeric:.{digits}f}"


def _text_table(headers: list[str], rows: list[list[str]], alignments: list[str] | None = None) -> str:
    if alignments is None:
        alignments = ["left"] * len(headers)
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    lines = []
    header_line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    lines.append(header_line)
    lines.append("-" * len(header_line))
    for row in rows:
        formatted_cells = []
        for index, cell in enumerate(row):
            text = str(cell)
            formatted_cells.append(text.rjust(widths[index]) if alignments[index] == "right" else text.ljust(widths[index]))
        lines.append("  ".join(formatted_cells))
    return "\n".join(lines)


def _truncate_text(value: object, max_width: int = 14) -> str:
    text = str(value or "")
    if len(text) <= max_width:
        return text
    if max_width <= 1:
        return text[:max_width]
    return text[: max_width - 1] + "~"


def _fmt_runtime(value: object, digits: int = 1) -> str:
    numeric = _coerce_numeric(value)
    return "" if numeric is None else f"{numeric:.{digits}f}s"


def individual_table_rows(individual: pd.DataFrame) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Dataset",
        "Attr",
        "Nodes",
        "Edges",
        "Budget",
        "Seed",
        "RR",
        "MC",
        "F-score",
        "MF",
        "DCV_short",
        "DCV_disp",
        "Coverage",
        "Spread",
        "Runtime(s)",
        "Quality",
    ]
    rows = []
    for _, row in individual.iterrows():
        rows.append(
            [
                str(row.get("dataset") or ""),
                str(row.get("protected_attribute") or ""),
                _fmt_int(row.get("nodes")),
                _fmt_int(row.get("edges")),
                _fmt_int(row.get("budget")),
                _fmt_int(row.get("random_seed")),
                _fmt_int(row.get("rr_sets")),
                _fmt_int(row.get("mc_runs_eval")),
                _fmt_float(row.get("f_score"), 4),
                _fmt_float(row.get("mf"), 4),
                _fmt_float(row.get("dcv_shortfall"), 4),
                _fmt_float(row.get("dcv_disparity"), 4),
                _fmt_float(row.get("target_coverage_ratio"), 4),
                _fmt_float(row.get("spread"), 3),
                _fmt_float(row.get("runtime_seconds"), 2),
                str(row.get("quality") or "FAIL"),
            ]
        )
    return headers, rows


def aggregate_table_rows(aggregate: pd.DataFrame) -> tuple[list[str], list[list[str]]]:
    headers = [
        "Dataset",
        "Attr",
        "Nodes",
        "Edges",
        "Budget",
        "Runs",
        f"F-score mean{PM}std",
        f"MF mean{PM}std",
        f"DCV_short mean{PM}std",
        f"DCV_disp mean{PM}std",
        "Coverage Pass",
        "DCV=0 Pass",
        f"Spread mean{PM}std",
        f"Runtime mean{PM}std",
        "Quality Pass Rate",
        "Verdict",
    ]
    rows = []
    for _, row in aggregate.iterrows():
        rows.append(
            [
                str(row.get("dataset") or ""),
                str(row.get("protected_attribute") or ""),
                _fmt_int(row.get("nodes")),
                _fmt_int(row.get("edges")),
                _fmt_int(row.get("budget")),
                _fmt_int(row.get("run_count")),
                _fmt_mean_std(row.get("mean_f_score"), row.get("std_f_score"), 4),
                _fmt_mean_std(row.get("mean_mf"), row.get("std_mf"), 4),
                _fmt_mean_std(row.get("mean_dcv_shortfall"), row.get("std_dcv_shortfall"), 4),
                _fmt_mean_std(row.get("mean_dcv_disparity"), row.get("std_dcv_disparity"), 4),
                _fmt_rate(row.get("target_coverage_pass_rate")),
                _fmt_rate(row.get("dcv_shortfall_zero_rate")),
                _fmt_mean_std(row.get("mean_spread"), row.get("std_spread"), 3),
                _fmt_mean_std(row.get("mean_runtime_seconds"), row.get("std_runtime_seconds"), 2),
                _fmt_rate(row.get("fairness_quality_pass_rate")),
                str(row.get("verdict") or ""),
            ]
        )
    return headers, rows


def compact_aggregate_table_rows(aggregate: pd.DataFrame) -> tuple[list[str], list[list[str]], list[str]]:
    headers = [
        "Dataset",
        "Nodes",
        "Edges",
        "Budget",
        "Runs",
        "F-score",
        "MF",
        "DCV_s",
        "Coverage",
        "Runtime",
        "Verdict",
    ]
    alignments = ["left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "left"]
    rows = []
    for _, row in aggregate.iterrows():
        rows.append(
            [
                _truncate_text(row.get("dataset"), 14),
                _fmt_int(row.get("nodes")),
                _fmt_int(row.get("edges")),
                _fmt_int(row.get("budget")),
                _fmt_int(row.get("run_count")),
                _fmt_float(row.get("mean_f_score"), 4),
                _fmt_float(row.get("mean_mf"), 4),
                _fmt_float(row.get("mean_dcv_shortfall"), 4),
                _fmt_rate(row.get("target_coverage_pass_rate")),
                _fmt_runtime(row.get("mean_runtime_seconds"), 1),
                str(row.get("verdict") or ""),
            ]
        )
    return headers, rows, alignments


def _fmt_vertical_mean_std(row: pd.Series, mean_column: str, std_column: str, digits: int, suffix: str = "") -> str:
    mean_value = _coerce_numeric(row.get(mean_column))
    std_value = _coerce_numeric(row.get(std_column))
    if mean_value is None:
        return ""
    if std_value is None:
        std_value = 0.0
    if suffix:
        return f"{mean_value:.{digits}f}{suffix} {PM} {std_value:.{digits}f}{suffix}"
    return f"{mean_value:.{digits}f} {PM} {std_value:.{digits}f}"


def _print_compact_aggregate_table(aggregate: pd.DataFrame) -> None:
    print("\nScalability Summary Compact")
    print("-" * 96)
    headers, rows, alignments = compact_aggregate_table_rows(aggregate)
    print(_text_table(headers, rows, alignments) if rows else "No aggregated runs found.")


def _print_vertical_aggregate_table(aggregate: pd.DataFrame) -> None:
    print("\nScalability Summary Vertical")
    print("-" * 80)
    if aggregate.empty:
        print("No aggregated runs found.")
        return
    for index, (_, row) in enumerate(aggregate.iterrows()):
        if index:
            print("")
        print(f"Dataset: {row.get('dataset') or ''}")
        print(f"  {'Nodes / Edges':<20}: {_fmt_int(row.get('nodes'))} / {_fmt_int(row.get('edges'))}")
        print(f"  {'Budget':<20}: {_fmt_int(row.get('budget'))}")
        print(f"  {'Runs':<20}: {_fmt_int(row.get('run_count'))}")
        print(f"  {'F-score':<20}: {_fmt_vertical_mean_std(row, 'mean_f_score', 'std_f_score', 4)}")
        print(f"  {'MF':<20}: {_fmt_vertical_mean_std(row, 'mean_mf', 'std_mf', 4)}")
        print(f"  {'DCV_shortfall':<20}: {_fmt_vertical_mean_std(row, 'mean_dcv_shortfall', 'std_dcv_shortfall', 4)}")
        print(f"  {'Coverage Pass':<20}: {_fmt_rate(row.get('target_coverage_pass_rate'))}")
        print(f"  {'Runtime':<20}: {_fmt_vertical_mean_std(row, 'mean_runtime_seconds', 'std_runtime_seconds', 1, 's')}")
        print(f"  {'Verdict':<20}: {row.get('verdict') or ''}")


def _print_wide_individual_table(individual: pd.DataFrame) -> None:
    print("\nIndividual Runs")
    print("-" * 120)
    headers, rows = individual_table_rows(individual)
    print(_text_table(headers, rows) if rows else "No individual runs found.")


def _print_std_aggregate_table(aggregate: pd.DataFrame) -> None:
    print("\nScalability Summary Detailed")
    print("-" * 120)
    headers, rows = aggregate_table_rows(aggregate)
    print(_text_table(headers, rows) if rows else "No aggregated runs found.")


def resolve_table_mode(table_mode: str, *, terminal_width: int) -> str:
    mode = str(table_mode or "auto").strip().lower()
    if mode == "auto":
        return "compact" if terminal_width < 140 else "wide"
    return mode


def print_terminal_tables(
    individual: pd.DataFrame,
    aggregate: pd.DataFrame,
    *,
    table_mode: str,
    show_std_table: bool,
) -> None:
    terminal_width = shutil.get_terminal_size((120, 20)).columns
    resolved_mode = resolve_table_mode(table_mode, terminal_width=terminal_width)

    if resolved_mode == "vertical":
        _print_vertical_aggregate_table(aggregate)
    else:
        if resolved_mode == "wide":
            _print_wide_individual_table(individual)
        _print_compact_aggregate_table(aggregate)

    if show_std_table:
        _print_std_aggregate_table(aggregate)


def _jsonable(value: object) -> object:
    value = _clean_scalar(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _records_for_json(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    clean = frame.astype(object).where(pd.notna(frame), None)
    return [{key: _jsonable(value) for key, value in row.items()} for row in clean.to_dict(orient="records")]


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows found._"
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def key_findings(individual: pd.DataFrame, aggregate: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    if individual.empty:
        return ["No result rows were found for the selected filters."]

    runtime_series = pd.to_numeric(individual["runtime_seconds"], errors="coerce")
    if runtime_series.notna().any():
        fastest = individual.loc[runtime_series.idxmin()]
        slowest = individual.loc[runtime_series.idxmax()]
        findings.append(
            f"Fastest run: {fastest.get('dataset')} seed {fastest.get('random_seed') or 'n/a'} "
            f"at {_fmt_float(fastest.get('runtime_seconds'), 2)} seconds."
        )
        findings.append(
            f"Slowest run: {slowest.get('dataset')} seed {slowest.get('random_seed') or 'n/a'} "
            f"at {_fmt_float(slowest.get('runtime_seconds'), 2)} seconds."
        )

    if not aggregate.empty:
        f_scores = pd.to_numeric(aggregate["mean_f_score"], errors="coerce")
        if f_scores.notna().any():
            best = aggregate.loc[f_scores.idxmax()]
            findings.append(
                f"Highest average F-score: {best.get('dataset')} "
                f"({_fmt_float(best.get('mean_f_score'), 4)})."
            )
        mf_values = pd.to_numeric(aggregate["mean_mf"], errors="coerce")
        if mf_values.notna().any():
            lowest = aggregate.loc[mf_values.idxmin()]
            findings.append(f"Lowest average MF: {lowest.get('dataset')} ({_fmt_float(lowest.get('mean_mf'), 4)}).")

    all_dcv_zero = bool(individual["dcv_shortfall_zero"].all()) if "dcv_shortfall_zero" in individual else False
    all_coverage = bool(individual["target_coverage_pass"].all()) if "target_coverage_pass" in individual else False
    if all_dcv_zero and all_coverage:
        findings.append("All evaluated runs satisfied DCV_shortfall = 0 and Target Coverage Ratio = 1.0.")
    else:
        findings.append("At least one evaluated run did not satisfy both DCV_shortfall = 0 and Target Coverage Ratio = 1.0.")

    if not aggregate.empty and "twitter" in set(aggregate["dataset"].astype(str).str.lower()):
        twitter_rows = aggregate[aggregate["dataset"].astype(str).str.lower() == "twitter"]
        best_twitter = twitter_rows.sort_values("run_count", ascending=False).iloc[0]
        findings.append(
            f"Twitter aggregated {int(best_twitter.get('run_count') or 0)} run(s) with average runtime around "
            f"{_fmt_float(best_twitter.get('mean_runtime_seconds'), 2)} seconds."
        )

    if not aggregate.empty:
        nodes = pd.to_numeric(aggregate["nodes"], errors="coerce")
        runtimes = pd.to_numeric(aggregate["mean_runtime_seconds"], errors="coerce")
        valid = aggregate[nodes.notna() & runtimes.notna()]
        if len(valid) >= 2 and valid["nodes"].nunique(dropna=True) >= 2 and valid["mean_runtime_seconds"].nunique(dropna=True) >= 2:
            corr = nodes[nodes.notna() & runtimes.notna()].corr(runtimes[nodes.notna() & runtimes.notna()])
            if pd.notna(corr) and corr >= 0.5:
                findings.append("Runtime generally increases with graph size in the selected result set.")
            else:
                findings.append("Runtime scaling is mixed; graph size is not the only driver in the selected result set.")
        else:
            findings.append("Runtime scaling trend needs at least two datasets with node counts and runtimes.")
    findings.append("Runtime also depends on candidate pool size, RR sets, optimizer settings, and final Monte Carlo runs.")
    return findings


def build_markdown_report(individual: pd.DataFrame, aggregate: pd.DataFrame, findings: list[str]) -> str:
    individual_headers, individual_rows = individual_table_rows(individual)
    aggregate_headers, aggregate_rows = aggregate_table_rows(aggregate)
    lines = [
        "# Scalability Evaluation Summary",
        "",
        "## Individual Runs",
        _markdown_table(individual_headers, individual_rows),
        "",
        "## Aggregated Results",
        _markdown_table(aggregate_headers, aggregate_rows),
        "",
        "## Key Findings",
        *[f"- {finding}" for finding in findings],
        "",
    ]
    return "\n".join(lines)


def build_text_report(individual: pd.DataFrame, aggregate: pd.DataFrame, findings: list[str]) -> str:
    individual_headers, individual_rows = individual_table_rows(individual)
    aggregate_headers, aggregate_rows = aggregate_table_rows(aggregate)
    lines = [
        "Scalability Evaluation Summary",
        "",
        "Individual Runs",
        _text_table(individual_headers, individual_rows) if individual_rows else "No individual runs found.",
        "",
        "Aggregated Results",
        _text_table(aggregate_headers, aggregate_rows) if aggregate_rows else "No aggregated runs found.",
        "",
        "Key Findings",
        *[f"- {finding}" for finding in findings],
        "",
    ]
    return "\n".join(lines)


def save_outputs(
    individual: pd.DataFrame,
    aggregate: pd.DataFrame,
    findings: list[str],
    output_dir: Path,
    *,
    save_csv: bool,
    save_json: bool,
    save_markdown: bool,
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_csv:
        individual.to_csv(output_dir / "scalability_individual_runs.csv", index=False)
        aggregate.to_csv(output_dir / "scalability_aggregated_summary.csv", index=False)
    if save_json:
        payload = {
            "metadata": {key: _jsonable(value) for key, value in metadata.items()},
            "individual_runs": _records_for_json(individual),
            "aggregated_summary": _records_for_json(aggregate),
            "key_findings": findings,
        }
        (output_dir / "scalability_summary.json").write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
    if save_markdown:
        markdown = build_markdown_report(individual, aggregate, findings)
        text = build_text_report(individual, aggregate, findings)
        (output_dir / "scalability_report.md").write_text(markdown, encoding="utf-8")
        (output_dir / "scalability_report.txt").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results", help="Root directory containing saved result folders.")
    parser.add_argument("--output-dir", default="results/scalability_summary", help="Directory for summary outputs.")
    parser.add_argument("--include-pattern", action="append", default=[], help="Substring or glob pattern to include.")
    parser.add_argument("--exclude-pattern", action="append", default=[], help="Substring or glob pattern to exclude.")
    parser.add_argument("--group-by", default="dataset", help="Comma or space separated leading group fields.")
    parser.add_argument("--print-table", action="store_true", help="Print terminal tables.")
    parser.add_argument("--save-csv", action="store_true", help="Save CSV outputs.")
    parser.add_argument("--save-json", action="store_true", help="Save JSON output.")
    parser.add_argument("--save-markdown", action="store_true", help="Save Markdown and text reports.")
    parser.add_argument("--sort-by", default="nodes", help="Column used for table sorting.")
    parser.add_argument(
        "--table-mode",
        choices=["compact", "wide", "vertical", "auto"],
        default="auto",
        help="Terminal table layout. Auto uses compact below 140 columns and wide otherwise.",
    )
    parser.add_argument(
        "--show-std-table",
        action="store_true",
        help="Also print the detailed mean±std aggregate table in the terminal.",
    )
    parser.add_argument("--compact", action="store_true", help="Legacy shortcut for --table-mode compact.")
    parser.add_argument("--verbose", action="store_true", help="Print warnings for missing or unreadable fields.")
    return parser.parse_args()


def _print_saved_outputs_message(output_dir: Path) -> None:
    print("\nFull detailed outputs saved:")
    for filename in (
        "scalability_individual_runs.csv",
        "scalability_aggregated_summary.csv",
        "scalability_report.md",
        "scalability_summary.json",
    ):
        print(f"- {filename}")
    print(f"Output directory: {output_dir}")


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    group_by = _parse_group_by(args.group_by)
    run_dirs = discover_run_directories(
        results_root,
        include_patterns=list(args.include_pattern or []),
        exclude_patterns=list(args.exclude_pattern or []),
    )
    individual = load_individual_runs(run_dirs, verbose=bool(args.verbose))
    aggregate = aggregate_runs(individual, group_by=group_by)

    individual = _sort_frame(individual, args.sort_by)
    aggregate = _sort_frame(aggregate, args.sort_by)
    findings = key_findings(individual, aggregate)

    table_mode = "compact" if bool(args.compact) and str(args.table_mode) == "auto" else str(args.table_mode)

    if args.print_table:
        print_terminal_tables(
            individual,
            aggregate,
            table_mode=table_mode,
            show_std_table=bool(args.show_std_table),
        )

    save_outputs(
        individual,
        aggregate,
        findings,
        output_dir,
        save_csv=bool(args.save_csv),
        save_json=bool(args.save_json),
        save_markdown=bool(args.save_markdown),
        metadata={
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "results_root": str(results_root),
            "output_dir": str(output_dir),
            "include_pattern": list(args.include_pattern or []),
            "exclude_pattern": list(args.exclude_pattern or []),
            "group_by": group_by,
            "sort_by": args.sort_by,
            "table_mode": table_mode,
            "terminal_width": shutil.get_terminal_size((120, 20)).columns,
            "show_std_table": bool(args.show_std_table),
            "run_directories_scanned": len(run_dirs),
            "individual_run_count": int(len(individual)),
            "aggregate_row_count": int(len(aggregate)),
        },
    )

    if any([args.save_csv, args.save_json, args.save_markdown]):
        _print_saved_outputs_message(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
