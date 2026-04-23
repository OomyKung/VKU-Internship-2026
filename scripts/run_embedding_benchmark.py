"""CLI for the modular graph embedding benchmark framework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.data_loader import load_dataset, resolve_dataset_config  # noqa: E402
from fim_hybrid.embeddings import (  # noqa: E402
    available_embedding_methods,
    available_evaluation_tasks,
    evaluate_embedding_benchmark,
    evaluate_embedding_benchmark_repeated,
    resolve_method_names,
    resolve_evaluation_tasks,
    run_embedding_benchmark,
)


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def build_dataset_config(args: argparse.Namespace):
    """Resolve CLI dataset arguments into the shared dataset config."""

    dataset_format = None if getattr(args, "dataset_format", None) in {None, "auto"} else args.dataset_format
    return resolve_dataset_config(
        args.dataset,
        base_dir=ROOT,
        graph_path=args.graph_path,
        attributes_path=args.attributes_path,
        dataset_format=dataset_format,
        dataset_config=args.dataset_config,
        directed=args.directed,
        source_col=args.source_col,
        target_col=args.target_col,
        node_id_col=args.node_id_col,
    )


def build_method_configs(args: argparse.Namespace, methods: list[str]) -> dict[str, dict[str, object]]:
    """Build per-method config dictionaries from shared CLI flags."""

    configs: dict[str, dict[str, object]] = {}
    for method_name in methods:
        config: dict[str, object] = {
            "embedding_dim": args.embedding_dim,
            "random_seed": args.random_seed,
        }
        if method_name in {"deepwalk", "node2vec"}:
            config.update(
                {
                    "walk_length": args.walk_length,
                    "num_walks": args.num_walks,
                    "window_size": args.window_size,
                }
            )
        if method_name == "node2vec":
            config.update({"p": args.node2vec_p, "q": args.node2vec_q})
        if method_name == "line":
            config["order"] = args.line_order
        if method_name in {"gcn", "graphsage", "vgae", "dgi", "graphcl"}:
            config.update(
                {
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "epochs": args.epochs,
                }
            )
        if method_name == "graphcl":
            config.update(
                {
                    "projection_dim": args.projection_dim,
                    "temperature": args.temperature,
                    "edge_dropout_probability": args.edge_dropout_probability,
                    "feature_mask_probability": args.feature_mask_probability,
                }
            )
        configs[method_name] = config
    return configs


def build_evaluation_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Build shared downstream-evaluation kwargs from CLI flags."""

    return {
        "node_test_fraction": getattr(args, "node_test_fraction", 0.25),
        "link_test_fraction": getattr(args, "link_test_fraction", 0.25),
        "link_negative_ratio": getattr(args, "link_negative_ratio", 1.0),
        "link_prediction_edge_feature": getattr(args, "link_prediction_edge_feature", "hadamard"),
        "node_classification_model": getattr(args, "node_classification_model", "logistic_regression"),
        "clustering_method": getattr(args, "clustering_method", "kmeans"),
        "clustering_input_mode": getattr(args, "clustering_input_mode", "embedding"),
        "clustering_n_clusters": getattr(args, "n_clusters", None),
        "clustering_min_cluster_size": getattr(args, "min_cluster_size", None),
        "clustering_method_config": getattr(args, "clustering_method_config", None),
        "training_mode": getattr(args, "training_mode", None),
        "imbalance_mode": getattr(args, "imbalance_mode", "none"),
        "debias_mode": getattr(args, "debias_mode", "none"),
        "focal_gamma": getattr(args, "focal_gamma", 2.0),
        "use_stratified_split": getattr(args, "use_stratified_split", True),
        "early_stop_metric": getattr(args, "early_stop_metric", "accuracy"),
        "early_stop_patience": getattr(args, "early_stop_patience", 0),
        "class_weight_smoothing": getattr(args, "class_weight_smoothing", 0.0),
        "node_validation_fraction": getattr(args, "node_validation_fraction", 0.2),
        "group_robust_weight": getattr(args, "group_robust_weight", 0.0),
        "group_weight_mode": getattr(args, "group_weight_mode", "none"),
        "worst_group_boost_factor": getattr(args, "worst_group_boost_factor", 2.0),
        "min_support_boost_factor": getattr(args, "min_support_boost_factor", 2.0),
        "min_group_support_threshold": getattr(args, "min_group_support_threshold", None),
        "min_group_support_train": getattr(args, "min_group_support_train", None),
        "min_group_support_eval": getattr(args, "min_group_support_eval", None),
        "report_small_group_metrics": getattr(args, "report_small_group_metrics", False),
        "rebalance_batches_by_group": getattr(args, "rebalance_batches_by_group", False),
        "fairness_score_alpha": getattr(args, "fairness_score_alpha", 0.25),
        "fairness_score_beta": getattr(args, "fairness_score_beta", 0.25),
        "probe_model_type": getattr(args, "probe_model_type", "linear"),
        "adversary_loss_weight": getattr(args, "adversary_loss_weight", 1.0),
        "gradient_reversal_lambda": getattr(args, "gradient_reversal_lambda", 1.0),
        "adversary_warmup_epochs": getattr(args, "adversary_warmup_epochs", 0),
        "group_robust_warmup_epochs": getattr(args, "group_robust_warmup_epochs", 0),
        "adversary_hidden_dim": getattr(args, "adversary_hidden_dim", 64),
        "adversary_num_layers": getattr(args, "adversary_num_layers", 1),
        "adversary_dropout": getattr(args, "adversary_dropout", 0.2),
        "protected_attribute_column": getattr(args, "protected_attribute_column", None),
    }


def _rule(character: str = "=") -> str:
    width = max(80, min(120, shutil.get_terminal_size((100, 20)).columns))
    return character * width


def _format_runtime(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.3f}s"


def _format_dim(value: object) -> str:
    if pd.isna(value):
        return "-"
    return str(int(value))


def _display_path(value: object) -> str:
    if pd.isna(value):
        return "-"
    return str(value)


def _format_optional_text(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    text = str(value).strip()
    return text if text else "-"


def _has_text(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return bool(str(value).strip())


def benchmark_report_path(output_dir: Path | None, dataset_name: str) -> Path | None:
    """Return the text report output path when an output directory is available."""

    if output_dir is None:
        return None
    benchmark_dir = output_dir / dataset_name / "reports"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    return benchmark_dir / f"{dataset_name}_embedding_benchmark_report.txt"


def graphsage_comparison_summary_path(output_dir: Path | None, dataset_name: str) -> Path | None:
    """Return the preset GraphSAGE comparison CSV path when available."""

    if output_dir is None:
        return None
    benchmark_dir = output_dir / dataset_name / "embeddings"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    return benchmark_dir / f"{dataset_name}_graphsage_comparison_summary.csv"


def graphsage_tuning_summary_path(output_dir: Path | None, dataset_name: str) -> Path | None:
    """Return the preset GraphSAGE tuning CSV path when available."""

    if output_dir is None:
        return None
    benchmark_dir = output_dir / dataset_name / "embeddings"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    return benchmark_dir / f"{dataset_name}_graphsage_tuning_summary.csv"


def repeated_evaluation_runs_path(output_dir: Path | None, dataset_name: str) -> Path | None:
    """Return the repeated-evaluation per-run CSV path when available."""

    if output_dir is None:
        return None
    benchmark_dir = output_dir / dataset_name / "embeddings"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    return benchmark_dir / f"{dataset_name}_embedding_evaluation_repeated_runs.csv"


def repeated_evaluation_summary_path(output_dir: Path | None, dataset_name: str) -> Path | None:
    """Return the repeated-evaluation aggregate CSV path when available."""

    if output_dir is None:
        return None
    benchmark_dir = output_dir / dataset_name / "embeddings"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    return benchmark_dir / f"{dataset_name}_embedding_evaluation_repeated_summary.csv"


def graphsage_repeated_comparison_runs_path(output_dir: Path | None, dataset_name: str) -> Path | None:
    """Return the repeated GraphSAGE comparison per-run CSV path when available."""

    if output_dir is None:
        return None
    benchmark_dir = output_dir / dataset_name / "embeddings"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    return benchmark_dir / f"{dataset_name}_graphsage_comparison_repeated_runs.csv"


def graphsage_repeated_comparison_summary_path(output_dir: Path | None, dataset_name: str) -> Path | None:
    """Return the repeated GraphSAGE comparison aggregate CSV path when available."""

    if output_dir is None:
        return None
    benchmark_dir = output_dir / dataset_name / "embeddings"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    return benchmark_dir / f"{dataset_name}_graphsage_comparison_repeated_summary.csv"


def attribute_report_path(output_dir: Path | None, dataset_name: str, attribute_name: str) -> Path | None:
    """Return the attribute-specific text report output path when available."""

    if output_dir is None:
        return None
    benchmark_dir = output_dir / dataset_name / "reports"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    safe_attribute = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in attribute_name.strip()
    )
    safe_attribute = safe_attribute or "attribute"
    return benchmark_dir / f"{dataset_name}_embedding_benchmark_report_{safe_attribute}.txt"


def all_attributes_report_path(output_dir: Path | None, dataset_name: str) -> Path | None:
    """Return the aggregate all-attributes text report path when available."""

    if output_dir is None:
        return None
    benchmark_dir = output_dir / dataset_name / "reports"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    return benchmark_dir / f"{dataset_name}_all_attributes_embedding_benchmark_report.txt"


def save_benchmark_report(report_text: str, *, output_dir: Path | None, dataset_name: str) -> Path | None:
    """Persist the terminal report as a text file when possible."""

    report_path = benchmark_report_path(output_dir, dataset_name)
    if report_path is None:
        return None
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def save_attribute_benchmark_report(
    report_text: str,
    *,
    output_dir: Path | None,
    dataset_name: str,
    attribute_name: str,
) -> Path | None:
    """Persist an attribute-specific report as a text file when possible."""

    report_path = attribute_report_path(output_dir, dataset_name, attribute_name)
    if report_path is None:
        return None
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def save_all_attributes_benchmark_report(
    report_text: str,
    *,
    output_dir: Path | None,
    dataset_name: str,
) -> Path | None:
    """Persist the aggregate all-attributes report when possible."""

    report_path = all_attributes_report_path(output_dir, dataset_name)
    if report_path is None:
        return None
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def _format_metric(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def _load_json_payload(value: object) -> Any | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _format_group_support_counts_brief(value: object) -> str:
    payload = _load_json_payload(value)
    if not isinstance(payload, list) or not payload:
        return "-"
    return ",".join(
        f"{item.get('group')}:{int(item.get('count', 0) or 0)}"
        for item in payload
        if isinstance(item, dict) and item.get("group") is not None
    ) or "-"


def _format_small_groups_brief(value: object) -> str:
    payload = _load_json_payload(value)
    if not isinstance(payload, list) or not payload:
        return "-"
    return ",".join(
        f"{item.get('group')}:{int(item.get('count', 0) or 0)}"
        for item in payload
        if isinstance(item, dict) and item.get("group") is not None
    ) or "-"


def _format_split_diagnostics_brief(value: object) -> str:
    payload = _load_json_payload(value)
    if not isinstance(payload, dict):
        return "-"
    ordered_split_names = ["train_full", "train", "validation", "test"]
    parts: list[str] = []
    for split_name in ordered_split_names:
        split_payload = payload.get(split_name)
        if not isinstance(split_payload, dict):
            continue
        underrepresented_groups = split_payload.get("underrepresented_protected_groups", [])
        small_group_count = len(underrepresented_groups) if isinstance(underrepresented_groups, list) else 0
        missing_count = int(split_payload.get("protected_missing_count", 0) or 0)
        protected_counts = split_payload.get("protected_counts", {})
        protected_counts_text = (
            ",".join(
                f"{group_name}:{int(count)}"
                for group_name, count in protected_counts.items()
            )
            if isinstance(protected_counts, dict) and protected_counts
            else "-"
        )
        parts.append(
            f"{split_name}:n={split_payload.get('count', '-')},groups={protected_counts_text},small_groups={small_group_count},missing={missing_count},threshold={split_payload.get('underrepresented_threshold', '-')}"
        )
    return " | ".join(parts) if parts else "-"


def _format_group_error_brief(value: object) -> str:
    if isinstance(value, str) and not value.strip():
        return "-"
    payload = _load_json_payload(value)
    if not isinstance(payload, list) or not payload:
        return "-"
    collapse_groups = [
        str(group_payload.get("group"))
        for group_payload in payload
        if bool(group_payload.get("prediction_collapse"))
    ]
    max_misclassification_rate = max(
        float(group_payload.get("misclassification_rate", 0.0) or 0.0)
        for group_payload in payload
    )
    collapse_text = ",".join(collapse_groups[:3])
    if len(collapse_groups) > 3:
        collapse_text += ",..."
    if not collapse_text:
        collapse_text = "-"
    return (
        f"collapse_groups={collapse_text} | "
        f"max_group_error={max_misclassification_rate:.4f}"
    )


def _format_training_history_brief(value: object, *, best_epoch: object) -> str:
    payload = _load_json_payload(value)
    if not isinstance(payload, list) or not payload:
        return "-"
    collapse_epochs = sum(1 for item in payload if isinstance(item, dict) and item.get("collapsed_groups"))
    parts = [
        f"best_epoch={_format_dim(best_epoch)}",
        f"epochs={len(payload)}",
        f"collapse_epochs={collapse_epochs}",
    ]
    last_entry = payload[-1]
    if isinstance(last_entry, dict):
        if "worst_group_f1_raw" in last_entry:
            parts.append(f"last_worst_group_f1_raw={_format_metric(last_entry.get('worst_group_f1_raw', pd.NA))}")
        elif "worst_group_f1" in last_entry:
            parts.append(f"last_worst_group_f1_raw={_format_metric(last_entry.get('worst_group_f1', pd.NA))}")
        if "worst_group_f1_supported" in last_entry:
            parts.append(
                f"last_worst_group_f1_supported={_format_metric(last_entry.get('worst_group_f1_supported', pd.NA))}"
            )
        if "protected_probe_macro_f1" in last_entry:
            parts.append(
                f"last_probe_macro_f1={_format_metric(last_entry.get('protected_probe_macro_f1', pd.NA))}"
            )
    return " | ".join(parts)


def _comparison_metric_desc(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(float("-inf"))


def _comparison_metric_asc(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(float("inf"))


def _sorted_graphsage_comparison_frame(comparison_frame: pd.DataFrame) -> pd.DataFrame:
    ordered = comparison_frame.copy()
    baseline_macro_f1 = float("-inf")
    baseline_rows = ordered.loc[ordered["comparison_mode"].astype(str) == "baseline"]
    if not baseline_rows.empty:
        baseline_metric = pd.to_numeric(baseline_rows["macro_f1"], errors="coerce").dropna()
        if not baseline_metric.empty:
            baseline_macro_f1 = float(baseline_metric.iloc[0])
    ordered["macro_f1_drop_vs_baseline"] = baseline_macro_f1 - pd.to_numeric(
        ordered["macro_f1"],
        errors="coerce",
    )
    worst_group_f1_supported = ordered.get("worst_group_f1_supported", ordered.get("worst_group_f1", pd.Series(pd.NA, index=ordered.index)))
    worst_group_f1_raw = ordered.get("worst_group_f1_raw", ordered.get("worst_group_f1", pd.Series(pd.NA, index=ordered.index)))
    macro_f1_gap_supported = ordered.get("macro_f1_gap_supported", ordered.get("macro_f1_gap", pd.Series(pd.NA, index=ordered.index)))
    macro_f1_gap_raw = ordered.get("macro_f1_gap_raw", ordered.get("macro_f1_gap", pd.Series(pd.NA, index=ordered.index)))
    protected_probe_macro_f1 = ordered.get("protected_probe_macro_f1", pd.Series(pd.NA, index=ordered.index))
    ordered["_status_rank"] = ordered["status"].map(_status_sort_rank)
    ordered["_macro_f1_sort"] = _comparison_metric_desc(ordered["macro_f1"])
    ordered["_worst_group_f1_raw_sort"] = _comparison_metric_desc(worst_group_f1_raw)
    ordered["_worst_group_f1_supported_sort"] = _comparison_metric_desc(worst_group_f1_supported)
    ordered["_macro_f1_gap_supported_sort"] = _comparison_metric_asc(macro_f1_gap_supported)
    ordered["_probe_f1_sort"] = _comparison_metric_asc(protected_probe_macro_f1)
    ordered["_macro_f1_drop_sort"] = _comparison_metric_asc(ordered["macro_f1_drop_vs_baseline"])
    ordered["_macro_f1_gap_raw_sort"] = _comparison_metric_asc(macro_f1_gap_raw)
    ordered = ordered.sort_values(
        by=[
            "_status_rank",
            "_worst_group_f1_raw_sort",
            "_worst_group_f1_supported_sort",
            "_macro_f1_gap_supported_sort",
            "_probe_f1_sort",
            "_macro_f1_drop_sort",
            "_macro_f1_sort",
            "method",
            "comparison_mode",
        ],
        ascending=[True, False, False, True, True, True, False, True, True],
    ).reset_index(drop=True)
    ordered["comparison_rank"] = np.arange(1, len(ordered) + 1, dtype=int)
    return ordered.drop(
        columns=[
            "_status_rank",
            "_worst_group_f1_raw_sort",
            "_worst_group_f1_supported_sort",
            "_macro_f1_gap_supported_sort",
            "_probe_f1_sort",
            "_macro_f1_drop_sort",
            "_macro_f1_sort",
            "_macro_f1_gap_raw_sort",
        ]
    )


def _format_graphsage_comparison_lines(comparison_frame: pd.DataFrame | None) -> list[str]:
    if comparison_frame is None or comparison_frame.empty:
        return []

    lines = [
        _rule("-"),
        "GraphSAGE Fairness Comparison",
        _rule("-"),
        "Ranked by worst_group_f1_raw desc, worst_group_f1_supported desc, macro_f1_gap_supported asc, leakage asc, macro_f1 drop vs baseline asc.",
    ]
    ordered = _sorted_graphsage_comparison_frame(comparison_frame)
    for _, row in ordered.iterrows():
        lines.extend(
            [
                "",
                f"{int(row['comparison_rank'])}. {row['method']} | {row['comparison_mode']} [{row['status']}]",
                (
                    f"   macro_f1={_format_metric(row.get('macro_f1', pd.NA))} | "
                    f"worst_group_f1_raw={_format_metric(row.get('worst_group_f1_raw', row.get('worst_group_f1', pd.NA)))} | "
                    f"worst_group_f1_supported={_format_metric(row.get('worst_group_f1_supported', pd.NA))} | "
                    f"macro_f1_gap_raw={_format_metric(row.get('macro_f1_gap_raw', row.get('macro_f1_gap', pd.NA)))} | "
                    f"macro_f1_gap_supported={_format_metric(row.get('macro_f1_gap_supported', pd.NA))} | "
                    f"probe_macro_f1={_format_metric(row.get('protected_probe_macro_f1', pd.NA))}"
                ),
                (
                    f"   training: mode={_format_optional_text(row.get('training_mode', pd.NA))} | "
                    f"debias_mode={_format_optional_text(row.get('debias_mode', pd.NA))} | "
                    f"group_weight_mode={_format_optional_text(row.get('group_weight_mode', pd.NA))} | "
                    f"group_robust_weight={_format_metric(row.get('group_robust_weight', pd.NA))} | "
                    f"macro_f1_drop_vs_baseline={_format_metric(row.get('macro_f1_drop_vs_baseline', pd.NA))} | "
                    f"rebalance={_format_optional_text(row.get('rebalance_batches_by_group', pd.NA))}"
                ),
            ]
        )
        lines.append(
            "   diagnostics: "
            f"collapsed_groups={_format_optional_text(row.get('collapsed_groups', pd.NA))} | "
            f"support_threshold={_format_dim(row.get('support_threshold_used', row.get('min_group_support_eval', pd.NA)))} | "
            f"small_group_count={_format_dim(row.get('small_group_count', pd.NA))}"
        )
    return lines


def _sorted_graphsage_repeated_comparison_frame(comparison_frame: pd.DataFrame) -> pd.DataFrame:
    ordered = comparison_frame.copy()
    baseline_macro_f1 = float("-inf")
    baseline_rows = ordered.loc[ordered["comparison_mode"].astype(str) == "baseline"]
    if not baseline_rows.empty:
        baseline_metric = pd.to_numeric(baseline_rows["mean_macro_f1"], errors="coerce").dropna()
        if not baseline_metric.empty:
            baseline_macro_f1 = float(baseline_metric.iloc[0])
    ordered["macro_f1_drop_vs_baseline"] = baseline_macro_f1 - pd.to_numeric(
        ordered["mean_macro_f1"],
        errors="coerce",
    )
    ordered["_status_rank"] = ordered["status"].map(_status_sort_rank)
    ordered["_worst_group_f1_raw_sort"] = _comparison_metric_desc(
        ordered.get("mean_worst_group_f1_raw", pd.Series(pd.NA, index=ordered.index))
    )
    ordered["_worst_group_f1_supported_sort"] = _comparison_metric_desc(
        ordered.get("mean_worst_group_f1_supported", pd.Series(pd.NA, index=ordered.index))
    )
    ordered["_macro_f1_gap_supported_sort"] = _comparison_metric_asc(
        ordered.get("mean_macro_f1_gap_supported", pd.Series(pd.NA, index=ordered.index))
    )
    ordered["_probe_sort"] = _comparison_metric_asc(
        ordered.get("mean_protected_probe_macro_f1", pd.Series(pd.NA, index=ordered.index))
    )
    ordered["_macro_f1_drop_sort"] = _comparison_metric_asc(ordered["macro_f1_drop_vs_baseline"])
    ordered = ordered.sort_values(
        by=[
            "_status_rank",
            "_worst_group_f1_raw_sort",
            "_worst_group_f1_supported_sort",
            "_macro_f1_gap_supported_sort",
            "_probe_sort",
            "_macro_f1_drop_sort",
            "method",
            "comparison_mode",
        ],
        ascending=[True, False, False, True, True, True, True, True],
    ).reset_index(drop=True)
    ordered["comparison_rank"] = np.arange(1, len(ordered) + 1, dtype=int)
    return ordered.drop(
        columns=[
            "_status_rank",
            "_worst_group_f1_raw_sort",
            "_worst_group_f1_supported_sort",
            "_macro_f1_gap_supported_sort",
            "_probe_sort",
            "_macro_f1_drop_sort",
        ]
    )


def _format_graphsage_repeated_comparison_lines(comparison_frame: pd.DataFrame | None) -> list[str]:
    if comparison_frame is None or comparison_frame.empty:
        return []

    lines = [
        _rule("-"),
        "GraphSAGE Repeated Fairness Comparison",
        _rule("-"),
        "Ranked by mean_worst_group_f1_raw desc, mean_worst_group_f1_supported desc, mean_macro_f1_gap_supported asc, mean_protected_probe_macro_f1 asc, macro_f1 drop vs baseline asc.",
    ]
    ordered = _sorted_graphsage_repeated_comparison_frame(comparison_frame)
    for _, row in ordered.iterrows():
        lines.extend(
            [
                "",
                f"{int(row['comparison_rank'])}. {row['method']} | {row['comparison_mode']} [{row['status']}]",
                (
                    f"   runs={_format_dim(row.get('run_count', pd.NA))} | "
                    f"mean_macro_f1={_format_metric(row.get('mean_macro_f1', pd.NA))} +/- {_format_metric(row.get('std_macro_f1', pd.NA))} | "
                    f"mean_worst_group_f1_raw={_format_metric(row.get('mean_worst_group_f1_raw', pd.NA))} +/- {_format_metric(row.get('std_worst_group_f1_raw', pd.NA))}"
                ),
                (
                    f"   mean_worst_group_f1_supported={_format_metric(row.get('mean_worst_group_f1_supported', pd.NA))} +/- {_format_metric(row.get('std_worst_group_f1_supported', pd.NA))} | "
                    f"mean_macro_f1_gap_raw={_format_metric(row.get('mean_macro_f1_gap_raw', pd.NA))} | "
                    f"mean_macro_f1_gap_supported={_format_metric(row.get('mean_macro_f1_gap_supported', pd.NA))}"
                ),
                (
                    f"   leakage={_format_metric(row.get('mean_protected_probe_macro_f1', pd.NA))} +/- {_format_metric(row.get('std_protected_probe_macro_f1', pd.NA))} | "
                    f"macro_f1_drop_vs_baseline={_format_metric(row.get('macro_f1_drop_vs_baseline', pd.NA))} | "
                    f"collapsed_groups={_format_optional_text(row.get('collapsed_groups_notes', pd.NA))}"
                ),
            ]
        )
    return lines


def _format_repeated_evaluation_summary_lines(summary_frame: pd.DataFrame | None) -> list[str]:
    if summary_frame is None or summary_frame.empty:
        return []

    lines = [
        _rule("-"),
        "Repeated Evaluation Summary",
        _rule("-"),
        "Aggregate statistics over evaluation_random_seeds x repeated_split_count for the GraphSAGE node-classification path.",
    ]
    ordered = summary_frame.sort_values(
        by=["task", "method"],
        ascending=[True, True],
    ).reset_index(drop=True)
    for index, row in ordered.iterrows():
        lines.extend(
            [
                "",
                f"{index + 1}. {row['method']} | {row['task']} [{row['status']}]",
                (
                    f"   runs={_format_dim(row.get('run_count', pd.NA))} | "
                    f"mean_macro_f1={_format_metric(row.get('mean_macro_f1', pd.NA))} +/- {_format_metric(row.get('std_macro_f1', pd.NA))} | "
                    f"mean_worst_group_f1_raw={_format_metric(row.get('mean_worst_group_f1_raw', pd.NA))} +/- {_format_metric(row.get('std_worst_group_f1_raw', pd.NA))}"
                ),
                (
                    f"   mean_worst_group_f1_supported={_format_metric(row.get('mean_worst_group_f1_supported', pd.NA))} +/- {_format_metric(row.get('std_worst_group_f1_supported', pd.NA))} | "
                    f"mean_macro_f1_gap_supported={_format_metric(row.get('mean_macro_f1_gap_supported', pd.NA))} | "
                    f"mean_probe_macro_f1={_format_metric(row.get('mean_protected_probe_macro_f1', pd.NA))}"
                ),
                (
                    f"   training: comparison_mode={_format_optional_text(row.get('comparison_mode', pd.NA))} | "
                    f"group_weight_mode={_format_optional_text(row.get('group_weight_mode', pd.NA))} | "
                    f"debias_mode={_format_optional_text(row.get('debias_mode', pd.NA))} | "
                    f"collapsed_groups={_format_optional_text(row.get('collapsed_groups_notes', pd.NA))}"
                ),
            ]
        )
    return lines


def _sorted_graphsage_tuning_frame(tuning_frame: pd.DataFrame) -> pd.DataFrame:
    """Rank a GraphSAGE tuning sweep by fairness first, subject to utility budget."""

    ordered = tuning_frame.copy()
    baseline_rows = ordered.loc[ordered["comparison_mode"].astype(str) == "baseline"]
    baseline_macro_f1 = float("-inf")
    if not baseline_rows.empty:
        baseline_metric = pd.to_numeric(baseline_rows["macro_f1"], errors="coerce").dropna()
        if not baseline_metric.empty:
            baseline_macro_f1 = float(baseline_metric.iloc[0])
    ordered["macro_f1"] = pd.to_numeric(ordered["macro_f1"], errors="coerce")
    ordered["macro_f1_drop"] = baseline_macro_f1 - ordered["macro_f1"]
    ordered["within_macro_f1_budget"] = ordered["macro_f1_drop"] <= 0.03 + 1e-12
    ordered["_status_rank"] = ordered["status"].map(_status_sort_rank)
    ordered["_within_budget_sort"] = ordered["within_macro_f1_budget"].astype(int)
    ordered["_worst_group_f1_sort"] = _comparison_metric_desc(ordered["worst_group_f1"])
    ordered["_worst_group_accuracy_sort"] = _comparison_metric_desc(ordered["worst_group_accuracy"])
    ordered["_macro_f1_gap_sort"] = _comparison_metric_asc(ordered["macro_f1_gap"])
    ordered["_probe_f1_sort"] = _comparison_metric_asc(ordered["protected_probe_macro_f1"])
    ordered["_macro_f1_sort"] = _comparison_metric_desc(ordered["macro_f1"])
    ordered["_accuracy_gap_sort"] = _comparison_metric_asc(ordered["accuracy_gap"])
    ordered = ordered.sort_values(
        by=[
            "_status_rank",
            "_within_budget_sort",
            "_worst_group_f1_sort",
            "_worst_group_accuracy_sort",
            "_macro_f1_gap_sort",
            "_probe_f1_sort",
            "_macro_f1_sort",
            "_accuracy_gap_sort",
            "tuning_label",
            "method",
        ],
        ascending=[True, False, False, False, True, True, False, True, True, True],
    ).reset_index(drop=True)
    ordered["tuning_rank"] = np.arange(1, len(ordered) + 1, dtype=int)
    return ordered.drop(
        columns=[
            "_status_rank",
            "_within_budget_sort",
            "_worst_group_f1_sort",
            "_worst_group_accuracy_sort",
            "_macro_f1_gap_sort",
            "_probe_f1_sort",
            "_macro_f1_sort",
            "_accuracy_gap_sort",
        ]
    )


def _format_graphsage_tuning_lines(tuning_frame: pd.DataFrame | None) -> list[str]:
    """Render one compact GraphSAGE tuning-sweep section."""

    if tuning_frame is None or tuning_frame.empty:
        return []

    lines = [
        _rule("-"),
        "GraphSAGE Mild Tuning Sweep",
        _rule("-"),
        "Ranked by <=3pt Macro F1 drop, worst_group_f1 desc, worst_group_accuracy desc, macro_f1_gap asc, leakage asc, macro_f1 desc, accuracy_gap asc.",
    ]
    ordered = _sorted_graphsage_tuning_frame(tuning_frame)
    for _, row in ordered.iterrows():
        lines.extend(
            [
                "",
                f"{int(row['tuning_rank'])}. {row['method']} | {row['tuning_label']} [{row['status']}]",
                (
                    f"   macro_f1={_format_metric(row.get('macro_f1', pd.NA))} | "
                    f"macro_f1_drop={_format_metric(row.get('macro_f1_drop', pd.NA))} | "
                    f"within_budget={str(bool(row.get('within_macro_f1_budget', False))).lower()} | "
                    f"worst_group_f1={_format_metric(row.get('worst_group_f1', pd.NA))} | "
                    f"worst_group_accuracy={_format_metric(row.get('worst_group_accuracy', pd.NA))} | "
                    f"macro_f1_gap={_format_metric(row.get('macro_f1_gap', pd.NA))} | "
                    f"probe_macro_f1={_format_metric(row.get('protected_probe_macro_f1', pd.NA))}"
                ),
                (
                    f"   training: mode={_format_optional_text(row.get('training_mode', pd.NA))} | "
                    f"debias_mode={_format_optional_text(row.get('debias_mode', pd.NA))} | "
                    f"group_weight_mode={_format_optional_text(row.get('group_weight_mode', pd.NA))} | "
                    f"group_robust_weight={_format_metric(row.get('group_robust_weight', pd.NA))} | "
                    f"adversary_loss_weight={_format_metric(row.get('adversary_loss_weight', pd.NA))} | "
                    f"grl_lambda={_format_metric(row.get('gradient_reversal_lambda', pd.NA))}"
                ),
            ]
        )
    return lines


def _graphsage_comparison_variants(base_kwargs: dict[str, object]) -> list[tuple[str, dict[str, object]]]:
    """Build the preset four-way comparison variants from one base config."""

    return [
        (
            "baseline",
            {
                **base_kwargs,
                "training_mode": "baseline",
                "imbalance_mode": "none",
                "debias_mode": "none",
                "group_weight_mode": "none",
                "group_robust_weight": 0.0,
                "rebalance_batches_by_group": False,
            },
        ),
        (
            "support_aware_group_robust",
            {
                **base_kwargs,
                "training_mode": "support_aware_group_robust",
            },
        ),
        (
            "anti_collapse_group_robust",
            {
                **base_kwargs,
                "training_mode": "anti_collapse_group_robust",
            },
        ),
        (
            "anti_collapse_group_robust_with_mild_adversarial",
            {
                **base_kwargs,
                "training_mode": "anti_collapse_group_robust_with_mild_adversarial",
            },
        ),
    ]


def _mild_strength_levels(
    explicit_value: object,
    *,
    defaults: tuple[float, ...],
    default_reference: float,
) -> list[float]:
    """Return a small, deterministic list of mild sweep strengths."""

    numeric_value = float(explicit_value)
    if abs(numeric_value - float(default_reference)) > 1e-12 and numeric_value > 0.0:
        levels = [numeric_value / 2.0, numeric_value]
    else:
        levels = list(defaults)
    normalized = sorted(
        {
            round(max(0.0, float(level)), 6)
            for level in levels
            if float(level) > 0.0
        }
    )
    return normalized


def _graphsage_tuning_variants(base_kwargs: dict[str, object]) -> list[tuple[str, dict[str, object]]]:
    """Build one lightweight GraphSAGE mild-tuning sweep from base settings."""

    robust_mode = str(base_kwargs.get("group_weight_mode", "none")).strip().lower()
    if robust_mode == "none":
        robust_mode = "worst_group_boost"
    robust_levels = _mild_strength_levels(
        base_kwargs.get("group_robust_weight", 0.0),
        defaults=(0.05, 0.1),
        default_reference=0.0,
    )
    adversary_levels = _mild_strength_levels(
        base_kwargs.get("adversary_loss_weight", 0.0),
        defaults=(0.02, 0.05),
        default_reference=1.0,
    )
    grl_levels = _mild_strength_levels(
        base_kwargs.get("gradient_reversal_lambda", 0.0),
        defaults=(0.02, 0.05),
        default_reference=1.0,
    )
    paired_adv_levels = list(zip(adversary_levels, grl_levels, strict=False))
    adversary_warmup = int(base_kwargs.get("adversary_warmup_epochs", 0) or 20)
    group_warmup = int(base_kwargs.get("group_robust_warmup_epochs", 0) or 20)

    variants: list[tuple[str, dict[str, object]]] = [
        (
            "baseline",
            {
                **base_kwargs,
                "training_mode": "baseline",
                "debias_mode": "none",
                "group_weight_mode": "none",
                "group_robust_weight": 0.0,
                "adversary_loss_weight": 0.0,
                "gradient_reversal_lambda": 0.0,
                "adversary_warmup_epochs": 0,
                "group_robust_warmup_epochs": 0,
                "rebalance_batches_by_group": False,
            },
        )
    ]
    for group_weight in robust_levels:
        variants.append(
            (
                f"group_robust:w={group_weight:.3f}",
                {
                    **base_kwargs,
                    "training_mode": "group_robust",
                    "debias_mode": "none",
                    "group_weight_mode": robust_mode,
                    "group_robust_weight": float(group_weight),
                    "adversary_loss_weight": 0.0,
                    "gradient_reversal_lambda": 0.0,
                    "adversary_warmup_epochs": 0,
                    "group_robust_warmup_epochs": group_warmup,
                },
            ),
        )
    for adversary_loss_weight, grl_lambda in paired_adv_levels:
        variants.append(
            (
                f"adversarial:a={adversary_loss_weight:.3f},grl={grl_lambda:.3f}",
                {
                    **base_kwargs,
                    "training_mode": "adversarial",
                    "debias_mode": "adversarial",
                    "group_weight_mode": "none",
                    "group_robust_weight": 0.0,
                    "adversary_loss_weight": float(adversary_loss_weight),
                    "gradient_reversal_lambda": float(grl_lambda),
                    "adversary_warmup_epochs": adversary_warmup,
                    "group_robust_warmup_epochs": 0,
                    "rebalance_batches_by_group": False,
                },
            ),
        )
    for index, (adversary_loss_weight, grl_lambda) in enumerate(paired_adv_levels):
        group_weight = robust_levels[min(index, len(robust_levels) - 1)]
        variants.append(
            (
                f"adversarial_group_robust:a={adversary_loss_weight:.3f},grl={grl_lambda:.3f},w={group_weight:.3f}",
                {
                    **base_kwargs,
                    "training_mode": "adversarial_group_robust",
                    "debias_mode": "adversarial",
                    "group_weight_mode": robust_mode,
                    "group_robust_weight": float(group_weight),
                    "adversary_loss_weight": float(adversary_loss_weight),
                    "gradient_reversal_lambda": float(grl_lambda),
                    "adversary_warmup_epochs": adversary_warmup,
                    "group_robust_warmup_epochs": group_warmup,
                },
            ),
        )
    return variants


def _task_sort_rank(task_name: object) -> int:
    task_order = {
        "node_classification": 0,
        "link_prediction": 1,
        "node_clustering": 2,
    }
    return task_order.get(str(task_name), len(task_order))


def _status_sort_rank(status_name: object) -> int:
    status_order = {
        "ok": 0,
        "skipped": 1,
        "failed": 2,
    }
    return status_order.get(str(status_name), len(status_order))


def _primary_metric_column(task_name: object) -> tuple[str, str]:
    metric_by_task = {
        "node_classification": ("accuracy", "accuracy"),
        "link_prediction": ("roc_auc", "roc_auc"),
        "node_clustering": ("nmi", "nmi"),
    }
    return metric_by_task.get(str(task_name), ("accuracy", "accuracy"))


def _primary_metric_value(row: pd.Series) -> float:
    metric_column, _ = _primary_metric_column(row.get("task"))
    metric_value = row.get(metric_column, pd.NA)
    if pd.isna(metric_value):
        return float("-inf")
    return float(metric_value)


def _sorted_evaluation_frame(evaluation_frame: pd.DataFrame) -> pd.DataFrame:
    ordered = evaluation_frame.copy()
    ordered["_task_rank"] = ordered["task"].map(_task_sort_rank)
    ordered["_status_rank"] = ordered["status"].map(_status_sort_rank)
    ordered["_primary_metric_value"] = ordered.apply(_primary_metric_value, axis=1)
    ordered = ordered.sort_values(
        by=["_task_rank", "_status_rank", "_primary_metric_value", "method"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    return ordered.drop(columns=["_task_rank", "_status_rank", "_primary_metric_value"])


def _format_evaluation_summary_lines(evaluation_frame: pd.DataFrame | None) -> list[str]:
    if evaluation_frame is None or evaluation_frame.empty:
        return ["No evaluation rows were produced."]

    lines = [
        _rule("-"),
        "Graph Embedding Evaluation Summary",
        _rule("-"),
        "Sorted by primary metric within each task: accuracy for node classification, roc_auc for link prediction, nmi for clustering.",
    ]
    ordered_evaluation = _sorted_evaluation_frame(evaluation_frame)
    for index, row in ordered_evaluation.iterrows():
        primary_metric_column, primary_metric_label = _primary_metric_column(row["task"])
        metric_parts = [
            f"accuracy={_format_metric(row['accuracy'])}",
            f"macro_f1={_format_metric(row['macro_f1'])}",
            f"micro_f1={_format_metric(row['micro_f1'])}",
            f"weighted_f1={_format_metric(row.get('weighted_f1', pd.NA))}",
            f"balanced_accuracy={_format_metric(row.get('balanced_accuracy', pd.NA))}",
            f"macro_precision={_format_metric(row.get('macro_precision', pd.NA))}",
            f"macro_recall={_format_metric(row.get('macro_recall', pd.NA))}",
            f"roc_auc={_format_metric(row['roc_auc'])}",
            f"ap={_format_metric(row['average_precision'])}",
            f"nmi={_format_metric(row['nmi'])}",
            f"ari={_format_metric(row['ari'])}",
        ]
        lines.extend(
            [
                "",
                f"{index + 1}. {row['method']} | {row['task']} [{row['status']}]",
                (
                    f"   runtime={_format_runtime(row['runtime_seconds'])} | "
                    f"embedding_runtime={_format_runtime(row['embedding_runtime_seconds'])} | "
                    f"dim={_format_dim(row['embedding_dim'])} | "
                    f"evaluated={_format_dim(row['evaluated_count'])}"
                ),
                (
                    f"   train={_format_dim(row['train_count'])} | "
                    f"test={_format_dim(row['test_count'])} | "
                    f"classifier={row['classifier'] if not pd.isna(row['classifier']) else '-'} | "
                    f"label_column={row['label_column'] if not pd.isna(row['label_column']) else '-'}"
                ),
                f"   primary_metric={primary_metric_label} ({_format_metric(row.get(primary_metric_column, pd.NA))})",
                f"   metrics: {', '.join(metric_parts)}",
            ]
        )
        if not pd.isna(row.get("edge_feature", pd.NA)):
            lines.append(f"   edge_feature={row['edge_feature']}")
        if str(row.get("task", "")).strip() == "node_clustering":
            lines.append(
                "   clustering: "
                f"method={_format_optional_text(row.get('clustering_method', row.get('classifier', pd.NA)))} | "
                f"category={_format_optional_text(row.get('clustering_category', pd.NA))} | "
                f"requested_input_mode={_format_optional_text(row.get('clustering_requested_input_mode', pd.NA))} | "
                f"input_mode={_format_optional_text(row.get('clustering_input_mode', pd.NA))} | "
                f"clusters={_format_dim(row.get('num_clusters_found', pd.NA))} | "
                f"cluster_sizes={_format_optional_text(row.get('cluster_size_summary', pd.NA))}"
            )
            if (
                not pd.isna(row.get("silhouette_score", pd.NA))
                or not pd.isna(row.get("davies_bouldin_score", pd.NA))
                or not pd.isna(row.get("calinski_harabasz_score", pd.NA))
            ):
                lines.append(
                    "   clustering_metrics: "
                    f"silhouette={_format_metric(row.get('silhouette_score', pd.NA))} | "
                    f"davies_bouldin={_format_metric(row.get('davies_bouldin_score', pd.NA))} | "
                    f"calinski_harabasz={_format_metric(row.get('calinski_harabasz_score', pd.NA))}"
                )
        if not pd.isna(row.get("worst_group_accuracy", pd.NA)):
            lines.append(
                "   fairness_raw: "
                f"protected_attribute={_format_optional_text(row.get('protected_attribute_column', pd.NA))} | "
                f"worst_group_accuracy_raw={_format_metric(row.get('worst_group_accuracy_raw', row.get('worst_group_accuracy', pd.NA)))} | "
                f"accuracy_gap_raw={_format_metric(row.get('accuracy_gap_raw', row.get('accuracy_gap', row.get('group_accuracy_gap', pd.NA))))} | "
                f"worst_group_f1_raw={_format_metric(row.get('worst_group_f1_raw', row.get('worst_group_f1', row.get('worst_group_macro_f1', pd.NA))))} | "
                f"macro_f1_gap_raw={_format_metric(row.get('macro_f1_gap_raw', row.get('macro_f1_gap', row.get('group_macro_f1_gap', pd.NA))))}"
            )
            lines.append(
                "   fairness_supported: "
                f"support_threshold={_format_dim(row.get('support_threshold_used', row.get('min_group_support_eval', pd.NA)))} | "
                f"supported_groups={_format_dim(row.get('supported_group_count', pd.NA))} | "
                f"worst_group_accuracy_supported={_format_metric(row.get('worst_group_accuracy_supported', pd.NA))} | "
                f"accuracy_gap_supported={_format_metric(row.get('accuracy_gap_supported', pd.NA))} | "
                f"worst_group_f1_supported={_format_metric(row.get('worst_group_f1_supported', pd.NA))} | "
                f"macro_f1_gap_supported={_format_metric(row.get('macro_f1_gap_supported', pd.NA))}"
            )
            worst_accuracy_group = _format_optional_text(row.get("worst_group_accuracy_name", pd.NA))
            worst_f1_group = _format_optional_text(row.get("worst_group_f1_name", pd.NA))
            if worst_accuracy_group != "-" or worst_f1_group != "-":
                lines.append(
                    "   worst_groups: "
                    f"accuracy={worst_accuracy_group}(n={_format_dim(row.get('worst_group_accuracy_count', pd.NA))}) | "
                    f"f1={worst_f1_group}(n={_format_dim(row.get('worst_group_f1_count', pd.NA))})"
                )
        training_mode_value = row.get("training_mode", pd.NA)
        debias_mode_value = row.get("debias_mode", pd.NA)
        if not pd.isna(training_mode_value) or not pd.isna(debias_mode_value):
            comparison_mode_text = _format_optional_text(row.get("comparison_mode", pd.NA))
            comparison_prefix = ""
            if comparison_mode_text != "-":
                comparison_prefix = f"comparison_mode={comparison_mode_text} | "
            lines.append(
                "   training: "
                f"{comparison_prefix}"
                f"mode={_format_optional_text(training_mode_value)} | "
                f"debias_mode={_format_optional_text(debias_mode_value)} | "
                f"imbalance_mode={_format_optional_text(row.get('imbalance_mode', pd.NA))} | "
                f"group_weight_mode={_format_optional_text(row.get('group_weight_mode', pd.NA))} | "
                f"group_robust_weight={_format_metric(row.get('group_robust_weight', pd.NA))} | "
                f"rebalance_batches_by_group={_format_optional_text(row.get('rebalance_batches_by_group', pd.NA))} | "
                f"early_stop_metric={_format_optional_text(row.get('early_stop_metric', pd.NA))}"
            )
        if not pd.isna(row.get("min_group_support_train", pd.NA)) or not pd.isna(row.get("min_group_support_eval", pd.NA)):
            lines.append(
                "   support: "
                f"train_threshold={_format_dim(row.get('min_group_support_train', row.get('min_group_support_threshold', pd.NA)))} | "
                f"eval_threshold={_format_dim(row.get('min_group_support_eval', row.get('min_group_support_threshold', pd.NA)))} | "
                f"report_small_group_metrics={_format_optional_text(row.get('report_small_group_metrics', pd.NA))}"
            )
        if _format_optional_text(row.get("group_weight_mode", pd.NA)) != "none":
            robust_parts = []
            if _format_optional_text(row.get("group_weight_mode", pd.NA)) == "min_support_boost":
                robust_parts.append(
                    f"min_support_boost_factor={_format_metric(row.get('min_support_boost_factor', pd.NA))}"
                )
            else:
                robust_parts.append(
                    f"boost_factor={_format_metric(row.get('worst_group_boost_factor', pd.NA))}"
                )
            robust_parts.append(
                f"warmup_epochs={_format_dim(row.get('group_robust_warmup_epochs', pd.NA))}"
            )
            lines.append(f"   robust: {' | '.join(robust_parts)}")
        if str(row.get("early_stop_metric", "")).strip() == "fairness_score":
            lines.append(
                "   fairness_score: "
                f"alpha={_format_metric(row.get('fairness_score_alpha', pd.NA))} | "
                f"beta={_format_metric(row.get('fairness_score_beta', pd.NA))}"
            )
        if not pd.isna(debias_mode_value) and str(debias_mode_value).strip() == "adversarial":
            lines.append(
                "   adversary: "
                f"loss_weight={_format_metric(row.get('adversary_loss_weight', pd.NA))} | "
                f"grl_lambda={_format_metric(row.get('gradient_reversal_lambda', pd.NA))} | "
                f"warmup_epochs={_format_dim(row.get('adversary_warmup_epochs', pd.NA))} | "
                f"hidden_dim={_format_dim(row.get('adversary_hidden_dim', pd.NA))} | "
                f"layers={_format_dim(row.get('adversary_num_layers', pd.NA))} | "
                f"dropout={_format_metric(row.get('adversary_dropout', pd.NA))}"
            )
        protected_probe_status_value = row.get("protected_probe_status", pd.NA)
        protected_probe_status = (
            "" if pd.isna(protected_probe_status_value) else str(protected_probe_status_value).strip()
        )
        if protected_probe_status:
            lines.append(
                "   leakage: "
                f"protected_probe={protected_probe_status} | "
                f"accuracy={_format_metric(row.get('protected_probe_accuracy', pd.NA))} | "
                f"macro_f1={_format_metric(row.get('protected_probe_macro_f1', pd.NA))} | "
                f"type={_format_optional_text(row.get('protected_probe_model_type', pd.NA))}"
            )
        diagnostics_brief = _format_split_diagnostics_brief(row.get("split_diagnostics_json", pd.NA))
        if diagnostics_brief != "-":
            lines.append(f"   diagnostics: {diagnostics_brief}")
        collapsed_groups_text = _format_optional_text(row.get("collapsed_groups", pd.NA))
        if (
            collapsed_groups_text != "-"
            or not pd.isna(row.get("small_group_count", pd.NA))
            or not pd.isna(row.get("max_group_error", pd.NA))
        ):
            lines.append(
                "   diagnostics_summary: "
                f"collapsed_groups={collapsed_groups_text} | "
                f"small_group_count={_format_dim(row.get('small_group_count', pd.NA))} | "
                f"max_group_error={_format_metric(row.get('max_group_error', pd.NA))}"
            )
        group_support_counts_text = _format_group_support_counts_brief(row.get("group_support_counts_json", pd.NA))
        if group_support_counts_text != "-":
            lines.append(
                "   support_counts: "
                f"{group_support_counts_text} | "
                f"below_threshold={_format_small_groups_brief(row.get('groups_below_support_threshold_json', pd.NA))}"
            )
        group_error_brief = _format_group_error_brief(row.get("group_error_diagnostics_json", pd.NA))
        if group_error_brief != "-":
            lines.append(f"   group_errors: {group_error_brief}")
        training_history_brief = _format_training_history_brief(
            row.get("training_history_json", pd.NA),
            best_epoch=row.get("best_epoch", pd.NA),
        )
        if training_history_brief != "-":
            lines.append(f"   history: {training_history_brief}")
        if _has_text(row.get("protected_probe_reason", pd.NA)):
            lines.append(f"   protected_probe_reason={row['protected_probe_reason']}")
        if _has_text(row.get("notes", pd.NA)):
            lines.append(f"   notes={row['notes']}")
        if _has_text(row.get("unavailable_reason", pd.NA)):
            lines.append(f"   unavailable_reason={row['unavailable_reason']}")
        if _has_text(row.get("skipped_reason", pd.NA)):
            lines.append(f"   skipped_reason={row['skipped_reason']}")
    return lines


def _format_summary_row_lines(index: int, row: pd.Series) -> list[str]:
    """Render one embedding-method summary block."""

    lines = [
        "",
        f"{index + 1}. {row['method']} [{row['status']}]",
        (
            f"   runtime={_format_runtime(row['runtime_seconds'])} | "
            f"dim={_format_dim(row['embedding_dim'])} | "
            f"nodes={_format_dim(row['node_count'])} | "
            f"ml_ready={row['ml_ready']}"
        ),
        (
            f"   all_nodes={row['all_nodes_embedded']} | "
            f"all_finite={row['all_finite']} | "
            f"cosine_mean={row['mean_pairwise_cosine'] if not pd.isna(row['mean_pairwise_cosine']) else '-'}"
        ),
        (
            f"   label_probe={row.get('label_probe_status', 'not_requested')} | "
            f"accuracy={_format_metric(row.get('label_probe_accuracy', pd.NA))} | "
            f"macro_f1={_format_metric(row.get('label_probe_macro_f1', pd.NA))} | "
            f"type={_format_optional_text(row.get('label_probe_model_type', pd.NA))}"
        ),
    ]
    protected_probe_status_value = row.get("protected_probe_status", pd.NA)
    protected_probe_status = (
        "" if pd.isna(protected_probe_status_value) else str(protected_probe_status_value).strip()
    )
    if protected_probe_status and protected_probe_status != "not_requested":
        lines.append(
            "   "
            f"protected_probe={protected_probe_status} | "
            f"column={_format_optional_text(row.get('protected_probe_column', pd.NA))} | "
            f"accuracy={_format_metric(row.get('protected_probe_accuracy', pd.NA))} | "
            f"macro_f1={_format_metric(row.get('protected_probe_macro_f1', pd.NA))} | "
            f"type={_format_optional_text(row.get('protected_probe_model_type', pd.NA))}"
        )
    if _has_text(row.get("protected_probe_reason", pd.NA)):
        lines.append(f"   protected_probe_reason={row['protected_probe_reason']}")
    if _has_text(row.get("label_probe_reason", pd.NA)):
        lines.append(f"   label_probe_reason={row['label_probe_reason']}")
    if not pd.isna(row["csv_path"]) or not pd.isna(row["pickle_path"]) or not pd.isna(row["npy_path"]):
        lines.append(
            "   "
            f"exports: csv={_display_path(row['csv_path'])} | "
            f"pickle={_display_path(row['pickle_path'])} | "
            f"npy={_display_path(row['npy_path'])}"
        )
    if _has_text(row.get("skip_reason", pd.NA)):
        lines.append(f"   skip_reason={row['skip_reason']}")
    if _has_text(row.get("error_message", pd.NA)):
        lines.append(f"   error={row['error_message']}")
    return lines


def format_benchmark_report(
    summary_frame: pd.DataFrame,
    *,
    evaluation_frame: pd.DataFrame | None = None,
    repeated_evaluation_summary_frame: pd.DataFrame | None = None,
    comparison_frame: pd.DataFrame | None = None,
    repeated_comparison_frame: pd.DataFrame | None = None,
    tuning_frame: pd.DataFrame | None = None,
    dataset_name: str,
    output_dir: Path | None,
) -> str:
    """Render a compact terminal report for the embedding benchmark."""

    if summary_frame.empty:
        return "No embedding benchmark results were produced."

    lines = [
        _rule("="),
        "Graph Embedding Benchmark Summary",
        _rule("="),
        f"Dataset: {dataset_name}",
        f"Output directory: {output_dir if output_dir is not None else '-'}",
        f"Methods available: {', '.join(available_embedding_methods())}",
    ]

    ordered = summary_frame.sort_values(
        by=["status", "runtime_seconds", "method"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    for index, row in ordered.iterrows():
        lines.extend(_format_summary_row_lines(index, row))

    if evaluation_frame is not None and not evaluation_frame.empty:
        lines.extend(["", *_format_evaluation_summary_lines(evaluation_frame)])
    if repeated_evaluation_summary_frame is not None and not repeated_evaluation_summary_frame.empty:
        lines.extend(["", *_format_repeated_evaluation_summary_lines(repeated_evaluation_summary_frame)])
    if comparison_frame is not None and not comparison_frame.empty:
        lines.extend(["", *_format_graphsage_comparison_lines(comparison_frame)])
    if repeated_comparison_frame is not None and not repeated_comparison_frame.empty:
        lines.extend(["", *_format_graphsage_repeated_comparison_lines(repeated_comparison_frame)])
    if tuning_frame is not None and not tuning_frame.empty:
        lines.extend(["", *_format_graphsage_tuning_lines(tuning_frame)])

    return "\n".join(lines)


def render_saved_benchmark_summary_text(
    result,
    *,
    report_path: Path,
    repeated_evaluation_summary_frame: pd.DataFrame | None = None,
    comparison_frame: pd.DataFrame | None = None,
    repeated_comparison_frame: pd.DataFrame | None = None,
    tuning_frame: pd.DataFrame | None = None,
) -> str:
    """Render the exact saved-summary text for both terminal and file output."""

    lines = [f"Saved benchmark report to {report_path}"]
    if not result.summary_frame.empty:
        status_counts = result.summary_frame["status"].astype(str).value_counts().to_dict()
        lines.append(f"Method status counts: {status_counts}")
        ordered = result.summary_frame.sort_values(
            by=["status", "runtime_seconds", "method"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
        for index, row in ordered.iterrows():
            lines.extend(["", *_format_summary_row_lines(index, row)])
    if not result.evaluation_frame.empty:
        evaluation_status_counts = result.evaluation_frame["status"].astype(str).value_counts().to_dict()
        lines.append(f"Evaluation status counts: {evaluation_status_counts}")
        lines.extend(["", *_format_evaluation_summary_lines(result.evaluation_frame)])
    if repeated_evaluation_summary_frame is not None and not repeated_evaluation_summary_frame.empty:
        repeated_status_counts = repeated_evaluation_summary_frame["status"].astype(str).value_counts().to_dict()
        lines.append(f"Repeated evaluation status counts: {repeated_status_counts}")
        lines.extend(["", *_format_repeated_evaluation_summary_lines(repeated_evaluation_summary_frame)])
    if comparison_frame is not None and not comparison_frame.empty:
        comparison_status_counts = comparison_frame["status"].astype(str).value_counts().to_dict()
        lines.append(f"Comparison status counts: {comparison_status_counts}")
        lines.extend(["", *_format_graphsage_comparison_lines(comparison_frame)])
    if repeated_comparison_frame is not None and not repeated_comparison_frame.empty:
        repeated_comparison_status_counts = repeated_comparison_frame["status"].astype(str).value_counts().to_dict()
        lines.append(f"Repeated comparison status counts: {repeated_comparison_status_counts}")
        lines.extend(["", *_format_graphsage_repeated_comparison_lines(repeated_comparison_frame)])
    if tuning_frame is not None and not tuning_frame.empty:
        tuning_status_counts = tuning_frame["status"].astype(str).value_counts().to_dict()
        lines.append(f"Tuning status counts: {tuning_status_counts}")
        lines.extend(["", *_format_graphsage_tuning_lines(tuning_frame)])
    return "\n".join(lines)


def print_saved_benchmark_summary(
    result,
    *,
    report_path: Path,
) -> None:
    """Print a compact terminal summary when the report was written to disk."""

    print(render_saved_benchmark_summary_text(result, report_path=report_path))


def _attribute_column_names(dataset) -> list[str]:
    return [column for column in dataset.node_attributes.columns if column != "node_id"]


def _all_attributes_requested(args: argparse.Namespace) -> bool:
    label_values = [
        args.label_column,
        args.classification_label_column,
        args.clustering_label_column,
    ]
    return any(
        value is not None and str(value).strip().lower() == "all"
        for value in label_values
    )


def _validate_all_attributes_mode(args: argparse.Namespace) -> None:
    label_values = [
        args.label_column,
        args.classification_label_column,
        args.clustering_label_column,
    ]
    explicit_values = {
        str(value).strip().lower()
        for value in label_values
        if value is not None and str(value).strip()
    }
    if "all" in explicit_values and explicit_values.difference({"all"}):
        raise ValueError(
            "When using all-attributes mode, do not mix 'all' with explicit label columns. "
            "Use --label-column all by itself."
        )


def format_attribute_benchmark_report(
    summary_frame: pd.DataFrame,
    *,
    evaluation_frame: pd.DataFrame,
    dataset_name: str,
    attribute_name: str,
    output_dir: Path | None,
) -> str:
    """Render one readable text report for a single dataset attribute."""

    base_report = benchmark_report_path(output_dir, dataset_name)
    header_lines = [
        _rule("="),
        "Graph Embedding Benchmark Report By Attribute",
        _rule("="),
        f"Dataset: {dataset_name}",
        f"Attribute: {attribute_name}",
        f"Base embedding report: {base_report if base_report is not None else '-'}",
        "The base embedding summary is shared across attributes; only the evaluation rows below change by attribute.",
        "",
    ]
    embedding_summary_lines = [
        "Embedding summary:",
        f"- methods={', '.join(summary_frame['method'].astype(str).tolist())}",
        f"- statuses={summary_frame['status'].astype(str).value_counts().to_dict()}",
        "",
        *_format_evaluation_summary_lines(evaluation_frame),
    ]
    return "\n".join(header_lines + embedding_summary_lines)


def format_all_attributes_benchmark_report(
    dataset_name: str,
    *,
    base_report_text: str,
    attribute_reports: list[dict[str, Any]],
    output_dir: Path | None,
) -> str:
    """Render one readable report covering all attributes for a dataset."""

    lines = [
        _rule("="),
        "All-Attributes Graph Embedding Benchmark Summary",
        _rule("="),
        f"Dataset: {dataset_name}",
        f"Output directory: {output_dir if output_dir is not None else '-'}",
        f"Attributes evaluated: {', '.join(report['attribute_name'] for report in attribute_reports)}",
        "",
        _rule("-"),
        "Base Embedding Benchmark",
        _rule("-"),
        "",
        base_report_text,
    ]

    if attribute_reports:
        lines.extend(["", _rule("-"), "Attribute-Specific Reports", _rule("-")])
        for report in attribute_reports:
            evaluation_frame = report.get("evaluation_frame")
            lines.extend(
                [
                    "",
                    _rule("="),
                    f"Attribute: {report['attribute_name']}",
                    _rule("="),
                    f"Report path: {report['report_path'] if report['report_path'] is not None else '-'}",
                    "",
                    *_format_evaluation_summary_lines(evaluation_frame),
                ]
            )

    return "\n".join(lines)


def _run_graphsage_comparison_preset(
    dataset,
    benchmark_result,
    *,
    preset_name: str,
    evaluation_kwargs: dict[str, object],
    label_column: str | None,
    classification_label_column: str | None,
    clustering_label_column: str | None,
    random_seed: int,
    symmetrize_directed: bool,
) -> pd.DataFrame | None:
    """Run one lightweight GraphSAGE comparison preset on existing embeddings."""

    normalized_preset = str(preset_name).strip().lower()
    if normalized_preset in {"", "none"}:
        return None
    if normalized_preset != "fairness_tradeoff":
        raise ValueError(
            "Unsupported graphsage_comparison_preset "
            f"'{preset_name}'. Supported values: ['none', 'fairness_tradeoff']."
        )
    if str(evaluation_kwargs.get("node_classification_model", "")).strip().lower() != "graphsage":
        raise ValueError("graphsage_comparison_preset requires --node-classification-model graphsage.")
    if evaluation_kwargs.get("protected_attribute_column") in {None, ""}:
        raise ValueError("graphsage_comparison_preset requires --protected-attribute-column.")
    base_kwargs = dict(evaluation_kwargs)
    comparison_frames: list[pd.DataFrame] = []
    for expected_mode, variant_kwargs in _graphsage_comparison_variants(base_kwargs):
        frame = evaluate_embedding_benchmark(
            dataset,
            benchmark_result,
            tasks=["node_classification"],
            random_seed=random_seed,
            label_column=label_column,
            classification_label_column=classification_label_column,
            clustering_label_column=clustering_label_column,
            symmetrize_directed=symmetrize_directed,
            **variant_kwargs,
        )
        if frame.empty:
            continue
        frame = frame.copy()
        frame["requested_comparison_mode"] = expected_mode
        comparison_frames.append(frame)
    if not comparison_frames:
        return None

    comparison_frame = pd.concat(comparison_frames, ignore_index=True)
    comparison_frame = _sorted_graphsage_comparison_frame(comparison_frame)
    comparison_path = graphsage_comparison_summary_path(
        benchmark_result.output_dir,
        benchmark_result.dataset_name,
    )
    if comparison_path is not None:
        comparison_frame.to_csv(comparison_path, index=False)
    return comparison_frame


def _run_graphsage_repeated_comparison_preset(
    dataset,
    benchmark_result,
    *,
    preset_name: str,
    evaluation_kwargs: dict[str, object],
    label_column: str | None,
    classification_label_column: str | None,
    clustering_label_column: str | None,
    random_seed: int,
    evaluation_random_seeds: list[int] | None,
    repeated_split_count: int,
    symmetrize_directed: bool,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Run repeated seeded/split evaluation for the four GraphSAGE comparison modes."""

    normalized_preset = str(preset_name).strip().lower()
    if normalized_preset in {"", "none"}:
        return None, None

    base_kwargs = dict(evaluation_kwargs)
    comparison_run_frames: list[pd.DataFrame] = []
    comparison_summary_frames: list[pd.DataFrame] = []
    for expected_mode, variant_kwargs in _graphsage_comparison_variants(base_kwargs):
        per_run_frame, aggregate_frame = evaluate_embedding_benchmark_repeated(
            dataset,
            benchmark_result,
            evaluation_random_seeds=evaluation_random_seeds,
            repeated_split_count=repeated_split_count,
            tasks=["node_classification"],
            random_seed=random_seed,
            label_column=label_column,
            classification_label_column=classification_label_column,
            clustering_label_column=clustering_label_column,
            symmetrize_directed=symmetrize_directed,
            **variant_kwargs,
        )
        if not per_run_frame.empty:
            per_run_frame = per_run_frame.copy()
            per_run_frame["requested_comparison_mode"] = expected_mode
            comparison_run_frames.append(per_run_frame)
        if not aggregate_frame.empty:
            aggregate_frame = aggregate_frame.copy()
            aggregate_frame["requested_comparison_mode"] = expected_mode
            comparison_summary_frames.append(aggregate_frame)

    combined_run_frame = (
        pd.concat(comparison_run_frames, ignore_index=True)
        if comparison_run_frames
        else None
    )
    combined_summary_frame = (
        _sorted_graphsage_repeated_comparison_frame(
            pd.concat(comparison_summary_frames, ignore_index=True)
        )
        if comparison_summary_frames
        else None
    )
    runs_path = graphsage_repeated_comparison_runs_path(
        benchmark_result.output_dir,
        benchmark_result.dataset_name,
    )
    if runs_path is not None and combined_run_frame is not None:
        combined_run_frame.to_csv(runs_path, index=False)
    summary_path = graphsage_repeated_comparison_summary_path(
        benchmark_result.output_dir,
        benchmark_result.dataset_name,
    )
    if summary_path is not None and combined_summary_frame is not None:
        combined_summary_frame.to_csv(summary_path, index=False)
    return combined_run_frame, combined_summary_frame


def _run_graphsage_tuning_preset(
    dataset,
    benchmark_result,
    *,
    preset_name: str,
    evaluation_kwargs: dict[str, object],
    label_column: str | None,
    classification_label_column: str | None,
    clustering_label_column: str | None,
    random_seed: int,
    symmetrize_directed: bool,
) -> pd.DataFrame | None:
    """Run one lightweight GraphSAGE mild-tuning sweep on existing embeddings."""

    normalized_preset = str(preset_name).strip().lower()
    if normalized_preset in {"", "none"}:
        return None
    if normalized_preset != "mild_tradeoff":
        raise ValueError(
            "Unsupported graphsage_tuning_preset "
            f"'{preset_name}'. Supported values: ['none', 'mild_tradeoff']."
        )
    if str(evaluation_kwargs.get("node_classification_model", "")).strip().lower() != "graphsage":
        raise ValueError("graphsage_tuning_preset requires --node-classification-model graphsage.")
    if evaluation_kwargs.get("protected_attribute_column") in {None, ""}:
        raise ValueError("graphsage_tuning_preset requires --protected-attribute-column.")

    base_kwargs = dict(evaluation_kwargs)
    tuning_frames: list[pd.DataFrame] = []
    for tuning_label, variant_kwargs in _graphsage_tuning_variants(base_kwargs):
        frame = evaluate_embedding_benchmark(
            dataset,
            benchmark_result,
            tasks=["node_classification"],
            random_seed=random_seed,
            label_column=label_column,
            classification_label_column=classification_label_column,
            clustering_label_column=clustering_label_column,
            symmetrize_directed=symmetrize_directed,
            **variant_kwargs,
        )
        if frame.empty:
            continue
        frame = frame.copy()
        frame["tuning_label"] = tuning_label
        tuning_frames.append(frame)
    if not tuning_frames:
        return None

    tuning_frame = pd.concat(tuning_frames, ignore_index=True)
    tuning_frame = _sorted_graphsage_tuning_frame(tuning_frame)
    tuning_path = graphsage_tuning_summary_path(
        benchmark_result.output_dir,
        benchmark_result.dataset_name,
    )
    if tuning_path is not None:
        tuning_frame.to_csv(tuning_path, index=False)
    return tuning_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modular graph embedding benchmark.")
    parser.add_argument(
        "--dataset",
        default="graph_spa_500_0",
        help="Built-in dataset name, a supported custom dataset file path, or a dataset stem under networks/.",
    )
    parser.add_argument("--graph-path", default=None, help="Path to an external graph file.")
    parser.add_argument(
        "--attributes-path",
        "--attribute-path",
        dest="attributes_path",
        default=None,
        help="Optional path to a separate node-attribute file.",
    )
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "pickle", "pkl", "txt", "csv"],
        default="auto",
        help="External graph format. Use 'auto' to infer from the file extension.",
    )
    parser.add_argument("--dataset-config", default=None, help="Optional JSON dataset config file.")
    parser.add_argument(
        "--directed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the external graph directed flag when needed.",
    )
    parser.add_argument("--source-col", default=None, help="Source column name for CSV edge lists.")
    parser.add_argument("--target-col", default=None, help="Target column name for CSV edge lists.")
    parser.add_argument("--node-id-col", default=None, help="Node ID column name for attribute files.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Embedding methods to run. Use 'all' for every registered method.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Embedding methods to run concurrently. Use 0 to run all selected methods at once, or 1 for serial execution.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Base output directory. Per-method embeddings are saved under results/<dataset>/embeddings/.",
    )
    parser.add_argument(
        "--export-formats",
        nargs="+",
        default=["csv", "pickle"],
        choices=["csv", "pickle", "npy"],
        help="Embedding export formats.",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Fallback label column for the simple probe and downstream evaluation tasks. Use 'all' to evaluate every dataset attribute.",
    )
    parser.add_argument(
        "--evaluation-tasks",
        nargs="+",
        default=[],
        choices=["all", *available_evaluation_tasks()],
        help="Optional downstream evaluation tasks to run after embedding generation.",
    )
    parser.add_argument(
        "--classification-label-column",
        default=None,
        help="Optional node-classification label column. Falls back to --label-column when omitted.",
    )
    parser.add_argument(
        "--clustering-label-column",
        default=None,
        help="Optional node-clustering label column. Falls back to --label-column when omitted.",
    )
    parser.add_argument(
        "--symmetrize-directed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Symmetrize directed graphs before benchmarking for cross-method comparability.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep running remaining methods when one method fails.",
    )
    parser.add_argument("--embedding-dim", type=int, default=64, help="Final embedding width for every method.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducible runs.")
    parser.add_argument("--walk-length", type=int, default=20, help="Walk length for DeepWalk/Node2Vec.")
    parser.add_argument("--num-walks", type=int, default=10, help="Walks per node for DeepWalk/Node2Vec.")
    parser.add_argument("--window-size", type=int, default=5, help="Context window for DeepWalk/Node2Vec.")
    parser.add_argument("--node2vec-p", type=float, default=1.0, help="Node2Vec return parameter p.")
    parser.add_argument("--node2vec-q", type=float, default=1.0, help="Node2Vec in-out parameter q.")
    parser.add_argument(
        "--line-order",
        choices=["first", "second", "both"],
        default="both",
        help="LINE proximity order.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden width for GNN-based methods.")
    parser.add_argument("--num-layers", type=int, default=2, help="Message-passing layers for GNN methods.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for GNN methods.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for GNN methods.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay for GNN methods.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs for GNN methods.")
    parser.add_argument("--projection-dim", type=int, default=64, help="Projection width for GraphCL.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Contrastive temperature for GraphCL.")
    parser.add_argument(
        "--node-test-fraction",
        type=float,
        default=0.25,
        help="Test fraction for seeded node-classification splits.",
    )
    parser.add_argument(
        "--link-test-fraction",
        type=float,
        default=0.25,
        help="Test fraction for seeded link-prediction positive-edge splits.",
    )
    parser.add_argument(
        "--link-negative-ratio",
        type=float,
        default=1.0,
        help="Negative edge ratio relative to positives for link prediction.",
    )
    parser.add_argument(
        "--link-prediction-edge-feature",
        choices=["hadamard", "abs_diff", "concat", "dot"],
        default="hadamard",
        help="Edge feature construction for link prediction.",
    )
    parser.add_argument(
        "--node-classification-model",
        choices=["logistic_regression", "graphsage"],
        default="logistic_regression",
        help="Classifier used for node-classification evaluation.",
    )
    parser.add_argument(
        "--clustering-method",
        choices=[
            "louvain",
            "leiden",
            "multilevel",
            "infomap",
            "label_propagation",
            "walktrap",
            "kmeans",
            "spectral",
            "agglomerative",
            "dbscan_or_hdbscan",
            "gmm",
        ],
        default="kmeans",
        help="Clustering method used for node-clustering evaluation.",
    )
    parser.add_argument(
        "--clustering-input-mode",
        choices=["graph", "embedding", "auto"],
        default="embedding",
        help="Input routing for node-clustering evaluation. Graph-native methods ignore this and use the graph directly.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Optional fixed cluster count for embedding-space node-clustering methods.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        help="Optional minimum cluster size for density-based clustering methods.",
    )
    parser.add_argument(
        "--clustering-method-config-json",
        default="{}",
        help="JSON object with additional clustering-method overrides, e.g. '{\"eps\":0.4}'.",
    )
    parser.add_argument(
        "--training-mode",
        choices=[
            "baseline",
            "imbalance_only",
            "group_robust",
            "adversarial",
            "adversarial_group_robust",
            "support_aware_group_robust",
            "anti_collapse_group_robust",
            "anti_collapse_group_robust_with_mild_adversarial",
        ],
        default=None,
        help="Optional high-level GraphSAGE training mode. When omitted, it is inferred from the low-level debiasing and weighting flags.",
    )
    parser.add_argument(
        "--debias-mode",
        choices=["none", "adversarial"],
        default="none",
        help="Optional debiasing mode for GraphSAGE node classification.",
    )
    parser.add_argument(
        "--protected-attribute-column",
        default=None,
        help="Optional protected attribute column for per-group diagnostics and protected-attribute probing.",
    )
    parser.add_argument(
        "--run-protected-attribute-probe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a simple auxiliary probe on embeddings to predict the protected attribute column.",
    )
    parser.add_argument(
        "--imbalance-mode",
        choices=["none", "class_weighted", "focal_loss", "weighted_sampler"],
        default="none",
        help="Optional class-imbalance handling for GraphSAGE node classification.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for focal loss when --imbalance-mode focal_loss is used.",
    )
    parser.add_argument(
        "--use-stratified-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use stratified seeded train/test and train/validation splits when possible.",
    )
    parser.add_argument(
        "--early-stop-metric",
        choices=["accuracy", "macro_f1", "worst_group_f1", "worst_group_f1_raw", "worst_group_f1_supported", "fairness_score"],
        default="accuracy",
        help="Validation metric for GraphSAGE early stopping.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="GraphSAGE early-stopping patience in epochs; 0 disables early stopping.",
    )
    parser.add_argument(
        "--class-weight-smoothing",
        type=float,
        default=0.0,
        help="Optional additive smoothing for class-weight and weighted-sampler counts.",
    )
    parser.add_argument(
        "--node-validation-fraction",
        type=float,
        default=0.2,
        help="Validation fraction carved from the training nodes for GraphSAGE node classification.",
    )
    parser.add_argument(
        "--group-robust-weight",
        type=float,
        default=0.0,
        help="Strength of protected-group robust weighting for GraphSAGE when group weighting is enabled.",
    )
    parser.add_argument(
        "--group-weight-mode",
        choices=["none", "inverse_frequency", "worst_group_boost", "group_dro", "min_support_boost"],
        default="none",
        help="Protected-group weighting mode for GraphSAGE robust training.",
    )
    parser.add_argument(
        "--worst-group-boost-factor",
        type=float,
        default=2.0,
        help="Boost applied to collapsed or worst-loss protected groups in worst_group_boost mode.",
    )
    parser.add_argument(
        "--min-support-boost-factor",
        type=float,
        default=2.0,
        help="Additional multiplier applied to inverse-frequency weights for protected groups below the train support threshold in min_support_boost mode.",
    )
    parser.add_argument(
        "--min-group-support-threshold",
        type=int,
        default=None,
        help="Legacy shared protected-group support threshold alias. When set without the new train/eval flags, it is applied to both.",
    )
    parser.add_argument(
        "--min-group-support-train",
        type=int,
        default=None,
        help="Minimum protected-group support used by training-time robust weighting and train/validation diagnostics.",
    )
    parser.add_argument(
        "--min-group-support-eval",
        type=int,
        default=None,
        help="Minimum protected-group support used by support-aware fairness metrics and test diagnostics.",
    )
    parser.add_argument(
        "--report-small-group-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include explicit small-group metric payloads in saved evaluation rows and reports.",
    )
    parser.add_argument(
        "--rebalance-batches-by-group",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deterministically rebalance supervised GraphSAGE train samples by protected group each epoch.",
    )
    parser.add_argument(
        "--evaluation-random-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of base random seeds for repeated GraphSAGE fairness evaluation.",
    )
    parser.add_argument(
        "--repeated-split-count",
        type=int,
        default=1,
        help="Repeated split count per evaluation seed. Effective seed = base_seed + repeat_index * 1000.",
    )
    parser.add_argument(
        "--fairness-score-alpha",
        type=float,
        default=0.25,
        help="Alpha penalty on macro_f1_gap when --early-stop-metric fairness_score is used.",
    )
    parser.add_argument(
        "--fairness-score-beta",
        type=float,
        default=0.25,
        help="Beta penalty on protected_probe_macro_f1 when --early-stop-metric fairness_score is used.",
    )
    parser.add_argument(
        "--probe-model-type",
        choices=["linear", "nonlinear"],
        default="linear",
        help="Model family used for protected-attribute and label probes.",
    )
    parser.add_argument(
        "--adversary-loss-weight",
        type=float,
        default=1.0,
        help="Loss weight for the protected-attribute adversary when --debias-mode adversarial is used.",
    )
    parser.add_argument(
        "--gradient-reversal-lambda",
        type=float,
        default=1.0,
        help="Gradient-reversal strength for adversarial GraphSAGE training.",
    )
    parser.add_argument(
        "--adversary-warmup-epochs",
        type=int,
        default=0,
        help="Linear warm-up epochs for adversarial loss and gradient-reversal pressure.",
    )
    parser.add_argument(
        "--group-robust-warmup-epochs",
        type=int,
        default=0,
        help="Linear warm-up epochs for protected-group robust weighting pressure.",
    )
    parser.add_argument(
        "--adversary-hidden-dim",
        type=int,
        default=64,
        help="Hidden width for the protected-attribute adversary head.",
    )
    parser.add_argument(
        "--adversary-num-layers",
        type=int,
        default=1,
        help="Number of layers in the protected-attribute adversary head.",
    )
    parser.add_argument(
        "--adversary-dropout",
        type=float,
        default=0.2,
        help="Dropout used inside the protected-attribute adversary head.",
    )
    parser.add_argument(
        "--graphsage-comparison-preset",
        choices=["none", "fairness_tradeoff"],
        default="none",
        help="Optional evaluation-only four-way GraphSAGE comparison preset.",
    )
    parser.add_argument(
        "--graphsage-tuning-preset",
        choices=["none", "mild_tradeoff"],
        default="none",
        help="Optional evaluation-only mild GraphSAGE tuning sweep around low adversarial and robust strengths.",
    )
    parser.add_argument(
        "--edge-dropout-probability",
        type=float,
        default=0.2,
        help="Edge dropout probability for GraphCL augmentations.",
    )
    parser.add_argument(
        "--feature-mask-probability",
        type=float,
        default=0.2,
        help="Feature masking probability for GraphCL augmentations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_all_attributes_mode(args)
    if args.graphsage_comparison_preset != "none" and _all_attributes_requested(args):
        raise ValueError("graphsage_comparison_preset is only supported for a single label column, not all-attributes mode.")
    if args.graphsage_tuning_preset != "none" and _all_attributes_requested(args):
        raise ValueError("graphsage_tuning_preset is only supported for a single label column, not all-attributes mode.")
    dataset_config = build_dataset_config(args)
    dataset = load_dataset(dataset_config)
    methods = list(resolve_method_names([str(method).strip().lower() for method in args.methods]))
    method_configs = build_method_configs(args, methods)
    output_dir = _resolve_repo_path(args.output_dir)
    normalized_evaluation_tasks = resolve_evaluation_tasks(args.evaluation_tasks)
    clustering_method_config = json.loads(args.clustering_method_config_json)
    if not isinstance(clustering_method_config, dict):
        raise ValueError("--clustering-method-config-json must parse to a JSON object.")
    args.clustering_method_config = clustering_method_config
    evaluation_kwargs = build_evaluation_kwargs(args)
    repeated_evaluation_requested = args.evaluation_random_seeds is not None or int(args.repeated_split_count) > 1

    if not _all_attributes_requested(args):
        result = run_embedding_benchmark(
            dataset,
            methods=methods,
            method_configs=method_configs,
            output_dir=output_dir,
            export_formats=args.export_formats,
            label_column=args.label_column,
            evaluation_tasks=normalized_evaluation_tasks,
            classification_label_column=args.classification_label_column,
            clustering_label_column=args.clustering_label_column,
            max_workers=args.max_workers,
            symmetrize_directed=args.symmetrize_directed,
            continue_on_error=args.continue_on_error,
            run_protected_attribute_probe=args.run_protected_attribute_probe,
            **evaluation_kwargs,
        )
        repeated_evaluation_summary_frame = None
        if (
            repeated_evaluation_requested
            and "node_classification" in normalized_evaluation_tasks
            and str(args.node_classification_model).strip().lower() == "graphsage"
        ):
            repeated_runs_frame, repeated_evaluation_summary_frame = evaluate_embedding_benchmark_repeated(
                dataset,
                result,
                evaluation_random_seeds=args.evaluation_random_seeds,
                repeated_split_count=args.repeated_split_count,
                tasks=["node_classification"],
                random_seed=args.random_seed,
                label_column=args.label_column,
                classification_label_column=args.classification_label_column,
                clustering_label_column=args.clustering_label_column,
                symmetrize_directed=args.symmetrize_directed,
                **evaluation_kwargs,
            )
            repeated_runs_path = repeated_evaluation_runs_path(result.output_dir, result.dataset_name)
            if repeated_runs_path is not None and repeated_runs_frame is not None and not repeated_runs_frame.empty:
                repeated_runs_frame.to_csv(repeated_runs_path, index=False)
            repeated_summary_path = repeated_evaluation_summary_path(result.output_dir, result.dataset_name)
            if (
                repeated_summary_path is not None
                and repeated_evaluation_summary_frame is not None
                and not repeated_evaluation_summary_frame.empty
            ):
                repeated_evaluation_summary_frame.to_csv(repeated_summary_path, index=False)
        comparison_frame = _run_graphsage_comparison_preset(
            dataset,
            result,
            preset_name=args.graphsage_comparison_preset,
            evaluation_kwargs=evaluation_kwargs,
            label_column=args.label_column,
            classification_label_column=args.classification_label_column,
            clustering_label_column=args.clustering_label_column,
            random_seed=args.random_seed,
            symmetrize_directed=args.symmetrize_directed,
        )
        _, repeated_comparison_frame = _run_graphsage_repeated_comparison_preset(
            dataset,
            result,
            preset_name=args.graphsage_comparison_preset if repeated_evaluation_requested else "none",
            evaluation_kwargs=evaluation_kwargs,
            label_column=args.label_column,
            classification_label_column=args.classification_label_column,
            clustering_label_column=args.clustering_label_column,
            random_seed=args.random_seed,
            evaluation_random_seeds=args.evaluation_random_seeds,
            repeated_split_count=args.repeated_split_count,
            symmetrize_directed=args.symmetrize_directed,
        )
        tuning_frame = _run_graphsage_tuning_preset(
            dataset,
            result,
            preset_name=args.graphsage_tuning_preset,
            evaluation_kwargs=evaluation_kwargs,
            label_column=args.label_column,
            classification_label_column=args.classification_label_column,
            clustering_label_column=args.clustering_label_column,
            random_seed=args.random_seed,
            symmetrize_directed=args.symmetrize_directed,
        )
        report_text = format_benchmark_report(
            result.summary_frame,
            evaluation_frame=result.evaluation_frame,
            repeated_evaluation_summary_frame=repeated_evaluation_summary_frame,
            comparison_frame=comparison_frame,
            repeated_comparison_frame=repeated_comparison_frame,
            tuning_frame=tuning_frame,
            dataset_name=result.dataset_name,
            output_dir=result.output_dir,
        )
        report_path = benchmark_report_path(result.output_dir, result.dataset_name)
        if report_path is None:
            print(report_text)
            return

        saved_report_text = render_saved_benchmark_summary_text(
            result,
            report_path=report_path,
            repeated_evaluation_summary_frame=repeated_evaluation_summary_frame,
            comparison_frame=comparison_frame,
            repeated_comparison_frame=repeated_comparison_frame,
            tuning_frame=tuning_frame,
        )
        save_benchmark_report(
            saved_report_text,
            output_dir=result.output_dir,
            dataset_name=result.dataset_name,
        )
        print(saved_report_text)
        return

    attribute_names = _attribute_column_names(dataset)
    if not attribute_names:
        raise ValueError(f"Dataset '{dataset.name}' has no attribute columns other than node_id.")

    attribute_specific_tasks = [task for task in normalized_evaluation_tasks if task in {"node_classification", "node_clustering"}]
    base_tasks = [task for task in normalized_evaluation_tasks if task == "link_prediction"]
    if not attribute_specific_tasks:
        raise ValueError(
            "All-attributes mode requires at least one label-dependent evaluation task such as "
            "node_classification or node_clustering."
        )

    base_result = run_embedding_benchmark(
        dataset,
        methods=methods,
        method_configs=method_configs,
        output_dir=output_dir,
        export_formats=args.export_formats,
        label_column=None,
        evaluation_tasks=base_tasks,
        classification_label_column=None,
        clustering_label_column=None,
        max_workers=args.max_workers,
        symmetrize_directed=args.symmetrize_directed,
        continue_on_error=args.continue_on_error,
        run_protected_attribute_probe=args.run_protected_attribute_probe,
        **evaluation_kwargs,
    )
    base_report_text = format_benchmark_report(
        base_result.summary_frame,
        evaluation_frame=base_result.evaluation_frame,
        dataset_name=base_result.dataset_name,
        output_dir=base_result.output_dir,
    )
    base_report_path = benchmark_report_path(base_result.output_dir, base_result.dataset_name)
    if base_report_path is not None:
        save_benchmark_report(
            render_saved_benchmark_summary_text(base_result, report_path=base_report_path),
            output_dir=base_result.output_dir,
            dataset_name=base_result.dataset_name,
        )

    attribute_reports: list[dict[str, Any]] = []
    for attribute_name in attribute_names:
        evaluation_frame = evaluate_embedding_benchmark(
            dataset,
            base_result,
            tasks=attribute_specific_tasks,
            random_seed=args.random_seed,
            label_column=attribute_name,
            classification_label_column=attribute_name,
            clustering_label_column=attribute_name,
            symmetrize_directed=args.symmetrize_directed,
            **evaluation_kwargs,
        )
        report_text = format_attribute_benchmark_report(
            base_result.summary_frame,
            evaluation_frame=evaluation_frame,
            dataset_name=base_result.dataset_name,
            attribute_name=attribute_name,
            output_dir=base_result.output_dir,
        )
        report_path = save_attribute_benchmark_report(
            report_text,
            output_dir=base_result.output_dir,
            dataset_name=base_result.dataset_name,
            attribute_name=attribute_name,
        )
        attribute_reports.append(
            {
                "attribute_name": attribute_name,
                "report_text": report_text,
                "report_path": report_path,
                "evaluation_frame": evaluation_frame,
            }
        )

    combined_report_text = format_all_attributes_benchmark_report(
        base_result.dataset_name,
        base_report_text=base_report_text,
        attribute_reports=attribute_reports,
        output_dir=base_result.output_dir,
    )
    combined_report_path = save_all_attributes_benchmark_report(
        combined_report_text,
        output_dir=base_result.output_dir,
        dataset_name=base_result.dataset_name,
    )

    print(f"Saved base benchmark report to {base_report_path if base_report_path is not None else '-'}")
    print(f"Saved all-attributes report to {combined_report_path if combined_report_path is not None else '-'}")
    print(f"Attributes evaluated: {', '.join(attribute_names)}")
    for report in attribute_reports:
        print(f"{report['attribute_name']}: report={report['report_path'] if report['report_path'] is not None else '-'}")


if __name__ == "__main__":
    main()
