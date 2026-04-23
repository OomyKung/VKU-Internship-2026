"""CLI for the shared clustering and community-detection benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.clustering import (  # noqa: E402
    available_clustering_methods,
    run_clustering_benchmark,
)
from fim_hybrid.data_loader import load_dataset, resolve_dataset_config  # noqa: E402
from fim_hybrid.embeddings.features import prepare_benchmark_features  # noqa: E402


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


def _format_metric(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def _format_runtime(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.3f}s"


def _format_int(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return str(int(value))


def _format_text(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    text = str(value).strip()
    return text if text else "-"


def build_dataset_config(args: argparse.Namespace):
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


def clustering_report_path(output_dir: Path | None, dataset_name: str) -> Path | None:
    if output_dir is None:
        return None
    report_dir = output_dir / dataset_name / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"{dataset_name}_clustering_benchmark_report.txt"


def save_clustering_report(report_text: str, *, output_dir: Path | None, dataset_name: str) -> Path | None:
    report_path = clustering_report_path(output_dir, dataset_name)
    if report_path is None:
        return None
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def _load_embeddings_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Embedding CSV file not found: {path}")
    frame = pd.read_csv(path)
    if "node_id" not in frame.columns:
        raise ValueError(f"Embedding CSV '{path}' must contain a node_id column.")
    return frame


def _labels_from_dataset(dataset, label_column: str | None) -> pd.Series | None:
    if label_column is None:
        return None
    if label_column not in dataset.node_attributes.columns:
        raise ValueError(
            f"Label column '{label_column}' is not present in dataset.node_attributes."
        )
    return dataset.node_attributes.set_index("node_id", drop=False)[label_column].astype(str)


def _build_method_configs(
    methods: list[str],
    *,
    config_payload: dict[str, Any],
    n_clusters: int | None,
    min_cluster_size: int | None,
) -> dict[str, dict[str, object]]:
    global_config: dict[str, object] = {}
    per_method_config: dict[str, dict[str, object]] = {}
    for key, value in config_payload.items():
        if isinstance(value, dict):
            per_method_config[str(key).strip().lower()] = dict(value)
        else:
            global_config[str(key)] = value

    configs: dict[str, dict[str, object]] = {}
    for method_name in methods:
        config = dict(global_config)
        config.update(per_method_config.get(method_name, {}))
        if n_clusters is not None:
            config.setdefault("n_clusters", int(n_clusters))
        if min_cluster_size is not None:
            config.setdefault("min_cluster_size", int(min_cluster_size))
        configs[method_name] = config
    return configs


def format_clustering_report(summary_frame: pd.DataFrame, *, dataset_name: str) -> str:
    if summary_frame.empty:
        return "No clustering benchmark rows were produced."

    lines = [
        _rule("="),
        "Clustering Benchmark Summary",
        _rule("="),
        f"Dataset: {dataset_name}",
        f"Method status counts: {summary_frame['status'].value_counts(dropna=False).to_dict()}",
        "",
        _rule("-"),
        "Clustering Comparison",
        _rule("-"),
    ]
    ordered = summary_frame.copy()
    ordered["_status_rank"] = ordered["status"].map({"ok": 0, "skipped": 1}).fillna(2)
    ordered["_clusters_sort"] = pd.to_numeric(ordered["num_clusters"], errors="coerce").fillna(-1.0)
    ordered = ordered.sort_values(
        ["_status_rank", "category", "runtime_seconds", "_clusters_sort", "method"],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)
    for index, row in ordered.iterrows():
        lines.extend(
            [
                f"{index + 1}. {row['method']} | {row['category']} [{row['status']}]",
                (
                    f"   input_mode={_format_text(row.get('resolved_input_mode', pd.NA))} "
                    f"(requested={_format_text(row.get('requested_input_mode', pd.NA))}) | "
                    f"runtime={_format_runtime(row.get('runtime_seconds', pd.NA))} | "
                    f"clusters={_format_int(row.get('num_clusters', pd.NA))}"
                ),
                (
                    f"   cluster_sizes={_format_text(row.get('cluster_size_summary', pd.NA))} | "
                    f"largest={_format_int(row.get('largest_cluster_size', pd.NA))} | "
                    f"smallest={_format_int(row.get('smallest_cluster_size', pd.NA))} | "
                    f"mean={_format_metric(row.get('average_cluster_size', pd.NA))}"
                ),
                (
                    f"   metrics: modularity={_format_metric(row.get('modularity', pd.NA))} | "
                    f"conductance={_format_metric(row.get('mean_conductance', pd.NA))} | "
                    f"silhouette={_format_metric(row.get('silhouette_score', pd.NA))} | "
                    f"davies_bouldin={_format_metric(row.get('davies_bouldin_score', pd.NA))} | "
                    f"calinski_harabasz={_format_metric(row.get('calinski_harabasz_score', pd.NA))} | "
                    f"nmi={_format_metric(row.get('nmi', pd.NA))} | "
                    f"ari={_format_metric(row.get('ari', pd.NA))}"
                ),
            ]
        )
        if _format_text(row.get("assignments_csv_path", pd.NA)) != "-":
            lines.append(
                "   outputs: "
                f"assignments={_format_text(row.get('assignments_csv_path', pd.NA))} | "
                f"cluster_sizes={_format_text(row.get('cluster_sizes_csv_path', pd.NA))}"
            )
        if _format_text(row.get("skip_reason", pd.NA)) != "-":
            lines.append(f"   skip_reason={row['skip_reason']}")
        if index < len(ordered) - 1:
            lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a clustering/community-detection benchmark.")
    parser.add_argument("--dataset", required=True, help="Built-in dataset name or custom graph path stem.")
    parser.add_argument("--graph-path", default=None, help="Optional explicit graph path.")
    parser.add_argument("--attributes-path", default=None, help="Optional attribute CSV path.")
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
        help="Treat external edge-list datasets as directed. Pickled graphs keep their stored graph type.",
    )
    parser.add_argument("--source-col", default=None, help="Source column name for CSV edge lists.")
    parser.add_argument("--target-col", default=None, help="Target column name for CSV edge lists.")
    parser.add_argument("--node-id-col", default=None, help="Node ID column name for separate attribute files.")
    parser.add_argument(
        "--clustering-methods",
        nargs="+",
        default=list(available_clustering_methods()),
        help="Clustering/community methods to compare.",
    )
    parser.add_argument(
        "--clustering-input-mode",
        choices=["graph", "embedding", "auto"],
        default="auto",
        help="Requested input mode. Graph-native methods always use graph input directly.",
    )
    parser.add_argument(
        "--embedding-source",
        choices=["auto", "csv", "feature"],
        default="auto",
        help="Embedding-space input source. 'auto' prefers --embedding-csv and otherwise falls back to prepared features.",
    )
    parser.add_argument("--embedding-csv", default=None, help="Optional embedding CSV for embedding-space clustering.")
    parser.add_argument("--label-column", default=None, help="Optional ground-truth label column for NMI/ARI.")
    parser.add_argument("--n-clusters", type=int, default=None, help="Optional fixed cluster count for fixed-k methods.")
    parser.add_argument("--min-cluster-size", type=int, default=None, help="Optional minimum cluster size for density-based methods.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", default="results", help="Directory for CSV outputs.")
    parser.add_argument(
        "--save-cluster-assignments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist per-method node->cluster assignments and cluster-size CSVs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip methods that fail cleanly instead of aborting the whole benchmark.",
    )
    parser.add_argument(
        "--clustering-method-config-json",
        default="{}",
        help="JSON object with additional clustering config. Nested method-name objects override shared values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = build_dataset_config(args)
    dataset = load_dataset(dataset_config)
    output_dir = _resolve_repo_path(args.output_dir)
    embedding_csv_path = _resolve_repo_path(args.embedding_csv)
    embeddings = _load_embeddings_csv(embedding_csv_path)
    labels = _labels_from_dataset(dataset, args.label_column)
    config_payload = json.loads(args.clustering_method_config_json)
    if not isinstance(config_payload, dict):
        raise ValueError("--clustering-method-config-json must parse to a JSON object.")

    methods = [str(method).strip().lower() for method in args.clustering_methods]
    method_configs = _build_method_configs(
        methods,
        config_payload=config_payload,
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size,
    )
    features = None
    if args.embedding_source == "feature":
        features = prepare_benchmark_features(dataset, graph=dataset.graph, random_seed=args.random_seed).feature_frame

    result = run_clustering_benchmark(
        dataset,
        methods=methods,
        embeddings=embeddings,
        features=features,
        labels=labels,
        output_dir=output_dir,
        input_mode=args.clustering_input_mode,
        method_configs=method_configs,
        random_seed=args.random_seed,
        continue_on_error=args.continue_on_error,
        save_cluster_assignments=args.save_cluster_assignments,
    )
    report_text = format_clustering_report(result.summary_frame, dataset_name=result.dataset_name)
    report_path = save_clustering_report(
        report_text,
        output_dir=result.output_dir,
        dataset_name=result.dataset_name,
    )
    if report_path is not None:
        print(f"Saved clustering report to {report_path}")
    print(report_text)


if __name__ == "__main__":
    main()
