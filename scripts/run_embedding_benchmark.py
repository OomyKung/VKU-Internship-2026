"""CLI for the modular graph embedding benchmark framework."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.data_loader import load_dataset, resolve_dataset_config  # noqa: E402
from fim_hybrid.embeddings import (  # noqa: E402
    available_embedding_methods,
    available_evaluation_tasks,
    resolve_method_names,
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


def _format_metric(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def format_benchmark_report(
    summary_frame: pd.DataFrame,
    *,
    evaluation_frame: pd.DataFrame | None = None,
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
        lines.extend(
            [
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
                    f"   label_probe={row['label_probe_status']} | "
                    f"accuracy={row['label_probe_accuracy'] if not pd.isna(row['label_probe_accuracy']) else '-'}"
                ),
            ]
        )
        if not pd.isna(row["csv_path"]) or not pd.isna(row["pickle_path"]) or not pd.isna(row["npy_path"]):
            lines.append(
                "   "
                f"exports: csv={_display_path(row['csv_path'])} | "
                f"pickle={_display_path(row['pickle_path'])} | "
                f"npy={_display_path(row['npy_path'])}"
            )
        if str(row.get("skip_reason", "")).strip():
            lines.append(f"   skip_reason={row['skip_reason']}")
        if str(row.get("error_message", "")).strip():
            lines.append(f"   error={row['error_message']}")

    if evaluation_frame is not None and not evaluation_frame.empty:
        lines.extend(
            [
                "",
                _rule("-"),
                "Graph Embedding Evaluation Summary",
                _rule("-"),
            ]
        )
        ordered_evaluation = evaluation_frame.sort_values(
            by=["method", "task", "status"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
        for index, row in ordered_evaluation.iterrows():
            metric_parts = [
                f"accuracy={_format_metric(row['accuracy'])}",
                f"macro_f1={_format_metric(row['macro_f1'])}",
                f"micro_f1={_format_metric(row['micro_f1'])}",
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
                    f"   metrics: {', '.join(metric_parts)}",
                ]
            )
            if not pd.isna(row.get("edge_feature", pd.NA)):
                lines.append(f"   edge_feature={row['edge_feature']}")
            if str(row.get("notes", "")).strip():
                lines.append(f"   notes={row['notes']}")
            if str(row.get("skipped_reason", "")).strip():
                lines.append(f"   skipped_reason={row['skipped_reason']}")

    return "\n".join(lines)


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
        help="Fallback label column for the simple probe and downstream evaluation tasks.",
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
    dataset_config = build_dataset_config(args)
    dataset = load_dataset(dataset_config)
    methods = list(resolve_method_names([str(method).strip().lower() for method in args.methods]))
    method_configs = build_method_configs(args, methods)
    output_dir = _resolve_repo_path(args.output_dir)
    result = run_embedding_benchmark(
        dataset,
        methods=methods,
        method_configs=method_configs,
        output_dir=output_dir,
        export_formats=args.export_formats,
        label_column=args.label_column,
        evaluation_tasks=args.evaluation_tasks,
        classification_label_column=args.classification_label_column,
        clustering_label_column=args.clustering_label_column,
        node_test_fraction=args.node_test_fraction,
        link_test_fraction=args.link_test_fraction,
        link_negative_ratio=args.link_negative_ratio,
        link_prediction_edge_feature=args.link_prediction_edge_feature,
        max_workers=args.max_workers,
        symmetrize_directed=args.symmetrize_directed,
        continue_on_error=args.continue_on_error,
    )
    print(
        format_benchmark_report(
            result.summary_frame,
            evaluation_frame=result.evaluation_frame,
            dataset_name=result.dataset_name,
            output_dir=result.output_dir,
        )
    )


if __name__ == "__main__":
    main()
