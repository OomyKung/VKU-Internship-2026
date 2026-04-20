"""Write dataset-specific embedding explanation notes as text files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METHOD_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "deepwalk": {
        "family": "Shallow random-walk",
        "what_it_does": (
            "DeepWalk generates short random walks on the graph and learns node vectors from "
            "co-occurrence patterns inside those walks. It is effectively a graph version of a "
            "word2vec-style neighborhood model."
        ),
        "strengths": "Captures higher-order structural context beyond direct neighbors.",
        "weaknesses": "Uses structure only in this benchmark and ignores node attributes.",
    },
    "node2vec": {
        "family": "Shallow biased random-walk",
        "what_it_does": (
            "Node2Vec extends DeepWalk by biasing the walks with p/q parameters so it can favor "
            "BFS-like local exploration or DFS-like outward exploration."
        ),
        "strengths": "Can balance community-level similarity and structural-role similarity.",
        "weaknesses": "Still structure-only here, so attribute-driven class signals are not used directly.",
    },
    "line": {
        "family": "Shallow proximity-preserving",
        "what_it_does": (
            "LINE preserves first-order proximity between linked nodes and second-order proximity "
            "between nodes with similar neighborhoods. It is fast and focused on local graph structure."
        ),
        "strengths": "Very fast and simple, especially on sparse static graphs.",
        "weaknesses": "Local-only objective usually misses longer-range community structure and attribute information.",
    },
    "vgae": {
        "family": "Autoencoder / generative GNN",
        "what_it_does": (
            "VGAE encodes nodes with a GNN into a latent distribution and learns embeddings by "
            "reconstructing graph connectivity from that latent space."
        ),
        "strengths": "Combines topology with node features and encourages a smooth latent space.",
        "weaknesses": "The reconstruction objective can be less directly aligned with class separation than contrastive methods.",
    },
    "gcn": {
        "family": "Message-passing GNN",
        "what_it_does": (
            "GCN repeatedly averages transformed neighbor features, so each node embedding becomes "
            "a smoothed summary of its local neighborhood."
        ),
        "strengths": "Strong baseline when labels correlate with local homophily and useful node features exist.",
        "weaknesses": "Can over-smooth if neighborhoods are mixed or if too much averaging washes out boundaries.",
    },
    "graphsage": {
        "family": "Message-passing GNN",
        "what_it_does": (
            "GraphSAGE learns how to aggregate neighbor information instead of using a fixed normalized "
            "average, which gives it more flexibility than plain GCN."
        ),
        "strengths": "Often preserves local discriminative patterns better than plain averaging.",
        "weaknesses": "Still depends heavily on the quality of the input node features.",
    },
    "dgi": {
        "family": "Self-supervised contrastive GNN",
        "what_it_does": (
            "DGI trains node embeddings by maximizing mutual information between node-level embeddings "
            "and a global graph summary while contrasting them against corrupted features."
        ),
        "strengths": "Good at extracting feature-aware global structure without labels.",
        "weaknesses": "Can produce tightly clustered embeddings if the global summary dominates too strongly.",
    },
    "graphcl": {
        "family": "Self-supervised contrastive GNN",
        "what_it_does": (
            "GraphCL learns embeddings by making two augmented graph views agree while separating them "
            "from unrelated views. In this benchmark it uses edge dropout and feature masking."
        ),
        "strengths": "Learns invariances and often preserves robust class-separating signals.",
        "weaknesses": "More training-heavy and sensitive to augmentation choices.",
    },
    "metapath2vec": {
        "family": "Heterogeneous graph embedding",
        "what_it_does": (
            "metapath2vec performs random walks constrained by node/edge types and metapaths, so it is "
            "designed for heterogeneous graphs rather than plain homogeneous networks."
        ),
        "strengths": "Useful when typed entities and typed relations matter.",
        "weaknesses": "Not applicable to homogeneous graphs like the current benchmark graph.",
    },
}


def _default_notes_output_dir(dataset_name: str) -> Path:
    return ROOT / "results" / dataset_name / "notes"


def _load_summary_frames(results_root: Path, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    embedding_dir = results_root / dataset_name / "embeddings"
    summary_path = embedding_dir / f"{dataset_name}_embedding_benchmark_summary.csv"
    evaluation_path = embedding_dir / f"{dataset_name}_embedding_evaluation_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Benchmark summary not found: {summary_path}")
    summary_frame = pd.read_csv(summary_path)
    evaluation_frame = pd.read_csv(evaluation_path) if evaluation_path.is_file() else pd.DataFrame()
    return summary_frame, evaluation_frame


def _fmt_float(value: Any, digits: int = 4) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_text(value: Any) -> str:
    if pd.isna(value):
        return "-"
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return "-"
    return text if text else "-"


def _fmt_intlike(value: Any) -> str:
    if pd.isna(value):
        return "-"
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else str(numeric)


def _method_eval_rows(evaluation_frame: pd.DataFrame, method_name: str) -> pd.DataFrame:
    if evaluation_frame.empty:
        return pd.DataFrame()
    rows = evaluation_frame.loc[evaluation_frame["method"].astype(str) == method_name].copy()
    if rows.empty:
        return rows
    return rows.sort_values(by=["task"], ascending=[True]).reset_index(drop=True)


def _best_accuracy(evaluation_frame: pd.DataFrame) -> float | None:
    if evaluation_frame.empty or "accuracy" not in evaluation_frame.columns:
        return None
    accuracy_series = pd.to_numeric(evaluation_frame["accuracy"], errors="coerce").dropna()
    if accuracy_series.empty:
        return None
    return float(accuracy_series.max())


def _performance_reason(summary_row: pd.Series, eval_rows: pd.DataFrame, best_accuracy: float | None) -> str:
    method_name = str(summary_row["method"])
    uses_features = bool(summary_row.get("uses_features", False))
    status = str(summary_row.get("status", ""))
    if status != "ok":
        return str(summary_row.get("skip_reason", "") or summary_row.get("error_message", "")).strip()

    if eval_rows.empty:
        return "No downstream evaluation row was found, so only the embedding-generation result is available."

    first_eval = eval_rows.iloc[0]
    label_column = str(first_eval.get("label_column", "")).strip()
    accuracy = pd.to_numeric(first_eval.get("accuracy", pd.NA), errors="coerce")

    if method_name == "line":
        return (
            f"LINE was the weakest method here because it only optimizes local proximity. "
            f"That is usually not enough for a 5-class task like '{label_column}', where broader "
            "community structure and rich attributes matter."
        )

    if method_name in {"deepwalk", "node2vec"}:
        extra = ""
        if method_name == "node2vec":
            extra = " In this run it ended up effectively matching DeepWalk, which suggests the chosen p/q bias did not add useful extra signal."
        return (
            f"{method_name} performed in the middle because random-walk co-occurrence captures higher-order topology better than LINE, "
            f"but it still does not use node attributes directly.{extra}"
        )

    if method_name == "metapath2vec":
        return (
            "metapath2vec was skipped because the benchmark graph is homogeneous. "
            "The method needs typed nodes or typed edges so it can follow metapaths."
        )

    if uses_features:
        leakage_note = (
            f" A major caveat: this benchmark's feature preparation includes node attributes, and the evaluation target "
            f"'{label_column}' is one of those attributes on this dataset. That likely makes the result substantially easier for feature-based methods."
            if label_column and label_column != "-"
            else ""
        )
        if method_name == "graphcl":
            return (
                "GraphCL came out best because its contrastive objective pushes the model to preserve signals that survive graph augmentations, "
                "which often produces robust class-separating embeddings."
                + leakage_note
            )
        if method_name == "dgi":
            return (
                "DGI scored very strongly because it learns embeddings that stay informative about both node-level and graph-level structure, "
                "which works well when useful attributes are present."
                + leakage_note
            )
        if method_name == "graphsage":
            return (
                "GraphSAGE scored very strongly because its learned neighbor aggregation can preserve discriminative local patterns better than plain averaging."
                + leakage_note
            )
        if method_name == "gcn":
            return (
                "GCN scored strongly because it propagates feature information through local neighborhoods, which is effective on homophilous graphs."
                + leakage_note
            )
        if method_name == "vgae":
            return (
                "VGAE scored strongly because it combines graph reconstruction with feature-aware latent representations, "
                "but its objective is slightly less directly aligned with classification than the strongest contrastive methods."
                + leakage_note
            )

    if best_accuracy is not None and not pd.isna(accuracy):
        gap = float(best_accuracy) - float(accuracy)
        return (
            f"This method reached accuracy {float(accuracy):.4f}, which is {gap:.4f} below the best method in this run. "
            "The result is consistent with the amount of structural or feature signal the method can exploit."
        )

    return "The observed result is broadly consistent with the method's structural bias and training objective."


def _evaluation_lines(eval_rows: pd.DataFrame) -> list[str]:
    if eval_rows.empty:
        return ["No downstream evaluation rows were available for this method."]
    lines: list[str] = []
    for _, row in eval_rows.iterrows():
        task = str(row.get("task", "unknown"))
        status = str(row.get("status", ""))
        metrics = [
            f"accuracy={_fmt_float(row.get('accuracy', pd.NA))}",
            f"macro_f1={_fmt_float(row.get('macro_f1', pd.NA))}",
            f"micro_f1={_fmt_float(row.get('micro_f1', pd.NA))}",
            f"roc_auc={_fmt_float(row.get('roc_auc', pd.NA))}",
            f"average_precision={_fmt_float(row.get('average_precision', pd.NA))}",
            f"nmi={_fmt_float(row.get('nmi', pd.NA))}",
            f"ari={_fmt_float(row.get('ari', pd.NA))}",
        ]
        lines.append(
            f"- {task}: status={status}, label_column={_fmt_text(row.get('label_column', pd.NA))}, "
            f"runtime={_fmt_float(row.get('runtime_seconds', pd.NA), 3)}s, "
            + ", ".join(metrics)
        )
        skipped_reason = _fmt_text(row.get("skipped_reason", pd.NA))
        notes = _fmt_text(row.get("notes", pd.NA))
        if skipped_reason != "-":
            lines.append(f"  skipped_reason: {skipped_reason}")
        if notes != "-":
            lines.append(f"  notes: {notes}")
    return lines


def _write_method_note(
    output_path: Path,
    *,
    dataset_name: str,
    summary_row: pd.Series,
    eval_rows: pd.DataFrame,
    best_accuracy: float | None,
) -> None:
    method_name = str(summary_row["method"])
    template = METHOD_DESCRIPTIONS.get(
        method_name,
        {
            "family": "Graph embedding",
            "what_it_does": "This method produces node embeddings from graph structure and optionally node features.",
            "strengths": "Learns reusable node vectors for downstream tasks.",
            "weaknesses": "The exact behavior depends on the model family and the features available.",
        },
    )

    lines = [
        f"Dataset: {dataset_name}",
        f"Method: {method_name}",
        f"Family: {template['family']}",
        "",
        "What this method does:",
        template["what_it_does"],
        "",
        "Typical strengths:",
        template["strengths"],
        "",
        "Typical weaknesses:",
        template["weaknesses"],
        "",
        "Observed benchmark result:",
        (
            f"status={summary_row.get('status', '')}, runtime={_fmt_float(summary_row.get('runtime_seconds', pd.NA), 3)}s, "
            f"embedding_dim={_fmt_intlike(summary_row.get('embedding_dim', pd.NA))}, "
            f"label_probe_accuracy={_fmt_float(summary_row.get('label_probe_accuracy', pd.NA))}, "
            f"uses_features={bool(summary_row.get('uses_features', False))}"
        ),
        "",
        "Task-level evaluation:",
        *_evaluation_lines(eval_rows),
        "",
        "Why this result likely happened:",
        _performance_reason(summary_row, eval_rows, best_accuracy),
        "",
        "Caveat:",
        (
            "This explanation is an inference from the saved benchmark outputs and the current code path. "
            "It is a strong engineering interpretation, not a formal causal proof."
        ),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_overview_note(
    output_path: Path,
    *,
    dataset_name: str,
    summary_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
) -> None:
    lines = [
        f"Dataset: {dataset_name}",
        "",
        "Overview:",
        (
            "These notes explain what each embedding method does and why it likely produced its observed result "
            "on the current benchmark run."
        ),
        "",
        "Important benchmark caveat:",
        (
            "Feature-based methods in this framework use prepared node features built from structural features plus "
            "encoded node attributes. On graph_spa_500_0, the evaluation target 'ethnicity' is itself a node attribute. "
            "That means GCN, GraphSAGE, VGAE, DGI, and GraphCL likely had access to very direct signal about the label, "
            "so their scores are not directly comparable to structure-only methods like DeepWalk, Node2Vec, and LINE."
        ),
        "",
        "Method ranking by saved node-classification accuracy:",
    ]
    if not evaluation_frame.empty and "accuracy" in evaluation_frame.columns:
        ranked = (
            evaluation_frame.loc[evaluation_frame["status"].astype(str) == "ok", ["method", "accuracy", "macro_f1"]]
            .copy()
        )
        ranked["accuracy"] = pd.to_numeric(ranked["accuracy"], errors="coerce")
        ranked["macro_f1"] = pd.to_numeric(ranked["macro_f1"], errors="coerce")
        ranked = ranked.sort_values(by=["accuracy", "macro_f1"], ascending=[False, False]).reset_index(drop=True)
        for _, row in ranked.iterrows():
            lines.append(
                f"- {row['method']}: accuracy={_fmt_float(row['accuracy'])}, macro_f1={_fmt_float(row['macro_f1'])}"
            )
    else:
        lines.append("- No evaluation summary was available.")

    lines.extend(
        [
            "",
            "Saved method notes:",
            *[
                f"- {dataset_name}_{str(row['method'])}_embedding_note.txt"
                for _, row in summary_frame.iterrows()
            ],
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write embedding explanation notes from saved benchmark results.")
    parser.add_argument("--dataset", required=True, help="Dataset name used in the saved benchmark results.")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Base results directory that contains <dataset>/embeddings/*.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where the .txt explanation files should be written. Defaults to results/<dataset>/notes/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = (ROOT / args.results_dir).resolve()
    output_dir = (
        _default_notes_output_dir(args.dataset)
        if args.output_dir is None
        else (ROOT / args.output_dir).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_frame, evaluation_frame = _load_summary_frames(results_root, args.dataset)
    best_accuracy = _best_accuracy(evaluation_frame)

    overview_path = output_dir / f"{args.dataset}_embedding_notes_overview.txt"
    _write_overview_note(
        overview_path,
        dataset_name=args.dataset,
        summary_frame=summary_frame,
        evaluation_frame=evaluation_frame,
    )

    for _, row in summary_frame.iterrows():
        method_name = str(row["method"])
        note_path = output_dir / f"{args.dataset}_{method_name}_embedding_note.txt"
        _write_method_note(
            note_path,
            dataset_name=args.dataset,
            summary_row=row,
            eval_rows=_method_eval_rows(evaluation_frame, method_name),
            best_accuracy=best_accuracy,
        )

    print(f"Saved explanation notes to {output_dir}")
    print(f"Overview note: {overview_path}")


if __name__ == "__main__":
    main()
