"""CLI for Phase 1 dataset and protected-group verification."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.config import DatasetConfig  # noqa: E402
from fim_hybrid.data_loader import (  # noqa: E402
    format_phase1_report,
    resolve_builtin_dataset,
    verify_dataset_phase1,
)


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 dataset verification.")
    parser.add_argument("--dataset", default="graph_spa_500_0", help="Built-in dataset name.")
    parser.add_argument("--protected-attribute", required=True, help="Protected attribute to verify.")
    parser.add_argument("--edge-path", default=None, help="Path to a simple edge-list file.")
    parser.add_argument("--pickle-path", default=None, help="Path to a pickled NetworkX graph.")
    parser.add_argument("--attribute-path", default=None, help="Path to a node-attribute CSV file.")
    parser.add_argument("--delimiter", default=None, help="Optional edge-list delimiter override.")
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="Load explicit edge lists as undirected graphs.",
    )
    parser.add_argument(
        "--node-id-column",
        default="node_id",
        help="Node id column name in the attribute CSV.",
    )
    return parser.parse_args()


def build_dataset_config(args: argparse.Namespace) -> DatasetConfig:
    if args.edge_path or args.pickle_path:
        return DatasetConfig(
            name=args.dataset,
            edge_path=_resolve_repo_path(args.edge_path),
            pickle_path=_resolve_repo_path(args.pickle_path),
            attribute_path=_resolve_repo_path(args.attribute_path),
            directed=not args.undirected,
            delimiter=args.delimiter,
            node_id_column=args.node_id_column,
        )

    config = resolve_builtin_dataset(args.dataset, ROOT)
    if args.attribute_path is not None:
        config.attribute_path = _resolve_repo_path(args.attribute_path)
        config.node_id_column = args.node_id_column
    return config


def main() -> None:
    args = parse_args()
    config = build_dataset_config(args)
    report = verify_dataset_phase1(config, args.protected_attribute)
    print(format_phase1_report(report))


if __name__ == "__main__":
    main()
