"""Compatibility helpers for dataset resolution.

The legacy experiment runner was retired with the Swarm Intelligence removal.
Use scripts/run_fim_stack.py for the active FIM framework.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fim_hybrid.config import DatasetConfig  # noqa: E402
from fim_hybrid.data_loader import resolve_dataset_config  # noqa: E402
from fim_hybrid.removed_swarm import REMOVED_SWARM_FEATURE_MESSAGE  # noqa: E402


def build_dataset_config(args: argparse.Namespace) -> DatasetConfig:
    """Resolve CLI dataset arguments into the shared internal dataset config."""

    dataset_format = None if getattr(args, "dataset_format", None) in {None, "auto"} else args.dataset_format
    return resolve_dataset_config(
        getattr(args, "dataset", None),
        base_dir=ROOT,
        graph_path=getattr(args, "graph_path", None),
        attributes_path=getattr(args, "attributes_path", None),
        dataset_format=dataset_format,
        dataset_config=getattr(args, "dataset_config", None),
        directed=getattr(args, "directed", None),
        delimiter=getattr(args, "delimiter", None),
        source_col=getattr(args, "source_col", None),
        target_col=getattr(args, "target_col", None),
        node_id_col=getattr(args, "node_id_col", None),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--graph-path")
    parser.add_argument("--attributes-path")
    parser.add_argument("--dataset-format", default="auto")
    parser.add_argument("--dataset-config")
    parser.add_argument("--directed", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--delimiter")
    parser.add_argument("--source-col")
    parser.add_argument("--target-col")
    parser.add_argument("--node-id-col")
    parser.parse_args()
    parser.error(REMOVED_SWARM_FEATURE_MESSAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
