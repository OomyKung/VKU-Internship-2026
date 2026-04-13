"""Configuration models used by the staged FIM rebuild."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DatasetConfig:
    """Describe how a graph dataset should be located and parsed."""

    name: str
    edge_path: Path | None = None
    pickle_path: Path | None = None
    attribute_path: Path | None = None
    dataset_format: str | None = None
    directed: bool = True
    delimiter: str | None = None
    source_column: str = "source"
    target_column: str = "target"
    node_id_column: str = "node_id"
