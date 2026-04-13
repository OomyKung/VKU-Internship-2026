"""Phase 1 graph loading and protected-group verification."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from typing import Any
import pickle
import warnings

import networkx as nx
from numpy.exceptions import VisibleDeprecationWarning
import pandas as pd

from .config import DatasetConfig


@dataclass(slots=True)
class LoadedDataset:
    """Loaded graph plus the node-attribute table derived from it."""

    name: str
    graph: nx.Graph
    node_attributes: pd.DataFrame


@dataclass(slots=True)
class ProtectedGroupReport:
    """Validated protected-group summary for one dataset."""

    dataset_name: str
    protected_attribute: str
    node_count: int
    edge_count: int
    is_directed: bool
    sample_node_attributes: list[dict[str, Any]]
    protected_groups: dict[str, tuple[Any, ...]]
    group_sizes: dict[str, int]


_BUILTIN_DATASETS: dict[str, dict[str, str]] = {
    "email_eu_core": {
        "edge": "networks/email_Eu_core.txt",
        "attribute": "networks/email_Eu_core_department_labels.csv",
    },
    "twitter": {"pickle": "networks/twitter.pickle", "edge": "networks/twitter.txt"},
    "synth2": {"pickle": "networks/synth2.pickle", "edge": "networks/synth2.txt"},
    "synth3": {"pickle": "networks/synth3.pickle", "edge": "networks/synth3.txt"},
    "rice-facebook": {"pickle": "networks/rice_subset.pickle", "edge": "networks/rice_subset.txt"},
    "rice_subset": {"pickle": "networks/rice_subset.pickle", "edge": "networks/rice_subset.txt"},
    "graph_spa_500_0": {
        "pickle": "networks/graph_spa_500_0.pickle",
        "edge": "networks/graph_spa_500_0.txt",
    },
}

_DATASET_FORMAT_ALIASES = {
    "auto": None,
    "pickle": "pickle",
    "pkl": "pickle",
    "txt": "edge_list",
    "edgelist": "edge_list",
    "csv": "csv",
}


def builtin_dataset_exists(name: str) -> bool:
    """Return whether a built-in dataset name is registered."""

    return name.lower() in _BUILTIN_DATASETS


def resolve_builtin_dataset(name: str, base_dir: Path | str = ".") -> DatasetConfig:
    """Resolve a built-in dataset name into local file paths."""

    key = name.lower()
    if key not in _BUILTIN_DATASETS:
        raise FileNotFoundError(f"Unknown built-in dataset '{name}'.")

    base_dir = Path(base_dir)
    paths = _BUILTIN_DATASETS[key]
    return DatasetConfig(
        name=name,
        edge_path=base_dir / paths["edge"] if "edge" in paths else None,
        pickle_path=base_dir / paths["pickle"] if "pickle" in paths else None,
        attribute_path=base_dir / paths["attribute"] if "attribute" in paths else None,
        directed=True,
    )


def _normalize_dataset_format(dataset_format: str | None) -> str | None:
    if dataset_format is None:
        return None
    key = str(dataset_format).strip().lower()
    if key in {"pickle", "edge_list", "csv"}:
        return key
    if key not in _DATASET_FORMAT_ALIASES:
        supported = ", ".join(sorted(alias for alias in _DATASET_FORMAT_ALIASES if alias != "auto"))
        raise ValueError(f"Unsupported dataset format '{dataset_format}'. Supported values: auto, {supported}.")
    return _DATASET_FORMAT_ALIASES[key]


def _path_suffixes(path: Path) -> tuple[str, ...]:
    return tuple(suffix.lower() for suffix in path.suffixes)


def _infer_dataset_format_from_path(path: Path) -> str:
    suffixes = _path_suffixes(path)
    if suffixes[-1:] in [(".pickle",), (".pkl",)]:
        return "pickle"
    if suffixes[-1:] == (".csv",):
        return "csv"
    if suffixes[-2:] == (".csv", ".gz"):
        return "csv"
    if suffixes[-1:] == (".txt",):
        return "edge_list"
    if suffixes[-2:] == (".txt", ".gz"):
        return "edge_list"
    raise ValueError(
        f"Unsupported dataset format for '{path}'. "
        "Supported graph formats: .pickle/.pkl, .txt/.txt.gz edge lists, .csv edge lists."
    )


def _default_dataset_name_from_path(path: Path) -> str:
    suffixes = _path_suffixes(path)
    if suffixes[-2:] in {( ".txt", ".gz"), (".csv", ".gz")}:
        return path.name[: -len("".join(suffixes[-2:]))]
    if suffixes:
        return path.name[: -len(suffixes[-1])]
    return path.name


def _resolve_input_path(path_value: str | Path, base_dir: Path | str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def _dataset_candidates_from_name(name: str, base_dir: Path | str) -> list[Path]:
    base_path = Path(base_dir)
    relative_candidate = _resolve_input_path(name, base_path)
    candidates = [relative_candidate]
    networks_dir = base_path / "networks"
    for suffix in (".pickle", ".pkl", ".txt.gz", ".txt", ".csv"):
        candidates.append(networks_dir / f"{name}{suffix}")
    return candidates


def load_dataset_config_file(config_path: Path | str, base_dir: Path | str = ".") -> DatasetConfig:
    """Load a dataset configuration from a JSON file."""

    path = _resolve_input_path(config_path, base_dir)
    _ensure_exists(path, "Dataset config")
    if path.suffix.lower() != ".json":
        raise ValueError(f"Unsupported dataset config format '{path.suffix}'. Only JSON config files are supported.")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset config '{path}' must contain a JSON object.")

    config_dir = path.parent
    name = str(payload.get("name") or "")
    graph_path_value = payload.get("graph_path") or payload.get("edge_path") or payload.get("pickle_path")
    dataset_format = payload.get("dataset_format") or payload.get("graph_format")
    attribute_path_value = payload.get("attributes_path") or payload.get("attribute_path")

    if graph_path_value is None:
        raise ValueError(f"Dataset config '{path}' must define one of: graph_path, edge_path, or pickle_path.")

    graph_path = _resolve_input_path(str(graph_path_value), config_dir)
    resolved_format = _normalize_dataset_format(dataset_format)
    inferred_format = resolved_format or _infer_dataset_format_from_path(graph_path)
    dataset_name = name or _default_dataset_name_from_path(graph_path)

    config = DatasetConfig(
        name=dataset_name,
        attribute_path=(
            _resolve_input_path(str(attribute_path_value), config_dir)
            if attribute_path_value is not None
            else None
        ),
        dataset_format=inferred_format,
        directed=bool(payload.get("directed", True)),
        delimiter=payload.get("delimiter"),
        source_column=str(payload.get("source_col") or payload.get("source_column") or "source"),
        target_column=str(payload.get("target_col") or payload.get("target_column") or "target"),
        node_id_column=str(payload.get("node_id_col") or payload.get("node_id_column") or "node_id"),
    )
    if inferred_format == "pickle":
        config.pickle_path = graph_path
    else:
        config.edge_path = graph_path
    return config


def resolve_dataset_config(
    dataset: str | None,
    *,
    base_dir: Path | str = ".",
    graph_path: str | Path | None = None,
    attributes_path: str | Path | None = None,
    dataset_format: str | None = None,
    dataset_config: str | Path | None = None,
    directed: bool | None = None,
    delimiter: str | None = None,
    source_col: str | None = None,
    target_col: str | None = None,
    node_id_col: str | None = None,
) -> DatasetConfig:
    """Resolve built-in and external dataset references into one DatasetConfig."""

    if dataset_config is not None:
        config = load_dataset_config_file(dataset_config, base_dir=base_dir)
        if config.name == "" and dataset is not None and not builtin_dataset_exists(dataset):
            config.name = str(dataset)
        if attributes_path is not None:
            config.attribute_path = _resolve_input_path(attributes_path, base_dir)
        if dataset_format is not None:
            config.dataset_format = _normalize_dataset_format(dataset_format)
        if directed is not None:
            config.directed = directed
        if delimiter is not None:
            config.delimiter = delimiter
        if source_col is not None:
            config.source_column = source_col
        if target_col is not None:
            config.target_column = target_col
        if node_id_col is not None:
            config.node_id_column = node_id_col
        return config

    if graph_path is not None:
        resolved_graph_path = _resolve_input_path(graph_path, base_dir)
        resolved_format = _normalize_dataset_format(dataset_format) or _infer_dataset_format_from_path(resolved_graph_path)
        config = DatasetConfig(
            name=(
                str(dataset)
                if dataset is not None and not builtin_dataset_exists(dataset)
                else _default_dataset_name_from_path(resolved_graph_path)
            ),
            attribute_path=_resolve_input_path(attributes_path, base_dir) if attributes_path is not None else None,
            dataset_format=resolved_format,
            directed=True if directed is None else directed,
            delimiter=delimiter,
            source_column=source_col or "source",
            target_column=target_col or "target",
            node_id_column=node_id_col or "node_id",
        )
        if resolved_format == "pickle":
            config.pickle_path = resolved_graph_path
        else:
            config.edge_path = resolved_graph_path
        return config

    if dataset is None:
        raise FileNotFoundError(
            "No dataset source was provided. Supply a built-in --dataset, --graph-path, or --dataset-config."
        )

    if builtin_dataset_exists(dataset):
        config = resolve_builtin_dataset(dataset, base_dir)
        if attributes_path is not None:
            config.attribute_path = _resolve_input_path(attributes_path, base_dir)
        if node_id_col is not None:
            config.node_id_column = node_id_col
        return config

    for candidate_path in _dataset_candidates_from_name(dataset, base_dir):
        if candidate_path.exists() and candidate_path.is_file():
            inferred_format = _normalize_dataset_format(dataset_format) or _infer_dataset_format_from_path(candidate_path)
            config = DatasetConfig(
                name=_default_dataset_name_from_path(candidate_path),
                attribute_path=_resolve_input_path(attributes_path, base_dir) if attributes_path is not None else None,
                dataset_format=inferred_format,
                directed=True if directed is None else directed,
                delimiter=delimiter,
                source_column=source_col or "source",
                target_column=target_col or "target",
                node_id_column=node_id_col or "node_id",
            )
            if inferred_format == "pickle":
                config.pickle_path = candidate_path
            else:
                config.edge_path = candidate_path
            return config

    raise FileNotFoundError(
        f"Unknown built-in dataset '{dataset}'. "
        "Provide --graph-path or --dataset-config for custom datasets, or place a supported file under networks/."
    )


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")


def _graph_node_id_type(graph: nx.Graph) -> type[Any]:
    if graph.number_of_nodes() == 0:
        raise ValueError("Graph is empty.")

    node_types = {type(node) for node in graph.nodes()}
    if len(node_types) != 1:
        names = ", ".join(sorted(node_type.__name__ for node_type in node_types))
        raise ValueError(f"Inconsistent node ID types in graph: {names}.")
    return next(iter(node_types))


def _infer_delimiter(path: Path, delimiter: str | None) -> str | None:
    if delimiter is not None:
        return delimiter
    suffixes = _path_suffixes(path)
    if suffixes[-1:] == (".csv",) or suffixes[-2:] == (".csv", ".gz"):
        return ","
    if suffixes[-1:] == (".tsv",) or suffixes[-2:] == (".tsv", ".gz"):
        return "\t"
    return None


def _is_int_token(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def _is_float_token(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _infer_node_parser(tokens: list[str]) -> Any:
    if tokens and all(_is_int_token(token) for token in tokens):
        return int
    if tokens and all(_is_float_token(token) for token in tokens):
        return float
    return str


def _open_text_file(path: Path):
    if _path_suffixes(path)[-1:] == (".gz",):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _read_simple_edge_list(path: Path, directed: bool, delimiter: str | None) -> nx.Graph:
    graph_type = nx.DiGraph if directed else nx.Graph
    edges: list[tuple[str, str]] = []

    with _open_text_file(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(delimiter) if delimiter is not None else line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Edge list '{path}' has line {line_number} with {len(parts)} columns; expected exactly 2."
                )
            edges.append((parts[0].strip(), parts[1].strip()))

    if not edges:
        raise ValueError(f"Edge list '{path}' does not contain any edges.")

    parser = _infer_node_parser([token for edge in edges for token in edge])
    graph = graph_type()
    graph.add_edges_from((parser(source), parser(target)) for source, target in edges)
    return graph


def _read_csv_edge_list(config: DatasetConfig) -> nx.Graph:
    if config.edge_path is None:
        raise ValueError("edge_path is required for CSV edge-list loading.")

    graph_type = nx.DiGraph if config.directed else nx.Graph
    edge_path = Path(config.edge_path)
    frame = pd.read_csv(
        edge_path,
        sep=_infer_delimiter(edge_path, config.delimiter) or ",",
        compression="infer",
    )
    if config.source_column not in frame.columns or config.target_column not in frame.columns:
        raise ValueError(
            f"CSV edge list '{edge_path}' must contain source column '{config.source_column}' "
            f"and target column '{config.target_column}'."
        )

    edge_frame = frame[[config.source_column, config.target_column]].copy()
    if edge_frame.isna().any().any():
        raise ValueError(f"CSV edge list '{edge_path}' contains null source or target node IDs.")

    graph = nx.from_pandas_edgelist(
        edge_frame,
        source=config.source_column,
        target=config.target_column,
        create_using=graph_type(),
    )
    if graph.number_of_edges() == 0:
        raise ValueError(f"CSV edge list '{edge_path}' does not contain any edges.")
    return graph


def _extract_node_attributes(graph: nx.Graph) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for node_id in sorted(graph.nodes()):
        record = {"node_id": node_id}
        record.update(graph.nodes[node_id])
        records.append(record)
    return pd.DataFrame(records).set_index("node_id", drop=False)


def _coerce_node_id(value: Any, expected_type: type[Any]) -> Any:
    if pd.isna(value):
        raise ValueError("Attribute file contains null node IDs.")
    try:
        return expected_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Inconsistent node ID types between graph ({expected_type.__name__}) "
            f"and attribute data value '{value}'."
        ) from exc


def _read_attribute_table(attribute_path: Path) -> pd.DataFrame:
    delimiter = _infer_delimiter(attribute_path, None)
    if delimiter is not None:
        return pd.read_csv(attribute_path, sep=delimiter, compression="infer")
    return pd.read_csv(attribute_path, sep=None, engine="python", compression="infer")


def _attach_attributes_from_table(attribute_path: Path, graph: nx.Graph, node_id_column: str) -> None:
    _ensure_exists(attribute_path, "Attribute")

    frame = _read_attribute_table(attribute_path)
    if node_id_column not in frame.columns:
        raise ValueError(
            f"Attribute file '{attribute_path}' does not contain node id column '{node_id_column}'."
        )

    expected_type = _graph_node_id_type(graph)
    coerced_ids = frame[node_id_column].map(lambda value: _coerce_node_id(value, expected_type))
    if coerced_ids.duplicated().any():
        duplicates = sorted({value for value in coerced_ids[coerced_ids.duplicated()].tolist()})
        raise ValueError(f"Attribute file '{attribute_path}' contains duplicate node IDs: {duplicates}.")

    frame = frame.copy()
    frame[node_id_column] = coerced_ids

    graph_nodes = set(graph.nodes())
    missing_nodes = sorted(set(frame[node_id_column]) - graph_nodes)
    if missing_nodes:
        preview = ", ".join(str(node_id) for node_id in missing_nodes[:5])
        raise ValueError(
            f"Attribute file '{attribute_path}' contains node IDs not present in the graph: {preview}."
        )

    for row in frame.to_dict(orient="records"):
        node_id = row[node_id_column]
        for key, value in row.items():
            if key == node_id_column or pd.isna(value):
                continue
            graph.nodes[node_id][key] = value


def load_graph_from_pickle(config: DatasetConfig) -> LoadedDataset:
    """Load a pickled NetworkX graph and optional external node attributes."""

    if config.pickle_path is None:
        raise ValueError("pickle_path is required for pickle loading.")

    pickle_path = Path(config.pickle_path)
    _ensure_exists(pickle_path, "Pickle")

    with pickle_path.open("rb") as handle:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
                category=VisibleDeprecationWarning,
            )
            graph = pickle.load(handle)

    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Pickle at '{pickle_path}' does not contain a NetworkX graph.")

    _graph_node_id_type(graph)

    if config.attribute_path is not None:
        _attach_attributes_from_table(Path(config.attribute_path), graph, config.node_id_column)

    return LoadedDataset(
        name=config.name,
        graph=graph,
        node_attributes=_extract_node_attributes(graph),
    )


def load_graph_from_edgelist(config: DatasetConfig) -> LoadedDataset:
    """Load a graph from a simple edge list and optional attribute CSV."""

    if config.edge_path is None:
        raise ValueError("edge_path is required for edge-list loading.")

    edge_path = Path(config.edge_path)
    _ensure_exists(edge_path, "Edge list")

    edge_format = _normalize_dataset_format(config.dataset_format) or _infer_dataset_format_from_path(edge_path)
    if edge_format == "csv":
        graph = _read_csv_edge_list(config)
    elif edge_format == "edge_list":
        graph = _read_simple_edge_list(
            path=edge_path,
            directed=config.directed,
            delimiter=_infer_delimiter(edge_path, config.delimiter),
        )
    else:
        raise ValueError(f"Unsupported edge-list dataset format '{config.dataset_format}'.")
    _graph_node_id_type(graph)

    if config.attribute_path is not None:
        _attach_attributes_from_table(Path(config.attribute_path), graph, config.node_id_column)

    return LoadedDataset(
        name=config.name,
        graph=graph,
        node_attributes=_extract_node_attributes(graph),
    )


def load_dataset(config: DatasetConfig) -> LoadedDataset:
    """Load a dataset from the first available supported source."""

    if config.pickle_path is not None and Path(config.pickle_path).exists():
        return load_graph_from_pickle(config)
    if config.edge_path is not None and Path(config.edge_path).exists():
        return load_graph_from_edgelist(config)

    if config.pickle_path is not None:
        raise FileNotFoundError(f"Pickle file not found: {config.pickle_path}")
    if config.edge_path is not None:
        raise FileNotFoundError(f"Edge list file not found: {config.edge_path}")
    raise FileNotFoundError(f"Dataset '{config.name}' is missing both pickle_path and edge_path.")


def _normalize_group_label(value: Any) -> str:
    return str(value).strip()


def verify_protected_groups(
    dataset: LoadedDataset,
    protected_attribute: str,
    sample_size: int = 5,
) -> ProtectedGroupReport:
    """Validate the protected attribute and build deterministic protected groups."""

    if protected_attribute not in dataset.node_attributes.columns:
        raise ValueError(
            f"Protected attribute '{protected_attribute}' is missing from dataset '{dataset.name}'."
        )

    attribute_series = dataset.node_attributes[protected_attribute]
    null_mask = attribute_series.isna() | (
        attribute_series.astype(str).str.strip() == ""
    )

    if null_mask.all():
        raise ValueError(
            f"Protected attribute '{protected_attribute}' in dataset '{dataset.name}' is all-null."
        )
    if null_mask.any():
        raise ValueError(
            f"Protected attribute '{protected_attribute}' in dataset '{dataset.name}' "
            f"is missing for {int(null_mask.sum())} nodes."
        )

    groups: dict[str, list[Any]] = defaultdict(list)
    for node_id, value in attribute_series.items():
        groups[_normalize_group_label(value)].append(node_id)

    if not groups:
        raise ValueError(
            f"Protected attribute '{protected_attribute}' in dataset '{dataset.name}' produced no groups."
        )
    if any(len(node_ids) == 0 for node_ids in groups.values()):
        raise ValueError(
            f"Protected attribute '{protected_attribute}' in dataset '{dataset.name}' contains an empty group."
        )

    sample_rows = dataset.node_attributes.head(sample_size).where(
        pd.notna(dataset.node_attributes.head(sample_size)),
        None,
    )
    sample_node_attributes = sample_rows.to_dict(orient="records")

    protected_groups = {
        group_name: tuple(sorted(node_ids))
        for group_name, node_ids in sorted(groups.items(), key=lambda item: item[0])
    }
    group_sizes = {group_name: len(node_ids) for group_name, node_ids in protected_groups.items()}

    return ProtectedGroupReport(
        dataset_name=dataset.name,
        protected_attribute=protected_attribute,
        node_count=dataset.graph.number_of_nodes(),
        edge_count=dataset.graph.number_of_edges(),
        is_directed=dataset.graph.is_directed(),
        sample_node_attributes=sample_node_attributes,
        protected_groups=protected_groups,
        group_sizes=group_sizes,
    )


def verify_dataset_phase1(config: DatasetConfig, protected_attribute: str) -> ProtectedGroupReport:
    """Load a dataset and return the validated Phase 1 protected-group report."""

    dataset = load_dataset(config)
    return verify_protected_groups(dataset, protected_attribute)


def format_phase1_report(report: ProtectedGroupReport) -> str:
    """Render a human-readable deterministic Phase 1 validation summary."""

    lines = [
        f"dataset: {report.dataset_name}",
        f"protected_attribute: {report.protected_attribute}",
        f"graph: nodes={report.node_count} edges={report.edge_count} directed={report.is_directed}",
        "sample_node_attributes:",
    ]
    for sample in report.sample_node_attributes:
        lines.append(f"  - {sample}")

    lines.append(f"protected_group_names: {list(report.protected_groups)}")
    lines.append("protected_group_sizes:")
    for group_name, size in report.group_sizes.items():
        lines.append(f"  - {group_name}: {size}")
    return "\n".join(lines)
