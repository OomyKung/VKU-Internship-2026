"""Graph loading utilities for Fair Influence Maximization experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import pandas as pd
import pickle

from .config import DatasetConfig


@dataclass(slots=True)
class LoadedDataset:
    """Loaded dataset bundle."""

    name: str
    graph: nx.Graph
    node_attributes: pd.DataFrame


_BUILTIN_DATASETS: dict[str, dict[str, str]] = {
    # Short names used by the CLI and experiment runner.
    "twitter": {"pickle": "networks/twitter.pickle", "edge": "networks/twitter.txt"},
    "synth1": {"edge": "networks/synth1.txt"},
    "synth2": {"pickle": "networks/synth2.pickle", "edge": "networks/synth2.txt"},
    "synth3": {"pickle": "networks/synth3.pickle", "edge": "networks/synth3.txt"},
    "rice-facebook": {"pickle": "networks/rice_subset.pickle", "edge": "networks/rice_subset.txt"},
    "rice_subset": {"pickle": "networks/rice_subset.pickle", "edge": "networks/rice_subset.txt"},
    "graph_spa_500_0": {"pickle": "networks/graph_spa_500_0.pickle", "edge": "networks/graph_spa_500_0.txt"},
}


def resolve_builtin_dataset(name: str, base_dir: Path | str = ".") -> DatasetConfig:
    """Resolve a built-in dataset name to local file paths."""

    key = name.lower()
    if key not in _BUILTIN_DATASETS:
        raise FileNotFoundError(f"Unknown built-in dataset '{name}'.")

    base_dir = Path(base_dir)
    paths = _BUILTIN_DATASETS[key]
    edge_path = base_dir / paths["edge"] if "edge" in paths else None
    pickle_path = base_dir / paths["pickle"] if "pickle" in paths else None

    return DatasetConfig(
        name=name,
        edge_path=edge_path,
        pickle_path=pickle_path,
        directed=False,
    )


def _ensure_exists(path: Path, label: str) -> None:
    # Centralized existence checks keep error messages consistent.
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")


def _coerce_node_id(value: Any, template_node: Any) -> Any:
    # CSV loaders often read ids as strings; cast them back to the graph's node
    # type when possible so joins work correctly.
    template_type = type(template_node)
    try:
        return template_type(value)
    except (TypeError, ValueError):
        return value


def _attach_edge_probabilities(graph: nx.Graph, probability: float) -> None:
    # Downstream diffusion code expects each edge to expose `data["p"]`.
    for _, _, data in graph.edges(data=True):
        data["p"] = float(probability)


def _graph_node_attributes(graph: nx.Graph) -> pd.DataFrame:
    # Convert attached node metadata into a DataFrame for feature extraction and ML.
    records: list[dict[str, Any]] = []
    for node, attrs in graph.nodes(data=True):
        record = {"node_id": node}
        record.update(attrs)
        records.append(record)
    return pd.DataFrame(records).set_index("node_id", drop=False)


def _load_attributes_csv(
    attribute_path: Path,
    graph: nx.Graph,
    node_id_column: str,
) -> pd.DataFrame:
    # Keep attribute loading separate so the project can support edge lists with
    # external CSV metadata as well as pickled graphs with embedded metadata.
    _ensure_exists(attribute_path, "Attribute")
    frame = pd.read_csv(attribute_path)

    if node_id_column not in frame.columns:
        raise ValueError(
            f"Attribute file '{attribute_path}' does not contain node id column '{node_id_column}'."
        )

    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot attach attributes to an empty graph.")

    template_node = next(iter(graph.nodes()))
    frame = frame.copy()
    # Cast CSV ids to the graph's node type before merging attributes.
    frame[node_id_column] = frame[node_id_column].map(lambda value: _coerce_node_id(value, template_node))
    frame = frame.set_index(node_id_column, drop=False)

    for node, row in frame.iterrows():
        if node not in graph:
            continue
        for key, value in row.items():
            if pd.isna(value):
                continue
            # Store non-missing CSV fields directly on each node.
            graph.nodes[node][key] = value

    return frame.rename(columns={node_id_column: "node_id"})


def load_graph_from_pickle(config: DatasetConfig) -> LoadedDataset:
    """Load a dataset from a pickle graph file."""

    if config.pickle_path is None:
        raise ValueError("pickle_path is required for pickle loading.")

    pickle_path = Path(config.pickle_path)
    _ensure_exists(pickle_path, "Pickle")

    with pickle_path.open("rb") as handle:
        graph = pickle.load(handle)

    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Pickle at '{pickle_path}' does not contain a NetworkX graph.")

    if not config.directed and graph.is_directed():
        # Most social-network experiments in this prototype use undirected graphs.
        graph = graph.to_undirected()

    _attach_edge_probabilities(graph, config.propagation_probability)

    if config.attribute_path is not None:
        attributes = _load_attributes_csv(Path(config.attribute_path), graph, config.node_id_column)
    else:
        attributes = _graph_node_attributes(graph)

    return LoadedDataset(name=config.name, graph=graph, node_attributes=attributes)


def load_graph_from_edgelist(config: DatasetConfig) -> LoadedDataset:
    """Load a dataset from an edge list and optional attribute CSV."""

    if config.edge_path is None:
        raise ValueError("edge_path is required for edge-list loading.")

    edge_path = Path(config.edge_path)
    _ensure_exists(edge_path, "Edge list")

    graph_type = nx.DiGraph if config.directed else nx.Graph
    try:
        # Prefer integer node ids because the current built-in datasets use them.
        graph = nx.read_edgelist(
            edge_path,
            create_using=graph_type(),
            delimiter=config.delimiter,
            nodetype=int,
        )
    except ValueError:
        # Fall back to raw strings for custom datasets with non-integer ids.
        graph = nx.read_edgelist(
            edge_path,
            create_using=graph_type(),
            delimiter=config.delimiter,
        )

    _attach_edge_probabilities(graph, config.propagation_probability)

    if config.attribute_path is not None:
        attributes = _load_attributes_csv(Path(config.attribute_path), graph, config.node_id_column)
    else:
        attributes = _graph_node_attributes(graph)

    return LoadedDataset(name=config.name, graph=graph, node_attributes=attributes)


def load_dataset(config: DatasetConfig) -> LoadedDataset:
    """Load a dataset from pickle or edge list with basic validation."""

    # Prefer pickled graphs when available because they preserve structure and
    # attributes exactly as originally serialized.
    if config.pickle_path is not None and Path(config.pickle_path).exists():
        return load_graph_from_pickle(config)

    if config.edge_path is not None:
        return load_graph_from_edgelist(config)

    raise FileNotFoundError(
        f"Dataset '{config.name}' is missing both a usable pickle_path and edge_path."
    )
