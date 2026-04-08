"""Configuration dataclasses for FIM experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class DatasetConfig:
    """Dataset configuration for loading a graph and optional node attributes."""

    # Short dataset label used in logs, output files, and result tables.
    name: str
    # The loader can work from either a raw edge list or a serialized pickle.
    edge_path: Optional[Path] = None
    attribute_path: Optional[Path] = None
    pickle_path: Optional[Path] = None
    # Node id column used when attaching a CSV of node attributes.
    node_id_column: str = "node_id"
    directed: bool = False
    delimiter: Optional[str] = None
    # Default IC propagation probability attached to each edge at load time.
    propagation_probability: float = 0.01


@dataclass(slots=True)
class CommunityConfig:
    """Community detection configuration."""

    # All community algorithms are selected through one field for easy sweeps.
    method: str = "leiden"
    weight_attribute: Optional[str] = None
    resolution: float = 1.0
    seed: int = 42


@dataclass(slots=True)
class DiffusionConfig:
    """Independent Cascade diffusion configuration."""

    # Monte Carlo diffusion settings used by evaluation and label generation.
    propagation_probability: float = 0.01
    mc_runs: int = 100
    seed: int = 42


@dataclass(slots=True)
class FairnessConfig:
    """Fairness evaluation configuration."""

    # Protected attribute defines the demographic or categorical groups.
    protected_attribute: str = "color"
    lambda_weight: float = 0.5
    target_mode: str = "population_proportional"
    score_mode: str = "raw_mf"


@dataclass(slots=True)
class OptimizerConfig:
    """Unified hybrid SI+EA optimizer configuration."""

    # Population size, seed budget, and iteration count.
    budget: int = 10
    population_size: int = 20
    generations: int = 50
    # Evolutionary operators.
    crossover_rate: float = 0.7
    mutation_rate: float = 0.1
    elite_fraction: float = 0.2
    # Swarm-style guidance strengths.
    swarm_inertia_rate: float = 0.35
    swarm_cognitive_rate: float = 0.30
    swarm_social_rate: float = 0.25
    swarm_elite_rate: float = 0.15
    restart_rate: float = 0.05
    fairness_repair_bias: float = 0.35
    # Optional convergence-based early stopping.
    convergence_patience: Optional[int] = None
    disable_swarm_guidance: bool = False
    disable_crossover: bool = False
    disable_community_repair: bool = False
    debug_logging: bool = False
    debug_frequency: int = 1
    seed: int = 42


@dataclass(slots=True)
class MLConfig:
    """Machine-learning-guided search space reduction configuration."""

    # This stage is optional and only filters nodes before the optimizer runs.
    enabled: bool = False
    model_name: str = "random_forest"
    top_fraction: float = 0.3
    n_estimators: int = 200
    singleton_mc_runs: int = 30
    positive_fraction: float = 0.2
    seed: int = 42


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    # Bundle all experiment settings into one object passed to the runner.
    dataset: DatasetConfig
    community: CommunityConfig = field(default_factory=CommunityConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    fairness: FairnessConfig = field(default_factory=FairnessConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    output_dir: Path = Path("results")
    random_seed: int = 42
