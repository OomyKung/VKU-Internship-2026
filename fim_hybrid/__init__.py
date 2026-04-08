"""Research-friendly Fair Influence Maximization package."""

# Re-export the main entry points so scripts and notebooks can import from the
# package root instead of reaching into individual module files.
from .config import (
    CommunityConfig,
    DatasetConfig,
    DiffusionConfig,
    ExperimentConfig,
    FairnessConfig,
    MLConfig,
    OptimizerConfig,
)
from .community_detection import CommunityDetectionResult, detect_communities
from .data_loader import LoadedDataset, load_dataset, resolve_builtin_dataset
from .diffusion import DiffusionResult, IndependentCascadeSimulator
from .fairness import FairnessMetrics, evaluate_fairness
from .feature_extraction import compute_node_features
from .hybrid_optimizer import HybridOptimizationResult, HybridSIEAOptimizer

# Keep the public package surface explicit for readability and stability.
__all__ = [
    "CommunityConfig",
    "CommunityDetectionResult",
    "DatasetConfig",
    "DiffusionConfig",
    "DiffusionResult",
    "ExperimentConfig",
    "FairnessConfig",
    "FairnessMetrics",
    "HybridOptimizationResult",
    "HybridSIEAOptimizer",
    "IndependentCascadeSimulator",
    "LoadedDataset",
    "MLConfig",
    "OptimizerConfig",
    "compute_node_features",
    "detect_communities",
    "evaluate_fairness",
    "load_dataset",
    "resolve_builtin_dataset",
]
