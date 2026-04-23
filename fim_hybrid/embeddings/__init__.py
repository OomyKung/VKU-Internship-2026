"""Modular graph embedding benchmark framework."""

from .base import (
    EmbeddingFrameworkError,
    EmbeddingResult,
    GraphEmbeddingModel,
    OptionalDependencyError,
    UnsupportedGraphTypeError,
)
from .benchmark import BenchmarkRunResult, run_embedding_benchmark, run_embedding_method
from .evaluation import (
    available_evaluation_tasks,
    evaluate_embedding_benchmark,
    evaluate_embedding_benchmark_repeated,
    evaluate_embedding_result,
    evaluation_result_columns,
    resolve_evaluation_tasks,
)
from .features import PreparedFeatures, coerce_input_features, prepare_benchmark_features
from .registry import (
    available_embedding_methods,
    create_embedding_model,
    get_method_spec,
    method_requires_features,
    resolve_method_names,
)

__all__ = [
    "available_embedding_methods",
    "available_evaluation_tasks",
    "BenchmarkRunResult",
    "coerce_input_features",
    "create_embedding_model",
    "EmbeddingFrameworkError",
    "EmbeddingResult",
    "evaluate_embedding_benchmark",
    "evaluate_embedding_benchmark_repeated",
    "evaluate_embedding_result",
    "evaluation_result_columns",
    "get_method_spec",
    "GraphEmbeddingModel",
    "method_requires_features",
    "OptionalDependencyError",
    "PreparedFeatures",
    "prepare_benchmark_features",
    "resolve_evaluation_tasks",
    "resolve_method_names",
    "run_embedding_benchmark",
    "run_embedding_method",
    "UnsupportedGraphTypeError",
]
