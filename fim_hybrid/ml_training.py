"""Tabular ML training utilities for candidate ranking in fair FIM."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency.
    XGBRegressor = None


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _rank_nodes(node_scores: dict[Any, float]) -> tuple[Any, ...]:
    return tuple(
        sorted(
            node_scores,
            key=lambda node_id: (-float(node_scores[node_id]), _sort_key(node_id)),
        )
    )


def _safe_spearman(y_true: pd.Series, y_pred: pd.Series) -> float:
    correlation = y_true.astype(float).corr(y_pred.astype(float), method="spearman")
    if pd.isna(correlation):
        return 0.0
    return float(correlation)


def _precision_at_k(
    node_ids: pd.Series,
    true_scores: pd.Series,
    predicted_scores: pd.Series,
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1.")

    ranking_frame = pd.DataFrame(
        {
            "node_id": node_ids.tolist(),
            "true_score": true_scores.astype(float).tolist(),
            "predicted_score": predicted_scores.astype(float).tolist(),
        }
    )
    predicted_top = ranking_frame.sort_values(
        by=["predicted_score", "node_id"],
        ascending=[False, True],
    )["node_id"].head(k)
    true_top = ranking_frame.sort_values(
        by=["true_score", "node_id"],
        ascending=[False, True],
    )["node_id"].head(k)
    overlap = set(predicted_top.tolist()) & set(true_top.tolist())
    return float(len(overlap) / float(k))


@dataclass(frozen=True, slots=True)
class RankingModelSpec:
    """Static metadata for one tabular or score-producing ranking model."""

    name: str
    backend: str
    objective_type: str
    optional_dependencies: tuple[str, ...] = ()
    description: str = ""


def _build_feature_matrix(feature_frame: pd.DataFrame, label_frame: pd.DataFrame) -> pd.DataFrame:
    left = feature_frame.reset_index(drop=True).copy()
    right = label_frame.reset_index(drop=True).copy()

    if left["node_id"].duplicated().any():
        raise ValueError("feature_frame contains duplicate node_id values.")
    if right["node_id"].duplicated().any():
        raise ValueError("label_frame contains duplicate node_id values.")

    merged = left.merge(
        right,
        on="node_id",
        how="inner",
        validate="one_to_one",
    )
    missing_nodes = sorted(
        set(left["node_id"]) ^ set(right["node_id"]),
        key=_sort_key,
    )
    if missing_nodes:
        raise ValueError(f"feature_frame and label_frame must cover the same nodes: {missing_nodes[:5]}.")
    return merged


def _ranking_model_registry() -> dict[str, RankingModelSpec]:
    return {
        "random_forest": RankingModelSpec(
            name="random_forest",
            backend="tabular",
            objective_type="regression",
            description="Random-forest regressor over handcrafted node features.",
        ),
        "xgboost": RankingModelSpec(
            name="xgboost",
            backend="tabular",
            objective_type="regression",
            optional_dependencies=("xgboost",),
            description="Gradient-boosted tree regressor over handcrafted node features.",
        ),
        "mlp": RankingModelSpec(
            name="mlp",
            backend="tabular",
            objective_type="regression",
            description="Feed-forward MLP regressor over handcrafted node features.",
        ),
        "logistic_regression": RankingModelSpec(
            name="logistic_regression",
            backend="tabular",
            objective_type="binary_top_budget_classification",
            description=(
                "Logistic classifier that ranks nodes by probability of belonging to the "
                "top-budget singleton-utility set."
            ),
        ),
        "graphsage": RankingModelSpec(
            name="graphsage",
            backend="gnn",
            objective_type="regression",
            optional_dependencies=("torch", "torch_geometric"),
            description="GraphSAGE regressor used by the GNN ranking backend.",
        ),
        "gcn": RankingModelSpec(
            name="gcn",
            backend="gnn",
            objective_type="regression",
            optional_dependencies=("torch", "torch_geometric"),
            description="GCN regressor used by the GNN ranking backend.",
        ),
        "ris_guidance": RankingModelSpec(
            name="ris_guidance",
            backend="ris",
            objective_type="coverage_prior",
            description="RIS node-coverage prior used for search-time guidance only.",
        ),
    }


def available_ranking_models() -> tuple[str, ...]:
    """Return all supported ranking/scoring model names."""

    return tuple(_ranking_model_registry())


def get_ranking_model_spec(model_type: str) -> RankingModelSpec:
    """Return static metadata for one ranking model."""

    key = str(model_type).strip().lower()
    registry = _ranking_model_registry()
    if key not in registry:
        raise ValueError(f"Unknown ranking model '{model_type}'.")
    return registry[key]


def select_ml_candidate_nodes(
    ranked_nodes: tuple[Any, ...],
    budget: int,
    top_fraction: float | None = None,
    top_n: int | None = None,
    max_nodes: int | None = None,
) -> tuple[Any, ...]:
    """Select the candidate pool from a global predicted ranking."""

    if budget < 1:
        raise ValueError("budget must be at least 1.")
    if not ranked_nodes:
        raise ValueError("ranked_nodes must contain at least one node.")
    if top_fraction is not None and not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in the interval (0.0, 1.0].")
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be at least 1.")
    if max_nodes is not None and max_nodes < 1:
        raise ValueError("max_nodes must be at least 1.")

    candidate_count = len(ranked_nodes)
    if top_n is not None:
        candidate_count = top_n
    elif top_fraction is not None:
        candidate_count = int(math.ceil(len(ranked_nodes) * top_fraction))

    if max_nodes is not None:
        candidate_count = min(candidate_count, max_nodes)

    candidate_count = max(budget, candidate_count)
    candidate_count = min(candidate_count, len(ranked_nodes))
    return tuple(ranked_nodes[:candidate_count])


def _split_training_frame(
    training_frame: pd.DataFrame,
    random_seed: int,
    stratify_labels: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_count = len(training_frame)
    if node_count < 4:
        return training_frame.copy(), training_frame.copy()

    test_size = max(1, int(round(node_count * 0.25)))
    test_size = min(test_size, node_count - 2)
    stratify = None
    if stratify_labels is not None:
        unique_values = set(pd.Series(stratify_labels).dropna().astype(int).tolist())
        if len(unique_values) >= 2:
            stratify = stratify_labels
    train_frame, validation_frame = train_test_split(
        training_frame,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
        stratify=stratify,
    )
    return train_frame.copy(), validation_frame.copy()


def _build_random_forest_pipeline(
    categorical_columns: list[str],
    numeric_columns: list[str],
    random_seed: int,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
            ("numeric", "passthrough", numeric_columns),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=random_seed,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _build_xgboost_pipeline(
    categorical_columns: list[str],
    numeric_columns: list[str],
    random_seed: int,
) -> Pipeline:
    if XGBRegressor is None:
        raise ValueError("model_type='xgboost' requires the xgboost package to be installed.")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
            ("numeric", "passthrough", numeric_columns),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=random_seed,
                    n_jobs=1,
                    verbosity=0,
                ),
            ),
        ]
    )


def _build_mlp_pipeline(
    categorical_columns: list[str],
    numeric_columns: list[str],
    random_seed: int,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
            ("numeric", "passthrough", numeric_columns),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def _build_logistic_regression_pipeline(
    categorical_columns: list[str],
    numeric_columns: list[str],
    random_seed: int,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
            ("numeric", "passthrough", numeric_columns),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def _build_model_pipeline(
    model_type: str,
    categorical_columns: list[str],
    numeric_columns: list[str],
    random_seed: int,
) -> Pipeline:
    if model_type == "random_forest":
        return _build_random_forest_pipeline(categorical_columns, numeric_columns, random_seed)
    if model_type == "xgboost":
        return _build_xgboost_pipeline(categorical_columns, numeric_columns, random_seed)
    if model_type == "mlp":
        return _build_mlp_pipeline(categorical_columns, numeric_columns, random_seed)
    if model_type == "logistic_regression":
        return _build_logistic_regression_pipeline(categorical_columns, numeric_columns, random_seed)
    raise ValueError(
        "model_type must be one of ['random_forest', 'xgboost', 'mlp', 'logistic_regression']."
    )


def _classification_target(
    training_frame: pd.DataFrame,
    budget: int,
) -> pd.Series:
    if budget < 1:
        raise ValueError("budget must be at least 1 for logistic_regression ranking.")
    if len(training_frame) < 2:
        raise ValueError("logistic_regression ranking requires at least two nodes.")
    positive_count = min(max(int(budget), 1), len(training_frame) - 1)
    ordered = training_frame.sort_values(
        by=["label_score", "node_id"],
        ascending=[False, True],
    )
    target = pd.Series(0, index=training_frame.index, dtype=int)
    target.loc[ordered.index[:positive_count]] = 1
    if target.nunique() < 2:
        raise ValueError(
            "logistic_regression ranking requires both positive and negative training labels."
        )
    return target


def _predict_scores(
    model_type: str,
    model: Pipeline,
    feature_frame: pd.DataFrame,
) -> pd.Series:
    if model_type == "logistic_regression":
        probabilities = model.predict_proba(feature_frame)
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError("logistic_regression ranking requires binary class probabilities.")
        return pd.Series(probabilities[:, 1], index=feature_frame.index, dtype=float)
    return pd.Series(model.predict(feature_frame), index=feature_frame.index, dtype=float)


@dataclass(slots=True)
class MLTrainingResult:
    """Fitted model outputs and ranking diagnostics for ML-guided FIM."""

    model: Pipeline
    training_frame: pd.DataFrame
    predicted_scores: dict[Any, float]
    ranked_nodes: tuple[Any, ...]
    candidate_nodes: tuple[Any, ...]
    validation_spearman: float
    validation_precision_at_budget: float
    runtime_seconds: float
    model_type: str
    target_type: str


def train_node_utility_model(
    feature_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    budget: int,
    model_type: str = "random_forest",
    top_fraction: float | None = None,
    top_n: int | None = None,
    max_nodes: int | None = None,
    random_seed: int = 42,
) -> MLTrainingResult:
    """Train a tabular regressor and derive a filtered candidate pool."""

    start = perf_counter()
    training_frame = _build_feature_matrix(feature_frame, label_frame)
    if "label_score" not in training_frame.columns:
        raise ValueError("label_frame must contain a label_score column.")

    excluded_columns = {
        "node_id",
        "singleton_total_spread",
        "singleton_mf",
        "singleton_soft_mf",
        "singleton_dcv",
        "singleton_soft_fair_score",
        "spread_norm",
        "soft_fair_norm",
        "label_score",
    }
    categorical_columns = ["community_id", "protected_group"]
    numeric_columns = [
        column_name
        for column_name in training_frame.columns
        if column_name not in excluded_columns and column_name not in categorical_columns
    ]
    target_type = "regression"
    binary_target: pd.Series | None = None
    if model_type == "logistic_regression":
        binary_target = _classification_target(training_frame, budget=budget)
        target_type = "binary_top_budget_classification"

    train_frame, validation_frame = _split_training_frame(
        training_frame,
        random_seed=random_seed,
        stratify_labels=binary_target,
    )

    validation_pipeline = _build_model_pipeline(
        model_type=model_type,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        random_seed=random_seed,
    )
    if binary_target is not None:
        train_target = binary_target.loc[train_frame.index]
        full_target = binary_target
    else:
        train_target = train_frame["label_score"]
        full_target = training_frame["label_score"]
    validation_pipeline.fit(
        train_frame[categorical_columns + numeric_columns],
        train_target,
    )
    validation_predictions = _predict_scores(
        model_type,
        validation_pipeline,
        validation_frame[categorical_columns + numeric_columns],
    )
    validation_spearman = _safe_spearman(validation_frame["label_score"], validation_predictions)
    validation_precision_at_budget = _precision_at_k(
        node_ids=validation_frame["node_id"],
        true_scores=validation_frame["label_score"],
        predicted_scores=validation_predictions,
        k=min(budget, len(validation_frame)),
    )

    final_model = _build_model_pipeline(
        model_type=model_type,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        random_seed=random_seed,
    )
    final_model.fit(
        training_frame[categorical_columns + numeric_columns],
        full_target,
    )
    predicted_series = _predict_scores(
        model_type,
        final_model,
        training_frame[categorical_columns + numeric_columns],
    )
    predicted_scores = {
        row.node_id: float(score)
        for row, score in zip(training_frame.itertuples(index=False), predicted_series, strict=True)
    }
    ranked_nodes = _rank_nodes(predicted_scores)
    candidate_nodes = select_ml_candidate_nodes(
        ranked_nodes=ranked_nodes,
        budget=budget,
        top_fraction=top_fraction,
        top_n=top_n,
        max_nodes=max_nodes,
    )
    runtime_seconds = perf_counter() - start

    return MLTrainingResult(
        model=final_model,
        training_frame=training_frame,
        predicted_scores=predicted_scores,
        ranked_nodes=ranked_nodes,
        candidate_nodes=candidate_nodes,
        validation_spearman=validation_spearman,
        validation_precision_at_budget=validation_precision_at_budget,
        runtime_seconds=runtime_seconds,
        model_type=model_type,
        target_type=target_type,
    )
