"""Machine learning utilities for candidate-pool restriction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

from .config import MLConfig


@dataclass(slots=True)
class MLTrainingResult:
    """Trained node-utility model and metadata."""

    # We also store the encoded training columns so inference can align features exactly.
    model_name: str
    model: object
    encoded_feature_columns: list[str]
    training_r2: float


def _base_feature_columns(frame: pd.DataFrame) -> list[str]:
    # Exclude targets and metadata so the model only sees explanatory features.
    excluded = {
        "node_id",
        "singleton_spread",
        "singleton_mf",
        "singleton_dcv",
        "singleton_score",
        "ml_label",
        "label_generation_seconds",
    }
    return [column for column in frame.columns if column not in excluded]


def _encode_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    # One-hot encoding keeps the implementation simple and works for mixed numeric
    # and categorical research features.
    subset = frame[feature_columns].copy()
    return pd.get_dummies(subset, dummy_na=True)


def train_node_ranker(
    training_frame: pd.DataFrame,
    config: MLConfig,
    target_column: str = "singleton_score",
) -> MLTrainingResult:
    """Train a node utility predictor."""

    feature_columns = _base_feature_columns(training_frame)
    encoded = _encode_features(training_frame, feature_columns)
    target = training_frame[target_column].astype(float)

    if config.model_name.lower() == "xgboost" and XGBRegressor is not None:
        # Use XGBoost only when explicitly requested and available locally.
        model = XGBRegressor(
            n_estimators=config.n_estimators,
            random_state=config.seed,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            verbosity=0,
        )
        model_name = "xgboost"
    else:
        # Random Forest is the default because it has minimal setup burden.
        model = RandomForestRegressor(
            n_estimators=config.n_estimators,
            random_state=config.seed,
            n_jobs=-1,
        )
        model_name = "random_forest"

    model.fit(encoded, target)
    # Training R^2 is reported mainly as a quick sanity check.
    predictions = model.predict(encoded)
    training_r2 = float(r2_score(target, predictions))

    return MLTrainingResult(
        model_name=model_name,
        model=model,
        encoded_feature_columns=encoded.columns.tolist(),
        training_r2=training_r2,
    )


def predict_node_utilities(
    feature_frame: pd.DataFrame,
    training_result: MLTrainingResult,
) -> pd.Series:
    """Predict node utility scores for all nodes."""

    feature_columns = _base_feature_columns(feature_frame)
    encoded = _encode_features(feature_frame, feature_columns)
    # Reindex to the training columns so inference works even if some dummy
    # columns are missing from the current frame.
    encoded = encoded.reindex(columns=training_result.encoded_feature_columns, fill_value=0.0)
    predictions = training_result.model.predict(encoded)
    return pd.Series(predictions, index=feature_frame.index, name="predicted_utility")


def select_top_candidate_pool(
    feature_frame: pd.DataFrame,
    predicted_scores: pd.Series,
    top_fraction: float,
) -> list[object]:
    """Restrict the search space to the top fraction of predicted nodes."""

    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in the interval (0, 1].")

    merged = feature_frame.copy()
    merged["predicted_utility"] = predicted_scores
    # Restricting the optimizer to the top-k% nodes is the main benefit of the ML stage.
    top_k = max(1, int(len(merged) * top_fraction))
    ranked = merged.sort_values("predicted_utility", ascending=False)
    return ranked.head(top_k)["node_id"].tolist()
