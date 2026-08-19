import os
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from data.database import get_engine, load_weather_and_air_quality
from models.dataset import FEATURE_COLUMNS, TARGET_HORIZONS, baseline_predict, build_training_frame, chronological_split
from models.evaluate import compute_metrics
from utils.config import DEFAULT_LOCATION, SAVED_MODELS_DIR
from utils.logging_config import get_logger

logger = get_logger(__name__)

MIN_TRAINING_ROWS = 50
METRICS_LOG_PATH = os.path.join(SAVED_MODELS_DIR, "metrics.csv")


def _candidate_models() -> dict:
    return {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "xgboost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    }


def train_and_evaluate(location: str, engine=None) -> dict:
    engine = engine or get_engine()
    raw_df = load_weather_and_air_quality(engine, location)
    frame = build_training_frame(raw_df)

    if len(frame) < MIN_TRAINING_ROWS:
        logger.warning(
            "Only %d usable rows for %s after feature engineering (need >= %d); skipping training",
            len(frame), location, MIN_TRAINING_ROWS,
        )
        return {}

    train_df, val_df, test_df = chronological_split(frame)
    results = {}
    metrics_rows = []

    for label in TARGET_HORIZONS:
        target_col = f"aqi_target_{label}"
        X_train, y_train = train_df[FEATURE_COLUMNS], train_df[target_col]
        X_val, y_val = val_df[FEATURE_COLUMNS], val_df[target_col]
        X_test, y_test = test_df[FEATURE_COLUMNS], test_df[target_col]

        baseline_val_metrics = compute_metrics(y_val, baseline_predict(val_df))
        metrics_rows.append({"horizon": label, "model": "baseline", "split": "validation", **baseline_val_metrics})

        val_scores = {}
        fitted = {}
        for name, estimator in _candidate_models().items():
            estimator.fit(X_train, y_train)
            fitted[name] = estimator
            val_metrics = compute_metrics(y_val, estimator.predict(X_val))
            val_scores[name] = val_metrics["mae"]
            metrics_rows.append({"horizon": label, "model": name, "split": "validation", **val_metrics})

        best_name = min(val_scores, key=val_scores.get)
        best_model = fitted[best_name]
        test_metrics = compute_metrics(y_test, best_model.predict(X_test))
        metrics_rows.append({"horizon": label, "model": best_name, "split": "test", **test_metrics})

        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        model_path = os.path.join(SAVED_MODELS_DIR, f"aqi_{label}.joblib")
        name_path = os.path.join(SAVED_MODELS_DIR, f"aqi_{label}_model_name.txt")
        joblib.dump(best_model, model_path)
        with open(name_path, "w") as f:
            f.write(best_name)

        logger.info(
            "Horizon %s: best model=%s (val MAE=%.2f), test MAE=%.2f, saved to %s",
            label, best_name, val_scores[best_name], test_metrics["mae"], model_path,
        )
        results[label] = {"best_model": best_name, "test_metrics": test_metrics, "model_path": model_path}

    _append_metrics_log(metrics_rows, location, len(frame))
    return results


def _append_metrics_log(metrics_rows: list[dict], location: str, dataset_size: int) -> None:
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    log_df = pd.DataFrame(metrics_rows)
    log_df["location"] = location
    log_df["dataset_size"] = dataset_size
    log_df["trained_at"] = datetime.now(timezone.utc).isoformat()
    if os.path.exists(METRICS_LOG_PATH):
        log_df.to_csv(METRICS_LOG_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(METRICS_LOG_PATH, index=False)


if __name__ == "__main__":
    location = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOCATION
    train_and_evaluate(location)
