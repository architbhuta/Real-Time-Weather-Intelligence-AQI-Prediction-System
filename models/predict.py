import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

import joblib
import pandas as pd

from models.dataset import FEATURE_COLUMNS, baseline_predict
from utils.config import SAVED_MODELS_DIR


def load_model(horizon: str) -> tuple[object, str] | None:
    model_path = os.path.join(SAVED_MODELS_DIR, f"aqi_{horizon}.joblib")
    name_path = os.path.join(SAVED_MODELS_DIR, f"aqi_{horizon}_model_name.txt")
    if not (os.path.exists(model_path) and os.path.exists(name_path)):
        return None
    model = joblib.load(model_path)
    with open(name_path) as f:
        model_name = f.read().strip()
    return model, model_name


def predict_aqi(horizon: str, feature_row: pd.DataFrame) -> tuple[float, str]:
    loaded = load_model(horizon)
    if loaded is None:
        predicted = float(baseline_predict(feature_row).iloc[0])
        return predicted, "baseline"

    model, model_name = loaded
    predicted = float(model.predict(feature_row[FEATURE_COLUMNS])[0])
    return predicted, model_name
