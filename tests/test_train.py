import os

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from data.database import AirQualityData, WeatherData, get_engine, init_db


def _seed_synthetic_history(engine, location="Delhi", n=200):
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-01-01 00:00", periods=n, freq="h")
    base_aqi = 100 + 40 * np.sin(np.linspace(0, 8 * np.pi, n)) + rng.normal(0, 5, n)
    base_pm25 = base_aqi * 0.5 + rng.normal(0, 3, n)

    with Session(engine) as session:
        for i, ts in enumerate(timestamps):
            session.add(WeatherData(
                timestamp=ts.to_pydatetime(), location=location, latitude=28.7041, longitude=77.1025,
                temperature=28 + 5 * np.sin(i / 24), feels_like=30.0, humidity=55.0, pressure=1005.0,
                wind_speed=10.0, wind_direction=200.0, rainfall=0.0, visibility=None,
                cloud_cover=20.0, uv_index=None,
            ))
            session.add(AirQualityData(
                timestamp=ts.to_pydatetime(), location=location,
                pm25=float(base_pm25[i]), pm10=float(base_pm25[i] + 40), co=450.0,
                no2=30.0, so2=8.0, o3=35.0, aqi=int(base_aqi[i]),
            ))
        session.commit()


def test_train_and_evaluate_produces_a_model_per_horizon(tmp_path, monkeypatch):
    import models.train as train_module

    saved_models_dir = str(tmp_path / "saved_models")
    monkeypatch.setattr(train_module, "SAVED_MODELS_DIR", saved_models_dir)

    engine = get_engine(":memory:")
    init_db(engine)
    _seed_synthetic_history(engine)

    results = train_module.train_and_evaluate("Delhi", engine=engine)

    assert set(results.keys()) == {"1h", "3h", "6h"}
    for label, result in results.items():
        assert result["best_model"] in {"linear_regression", "random_forest", "xgboost"}
        assert "mae" in result["test_metrics"]
        assert os.path.exists(result["model_path"])
        assert os.path.exists(os.path.join(saved_models_dir, f"aqi_{label}_model_name.txt"))

    assert os.path.exists(os.path.join(saved_models_dir, "metrics.csv"))
    metrics_df = pd.read_csv(os.path.join(saved_models_dir, "metrics.csv"))
    assert set(metrics_df["model"]) >= {"baseline", "linear_regression", "random_forest", "xgboost"}


def test_train_and_evaluate_skips_horizon_with_insufficient_data(tmp_path, monkeypatch):
    import models.train as train_module

    saved_models_dir = str(tmp_path / "saved_models")
    monkeypatch.setattr(train_module, "SAVED_MODELS_DIR", saved_models_dir)

    engine = get_engine(":memory:")
    init_db(engine)
    _seed_synthetic_history(engine, n=20)  # well under MIN_TRAINING_ROWS after feature/target trimming

    results = train_module.train_and_evaluate("Delhi", engine=engine)

    assert results == {}
    assert not os.path.exists(saved_models_dir)


def test_train_and_evaluate_returns_empty_when_location_has_no_stored_rows(tmp_path, monkeypatch):
    import models.train as train_module

    saved_models_dir = str(tmp_path / "saved_models")
    monkeypatch.setattr(train_module, "SAVED_MODELS_DIR", saved_models_dir)

    engine = get_engine(":memory:")
    init_db(engine)  # tables exist, but nothing was ever backfilled for Mumbai

    results = train_module.train_and_evaluate("Mumbai", engine=engine)

    assert results == {}
    assert not os.path.exists(saved_models_dir)


def test_train_and_evaluate_returns_empty_when_tables_do_not_exist(tmp_path, monkeypatch):
    import models.train as train_module

    saved_models_dir = str(tmp_path / "saved_models")
    monkeypatch.setattr(train_module, "SAVED_MODELS_DIR", saved_models_dir)

    # A fresh engine whose schema was never created: train_and_evaluate must
    # init it itself rather than raising "no such table: weather_data".
    engine = get_engine(str(tmp_path / "fresh.db"))

    results = train_module.train_and_evaluate("Delhi", engine=engine)

    assert results == {}
    assert not os.path.exists(saved_models_dir)


def test_train_and_evaluate_returns_empty_when_database_is_unusable(tmp_path, monkeypatch):
    import models.train as train_module

    saved_models_dir = str(tmp_path / "saved_models")
    monkeypatch.setattr(train_module, "SAVED_MODELS_DIR", saved_models_dir)

    class _BrokenEngine:
        pass

    results = train_module.train_and_evaluate("Delhi", engine=_BrokenEngine())

    assert results == {}
    assert not os.path.exists(saved_models_dir)
