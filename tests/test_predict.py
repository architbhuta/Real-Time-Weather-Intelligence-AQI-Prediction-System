import os

import joblib
import pandas as pd
import pytest

from models.dataset import FEATURE_COLUMNS


def _feature_row(aqi_rolling_3=120.0):
    row = {column: 1.0 for column in FEATURE_COLUMNS}
    row["aqi_rolling_3"] = aqi_rolling_3
    return pd.DataFrame([row])


class _DummyModel:
    """Module-level (not test-local) so joblib/pickle can serialize it."""

    def predict(self, X):
        return [42.0] * len(X)


def test_predict_aqi_falls_back_to_baseline_when_no_model_saved(tmp_path, monkeypatch):
    import models.predict as predict_module

    monkeypatch.setattr(predict_module, "SAVED_MODELS_DIR", str(tmp_path))

    predicted, model_name = predict_module.predict_aqi("1h", _feature_row(aqi_rolling_3=133.0))

    assert predicted == 133.0
    assert model_name == "baseline"


def test_predict_aqi_returns_unavailable_when_baseline_is_nan(tmp_path, monkeypatch):
    import models.predict as predict_module

    monkeypatch.setattr(predict_module, "SAVED_MODELS_DIR", str(tmp_path))

    # Fewer than 3 hours of history: the rolling-3h baseline is NaN.
    predicted, model_name = predict_module.predict_aqi("1h", _feature_row(aqi_rolling_3=float("nan")))

    assert predicted is None
    assert model_name == "unavailable"


def test_predict_aqi_returns_unavailable_for_empty_or_incomplete_feature_row(tmp_path, monkeypatch):
    import models.predict as predict_module

    monkeypatch.setattr(predict_module, "SAVED_MODELS_DIR", str(tmp_path))

    assert predict_module.predict_aqi("1h", pd.DataFrame()) == (None, "unavailable")

    row_without_rolling = _feature_row().drop(columns=["aqi_rolling_3"])
    assert predict_module.predict_aqi("1h", row_without_rolling) == (None, "unavailable")


def test_predict_aqi_uses_saved_model_when_present(tmp_path, monkeypatch):
    import models.predict as predict_module

    monkeypatch.setattr(predict_module, "SAVED_MODELS_DIR", str(tmp_path))

    joblib.dump(_DummyModel(), os.path.join(str(tmp_path), "aqi_3h.joblib"))
    with open(os.path.join(str(tmp_path), "aqi_3h_model_name.txt"), "w") as f:
        f.write("random_forest")

    predicted, model_name = predict_module.predict_aqi("3h", _feature_row())

    assert predicted == 42.0
    assert model_name == "random_forest"


def test_load_model_returns_none_when_only_one_of_the_pair_exists(tmp_path, monkeypatch):
    import models.predict as predict_module

    monkeypatch.setattr(predict_module, "SAVED_MODELS_DIR", str(tmp_path))
    joblib.dump(object(), os.path.join(str(tmp_path), "aqi_6h.joblib"))
    # no matching aqi_6h_model_name.txt written

    assert predict_module.load_model("6h") is None
