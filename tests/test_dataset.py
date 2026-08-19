import pandas as pd

from models.dataset import (
    FEATURE_COLUMNS,
    TARGET_HORIZONS,
    add_target_columns,
    baseline_predict,
    build_training_frame,
    chronological_split,
)


def _sample_raw_df(n=40):
    timestamps = pd.date_range("2026-05-01 00:00", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "location": ["Delhi"] * n,
        "temperature": [30.0 + i * 0.1 for i in range(n)],
        "feels_like": [32.0] * n,
        "humidity": [55.0] * n,
        "pressure": [1005.0] * n,
        "wind_speed": [10.0] * n,
        "wind_direction": [200.0] * n,
        "rainfall": [0.0] * n,
        "visibility": [None] * n,
        "cloud_cover": [20.0] * n,
        "uv_index": [None] * n,
        "pm25": [80.0 + i for i in range(n)],
        "pm10": [130.0 + i for i in range(n)],
        "co": [450.0] * n,
        "no2": [30.0] * n,
        "so2": [8.0] * n,
        "o3": [35.0] * n,
        "aqi": [100 + i * 2 for i in range(n)],
    })


def test_add_target_columns_shifts_future_aqi_backward():
    df = add_target_columns(_sample_raw_df())

    assert df["aqi_target_1h"].iloc[0] == df["aqi"].iloc[1]
    assert df["aqi_target_3h"].iloc[0] == df["aqi"].iloc[3]
    assert df["aqi_target_6h"].iloc[0] == df["aqi"].iloc[6]
    assert pd.isna(df["aqi_target_6h"].iloc[-1])


def test_build_training_frame_drops_rows_with_missing_required_values():
    frame = build_training_frame(_sample_raw_df())

    assert len(frame) > 0
    assert len(frame) < 40  # warmup (lags/rolling) and lookahead (targets) both trim rows
    for column in FEATURE_COLUMNS:
        assert frame[column].isna().sum() == 0
    for label in TARGET_HORIZONS:
        assert frame[f"aqi_target_{label}"].isna().sum() == 0


def test_chronological_split_is_contiguous_and_unshuffled():
    frame = build_training_frame(_sample_raw_df())

    train_df, val_df, test_df = chronological_split(frame)

    assert len(train_df) + len(val_df) + len(test_df) == len(frame)
    assert list(train_df["timestamp"]) == sorted(train_df["timestamp"])
    if len(train_df) and len(val_df):
        assert train_df["timestamp"].max() <= val_df["timestamp"].min()
    if len(val_df) and len(test_df):
        assert val_df["timestamp"].max() <= test_df["timestamp"].min()


def test_baseline_predict_returns_rolling_average_column():
    frame = build_training_frame(_sample_raw_df())

    predicted = baseline_predict(frame)

    assert (predicted == frame["aqi_rolling_3"]).all()
