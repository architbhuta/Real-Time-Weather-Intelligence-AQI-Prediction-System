import pandas as pd

from data.feature_engineering import add_lag_features, add_rolling_features, add_time_features, build_features


def _hourly_df(n=10):
    timestamps = pd.date_range("2026-08-19 00:00", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "aqi": [100 + i * 10 for i in range(n)],
        "pm25": [50 + i * 5 for i in range(n)],
    })


def test_add_time_features():
    df = add_time_features(_hourly_df())

    assert list(df["hour"][:3]) == [0, 1, 2]
    assert "day_of_week" in df.columns
    assert "is_weekend" in df.columns


def test_add_lag_features_shifts_correctly():
    df = add_lag_features(_hourly_df(), columns=["aqi"], lags=[1])

    assert pd.isna(df["aqi_lag_1"].iloc[0])
    assert df["aqi_lag_1"].iloc[1] == 100  # aqi at row 0


def test_add_rolling_features_requires_full_window():
    df = add_rolling_features(_hourly_df(), columns=["aqi"], windows=[3])

    assert pd.isna(df["aqi_rolling_3"].iloc[1])  # only 2 points so far
    assert df["aqi_rolling_3"].iloc[2] == (100 + 110 + 120) / 3


def test_build_features_sorts_by_timestamp_before_computing():
    df = _hourly_df()
    df["pm10"] = df["pm25"] + 20
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)

    result = build_features(shuffled)

    assert list(result["timestamp"]) == list(df["timestamp"])
    assert result["aqi_lag_1"].iloc[1] == 100
