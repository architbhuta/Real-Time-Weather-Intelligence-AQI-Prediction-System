import pandas as pd
import pytest

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


def test_build_features_accepts_a_single_location_frame():
    df = _hourly_df()
    df["pm10"] = df["pm25"] + 20
    df["location"] = "Delhi"

    result = build_features(df)

    assert result["aqi_lag_1"].iloc[1] == 100


def test_build_features_rejects_multi_location_frame():
    delhi = _hourly_df()
    delhi["pm10"] = delhi["pm25"] + 20
    delhi["location"] = "Delhi"
    mumbai = delhi.copy()
    mumbai["location"] = "Mumbai"
    multi_city = pd.concat([delhi, mumbai], ignore_index=True)

    with pytest.raises(AssertionError, match="single-location"):
        build_features(multi_city)


def _with_pm10(df):
    df["pm10"] = df["pm25"] + 20
    return df


def test_build_features_lags_are_time_based_across_a_gap():
    df = _with_pm10(_hourly_df(10))
    df = df.drop(index=[4, 5, 6]).reset_index(drop=True)  # hours 04-06 missing

    result = build_features(df).set_index("timestamp")

    # 07:00's previous hour (06:00) is missing, not the 03:00 row that precedes it positionally
    assert pd.isna(result.loc["2026-08-19 07:00", "aqi_lag_1"])
    # 09:00's lag_3 is 06:00 (missing); lag_6 is 03:00 (present)
    assert pd.isna(result.loc["2026-08-19 09:00", "aqi_lag_3"])
    assert result.loc["2026-08-19 09:00", "aqi_lag_6"] == 130


def test_build_features_collapses_sub_hourly_readings_to_one_per_hour():
    df = _with_pm10(_hourly_df(4))
    extra = df.iloc[[3]].copy()
    extra["timestamp"] = extra["timestamp"] + pd.Timedelta(minutes=30)
    extra["aqi"] = 999
    result = build_features(pd.concat([df, extra], ignore_index=True))

    assert list(result["timestamp"]) == list(pd.date_range("2026-08-19 00:00", periods=4, freq="h"))
    assert result["aqi"].iloc[3] == 999  # latest reading within the hour wins
    assert result["aqi_lag_1"].iloc[3] == 120
