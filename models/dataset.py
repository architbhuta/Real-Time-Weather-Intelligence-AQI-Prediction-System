import pandas as pd

from data.feature_engineering import build_features

TARGET_HORIZONS = {"1h": 1, "3h": 3, "6h": 6}

FEATURE_COLUMNS = [
    "hour", "day_of_week", "month", "is_weekend",
    "temperature", "humidity", "pressure", "wind_speed", "rainfall", "cloud_cover",
    "pm25", "pm10",
    "aqi_lag_1", "aqi_lag_3", "aqi_lag_6",
    "pm25_lag_1", "pm25_lag_3", "pm25_lag_6",
    "pm10_lag_1", "pm10_lag_3", "pm10_lag_6",
    "aqi_rolling_3", "aqi_rolling_6",
    "pm25_rolling_3", "pm25_rolling_6",
]


def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for label, hours in TARGET_HORIZONS.items():
        df[f"aqi_target_{label}"] = df["aqi"].shift(-hours)
    return df


def build_training_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = build_features(raw_df)
    df["is_weekend"] = df["is_weekend"].astype(int)
    df = add_target_columns(df)

    required_columns = FEATURE_COLUMNS + [f"aqi_target_{label}" for label in TARGET_HORIZONS]
    df = df.dropna(subset=required_columns)
    return df.reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


def baseline_predict(df: pd.DataFrame) -> pd.Series:
    return df["aqi_rolling_3"]
