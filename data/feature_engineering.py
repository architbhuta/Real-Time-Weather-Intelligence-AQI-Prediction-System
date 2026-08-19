import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    return df


def add_lag_features(df: pd.DataFrame, columns: list[str], lags: list[int]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        for lag in lags:
            df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, columns: list[str], windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        for window in windows:
            df[f"{column}_rolling_{window}"] = (
                df[column].rolling(window=window, min_periods=window).mean()
            )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_time_features(df)
    df = add_lag_features(df, columns=["aqi", "pm25", "pm10"], lags=[1, 3, 6])
    df = add_rolling_features(df, columns=["aqi", "pm25"], windows=[3, 6])
    return df
