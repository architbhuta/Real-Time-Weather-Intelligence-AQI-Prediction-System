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


def to_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex onto a strict hourly grid so row shifts equal clock-hour shifts.

    Readings are floored to the hour (the latest reading within an hour wins),
    and hours with no reading become NaN rows. Without this, a gap in collection
    or a sub-hourly reading would make "1 row ago" mean something other than
    "1 hour ago".
    """
    df = df.sort_values("timestamp").copy()
    df["timestamp"] = df["timestamp"].dt.floor("h")
    df = df.drop_duplicates(subset="timestamp", keep="last")
    df = df.set_index("timestamp").asfreq("h")
    if "location" in df.columns:
        df["location"] = df["location"].ffill()
    return df.reset_index()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build time, lag and rolling features for a SINGLE location's history.

    Lag and rolling features are shifts over an hourly grid, so a multi-city
    frame would mix one city's readings into another's history. The caller must
    filter by location first; that precondition is asserted below rather than
    silently producing corrupt features. Hours with no reading yield NaN lags.
    """
    if "location" in df.columns:
        assert df["location"].nunique() <= 1, (
            "build_features expects a single-location DataFrame; "
            "filter by location before calling"
        )
    df = to_hourly_grid(df)
    df = add_time_features(df)
    df = add_lag_features(df, columns=["aqi", "pm25", "pm10"], lags=[1, 3, 6])
    df = add_rolling_features(df, columns=["aqi", "pm25"], windows=[3, 6])
    return df
