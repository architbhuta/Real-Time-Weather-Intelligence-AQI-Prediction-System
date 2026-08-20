import pandas as pd

METRICS = ["pm25", "pm10", "aqi", "temperature", "humidity"]


def compute_iqr_bounds(values: pd.Series) -> tuple[float, float, float]:
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper, iqr


def classify_severity(distance_ratio: float) -> str:
    if distance_ratio < 1.0:
        return "Low"
    if distance_ratio < 3.0:
        return "Medium"
    return "High"


def detect_anomalies_for_metric(df: pd.DataFrame, metric: str, min_history: int = 30) -> list[dict]:
    values = df[metric].dropna()
    if len(values) < min_history:
        return []

    lower, upper, iqr = compute_iqr_bounds(values)
    if iqr == 0:
        return []

    expected = float(values.median())
    anomalies = []
    for idx, value in values.items():
        if value < lower:
            distance_ratio = (lower - value) / iqr
        elif value > upper:
            distance_ratio = (value - upper) / iqr
        else:
            continue

        anomalies.append({
            "timestamp": df.loc[idx, "timestamp"],
            "metric": metric,
            "observed_value": round(float(value), 2),
            "expected_value": round(expected, 2),
            "anomaly_score": round(float(distance_ratio), 2),
            "severity": classify_severity(distance_ratio),
        })
    return anomalies


def detect_anomalies_for_location(df: pd.DataFrame, location: str) -> list[dict]:
    all_anomalies = []
    for metric in METRICS:
        if metric not in df.columns:
            continue
        for record in detect_anomalies_for_metric(df, metric):
            record["location"] = location
            all_anomalies.append(record)
    return all_anomalies
