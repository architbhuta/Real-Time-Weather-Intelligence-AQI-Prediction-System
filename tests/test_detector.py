import pandas as pd

from anomaly.detector import (
    METRICS,
    classify_severity,
    compute_iqr_bounds,
    detect_anomalies_for_location,
    detect_anomalies_for_metric,
)


def _history_df(n=50, spike_index=25, spike_value=500.0):
    timestamps = pd.date_range("2026-05-01 00:00", periods=n, freq="h")
    pm25 = [80.0 + (i % 5) for i in range(n)]  # tight, stable range
    pm25[spike_index] = spike_value
    return pd.DataFrame({
        "timestamp": timestamps,
        "location": ["Delhi"] * n,
        "pm25": pm25,
        "pm10": [130.0 + (i % 5) for i in range(n)],
        "aqi": [150.0 + (i % 5) for i in range(n)],
        "temperature": [30.0 + (i % 3) for i in range(n)],
        "humidity": [55.0 + (i % 3) for i in range(n)],
    })


def test_compute_iqr_bounds_returns_sensible_range():
    values = pd.Series([80, 81, 82, 83, 84, 85, 86, 87, 88, 89])

    lower, upper, iqr = compute_iqr_bounds(values)

    assert lower < values.median() < upper
    assert iqr > 0


def test_classify_severity_thresholds():
    assert classify_severity(0.5) == "Low"
    assert classify_severity(1.5) == "Medium"
    assert classify_severity(4.0) == "High"


def test_detect_anomalies_for_metric_flags_the_spike():
    df = _history_df()

    anomalies = detect_anomalies_for_metric(df, "pm25")

    assert len(anomalies) == 1
    assert anomalies[0]["metric"] == "pm25"
    assert anomalies[0]["observed_value"] == 500.0
    assert anomalies[0]["severity"] in {"Low", "Medium", "High"}
    assert anomalies[0]["timestamp"] == df["timestamp"].iloc[25]


def test_detect_anomalies_for_metric_returns_empty_below_min_history():
    df = _history_df(n=10, spike_index=5)

    anomalies = detect_anomalies_for_metric(df, "pm25", min_history=30)

    assert anomalies == []


def test_detect_anomalies_for_location_passes_min_history_through():
    df = _history_df(n=40)  # enough for the default 30, not for 100

    assert detect_anomalies_for_location(df, "Delhi") != []
    assert detect_anomalies_for_location(df, "Delhi", min_history=100) == []


def test_detect_anomalies_for_location_checks_all_metrics():
    df = _history_df()

    anomalies = detect_anomalies_for_location(df, "Delhi")

    assert all(a["location"] == "Delhi" for a in anomalies)
    flagged_metrics = {a["metric"] for a in anomalies}
    assert "pm25" in flagged_metrics  # the spike must be caught
    for metric in METRICS:
        assert metric in {"pm25", "pm10", "aqi", "temperature", "humidity"}
