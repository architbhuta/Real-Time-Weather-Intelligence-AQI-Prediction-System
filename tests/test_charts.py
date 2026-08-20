import pandas as pd
import plotly.graph_objects as go

from visualization.charts import (
    aqi_timeseries_chart,
    anomaly_scatter_chart,
    correlation_heatmap,
    pollutant_comparison_chart,
    prediction_chart,
    scatter_chart,
)


def _sample_history(n=10):
    timestamps = pd.date_range("2026-05-01 00:00", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "aqi": [100 + i for i in range(n)],
        "pm25": [60 + i for i in range(n)],
        "pm10": [110 + i for i in range(n)],
        "temperature": [28 + i * 0.1 for i in range(n)],
        "humidity": [55 - i * 0.5 for i in range(n)],
        "wind_speed": [10.0] * n,
        "pressure": [1005.0] * n,
        "rainfall": [0.0] * n,
    })


def test_aqi_timeseries_chart_returns_figure_with_data():
    fig = aqi_timeseries_chart(_sample_history())

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    assert list(fig.data[0].y) == list(_sample_history()["aqi"])


def test_pollutant_comparison_chart_includes_pm25_and_pm10():
    fig = pollutant_comparison_chart(_sample_history())

    assert isinstance(fig, go.Figure)
    trace_names = {trace.name for trace in fig.data}
    assert "PM25" in trace_names
    assert "PM10" in trace_names


def test_scatter_chart_returns_figure():
    fig = scatter_chart(_sample_history(), "temperature", "aqi", "Temperature vs AQI")

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Temperature vs AQI"


def test_prediction_chart_skips_unavailable_horizons():
    history = _sample_history()
    predictions = {
        "1h": (160.0, "random_forest"),
        "3h": (None, "unavailable"),
        "6h": (175.0, "random_forest"),
    }

    fig = prediction_chart(history, predictions)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # observed line + predicted line
    predicted_trace = fig.data[1]
    assert len(predicted_trace.y) == 3  # last observed point + 1h + 6h (3h skipped)


def test_prediction_chart_handles_empty_history():
    fig = prediction_chart(pd.DataFrame(columns=["timestamp", "aqi"]), {"1h": (None, "unavailable")})

    assert isinstance(fig, go.Figure)


def test_correlation_heatmap_returns_figure():
    fig = correlation_heatmap(_sample_history(), ["aqi", "pm25", "pm10", "temperature"])

    assert isinstance(fig, go.Figure)


def test_anomaly_scatter_chart_returns_figure():
    anomalies = pd.DataFrame({
        "timestamp": pd.date_range("2026-05-01", periods=3, freq="h"),
        "metric": ["pm25", "pm10", "aqi"],
        "observed_value": [200.0, 300.0, 400.0],
        "expected_value": [80.0, 130.0, 150.0],
        "anomaly_score": [1.5, 2.5, 3.0],
        "severity": ["Medium", "High", "High"],
    })

    fig = anomaly_scatter_chart(anomalies)

    assert isinstance(fig, go.Figure)
