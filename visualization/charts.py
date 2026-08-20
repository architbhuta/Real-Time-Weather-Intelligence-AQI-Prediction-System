import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3"]
HORIZON_HOURS = {"1h": 1, "3h": 3, "6h": 6}


def aqi_timeseries_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.line(df, x="timestamp", y="aqi", title="AQI Over Time")
    fig.update_layout(xaxis_title="Time", yaxis_title="AQI")
    return fig


def pollutant_comparison_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for pollutant in POLLUTANTS:
        if pollutant in df.columns:
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df[pollutant], name=pollutant.upper(), mode="lines"))
    fig.update_layout(title="Pollutant Comparison", xaxis_title="Time", yaxis_title="Concentration")
    return fig


def scatter_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = px.scatter(df, x=x, y=y, title=title)
    return fig


def prediction_chart(history_df: pd.DataFrame, predictions: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df["timestamp"] if "timestamp" in history_df.columns else [],
        y=history_df["aqi"] if "aqi" in history_df.columns else [],
        name="Observed AQI", mode="lines", line=dict(color="royalblue"),
    ))

    if not history_df.empty:
        last_timestamp = history_df["timestamp"].iloc[-1]
        last_aqi = history_df["aqi"].iloc[-1]
        pred_x = [last_timestamp]
        pred_y = [last_aqi]
        for horizon in ["1h", "3h", "6h"]:
            value, _model_name = predictions.get(horizon, (None, None))
            if value is not None:
                pred_x.append(last_timestamp + pd.Timedelta(hours=HORIZON_HOURS[horizon]))
                pred_y.append(value)
        if len(pred_x) > 1:
            fig.add_trace(go.Scatter(
                x=pred_x, y=pred_y, name="Predicted AQI",
                mode="lines+markers", line=dict(color="orange", dash="dash"),
            ))

    fig.update_layout(title="AQI: Observed vs Predicted", xaxis_title="Time", yaxis_title="AQI")
    return fig


def correlation_heatmap(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    corr = df[columns].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    return fig


def anomaly_scatter_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="timestamp", y="observed_value", color="severity", symbol="metric",
        title="Anomalies Over Time", hover_data=["metric", "expected_value", "anomaly_score"],
    )
    return fig
