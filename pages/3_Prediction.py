import pandas as pd
import streamlit as st

from data.aqi_calculator import aqi_category
from data.database import get_engine, init_db, load_weather_and_air_quality
from data.feature_engineering import build_features
from models.predict import predict_aqi
from visualization.charts import prediction_chart
from visualization.dashboard_common import render_location_selector

st.set_page_config(page_title="AQI Prediction", layout="wide")

location = render_location_selector()
engine = get_engine()
init_db(engine)

st.title(f"AQI Prediction — {location}")

raw_df = load_weather_and_air_quality(engine, location)

if raw_df.empty:
    st.info(f"No stored history for {location} yet. Run the collector or backfill script first.")
else:
    features_df = build_features(raw_df)
    features_df["is_weekend"] = features_df["is_weekend"].astype(int)
    latest = features_df.dropna(subset=["aqi_rolling_3"]).tail(1)

    if latest.empty:
        st.info("Not enough recent history to compute a prediction yet.")
    else:
        current_aqi = raw_df["aqi"].iloc[-1]
        st.metric("Current AQI", "N/A" if pd.isna(current_aqi) else f"{current_aqi:.0f}")

        predictions = {horizon: predict_aqi(horizon, latest) for horizon in ["1h", "3h", "6h"]}

        st.subheader("Predicted AQI")
        pred_cols = st.columns(3)
        labels = {"1h": "+1 Hour", "3h": "+3 Hours", "6h": "+6 Hours"}
        for col, horizon in zip(pred_cols, ["1h", "3h", "6h"]):
            value, model_name = predictions[horizon]
            if value is None:
                col.metric(labels[horizon], "Unavailable")
                col.caption("Insufficient historical data for a reliable prediction.")
            else:
                col.metric(labels[horizon], f"{value:.0f} ({aqi_category(int(value))})")
                if model_name == "baseline":
                    col.caption("Baseline estimate (rolling average) — insufficient data for a trained model.")
                else:
                    col.caption(f"Model: {model_name}")

        st.caption("Predictions are estimates based on historical patterns, not guarantees of future air quality.")

        recent_history = raw_df.tail(48)
        st.plotly_chart(prediction_chart(recent_history, predictions), use_container_width=True)
