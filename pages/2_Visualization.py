import pandas as pd
import streamlit as st

from data.database import get_engine, init_db, load_weather_and_air_quality
from visualization.charts import aqi_timeseries_chart, pollutant_comparison_chart, scatter_chart
from visualization.dashboard_common import render_location_selector

st.set_page_config(page_title="Visualization", layout="wide")

location = render_location_selector()
engine = get_engine()
init_db(engine)

st.title(f"Live Visualization — {location}")

df = load_weather_and_air_quality(engine, location)

if df.empty:
    st.info(f"No stored history for {location} yet.")
else:
    range_option = st.selectbox(
        "Time range", ["Last 6 hours", "Last 24 hours", "Last 7 days", "Last 30 days", "All available"],
    )
    range_hours = {
        "Last 6 hours": 6, "Last 24 hours": 24, "Last 7 days": 24 * 7,
        "Last 30 days": 24 * 30, "All available": None,
    }[range_option]

    if range_hours is not None:
        cutoff = df["timestamp"].max() - pd.Timedelta(hours=range_hours)
        filtered = df[df["timestamp"] >= cutoff]
    else:
        filtered = df

    if filtered.empty:
        st.warning(f"No data available for '{range_option}'. Showing all available data instead.")
        filtered = df

    st.plotly_chart(aqi_timeseries_chart(filtered), use_container_width=True)
    st.plotly_chart(pollutant_comparison_chart(filtered), use_container_width=True)

    scatter_cols = st.columns(2)
    with scatter_cols[0]:
        st.plotly_chart(scatter_chart(filtered, "temperature", "aqi", "Temperature vs AQI"), use_container_width=True)
    with scatter_cols[1]:
        st.plotly_chart(scatter_chart(filtered, "humidity", "aqi", "Humidity vs AQI"), use_container_width=True)
