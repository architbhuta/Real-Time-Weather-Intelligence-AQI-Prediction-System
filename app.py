import streamlit as st

from visualization.dashboard_common import render_location_selector

st.set_page_config(page_title="AirSense", layout="wide")

location = render_location_selector()

st.title("AirSense")
st.write("Real-Time Weather Intelligence & AQI Prediction System")
st.write(
    f"Currently viewing **{location}**. Use the sidebar to switch locations and to navigate "
    "between Live Overview, Visualization, Prediction, Anomalies, and Historical Analytics."
)
st.caption(
    "Predictions are estimates based on historical patterns, not guarantees of future air quality. "
    "Anomalies are statistically unusual readings, not necessarily dangerous ones."
)
