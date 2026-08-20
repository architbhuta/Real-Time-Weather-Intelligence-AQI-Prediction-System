from data.aqi_calculator import aqi_category
from data.database import get_engine, get_latest_reading, init_db
from collect import collect_for_location
import streamlit as st

from visualization.dashboard_common import render_location_selector

st.set_page_config(page_title="Live Overview", layout="wide")

location = render_location_selector()
engine = get_engine()
init_db(engine)

st.title(f"Live Overview — {location}")

if st.sidebar.button("Refresh Live Data"):
    with st.spinner("Fetching live data..."):
        success = collect_for_location(location, engine=engine)
    if success:
        st.sidebar.success("Live data refreshed.")
    else:
        st.sidebar.warning("Live data temporarily unavailable. Showing the most recent stored data.")

reading = get_latest_reading(engine, location)

if reading is None:
    st.info(f"No data yet for {location}. Click 'Refresh Live Data' in the sidebar to fetch current conditions.")
else:
    st.caption(f"Last updated: {reading['timestamp']}")

    weather_cols = st.columns(5)
    weather_cols[0].metric("Temperature", f"{reading['temperature']:.1f} C" if reading["temperature"] is not None else "N/A")
    weather_cols[1].metric("Humidity", f"{reading['humidity']:.0f}%" if reading["humidity"] is not None else "N/A")
    weather_cols[2].metric("Wind Speed", f"{reading['wind_speed']:.1f} km/h" if reading["wind_speed"] is not None else "N/A")
    weather_cols[3].metric("Pressure", f"{reading['pressure']:.0f} hPa" if reading["pressure"] is not None else "N/A")
    weather_cols[4].metric("Rainfall", f"{reading['rainfall']:.1f} mm" if reading["rainfall"] is not None else "N/A")

    aqi = reading.get("aqi")
    if aqi is not None:
        st.subheader(f"Air Quality Index: {int(aqi)} — {aqi_category(int(aqi))}")
    else:
        st.subheader("Air Quality Index: Not available")

    pollutant_cols = st.columns(6)
    pollutant_cols[0].metric("PM2.5", f"{reading['pm25']:.1f}" if reading["pm25"] is not None else "N/A")
    pollutant_cols[1].metric("PM10", f"{reading['pm10']:.1f}" if reading["pm10"] is not None else "N/A")
    pollutant_cols[2].metric("NO2", f"{reading['no2']:.1f}" if reading["no2"] is not None else "N/A")
    pollutant_cols[3].metric("SO2", f"{reading['so2']:.1f}" if reading["so2"] is not None else "N/A")
    pollutant_cols[4].metric("CO", f"{reading['co']:.1f}" if reading["co"] is not None else "N/A")
    pollutant_cols[5].metric("O3", f"{reading['o3']:.1f}" if reading["o3"] is not None else "N/A")
