from data.database import get_engine, init_db, load_weather_and_air_quality
import streamlit as st

from visualization.charts import correlation_heatmap
from visualization.dashboard_common import render_location_selector

st.set_page_config(page_title="Historical Analytics", layout="wide")

location = render_location_selector()
engine = get_engine()
init_db(engine)

st.title(f"Historical Analytics — {location}")

df = load_weather_and_air_quality(engine, location)

if df.empty:
    st.info(f"No stored history for {location} yet.")
else:
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["hour"] = df["timestamp"].dt.hour

    st.subheader("Daily Average AQI")
    daily = df.groupby("date")["aqi"].mean()
    st.line_chart(daily)

    st.subheader("Average AQI by Day of Week")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_day = df.groupby("day_of_week")["aqi"].mean().reindex(day_order)
    st.bar_chart(by_day)

    st.subheader("Average AQI by Hour of Day")
    by_hour = df.groupby("hour")["aqi"].mean()
    st.bar_chart(by_hour)

    peak_hour = int(by_hour.idxmax())
    lowest_hour = int(by_hour.idxmin())
    st.write(
        f"Highest average AQI observed around **{peak_hour}:00**, "
        f"lowest average AQI around **{lowest_hour}:00** "
        f"(based on {len(df)} stored readings)."
    )

    st.subheader("Correlation Matrix")
    numeric_columns = ["aqi", "pm25", "pm10", "temperature", "humidity", "wind_speed", "pressure", "rainfall"]
    available_columns = [c for c in numeric_columns if c in df.columns]
    st.plotly_chart(correlation_heatmap(df, available_columns), use_container_width=True)
    st.caption("Correlation does not imply causation.")
