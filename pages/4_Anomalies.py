from data.database import get_engine, get_recent_anomalies, init_db
import streamlit as st

from visualization.charts import anomaly_scatter_chart
from visualization.dashboard_common import render_location_selector

st.set_page_config(page_title="Anomaly Detection", layout="wide")

location = render_location_selector()
engine = get_engine()
init_db(engine)

st.title(f"Anomaly Detection — {location}")
st.caption(
    "Anomalies are statistically unusual readings compared to recent history. "
    "This does not necessarily mean a reading is dangerous."
)

anomalies = get_recent_anomalies(engine, location, limit=100)

if anomalies.empty:
    st.info(f"No anomalies detected for {location} yet. Run the anomaly scan first.")
else:
    st.metric("Total Anomalies (recent)", len(anomalies))

    severity_counts = anomalies["severity"].value_counts()
    severity_cols = st.columns(3)
    severity_cols[0].metric("Low", int(severity_counts.get("Low", 0)))
    severity_cols[1].metric("Medium", int(severity_counts.get("Medium", 0)))
    severity_cols[2].metric("High", int(severity_counts.get("High", 0)))

    st.plotly_chart(anomaly_scatter_chart(anomalies), use_container_width=True)

    st.subheader("Recent Anomalies")
    st.dataframe(anomalies.sort_values("timestamp", ascending=False), use_container_width=True)
