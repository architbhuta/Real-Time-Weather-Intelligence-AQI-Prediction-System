# Streamlit Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 5-page Streamlit dashboard (Live Overview, Visualization, Prediction, Anomalies, Historical Analytics) wired to the already-working backend: `data/database.py`'s read paths, `models/predict.py`, and `anomaly/detector.py`'s stored results.

**Architecture:** A shared data-helpers layer (`data/database.py` additions) and a Plotly chart-builder layer (`visualization/charts.py`) sit underneath 5 Streamlit page scripts under `pages/`, auto-discovered by Streamlit's native multipage routing from `app.py`. A tiny shared module (`visualization/dashboard_common.py`) renders the location selector consistently on every page via `st.session_state`. Given the time constraint on this plan, task granularity is 3 larger tasks rather than 7 small ones — a deliberate speed/quality tradeoff (backend helpers are still fully unit-tested; pages are tested with Streamlit's `AppTest` framework, which runs each page script headlessly and lets us assert on rendered output without a browser).

**Tech Stack:** Streamlit, Plotly (both already in `requirements.txt`, unused until now). `streamlit.testing.v1.AppTest` for page testing (ships with Streamlit >=1.28, already satisfied by the pinned `streamlit>=1.32`).

**Spec:** `docs/superpowers/specs/2026-08-19-aqi-prediction-design.md`

**Prior plans this builds on:** Plan 1 (data pipeline), Plan 2 (ML training/prediction), Plan 3 (anomaly detection) — all merged and pushed. This plan adds no new backend logic beyond two small read-path helpers; everything else is presentation wired to what already exists and is already tested.

## Global Constraints

- Predictions are never presented as guaranteed — every prediction display carries a caption saying so (per the spec).
- A `"baseline"` or `"unavailable"` prediction (from `models.predict.predict_aqi`'s cold-start fallback) must be visibly labeled as such, never presented as if it came from a trained model.
- Anomalies are labeled "statistically unusual," never implied to be dangerous (per Plan 3's established framing).
- Correlation charts carry a "correlation does not imply causation" caption.
- No API keys, no new dependencies beyond what's already in `requirements.txt`.
- Real data only — the dashboard reads from `db/aqi_system.db`, which already has 2232 real backfilled Delhi hours, trained models, and 104 real detected anomalies from Plans 2-3. Page tests run against this real, already-populated database rather than a synthetic in-memory fixture (Streamlit's `AppTest` doesn't accept an injectable engine the way earlier tasks' functions did) — this is a deliberate, disclosed tradeoff for this plan, not an oversight.

---

### Task 1: Dashboard Data Helpers + Chart Builders

**Files:**
- Modify: `data/database.py` (add `get_latest_reading`, `get_recent_anomalies`)
- Create: `visualization/charts.py`
- Test: `tests/test_database.py` (append), `tests/test_charts.py`

**Interfaces:**
- Produces: `get_latest_reading(engine, location: str) -> dict | None` (most recent joined weather+air-quality row as a dict, keyed the same as `load_weather_and_air_quality`'s columns, or `None` if no history); `get_recent_anomalies(engine, location: str, limit: int = 20) -> pd.DataFrame` (columns: `timestamp, metric, observed_value, expected_value, anomaly_score, severity`, newest first); `visualization.charts.aqi_timeseries_chart(df) -> go.Figure`; `pollutant_comparison_chart(df) -> go.Figure`; `scatter_chart(df, x, y, title) -> go.Figure`; `prediction_chart(history_df, predictions: dict[str, tuple[float | None, str]]) -> go.Figure`; `correlation_heatmap(df, columns: list[str]) -> go.Figure`; `anomaly_scatter_chart(df) -> go.Figure`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_database.py
from data.database import get_latest_reading, get_recent_anomalies


def test_get_latest_reading_returns_the_newest_row():
    engine = get_engine(":memory:")
    init_db(engine)
    insert_weather_and_air_quality(engine, {
        "timestamp": datetime(2026, 5, 1, 0, 0), "location": "Delhi",
        "latitude": 28.7041, "longitude": 77.1025, "temperature": 28.0,
        "feels_like": 30.0, "humidity": 50.0, "pressure": 1004.0, "wind_speed": 8.0,
        "wind_direction": 190.0, "rainfall": 0.0, "visibility": None,
        "cloud_cover": 15.0, "uv_index": None,
    }, {
        "timestamp": datetime(2026, 5, 1, 0, 0), "location": "Delhi",
        "pm25": 70.0, "pm10": 120.0, "co": 400.0, "no2": 28.0, "so2": 7.0, "o3": 30.0, "aqi": 140,
    })
    insert_weather_and_air_quality(engine, {
        "timestamp": datetime(2026, 5, 1, 1, 0), "location": "Delhi",
        "latitude": 28.7041, "longitude": 77.1025, "temperature": 29.0,
        "feels_like": 31.0, "humidity": 52.0, "pressure": 1004.5, "wind_speed": 9.0,
        "wind_direction": 195.0, "rainfall": 0.0, "visibility": None,
        "cloud_cover": 18.0, "uv_index": None,
    }, {
        "timestamp": datetime(2026, 5, 1, 1, 0), "location": "Delhi",
        "pm25": 85.0, "pm10": 140.0, "co": 420.0, "no2": 32.0, "so2": 9.0, "o3": 33.0, "aqi": 158,
    })

    reading = get_latest_reading(engine, "Delhi")

    assert reading is not None
    assert reading["aqi"] == 158
    assert reading["timestamp"] == datetime(2026, 5, 1, 1, 0)


def test_get_latest_reading_returns_none_for_unknown_location():
    engine = get_engine(":memory:")
    init_db(engine)

    assert get_latest_reading(engine, "Atlantis") is None


def test_get_recent_anomalies_orders_newest_first_and_respects_limit():
    engine = get_engine(":memory:")
    init_db(engine)
    insert_anomalies(engine, [
        {"timestamp": datetime(2026, 5, 1, 0, 0), "location": "Delhi", "metric": "pm25",
         "observed_value": 200.0, "expected_value": 80.0, "anomaly_score": 1.5, "severity": "Medium"},
        {"timestamp": datetime(2026, 5, 1, 2, 0), "location": "Delhi", "metric": "pm10",
         "observed_value": 300.0, "expected_value": 130.0, "anomaly_score": 2.5, "severity": "High"},
        {"timestamp": datetime(2026, 5, 1, 1, 0), "location": "Delhi", "metric": "aqi",
         "observed_value": 400.0, "expected_value": 150.0, "anomaly_score": 3.0, "severity": "High"},
    ])

    anomalies = get_recent_anomalies(engine, "Delhi", limit=2)

    assert len(anomalies) == 2
    assert list(anomalies["timestamp"]) == [datetime(2026, 5, 1, 2, 0), datetime(2026, 5, 1, 1, 0)]
```

```python
# tests/test_charts.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_database.py tests/test_charts.py -v`
Expected: FAIL — `get_latest_reading`/`get_recent_anomalies` don't exist; `ModuleNotFoundError: No module named 'visualization.charts'`

- [ ] **Step 3: Add the two functions to `data/database.py`**

```python
def get_latest_reading(engine, location: str) -> dict | None:
    df = load_weather_and_air_quality(engine, location)
    if df.empty:
        return None
    return df.iloc[-1].to_dict()


def get_recent_anomalies(engine, location: str, limit: int = 20) -> pd.DataFrame:
    with Session(engine) as session:
        rows = (
            session.query(Anomaly)
            .filter(Anomaly.location == location)
            .order_by(Anomaly.timestamp.desc())
            .limit(limit)
            .all()
        )
    return pd.DataFrame([{
        "timestamp": r.timestamp,
        "metric": r.metric,
        "observed_value": r.observed_value,
        "expected_value": r.expected_value,
        "anomaly_score": r.anomaly_score,
        "severity": r.severity,
    } for r in rows])
```

- [ ] **Step 4: Create `visualization/__init__.py` and write `visualization/charts.py`**

```bash
mkdir -p visualization
touch visualization/__init__.py
```

```python
# visualization/charts.py
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_database.py tests/test_charts.py -v`
Expected: PASS (all tests in both files)

- [ ] **Step 6: Commit**

```bash
git add data/database.py visualization/__init__.py visualization/charts.py tests/test_database.py tests/test_charts.py
git commit -m "feat: add dashboard data helpers and Plotly chart builders"
```

---

### Task 2: App Entry + Live Overview + Visualization Pages

**Files:**
- Create: `visualization/dashboard_common.py`
- Create: `app.py`
- Create: `pages/1_Live_Overview.py`
- Create: `pages/2_Visualization.py`
- Test: `tests/test_pages_overview_and_visualization.py`

**Interfaces:**
- Consumes: `data.database.get_engine`, `init_db`, `load_weather_and_air_quality`, `get_latest_reading` (Task 1); `collect.collect_for_location` (Plan 1); `data.aqi_calculator.aqi_category` (Plan 1); `visualization.charts.aqi_timeseries_chart`, `pollutant_comparison_chart`, `scatter_chart` (Task 1); `utils.config.LOCATIONS`, `DEFAULT_LOCATION`
- Produces: `visualization.dashboard_common.render_location_selector() -> str` (renders a sidebar location dropdown backed by `st.session_state["location"]`, returns the selected location — every page calls this first)

- [ ] **Step 1: Write `visualization/dashboard_common.py`**

```python
import streamlit as st

from utils.config import DEFAULT_LOCATION, LOCATIONS


def render_location_selector() -> str:
    if "location" not in st.session_state:
        st.session_state["location"] = DEFAULT_LOCATION

    options = list(LOCATIONS.keys())
    current = st.session_state["location"]
    index = options.index(current) if current in options else 0

    location = st.sidebar.selectbox("Location", options=options, index=index, key="location_selector")
    st.session_state["location"] = location
    return location
```

- [ ] **Step 2: Write `app.py`**

```python
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
```

- [ ] **Step 3: Write `pages/1_Live_Overview.py`**

```python
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
```

- [ ] **Step 4: Write `pages/2_Visualization.py`**

```python
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
```

- [ ] **Step 5: Write the AppTest-based page tests**

```python
# tests/test_pages_overview_and_visualization.py
from streamlit.testing.v1 import AppTest


def test_app_entry_runs_without_exception():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)

    assert not at.exception
    assert any("AirSense" in str(el.value) for el in at.title)


def test_live_overview_page_runs_without_exception():
    at = AppTest.from_file("pages/1_Live_Overview.py")
    at.run(timeout=30)

    assert not at.exception
    assert any("Live Overview" in str(el.value) for el in at.title)


def test_visualization_page_runs_without_exception():
    at = AppTest.from_file("pages/2_Visualization.py")
    at.run(timeout=30)

    assert not at.exception
    assert any("Live Visualization" in str(el.value) for el in at.title)
```

Note: these tests run against the REAL `db/aqi_system.db` (per this plan's Global Constraints — Streamlit's `AppTest` doesn't accept an injectable engine parameter the way this project's other functions do). This database already has real backfilled Delhi data from Plan 2, so the pages exercise their real data path, not just their empty-state path. If the real database is ever absent (e.g., a fresh clone with no `db/aqi_system.db` yet), these tests should still pass via each page's `if df.empty:` / `if reading is None:` branches — verify this is genuinely true by reasoning through the code, since you can't easily simulate "fresh clone" without moving the real file.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_pages_overview_and_visualization.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add visualization/dashboard_common.py app.py pages/1_Live_Overview.py pages/2_Visualization.py tests/test_pages_overview_and_visualization.py
git commit -m "feat: add dashboard entry point, live overview, and visualization pages"
```

---

### Task 3: Prediction, Anomalies, and Historical Analytics Pages

**Files:**
- Create: `pages/3_Prediction.py`
- Create: `pages/4_Anomalies.py`
- Create: `pages/5_Historical_Analytics.py`
- Test: `tests/test_pages_prediction_anomalies_analytics.py`

**Interfaces:**
- Consumes: `data.database.get_engine`, `init_db`, `load_weather_and_air_quality`, `get_recent_anomalies` (Task 1/Plan 1/2); `data.feature_engineering.build_features` (Plan 1); `models.predict.predict_aqi` (Plan 2); `data.aqi_calculator.aqi_category` (Plan 1); `visualization.charts.prediction_chart`, `anomaly_scatter_chart`, `correlation_heatmap` (Task 1); `visualization.dashboard_common.render_location_selector` (Task 2)

- [ ] **Step 1: Write `pages/3_Prediction.py`**

```python
from data.aqi_calculator import aqi_category
from data.database import get_engine, init_db, load_weather_and_air_quality
from data.feature_engineering import build_features
from models.predict import predict_aqi
import streamlit as st

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
        st.metric("Current AQI", f"{current_aqi:.0f}" if current_aqi is not None else "N/A")

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
```

- [ ] **Step 2: Write `pages/4_Anomalies.py`**

```python
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
```

- [ ] **Step 3: Write `pages/5_Historical_Analytics.py`**

```python
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
```

- [ ] **Step 4: Write the AppTest-based page tests**

```python
# tests/test_pages_prediction_anomalies_analytics.py
from streamlit.testing.v1 import AppTest


def test_prediction_page_runs_without_exception():
    at = AppTest.from_file("pages/3_Prediction.py")
    at.run(timeout=30)

    assert not at.exception
    assert any("AQI Prediction" in str(el.value) for el in at.title)


def test_anomalies_page_runs_without_exception():
    at = AppTest.from_file("pages/4_Anomalies.py")
    at.run(timeout=30)

    assert not at.exception
    assert any("Anomaly Detection" in str(el.value) for el in at.title)


def test_historical_analytics_page_runs_without_exception():
    at = AppTest.from_file("pages/5_Historical_Analytics.py")
    at.run(timeout=30)

    assert not at.exception
    assert any("Historical Analytics" in str(el.value) for el in at.title)
```

Same note as Task 2's test file: these run against the real `db/aqi_system.db`, which has real Delhi data, real trained models, and real detected anomalies — so this exercises the genuine data path for all three pages, not just the empty-state path.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_pages_prediction_anomalies_analytics.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Run the full test suite**

Run: `pytest -q` (from the activated venv)
Expected: all tests pass, no regressions

- [ ] **Step 7: Manual verification — launch the real app in a browser**

```bash
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`). Confirm: the sidebar shows all 5 pages plus the location selector defaulting to Delhi; Live Overview shows real current-ish weather/AQI cards (click "Refresh Live Data" to confirm it fetches live data without crashing); Visualization shows real charts with data across the time-range selector; Prediction shows real predicted AQI values labeled with `random_forest` (not `baseline`, since Delhi has trained models) plus the observed-vs-predicted chart; Anomalies shows the 104 real detected anomalies with a severity breakdown; Historical Analytics shows real daily/day-of-week/hourly charts and a correlation matrix. Switch the location selector to a city other than Delhi (e.g. Mumbai) and confirm the app degrades gracefully — no crash, appropriate "no data yet" / "Unavailable" messaging per page, matching the cold-start requirement. Report what you actually saw, including any visual issues, not just "it worked."

- [ ] **Step 8: Commit**

```bash
git add pages/3_Prediction.py pages/4_Anomalies.py pages/5_Historical_Analytics.py tests/test_pages_prediction_anomalies_analytics.py
git commit -m "feat: add prediction, anomalies, and historical analytics pages"
```

---

## What's next

This plan delivers the full 5-page dashboard (Objectives 2 and 4's UI half) wired to real, already-tested backend data. Still to come, as smaller follow-ups: a README, multi-city backfill/training beyond Delhi, and deployment instructions.
