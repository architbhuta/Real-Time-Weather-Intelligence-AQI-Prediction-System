# Anomaly Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect statistically unusual pollution/weather readings (PM2.5, PM10, AQI, temperature, humidity) for a location using an IQR-based method, and store flagged anomalies in Plan 1's `anomalies` table.

**Architecture:** `anomaly/detector.py` reads a location's history via Plan 2's `data.database.load_weather_and_air_quality`, computes IQR bounds per metric against the full historical distribution, flags any value outside the bounds with a severity graded by how far outside, and returns anomaly records. A new `insert_anomalies` function in `data/database.py` stores them idempotently (a `UniqueConstraint` on `(timestamp, location, metric)` makes re-running the scan a safe no-op on already-flagged points). `anomaly_scan.py` at the repo root wires detection to storage for one location, mirroring the `collect.py`/`backfill.py` pattern.

**Tech Stack:** pandas (already installed) — no new dependency. IQR is chosen over Isolation Forest for this plan: the spec explicitly offers either as "simple and explainable," and IQR maps naturally onto the `anomalies` table's per-metric `observed_value`/`expected_value`/`anomaly_score` columns without needing a persisted model, matching this project's "don't over-engineer" principle.

**Spec:** `docs/superpowers/specs/2026-08-19-aqi-prediction-design.md`

**Prior plans this builds on:** `docs/superpowers/plans/2026-08-19-foundation-and-data-pipeline.md` (Plan 1 — `data/database.py`'s `Anomaly` model already exists with the right columns), `docs/superpowers/plans/2026-08-20-historical-backfill-and-ml.md` (Plan 2 — `load_weather_and_air_quality` already exists and has real Delhi data to detect against).

## Global Constraints

- Anomalies are **statistically unusual, not necessarily dangerous** — per the spec, never claim an anomaly is a hazard; severity describes how extreme the deviation is, not real-world danger.
- Detection is idempotent: re-running the scan on the same history must not create duplicate rows for the same `(timestamp, location, metric)`.
- No API keys, no new dependencies.
- Metrics checked: `pm25`, `pm10`, `aqi`, `temperature`, `humidity` — per the spec's explicit list in the anomaly detection section.

---

### Task 1: Anomaly Table Dedup + Storage

**Files:**
- Modify: `data/database.py` (add `UniqueConstraint` to `Anomaly`, add `insert_anomalies`)
- Test: `tests/test_database.py` (append)

**Interfaces:**
- Produces: `insert_anomalies(engine, records: list[dict]) -> int` — each record has keys `timestamp` (datetime), `location`, `metric`, `observed_value`, `expected_value`, `anomaly_score`, `severity`; returns the count of rows actually stored (duplicates silently skipped, matching Plan 1's `insert_weather_record`/`insert_air_quality_record` pattern)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_database.py
from data.database import Anomaly, insert_anomalies


def _anomaly_record(timestamp, metric="pm25"):
    return {
        "timestamp": timestamp, "location": "Delhi", "metric": metric,
        "observed_value": 250.0, "expected_value": 90.0,
        "anomaly_score": 2.3, "severity": "Medium",
    }


def test_insert_anomalies_stores_new_records():
    engine = get_engine(":memory:")
    init_db(engine)

    stored = insert_anomalies(engine, [
        _anomaly_record(datetime(2026, 5, 1, 0, 0)),
        _anomaly_record(datetime(2026, 5, 1, 1, 0)),
    ])

    assert stored == 2
    with Session(engine) as session:
        assert session.query(Anomaly).count() == 2


def test_insert_anomalies_skips_duplicates_on_rerun():
    engine = get_engine(":memory:")
    init_db(engine)

    first = insert_anomalies(engine, [_anomaly_record(datetime(2026, 5, 1, 0, 0))])
    second = insert_anomalies(engine, [_anomaly_record(datetime(2026, 5, 1, 0, 0))])

    assert first == 1
    assert second == 0
    with Session(engine) as session:
        assert session.query(Anomaly).count() == 1


def test_insert_anomalies_allows_different_metrics_at_same_timestamp():
    engine = get_engine(":memory:")
    init_db(engine)

    stored = insert_anomalies(engine, [
        _anomaly_record(datetime(2026, 5, 1, 0, 0), metric="pm25"),
        _anomaly_record(datetime(2026, 5, 1, 0, 0), metric="temperature"),
    ])

    assert stored == 2
```

Note: `tests/test_database.py` already imports `datetime`, `get_engine`, `init_db`, `Session` from Plan 1/2 — only the `Anomaly`/`insert_anomalies` import shown above is new (merge into the existing `from data.database import (...)` line).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_database.py -v`
Expected: FAIL — `insert_anomalies` does not exist (`ImportError`)

- [ ] **Step 3: Add the `UniqueConstraint` and `insert_anomalies` to `data/database.py`**

Modify the existing `Anomaly` class to add `__table_args__`:

```python
class Anomaly(Base):
    __tablename__ = "anomalies"
    __table_args__ = (UniqueConstraint("timestamp", "location", "metric", name="uq_anomaly_timestamp_location_metric"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    location = Column(String, nullable=False)
    metric = Column(String, nullable=False)
    observed_value = Column(Float, nullable=False)
    expected_value = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    severity = Column(String, nullable=False)
```

(This only adds `__table_args__` — the column definitions are unchanged from Plan 1. `UniqueConstraint` is already imported in this file from Plan 1's earlier fix wave; if it isn't, add it to the existing `from sqlalchemy import (...)` line.)

Add the insert function, following the existing `IntegrityError`-catch pattern already used by `insert_weather_record`/`insert_air_quality_record`/`insert_weather_and_air_quality` in this file:

```python
def insert_anomalies(engine, records: list[dict]) -> int:
    stored = 0
    for record in records:
        with Session(engine) as session:
            try:
                session.add(Anomaly(**record))
                session.commit()
                stored += 1
            except IntegrityError:
                session.rollback()
    return stored
```

(`IntegrityError` is already imported in this file from Plan 1's fix wave.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_database.py -v`
Expected: PASS (all tests in the file, including the 3 new ones)

- [ ] **Step 5: Commit**

```bash
git add data/database.py tests/test_database.py
git commit -m "feat: add anomaly storage with per-metric dedup"
```

---

### Task 2: IQR-Based Anomaly Detector

**Files:**
- Create: `anomaly/detector.py`
- Test: `tests/test_detector.py`

**Interfaces:**
- Consumes: nothing from earlier tasks directly (pure pandas module, tested with synthetic data) — but designed to be called with the DataFrame `data.database.load_weather_and_air_quality` (Plan 2) returns
- Produces: `METRICS: list[str]` (`["pm25", "pm10", "aqi", "temperature", "humidity"]`); `compute_iqr_bounds(values: pd.Series) -> tuple[float, float, float]` (returns `(lower_bound, upper_bound, iqr)`); `classify_severity(distance_ratio: float) -> str` (`"Low"` if `< 1.0`, `"Medium"` if `< 3.0`, else `"High"`); `detect_anomalies_for_metric(df: pd.DataFrame, metric: str, min_history: int = 30) -> list[dict]`; `detect_anomalies_for_location(df: pd.DataFrame, location: str) -> list[dict]` (each record has keys `timestamp`, `location`, `metric`, `observed_value`, `expected_value`, `anomaly_score`, `severity` — ready for `insert_anomalies`)

- [ ] **Step 1: Write the failing test**

```python
# anomaly/__init__.py — empty file, makes anomaly a package
```

```python
# tests/test_detector.py
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
    df = _history_df(n=10)

    anomalies = detect_anomalies_for_metric(df, "pm25", min_history=30)

    assert anomalies == []


def test_detect_anomalies_for_location_checks_all_metrics():
    df = _history_df()

    anomalies = detect_anomalies_for_location(df, "Delhi")

    assert all(a["location"] == "Delhi" for a in anomalies)
    flagged_metrics = {a["metric"] for a in anomalies}
    assert "pm25" in flagged_metrics  # the spike must be caught
    for metric in METRICS:
        assert metric in {"pm25", "pm10", "aqi", "temperature", "humidity"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'anomaly'`

- [ ] **Step 3: Create the `anomaly` package and write `anomaly/detector.py`**

```bash
mkdir -p anomaly
touch anomaly/__init__.py
```

```python
# anomaly/detector.py
import pandas as pd

METRICS = ["pm25", "pm10", "aqi", "temperature", "humidity"]


def compute_iqr_bounds(values: pd.Series) -> tuple[float, float, float]:
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper, iqr


def classify_severity(distance_ratio: float) -> str:
    if distance_ratio < 1.0:
        return "Low"
    if distance_ratio < 3.0:
        return "Medium"
    return "High"


def detect_anomalies_for_metric(df: pd.DataFrame, metric: str, min_history: int = 30) -> list[dict]:
    values = df[metric].dropna()
    if len(values) < min_history:
        return []

    lower, upper, iqr = compute_iqr_bounds(values)
    if iqr == 0:
        return []

    expected = float(values.median())
    anomalies = []
    for idx, value in values.items():
        if value < lower:
            distance_ratio = (lower - value) / iqr
        elif value > upper:
            distance_ratio = (value - upper) / iqr
        else:
            continue

        anomalies.append({
            "timestamp": df.loc[idx, "timestamp"],
            "metric": metric,
            "observed_value": round(float(value), 2),
            "expected_value": round(expected, 2),
            "anomaly_score": round(float(distance_ratio), 2),
            "severity": classify_severity(distance_ratio),
        })
    return anomalies


def detect_anomalies_for_location(df: pd.DataFrame, location: str) -> list[dict]:
    all_anomalies = []
    for metric in METRICS:
        if metric not in df.columns:
            continue
        for record in detect_anomalies_for_metric(df, metric):
            record["location"] = location
            all_anomalies.append(record)
    return all_anomalies
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_detector.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add anomaly/__init__.py anomaly/detector.py tests/test_detector.py
git commit -m "feat: add IQR-based anomaly detector"
```

---

### Task 3: Anomaly Scan Script

**Files:**
- Create: `anomaly_scan.py`
- Test: `tests/test_anomaly_scan.py`

**Interfaces:**
- Consumes: `data.database.get_engine`, `data.database.init_db`, `data.database.load_weather_and_air_quality` (Plan 2), `data.database.insert_anomalies` (Task 1), `anomaly.detector.detect_anomalies_for_location` (Task 2), `utils.config.LOCATIONS`, `utils.config.DEFAULT_LOCATION`
- Produces: `scan_location(location: str, engine=None) -> int` (returns the count of anomalies newly stored; never raises)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_anomaly_scan.py
from datetime import datetime

import pandas as pd

from data.database import AirQualityData, WeatherData, get_engine, init_db
from sqlalchemy.orm import Session


def _seed_history_with_spike(engine, location="Delhi", n=50, spike_index=25):
    timestamps = pd.date_range("2026-05-01 00:00", periods=n, freq="h")
    with Session(engine) as session:
        for i, ts in enumerate(timestamps):
            pm25 = 80.0 + (i % 5)
            if i == spike_index:
                pm25 = 500.0
            session.add(WeatherData(
                timestamp=ts.to_pydatetime(), location=location, latitude=28.7041, longitude=77.1025,
                temperature=30.0, feels_like=32.0, humidity=55.0, pressure=1005.0,
                wind_speed=10.0, wind_direction=200.0, rainfall=0.0, visibility=None,
                cloud_cover=20.0, uv_index=None,
            ))
            session.add(AirQualityData(
                timestamp=ts.to_pydatetime(), location=location,
                pm25=pm25, pm10=130.0 + (i % 5), co=450.0, no2=30.0, so2=8.0, o3=35.0,
                aqi=150 + (i % 5),
            ))
        session.commit()


def test_scan_location_stores_detected_anomalies():
    from anomaly_scan import scan_location

    engine = get_engine(":memory:")
    init_db(engine)
    _seed_history_with_spike(engine)

    stored = scan_location("Delhi", engine=engine)

    assert stored > 0


def test_scan_location_is_idempotent_on_rerun():
    from anomaly_scan import scan_location

    engine = get_engine(":memory:")
    init_db(engine)
    _seed_history_with_spike(engine)

    first = scan_location("Delhi", engine=engine)
    second = scan_location("Delhi", engine=engine)

    assert first > 0
    assert second == 0  # same history, already-flagged points skipped


def test_scan_location_returns_zero_for_unknown_location():
    from anomaly_scan import scan_location

    engine = get_engine(":memory:")
    init_db(engine)

    stored = scan_location("Atlantis", engine=engine)

    assert stored == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_anomaly_scan.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'anomaly_scan'`

- [ ] **Step 3: Write `anomaly_scan.py`**

```python
import sys

from anomaly.detector import detect_anomalies_for_location
from data.database import get_engine, init_db, insert_anomalies, load_weather_and_air_quality
from utils.config import DEFAULT_LOCATION, LOCATIONS
from utils.logging_config import get_logger

logger = get_logger(__name__)


def scan_location(location: str, engine=None) -> int:
    if location not in LOCATIONS:
        logger.error("Unknown location '%s'", location)
        return 0

    engine = engine or get_engine()
    try:
        init_db(engine)
        df = load_weather_and_air_quality(engine, location)
    except Exception as exc:
        logger.error("Could not load history for %s: %s", location, exc)
        return 0

    if df.empty:
        logger.warning("No stored history for %s; skipping anomaly scan", location)
        return 0

    anomalies = detect_anomalies_for_location(df, location)
    if not anomalies:
        logger.info("No anomalies detected for %s", location)
        return 0

    stored = insert_anomalies(engine, anomalies)
    logger.info("Anomaly scan for %s: %d flagged, %d newly stored", location, len(anomalies), stored)
    return stored


if __name__ == "__main__":
    location = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOCATION
    scan_location(location)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_anomaly_scan.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full test suite**

Run: `pytest -q` (from the activated venv)
Expected: all tests pass, no regressions

- [ ] **Step 6: Manual verification against the real backfilled Delhi data**

```bash
python anomaly_scan.py Delhi
sqlite3 db/aqi_system.db "select metric, count(*), severity from anomalies group by metric, severity;"
python anomaly_scan.py Delhi
sqlite3 db/aqi_system.db "select count(*) from anomalies;"
```

Expected: the first run stores some number of anomalies (report the actual counts and which metrics/severities showed up — Delhi's real PM2.5/AQI data over 92 days likely has genuine outliers worth reporting honestly, don't fabricate); the second run reports 0 newly stored (idempotent) and the total row count in `anomalies` is unchanged between the two `select count(*)` checks around it.

- [ ] **Step 7: Commit**

```bash
git add anomaly_scan.py tests/test_anomaly_scan.py
git commit -m "feat: add anomaly scan script wiring detection to storage"
```

---

## What's next

This plan delivers a working, tested anomaly detection module (Objective 4) using real historical data. Still to come: the Streamlit dashboard (Objectives 2 and 4's remaining half) — 5 pages wired to `data/database.py`'s read paths, `models/predict.py`, and `anomaly/detector.py`'s stored results.
