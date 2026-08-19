# Historical Backfill & ML Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backfill historical weather + air-quality data from Open-Meteo's archive APIs, assemble it into a training dataset using the existing feature-engineering module, train and compare baseline/Linear Regression/Random Forest/XGBoost per prediction horizon (+1h/+3h/+6h), save the best model per horizon, and provide a `predict_aqi` function that automatically falls back to the baseline when no trained model exists yet (cold start).

**Architecture:** `data/historical.py` (new API calls to Open-Meteo's archive + air-quality-history endpoints) → `backfill.py` (fetch → validate → clean → store, reusing Plan 1's cleaning/DB layer) → `data/database.py` gains a read path (`load_weather_and_air_quality`) → `models/dataset.py` (features + targets + chronological split, built on Plan 1's `build_features`) → `models/train.py` (fits all candidates per horizon, picks the validation winner, evaluates once on test, saves it) → `models/predict.py` (loads a saved model, or falls back to the baseline if none exists).

**Tech Stack:** Everything already in `requirements.txt` from Plan 1 (scikit-learn, xgboost, joblib is a scikit-learn dependency so no new line needed, pandas). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-19-aqi-prediction-design.md`

**Prior plan this builds on:** `docs/superpowers/plans/2026-08-19-foundation-and-data-pipeline.md` (Plan 1 — already merged. Its `data/api_client.py`, `data/data_cleaning.py`, `data/database.py`, `data/feature_engineering.py`, `utils/config.py` all exist and are used here unchanged except where a task below explicitly extends them.)

## Global Constraints

- Chronological 70/15/15 split, never shuffled — inherited from Plan 1's `build_features`, extended here to the train/val/test split itself.
- `StandardScaler` (inside the Linear Regression `Pipeline`) is fit only on the training fold — never on validation or test — to avoid leakage.
- Model selection: compare baseline + all 3 ML candidates on the **validation** set; the winner is evaluated on the **test** set exactly once. Never report only training performance.
- Open-Meteo's historical archive does not provide `visibility` or `uv_index` (those are live-only fields in Open-Meteo's `current` endpoint, not present in the ERA5-based archive). These two fields are therefore excluded from the ML feature set entirely — even for live rows — so training and inference always see the same feature columns.
- Cold start: if a horizon has fewer than `MIN_TRAINING_ROWS` (50) usable rows after feature engineering, training is skipped for that horizon and no model file is written. `predict_aqi` detects the missing file and falls back to the rolling-average baseline, labeling the result `"baseline"` rather than a model name — never silently presenting a baseline number as an ML prediction.
- AQI is still always the CPCB-calculated value from Plan 1 — nothing in this plan touches AQI calculation itself, only historical backfill and modeling on top of it.
- No API keys — Open-Meteo's archive and air-quality-history endpoints are free/keyless, same as Plan 1.

---

### Task 1: Historical Weather & Air Quality API Clients

**Files:**
- Create: `data/historical.py`
- Modify: `utils/config.py` (add `HISTORICAL_WEATHER_API_URL`)
- Test: `tests/test_historical.py`

**Interfaces:**
- Consumes: `data.api_client._get_with_retry` (Plan 1), `utils.config.AIR_QUALITY_API_URL` (Plan 1, reused — Open-Meteo's air-quality endpoint serves both `current` and historical `hourly` data from the same URL)
- Produces: `fetch_historical_weather(latitude: float, longitude: float, start_date: str, end_date: str) -> list[dict]` (each dict has the same keys as Plan 1's `fetch_current_weather`, one per hour in range; `visibility` and `uv_index` are always `None`); `fetch_historical_air_quality(latitude: float, longitude: float, start_date: str, end_date: str) -> list[dict]` (same keys as `fetch_current_air_quality`, one per hour)

- [ ] **Step 1: Add `HISTORICAL_WEATHER_API_URL` to `utils/config.py`**

Add this line near the other API URL constants:

```python
HISTORICAL_WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_historical.py
from unittest.mock import MagicMock, patch

SAMPLE_HISTORICAL_WEATHER_RESPONSE = {
    "hourly": {
        "time": ["2026-05-01T00:00", "2026-05-01T01:00"],
        "temperature_2m": [28.1, 27.5],
        "relative_humidity_2m": [60, 62],
        "apparent_temperature": [30.0, 29.2],
        "precipitation": [0.0, 0.0],
        "cloud_cover": [10, 15],
        "pressure_msl": [1008.1, 1008.3],
        "wind_speed_10m": [8.2, 7.9],
        "wind_direction_10m": [180, 175],
    }
}


@patch("data.historical.requests.get")
def test_fetch_historical_weather_parses_response(mock_get):
    from data.historical import fetch_historical_weather

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = SAMPLE_HISTORICAL_WEATHER_RESPONSE
    mock_get.return_value = mock_response

    records = fetch_historical_weather(28.7041, 77.1025, "2026-05-01", "2026-05-01")

    assert len(records) == 2
    assert records[0]["timestamp"] == "2026-05-01T00:00"
    assert records[0]["temperature"] == 28.1
    assert records[0]["humidity"] == 60
    assert records[0]["visibility"] is None
    assert records[0]["uv_index"] is None
    assert records[1]["temperature"] == 27.5


@patch("data.historical.requests.get")
def test_fetch_historical_weather_returns_empty_list_on_failure(mock_get):
    import requests

    from data.historical import fetch_historical_weather

    mock_get.side_effect = requests.exceptions.Timeout("timed out")

    records = fetch_historical_weather(28.7041, 77.1025, "2026-05-01", "2026-05-01")

    assert records == []


SAMPLE_HISTORICAL_AIR_QUALITY_RESPONSE = {
    "hourly": {
        "time": ["2026-05-01T00:00", "2026-05-01T01:00"],
        "pm2_5": [80.0, 82.5],
        "pm10": [130.0, 135.0],
        "carbon_monoxide": [450.0, 460.0],
        "nitrogen_dioxide": [30.0, 32.0],
        "sulphur_dioxide": [8.0, 9.0],
        "ozone": [35.0, 37.0],
    }
}


@patch("data.historical.requests.get")
def test_fetch_historical_air_quality_parses_response(mock_get):
    from data.historical import fetch_historical_air_quality

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = SAMPLE_HISTORICAL_AIR_QUALITY_RESPONSE
    mock_get.return_value = mock_response

    records = fetch_historical_air_quality(28.7041, 77.1025, "2026-05-01", "2026-05-01")

    assert len(records) == 2
    assert records[0]["timestamp"] == "2026-05-01T00:00"
    assert records[0]["pm25"] == 80.0
    assert records[1]["pm10"] == 135.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_historical.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.historical'`

- [ ] **Step 4: Write `data/historical.py`**

```python
import requests

from data.api_client import _get_with_retry
from utils.config import AIR_QUALITY_API_URL, HISTORICAL_WEATHER_API_URL


def fetch_historical_weather(latitude: float, longitude: float, start_date: str, end_date: str) -> list[dict]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "cloud_cover",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
        ]),
        "timezone": "auto",
    }
    payload = _get_with_retry(HISTORICAL_WEATHER_API_URL, params)
    if payload is None or "hourly" not in payload:
        return []

    hourly = payload["hourly"]
    times = hourly.get("time", [])
    records = []
    for i, timestamp in enumerate(times):
        records.append({
            "timestamp": timestamp,
            "latitude": latitude,
            "longitude": longitude,
            "temperature": hourly.get("temperature_2m", [None] * len(times))[i],
            "feels_like": hourly.get("apparent_temperature", [None] * len(times))[i],
            "humidity": hourly.get("relative_humidity_2m", [None] * len(times))[i],
            "pressure": hourly.get("pressure_msl", [None] * len(times))[i],
            "wind_speed": hourly.get("wind_speed_10m", [None] * len(times))[i],
            "wind_direction": hourly.get("wind_direction_10m", [None] * len(times))[i],
            "rainfall": hourly.get("precipitation", [None] * len(times))[i],
            "visibility": None,
            "cloud_cover": hourly.get("cloud_cover", [None] * len(times))[i],
            "uv_index": None,
        })
    return records


def fetch_historical_air_quality(latitude: float, longitude: float, start_date: str, end_date: str) -> list[dict]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
        ]),
        "timezone": "auto",
    }
    payload = _get_with_retry(AIR_QUALITY_API_URL, params)
    if payload is None or "hourly" not in payload:
        return []

    hourly = payload["hourly"]
    times = hourly.get("time", [])
    records = []
    for i, timestamp in enumerate(times):
        records.append({
            "timestamp": timestamp,
            "latitude": latitude,
            "longitude": longitude,
            "pm25": hourly.get("pm2_5", [None] * len(times))[i],
            "pm10": hourly.get("pm10", [None] * len(times))[i],
            "co": hourly.get("carbon_monoxide", [None] * len(times))[i],
            "no2": hourly.get("nitrogen_dioxide", [None] * len(times))[i],
            "so2": hourly.get("sulphur_dioxide", [None] * len(times))[i],
            "o3": hourly.get("ozone", [None] * len(times))[i],
        })
    return records
```

Note: `_get_with_retry` already catches `(requests.exceptions.RequestException, ValueError)` per Plan 1's Task 2 fix, so a network failure or malformed JSON here returns `None` (handled above) rather than raising — the `import requests` in the test file is only needed to reference `requests.exceptions.Timeout`.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_historical.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add data/historical.py tests/test_historical.py utils/config.py
git commit -m "feat: add historical weather and air quality API clients"
```

---

### Task 2: Backfill Script

**Files:**
- Create: `backfill.py`
- Test: `tests/test_backfill.py`

**Interfaces:**
- Consumes: `data.historical.fetch_historical_weather`, `data.historical.fetch_historical_air_quality`, `data.data_cleaning.validate_weather_record`, `data.data_cleaning.validate_air_quality_record`, `data.data_cleaning.clean_weather_record`, `data.data_cleaning.clean_air_quality_record`, `data.database.get_engine`, `data.database.init_db`, `data.database.insert_weather_and_air_quality` (Plan 1's atomic paired-insert helper, already handles duplicate `(timestamp, location)` as a silent no-op), `utils.config.LOCATIONS`, `utils.config.DEFAULT_LOCATION`
- Produces: `backfill_location(location: str, days: int = 92, engine=None) -> int` (returns the count of hours successfully stored; never raises)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backfill.py
from unittest.mock import patch

from data.database import AirQualityData, WeatherData, get_engine, init_db
from sqlalchemy.orm import Session


def _weather_record(timestamp):
    return {
        "timestamp": timestamp, "latitude": 28.7041, "longitude": 77.1025,
        "temperature": 30.0, "feels_like": 32.0, "humidity": 55, "pressure": 1005.0,
        "wind_speed": 10.0, "wind_direction": 200, "rainfall": 0.0,
        "visibility": None, "cloud_cover": 20, "uv_index": None,
    }


def _air_quality_record(timestamp):
    return {
        "timestamp": timestamp, "latitude": 28.7041, "longitude": 77.1025,
        "pm25": 80.0, "pm10": 130.0, "co": 450.0, "no2": 30.0, "so2": 8.0, "o3": 35.0,
    }


@patch("backfill.fetch_historical_air_quality")
@patch("backfill.fetch_historical_weather")
def test_backfill_location_stores_only_overlapping_timestamps(mock_weather, mock_air_quality):
    from backfill import backfill_location

    mock_weather.return_value = [
        _weather_record("2026-05-01T00:00"),
        _weather_record("2026-05-01T01:00"),
        _weather_record("2026-05-01T02:00"),  # no matching air-quality row
    ]
    mock_air_quality.return_value = [
        _air_quality_record("2026-05-01T00:00"),
        _air_quality_record("2026-05-01T01:00"),
        _air_quality_record("2026-05-01T03:00"),  # no matching weather row
    ]

    engine = get_engine(":memory:")
    init_db(engine)

    stored = backfill_location("Delhi", days=1, engine=engine)

    assert stored == 2
    with Session(engine) as session:
        assert session.query(WeatherData).count() == 2
        assert session.query(AirQualityData).count() == 2


@patch("backfill.fetch_historical_air_quality")
@patch("backfill.fetch_historical_weather")
def test_backfill_location_is_idempotent_on_rerun(mock_weather, mock_air_quality):
    from backfill import backfill_location

    mock_weather.return_value = [_weather_record("2026-05-01T00:00")]
    mock_air_quality.return_value = [_air_quality_record("2026-05-01T00:00")]

    engine = get_engine(":memory:")
    init_db(engine)

    first_run = backfill_location("Delhi", days=1, engine=engine)
    second_run = backfill_location("Delhi", days=1, engine=engine)

    assert first_run == 1
    assert second_run == 1  # counted as "processed", but stored as a no-op duplicate
    with Session(engine) as session:
        assert session.query(WeatherData).count() == 1


def test_backfill_location_returns_zero_for_unknown_location():
    from backfill import backfill_location

    engine = get_engine(":memory:")
    init_db(engine)

    stored = backfill_location("Atlantis", days=1, engine=engine)

    assert stored == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backfill.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backfill'`

- [ ] **Step 3: Write `backfill.py`**

```python
import sys
from datetime import date, timedelta

from data.data_cleaning import (
    clean_air_quality_record,
    clean_weather_record,
    validate_air_quality_record,
    validate_weather_record,
)
from data.database import get_engine, init_db, insert_weather_and_air_quality
from data.historical import fetch_historical_air_quality, fetch_historical_weather
from utils.config import DEFAULT_LOCATION, LOCATIONS
from utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_BACKFILL_DAYS = 92


def backfill_location(location: str, days: int = DEFAULT_BACKFILL_DAYS, engine=None) -> int:
    if location not in LOCATIONS:
        logger.error("Unknown location '%s'", location)
        return 0

    latitude, longitude = LOCATIONS[location]
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    engine = engine or get_engine()
    try:
        init_db(engine)
    except Exception as exc:
        logger.error("Failed to initialize database for backfill: %s", exc)
        return 0

    weather_records = fetch_historical_weather(latitude, longitude, start_date.isoformat(), end_date.isoformat())
    air_quality_records = fetch_historical_air_quality(latitude, longitude, start_date.isoformat(), end_date.isoformat())

    weather_by_timestamp = {r["timestamp"]: r for r in weather_records}
    air_quality_by_timestamp = {r["timestamp"]: r for r in air_quality_records}
    shared_timestamps = sorted(set(weather_by_timestamp) & set(air_quality_by_timestamp))

    stored = 0
    for timestamp in shared_timestamps:
        weather_raw = weather_by_timestamp[timestamp]
        air_quality_raw = air_quality_by_timestamp[timestamp]

        if not validate_weather_record(weather_raw) or not validate_air_quality_record(air_quality_raw):
            continue

        weather_clean = clean_weather_record(weather_raw, location)
        air_quality_clean = clean_air_quality_record(air_quality_raw, location)

        try:
            insert_weather_and_air_quality(engine, weather_clean, air_quality_clean)
            stored += 1
        except Exception as exc:
            logger.warning("Failed to store backfilled hour %s for %s: %s", timestamp, location, exc)

    logger.info(
        "Backfill for %s: processed %d/%d overlapping hours (%s to %s)",
        location, stored, len(shared_timestamps), start_date, end_date,
    )
    return stored


if __name__ == "__main__":
    location = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOCATION
    days = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BACKFILL_DAYS
    backfill_location(location, days)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backfill.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Manual verification against the real API**

```bash
python backfill.py Delhi 92
sqlite3 db/aqi_system.db "select count(*) from weather_data where location='Delhi';"
sqlite3 db/aqi_system.db "select count(*) from air_quality_data where location='Delhi';"
sqlite3 db/aqi_system.db "select min(timestamp), max(timestamp) from air_quality_data where location='Delhi';"
```

Expected: both counts are in the hundreds to low thousands (up to ~2200 for 92 days if Open-Meteo returns the full range), the min/max timestamps span roughly the last 92 days, and the command exits without error. Note the actual count in your report — if it's much lower than expected, that's useful information (it likely means Open-Meteo's historical air-quality coverage for this location doesn't extend the full 92 days; note the earliest available date you observe, this does not indicate a bug).

- [ ] **Step 6: Commit**

```bash
git add backfill.py tests/test_backfill.py
git commit -m "feat: add historical data backfill script"
```

---

### Task 3: Database Read Path

**Files:**
- Modify: `data/database.py` (add `load_weather_and_air_quality`)
- Test: `tests/test_database.py` (append)

**Interfaces:**
- Produces: `load_weather_and_air_quality(engine, location: str) -> pd.DataFrame` — inner join of `weather_data` and `air_quality_data` on `(timestamp, location)`, sorted ascending by `timestamp`, columns: `timestamp, location, temperature, feels_like, humidity, pressure, wind_speed, wind_direction, rainfall, visibility, cloud_cover, uv_index, pm25, pm10, co, no2, so2, o3, aqi`

- [ ] **Step 1: Write the failing test**

Note: `tests/test_database.py` already has `from datetime import datetime` and `from data.database import (AirQualityData, WeatherData, get_engine, init_db, insert_air_quality_record, insert_weather_record, insert_weather_and_air_quality)` at the top from Plan 1 — `insert_weather_and_air_quality` and `insert_weather_record` are already in scope, do not re-import them. Only add the two new imports shown below (merge `load_weather_and_air_quality` into the existing `from data.database import (...)` line rather than a separate line).

```python
# append to tests/test_database.py
import pandas as pd

from data.database import load_weather_and_air_quality  # merge into the existing import line


def test_load_weather_and_air_quality_joins_on_timestamp_and_location():
    engine = get_engine(":memory:")
    init_db(engine)

    insert_weather_and_air_quality(engine, {
        "timestamp": datetime(2026, 5, 1, 0, 0), "location": "Delhi",
        "latitude": 28.7041, "longitude": 77.1025, "temperature": 30.0,
        "feels_like": 32.0, "humidity": 55, "pressure": 1005.0, "wind_speed": 10.0,
        "wind_direction": 200, "rainfall": 0.0, "visibility": None,
        "cloud_cover": 20, "uv_index": None,
    }, {
        "timestamp": datetime(2026, 5, 1, 0, 0), "location": "Delhi",
        "pm25": 80.0, "pm10": 130.0, "co": 450.0, "no2": 30.0, "so2": 8.0, "o3": 35.0, "aqi": 158,
    })
    # A weather-only hour (no matching air-quality row) must not appear in the join
    insert_weather_record(engine, {
        "timestamp": datetime(2026, 5, 1, 1, 0), "location": "Delhi",
        "latitude": 28.7041, "longitude": 77.1025, "temperature": 29.0,
        "feels_like": 31.0, "humidity": 58, "pressure": 1005.5, "wind_speed": 9.0,
        "wind_direction": 195, "rainfall": 0.0, "visibility": None,
        "cloud_cover": 25, "uv_index": None,
    })

    df = load_weather_and_air_quality(engine, "Delhi")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["temperature"] == 30.0
    assert df.iloc[0]["aqi"] == 158


def test_load_weather_and_air_quality_returns_empty_frame_for_unknown_location():
    engine = get_engine(":memory:")
    init_db(engine)

    df = load_weather_and_air_quality(engine, "Nowhere")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
```

Note: this test imports `insert_weather_and_air_quality` (Plan 1's atomic paired-insert helper) — it's already in `data/database.py` from Plan 1's final-review fix wave, so no new import wiring is needed beyond what the existing test file already has from Plan 1.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_database.py -v`
Expected: FAIL — `load_weather_and_air_quality` does not exist (`ImportError`)

- [ ] **Step 3: Add `load_weather_and_air_quality` to `data/database.py`**

```python
import pandas as pd


def load_weather_and_air_quality(engine, location: str) -> pd.DataFrame:
    with Session(engine) as session:
        weather_rows = session.query(WeatherData).filter(WeatherData.location == location).all()
        air_quality_rows = session.query(AirQualityData).filter(AirQualityData.location == location).all()

    weather_df = pd.DataFrame([{
        "timestamp": r.timestamp, "location": r.location, "temperature": r.temperature,
        "feels_like": r.feels_like, "humidity": r.humidity, "pressure": r.pressure,
        "wind_speed": r.wind_speed, "wind_direction": r.wind_direction, "rainfall": r.rainfall,
        "visibility": r.visibility, "cloud_cover": r.cloud_cover, "uv_index": r.uv_index,
    } for r in weather_rows])

    air_quality_df = pd.DataFrame([{
        "timestamp": r.timestamp, "location": r.location, "pm25": r.pm25, "pm10": r.pm10,
        "co": r.co, "no2": r.no2, "so2": r.so2, "o3": r.o3, "aqi": r.aqi,
    } for r in air_quality_rows])

    if weather_df.empty or air_quality_df.empty:
        return pd.DataFrame(columns=[
            "timestamp", "location", "temperature", "feels_like", "humidity", "pressure",
            "wind_speed", "wind_direction", "rainfall", "visibility", "cloud_cover", "uv_index",
            "pm25", "pm10", "co", "no2", "so2", "o3", "aqi",
        ])

    merged = pd.merge(weather_df, air_quality_df, on=["timestamp", "location"], how="inner")
    return merged.sort_values("timestamp").reset_index(drop=True)
```

Add `import pandas as pd` at the top of `data/database.py` alongside the existing SQLAlchemy imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_database.py -v`
Expected: PASS (all tests in the file, including the 2 new ones)

- [ ] **Step 5: Commit**

```bash
git add data/database.py tests/test_database.py
git commit -m "feat: add read path joining weather and air quality history"
```

---

### Task 4: Training Dataset Assembly

**Files:**
- Create: `models/__init__.py` (empty, makes `models` a package)
- Create: `models/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `data.feature_engineering.build_features` (Plan 1)
- Produces: `FEATURE_COLUMNS: list[str]` (the fixed, ordered feature set used by every model); `TARGET_HORIZONS: dict[str, int]` (`{"1h": 1, "3h": 3, "6h": 6}`); `add_target_columns(df: pd.DataFrame) -> pd.DataFrame`; `build_training_frame(raw_df: pd.DataFrame) -> pd.DataFrame` (features + targets, rows with missing required values dropped); `chronological_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`; `baseline_predict(df: pd.DataFrame) -> pd.Series` (the rolling-average baseline: `df["aqi_rolling_3"]`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py
import pandas as pd

from models.dataset import (
    FEATURE_COLUMNS,
    TARGET_HORIZONS,
    add_target_columns,
    baseline_predict,
    build_training_frame,
    chronological_split,
)


def _sample_raw_df(n=40):
    timestamps = pd.date_range("2026-05-01 00:00", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "location": ["Delhi"] * n,
        "temperature": [30.0 + i * 0.1 for i in range(n)],
        "feels_like": [32.0] * n,
        "humidity": [55.0] * n,
        "pressure": [1005.0] * n,
        "wind_speed": [10.0] * n,
        "wind_direction": [200.0] * n,
        "rainfall": [0.0] * n,
        "visibility": [None] * n,
        "cloud_cover": [20.0] * n,
        "uv_index": [None] * n,
        "pm25": [80.0 + i for i in range(n)],
        "pm10": [130.0 + i for i in range(n)],
        "co": [450.0] * n,
        "no2": [30.0] * n,
        "so2": [8.0] * n,
        "o3": [35.0] * n,
        "aqi": [100 + i * 2 for i in range(n)],
    })


def test_add_target_columns_shifts_future_aqi_backward():
    df = add_target_columns(_sample_raw_df())

    assert df["aqi_target_1h"].iloc[0] == df["aqi"].iloc[1]
    assert df["aqi_target_3h"].iloc[0] == df["aqi"].iloc[3]
    assert df["aqi_target_6h"].iloc[0] == df["aqi"].iloc[6]
    assert pd.isna(df["aqi_target_6h"].iloc[-1])


def test_build_training_frame_drops_rows_with_missing_required_values():
    frame = build_training_frame(_sample_raw_df())

    assert len(frame) > 0
    assert len(frame) < 40  # warmup (lags/rolling) and lookahead (targets) both trim rows
    for column in FEATURE_COLUMNS:
        assert frame[column].isna().sum() == 0
    for label in TARGET_HORIZONS:
        assert frame[f"aqi_target_{label}"].isna().sum() == 0


def test_chronological_split_is_contiguous_and_unshuffled():
    frame = build_training_frame(_sample_raw_df())

    train_df, val_df, test_df = chronological_split(frame)

    assert len(train_df) + len(val_df) + len(test_df) == len(frame)
    assert list(train_df["timestamp"]) == sorted(train_df["timestamp"])
    if len(train_df) and len(val_df):
        assert train_df["timestamp"].max() <= val_df["timestamp"].min()
    if len(val_df) and len(test_df):
        assert val_df["timestamp"].max() <= test_df["timestamp"].min()


def test_baseline_predict_returns_rolling_average_column():
    frame = build_training_frame(_sample_raw_df())

    predicted = baseline_predict(frame)

    assert (predicted == frame["aqi_rolling_3"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models'`

- [ ] **Step 3: Create the `models` package and write `models/dataset.py`**

```bash
mkdir -p models
touch models/__init__.py
```

```python
# models/dataset.py
import pandas as pd

from data.feature_engineering import build_features

TARGET_HORIZONS = {"1h": 1, "3h": 3, "6h": 6}

FEATURE_COLUMNS = [
    "hour", "day_of_week", "month", "is_weekend",
    "temperature", "humidity", "pressure", "wind_speed", "rainfall", "cloud_cover",
    "pm25", "pm10",
    "aqi_lag_1", "aqi_lag_3", "aqi_lag_6",
    "pm25_lag_1", "pm25_lag_3", "pm25_lag_6",
    "pm10_lag_1", "pm10_lag_3", "pm10_lag_6",
    "aqi_rolling_3", "aqi_rolling_6",
    "pm25_rolling_3", "pm25_rolling_6",
]


def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for label, hours in TARGET_HORIZONS.items():
        df[f"aqi_target_{label}"] = df["aqi"].shift(-hours)
    return df


def build_training_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = build_features(raw_df)
    df["is_weekend"] = df["is_weekend"].astype(int)
    df = add_target_columns(df)

    required_columns = FEATURE_COLUMNS + [f"aqi_target_{label}" for label in TARGET_HORIZONS]
    df = df.dropna(subset=required_columns)
    return df.reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


def baseline_predict(df: pd.DataFrame) -> pd.Series:
    return df["aqi_rolling_3"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add models/__init__.py models/dataset.py tests/test_dataset.py
git commit -m "feat: add training dataset assembly (features, targets, chronological split)"
```

---

### Task 5: Evaluation Metrics

**Files:**
- Create: `models/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Produces: `compute_metrics(y_true, y_pred) -> dict` (keys: `mae`, `rmse`, `r2`, `mape` — all `float`, `mape` may be `None` if every true value is 0)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
from models.evaluate import compute_metrics


def test_compute_metrics_returns_expected_keys_and_reasonable_values():
    y_true = [100, 150, 200]
    y_pred = [110, 140, 210]

    metrics = compute_metrics(y_true, y_pred)

    assert set(metrics.keys()) == {"mae", "rmse", "r2", "mape"}
    assert metrics["mae"] == 10.0
    assert metrics["rmse"] > 0
    assert metrics["mape"] is not None
    assert metrics["mape"] > 0


def test_compute_metrics_perfect_prediction_has_zero_error():
    y_true = [50, 60, 70]
    y_pred = [50, 60, 70]

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mape"] == 0.0


def test_compute_metrics_handles_all_zero_true_values():
    y_true = [0, 0, 0]
    y_pred = [1, 2, 3]

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["mape"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.evaluate'`

- [ ] **Step 3: Write `models/evaluate.py`**

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    nonzero = y_true != 0
    if nonzero.any():
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    else:
        mape = None

    return {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4),
        "mape": round(mape, 2) if mape is not None else None,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add models/evaluate.py tests/test_evaluate.py
git commit -m "feat: add model evaluation metrics"
```

---

### Task 6: Model Training Pipeline

**Files:**
- Modify: `utils/config.py` (add `SAVED_MODELS_DIR`)
- Create: `models/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `data.database.get_engine`, `data.database.load_weather_and_air_quality` (Task 3); `models.dataset.FEATURE_COLUMNS`, `models.dataset.TARGET_HORIZONS`, `models.dataset.build_training_frame`, `models.dataset.chronological_split`, `models.dataset.baseline_predict` (Task 4); `models.evaluate.compute_metrics` (Task 5)
- Produces: `train_and_evaluate(location: str, engine=None) -> dict` (keys are horizon labels; each value has `best_model`, `test_metrics`, `model_path` — empty dict if training was skipped for lack of data). Writes `models/saved_models/aqi_{horizon}.joblib` and `models/saved_models/aqi_{horizon}_model_name.txt` per horizon that had enough data, and appends every candidate's metrics to `models/saved_models/metrics.csv`.

- [ ] **Step 1: Add `SAVED_MODELS_DIR` to `utils/config.py`**

Add near `DATABASE_PATH` (reuse the same `PROJECT_ROOT`-anchoring pattern Plan 1's final fix wave already established for `DATABASE_PATH`):

```python
SAVED_MODELS_DIR = str(PROJECT_ROOT / "models" / "saved_models")
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_train.py
import os

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from data.database import AirQualityData, WeatherData, get_engine, init_db


def _seed_synthetic_history(engine, location="Delhi", n=200):
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-01-01 00:00", periods=n, freq="h")
    base_aqi = 100 + 40 * np.sin(np.linspace(0, 8 * np.pi, n)) + rng.normal(0, 5, n)
    base_pm25 = base_aqi * 0.5 + rng.normal(0, 3, n)

    with Session(engine) as session:
        for i, ts in enumerate(timestamps):
            session.add(WeatherData(
                timestamp=ts.to_pydatetime(), location=location, latitude=28.7041, longitude=77.1025,
                temperature=28 + 5 * np.sin(i / 24), feels_like=30.0, humidity=55.0, pressure=1005.0,
                wind_speed=10.0, wind_direction=200.0, rainfall=0.0, visibility=None,
                cloud_cover=20.0, uv_index=None,
            ))
            session.add(AirQualityData(
                timestamp=ts.to_pydatetime(), location=location,
                pm25=float(base_pm25[i]), pm10=float(base_pm25[i] + 40), co=450.0,
                no2=30.0, so2=8.0, o3=35.0, aqi=int(base_aqi[i]),
            ))
        session.commit()


def test_train_and_evaluate_produces_a_model_per_horizon(tmp_path, monkeypatch):
    import models.train as train_module

    saved_models_dir = str(tmp_path / "saved_models")
    monkeypatch.setattr(train_module, "SAVED_MODELS_DIR", saved_models_dir)
    monkeypatch.setattr(train_module, "METRICS_LOG_PATH", os.path.join(saved_models_dir, "metrics.csv"))

    engine = get_engine(":memory:")
    init_db(engine)
    _seed_synthetic_history(engine)

    results = train_module.train_and_evaluate("Delhi", engine=engine)

    assert set(results.keys()) == {"1h", "3h", "6h"}
    for label, result in results.items():
        assert result["best_model"] in {"linear_regression", "random_forest", "xgboost"}
        assert "mae" in result["test_metrics"]
        assert os.path.exists(result["model_path"])
        assert os.path.exists(os.path.join(saved_models_dir, f"aqi_{label}_model_name.txt"))

    assert os.path.exists(os.path.join(saved_models_dir, "metrics.csv"))
    metrics_df = pd.read_csv(os.path.join(saved_models_dir, "metrics.csv"))
    assert set(metrics_df["model"]) >= {"baseline", "linear_regression", "random_forest", "xgboost"}


def test_train_and_evaluate_skips_horizon_with_insufficient_data(tmp_path, monkeypatch):
    import models.train as train_module

    saved_models_dir = str(tmp_path / "saved_models")
    monkeypatch.setattr(train_module, "SAVED_MODELS_DIR", saved_models_dir)
    monkeypatch.setattr(train_module, "METRICS_LOG_PATH", os.path.join(saved_models_dir, "metrics.csv"))

    engine = get_engine(":memory:")
    init_db(engine)
    _seed_synthetic_history(engine, n=20)  # well under MIN_TRAINING_ROWS after feature/target trimming

    results = train_module.train_and_evaluate("Delhi", engine=engine)

    assert results == {}
    assert not os.path.exists(saved_models_dir)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.train'`

- [ ] **Step 4: Write `models/train.py`**

```python
import os
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from data.database import get_engine, load_weather_and_air_quality
from models.dataset import FEATURE_COLUMNS, TARGET_HORIZONS, baseline_predict, build_training_frame, chronological_split
from models.evaluate import compute_metrics
from utils.config import DEFAULT_LOCATION, SAVED_MODELS_DIR
from utils.logging_config import get_logger

logger = get_logger(__name__)

MIN_TRAINING_ROWS = 50
METRICS_LOG_PATH = os.path.join(SAVED_MODELS_DIR, "metrics.csv")


def _candidate_models() -> dict:
    return {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "xgboost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    }


def train_and_evaluate(location: str, engine=None) -> dict:
    engine = engine or get_engine()
    raw_df = load_weather_and_air_quality(engine, location)
    frame = build_training_frame(raw_df)

    if len(frame) < MIN_TRAINING_ROWS:
        logger.warning(
            "Only %d usable rows for %s after feature engineering (need >= %d); skipping training",
            len(frame), location, MIN_TRAINING_ROWS,
        )
        return {}

    train_df, val_df, test_df = chronological_split(frame)
    results = {}
    metrics_rows = []

    for label in TARGET_HORIZONS:
        target_col = f"aqi_target_{label}"
        X_train, y_train = train_df[FEATURE_COLUMNS], train_df[target_col]
        X_val, y_val = val_df[FEATURE_COLUMNS], val_df[target_col]
        X_test, y_test = test_df[FEATURE_COLUMNS], test_df[target_col]

        baseline_val_metrics = compute_metrics(y_val, baseline_predict(val_df))
        metrics_rows.append({"horizon": label, "model": "baseline", "split": "validation", **baseline_val_metrics})

        val_scores = {}
        fitted = {}
        for name, estimator in _candidate_models().items():
            estimator.fit(X_train, y_train)
            fitted[name] = estimator
            val_metrics = compute_metrics(y_val, estimator.predict(X_val))
            val_scores[name] = val_metrics["mae"]
            metrics_rows.append({"horizon": label, "model": name, "split": "validation", **val_metrics})

        best_name = min(val_scores, key=val_scores.get)
        best_model = fitted[best_name]
        test_metrics = compute_metrics(y_test, best_model.predict(X_test))
        metrics_rows.append({"horizon": label, "model": best_name, "split": "test", **test_metrics})

        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        model_path = os.path.join(SAVED_MODELS_DIR, f"aqi_{label}.joblib")
        name_path = os.path.join(SAVED_MODELS_DIR, f"aqi_{label}_model_name.txt")
        joblib.dump(best_model, model_path)
        with open(name_path, "w") as f:
            f.write(best_name)

        logger.info(
            "Horizon %s: best model=%s (val MAE=%.2f), test MAE=%.2f, saved to %s",
            label, best_name, val_scores[best_name], test_metrics["mae"], model_path,
        )
        results[label] = {"best_model": best_name, "test_metrics": test_metrics, "model_path": model_path}

    _append_metrics_log(metrics_rows, location, len(frame))
    return results


def _append_metrics_log(metrics_rows: list[dict], location: str, dataset_size: int) -> None:
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    log_df = pd.DataFrame(metrics_rows)
    log_df["location"] = location
    log_df["dataset_size"] = dataset_size
    log_df["trained_at"] = datetime.now(timezone.utc).isoformat()
    if os.path.exists(METRICS_LOG_PATH):
        log_df.to_csv(METRICS_LOG_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(METRICS_LOG_PATH, index=False)


if __name__ == "__main__":
    location = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOCATION
    train_and_evaluate(location)
```

Note on the test's `monkeypatch.setattr(train_module, "SAVED_MODELS_DIR", ...)`: this only works if `models/train.py` reads the module-level `SAVED_MODELS_DIR` name at call time (which it does, since `_append_metrics_log` and `train_and_evaluate` reference the module global directly) — do not refactor this into a function default argument, which would bind the value at import time and make it unpatchable.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_train.py -v`
Expected: PASS (2 tests) — this will take a few seconds (fitting 3 real models × 3 horizons on ~185 rows)

- [ ] **Step 6: Manual verification against the real backfilled data**

Requires Task 2's backfill to have been run for Delhi first (`python backfill.py Delhi 92`, if not already done).

```bash
python models/train.py Delhi
cat models/saved_models/metrics.csv
ls models/saved_models/
```

Expected: `metrics.csv` has rows for `baseline`, `linear_regression`, `random_forest`, `xgboost` at each horizon (validation split) plus one `test` row per horizon for the winning model; `aqi_1h.joblib`, `aqi_3h.joblib`, `aqi_6h.joblib` and their `_model_name.txt` siblings exist. Note which model won each horizon and its test MAE in your report — don't fabricate these numbers, report what actually printed/was written.

- [ ] **Step 7: Commit**

```bash
git add utils/config.py models/train.py tests/test_train.py
git commit -m "feat: add model training pipeline (baseline, linear regression, random forest, xgboost)"
```

---

### Task 7: Prediction Module

**Files:**
- Create: `models/predict.py`
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: `models.dataset.FEATURE_COLUMNS`, `models.dataset.baseline_predict`; `utils.config.SAVED_MODELS_DIR`
- Produces: `load_model(horizon: str) -> tuple[object, str] | None`; `predict_aqi(horizon: str, feature_row: pd.DataFrame) -> tuple[float, str]` (second element is the model name, or `"baseline"` if no trained model exists for that horizon yet)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_predict.py
import os

import joblib
import pandas as pd
import pytest

from models.dataset import FEATURE_COLUMNS


def _feature_row(aqi_rolling_3=120.0):
    row = {column: 1.0 for column in FEATURE_COLUMNS}
    row["aqi_rolling_3"] = aqi_rolling_3
    return pd.DataFrame([row])


def test_predict_aqi_falls_back_to_baseline_when_no_model_saved(tmp_path, monkeypatch):
    import models.predict as predict_module

    monkeypatch.setattr(predict_module, "SAVED_MODELS_DIR", str(tmp_path))

    predicted, model_name = predict_module.predict_aqi("1h", _feature_row(aqi_rolling_3=133.0))

    assert predicted == 133.0
    assert model_name == "baseline"


def test_predict_aqi_uses_saved_model_when_present(tmp_path, monkeypatch):
    import models.predict as predict_module

    monkeypatch.setattr(predict_module, "SAVED_MODELS_DIR", str(tmp_path))

    class _DummyModel:
        def predict(self, X):
            return [42.0] * len(X)

    joblib.dump(_DummyModel(), os.path.join(str(tmp_path), "aqi_3h.joblib"))
    with open(os.path.join(str(tmp_path), "aqi_3h_model_name.txt"), "w") as f:
        f.write("random_forest")

    predicted, model_name = predict_module.predict_aqi("3h", _feature_row())

    assert predicted == 42.0
    assert model_name == "random_forest"


def test_load_model_returns_none_when_only_one_of_the_pair_exists(tmp_path, monkeypatch):
    import models.predict as predict_module

    monkeypatch.setattr(predict_module, "SAVED_MODELS_DIR", str(tmp_path))
    joblib.dump(object(), os.path.join(str(tmp_path), "aqi_6h.joblib"))
    # no matching aqi_6h_model_name.txt written

    assert predict_module.load_model("6h") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_predict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.predict'`

- [ ] **Step 3: Write `models/predict.py`**

```python
import os

import joblib
import pandas as pd

from models.dataset import FEATURE_COLUMNS, baseline_predict
from utils.config import SAVED_MODELS_DIR


def load_model(horizon: str) -> tuple[object, str] | None:
    model_path = os.path.join(SAVED_MODELS_DIR, f"aqi_{horizon}.joblib")
    name_path = os.path.join(SAVED_MODELS_DIR, f"aqi_{horizon}_model_name.txt")
    if not (os.path.exists(model_path) and os.path.exists(name_path)):
        return None
    model = joblib.load(model_path)
    with open(name_path) as f:
        model_name = f.read().strip()
    return model, model_name


def predict_aqi(horizon: str, feature_row: pd.DataFrame) -> tuple[float, str]:
    loaded = load_model(horizon)
    if loaded is None:
        predicted = float(baseline_predict(feature_row).iloc[0])
        return predicted, "baseline"

    model, model_name = loaded
    predicted = float(model.predict(feature_row[FEATURE_COLUMNS])[0])
    return predicted, model_name
```

Note: `load_model` and `predict_aqi` both read the module-level `SAVED_MODELS_DIR` name at call time, matching the same `monkeypatch`-friendly pattern used in Task 6 — do not read it into a default argument at import/definition time.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_predict.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Manual verification against the real trained model**

Requires Task 6's training to have been run for Delhi first.

```bash
python -c "
from data.database import get_engine, load_weather_and_air_quality
from data.feature_engineering import build_features
from models.predict import predict_aqi

engine = get_engine()
raw_df = load_weather_and_air_quality(engine, 'Delhi')
features_df = build_features(raw_df)
features_df['is_weekend'] = features_df['is_weekend'].astype(int)
latest_row = features_df.dropna(subset=['aqi_rolling_3']).tail(1)
for horizon in ['1h', '3h', '6h']:
    predicted, model_name = predict_aqi(horizon, latest_row)
    print(f'{horizon}: predicted AQI={predicted:.1f} (model={model_name})')
"
```

Expected: three lines printed with plausible AQI numbers (roughly in the 0-500 range, ideally close to the current AQI) and each `model` should be `linear_regression`, `random_forest`, or `xgboost` (not `baseline`, since Task 6 should have trained real models against ~92 days of Delhi data). Report the actual printed output — don't fabricate it.

- [ ] **Step 6: Commit**

```bash
git add models/predict.py tests/test_predict.py
git commit -m "feat: add prediction module with baseline cold-start fallback"
```

---

## What's next

This plan delivers a working, evaluated ML pipeline (Objective 3) trained on real historical data, with an honest cold-start fallback. Still to come:

1. **Anomaly detection** — Isolation Forest module over stored data (Objective 4, first half).
2. **Dashboard** — the 5 Streamlit pages (Objectives 2 and 4, second half), wired to `data/database.py`'s read path and `models/predict.py`.
