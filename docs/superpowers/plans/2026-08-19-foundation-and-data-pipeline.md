# Foundation & Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the project scaffold and a working, tested pipeline that fetches live weather + air-quality data for a city from Open-Meteo, validates and cleans it, calculates CPCB AQI, and stores it in SQLite — with a feature-engineering module ready for the ML phase.

**Architecture:** Open-Meteo (weather + air quality, both free/keyless) → API client (retry/timeout) → validation → cleaning + CPCB AQI calculation → SQLAlchemy/SQLite → feature engineering module (used later by training). A thin `collect.py` script wires fetch → validate → clean → store for one location.

**Tech Stack:** Python 3.11+, requests, python-dotenv, SQLAlchemy, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-aqi-prediction-design.md`

## Global Constraints

- No API keys hardcoded — Open-Meteo needs none today; config still loads `.env` so keyed providers can be added later without code changes.
- AQI is always CPCB-calculated from raw pollutants — never use a provider's own AQI field.
- Time-series data is never shuffled; always sorted by timestamp before lag/rolling features.
- Missing optional fields stay `None` — never defaulted to 0.
- SQLite only, via SQLAlchemy.
- This plan covers roadmap phases 1–4 (Setup, API integration, Database, Data processing) — ML, anomaly detection, and the dashboard are separate follow-up plans.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`, `.env.example`, `.gitignore`, `pyproject.toml`
- Create: `utils/__init__.py`, `utils/config.py`, `utils/logging_config.py`
- Create: `data/__init__.py` (empty, makes `data` a package)
- Create: `tests/__init__.py` (empty)
- Create directories (via placeholder files where needed): `db/.gitkeep`, `logs/.gitkeep`, `models/saved_models/.gitkeep`, `anomaly/.gitkeep`, `visualization/.gitkeep`, `pages/.gitkeep`, `notebooks/.gitkeep`

**Interfaces:**
- Produces: `utils.config.LOCATIONS` (dict[str, tuple[float, float]]), `utils.config.DEFAULT_LOCATION` (str), `utils.config.DATABASE_PATH` (str), `utils.config.WEATHER_API_URL` / `AIR_QUALITY_API_URL` (str), `utils.logging_config.get_logger(name: str) -> logging.Logger`

- [ ] **Step 1: Create `requirements.txt`**

```
requests>=2.31
python-dotenv>=1.0
SQLAlchemy>=2.0
pandas>=2.0
numpy>=1.26
scikit-learn>=1.4
xgboost>=2.0
streamlit>=1.32
plotly>=5.20
folium>=0.16
streamlit-folium>=0.20
pytest>=8.0
```

- [ ] **Step 2: Create `.env.example`**

```
# Open-Meteo is free and keyless today. This file is reserved for
# future keyed providers (e.g. WAQI) if the project adds a second source.
DEFAULT_LOCATION=Delhi
DATABASE_PATH=db/aqi_system.db
LOG_LEVEL=INFO
```

- [ ] **Step 3: Create `.env` (copy of `.env.example`, not committed)**

```bash
cp .env.example .env
```

- [ ] **Step 4: Create `.gitignore`**

```
.env
__pycache__/
*.pyc
venv/
.venv/
db/*.db
logs/*.log
.DS_Store
*.egg-info/
```

- [ ] **Step 5: Create `pyproject.toml` (so `pytest` finds packages regardless of cwd)**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 6: Create placeholder directories**

```bash
mkdir -p db logs models/saved_models anomaly visualization pages notebooks
touch db/.gitkeep logs/.gitkeep models/saved_models/.gitkeep anomaly/.gitkeep visualization/.gitkeep pages/.gitkeep notebooks/.gitkeep
mkdir -p data tests
touch data/__init__.py tests/__init__.py
```

- [ ] **Step 7: Create `utils/__init__.py`**

```python
```

(empty file — makes `utils` a package)

- [ ] **Step 8: Create `utils/config.py`**

```python
import os

from dotenv import load_dotenv

load_dotenv()

# city -> (latitude, longitude)
LOCATIONS: dict[str, tuple[float, float]] = {
    "Delhi": (28.7041, 77.1025),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Pune": (18.5204, 73.8567),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Ahmedabad": (23.0225, 72.5714),
}

DEFAULT_LOCATION = os.getenv("DEFAULT_LOCATION", "Delhi")
DATABASE_PATH = os.getenv("DATABASE_PATH", "db/aqi_system.db")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
```

- [ ] **Step 9: Create `utils/logging_config.py`**

```python
import logging
import os

from utils.config import LOG_LEVEL

os.makedirs("logs", exist_ok=True)

_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL, logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[
                logging.FileHandler("logs/app.log"),
                logging.StreamHandler(),
            ],
        )
        _configured = True
    return logging.getLogger(name)
```

- [ ] **Step 10: Verify config imports cleanly**

Run: `python -c "from utils.config import LOCATIONS, DEFAULT_LOCATION; print(DEFAULT_LOCATION, LOCATIONS['Delhi'])"`
Expected: prints `Delhi (28.7041, 77.1025)`

- [ ] **Step 11: Init git and commit**

```bash
git init
git add requirements.txt .env.example .gitignore pyproject.toml utils/ data/__init__.py tests/__init__.py db/.gitkeep logs/.gitkeep models/saved_models/.gitkeep anomaly/.gitkeep visualization/.gitkeep pages/.gitkeep notebooks/.gitkeep docs/
git commit -m "chore: project scaffolding, config, logging"
```

---

### Task 2: Weather API Client

**Files:**
- Create: `data/api_client.py`
- Test: `tests/test_api_client.py`

**Interfaces:**
- Consumes: `utils.config.WEATHER_API_URL`, `utils.logging_config.get_logger`
- Produces: `fetch_current_weather(latitude: float, longitude: float) -> dict | None` — keys: `timestamp` (str, ISO), `latitude`, `longitude`, `temperature`, `feels_like`, `humidity`, `pressure`, `wind_speed`, `wind_direction`, `rainfall`, `visibility`, `cloud_cover`, `uv_index` (all `float | None` except timestamp)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_client.py
from unittest.mock import MagicMock, patch

SAMPLE_WEATHER_RESPONSE = {
    "current": {
        "time": "2026-08-19T12:00",
        "temperature_2m": 33.5,
        "relative_humidity_2m": 55,
        "apparent_temperature": 36.1,
        "precipitation": 0.0,
        "cloud_cover": 20,
        "pressure_msl": 1005.2,
        "wind_speed_10m": 12.3,
        "wind_direction_10m": 210,
        "visibility": 8000,
        "uv_index": 6.5,
    }
}


@patch("data.api_client.requests.get")
def test_fetch_current_weather_parses_response(mock_get):
    from data.api_client import fetch_current_weather

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = SAMPLE_WEATHER_RESPONSE
    mock_get.return_value = mock_response

    result = fetch_current_weather(28.7041, 77.1025)

    assert result is not None
    assert result["temperature"] == 33.5
    assert result["humidity"] == 55
    assert result["wind_speed"] == 12.3
    assert result["latitude"] == 28.7041
    assert result["longitude"] == 77.1025
    assert result["timestamp"] == "2026-08-19T12:00"


@patch("data.api_client.requests.get")
def test_fetch_current_weather_returns_none_after_retries_exhausted(mock_get):
    import requests

    from data.api_client import fetch_current_weather

    mock_get.side_effect = requests.exceptions.Timeout("timed out")

    result = fetch_current_weather(28.7041, 77.1025)

    assert result is None
    assert mock_get.call_count == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.api_client'`

- [ ] **Step 3: Write `data/api_client.py`**

```python
import time

import requests

from utils.config import WEATHER_API_URL
from utils.logging_config import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3
TIMEOUT_SECONDS = 10


def _get_with_retry(url: str, params: dict) -> dict | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            logger.warning("Request attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    logger.error("All %d retries failed for %s", MAX_RETRIES, url)
    return None


def fetch_current_weather(latitude: float, longitude: float) -> dict | None:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "cloud_cover",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
            "visibility",
            "uv_index",
        ]),
        "timezone": "auto",
    }
    payload = _get_with_retry(WEATHER_API_URL, params)
    if payload is None or "current" not in payload:
        return None

    current = payload["current"]
    return {
        "timestamp": current.get("time"),
        "latitude": latitude,
        "longitude": longitude,
        "temperature": current.get("temperature_2m"),
        "feels_like": current.get("apparent_temperature"),
        "humidity": current.get("relative_humidity_2m"),
        "pressure": current.get("pressure_msl"),
        "wind_speed": current.get("wind_speed_10m"),
        "wind_direction": current.get("wind_direction_10m"),
        "rainfall": current.get("precipitation"),
        "visibility": current.get("visibility"),
        "cloud_cover": current.get("cloud_cover"),
        "uv_index": current.get("uv_index"),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_client.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add data/api_client.py tests/test_api_client.py
git commit -m "feat: add weather API client with retry/timeout"
```

---

### Task 3: Air Quality API Client + CPCB AQI Calculator

**Files:**
- Create: `data/aqi_calculator.py`
- Modify: `data/api_client.py` (add `fetch_current_air_quality`)
- Test: `tests/test_aqi_calculator.py`
- Test: `tests/test_api_client.py` (append air-quality tests)

**Interfaces:**
- Consumes: `data.api_client._get_with_retry` (from Task 2)
- Produces: `fetch_current_air_quality(latitude, longitude) -> dict | None` (keys: `timestamp`, `latitude`, `longitude`, `pm25`, `pm10`, `co`, `no2`, `so2`, `o3`); `calculate_cpcb_aqi(pm25, pm10, co, no2, so2, o3) -> tuple[int | None, str | None]` (returns `(aqi, dominant_pollutant)`, all inputs `float | None`); `aqi_category(aqi: int) -> str`

- [ ] **Step 1: Write the failing test for the AQI calculator**

```python
# tests/test_aqi_calculator.py
from data.aqi_calculator import aqi_category, calculate_cpcb_aqi


def test_calculate_cpcb_aqi_pm25_only():
    # PM2.5 = 15 falls in the 0-30 -> 0-50 bracket: Ip = (50/30)*15 = 25
    aqi, dominant = calculate_cpcb_aqi(pm25=15, pm10=None, co=None, no2=None, so2=None, o3=None)
    assert aqi == 25
    assert dominant == "pm25"


def test_calculate_cpcb_aqi_takes_worst_pollutant():
    # PM2.5=15 (-> 25), PM10=300 (301-350 -> 201-300 bracket, high sub-index)
    aqi, dominant = calculate_cpcb_aqi(pm25=15, pm10=300, co=None, no2=None, so2=None, o3=None)
    assert dominant == "pm10"
    assert aqi > 25


def test_calculate_cpcb_aqi_returns_none_when_no_pollutants():
    aqi, dominant = calculate_cpcb_aqi(pm25=None, pm10=None, co=None, no2=None, so2=None, o3=None)
    assert aqi is None
    assert dominant is None


def test_aqi_category_boundaries():
    assert aqi_category(30) == "Good"
    assert aqi_category(75) == "Satisfactory"
    assert aqi_category(150) == "Moderate"
    assert aqi_category(250) == "Poor"
    assert aqi_category(350) == "Very Poor"
    assert aqi_category(450) == "Severe"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aqi_calculator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.aqi_calculator'`

- [ ] **Step 3: Write `data/aqi_calculator.py`**

```python
"""CPCB National Air Quality Index calculation.

Breakpoints are from CPCB's National AQI (2014) sub-index tables, in
µg/m3 for PM2.5, PM10, NO2, SO2, O3 (24-hr avg, O3 8-hr avg) and mg/m3
for CO (8-hr avg). Overall AQI is the maximum of the available
sub-indices, per CPCB methodology. Open-Meteo returns CO in µg/m3, so
it is converted to mg/m3 before lookup.

Each tuple is (concentration_low, concentration_high, index_low, index_high).
"""

PM25_BREAKPOINTS = [
    (0, 30, 0, 50),
    (30, 60, 50, 100),
    (60, 90, 100, 200),
    (90, 120, 200, 300),
    (120, 250, 300, 400),
    (250, 380, 400, 500),
]
PM10_BREAKPOINTS = [
    (0, 50, 0, 50),
    (50, 100, 50, 100),
    (100, 250, 100, 200),
    (250, 350, 200, 300),
    (350, 430, 300, 400),
    (430, 510, 400, 500),
]
NO2_BREAKPOINTS = [
    (0, 40, 0, 50),
    (40, 80, 50, 100),
    (80, 180, 100, 200),
    (180, 280, 200, 300),
    (280, 400, 300, 400),
    (400, 800, 400, 500),
]
SO2_BREAKPOINTS = [
    (0, 40, 0, 50),
    (40, 80, 50, 100),
    (80, 380, 100, 200),
    (380, 800, 200, 300),
    (800, 1600, 300, 400),
    (1600, 2100, 400, 500),
]
O3_BREAKPOINTS = [
    (0, 50, 0, 50),
    (50, 100, 50, 100),
    (100, 168, 100, 200),
    (168, 208, 200, 300),
    (208, 748, 300, 400),
    (748, 1000, 400, 500),
]
CO_BREAKPOINTS_MG = [
    (0, 1.0, 0, 50),
    (1.0, 2.0, 50, 100),
    (2.0, 10.0, 100, 200),
    (10.0, 17.0, 200, 300),
    (17.0, 34.0, 300, 400),
    (34.0, 50.0, 400, 500),
]

CATEGORY_THRESHOLDS = [
    (50, "Good"),
    (100, "Satisfactory"),
    (200, "Moderate"),
    (300, "Poor"),
    (400, "Very Poor"),
    (500, "Severe"),
]


def _sub_index(concentration: float, breakpoints: list[tuple[float, float, float, float]]) -> float:
    if concentration <= 0:
        concentration = 0
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= concentration <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo
    # above the top bracket: clamp to the max index
    return breakpoints[-1][3]


def calculate_cpcb_aqi(
    pm25: float | None,
    pm10: float | None,
    co: float | None,
    no2: float | None,
    so2: float | None,
    o3: float | None,
) -> tuple[int | None, str | None]:
    sub_indices: dict[str, float] = {}
    if pm25 is not None:
        sub_indices["pm25"] = _sub_index(pm25, PM25_BREAKPOINTS)
    if pm10 is not None:
        sub_indices["pm10"] = _sub_index(pm10, PM10_BREAKPOINTS)
    if no2 is not None:
        sub_indices["no2"] = _sub_index(no2, NO2_BREAKPOINTS)
    if so2 is not None:
        sub_indices["so2"] = _sub_index(so2, SO2_BREAKPOINTS)
    if o3 is not None:
        sub_indices["o3"] = _sub_index(o3, O3_BREAKPOINTS)
    if co is not None:
        sub_indices["co"] = _sub_index(co / 1000.0, CO_BREAKPOINTS_MG)

    if not sub_indices:
        return None, None

    dominant_pollutant = max(sub_indices, key=sub_indices.get)
    return round(sub_indices[dominant_pollutant]), dominant_pollutant


def aqi_category(aqi: int) -> str:
    for threshold, category in CATEGORY_THRESHOLDS:
        if aqi <= threshold:
            return category
    return "Severe"
```

- [ ] **Step 4: Run AQI calculator tests to verify they pass**

Run: `pytest tests/test_aqi_calculator.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Append air-quality client tests to `tests/test_api_client.py`**

```python
SAMPLE_AIR_QUALITY_RESPONSE = {
    "current": {
        "time": "2026-08-19T12:00",
        "pm2_5": 85.0,
        "pm10": 140.0,
        "carbon_monoxide": 500.0,
        "nitrogen_dioxide": 35.0,
        "sulphur_dioxide": 10.0,
        "ozone": 40.0,
    }
}


@patch("data.api_client.requests.get")
def test_fetch_current_air_quality_parses_response(mock_get):
    from data.api_client import fetch_current_air_quality

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = SAMPLE_AIR_QUALITY_RESPONSE
    mock_get.return_value = mock_response

    result = fetch_current_air_quality(28.7041, 77.1025)

    assert result is not None
    assert result["pm25"] == 85.0
    assert result["pm10"] == 140.0
    assert result["co"] == 500.0
    assert result["no2"] == 35.0
    assert result["so2"] == 10.0
    assert result["o3"] == 40.0
```

- [ ] **Step 6: Add `fetch_current_air_quality` to `data/api_client.py`**

```python
from utils.config import AIR_QUALITY_API_URL, WEATHER_API_URL  # replaces the WEATHER_API_URL-only import


def fetch_current_air_quality(latitude: float, longitude: float) -> dict | None:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join([
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
    if payload is None or "current" not in payload:
        return None

    current = payload["current"]
    return {
        "timestamp": current.get("time"),
        "latitude": latitude,
        "longitude": longitude,
        "pm25": current.get("pm2_5"),
        "pm10": current.get("pm10"),
        "co": current.get("carbon_monoxide"),
        "no2": current.get("nitrogen_dioxide"),
        "so2": current.get("sulphur_dioxide"),
        "o3": current.get("ozone"),
    }
```

- [ ] **Step 7: Run full test file to verify it passes**

Run: `pytest tests/test_api_client.py -v`
Expected: PASS (3 tests)

- [ ] **Step 8: Commit**

```bash
git add data/api_client.py data/aqi_calculator.py tests/test_api_client.py tests/test_aqi_calculator.py
git commit -m "feat: add air quality client and CPCB AQI calculator"
```

---

### Task 4: Database Layer

**Files:**
- Create: `data/database.py`
- Test: `tests/test_database.py`

**Interfaces:**
- Produces: ORM models `WeatherData`, `AirQualityData`, `Prediction`, `Anomaly` (all with the fields listed in the spec's schema section); `get_engine(db_path: str | None = None) -> Engine`; `init_db(engine) -> None`; `insert_weather_record(engine, record: dict) -> None`; `insert_air_quality_record(engine, record: dict) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_database.py
from datetime import datetime

from data.database import (
    AirQualityData,
    WeatherData,
    get_engine,
    init_db,
    insert_air_quality_record,
    insert_weather_record,
)
from sqlalchemy.orm import Session


def test_insert_and_query_weather_record():
    engine = get_engine(":memory:")
    init_db(engine)

    insert_weather_record(engine, {
        "timestamp": datetime(2026, 8, 19, 12, 0, 0),
        "location": "Delhi",
        "latitude": 28.7041,
        "longitude": 77.1025,
        "temperature": 33.5,
        "feels_like": 36.1,
        "humidity": 55,
        "pressure": 1005.2,
        "wind_speed": 12.3,
        "wind_direction": 210,
        "rainfall": 0.0,
        "visibility": 8000,
        "cloud_cover": 20,
        "uv_index": 6.5,
    })

    with Session(engine) as session:
        rows = session.query(WeatherData).all()
        assert len(rows) == 1
        assert rows[0].location == "Delhi"
        assert rows[0].temperature == 33.5


def test_insert_and_query_air_quality_record():
    engine = get_engine(":memory:")
    init_db(engine)

    insert_air_quality_record(engine, {
        "timestamp": datetime(2026, 8, 19, 12, 0, 0),
        "location": "Delhi",
        "pm25": 85.0,
        "pm10": 140.0,
        "co": 500.0,
        "no2": 35.0,
        "so2": 10.0,
        "o3": 40.0,
        "aqi": 158,
    })

    with Session(engine) as session:
        rows = session.query(AirQualityData).all()
        assert len(rows) == 1
        assert rows[0].aqi == 158
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_database.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.database'`

- [ ] **Step 3: Write `data/database.py`**

```python
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session

from utils.config import DATABASE_PATH


class Base(DeclarativeBase):
    pass


class WeatherData(Base):
    __tablename__ = "weather_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    location = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    temperature = Column(Float)
    feels_like = Column(Float)
    humidity = Column(Float)
    pressure = Column(Float)
    wind_speed = Column(Float)
    wind_direction = Column(Float)
    rainfall = Column(Float)
    visibility = Column(Float)
    cloud_cover = Column(Float)
    uv_index = Column(Float)


class AirQualityData(Base):
    __tablename__ = "air_quality_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    location = Column(String, nullable=False)
    pm25 = Column(Float)
    pm10 = Column(Float)
    co = Column(Float)
    no2 = Column(Float)
    so2 = Column(Float)
    o3 = Column(Float)
    aqi = Column(Integer)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    location = Column(String, nullable=False)
    prediction_horizon = Column(String, nullable=False)
    predicted_aqi = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)


class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    location = Column(String, nullable=False)
    metric = Column(String, nullable=False)
    observed_value = Column(Float, nullable=False)
    expected_value = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    severity = Column(String, nullable=False)


def get_engine(db_path: str | None = None):
    path = db_path or DATABASE_PATH
    return create_engine(f"sqlite:///{path}")


def init_db(engine) -> None:
    Base.metadata.create_all(engine)


def insert_weather_record(engine, record: dict) -> None:
    with Session(engine) as session:
        session.add(WeatherData(**record))
        session.commit()


def insert_air_quality_record(engine, record: dict) -> None:
    with Session(engine) as session:
        session.add(AirQualityData(**record))
        session.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_database.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add data/database.py tests/test_database.py
git commit -m "feat: add SQLAlchemy models and database layer"
```

---

### Task 5: Data Validation & Cleaning

**Files:**
- Create: `data/data_cleaning.py`
- Test: `tests/test_data_cleaning.py`

**Interfaces:**
- Consumes: `data.aqi_calculator.calculate_cpcb_aqi` (Task 3)
- Produces: `validate_weather_record(record: dict) -> bool`; `validate_air_quality_record(record: dict) -> bool`; `clean_weather_record(record: dict, location: str) -> dict` (ready for `insert_weather_record`; `timestamp` is a `datetime` object, not a string); `clean_air_quality_record(record: dict, location: str) -> dict` (ready for `insert_air_quality_record`, includes computed `aqi`; `timestamp` is a `datetime` object)

**Note:** the API clients return `timestamp` as an ISO string (e.g. `"2026-08-19T12:00"`, no seconds). The DB `timestamp` columns are `DateTime`, and SQLite's datetime parsing on read-back is picky about string format — so cleaning converts the string to a real `datetime` object via `datetime.fromisoformat` before it ever reaches the database. This avoids a silent parse failure later when Plan 2 queries historical rows for training.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_cleaning.py
from datetime import datetime

from data.data_cleaning import (
    clean_air_quality_record,
    clean_weather_record,
    validate_air_quality_record,
    validate_weather_record,
)


def test_validate_weather_record_requires_core_fields():
    valid = {"timestamp": "2026-08-19T12:00", "latitude": 28.7, "longitude": 77.1, "temperature": 30.0}
    invalid = {"timestamp": "2026-08-19T12:00", "latitude": 28.7, "longitude": 77.1, "temperature": None}

    assert validate_weather_record(valid) is True
    assert validate_weather_record(invalid) is False


def test_clean_weather_record_preserves_none_for_missing_optional_fields():
    raw = {
        "timestamp": "2026-08-19T12:00",
        "latitude": 28.7041,
        "longitude": 77.1025,
        "temperature": 33.456,
        "feels_like": None,
        "humidity": 55,
        "pressure": 1005.2,
        "wind_speed": 12.3,
        "wind_direction": 210,
        "rainfall": 0.0,
        "visibility": None,
        "cloud_cover": 20,
        "uv_index": 6.5,
    }
    cleaned = clean_weather_record(raw, location="Delhi")

    assert cleaned["location"] == "Delhi"
    assert cleaned["temperature"] == 33.46
    assert cleaned["feels_like"] is None
    assert cleaned["visibility"] is None
    assert cleaned["timestamp"] == datetime(2026, 8, 19, 12, 0)


def test_validate_air_quality_record_requires_at_least_one_pollutant():
    valid = {"timestamp": "2026-08-19T12:00", "pm25": 85.0, "pm10": None, "co": None, "no2": None, "so2": None, "o3": None}
    invalid = {"timestamp": "2026-08-19T12:00", "pm25": None, "pm10": None, "co": None, "no2": None, "so2": None, "o3": None}

    assert validate_air_quality_record(valid) is True
    assert validate_air_quality_record(invalid) is False


def test_clean_air_quality_record_computes_aqi():
    raw = {
        "timestamp": "2026-08-19T12:00",
        "pm25": 85.0,
        "pm10": 140.0,
        "co": 500.0,
        "no2": 35.0,
        "so2": 10.0,
        "o3": 40.0,
    }
    cleaned = clean_air_quality_record(raw, location="Delhi")

    assert cleaned["location"] == "Delhi"
    assert cleaned["aqi"] is not None
    assert isinstance(cleaned["aqi"], int)
    assert cleaned["timestamp"] == datetime(2026, 8, 19, 12, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_cleaning.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.data_cleaning'`

- [ ] **Step 3: Write `data/data_cleaning.py`**

```python
from datetime import datetime

from data.aqi_calculator import calculate_cpcb_aqi
from utils.logging_config import get_logger

logger = get_logger(__name__)

REQUIRED_WEATHER_FIELDS = ["timestamp", "latitude", "longitude", "temperature"]
POLLUTANT_FIELDS = ["pm25", "pm10", "co", "no2", "so2", "o3"]


def validate_weather_record(record: dict) -> bool:
    for field in REQUIRED_WEATHER_FIELDS:
        if record.get(field) is None:
            logger.warning("Weather record missing required field '%s'", field)
            return False
    return True


def validate_air_quality_record(record: dict) -> bool:
    if record.get("timestamp") is None:
        logger.warning("Air quality record missing timestamp")
        return False
    if all(record.get(field) is None for field in POLLUTANT_FIELDS):
        logger.warning("Air quality record has no pollutant readings at all")
        return False
    return True


def _round_or_none(value, digits=2):
    return round(value, digits) if value is not None else None


def clean_weather_record(record: dict, location: str) -> dict:
    return {
        "timestamp": datetime.fromisoformat(record["timestamp"]),
        "location": location,
        "latitude": record["latitude"],
        "longitude": record["longitude"],
        "temperature": _round_or_none(record.get("temperature")),
        "feels_like": _round_or_none(record.get("feels_like")),
        "humidity": _round_or_none(record.get("humidity")),
        "pressure": _round_or_none(record.get("pressure")),
        "wind_speed": _round_or_none(record.get("wind_speed")),
        "wind_direction": _round_or_none(record.get("wind_direction")),
        "rainfall": _round_or_none(record.get("rainfall")),
        "visibility": _round_or_none(record.get("visibility")),
        "cloud_cover": _round_or_none(record.get("cloud_cover")),
        "uv_index": _round_or_none(record.get("uv_index")),
    }


def clean_air_quality_record(record: dict, location: str) -> dict:
    aqi, _dominant_pollutant = calculate_cpcb_aqi(
        pm25=record.get("pm25"),
        pm10=record.get("pm10"),
        co=record.get("co"),
        no2=record.get("no2"),
        so2=record.get("so2"),
        o3=record.get("o3"),
    )
    return {
        "timestamp": datetime.fromisoformat(record["timestamp"]),
        "location": location,
        "pm25": _round_or_none(record.get("pm25")),
        "pm10": _round_or_none(record.get("pm10")),
        "co": _round_or_none(record.get("co")),
        "no2": _round_or_none(record.get("no2")),
        "so2": _round_or_none(record.get("so2")),
        "o3": _round_or_none(record.get("o3")),
        "aqi": aqi,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_cleaning.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add data/data_cleaning.py tests/test_data_cleaning.py
git commit -m "feat: add validation and cleaning for weather and air quality records"
```

---

### Task 6: Feature Engineering

**Files:**
- Create: `data/feature_engineering.py`
- Test: `tests/test_feature_engineering.py`

**Interfaces:**
- Produces: `add_time_features(df: pd.DataFrame) -> pd.DataFrame`; `add_lag_features(df: pd.DataFrame, columns: list[str], lags: list[int]) -> pd.DataFrame`; `add_rolling_features(df: pd.DataFrame, columns: list[str], windows: list[int]) -> pd.DataFrame`; `build_features(df: pd.DataFrame) -> pd.DataFrame` (orchestrates all three; requires a `timestamp` column, sorts ascending first, never shuffles)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_feature_engineering.py
import pandas as pd

from data.feature_engineering import add_lag_features, add_rolling_features, add_time_features, build_features


def _hourly_df(n=10):
    timestamps = pd.date_range("2026-08-19 00:00", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "aqi": [100 + i * 10 for i in range(n)],
        "pm25": [50 + i * 5 for i in range(n)],
    })


def test_add_time_features():
    df = add_time_features(_hourly_df())

    assert list(df["hour"][:3]) == [0, 1, 2]
    assert "day_of_week" in df.columns
    assert "is_weekend" in df.columns


def test_add_lag_features_shifts_correctly():
    df = add_lag_features(_hourly_df(), columns=["aqi"], lags=[1])

    assert pd.isna(df["aqi_lag_1"].iloc[0])
    assert df["aqi_lag_1"].iloc[1] == 100  # aqi at row 0


def test_add_rolling_features_requires_full_window():
    df = add_rolling_features(_hourly_df(), columns=["aqi"], windows=[3])

    assert pd.isna(df["aqi_rolling_3"].iloc[1])  # only 2 points so far
    assert df["aqi_rolling_3"].iloc[2] == (100 + 110 + 120) / 3


def test_build_features_sorts_by_timestamp_before_computing():
    df = _hourly_df()
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)

    result = build_features(shuffled)

    assert list(result["timestamp"]) == list(df["timestamp"])
    assert result["aqi_lag_1"].iloc[1] == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_feature_engineering.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.feature_engineering'`

- [ ] **Step 3: Write `data/feature_engineering.py`**

```python
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    return df


def add_lag_features(df: pd.DataFrame, columns: list[str], lags: list[int]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        for lag in lags:
            df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, columns: list[str], windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        for window in windows:
            df[f"{column}_rolling_{window}"] = (
                df[column].rolling(window=window, min_periods=window).mean()
            )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_time_features(df)
    df = add_lag_features(df, columns=["aqi", "pm25", "pm10"], lags=[1, 3, 6])
    df = add_rolling_features(df, columns=["aqi", "pm25"], windows=[3, 6])
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_feature_engineering.py -v`
Expected: PASS (4 tests)

Note: `build_features` references `pm10` lag columns even though the test fixture only has `aqi`/`pm25` — that's fine for this task (pandas raises `KeyError` only if the column is missing when *actually called via `build_features`*, so step 5 below adds a `pm10` column to the fixture used for that specific test).

- [ ] **Step 5: Fix `test_build_features_sorts_by_timestamp_before_computing` to include `pm10`**

```python
def test_build_features_sorts_by_timestamp_before_computing():
    df = _hourly_df()
    df["pm10"] = df["pm25"] + 20
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)

    result = build_features(shuffled)

    assert list(result["timestamp"]) == list(df["timestamp"])
    assert result["aqi_lag_1"].iloc[1] == 100
```

- [ ] **Step 6: Re-run full test file to verify it passes**

Run: `pytest tests/test_feature_engineering.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```bash
git add data/feature_engineering.py tests/test_feature_engineering.py
git commit -m "feat: add time, lag, and rolling feature engineering"
```

---

### Task 7: End-to-End Collector Script

**Files:**
- Create: `collect.py`
- Test: `tests/test_collect.py`

**Interfaces:**
- Consumes: `data.api_client.fetch_current_weather`, `data.api_client.fetch_current_air_quality`, `data.data_cleaning.validate_weather_record`, `data.data_cleaning.validate_air_quality_record`, `data.data_cleaning.clean_weather_record`, `data.data_cleaning.clean_air_quality_record`, `data.database.get_engine`, `data.database.init_db`, `data.database.insert_weather_record`, `data.database.insert_air_quality_record`, `utils.config.LOCATIONS`
- Produces: `collect_for_location(location: str, engine=None) -> bool` (returns `True` if both records were stored, `False` on any failure; never raises)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collect.py
from unittest.mock import patch

from data.database import AirQualityData, WeatherData, get_engine, init_db
from sqlalchemy.orm import Session


@patch("collect.fetch_current_air_quality")
@patch("collect.fetch_current_weather")
def test_collect_for_location_stores_both_records(mock_weather, mock_air_quality):
    from collect import collect_for_location

    mock_weather.return_value = {
        "timestamp": "2026-08-19T12:00",
        "latitude": 28.7041,
        "longitude": 77.1025,
        "temperature": 33.5,
        "feels_like": 36.1,
        "humidity": 55,
        "pressure": 1005.2,
        "wind_speed": 12.3,
        "wind_direction": 210,
        "rainfall": 0.0,
        "visibility": 8000,
        "cloud_cover": 20,
        "uv_index": 6.5,
    }
    mock_air_quality.return_value = {
        "timestamp": "2026-08-19T12:00",
        "pm25": 85.0,
        "pm10": 140.0,
        "co": 500.0,
        "no2": 35.0,
        "so2": 10.0,
        "o3": 40.0,
    }

    engine = get_engine(":memory:")
    init_db(engine)

    result = collect_for_location("Delhi", engine=engine)

    assert result is True
    with Session(engine) as session:
        assert session.query(WeatherData).count() == 1
        assert session.query(AirQualityData).count() == 1


@patch("collect.fetch_current_air_quality")
@patch("collect.fetch_current_weather")
def test_collect_for_location_returns_false_on_api_failure(mock_weather, mock_air_quality):
    from collect import collect_for_location

    mock_weather.return_value = None
    mock_air_quality.return_value = None

    engine = get_engine(":memory:")
    init_db(engine)

    result = collect_for_location("Delhi", engine=engine)

    assert result is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_collect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'collect'`

- [ ] **Step 3: Write `collect.py`**

```python
import sys

from data.api_client import fetch_current_air_quality, fetch_current_weather
from data.data_cleaning import (
    clean_air_quality_record,
    clean_weather_record,
    validate_air_quality_record,
    validate_weather_record,
)
from data.database import get_engine, init_db, insert_air_quality_record, insert_weather_record
from utils.config import DEFAULT_LOCATION, LOCATIONS
from utils.logging_config import get_logger

logger = get_logger(__name__)


def collect_for_location(location: str, engine=None) -> bool:
    if location not in LOCATIONS:
        logger.error("Unknown location '%s'", location)
        return False

    latitude, longitude = LOCATIONS[location]
    engine = engine or get_engine()
    init_db(engine)

    weather_raw = fetch_current_weather(latitude, longitude)
    air_quality_raw = fetch_current_air_quality(latitude, longitude)

    if weather_raw is None or air_quality_raw is None:
        logger.error("Live data temporarily unavailable for %s", location)
        return False

    if not validate_weather_record(weather_raw) or not validate_air_quality_record(air_quality_raw):
        logger.error("Validation failed for %s, discarding this fetch", location)
        return False

    weather_clean = clean_weather_record(weather_raw, location)
    air_quality_clean = clean_air_quality_record(air_quality_raw, location)

    insert_weather_record(engine, weather_clean)
    insert_air_quality_record(engine, air_quality_clean)
    logger.info("Stored weather + air quality for %s (AQI=%s)", location, air_quality_clean["aqi"])
    return True


if __name__ == "__main__":
    location = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOCATION
    success = collect_for_location(location)
    sys.exit(0 if success else 1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_collect.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: All tests pass (weather client, air quality client, AQI calculator, database, cleaning, feature engineering, collector)

- [ ] **Step 6: Manual verification against the real API**

```bash
python collect.py Delhi
sqlite3 db/aqi_system.db "select * from weather_data;"
sqlite3 db/aqi_system.db "select * from air_quality_data;"
```

Expected: one row in each table with real, current values for Delhi, and a `calculated AQI` that looks plausible (roughly 50–400 depending on the day).

- [ ] **Step 7: Commit**

```bash
git add collect.py tests/test_collect.py
git commit -m "feat: add end-to-end collector script wiring fetch to storage"
```

---

## What's next

This plan delivers a working, tested data pipeline (Objective 1) but stops before ML, anomaly detection, and the dashboard. Once this is verified, the next plans are:

1. **Historical backfill + ML** — pull historical data via Open-Meteo's archive API, train baseline/Linear Regression/Random Forest/XGBoost per horizon, evaluate, save the best model.
2. **Anomaly detection** — Isolation Forest module over stored data.
3. **Dashboard** — the 5 Streamlit pages, wired to the database and saved model.
