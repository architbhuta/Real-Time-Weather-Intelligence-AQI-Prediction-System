# AirSense — Real-Time Weather Intelligence & AQI Prediction System

## Problem
Public AQI tools show current readings only, with no short-term forecast, no historical context, and no flag for unusual readings. This project builds an end-to-end pipeline — live data → database → ML forecast → anomaly detection → dashboard — for Indian cities, starting with Delhi.

## Objectives
1. Real-time data pipeline for weather + air quality (fetch, validate, clean, store).
2. Interactive dashboard for live conditions and historical trends.
3. Short-term AQI forecasting (+1h/+3h/+6h) via baseline → Linear Regression → Random Forest → XGBoost, evaluated chronologically.
4. Anomaly detection + plain-language insights.

## APIs
- **Weather:** Open-Meteo forecast (`api.open-meteo.com`) + historical archive (`archive-api.open-meteo.com`). Free, no key.
- **Air quality:** Open-Meteo air quality API (`air-quality-api.open-meteo.com`). Free, no key, gives raw pollutant concentrations (PM2.5, PM10, CO, NO2, SO2, O3) plus historical hourly data.
- One provider covers both live and historical needs for weather and pollutants — no key management, no multi-provider alignment.

## AQI Standard
AQI is **always calculated in-app** from raw pollutant concentrations using India's CPCB National AQI breakpoints. Open-Meteo's own `us_aqi`/`european_aqi` fields are ignored entirely to avoid mixing standards. Stored as `calculated_aqi` — never presented as API-provided.

## Architecture
```
Open-Meteo Weather API ─┐
                         ├─► Fetcher (retry/timeout) → Validate → Clean → Feature Engineering
Open-Meteo AQ API ───────┘                                              ↓
                                                                     SQLite
                                                    ┌────────────────────┴───────────────────┐
                                              Historical backfill                       Live inference
                                              (train models)                            (latest row)
                                                    ↓                                        ↓
                                     Baseline / LinReg / RF / XGBoost ──► saved model ──► predictions + anomalies
                                                                                              ↓
                                                                              Streamlit dashboard (5 pages)
```

## Tech Stack
Python 3.11+, requests, python-dotenv, SQLite + SQLAlchemy, pandas/numpy, scikit-learn, XGBoost, IsolationForest, Streamlit + Plotly + Folium, pytest.

## Database
`weather_data`, `air_quality_data`, `predictions`, `anomalies` — fields as listed in the original spec (timestamp, location, pollutant/weather columns, prediction_horizon, anomaly_score/severity).

## ML Approach
- Targets: AQI at t+1h, t+3h, t+6h — one model per horizon.
- Features: time (hour/day/dow/weekend), AQI/PM2.5/PM10 lags (1h/3h/6h), rolling means (3h/6h), current weather.
- Chronological 70/15/15 split, no shuffling.
- Compare MAE/RMSE/R²/MAPE on test set; best model on validation wins.
- Cold start: while historical backfill is incomplete, dashboard falls back to rolling-average baseline, clearly labeled as non-ML.

## Folder Structure
As in the original master prompt (section 26) — `app.py`, `pages/`, `data/`, `models/`, `anomaly/`, `visualization/`, `utils/`, `tests/`, `.env`/`.env.example`, `requirements.txt`, `README.md`.

## Roadmap
Setup → API integration → Database → Data processing → Historical backfill (Delhi) → ML (baseline/LR/RF/XGBoost) → Anomaly detection → Dashboard → Integration → Testing → Deployment. One phase at a time, tested before moving on.
