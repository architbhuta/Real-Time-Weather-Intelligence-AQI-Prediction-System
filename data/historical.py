import requests

from data.api_client import _get_with_retry
from utils.config import AIR_QUALITY_API_URL, HISTORICAL_WEATHER_API_URL


def _safe_index(values: list, i: int):
    """Safely index into a list, returning None if index is out of bounds."""
    return values[i] if i < len(values) else None


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
            "temperature": _safe_index(hourly.get("temperature_2m", []), i),
            "feels_like": _safe_index(hourly.get("apparent_temperature", []), i),
            "humidity": _safe_index(hourly.get("relative_humidity_2m", []), i),
            "pressure": _safe_index(hourly.get("pressure_msl", []), i),
            "wind_speed": _safe_index(hourly.get("wind_speed_10m", []), i),
            "wind_direction": _safe_index(hourly.get("wind_direction_10m", []), i),
            "rainfall": _safe_index(hourly.get("precipitation", []), i),
            "visibility": None,
            "cloud_cover": _safe_index(hourly.get("cloud_cover", []), i),
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
            "pm25": _safe_index(hourly.get("pm2_5", []), i),
            "pm10": _safe_index(hourly.get("pm10", []), i),
            "co": _safe_index(hourly.get("carbon_monoxide", []), i),
            "no2": _safe_index(hourly.get("nitrogen_dioxide", []), i),
            "so2": _safe_index(hourly.get("sulphur_dioxide", []), i),
            "o3": _safe_index(hourly.get("ozone", []), i),
        })
    return records
