import time

import requests

from utils.config import AIR_QUALITY_API_URL, WEATHER_API_URL
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
        except (requests.exceptions.RequestException, ValueError) as exc:
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
