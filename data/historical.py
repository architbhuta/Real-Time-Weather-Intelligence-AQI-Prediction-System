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
