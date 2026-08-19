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
