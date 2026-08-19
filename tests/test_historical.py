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
