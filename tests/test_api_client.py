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
