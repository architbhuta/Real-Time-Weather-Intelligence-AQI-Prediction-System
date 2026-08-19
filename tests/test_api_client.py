from unittest.mock import MagicMock, call, patch

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


@patch("data.api_client.time.sleep")
@patch("data.api_client.requests.get")
def test_fetch_current_weather_returns_none_after_retries_exhausted(mock_get, mock_sleep):
    import requests

    from data.api_client import fetch_current_weather

    mock_get.side_effect = requests.exceptions.Timeout("timed out")

    result = fetch_current_weather(28.7041, 77.1025)

    assert result is None
    assert mock_get.call_count == 3
    # Exponential backoff between attempts, and none after the last one.
    assert mock_sleep.call_args_list == [call(2), call(4)]


@patch("data.api_client.time.sleep")
@patch("data.api_client.requests.get")
def test_fetch_current_weather_handles_malformed_json(mock_get, mock_sleep):
    from data.api_client import fetch_current_weather

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")

    mock_get.return_value = mock_response

    result = fetch_current_weather(28.7041, 77.1025)

    assert result is None
    assert mock_get.call_count == 3
    assert mock_sleep.call_args_list == [call(2), call(4)]


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
