from unittest.mock import patch

from data.database import AirQualityData, WeatherData, get_engine, init_db
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

SAMPLE_WEATHER = {
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

SAMPLE_AIR_QUALITY = {
    "timestamp": "2026-08-19T12:00",
    "pm25": 85.0,
    "pm10": 140.0,
    "co": 500.0,
    "no2": 35.0,
    "so2": 10.0,
    "o3": 40.0,
}


@patch("collect.fetch_current_air_quality")
@patch("collect.fetch_current_weather")
def test_collect_for_location_stores_both_records(mock_weather, mock_air_quality):
    from collect import collect_for_location

    mock_weather.return_value = dict(SAMPLE_WEATHER)
    mock_air_quality.return_value = dict(SAMPLE_AIR_QUALITY)

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


@patch("collect.insert_weather_and_air_quality")
@patch("collect.fetch_current_air_quality")
@patch("collect.fetch_current_weather")
def test_collect_for_location_returns_false_on_db_write_failure(
    mock_weather, mock_air_quality, mock_insert
):
    from collect import collect_for_location

    mock_weather.return_value = dict(SAMPLE_WEATHER)
    mock_air_quality.return_value = dict(SAMPLE_AIR_QUALITY)
    mock_insert.side_effect = RuntimeError("disk full")

    engine = get_engine(":memory:")
    init_db(engine)

    result = collect_for_location("Delhi", engine=engine)

    assert result is False


@patch("collect.get_engine")
@patch("collect.fetch_current_air_quality")
@patch("collect.fetch_current_weather")
def test_collect_for_location_returns_false_when_engine_creation_fails(
    mock_weather, mock_air_quality, mock_get_engine
):
    from collect import collect_for_location

    mock_weather.return_value = dict(SAMPLE_WEATHER)
    mock_air_quality.return_value = dict(SAMPLE_AIR_QUALITY)
    mock_get_engine.side_effect = OperationalError("unable to open database file", {}, None)

    result = collect_for_location("Delhi")

    assert result is False


@patch("collect.init_db")
@patch("collect.fetch_current_air_quality")
@patch("collect.fetch_current_weather")
def test_collect_for_location_returns_false_when_init_db_fails(
    mock_weather, mock_air_quality, mock_init_db
):
    from collect import collect_for_location

    mock_weather.return_value = dict(SAMPLE_WEATHER)
    mock_air_quality.return_value = dict(SAMPLE_AIR_QUALITY)
    mock_init_db.side_effect = OperationalError("database is locked", {}, None)

    engine = get_engine(":memory:")

    result = collect_for_location("Delhi", engine=engine)

    assert result is False


@patch("collect.fetch_current_air_quality")
@patch("collect.fetch_current_weather")
def test_collect_for_location_skips_duplicate_observation(mock_weather, mock_air_quality):
    from collect import collect_for_location

    mock_weather.return_value = dict(SAMPLE_WEATHER)
    mock_air_quality.return_value = dict(SAMPLE_AIR_QUALITY)

    engine = get_engine(":memory:")
    init_db(engine)

    assert collect_for_location("Delhi", engine=engine) is True
    assert collect_for_location("Delhi", engine=engine) is True

    with Session(engine) as session:
        assert session.query(WeatherData).count() == 1
        assert session.query(AirQualityData).count() == 1
