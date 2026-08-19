from unittest.mock import patch

from data.database import AirQualityData, WeatherData, get_engine, init_db
from sqlalchemy.orm import Session


def _weather_record(timestamp):
    return {
        "timestamp": timestamp, "latitude": 28.7041, "longitude": 77.1025,
        "temperature": 30.0, "feels_like": 32.0, "humidity": 55, "pressure": 1005.0,
        "wind_speed": 10.0, "wind_direction": 200, "rainfall": 0.0,
        "visibility": None, "cloud_cover": 20, "uv_index": None,
    }


def _air_quality_record(timestamp):
    return {
        "timestamp": timestamp, "latitude": 28.7041, "longitude": 77.1025,
        "pm25": 80.0, "pm10": 130.0, "co": 450.0, "no2": 30.0, "so2": 8.0, "o3": 35.0,
    }


@patch("backfill.fetch_historical_air_quality")
@patch("backfill.fetch_historical_weather")
def test_backfill_location_stores_only_overlapping_timestamps(mock_weather, mock_air_quality):
    from backfill import backfill_location

    mock_weather.return_value = [
        _weather_record("2026-05-01T00:00"),
        _weather_record("2026-05-01T01:00"),
        _weather_record("2026-05-01T02:00"),  # no matching air-quality row
    ]
    mock_air_quality.return_value = [
        _air_quality_record("2026-05-01T00:00"),
        _air_quality_record("2026-05-01T01:00"),
        _air_quality_record("2026-05-01T03:00"),  # no matching weather row
    ]

    engine = get_engine(":memory:")
    init_db(engine)

    stored = backfill_location("Delhi", days=1, engine=engine)

    assert stored == 2
    with Session(engine) as session:
        assert session.query(WeatherData).count() == 2
        assert session.query(AirQualityData).count() == 2


@patch("backfill.fetch_historical_air_quality")
@patch("backfill.fetch_historical_weather")
def test_backfill_location_is_idempotent_on_rerun(mock_weather, mock_air_quality):
    from backfill import backfill_location

    mock_weather.return_value = [_weather_record("2026-05-01T00:00")]
    mock_air_quality.return_value = [_air_quality_record("2026-05-01T00:00")]

    engine = get_engine(":memory:")
    init_db(engine)

    first_run = backfill_location("Delhi", days=1, engine=engine)
    second_run = backfill_location("Delhi", days=1, engine=engine)

    assert first_run == 1
    assert second_run == 1  # counted as "processed", but stored as a no-op duplicate
    with Session(engine) as session:
        assert session.query(WeatherData).count() == 1


def test_backfill_location_returns_zero_for_unknown_location():
    from backfill import backfill_location

    engine = get_engine(":memory:")
    init_db(engine)

    stored = backfill_location("Atlantis", days=1, engine=engine)

    assert stored == 0
