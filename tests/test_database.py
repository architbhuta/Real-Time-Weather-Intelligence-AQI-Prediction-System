from datetime import datetime

from data.database import (
    AirQualityData,
    WeatherData,
    get_engine,
    init_db,
    insert_air_quality_record,
    insert_weather_and_air_quality,
    insert_weather_record,
)
from sqlalchemy.orm import Session

WEATHER_RECORD = {
    "timestamp": datetime(2026, 8, 19, 12, 0, 0),
    "location": "Delhi",
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

AIR_QUALITY_RECORD = {
    "timestamp": datetime(2026, 8, 19, 12, 0, 0),
    "location": "Delhi",
    "pm25": 85.0,
    "pm10": 140.0,
    "co": 500.0,
    "no2": 35.0,
    "so2": 10.0,
    "o3": 40.0,
    "aqi": 158,
}


def test_insert_and_query_weather_record():
    engine = get_engine(":memory:")
    init_db(engine)

    insert_weather_record(engine, {
        "timestamp": datetime(2026, 8, 19, 12, 0, 0),
        "location": "Delhi",
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
    })

    with Session(engine) as session:
        rows = session.query(WeatherData).all()
        assert len(rows) == 1
        assert rows[0].location == "Delhi"
        assert rows[0].temperature == 33.5


def test_insert_and_query_air_quality_record():
    engine = get_engine(":memory:")
    init_db(engine)

    insert_air_quality_record(engine, {
        "timestamp": datetime(2026, 8, 19, 12, 0, 0),
        "location": "Delhi",
        "pm25": 85.0,
        "pm10": 140.0,
        "co": 500.0,
        "no2": 35.0,
        "so2": 10.0,
        "o3": 40.0,
        "aqi": 158,
    })

    with Session(engine) as session:
        rows = session.query(AirQualityData).all()
        assert len(rows) == 1
        assert rows[0].aqi == 158


def test_duplicate_weather_record_is_a_silent_no_op():
    engine = get_engine(":memory:")
    init_db(engine)

    insert_weather_record(engine, dict(WEATHER_RECORD))
    insert_weather_record(engine, dict(WEATHER_RECORD))

    with Session(engine) as session:
        assert session.query(WeatherData).count() == 1


def test_duplicate_air_quality_record_is_a_silent_no_op():
    engine = get_engine(":memory:")
    init_db(engine)

    insert_air_quality_record(engine, dict(AIR_QUALITY_RECORD))
    insert_air_quality_record(engine, dict(AIR_QUALITY_RECORD))

    with Session(engine) as session:
        assert session.query(AirQualityData).count() == 1


def test_same_timestamp_different_location_is_allowed():
    engine = get_engine(":memory:")
    init_db(engine)

    insert_weather_record(engine, dict(WEATHER_RECORD))
    insert_weather_record(engine, {**WEATHER_RECORD, "location": "Mumbai"})

    with Session(engine) as session:
        assert session.query(WeatherData).count() == 2


def test_insert_weather_and_air_quality_stores_both():
    engine = get_engine(":memory:")
    init_db(engine)

    insert_weather_and_air_quality(engine, dict(WEATHER_RECORD), dict(AIR_QUALITY_RECORD))

    with Session(engine) as session:
        assert session.query(WeatherData).count() == 1
        assert session.query(AirQualityData).count() == 1


def test_insert_weather_and_air_quality_is_atomic():
    engine = get_engine(":memory:")
    init_db(engine)

    # The air-quality row is a duplicate, so the whole transaction rolls back
    # and no orphan weather row survives.
    insert_air_quality_record(engine, dict(AIR_QUALITY_RECORD))
    insert_weather_and_air_quality(engine, dict(WEATHER_RECORD), dict(AIR_QUALITY_RECORD))

    with Session(engine) as session:
        assert session.query(WeatherData).count() == 0
        assert session.query(AirQualityData).count() == 1
