from datetime import datetime

from data.database import (
    AirQualityData,
    WeatherData,
    get_engine,
    init_db,
    insert_air_quality_record,
    insert_weather_record,
)
from sqlalchemy.orm import Session


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
