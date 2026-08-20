import logging
from datetime import datetime

import pandas as pd

from data.database import (
    AirQualityData,
    Anomaly,
    WeatherData,
    get_engine,
    init_db,
    insert_air_quality_record,
    insert_anomalies,
    insert_weather_and_air_quality,
    insert_weather_record,
    load_weather_and_air_quality,
)
from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table
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


def test_load_weather_and_air_quality_joins_on_timestamp_and_location():
    engine = get_engine(":memory:")
    init_db(engine)

    insert_weather_and_air_quality(engine, {
        "timestamp": datetime(2026, 5, 1, 0, 0), "location": "Delhi",
        "latitude": 28.7041, "longitude": 77.1025, "temperature": 30.0,
        "feels_like": 32.0, "humidity": 55, "pressure": 1005.0, "wind_speed": 10.0,
        "wind_direction": 200, "rainfall": 0.0, "visibility": None,
        "cloud_cover": 20, "uv_index": None,
    }, {
        "timestamp": datetime(2026, 5, 1, 0, 0), "location": "Delhi",
        "pm25": 80.0, "pm10": 130.0, "co": 450.0, "no2": 30.0, "so2": 8.0, "o3": 35.0, "aqi": 158,
    })
    # A weather-only hour (no matching air-quality row) must not appear in the join
    insert_weather_record(engine, {
        "timestamp": datetime(2026, 5, 1, 1, 0), "location": "Delhi",
        "latitude": 28.7041, "longitude": 77.1025, "temperature": 29.0,
        "feels_like": 31.0, "humidity": 58, "pressure": 1005.5, "wind_speed": 9.0,
        "wind_direction": 195, "rainfall": 0.0, "visibility": None,
        "cloud_cover": 25, "uv_index": None,
    })

    df = load_weather_and_air_quality(engine, "Delhi")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["temperature"] == 30.0
    assert df.iloc[0]["aqi"] == 158


def test_load_weather_and_air_quality_returns_empty_frame_for_unknown_location():
    engine = get_engine(":memory:")
    init_db(engine)

    df = load_weather_and_air_quality(engine, "Nowhere")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def _anomaly_record(timestamp, metric="pm25"):
    return {
        "timestamp": timestamp, "location": "Delhi", "metric": metric,
        "observed_value": 250.0, "expected_value": 90.0,
        "anomaly_score": 2.3, "severity": "Medium",
    }


def test_insert_anomalies_stores_new_records():
    engine = get_engine(":memory:")
    init_db(engine)

    stored = insert_anomalies(engine, [
        _anomaly_record(datetime(2026, 5, 1, 0, 0)),
        _anomaly_record(datetime(2026, 5, 1, 1, 0)),
    ])

    assert stored == 2
    with Session(engine) as session:
        assert session.query(Anomaly).count() == 2


def test_insert_anomalies_skips_duplicates_on_rerun():
    engine = get_engine(":memory:")
    init_db(engine)

    first = insert_anomalies(engine, [_anomaly_record(datetime(2026, 5, 1, 0, 0))])
    second = insert_anomalies(engine, [_anomaly_record(datetime(2026, 5, 1, 0, 0))])

    assert first == 1
    assert second == 0
    with Session(engine) as session:
        assert session.query(Anomaly).count() == 1


def test_insert_anomalies_allows_different_metrics_at_same_timestamp():
    engine = get_engine(":memory:")
    init_db(engine)

    stored = insert_anomalies(engine, [
        _anomaly_record(datetime(2026, 5, 1, 0, 0), metric="pm25"),
        _anomaly_record(datetime(2026, 5, 1, 0, 0), metric="temperature"),
    ])

    assert stored == 2


def _create_anomalies_table_without_unique_constraint(engine):
    """Build an `anomalies` table shaped like an older schema version: same columns,
    no UniqueConstraint. `create_all` will leave it exactly as-is."""
    metadata = MetaData()
    Table(
        "anomalies",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("timestamp", DateTime, nullable=False),
        Column("location", String, nullable=False),
        Column("metric", String, nullable=False),
        Column("observed_value", Float, nullable=False),
        Column("expected_value", Float, nullable=False),
        Column("anomaly_score", Float, nullable=False),
        Column("severity", String, nullable=False),
    ).create(engine)


def test_init_db_warns_when_anomalies_table_is_missing_its_unique_constraint(caplog):
    engine = get_engine(":memory:")
    _create_anomalies_table_without_unique_constraint(engine)

    with caplog.at_level(logging.WARNING):
        init_db(engine)

    assert any(
        "missing its uniqueness constraint" in record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_init_db_does_not_warn_on_a_healthy_database(caplog):
    engine = get_engine(":memory:")
    init_db(engine)

    with caplog.at_level(logging.WARNING):
        init_db(engine)  # second call: the table now exists and is correct

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


from data.database import get_latest_reading, get_recent_anomalies


def test_get_latest_reading_returns_the_newest_row():
    engine = get_engine(":memory:")
    init_db(engine)
    insert_weather_and_air_quality(engine, {
        "timestamp": datetime(2026, 5, 1, 0, 0), "location": "Delhi",
        "latitude": 28.7041, "longitude": 77.1025, "temperature": 28.0,
        "feels_like": 30.0, "humidity": 50.0, "pressure": 1004.0, "wind_speed": 8.0,
        "wind_direction": 190.0, "rainfall": 0.0, "visibility": None,
        "cloud_cover": 15.0, "uv_index": None,
    }, {
        "timestamp": datetime(2026, 5, 1, 0, 0), "location": "Delhi",
        "pm25": 70.0, "pm10": 120.0, "co": 400.0, "no2": 28.0, "so2": 7.0, "o3": 30.0, "aqi": 140,
    })
    insert_weather_and_air_quality(engine, {
        "timestamp": datetime(2026, 5, 1, 1, 0), "location": "Delhi",
        "latitude": 28.7041, "longitude": 77.1025, "temperature": 29.0,
        "feels_like": 31.0, "humidity": 52.0, "pressure": 1004.5, "wind_speed": 9.0,
        "wind_direction": 195.0, "rainfall": 0.0, "visibility": None,
        "cloud_cover": 18.0, "uv_index": None,
    }, {
        "timestamp": datetime(2026, 5, 1, 1, 0), "location": "Delhi",
        "pm25": 85.0, "pm10": 140.0, "co": 420.0, "no2": 32.0, "so2": 9.0, "o3": 33.0, "aqi": 158,
    })

    reading = get_latest_reading(engine, "Delhi")

    assert reading is not None
    assert reading["aqi"] == 158
    assert reading["timestamp"] == datetime(2026, 5, 1, 1, 0)


def test_get_latest_reading_returns_none_for_unknown_location():
    engine = get_engine(":memory:")
    init_db(engine)

    assert get_latest_reading(engine, "Atlantis") is None


def test_get_recent_anomalies_orders_newest_first_and_respects_limit():
    engine = get_engine(":memory:")
    init_db(engine)
    insert_anomalies(engine, [
        {"timestamp": datetime(2026, 5, 1, 0, 0), "location": "Delhi", "metric": "pm25",
         "observed_value": 200.0, "expected_value": 80.0, "anomaly_score": 1.5, "severity": "Medium"},
        {"timestamp": datetime(2026, 5, 1, 2, 0), "location": "Delhi", "metric": "pm10",
         "observed_value": 300.0, "expected_value": 130.0, "anomaly_score": 2.5, "severity": "High"},
        {"timestamp": datetime(2026, 5, 1, 1, 0), "location": "Delhi", "metric": "aqi",
         "observed_value": 400.0, "expected_value": 150.0, "anomaly_score": 3.0, "severity": "High"},
    ])

    anomalies = get_recent_anomalies(engine, "Delhi", limit=2)

    assert len(anomalies) == 2
    assert list(anomalies["timestamp"]) == [datetime(2026, 5, 1, 2, 0), datetime(2026, 5, 1, 1, 0)]
