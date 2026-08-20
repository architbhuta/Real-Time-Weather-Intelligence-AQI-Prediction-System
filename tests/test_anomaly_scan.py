from datetime import datetime

import pandas as pd

from data.database import AirQualityData, WeatherData, get_engine, init_db
from sqlalchemy.orm import Session


def _seed_history_with_spike(engine, location="Delhi", n=50, spike_index=25):
    timestamps = pd.date_range("2026-05-01 00:00", periods=n, freq="h")
    with Session(engine) as session:
        for i, ts in enumerate(timestamps):
            pm25 = 80.0 + (i % 5)
            if i == spike_index:
                pm25 = 500.0
            session.add(WeatherData(
                timestamp=ts.to_pydatetime(), location=location, latitude=28.7041, longitude=77.1025,
                temperature=30.0, feels_like=32.0, humidity=55.0, pressure=1005.0,
                wind_speed=10.0, wind_direction=200.0, rainfall=0.0, visibility=None,
                cloud_cover=20.0, uv_index=None,
            ))
            session.add(AirQualityData(
                timestamp=ts.to_pydatetime(), location=location,
                pm25=pm25, pm10=130.0 + (i % 5), co=450.0, no2=30.0, so2=8.0, o3=35.0,
                aqi=150 + (i % 5),
            ))
        session.commit()


def test_scan_location_stores_detected_anomalies():
    from anomaly_scan import scan_location

    engine = get_engine(":memory:")
    init_db(engine)
    _seed_history_with_spike(engine)

    stored = scan_location("Delhi", engine=engine)

    assert stored > 0


def test_scan_location_is_idempotent_on_rerun():
    from anomaly_scan import scan_location

    engine = get_engine(":memory:")
    init_db(engine)
    _seed_history_with_spike(engine)

    first = scan_location("Delhi", engine=engine)
    second = scan_location("Delhi", engine=engine)

    assert first > 0
    assert second == 0  # same history, already-flagged points skipped


def test_scan_location_returns_zero_for_unknown_location():
    from anomaly_scan import scan_location

    engine = get_engine(":memory:")
    init_db(engine)

    stored = scan_location("Atlantis", engine=engine)

    assert stored == 0
