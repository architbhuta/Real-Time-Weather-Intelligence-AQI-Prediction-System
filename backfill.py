import sys
from datetime import date, timedelta

from data.data_cleaning import (
    clean_air_quality_record,
    clean_weather_record,
    validate_air_quality_record,
    validate_weather_record,
)
from data.database import get_engine, init_db, insert_weather_and_air_quality
from data.historical import fetch_historical_air_quality, fetch_historical_weather
from utils.config import DEFAULT_LOCATION, LOCATIONS
from utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_BACKFILL_DAYS = 92


def backfill_location(location: str, days: int = DEFAULT_BACKFILL_DAYS, engine=None) -> int:
    if location not in LOCATIONS:
        logger.error("Unknown location '%s'", location)
        return 0

    latitude, longitude = LOCATIONS[location]
    # Open-Meteo's archive API lags ~1 day behind "today" (it doesn't yet have
    # data for the current day), so end_date intentionally targets yesterday.
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=days)

    engine = engine or get_engine()
    try:
        init_db(engine)
    except Exception as exc:
        logger.error("Failed to initialize database for backfill: %s", exc)
        return 0

    weather_records = fetch_historical_weather(latitude, longitude, start_date.isoformat(), end_date.isoformat())
    air_quality_records = fetch_historical_air_quality(latitude, longitude, start_date.isoformat(), end_date.isoformat())

    weather_by_timestamp = {r["timestamp"]: r for r in weather_records}
    air_quality_by_timestamp = {r["timestamp"]: r for r in air_quality_records}
    shared_timestamps = sorted(set(weather_by_timestamp) & set(air_quality_by_timestamp))

    stored = 0
    for timestamp in shared_timestamps:
        weather_raw = weather_by_timestamp[timestamp]
        air_quality_raw = air_quality_by_timestamp[timestamp]

        if not validate_weather_record(weather_raw) or not validate_air_quality_record(air_quality_raw):
            continue

        weather_clean = clean_weather_record(weather_raw, location)
        air_quality_clean = clean_air_quality_record(air_quality_raw, location)

        try:
            insert_weather_and_air_quality(engine, weather_clean, air_quality_clean)
            stored += 1
        except Exception as exc:
            logger.warning("Failed to store backfilled hour %s for %s: %s", timestamp, location, exc)

    logger.info(
        "Backfill for %s: processed %d/%d overlapping hours (%s to %s)",
        location, stored, len(shared_timestamps), start_date, end_date,
    )
    return stored


if __name__ == "__main__":
    location = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOCATION
    days = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BACKFILL_DAYS
    backfill_location(location, days)
