import sys

from data.api_client import fetch_current_air_quality, fetch_current_weather
from data.data_cleaning import (
    clean_air_quality_record,
    clean_weather_record,
    validate_air_quality_record,
    validate_weather_record,
)
from data.database import get_engine, init_db, insert_air_quality_record, insert_weather_record
from utils.config import DEFAULT_LOCATION, LOCATIONS
from utils.logging_config import get_logger

logger = get_logger(__name__)


def collect_for_location(location: str, engine=None) -> bool:
    if location not in LOCATIONS:
        logger.error("Unknown location '%s'", location)
        return False

    latitude, longitude = LOCATIONS[location]
    engine = engine or get_engine()
    init_db(engine)

    weather_raw = fetch_current_weather(latitude, longitude)
    air_quality_raw = fetch_current_air_quality(latitude, longitude)

    if weather_raw is None or air_quality_raw is None:
        logger.error("Live data temporarily unavailable for %s", location)
        return False

    if not validate_weather_record(weather_raw) or not validate_air_quality_record(air_quality_raw):
        logger.error("Validation failed for %s, discarding this fetch", location)
        return False

    try:
        weather_clean = clean_weather_record(weather_raw, location)
        air_quality_clean = clean_air_quality_record(air_quality_raw, location)

        insert_weather_record(engine, weather_clean)
        insert_air_quality_record(engine, air_quality_clean)
    except Exception as exc:
        logger.error("Failed to store data for %s: %s", location, exc)
        return False

    logger.info("Stored weather + air quality for %s (AQI=%s)", location, air_quality_clean["aqi"])
    return True


if __name__ == "__main__":
    location = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOCATION
    success = collect_for_location(location)
    sys.exit(0 if success else 1)
