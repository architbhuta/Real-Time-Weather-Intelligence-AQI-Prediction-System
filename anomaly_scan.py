import sys

from anomaly.detector import detect_anomalies_for_location
from data.database import get_engine, init_db, insert_anomalies, load_weather_and_air_quality
from utils.config import DEFAULT_LOCATION, LOCATIONS
from utils.logging_config import get_logger

logger = get_logger(__name__)


def scan_location(location: str, engine=None) -> int:
    if location not in LOCATIONS:
        logger.error("Unknown location '%s'", location)
        return 0

    engine = engine or get_engine()
    try:
        init_db(engine)
        df = load_weather_and_air_quality(engine, location)
    except Exception as exc:
        logger.error("Could not load history for %s: %s", location, exc)
        return 0

    if df.empty:
        logger.warning("No stored history for %s; skipping anomaly scan", location)
        return 0

    anomalies = detect_anomalies_for_location(df, location)
    if not anomalies:
        logger.info("No anomalies detected for %s", location)
        return 0

    stored = insert_anomalies(engine, anomalies)
    logger.info("Anomaly scan for %s: %d flagged, %d newly stored", location, len(anomalies), stored)
    return stored


if __name__ == "__main__":
    location = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOCATION
    scan_location(location)
