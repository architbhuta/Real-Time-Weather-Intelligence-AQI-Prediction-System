import sys

from anomaly.detector import detect_anomalies_for_location
from data.database import get_engine, init_db, insert_anomalies, load_weather_and_air_quality
from utils.config import DEFAULT_LOCATION, LOCATIONS
from utils.logging_config import get_logger

logger = get_logger(__name__)


def scan_location(location: str, engine=None) -> int:
    """Scan one location's stored history for anomalies and store them. Never raises.

    Returns the count of anomalies newly stored (0 on any failure).

    This is meant to run unattended alongside collect.py, so the whole body is
    guarded: engine creation, schema init, the read, detection, and storage. That
    last one matters — insert_anomalies only handles IntegrityError itself, so a
    concurrent writer's OperationalError ("database is locked") would otherwise
    escape a function documented as never raising.
    """
    try:
        if location not in LOCATIONS:
            logger.error("Unknown location '%s'", location)
            return 0

        engine = engine or get_engine()
        init_db(engine)
        df = load_weather_and_air_quality(engine, location)

        if df.empty:
            logger.warning("No stored history for %s; skipping anomaly scan", location)
            return 0

        anomalies = detect_anomalies_for_location(df, location)
        if not anomalies:
            logger.info("No anomalies detected for %s", location)
            return 0

        stored = insert_anomalies(engine, anomalies)
    except Exception as exc:
        logger.error("Anomaly scan failed for %s: %s", location, exc)
        return 0

    logger.info("Anomaly scan for %s: %d flagged, %d newly stored", location, len(anomalies), stored)
    return stored


if __name__ == "__main__":
    # scan_location returns a count, and 0 is a perfectly good outcome (a rerun over
    # already-flagged history stores nothing), so the exit code reports only whether
    # the entry point itself blew up.
    try:
        location = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOCATION
        scan_location(location)
        success = True
    except Exception:
        logger.exception("Unexpected failure in anomaly scan entry point")
        success = False
    sys.exit(0 if success else 1)
