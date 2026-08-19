from datetime import datetime

from data.aqi_calculator import calculate_cpcb_aqi
from utils.logging_config import get_logger

logger = get_logger(__name__)

REQUIRED_WEATHER_FIELDS = ["timestamp", "latitude", "longitude", "temperature"]
POLLUTANT_FIELDS = ["pm25", "pm10", "co", "no2", "so2", "o3"]


def validate_weather_record(record: dict) -> bool:
    for field in REQUIRED_WEATHER_FIELDS:
        if record.get(field) is None:
            logger.warning("Weather record missing required field '%s'", field)
            return False
    return True


def validate_air_quality_record(record: dict) -> bool:
    """Is this record worth storing at all?

    Deliberately looser than CPCB's AQI-validity rule: a record with a single
    pollutant still carries a real raw measurement worth keeping. Whether that
    is enough to report an AQI is decided by `calculate_cpcb_aqi`, which
    returns None below CPCB's three-pollutant / particulate minimum.
    """
    if record.get("timestamp") is None:
        logger.warning("Air quality record missing timestamp")
        return False
    if all(record.get(field) is None for field in POLLUTANT_FIELDS):
        logger.warning("Air quality record has no pollutant readings at all")
        return False
    return True


def _round_or_none(value, digits=2):
    return round(value, digits) if value is not None else None


def clean_weather_record(record: dict, location: str) -> dict:
    return {
        "timestamp": datetime.fromisoformat(record["timestamp"]),
        "location": location,
        "latitude": record["latitude"],
        "longitude": record["longitude"],
        "temperature": _round_or_none(record.get("temperature")),
        "feels_like": _round_or_none(record.get("feels_like")),
        "humidity": _round_or_none(record.get("humidity")),
        "pressure": _round_or_none(record.get("pressure")),
        "wind_speed": _round_or_none(record.get("wind_speed")),
        "wind_direction": _round_or_none(record.get("wind_direction")),
        "rainfall": _round_or_none(record.get("rainfall")),
        "visibility": _round_or_none(record.get("visibility")),
        "cloud_cover": _round_or_none(record.get("cloud_cover")),
        "uv_index": _round_or_none(record.get("uv_index")),
    }


def clean_air_quality_record(record: dict, location: str) -> dict:
    aqi, _dominant_pollutant = calculate_cpcb_aqi(
        pm25=record.get("pm25"),
        pm10=record.get("pm10"),
        co=record.get("co"),
        no2=record.get("no2"),
        so2=record.get("so2"),
        o3=record.get("o3"),
    )
    return {
        "timestamp": datetime.fromisoformat(record["timestamp"]),
        "location": location,
        "pm25": _round_or_none(record.get("pm25")),
        "pm10": _round_or_none(record.get("pm10")),
        "co": _round_or_none(record.get("co")),
        "no2": _round_or_none(record.get("no2")),
        "so2": _round_or_none(record.get("so2")),
        "o3": _round_or_none(record.get("o3")),
        "aqi": aqi,
    }
