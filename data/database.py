import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    inspect,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Session

from utils.config import DATABASE_PATH
from utils.logging_config import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


class WeatherData(Base):
    __tablename__ = "weather_data"
    __table_args__ = (
        UniqueConstraint("timestamp", "location", name="uq_weather_data_timestamp_location"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    location = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    temperature = Column(Float)
    feels_like = Column(Float)
    humidity = Column(Float)
    pressure = Column(Float)
    wind_speed = Column(Float)
    wind_direction = Column(Float)
    rainfall = Column(Float)
    visibility = Column(Float)
    cloud_cover = Column(Float)
    uv_index = Column(Float)


class AirQualityData(Base):
    __tablename__ = "air_quality_data"
    __table_args__ = (
        UniqueConstraint("timestamp", "location", name="uq_air_quality_data_timestamp_location"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    location = Column(String, nullable=False)
    pm25 = Column(Float)
    pm10 = Column(Float)
    co = Column(Float)
    no2 = Column(Float)
    so2 = Column(Float)
    o3 = Column(Float)
    aqi = Column(Integer)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    location = Column(String, nullable=False)
    prediction_horizon = Column(String, nullable=False)
    predicted_aqi = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)


class Anomaly(Base):
    __tablename__ = "anomalies"
    __table_args__ = (UniqueConstraint("timestamp", "location", "metric", name="uq_anomaly_timestamp_location_metric"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    location = Column(String, nullable=False)
    metric = Column(String, nullable=False)
    observed_value = Column(Float, nullable=False)
    expected_value = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    severity = Column(String, nullable=False)


def get_engine(db_path: str | None = None):
    path = db_path or DATABASE_PATH
    return create_engine(f"sqlite:///{path}")


# The dedup guarantee the whole anomaly path relies on, as a plain column set.
_ANOMALY_UNIQUE_COLUMNS = {"timestamp", "location", "metric"}


def _anomalies_table_is_unique_constrained(engine) -> bool:
    """True if the live `anomalies` table really carries its uniqueness constraint.

    SQLite reports a *named* ``UniqueConstraint`` through ``get_unique_constraints``,
    while an unnamed one can instead surface as a unique index, so both are checked.
    """
    inspector = inspect(engine)
    constraints = inspector.get_unique_constraints(Anomaly.__tablename__)
    if any(set(c.get("column_names") or ()) == _ANOMALY_UNIQUE_COLUMNS for c in constraints):
        return True
    indexes = inspector.get_indexes(Anomaly.__tablename__)
    return any(
        ix.get("unique") and set(ix.get("column_names") or ()) == _ANOMALY_UNIQUE_COLUMNS
        for ix in indexes
    )


def init_db(engine) -> None:
    """Create any missing tables, and warn if an existing `anomalies` table has drifted.

    ``create_all`` only creates tables that do not exist yet — it never retrofits a
    constraint onto a table left over from an older schema version. That has bitten
    this project twice on the same local dev database, and it fails silently: inserts
    that should be deduplicated instead pile up as duplicates with no error. So the
    one constraint the anomaly path depends on is checked explicitly here.

    Only a table that existed *before* this call is checked — a table ``create_all``
    just built is correct by construction and must not trigger a false warning.
    """
    anomalies_existed = inspect(engine).has_table(Anomaly.__tablename__)

    Base.metadata.create_all(engine)

    if anomalies_existed and not _anomalies_table_is_unique_constrained(engine):
        logger.warning(
            "anomalies table exists but is missing its uniqueness constraint on "
            "(timestamp, location, metric) (created by an older schema version) — "
            "delete %s and regenerate it via backfill.py/collect.py/anomaly_scan.py to fix",
            DATABASE_PATH,
        )


def insert_weather_record(engine, record: dict) -> None:
    """Insert one weather row. A duplicate (timestamp, location) is a no-op."""
    with Session(engine) as session:
        session.add(WeatherData(**record))
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            logger.info(
                "Weather record for %s at %s already stored; skipping duplicate",
                record.get("location"),
                record.get("timestamp"),
            )


def insert_air_quality_record(engine, record: dict) -> None:
    """Insert one air-quality row. A duplicate (timestamp, location) is a no-op."""
    with Session(engine) as session:
        session.add(AirQualityData(**record))
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            logger.info(
                "Air quality record for %s at %s already stored; skipping duplicate",
                record.get("location"),
                record.get("timestamp"),
            )


def insert_weather_and_air_quality(
    engine, weather_record: dict, air_quality_record: dict
) -> None:
    """Insert a paired weather + air-quality observation in one transaction.

    Both rows are added to the same session and committed once, so either both
    are stored or neither is — no orphan weather row if the air-quality row
    fails. A duplicate (timestamp, location) on either row is a no-op.
    """
    with Session(engine) as session:
        session.add(WeatherData(**weather_record))
        session.add(AirQualityData(**air_quality_record))
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            logger.info(
                "Observation for %s at %s already stored; skipping duplicate",
                weather_record.get("location"),
                weather_record.get("timestamp"),
            )


def load_weather_and_air_quality(engine, location: str) -> pd.DataFrame:
    with Session(engine) as session:
        weather_rows = session.query(WeatherData).filter(WeatherData.location == location).all()
        air_quality_rows = session.query(AirQualityData).filter(AirQualityData.location == location).all()

    weather_df = pd.DataFrame([{
        "timestamp": r.timestamp, "location": r.location, "temperature": r.temperature,
        "feels_like": r.feels_like, "humidity": r.humidity, "pressure": r.pressure,
        "wind_speed": r.wind_speed, "wind_direction": r.wind_direction, "rainfall": r.rainfall,
        "visibility": r.visibility, "cloud_cover": r.cloud_cover, "uv_index": r.uv_index,
    } for r in weather_rows])

    air_quality_df = pd.DataFrame([{
        "timestamp": r.timestamp, "location": r.location, "pm25": r.pm25, "pm10": r.pm10,
        "co": r.co, "no2": r.no2, "so2": r.so2, "o3": r.o3, "aqi": r.aqi,
    } for r in air_quality_rows])

    if weather_df.empty or air_quality_df.empty:
        return pd.DataFrame(columns=[
            "timestamp", "location", "temperature", "feels_like", "humidity", "pressure",
            "wind_speed", "wind_direction", "rainfall", "visibility", "cloud_cover", "uv_index",
            "pm25", "pm10", "co", "no2", "so2", "o3", "aqi",
        ])

    merged = pd.merge(weather_df, air_quality_df, on=["timestamp", "location"], how="inner")
    return merged.sort_values("timestamp").reset_index(drop=True)


def insert_anomalies(engine, records: list[dict]) -> int:
    """Insert anomaly rows. A duplicate (timestamp, location, metric) is a no-op.

    Known, deliberate limitation: a row is written once, on first detection, and is
    never refreshed by a later rescan. A stored row's ``expected_value``,
    ``anomaly_score`` and ``severity`` therefore describe the historical distribution
    *as it was when the point was first flagged*, not the current one — as more
    history accumulates the IQR fence moves, but already-stored rows do not follow it.
    This is not a bug: whether stale rows should be refreshed is a display question,
    and that decision belongs to the future dashboard plan that will actually consume
    and show these values.
    """
    stored = 0
    for record in records:
        with Session(engine) as session:
            try:
                session.add(Anomaly(**record))
                session.commit()
                stored += 1
            except IntegrityError:
                session.rollback()
                logger.info(
                    "Anomaly for %s %s at %s already stored; skipping duplicate",
                    record.get("location"),
                    record.get("metric"),
                    record.get("timestamp"),
                )
    return stored


def get_latest_reading(engine, location: str) -> dict | None:
    df = load_weather_and_air_quality(engine, location)
    if df.empty:
        return None
    row = df.iloc[-1]
    # A NULL in a nullable numeric column (e.g. aqi, pm25 when a live fetch was
    # degraded) survives the SQL round-trip as pandas NaN, not Python None, once
    # the column is coerced to float64 — convert it back so callers can rely on
    # a plain `is None` check instead of every caller special-casing NaN.
    return row.where(pd.notna(row), None).to_dict()


def get_recent_anomalies(engine, location: str, limit: int = 20) -> pd.DataFrame:
    with Session(engine) as session:
        rows = (
            session.query(Anomaly)
            .filter(Anomaly.location == location)
            .order_by(Anomaly.timestamp.desc())
            .limit(limit)
            .all()
        )
    return pd.DataFrame([{
        "timestamp": r.timestamp,
        "metric": r.metric,
        "observed_value": r.observed_value,
        "expected_value": r.expected_value,
        "anomaly_score": r.anomaly_score,
        "severity": r.severity,
    } for r in rows])
