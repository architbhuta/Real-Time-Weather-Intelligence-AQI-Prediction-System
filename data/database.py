from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session

from utils.config import DATABASE_PATH


class Base(DeclarativeBase):
    pass


class WeatherData(Base):
    __tablename__ = "weather_data"

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


def init_db(engine) -> None:
    Base.metadata.create_all(engine)


def insert_weather_record(engine, record: dict) -> None:
    with Session(engine) as session:
        session.add(WeatherData(**record))
        session.commit()


def insert_air_quality_record(engine, record: dict) -> None:
    with Session(engine) as session:
        session.add(AirQualityData(**record))
        session.commit()
