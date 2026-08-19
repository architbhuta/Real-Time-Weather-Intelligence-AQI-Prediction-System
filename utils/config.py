import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Repo root, so paths never depend on the process's working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# city -> (latitude, longitude)
LOCATIONS: dict[str, tuple[float, float]] = {
    "Delhi": (28.7041, 77.1025),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Pune": (18.5204, 73.8567),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Ahmedabad": (23.0225, 72.5714),
}

DEFAULT_LOCATION = os.getenv("DEFAULT_LOCATION", "Delhi")

# A relative DATABASE_PATH (including the default) is resolved against the repo
# root, so `python collect.py` works from any working directory.
_database_path = os.getenv("DATABASE_PATH") or "db/aqi_system.db"
DATABASE_PATH = str(
    Path(_database_path) if os.path.isabs(_database_path) else PROJECT_ROOT / _database_path
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
