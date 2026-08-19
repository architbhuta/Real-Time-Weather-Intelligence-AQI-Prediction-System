import os

from dotenv import load_dotenv

load_dotenv()

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
DATABASE_PATH = os.getenv("DATABASE_PATH", "db/aqi_system.db")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
