import logging
import os

from utils.config import LOG_LEVEL

os.makedirs("logs", exist_ok=True)

_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL, logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[
                logging.FileHandler("logs/app.log"),
                logging.StreamHandler(),
            ],
        )
        _configured = True
    return logging.getLogger(name)
