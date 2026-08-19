import logging

from utils.config import LOG_LEVEL, PROJECT_ROOT

# Anchored to the repo root so logs land in the same place regardless of cwd.
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL, logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler(),
            ],
            # Without force=True, basicConfig is a no-op when the root logger
            # already has handlers (e.g. Streamlit configures logging on
            # startup), silently dropping our file handler.
            force=True,
        )
        _configured = True
    return logging.getLogger(name)
