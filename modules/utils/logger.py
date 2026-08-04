import logging
from pathlib import Path

from config import LOG_DIR


def get_logger(name: str = "ai_learning_assistant") -> logging.Logger:
    """Return a file-backed project logger."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(Path(LOG_DIR) / "app.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    return logger