"""Structured logging configuration for production FastAPI services."""

import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Formats log records as structured JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logging(log_level: str = "INFO", log_format: str = "text") -> None:
    """Configure structured logging across all application loggers."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    if log_format.lower() == "json":
        handler.setFormatter(JSONFormatter())
    else:
        text_format = "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        handler.setFormatter(logging.Formatter(fmt=text_format, datefmt=date_format))

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers = [handler]

    # Silence noisy third-party loggers
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)
