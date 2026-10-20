"""Local JSON events; request context also propagates to FastAPI's worker threads."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)
logger = logging.getLogger("ai_hardcore_engineer")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            **getattr(record, "event_fields", {}),
        }, ensure_ascii=False)


def configure_logging() -> None:
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def log_event(event: str, **fields: str | int | bool | None) -> None:
    # Callers supply only operational metadata, never messages, user IDs or tool data.
    logger.info(event, extra={"request_id": request_id_context.get(), "event_fields": fields})
