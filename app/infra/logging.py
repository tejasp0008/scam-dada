"""JSON logging with the current call id attached to every line.

`call_id` travels in a contextvar rather than being threaded through every signature,
because the orchestrator spawns background tasks (analysis, TTS) that must still be
greppable by call.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

_call_id: ContextVar[str | None] = ContextVar("call_id", default=None)

#: Attributes LogRecord always has; anything else was passed via `extra` and is ours.
_STANDARD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


def set_call_id(call_id: str | None) -> None:
    """Bind `call_id` to the current context so later log lines carry it."""
    _call_id.set(call_id)


def get_call_id() -> str | None:
    """Return the call id bound to the current context, if any."""
    return _call_id.get()


class JsonFormatter(logging.Formatter):
    """Renders records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        call_id = _call_id.get()
        if call_id is not None:
            payload["call_id"] = call_id
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger. Idempotent."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # Uvicorn's access log duplicates what our route handlers already report, and httpx
    # logs one INFO line per request, which on the audio path is one per TTS chunk batch.
    logging.getLogger("uvicorn.access").propagate = False
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
