"""Shared plumbing for the Gnani adapters.

Gnani returns errors in four different JSON shapes depending on which surface failed
(documented in api/introduction/errors). Normalising them in one place means the rest of
the codebase only ever sees our own exception types, and the retry policy only has to be
written once.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Final

from app.providers.base import (
    AuthError,
    BadRequest,
    OutOfCredit,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
    SynthesisFailed,
    TranscriptionFailed,
    UpstreamTimeout,
)

logger = logging.getLogger(__name__)

#: Gnani authenticates with this header on both REST and the WebSocket upgrade.
API_KEY_HEADER: Final[str] = "X-API-Key-ID"

#: The docs say to retry connection errors and these statuses with exponential backoff,
#: about three attempts starting around 0.5 s. Other 4xx "fail the same way".
RETRYABLE_STATUSES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})

#: Error types that mean one utterance failed while the session itself survives.
_UTTERANCE_FAILURES: Final[frozenset[str]] = frozenset({"TRANSCRIPTION_FAILED", "SYNTHESIS_FAILED"})


def extract_error(body: bytes | str | None) -> tuple[str | None, str]:
    """Pull a (code, message) pair out of any of Gnani's error shapes.

    Shapes seen in the docs:
      STT REST      {"success": false, "error": {"type": ..., "message": ...}}
      auth / rate   {"detail": {"error_code": ..., "message": ..., "status_code": ...}}
      validation    {"detail": [ ... ]}
      batch         {"error": ..., "message": ...}
      v3.0 504      {"error": "Gateway Timeout", "code": "UPSTREAM_TIMEOUT", "message": ...}
      TTS / VC      {"success": false, "message": ..., "status_code": ...}
      streaming     {"type": "error", "message": ...}
    """
    if not body:
        return None, ""
    try:
        payload: Any = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        text = body.decode(errors="replace") if isinstance(body, bytes) else body
        return None, text[:300]

    if not isinstance(payload, dict):
        return None, str(payload)[:300]

    detail = payload.get("detail")
    if isinstance(detail, dict):
        return detail.get("error_code"), str(detail.get("message", ""))
    if isinstance(detail, list):
        return "VALIDATION_ERROR", json.dumps(detail)[:300]

    error = payload.get("error")
    if isinstance(error, dict):
        return error.get("type"), str(error.get("message", ""))
    if isinstance(error, str):
        return payload.get("code") or error, str(payload.get("message", error))

    return payload.get("code"), str(payload.get("message", ""))


def error_for_status(status: int, body: bytes | str | None, *, surface: str) -> ProviderError:
    """Map an HTTP status and body onto one of our exception types.

    `surface` ("stt" or "tts") only decides which single-request failure type is raised, so
    a caller can tell "this utterance failed, carry on" from "the session is dead".
    """
    code, message = extract_error(body)
    label = f"{surface} HTTP {status}"
    detail = f"{label}: {code or 'no code'} {message}".strip()

    if status in (401, 403):
        return AuthError(detail)
    if status == 402:
        return OutOfCredit(detail)
    if status == 429:
        return RateLimited(detail)
    if status == 504:
        return UpstreamTimeout(detail)
    if code in _UTTERANCE_FAILURES:
        return TranscriptionFailed(detail) if surface == "stt" else SynthesisFailed(detail)
    if status >= 500:
        return ProviderUnavailable(detail)
    return BadRequest(detail)
