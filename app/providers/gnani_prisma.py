"""Prisma v3.0 streaming speech-to-text over WebSocket.

Three properties of this API shape everything downstream, and are worth stating here
because they are not what a speech API usually does:

* All configuration travels as **WebSocket upgrade headers**. Nothing is negotiated after
  the connection opens except `update_settings` and `commit`.
* Audio frames must be **exactly 1024 bytes** of raw PCM s16le mono, sent at real-time
  cadence. Anything else is rejected.
* There are **no partial transcripts** — `is_final` is always true — and turn boundaries
  arrive as separate `speech_start` / `processing` events. The adapter surfaces those as
  first-class events rather than discarding them, because they are the only endpointing
  signal available.

A session is also capped at 15 minutes upstream. That is handled a layer above, by the
rotation wrapper; here it surfaces as a fatal error event so the layer above can react.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any, Final

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from app.core.audio.framing import PRISMA_FRAME_BYTES
from app.providers.base import (
    AuthError,
    BadRequest,
    ProviderUnavailable,
    SttConfig,
    SttEvent,
    SttProcessing,
    SttReady,
    SttSpeechStart,
    SttStreamError,
    SttTranscript,
    UsageRecorder,
)
from app.providers.gnani_common import API_KEY_HEADER, error_for_status

logger = logging.getLogger(__name__)

#: Upstream closes a session at 15 minutes with code 1001.
SESSION_LIMIT_SECONDS: Final[int] = 15 * 60
#: Close code the server uses for "your session ran too long".
CLOSE_SESSION_EXPIRED: Final[int] = 1001
#: Close code the server uses for a bad handshake header.
CLOSE_BAD_HANDSHAKE: Final[int] = 1008

#: Flush usage this often rather than only at close, so the budget guard can stop a call
#: that is still in progress. A scambait is meant to run for many minutes.
BILLING_FLUSH_SECONDS: Final[float] = 30.0


class GnaniPrismaSession:
    """One live transcription session."""

    def __init__(
        self,
        connection: ClientConnection,
        config: SttConfig,
        *,
        usage: UsageRecorder | None = None,
        provider_name: str = "gnani-prisma",
    ) -> None:
        self._ws = connection
        self._config = config
        self._usage = usage
        self._provider = provider_name
        self._opened_at = time.monotonic()
        self._audio_seconds_sent = 0.0
        self._audio_seconds_billed = 0.0
        self._closed = False
        # 512 samples per frame; how long that is depends on the negotiated rate.
        self._seconds_per_frame = (PRISMA_FRAME_BYTES // 2) / config.sample_rate

    @property
    def frame_bytes(self) -> int:
        return PRISMA_FRAME_BYTES

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock age of this session, for the rotation wrapper."""
        return time.monotonic() - self._opened_at

    @property
    def audio_seconds_sent(self) -> float:
        """Audio pushed so far. Drives billing, and the minutes-wasted counter."""
        return self._audio_seconds_sent

    def _at_ms(self) -> int:
        return int((time.monotonic() - self._opened_at) * 1000)

    async def send_audio(self, frame: bytes) -> None:
        """Send one PCM frame of exactly `frame_bytes` bytes."""
        if self._closed:
            raise BadRequest("session is closed")
        if len(frame) != PRISMA_FRAME_BYTES:
            raise BadRequest(f"frame must be exactly {PRISMA_FRAME_BYTES} bytes, got {len(frame)}")
        try:
            await self._ws.send(frame)
        except ConnectionClosed as exc:
            raise _closed_error(exc) from exc
        self._audio_seconds_sent += self._seconds_per_frame
        if self._audio_seconds_sent - self._audio_seconds_billed >= BILLING_FLUSH_SECONDS:
            await self._flush_billing()

    async def commit(self) -> None:
        """Close the open utterance immediately instead of waiting for the VAD."""
        if self._closed:
            raise BadRequest("session is closed")
        try:
            await self._ws.send(json.dumps({"type": "commit"}))
        except ConnectionClosed as exc:
            raise _closed_error(exc) from exc

    async def events(self) -> AsyncIterator[SttEvent]:
        """Yield session events until the connection closes."""
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    continue  # the server is not documented to send binary; ignore it
                event = self._translate(message)
                if event is not None:
                    yield event
        except ConnectionClosed as exc:
            error = _closed_error(exc)
            yield SttStreamError(message=str(error), at_ms=self._at_ms(), fatal=True)

    def _translate(self, message: str) -> SttEvent | None:
        """Map one server frame onto our event vocabulary."""
        try:
            payload: Any = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("unparseable stt frame", extra={"raw": message[:200]})
            return None
        if not isinstance(payload, dict):
            return None

        kind = payload.get("type")
        if kind == "connected":
            config = payload.get("config") or {}
            return SttReady(
                sample_rate=int(config.get("sample_rate", self._config.sample_rate)),
                frame_bytes=PRISMA_FRAME_BYTES,
                request_id=payload.get("request_id"),
            )
        if kind == "speech_start":
            return SttSpeechStart(at_ms=self._at_ms(), utterance_id=payload.get("utterance_id"))
        if kind == "processing":
            return SttProcessing(at_ms=self._at_ms(), utterance_id=payload.get("utterance_id"))
        if kind == "transcript":
            return SttTranscript(
                text=str(payload.get("text", "")),
                # Prisma has no partial hypotheses; it documents this as always true.
                is_final=bool(payload.get("is_final", True)),
                at_ms=self._at_ms(),
                segment_id=payload.get("segment_id"),
                segment_index=payload.get("segment_index"),
                utterance_id=payload.get("utterance_id"),
                detected_language_code=payload.get("detected_language_code"),
                audio_duration_ms=payload.get("audio_duration_ms"),
                provider_latency_ms=_as_int(payload.get("latency")),
            )
        if kind == "error":
            return SttStreamError(
                message=str(payload.get("message", "stt error")),
                at_ms=self._at_ms(),
                fatal=False,
            )
        if kind == "settings_updated":
            return None  # an acknowledgement; nothing downstream needs it
        logger.debug("unhandled stt frame type", extra={"frame_type": kind})
        return None

    async def _flush_billing(self) -> None:
        """Charge for audio sent since the last flush."""
        if self._usage is None:
            return
        pending = self._audio_seconds_sent - self._audio_seconds_billed
        if pending <= 0:
            return
        charge = self._usage.cost.price_stt(pending)
        self._audio_seconds_billed = self._audio_seconds_sent
        await self._usage.record(self._provider, charge, call_id=self._config.call_id)

    async def aclose(self) -> None:
        """Close the connection and charge for any audio not yet billed."""
        if self._closed:
            return
        self._closed = True
        try:
            await self._flush_billing()
        finally:
            await self._ws.close()


class GnaniPrismaSTT:
    """Opens Prisma streaming sessions."""

    name = "gnani-prisma"

    def __init__(
        self,
        *,
        api_key: str,
        ws_url: str,
        model: str,
        usage: UsageRecorder | None = None,
        open_timeout: float = 10.0,
    ) -> None:
        if not api_key:
            raise ValueError("a Gnani API key is required")
        self._api_key = api_key
        self._url = ws_url
        self._model = model
        self._usage = usage
        self._open_timeout = open_timeout

    def _headers(self, config: SttConfig) -> dict[str, str]:
        """Build the upgrade headers. Everything Prisma takes is configured here."""
        headers = {
            API_KEY_HEADER.lower(): self._api_key,
            "model": config.model or self._model,
            "x-sample-rate": str(config.sample_rate),
        }
        output_format = config.output_format
        if output_format:
            headers["format"] = output_format
        # `language_code` is optional only for transliterate on v3.0, where it defaults to
        # indic-auto. Sending it anyway in that mode is harmless, but omitting it elsewhere
        # makes the server reject the handshake, so be explicit.
        if config.language_code:
            headers["language_code"] = config.language_code
        if config.idle_timeout_sec is not None:
            headers["x-idle-timeout-sec"] = str(config.idle_timeout_sec)
        if config.keyterms:
            headers["keyterms"] = json.dumps(list(config.keyterms))
        if config.min_silence_ms is not None:
            headers["x-min-silence-ms"] = str(config.min_silence_ms)
        if config.vad_threshold is not None:
            headers["x-vad-threshold"] = str(config.vad_threshold)
        return headers

    async def open_session(self, config: SttConfig) -> GnaniPrismaSession:
        """Connect and return a session ready for audio."""
        try:
            connection = await connect(
                self._url,
                additional_headers=self._headers(config),
                open_timeout=self._open_timeout,
                max_size=None,
            )
        except InvalidStatus as exc:
            # A rejected key never opens a WebSocket at all; it fails the HTTP upgrade.
            body = b""
            raise error_for_status(exc.response.status_code, body, surface="stt") from exc
        except OSError as exc:
            raise ProviderUnavailable(f"could not reach the stt endpoint: {exc}") from exc

        logger.info(
            "stt session opened",
            extra={
                "model": config.model,
                "language_code": config.language_code,
                "format": config.output_format,
                "sample_rate": config.sample_rate,
            },
        )
        return GnaniPrismaSession(connection, config, usage=self._usage, provider_name=self.name)


def _closed_error(exc: ConnectionClosed) -> Exception:
    """Translate a WebSocket close into something the orchestrator can act on."""
    code = exc.rcvd.code if exc.rcvd else None
    reason = (exc.rcvd.reason if exc.rcvd else "") or ""
    if code == CLOSE_SESSION_EXPIRED:
        return ProviderUnavailable(
            f"stt session hit the {SESSION_LIMIT_SECONDS // 60}-minute upstream limit; "
            "rotate to a new session"
        )
    if code == CLOSE_BAD_HANDSHAKE:
        return BadRequest(f"stt rejected the handshake: {reason}")
    if code in (401, 403):
        return AuthError(f"stt rejected the key: {reason}")
    return ProviderUnavailable(f"stt connection closed (code {code}): {reason}")


def _as_int(value: object) -> int | None:
    """Prisma reports latency as a number that may arrive as a float."""
    if value is None:
        return None
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
