"""Timbre v3.0 text-to-speech over Server-Sent Events.

Transport choice: SSE rather than the realtime WebSocket. Both return base64 audio inside
JSON rather than binary frames, and neither documents a cancel message, so a warm
WebSocket buys nothing we can use — on barge-in we have to abandon the stream either way.
SSE is one protocol fewer to keep alive and reconnect. `TTSProvider` hides the choice, so
swapping transports is a change to this file alone.

Audio is requested as headerless `container=raw` PCM: the browser needs to drop queued
audio instantly on barge-in, and Timbre's WAV headers carry a 0xFFFFFFFF placeholder size
that makes every reader misjudge the duration.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, Final

import httpx

from app.core.cost import Charge
from app.providers.base import (
    AudioChunk,
    ProviderError,
    ProviderUnavailable,
    SynthesisFailed,
    TtsRequest,
    UpstreamTimeout,
    UsageRecorder,
)
from app.providers.gnani_common import API_KEY_HEADER, error_for_status

logger = logging.getLogger(__name__)

#: Attempts for a failure that happens before any audio has been handed to the caller.
#: Once a chunk is out we cannot retry: the caller is already playing it.
MAX_ATTEMPTS: Final[int] = 3
BACKOFF_BASE_SECONDS: Final[float] = 0.5


class GnaniTimbreTTS:
    """Streams synthesised speech from Timbre."""

    name = "gnani-timbre"

    def __init__(
        self,
        *,
        api_key: str,
        sse_url: str,
        model: str,
        language: str,
        client: httpx.AsyncClient,
        usage: UsageRecorder | None = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("a Gnani API key is required")
        self._api_key = api_key
        self._url = sse_url
        self._model = model
        self._language = language
        self._client = client
        self._usage = usage
        self._timeout = httpx.Timeout(read_timeout, connect=connect_timeout)
        # Timbre permits exactly one synthesis in flight per API key: a second concurrent
        # request is refused with `429 RATE_LIMITED Only 1 active request is allowed at a
        # time`. Measured, not documented - see DECISIONS 0023. Serialising here rather than
        # in the caller because the limit is the vendor's and applies across every call this
        # process is handling, not just within one.
        self._in_flight = asyncio.Lock()

    def _body(self, request: TtsRequest) -> dict[str, Any]:
        return {
            "text": request.text,
            "voice": request.voice,
            "model": request.model or self._model,
            "language": request.language or self._language,
            "speed": request.speed,
            "audio_config": {
                "sample_rate": request.sample_rate,
                "num_channels": 1,
                "sample_width": 2,
                "encoding": "pcm_s16le",
                "container": "raw",
            },
        }

    async def synthesize(self, request: TtsRequest) -> AsyncGenerator[AudioChunk, None]:
        """Yield PCM chunks as Timbre produces them.

        Tag characters are billable, so the charge is priced on the text exactly as sent.

        Serialised: the account allows one active request, and the lock is held for the whole
        stream rather than just the connect, because "active" means until the response is
        finished. A barge-in that abandons a stream therefore has to close it before the next
        synthesis can start, which is what the `aclosing` in `PersonaSpeaker` guarantees.
        """
        async with self._in_flight:
            async for chunk in self._synthesize_locked(request):
                yield chunk

    async def _synthesize_locked(self, request: TtsRequest) -> AsyncIterator[AudioChunk]:
        charge: Charge | None = None
        if self._usage is not None:
            charge = self._usage.cost.price_tts(len(request.text))
            await self._usage.check(self.name, charge)

        emitted = 0
        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                async for chunk in self._stream_once(request):
                    emitted += 1
                    yield chunk
                if self._usage is not None and charge is not None:
                    await self._usage.record(self.name, charge, call_id=request.call_id)
                return
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = UpstreamTimeout(f"tts transport failure: {exc}")
            except (ProviderUnavailable, UpstreamTimeout) as exc:
                last_error = exc
            if emitted:
                # Audio is already playing; a retry would duplicate speech mid-sentence.
                raise SynthesisFailed(
                    f"tts stream failed after {emitted} chunks, cannot retry"
                ) from last_error
            if attempt < MAX_ATTEMPTS:
                delay = BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
                logger.warning(
                    "retrying tts",
                    extra={"attempt": attempt, "delay_s": delay, "error": str(last_error)},
                )
                await asyncio.sleep(delay)
        # Re-raise the underlying error rather than flattening it: the orchestrator reacts
        # differently to "upstream is struggling, a cached filler will do" than to "this
        # request is malformed and will never work".
        if isinstance(last_error, ProviderError):
            raise last_error
        raise SynthesisFailed(f"tts failed after {MAX_ATTEMPTS} attempts") from last_error

    async def _stream_once(self, request: TtsRequest) -> AsyncIterator[AudioChunk]:
        """One SSE request, translated into `AudioChunk`s."""
        headers = {API_KEY_HEADER: self._api_key, "Accept": "text/event-stream"}
        async with self._client.stream(
            "POST", self._url, headers=headers, json=self._body(request), timeout=self._timeout
        ) as response:
            if response.status_code != 200:
                raise error_for_status(response.status_code, await response.aread(), surface="tts")
            async for event, payload in _iter_sse(response):
                if event == "error" or payload.get("type") == "error":
                    raise SynthesisFailed(str(payload.get("message", "tts stream error")))
                audio_b64 = payload.get("audio")
                if not audio_b64:
                    # `start`, and the terminal `complete` whose audio field is empty.
                    continue
                yield AudioChunk(
                    index=int(payload.get("chunk_index", 0)),
                    pcm=base64.b64decode(audio_b64),
                    is_final=bool(payload.get("is_final", False)),
                )


async def _iter_sse(response: httpx.Response) -> AsyncIterator[tuple[str | None, dict[str, Any]]]:
    """Yield (event name, parsed data) for each SSE frame.

    Written by hand rather than with a library because the frames are simple and this keeps
    the dependency list short. Field names are tolerated in either `event:`/`data:` form or
    as bare `data:` lines, since the docs show both.
    """
    event_name: str | None = None
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line.startswith(":"):
            continue  # a comment / keepalive
        if not line.strip():
            if data_lines:
                raw = "\n".join(data_lines)
                data_lines = []
                name, event_name = event_name, None
                parsed = _parse_data(raw)
                if parsed is not None:
                    yield name, parsed
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        parsed = _parse_data("\n".join(data_lines))
        if parsed is not None:
            yield event_name, parsed


def _parse_data(raw: str) -> dict[str, Any] | None:
    """Parse an SSE data payload, unwrapping the `{"type":..,"data":{..}}` envelope."""
    if raw == "[DONE]":
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("unparseable tts sse frame", extra={"raw": raw[:200]})
        return None
    if not isinstance(payload, dict):
        return None
    inner = payload.get("data")
    if isinstance(inner, dict):
        # The realtime transport nests audio under `data`; keep `type` for error detection.
        merged = dict(inner)
        merged.setdefault("type", payload.get("type"))
        return merged
    return payload
