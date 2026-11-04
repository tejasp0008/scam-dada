"""Timbre adapter against a recorded SSE response."""

from __future__ import annotations

import httpx
import pytest

from app.providers.base import (
    AuthError,
    BadRequest,
    OutOfCredit,
    ProviderUnavailable,
    RateLimited,
    TtsRequest,
    UpstreamTimeout,
)
from app.providers.gnani_common import API_KEY_HEADER
from app.providers.gnani_timbre import GnaniTimbreTTS

pytestmark = pytest.mark.contract

REQUEST = TtsRequest(
    text="Haan beta, <sigh> kaun bol raha hai?",
    voice="Chetana",
    language="hi-en",
    model="timbre-v3.0",
    sample_rate=16000,
)


def adapter(handler: object, **kwargs: object) -> GnaniTimbreTTS:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    return GnaniTimbreTTS(
        api_key="test-key",
        sse_url="https://example.invalid/tts/sse",
        model="timbre-v3.0",
        language="hi-en",
        client=client,
        **kwargs,  # type: ignore[arg-type]
    )


async def test_a_recorded_sse_stream_decodes_to_pcm_chunks(fixture_text) -> None:
    body = fixture_text("timbre_sse_stream.txt")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    chunks = [c async for c in adapter(handler).synthesize(REQUEST)]
    # The `start` and `complete` frames carry no audio and must not become chunks.
    assert [c.index for c in chunks] == [0, 1]
    assert all(len(c.pcm) == 8 for c in chunks)
    assert not any(c.is_final for c in chunks)


async def test_the_request_asks_for_headerless_pcm_and_sends_the_key_header() -> None:
    """Raw PCM is required for instant barge-in, and Timbre's WAV headers lie about length."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, text="", headers={"content-type": "text/event-stream"})

    [c async for c in adapter(handler).synthesize(REQUEST)]
    body = seen["body"]
    assert body["audio_config"]["container"] == "raw"  # type: ignore[index]
    assert body["audio_config"]["encoding"] == "pcm_s16le"  # type: ignore[index]
    assert body["model"] == "timbre-v3.0"  # type: ignore[index]
    assert body["language"] == "hi-en"  # type: ignore[index]
    assert seen["headers"][API_KEY_HEADER.lower()] == "test-key"  # type: ignore[index]


async def test_the_full_text_including_tags_is_sent_verbatim() -> None:
    """Tag characters are billable and change the delivery, so they are not stripped."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["text"] = json.loads(request.content)["text"]
        return httpx.Response(200, text="", headers={"content-type": "text/event-stream"})

    [c async for c in adapter(handler).synthesize(REQUEST)]
    assert seen["text"] == REQUEST.text
    assert "<sigh>" in str(seen["text"])


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, b'{"detail": {"error_code": "INVALID_API_KEY", "message": "bad key"}}', AuthError),
        (403, b'{"detail": {"error_code": "SCOPE_NOT_ALLOWED", "message": "no"}}', AuthError),
        (402, b'{"detail": {"error_code": "NO_CREDIT", "message": "empty"}}', OutOfCredit),
        (429, b'{"detail": {"error_code": "RATE_LIMITED", "message": "slow down"}}', RateLimited),
        (400, b'{"success": false, "message": "bad voice", "status_code": 400}', BadRequest),
    ],
)
async def test_error_shapes_map_to_our_exception_types(
    status: int, body: bytes, expected: type[Exception]
) -> None:
    """Gnani returns four different error shapes; callers must only ever see ours."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body)

    with pytest.raises(expected):
        [c async for c in adapter(handler).synthesize(REQUEST)]


async def test_a_retryable_failure_is_retried_before_any_audio_is_emitted() -> None:
    attempts = {"n": 0}
    good = 'event: chunk\ndata: {"chunk_index": 0, "audio": "AAA=", "is_final": true}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503, content=b'{"success": false, "message": "down"}')
        return httpx.Response(200, text=good, headers={"content-type": "text/event-stream"})

    chunks = [c async for c in adapter(handler).synthesize(REQUEST)]
    assert attempts["n"] == 2
    assert len(chunks) == 1


async def test_a_timeout_before_the_first_chunk_is_retryable() -> None:
    attempts = {"n": 0}
    good = 'data: {"chunk_index": 0, "audio": "AAA=", "is_final": true}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ReadTimeout("too slow", request=request)
        return httpx.Response(200, text=good, headers={"content-type": "text/event-stream"})

    chunks = [c async for c in adapter(handler).synthesize(REQUEST)]
    assert attempts["n"] == 2
    assert chunks


async def test_a_nested_realtime_envelope_is_unwrapped(fixture_text) -> None:
    """The realtime transport nests audio under `data`; SSE does not. Accept both."""
    body = (
        'data: {"type": "audio", "data": {"chunk_index": 7, "audio": "AAA=", "is_final": true}}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    chunks = [c async for c in adapter(handler).synthesize(REQUEST)]
    assert chunks[0].index == 7
    assert chunks[0].is_final


async def test_an_in_stream_error_frame_raises() -> None:
    body = 'data: {"type": "error", "message": "SYNTHESIS_FAILED"}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    with pytest.raises(Exception, match="SYNTHESIS_FAILED"):
        [c async for c in adapter(handler).synthesize(REQUEST)]


async def test_usage_is_priced_on_the_text_as_sent(fixture_text) -> None:
    from app.core.cost import CostModel

    class Recorder:
        def __init__(self) -> None:
            self.cost = CostModel(rates_confirmed=True)
            self.checked: list[float] = []
            self.recorded: list[float] = []

        async def check(self, provider: str, charge: object) -> None:
            self.checked.append(charge.raw_units)  # type: ignore[attr-defined]

        async def record(self, provider: str, charge: object, **kw: object) -> None:
            self.recorded.append(charge.raw_units)  # type: ignore[attr-defined]

    recorder = Recorder()
    body = fixture_text("timbre_sse_stream.txt")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    tts = adapter(handler, usage=recorder)
    [c async for c in tts.synthesize(REQUEST)]
    assert recorder.checked == [float(len(REQUEST.text))]
    assert recorder.recorded == [float(len(REQUEST.text))]


async def test_an_empty_api_key_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="API key"):
        GnaniTimbreTTS(
            api_key="",
            sse_url="https://example.invalid",
            model="timbre-v3.0",
            language="hi-en",
            client=httpx.AsyncClient(),
        )


async def test_upstream_timeout_status_maps_to_a_retryable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            504, content=b'{"error": "Gateway Timeout", "code": "UPSTREAM_TIMEOUT", "message": "x"}'
        )

    with pytest.raises((UpstreamTimeout, ProviderUnavailable)):
        [c async for c in adapter(handler).synthesize(REQUEST)]
