"""Prisma adapter against recorded streaming frames.

The fixture is the sequence a live session actually produced: `connected`, `speech_start`,
two transcripts, a settings acknowledgement and an error — with **no `processing` event**.
See docs/DECISIONS.md 0012: the absence is the finding, so it is pinned here.
"""

from __future__ import annotations

import json

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from app.core.audio.framing import PRISMA_FRAME_BYTES
from app.providers.base import (
    BadRequest,
    ProviderUnavailable,
    SttConfig,
    SttEventKind,
    SttProcessing,
    SttReady,
    SttSpeechStart,
    SttStreamError,
    SttTranscript,
)
from app.providers.gnani_prisma import (
    CLOSE_BAD_HANDSHAKE,
    CLOSE_SESSION_EXPIRED,
    GnaniPrismaSession,
    GnaniPrismaSTT,
)

pytestmark = pytest.mark.contract

CONFIG = SttConfig(
    model="prisma-v3.0",
    language_code="indic-auto",
    sample_rate=16000,
    output_format="transliterate",
    idle_timeout_sec=900,
    keyterms=("UPI", "OTP"),
    min_silence_ms=500,
    vad_threshold=0.7,
)
FRAME = b"\x00" * PRISMA_FRAME_BYTES


class FakeConnection:
    """Replays recorded server frames and records what the adapter sent."""

    def __init__(self, frames: list[str], *, close: Close | None = None) -> None:
        self._frames = frames
        self._close = close
        self.sent_binary: list[bytes] = []
        self.sent_text: list[str] = []
        self.closed = False

    async def send(self, message: bytes | str) -> None:
        if isinstance(message, bytes):
            self.sent_binary.append(message)
        else:
            self.sent_text.append(message)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self) -> FakeConnection:
        self._iter = iter(self._frames)
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._iter)
        except StopIteration:
            if self._close is not None:
                raise ConnectionClosedError(self._close, None) from None
            raise StopAsyncIteration from None


def session_from(
    name: str = "prisma_stream_events.jsonl", **kwargs: object
) -> tuple[GnaniPrismaSession, FakeConnection]:
    from pathlib import Path

    path = Path(__file__).parent.parent / "fixtures" / name
    frames = [line for line in path.read_text().splitlines() if line.strip()]
    connection = FakeConnection(frames, **kwargs)  # type: ignore[arg-type]
    return GnaniPrismaSession(connection, CONFIG), connection  # type: ignore[arg-type]


async def test_recorded_frames_map_to_our_event_vocabulary() -> None:
    session, _ = session_from()
    events = [event async for event in session.events()]
    kinds = [event.kind for event in events]
    assert kinds == [
        SttEventKind.READY,
        SttEventKind.SPEECH_START,
        SttEventKind.TRANSCRIPT,
        SttEventKind.TRANSCRIPT,
        SttEventKind.ERROR,
    ]
    # `settings_updated` is an acknowledgement and is deliberately dropped.
    assert len(events) == 5


async def test_no_processing_event_is_invented() -> None:
    """DECISIONS 0012: the server was not observed to emit it, so filler timing cannot
    depend on it. If a future API version starts sending it, this test should fail and
    the decision should be revisited."""
    session, _ = session_from()
    events = [event async for event in session.events()]
    assert not any(isinstance(event, SttProcessing) for event in events)


async def test_the_ready_event_reports_the_frame_contract() -> None:
    session, _ = session_from()
    ready = await anext(session.events())
    assert isinstance(ready, SttReady)
    assert ready.frame_bytes == PRISMA_FRAME_BYTES
    assert ready.sample_rate == 16000
    assert ready.request_id == "01a1a7a8-4bd2-751f-bc7c-a685cce28f2e"


async def test_transcript_fields_are_carried_through() -> None:
    session, _ = session_from()
    transcripts = [e async for e in session.events() if isinstance(e, SttTranscript)]
    assert [t.text for t in transcripts] == ["haan beta", "Kaun bol raha hai?"]
    assert [t.segment_index for t in transcripts] == [1, 2]
    assert transcripts[0].audio_duration_ms == 900
    # Prisma reports latency as a float; our event type is an int.
    assert transcripts[0].provider_latency_ms == 69
    assert all(t.is_final for t in transcripts)


async def test_speech_start_carries_an_utterance_id() -> None:
    session, _ = session_from()
    start = next(e for e in [x async for x in session.events()] if isinstance(e, SttSpeechStart))
    assert start.utterance_id == "u-1"


async def test_an_error_frame_is_not_fatal() -> None:
    """A single failed utterance leaves the session usable."""
    session, _ = session_from()
    error = next(e for e in [x async for x in session.events()] if isinstance(e, SttStreamError))
    assert error.fatal is False
    assert "segment failed" in error.message


async def test_audio_frames_must_be_exactly_1024_bytes() -> None:
    session, connection = session_from()
    await session.send_audio(FRAME)
    assert connection.sent_binary == [FRAME]
    for bad in (b"", b"\x00" * 1023, b"\x00" * 1025):
        with pytest.raises(BadRequest, match="1024 bytes"):
            await session.send_audio(bad)


async def test_commit_sends_the_documented_json_message() -> None:
    session, connection = session_from()
    await session.commit()
    assert json.loads(connection.sent_text[0]) == {"type": "commit"}


async def test_audio_seconds_are_derived_from_the_negotiated_rate() -> None:
    """512 samples is 32 ms at 16 kHz and 64 ms at 8 kHz; billing depends on getting this right."""
    session, _ = session_from()
    for _ in range(10):
        await session.send_audio(FRAME)
    assert session.audio_seconds_sent == pytest.approx(0.32)

    slow = GnaniPrismaSession(
        FakeConnection([]),  # type: ignore[arg-type]
        CONFIG.model_copy(update={"sample_rate": 8000}),
    )
    for _ in range(10):
        await slow.send_audio(FRAME)
    assert slow.audio_seconds_sent == pytest.approx(0.64)


async def test_closing_charges_for_audio_not_yet_billed() -> None:
    from app.core.cost import CostModel

    class Recorder:
        def __init__(self) -> None:
            self.cost = CostModel(rates_confirmed=True, stt_rate_per_minute=1.0)
            self.charges: list[object] = []

        async def check(self, provider: str, charge: object) -> None:
            pass

        async def record(self, provider: str, charge: object, **kw: object) -> None:
            self.charges.append(charge)

    recorder = Recorder()
    session = GnaniPrismaSession(FakeConnection([]), CONFIG, usage=recorder)  # type: ignore[arg-type]
    for _ in range(10):
        await session.send_audio(FRAME)
    assert recorder.charges == []  # below the flush threshold
    await session.aclose()
    assert len(recorder.charges) == 1
    assert recorder.charges[0].raw_units == pytest.approx(0.32)  # type: ignore[attr-defined]
    # 0.32 s rounds up to the 0.05-minute minimum.
    assert recorder.charges[0].billed_units == pytest.approx(0.05)  # type: ignore[attr-defined]


async def test_a_closed_session_refuses_more_audio() -> None:
    session, connection = session_from()
    await session.aclose()
    assert connection.closed
    with pytest.raises(BadRequest, match="closed"):
        await session.send_audio(FRAME)
    with pytest.raises(BadRequest, match="closed"):
        await session.commit()


async def test_the_fifteen_minute_cap_surfaces_as_a_fatal_event() -> None:
    """Close code 1001 means the upstream session expired; the layer above must rotate."""
    session, _ = session_from(close=Close(CLOSE_SESSION_EXPIRED, "session limit"))
    events = [event async for event in session.events()]
    assert isinstance(events[-1], SttStreamError)
    assert events[-1].fatal is True
    assert "15-minute" in events[-1].message


async def test_a_rejected_handshake_surfaces_as_a_bad_request() -> None:
    session, _ = session_from(close=Close(CLOSE_BAD_HANDSHAKE, "Invalid x-substitution-map"))
    events = [event async for event in session.events()]
    assert isinstance(events[-1], SttStreamError)
    assert events[-1].fatal is True


def test_every_documented_setting_becomes_an_upgrade_header() -> None:
    """Prisma takes all its configuration as upgrade headers; nothing may be dropped."""
    provider = GnaniPrismaSTT(
        api_key="test-key", ws_url="wss://example.invalid/stream", model="prisma-v3.0"
    )
    headers = provider._headers(CONFIG)
    assert headers["x-api-key-id"] == "test-key"
    assert headers["model"] == "prisma-v3.0"
    assert headers["language_code"] == "indic-auto"
    assert headers["format"] == "transliterate"
    assert headers["x-sample-rate"] == "16000"
    assert headers["x-idle-timeout-sec"] == "900"
    assert json.loads(headers["keyterms"]) == ["UPI", "OTP"]
    assert headers["x-min-silence-ms"] == "500"
    assert headers["x-vad-threshold"] == "0.7"


def test_optional_settings_are_omitted_rather_than_sent_empty() -> None:
    provider = GnaniPrismaSTT(
        api_key="k", ws_url="wss://example.invalid/stream", model="prisma-v3.0"
    )
    headers = provider._headers(SttConfig(model="prisma-v3.0", language_code="hi-IN"))
    assert "keyterms" not in headers
    assert "x-idle-timeout-sec" not in headers
    assert "format" not in headers
    assert "x-vad-threshold" not in headers


def test_an_empty_api_key_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="API key"):
        GnaniPrismaSTT(api_key="", ws_url="wss://example.invalid", model="prisma-v3.0")


async def test_an_unknown_frame_type_is_ignored_rather_than_crashing() -> None:
    """Forward compatibility: a new server event must not take down a live call."""
    connection = FakeConnection(
        [json.dumps({"type": "something_new", "timestamp": "2026-11-04T18:00:00Z"})]
    )
    session = GnaniPrismaSession(connection, CONFIG)  # type: ignore[arg-type]
    assert [event async for event in session.events()] == []


async def test_unparseable_frames_are_ignored() -> None:
    connection = FakeConnection(["not json at all"])
    session = GnaniPrismaSession(connection, CONFIG)  # type: ignore[arg-type]
    assert [event async for event in session.events()] == []


def test_connection_loss_without_a_code_is_reported_as_unavailable() -> None:
    from app.providers.gnani_prisma import _closed_error

    error = _closed_error(ConnectionClosedError(None, None))
    assert isinstance(error, ProviderUnavailable)
