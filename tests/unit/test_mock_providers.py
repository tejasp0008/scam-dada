"""The mock providers' own contract.

Everything built in later phases is tested against these, so if they drift the whole suite
becomes meaningless.
"""

from __future__ import annotations

import re

import pytest
from pydantic import BaseModel

from app.core.audio.framing import PRISMA_FRAME_BYTES
from app.core.tags import ELDER_TAGS, sanitize_tags
from app.providers.base import (
    BadRequest,
    ChatMessage,
    Role,
    SttConfig,
    SttEventKind,
    SttTranscript,
    TtsRequest,
)
from app.providers.mock import (
    DEFAULT_PERSONA_REPLIES,
    DEFAULT_SCAMMER_SCRIPT,
    MockLLMProvider,
    MockSTTProvider,
    MockTTSProvider,
)

CONFIG = SttConfig(model="mock", language_code="hi-IN", sample_rate=16000)
FRAME = b"\x00" * PRISMA_FRAME_BYTES


async def collect_until_closed(session: object, limit: int) -> list[object]:
    events: list[object] = []
    async for event in session.events():  # type: ignore[attr-defined]
        events.append(event)
        if len(events) >= limit:
            break
    return events


async def test_a_session_announces_the_frame_contract() -> None:
    session = await MockSTTProvider().open_session(CONFIG)
    events = await collect_until_closed(session, 1)
    assert events[0].kind is SttEventKind.READY  # type: ignore[attr-defined]
    assert events[0].frame_bytes == PRISMA_FRAME_BYTES  # type: ignore[attr-defined]
    assert session.frame_bytes == PRISMA_FRAME_BYTES
    await session.aclose()


async def test_a_wrong_sized_frame_is_rejected() -> None:
    """The mock enforces Prisma's rule so the error surfaces in tests, not in production."""
    session = await MockSTTProvider().open_session(CONFIG)
    with pytest.raises(BadRequest, match="1024 bytes"):
        await session.send_audio(b"\x00" * 512)
    await session.aclose()


async def test_an_utterance_emits_speech_start_processing_then_transcript() -> None:
    session = await MockSTTProvider(frames_per_utterance=4).open_session(CONFIG)
    for _ in range(4):
        await session.send_audio(FRAME)
    kinds = [e.kind for e in await collect_until_closed(session, 5)]  # type: ignore[attr-defined]
    assert kinds[:4] == [
        SttEventKind.READY,
        SttEventKind.SPEECH_START,
        SttEventKind.PROCESSING,
        SttEventKind.TRANSCRIPT,
    ]
    await session.aclose()


async def test_commit_closes_the_utterance_early() -> None:
    session = await MockSTTProvider(frames_per_utterance=100).open_session(CONFIG)
    await session.send_audio(FRAME)
    await session.commit()
    events = await collect_until_closed(session, 4)
    transcripts = [e for e in events if isinstance(e, SttTranscript)]
    assert transcripts[0].text == DEFAULT_SCAMMER_SCRIPT[0]
    assert transcripts[0].is_final
    await session.aclose()


async def test_the_scripted_scammer_is_entirely_fictional() -> None:
    """The synthetic-data rule, enforced rather than trusted."""
    joined = " ".join(DEFAULT_SCAMMER_SCRIPT).lower()
    assert "@demo" in joined
    assert "fake" in joined


async def test_closing_a_session_ends_its_event_stream() -> None:
    session = await MockSTTProvider().open_session(CONFIG)
    await session.aclose()
    events = [event async for event in session.events()]
    assert events[0].kind is SttEventKind.READY  # type: ignore[attr-defined]
    with pytest.raises(BadRequest, match="closed"):
        await session.send_audio(FRAME)


async def test_synthesised_length_scales_with_the_text() -> None:
    tts = MockTTSProvider()
    short = TtsRequest(text="haan", voice="v", language="hi-en", model="m")
    long = TtsRequest(text="haan beta zara dheere boliye", voice="v", language="hi-en", model="m")
    short_bytes = sum([len(c.pcm) async for c in tts.synthesize(short)])
    long_bytes = sum([len(c.pcm) async for c in tts.synthesize(long)])
    assert long_bytes > short_bytes
    assert len(tts.requests) == 2


async def test_only_the_last_chunk_is_final() -> None:
    tts = MockTTSProvider()
    request = TtsRequest(text="ek minute beta", voice="v", language="hi-en", model="m")
    chunks = [chunk async for chunk in tts.synthesize(request)]
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert [c.is_final for c in chunks] == [False] * (len(chunks) - 1) + [True]


async def test_replies_stream_in_pieces_and_rotate() -> None:
    llm = MockLLMProvider()
    messages = [ChatMessage(role=Role.USER, content="hello")]
    first = "".join([token async for token in llm.stream_chat(messages)])
    second = "".join([token async for token in llm.stream_chat(messages)])
    assert first != second
    assert len(first.split()) > 1
    assert len(llm.chat_calls) == 2


class Analysis(BaseModel):
    stage: str = "hook"
    pressure: float = 0.0


async def test_complete_json_returns_a_queued_payload() -> None:
    llm = MockLLMProvider()
    llm.queue_json({"stage": "payment", "pressure": 0.9})
    result = await llm.complete_json([], Analysis)
    assert result.stage == "payment"


async def test_complete_json_falls_back_to_schema_defaults() -> None:
    """Analysis must degrade rather than break the persona loop."""
    assert (await MockLLMProvider().complete_json([], Analysis)).stage == "hook"


async def test_complete_json_raises_when_a_schema_needs_real_data() -> None:
    class Strict(BaseModel):
        required_field: str

    with pytest.raises(BadRequest, match="queue_json"):
        await MockLLMProvider().complete_json([], Strict)


class TestFixturesMatchThePersonas:
    """The mocks are what the integration suite and the dashboard are developed against.

    If a fixture uses a tag the persona is forbidden, the live filter strips it and the mock
    stops resembling what the real model is asked to produce - so the fixture drifts from
    reality in exactly the place that is hardest to notice.
    """

    @pytest.mark.parametrize("reply", DEFAULT_PERSONA_REPLIES)
    def test_every_canned_reply_uses_only_tags_an_elder_may_use(self, reply: str) -> None:
        cleaned, dropped = sanitize_tags(reply, ELDER_TAGS)
        assert dropped == (), f"{reply!r} uses {dropped}"
        assert cleaned == reply

    @pytest.mark.parametrize("reply", DEFAULT_PERSONA_REPLIES)
    def test_no_canned_reply_speaks_as_a_woman(self, reply: str) -> None:
        """The lead persona is a 74-year-old man; feminine verb forms are a leftover."""
        assert "rahi hoon" not in reply
        assert "bataungi" not in reply

    @pytest.mark.parametrize("line", DEFAULT_SCAMMER_SCRIPT)
    def test_the_scammer_script_stays_visibly_fictional(self, line: str) -> None:
        """Synthetic data only, and a fixture is the easiest place to break that rule."""
        assert not re.search(r"\b[6-9]\d{9}\b", line), "looks like a real mobile number"
        assert not re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", line), "looks like an Aadhaar"
        if "@" in line:
            assert "@demo" in line, "UPI handles in fixtures must be on @demo"
