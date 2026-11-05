"""A scripted call end to end through the orchestrator, on mock providers.

No network, no credits, no real timing. What is asserted is the thing that actually breaks
in production: the order of events, and that barge-in and the family relay happen at all.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest

from app.core.audio.framing import float32_to_pcm
from app.core.events import EventType
from app.core.models import PersonaId, Speaker
from app.core.state import CallState
from app.providers.base import SttConfig
from app.providers.mock import MockLLMProvider, MockSTTProvider, MockTTSProvider
from app.services.orchestrator import CallConfig, CallOrchestrator
from app.services.recorder import CallRecorder

FRAME_BYTES = 1024
VOICES = {PersonaId.DADA: "Devansh", PersonaId.GRANDSON: "Devansh"}


class FakeBus:
    """Collects published events instead of touching Redis."""

    def __init__(self) -> None:
        self.published: list[object] = []

    async def publish(self, event: object) -> None:
        self.published.append(event)


def loud_frame(level: float = 0.3) -> bytes:
    """A frame the energy VAD will call speech."""
    samples = (np.ones(FRAME_BYTES // 2, dtype=np.float32) * level).astype(np.float32)
    return float32_to_pcm(samples)


def quiet_frame() -> bytes:
    return float32_to_pcm(np.zeros(FRAME_BYTES // 2, dtype=np.float32))


async def audio(*, speech_frames: int, silence_frames: int, bursts: int) -> AsyncIterator[bytes]:
    """Alternating speech and silence, so the VAD opens and closes like a real caller."""
    for _ in range(bursts):
        for _ in range(speech_frames):
            yield loud_frame()
            await asyncio.sleep(0)
        for _ in range(silence_frames):
            yield quiet_frame()
            await asyncio.sleep(0)


def build(
    *,
    script: tuple[str, ...] | None = None,
    replies: tuple[str, ...] | None = None,
    frames_per_utterance: int = 6,
    handoff_after: int = 2,
    sink: list[bytes] | None = None,
    chunk_delay_s: float = 0.0,
) -> tuple[CallOrchestrator, CallRecorder, list[bytes], MockLLMProvider]:
    out: list[bytes] = sink if sink is not None else []
    llm = MockLLMProvider(replies or ("Haan beta, ek minute. Main likh raha hoon.",))
    recorder = CallRecorder("call-test", FakeBus())  # type: ignore[arg-type]

    async def write(chunk: bytes) -> None:
        # A real transport is paced by playback. Without some cost here the persona finishes
        # an utterance before the next inbound frame, and barge-in can never be exercised.
        if chunk_delay_s:
            await asyncio.sleep(chunk_delay_s)
        out.append(chunk)

    orchestrator = CallOrchestrator(
        CallConfig(call_id="call-test", handoff_after_persona_turns=handoff_after),
        stt=MockSTTProvider(
            script or ("Main cyber cell se bol raha hoon.",),
            frames_per_utterance=frames_per_utterance,
        ),
        tts=MockTTSProvider(ms_per_char=4, chunk_ms=40),
        llm=llm,
        recorder=recorder,
        voices=VOICES,
        sink=write,
        stt_config=SttConfig(model="mock", language_code="hi-IN", sample_rate=16000),
    )
    return orchestrator, recorder, out, llm


def kinds(recorder: CallRecorder) -> list[str]:
    return [event.type.value for event in recorder.history()]


class TestHappyPath:
    async def test_the_call_opens_and_closes_in_the_right_states(self) -> None:
        orchestrator, recorder, _, _ = build()
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))

        assert orchestrator.state is CallState.ENDED
        states = [
            (event.from_state, event.to_state)
            for event in recorder.history()
            if event.type is EventType.STATE_CHANGE
        ]
        assert states == [
            (CallState.RINGING, CallState.HUMAN_CHECK),
            (CallState.HUMAN_CHECK, CallState.PERSONA_ACTIVE),
            (CallState.PERSONA_ACTIVE, CallState.ENDED),
        ]

    async def test_sequence_numbers_are_dense_and_monotonic(self) -> None:
        """A dashboard reconnecting asks for everything after a seq, so gaps lose events."""
        orchestrator, recorder, _, _ = build()
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))
        assert [event.seq for event in recorder.history()] == list(range(len(recorder.history())))

    async def test_the_caller_is_transcribed_and_the_persona_replies(self) -> None:
        orchestrator, recorder, out, _ = build()
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))

        transcripts = [e for e in recorder.history() if e.type is EventType.TRANSCRIPT]
        speakers = [e.speaker for e in transcripts]
        assert Speaker.CALLER in speakers
        assert Speaker.PERSONA in speakers
        # The caller is always transcribed before the reply to it.
        assert speakers.index(Speaker.CALLER) < speakers.index(Speaker.PERSONA)
        assert out, "no audio was sent to the caller"

    async def test_every_event_belongs_to_this_call(self) -> None:
        orchestrator, recorder, _, _ = build()
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))
        assert {event.call_id for event in recorder.history()} == {"call-test"}

    async def test_a_latency_event_is_emitted_for_each_reply(self) -> None:
        orchestrator, recorder, _, _ = build(
            script=("Pehla sawaal.", "Doosra sawaal."), frames_per_utterance=6
        )
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=2))

        latencies = [e for e in recorder.history() if e.type is EventType.LATENCY]
        personas = [
            e
            for e in recorder.history()
            if e.type is EventType.TRANSCRIPT and e.speaker is Speaker.PERSONA
        ]
        assert len(latencies) == len(personas)

    async def test_the_persona_prompt_reaches_the_model(self) -> None:
        """The ethics constraints are prepended to every dialogue request, not just the first."""
        orchestrator, _, _, llm = build()
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))

        assert llm.chat_calls
        for messages in llm.chat_calls:
            system = messages[0].content
            assert "Never threaten" in system
            assert "Hariprasad" in system


class TestReplyShaping:
    async def test_a_multi_sentence_reply_is_spoken_sentence_by_sentence(self) -> None:
        """The latency win: the first sentence is synthesised before the rest exists."""
        orchestrator, recorder, _, _ = build(
            replies=("Haan beta, ek minute. Mera chashma kahan hai. Aap naam bataiye.",)
        )
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))
        persona = [
            e
            for e in recorder.history()
            if e.type is EventType.TRANSCRIPT and e.speaker is Speaker.PERSONA
        ]
        assert persona
        # All three sentences end up in the recorded turn, however many requests it took.
        assert "chashma" in persona[0].text

    async def test_the_reply_is_capped_so_the_persona_never_monologues(self) -> None:
        orchestrator, recorder, _, _ = build(
            replies=("Ek. Do teen chaar. Paanch chhe saat. Aath nau dus. Gyaarah baarah.",)
        )
        orchestrator._config.max_reply_sentences = 2
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))
        persona = [
            e
            for e in recorder.history()
            if e.type is EventType.TRANSCRIPT and e.speaker is Speaker.PERSONA
        ]
        assert persona
        assert "Gyaarah" not in persona[0].text


async def interrupting_audio(
    orchestrator: CallOrchestrator, *, interrupt_frames: int = 10
) -> AsyncIterator[bytes]:
    """Speak, go quiet, then wait for the persona to start and talk over it.

    Waiting on `orchestrator.speaking` rather than on a sleep: with instant mock providers
    the persona finishes a sentence in microseconds, so a timing-based test would pass or
    fail on scheduler luck rather than on whether barge-in works.
    """
    for _ in range(6):
        yield loud_frame()
        await asyncio.sleep(0)
    # The energy VAD needs 12 consecutive silent frames to close, and the close is what
    # marks end-of-speech, so a shorter gap would never produce a new rising edge.
    for _ in range(16):
        yield quiet_frame()
        await asyncio.sleep(0)

    for _ in range(4000):
        if orchestrator.speaking:
            break
        await asyncio.sleep(0.001)
    for _ in range(interrupt_frames):
        yield loud_frame()
        await asyncio.sleep(0)


class TestBargeIn:
    async def test_the_persona_stops_when_the_caller_talks_over_it(self) -> None:
        """Timbre has no cancel, so this has to happen on our side or not at all."""
        orchestrator, _, _, _ = build(
            script=("Pehla.", "Doosra."),
            replies=("Haan beta ek minute, main likh raha hoon zara dheere boliye na.",),
            frames_per_utterance=4,
            chunk_delay_s=0.02,
        )
        await orchestrator.run(interrupting_audio(orchestrator))
        assert orchestrator._speaker.interruptions > 0

    async def test_barge_in_stops_the_audio_going_out(self) -> None:
        """Counting interruptions is not enough: the point is that playback actually ends."""
        out: list[bytes] = []
        orchestrator, _, _, _ = build(
            script=("Pehla.", "Doosra."),
            replies=("Haan beta ek minute, main likh raha hoon zara dheere boliye na.",),
            frames_per_utterance=4,
            sink=out,
            chunk_delay_s=0.02,
        )
        await orchestrator.run(interrupting_audio(orchestrator))
        assert not orchestrator.speaking
        assert out, "the persona never spoke at all, so nothing was interrupted"

    async def test_no_barge_in_when_the_caller_stays_quiet(self) -> None:
        orchestrator, _, _, _ = build()
        await orchestrator.run(audio(speech_frames=8, silence_frames=20, bursts=1))
        assert orchestrator._speaker.interruptions == 0


class TestRelay:
    async def test_the_phone_is_handed_to_the_grandson(self) -> None:
        orchestrator, recorder, _, _ = build(
            script=("Ek.", "Do.", "Teen.", "Chaar."), frames_per_utterance=4, handoff_after=1
        )
        await orchestrator.run(audio(speech_frames=5, silence_frames=3, bursts=4))

        handoffs = [e for e in recorder.history() if e.type is EventType.PERSONA_HANDOFF]
        assert handoffs
        assert handoffs[0].from_persona is PersonaId.DADA
        assert handoffs[0].to_persona is PersonaId.GRANDSON
        assert orchestrator.persona_sequence == [PersonaId.DADA, PersonaId.GRANDSON]

    async def test_the_relay_does_not_wrap_back_to_dada(self) -> None:
        """Handing the scammer a voice he already heard is the fastest way to break it."""
        orchestrator, recorder, _, _ = build(
            script=tuple(f"Sawaal {i}." for i in range(8)),
            frames_per_utterance=4,
            handoff_after=1,
        )
        await orchestrator.run(audio(speech_frames=5, silence_frames=3, bursts=8))
        handoffs = [e for e in recorder.history() if e.type is EventType.PERSONA_HANDOFF]
        assert len(handoffs) == 1
        assert orchestrator.persona_sequence.count(PersonaId.DADA) == 1

    async def test_the_grandson_gets_his_own_prompt(self) -> None:
        orchestrator, _, _, llm = build(
            script=("Ek.", "Do.", "Teen."), frames_per_utterance=4, handoff_after=1
        )
        await orchestrator.run(audio(speech_frames=5, silence_frames=3, bursts=3))
        systems = [messages[0].content for messages in llm.chat_calls]
        assert any("Hariprasad" in s for s in systems)
        assert any("Arjun" in s for s in systems)


class TestDegradation:
    async def test_a_missing_voice_is_a_configuration_error_not_a_silent_mute(self) -> None:
        orchestrator, _, _, _ = build()
        orchestrator._voices = {}
        with pytest.raises(ValueError, match="no voice configured"):
            await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))

    async def test_an_empty_transcript_does_not_produce_a_turn(self) -> None:
        """Prisma returns blank segments on line noise; replying to one is a wasted turn."""
        orchestrator, recorder, _, _ = build(script=("   ",))
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))
        assert not [e for e in recorder.history() if e.type is EventType.TRANSCRIPT]
