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
from app.core.canary import issue
from app.core.events import EventType
from app.core.models import EntitySource, EntityType, PersonaId, ScamStage, Speaker
from app.core.personas import LEAD_PERSONA, REPLY_STYLE_NOTE
from app.core.state import CallState
from app.providers.base import Role, SttConfig
from app.providers.mock import MockLLMProvider, MockSTTProvider, MockTTSProvider
from app.services.canary_registry import CanaryRegistry
from app.services.orchestrator import MAX_FILLERS_PER_TURN, CallConfig, CallOrchestrator
from app.services.prompts import persona_fillers
from app.services.recorder import CallRecorder


class SlowLLM:
    """Streams a reply, but only after a delay longer than one filler phrase."""

    name = "slow-llm"

    def __init__(self, reply: str, delay_s: float = 0.6) -> None:
        self._reply = reply
        self._delay = delay_s
        self.chat_calls: list[object] = []

    async def stream_chat(self, messages: object, **kwargs: object) -> AsyncIterator[str]:
        self.chat_calls.append(messages)
        await asyncio.sleep(self._delay)
        for word in self._reply.split(" "):
            yield word + " "

    async def complete_json(self, messages: object, schema: type, **kwargs: object) -> object:
        return schema.model_validate({})


class SilentLLM:
    """Produces nothing at all, so only the fillers ever speak."""

    name = "silent-llm"

    def __init__(self) -> None:
        self.chat_calls: list[object] = []

    async def stream_chat(self, messages: object, **kwargs: object) -> AsyncIterator[str]:
        self.chat_calls.append(messages)
        await asyncio.sleep(1.2)
        return
        yield ""  # pragma: no cover - makes this an async generator

    async def complete_json(self, messages: object, schema: type, **kwargs: object) -> object:
        return schema.model_validate({})


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
    canary: CanaryRegistry | None = None,
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
        CallConfig(
            call_id="call-test",
            stt_config=SttConfig(model="mock", language_code="hi-IN", sample_rate=16000),
            tts_model="mock-tts",
            tts_language="hi-en",
            handoff_after_persona_turns=handoff_after,
        ),
        stt=MockSTTProvider(
            script or ("Main cyber cell se bol raha hoon.",),
            frames_per_utterance=frames_per_utterance,
        ),
        tts=MockTTSProvider(ms_per_char=4, chunk_ms=40),
        llm=llm,
        recorder=recorder,
        voices=VOICES,
        sink=write,
        canary=canary,
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
    """Who holds the phone. Rewritten after a live call where the grandson took it at turn 9
    and still had it at turn 21, and the scammer asked for Dada back and was refused."""

    async def test_the_phone_is_handed_to_the_grandson(self) -> None:
        orchestrator, recorder, _, _ = build(
            script=("Ek.", "Do.", "Teen.", "Chaar."), frames_per_utterance=4, handoff_after=1
        )
        await orchestrator.run(audio(speech_frames=5, silence_frames=3, bursts=4))

        handoffs = [e for e in recorder.history() if e.type is EventType.PERSONA_HANDOFF]
        assert handoffs
        assert handoffs[0].from_persona is PersonaId.DADA
        assert handoffs[0].to_persona is PersonaId.GRANDSON

    async def test_the_grandson_hands_the_phone_back(self) -> None:
        """He is an interlude. The elder does the baiting, so the phone returns to him."""
        orchestrator, recorder, _, _ = build(
            script=tuple(f"Sawaal {i}." for i in range(8)),
            frames_per_utterance=4,
            handoff_after=1,
        )
        orchestrator._config.grandson_turns_before_handback = 2
        await orchestrator.run(audio(speech_frames=5, silence_frames=3, bursts=8))

        moves = [
            (e.from_persona, e.to_persona)
            for e in recorder.history()
            if e.type is EventType.PERSONA_HANDOFF
        ]
        assert (PersonaId.DADA, PersonaId.GRANDSON) in moves
        assert (PersonaId.GRANDSON, PersonaId.DADA) in moves

    async def test_the_elder_ends_up_with_most_of_the_turns(self) -> None:
        """The thing that was actually wrong: the grandson did most of the talking."""
        orchestrator, recorder, _, _ = build(
            script=tuple(f"Sawaal {i}." for i in range(10)),
            frames_per_utterance=4,
            handoff_after=3,
        )
        await orchestrator.run(audio(speech_frames=5, silence_frames=3, bursts=10))

        said = [
            e.persona
            for e in recorder.history()
            if e.type is EventType.TRANSCRIPT and e.speaker is Speaker.PERSONA
        ]
        assert said.count(PersonaId.DADA) > said.count(PersonaId.GRANDSON)

    async def test_the_caller_asking_for_the_elder_brings_him_back(self) -> None:
        """A direct request beats any counter, and ignoring one is conspicuous."""
        orchestrator, recorder, _, _ = build(
            script=("Ek.", "Do.", "Aap apne dada ji se baat karwaiye."),
            frames_per_utterance=4,
            handoff_after=1,
        )
        orchestrator._config.grandson_turns_before_handback = 99
        await orchestrator.run(audio(speech_frames=5, silence_frames=3, bursts=3))

        handoffs = [e for e in recorder.history() if e.type is EventType.PERSONA_HANDOFF]
        back = [e for e in handoffs if e.to_persona is PersonaId.DADA]
        assert back, "the caller asked for Dada and did not get him"
        assert "asked for the elder" in back[-1].reason

    async def test_the_caller_asking_for_someone_else_brings_the_grandson(self) -> None:
        orchestrator, recorder, _, _ = build(
            script=("Mujhe samajh nahi aa raha, koi aur hai ghar mein?",),
            frames_per_utterance=6,
            handoff_after=99,
        )
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))

        handoffs = [e for e in recorder.history() if e.type is EventType.PERSONA_HANDOFF]
        assert handoffs
        assert handoffs[0].to_persona is PersonaId.GRANDSON
        assert "asked for someone else" in handoffs[0].reason

    async def test_a_handoff_reason_is_always_recorded(self) -> None:
        """It is shown on the dashboard, so an empty one is a blank line on camera."""
        orchestrator, recorder, _, _ = build(
            script=("Ek.", "Do.", "Teen."), frames_per_utterance=4, handoff_after=1
        )
        await orchestrator.run(audio(speech_frames=5, silence_frames=3, bursts=3))
        for event in recorder.history():
            if event.type is EventType.PERSONA_HANDOFF:
                assert event.reason.strip()

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


class TestHandoffContext:
    """The relay shares one history, which is why the new persona needs telling whose it is."""

    async def test_no_handoff_note_is_added_before_a_handoff(self) -> None:
        """The turn note is always sent - for the language rule - but says nothing of a relay."""
        orchestrator, _, _, llm = build()
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))
        for messages in llm.chat_calls:
            assert sum(m.role is Role.SYSTEM for m in messages) == 2
            assert "handed to you" not in messages[-1].content

    async def test_after_a_handoff_the_grandson_is_told_whose_lines_those_were(self) -> None:
        """On a real call he answered with his grandfather's line about his glasses."""
        orchestrator, _, _, llm = build(
            script=("Ek.", "Do.", "Teen."), frames_per_utterance=4, handoff_after=1
        )
        await orchestrator.run(audio(speech_frames=5, silence_frames=3, bursts=3))

        notes = [
            m.content for messages in llm.chat_calls for m in messages[1:] if m.role is Role.SYSTEM
        ]
        assert notes, "no handoff note was ever sent"
        assert "handed to you" in notes[-1]
        assert "Dada" in notes[-1]

    async def test_the_note_goes_last_so_it_is_closest_to_the_generation(self) -> None:
        orchestrator, _, _, llm = build(
            script=("Ek.", "Do.", "Teen."), frames_per_utterance=4, handoff_after=1
        )
        await orchestrator.run(audio(speech_frames=5, silence_frames=3, bursts=3))

        with_note = [m for m in llm.chat_calls if any(x.role is Role.SYSTEM for x in m[1:])]
        assert with_note
        assert with_note[-1][-1].role is Role.SYSTEM
        # The style reminder is the final line of that note, after the handoff text.
        assert with_note[-1][-1].content.endswith(REPLY_STYLE_NOTE)


class TestIntelligence:
    """The analysis pass, wired in. It must produce events and never affect the reply."""

    async def test_stage_entity_and_risk_events_are_emitted(self) -> None:
        orchestrator, recorder, _, _ = build(
            script=("Main Delhi cyber crime se inspector Sharma bol raha hoon.",),
            frames_per_utterance=6,
        )
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))

        kinds = {event.type for event in recorder.history()}
        assert EventType.STAGE in kinds
        assert EventType.RISK in kinds
        assert EventType.ENTITY in kinds

    async def test_the_stage_reaches_payment_on_a_payment_line(self) -> None:
        orchestrator, recorder, _, _ = build(
            script=("Do lakh fake.officer@demo par UPI se bhej dijiye.",), frames_per_utterance=6
        )
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))
        stages = [e for e in recorder.history() if e.type is EventType.STAGE]
        assert stages
        assert stages[-1].stage is ScamStage.PAYMENT

    async def test_entities_from_the_transcript_are_reported_once_each(self) -> None:
        """The dashboard list only grows, so a repeated handle must not appear twice."""
        line = "Do lakh fake.officer@demo par bhejiye."
        orchestrator, recorder, _, _ = build(script=(line, line), frames_per_utterance=5)
        await orchestrator.run(audio(speech_frames=6, silence_frames=3, bursts=2))

        handles = [
            e
            for e in recorder.history()
            if e.type is EventType.ENTITY and e.entity_type is EntityType.UPI_ID
        ]
        assert len(handles) == 1
        assert handles[0].value == "fake.officer@demo"
        assert handles[0].source is EntitySource.REGEX

    async def test_the_persona_still_replies_when_analysis_is_off(self) -> None:
        """Proof the reply path does not depend on the analysis path."""
        orchestrator, recorder, out, _ = build()
        orchestrator._config.analysis_enabled = False
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))

        assert out
        assert [
            e
            for e in recorder.history()
            if e.type is EventType.TRANSCRIPT and e.speaker is Speaker.PERSONA
        ]
        assert not [e for e in recorder.history() if e.type is EventType.STAGE]

    async def test_a_hook_is_armed_when_the_caller_threatens_to_leave(self) -> None:
        orchestrator, recorder, _, llm = build(
            script=(
                "Aap mera time waste kar rahe hain, main phone rakh raha hoon.",
                "Haan jaldi boliye.",
            ),
            frames_per_utterance=5,
            handoff_after=99,
        )
        await orchestrator.run(audio(speech_frames=6, silence_frames=3, bursts=2))

        risks = [e for e in recorder.history() if e.type is EventType.RISK]
        assert risks
        assert max(e.hangup_risk for e in risks) >= 0.45
        # The hook has to reach the dialogue prompt, or the risk score changed nothing.
        instructions = [
            m.content for messages in llm.chat_calls for m in messages if m.role is Role.SYSTEM
        ]
        assert any("close to ending the call" in text for text in instructions)

    async def test_no_hook_is_armed_on_a_calm_call(self) -> None:
        orchestrator, _, _, llm = build(
            script=("Haan ji, main cyber cell se bol raha hoon.",), frames_per_utterance=6
        )
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))
        instructions = [
            m.content for messages in llm.chat_calls for m in messages if m.role is Role.SYSTEM
        ]
        assert not any("close to ending the call" in text for text in instructions)


class TestCanaryInTheLoop:
    async def test_a_canary_from_an_earlier_call_is_flagged(self) -> None:
        """The strongest claim the system makes: two calls on one lead list."""
        earlier = issue("call-earlier")
        registry = CanaryRegistry("call-test")
        registry._prior = [("call-earlier", earlier)]

        orchestrator, recorder, _, _ = build(
            script=(f"Aapka account {earlier.upi_handle} verify karna hai.",),
            frames_per_utterance=6,
            canary=registry,
        )
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))

        matches = [e for e in recorder.history() if e.type is EventType.CANARY_MATCH]
        assert len(matches) == 1
        assert matches[0].issued_call_id == "call-earlier"
        assert matches[0].matched_field == "upi_handle"

    async def test_a_repeated_leak_is_reported_once(self) -> None:
        """Three identical alerts would make the dashboard look broken."""
        earlier = issue("call-earlier")
        registry = CanaryRegistry("call-test")
        registry._prior = [("call-earlier", earlier)]
        line = f"{earlier.upi_handle} par bhejiye."

        orchestrator, recorder, _, _ = build(
            script=(line, line, line), frames_per_utterance=4, canary=registry
        )
        await orchestrator.run(audio(speech_frames=5, silence_frames=3, bursts=3))
        assert len([e for e in recorder.history() if e.type is EventType.CANARY_MATCH]) == 1

    async def test_an_ordinary_call_flags_nothing(self) -> None:
        registry = CanaryRegistry("call-test")
        registry._prior = [("call-earlier", issue("call-earlier"))]
        orchestrator, recorder, _, _ = build(
            script=("Do lakh fake.officer@demo par bhejiye.",),
            frames_per_utterance=6,
            canary=registry,
        )
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))
        assert not [e for e in recorder.history() if e.type is EventType.CANARY_MATCH]

    async def test_each_call_gets_its_own_identity(self) -> None:
        assert CanaryRegistry("call-a").identity != CanaryRegistry("call-b").identity


class TestFillerCoverage:
    """There must never be dead silence while the model is thinking.

    One filler is not enough: a cached filler is served from disk and `speak` returns as soon
    as the bytes reach the transport, so the audio is queued rather than played. If the model
    is slower than one phrase, the line goes quiet the moment that phrase finishes.
    """

    async def test_a_slow_model_gets_more_than_one_filler(self) -> None:
        orchestrator, _, _, _ = build(replies=("Haan beta, ek minute. Main likh raha hoon.",))
        # Mock TTS reports a short duration, so the coverage loop decides it needs another.
        spoken: list[str] = []
        original = orchestrator._speaker.speak

        async def record(request: object, **kwargs: object) -> float | None:
            spoken.append(request.text)  # type: ignore[attr-defined]
            return await original(request, **kwargs)  # type: ignore[arg-type]

        orchestrator._speaker.speak = record  # type: ignore[method-assign]
        orchestrator._llm = SlowLLM("Haan beta, main sun raha hoon. Aap boliye na.")  # type: ignore[assignment]
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))

        fillers = [text for text in spoken if text in persona_fillers(LEAD_PERSONA)]
        assert len(fillers) >= 2, f"only {len(fillers)} filler(s) covered the wait"

    async def test_fillers_stop_once_the_reply_is_ready(self) -> None:
        """Otherwise the persona stalls three times and then answers."""
        orchestrator, _, _, _ = build(replies=("Haan beta ek minute. Main likh raha hoon.",))
        spoken: list[str] = []
        original = orchestrator._speaker.speak

        async def record(request: object, **kwargs: object) -> float | None:
            spoken.append(request.text)  # type: ignore[attr-defined]
            return await original(request, **kwargs)  # type: ignore[arg-type]

        orchestrator._speaker.speak = record  # type: ignore[method-assign]
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))

        fillers = [text for text in spoken if text in persona_fillers(LEAD_PERSONA)]
        assert len(fillers) <= MAX_FILLERS_PER_TURN

    async def test_the_cap_is_honoured_when_the_model_never_answers(self) -> None:
        """A dead model must not produce an endless stall."""
        orchestrator, _, _, _ = build()
        spoken: list[str] = []
        original = orchestrator._speaker.speak

        async def record(request: object, **kwargs: object) -> float | None:
            spoken.append(request.text)  # type: ignore[attr-defined]
            return await original(request, **kwargs)  # type: ignore[arg-type]

        orchestrator._speaker.speak = record  # type: ignore[method-assign]
        orchestrator._llm = SilentLLM()  # type: ignore[assignment]
        await orchestrator.run(audio(speech_frames=8, silence_frames=4, bursts=1))

        fillers = [text for text in spoken if text in persona_fillers(LEAD_PERSONA)]
        assert 1 <= len(fillers) <= MAX_FILLERS_PER_TURN

    async def test_a_filler_is_recorded_only_when_one_was_needed(self) -> None:
        """An instant model leaves no gap to cover, and `filler_played` says so.

        Reporting a filler that never played would be a fabricated number on the dashboard -
        the same class of mistake as the barge-in count that was removed in Phase 4.
        """
        instant, recorder, _, _ = build()
        await instant.run(audio(speech_frames=8, silence_frames=4, bursts=1))
        fast = [e for e in recorder.history() if e.type is EventType.LATENCY]
        assert fast
        assert not any(e.filler_played for e in fast)

    async def test_a_slow_model_does_record_a_filler(self) -> None:
        slow, recorder, _, _ = build()
        slow._llm = SlowLLM("Haan beta, main sun raha hoon. Aap boliye na.")  # type: ignore[assignment]
        await slow.run(audio(speech_frames=8, silence_frames=4, bursts=1))
        events = [e for e in recorder.history() if e.type is EventType.LATENCY]
        assert events
        assert any(e.filler_played for e in events)


class TestBargeInGate:
    """Barge-in needs a higher bar while the persona is audible, because on speakers the
    microphone hears the persona too and a flush is heard as the persona losing its voice."""

    async def test_a_quiet_echo_does_not_interrupt(self) -> None:
        """A live call produced four barge-ins in forty seconds and no audible reply."""
        orchestrator, _, _, _ = build(
            script=("Pehla.", "Doosra."),
            replies=("Haan beta ek minute, main likh raha hoon zara dheere boliye na.",),
            frames_per_utterance=4,
            chunk_delay_s=0.02,
        )
        orchestrator._config.barge_in_rms = 0.5  # far above the echo-level frames below
        await orchestrator.run(interrupting_audio(orchestrator))
        assert orchestrator._speaker.interruptions == 0

    async def test_sustained_loud_speech_still_interrupts(self) -> None:
        orchestrator, _, _, _ = build(
            script=("Pehla.", "Doosra."),
            replies=("Haan beta ek minute, main likh raha hoon zara dheere boliye na.",),
            frames_per_utterance=4,
            chunk_delay_s=0.02,
        )
        orchestrator._config.barge_in_rms = 0.1
        orchestrator._config.barge_in_frames = 3
        await orchestrator.run(interrupting_audio(orchestrator, interrupt_frames=20))
        assert orchestrator._speaker.interruptions > 0

    async def test_it_can_be_switched_off_entirely(self) -> None:
        orchestrator, _, _, _ = build(
            script=("Pehla.", "Doosra."),
            replies=("Haan beta ek minute, main likh raha hoon zara dheere boliye na.",),
            frames_per_utterance=4,
            chunk_delay_s=0.02,
        )
        orchestrator.set_barge_in(False)
        await orchestrator.run(interrupting_audio(orchestrator, interrupt_frames=30))
        assert orchestrator._speaker.interruptions == 0
