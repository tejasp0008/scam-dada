"""One object per call: listens, decides when to speak, speaks, and gets interrupted.

The shape of this loop is set by three measured facts about the providers, all recorded in
`docs/DECISIONS.md`:

* **Prisma emits no `processing` event** (0012), so there is no server-side end-of-speech
  signal. The local VAD closing its gate is the only cue we get that the caller has stopped,
  and it is therefore what starts the filler phrase.
* **Prisma returns no partial transcripts** (0002) and **Timbre documents no cancel**
  (0011), so barge-in is entirely local: the VAD opening cuts playback on our side and the
  audio already synthesised is simply discarded. It has already been billed either way.
* **Time to first token is 174-380 ms** (0010), which is only worth having if the reply is
  not then held back until it is complete. So the reply is streamed and cut at the first
  sentence boundary, and that first sentence goes to TTS while the rest is still arriving.

The dialogue path never awaits a database write, and the analysis path does not exist yet;
when it arrives it runs as a separate task whose failure cannot reach the caller's ear.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from app.core.audio.call_profile import PROFILES_BY_NAME, CallProfile
from app.core.audio.framing import FrameBuffer
from app.core.audio.vad import EnergyVad, Vad
from app.core.events import (
    LatencyEvent,
    PersonaHandoffEvent,
    StateChangeEvent,
    TranscriptEvent,
)
from app.core.models import PersonaId, Speaker, Turn
from app.core.personas import LEAD_PERSONA, PersonaProfile, next_in_relay, profile_for
from app.core.state import CallState, transition
from app.core.tags import sanitize_tags, tidy_for_speech
from app.core.turn import (
    FillerRotation,
    TurnBudget,
    flush_tail,
    is_handoff_due,
    recent_turns,
    take_sentence,
)
from app.providers.base import (
    ChatMessage,
    LLMProvider,
    ProviderError,
    Role,
    SttConfig,
    SttEvent,
    STTProvider,
    SttReady,
    SttSpeechStart,
    SttStreamError,
    SttTranscript,
    TTSProvider,
    TtsRequest,
)
from app.services.prompts import persona_fillers, persona_system_prompt
from app.services.recorder import CallRecorder
from app.services.speaker import PersonaSpeaker
from app.services.stt_rotation import RotatingSttSession

logger = logging.getLogger(__name__)

#: Audio out to the caller. Returns nothing; a sink that cannot keep up must buffer or drop
#: internally rather than applying backpressure to the dialogue path.
AudioSink = Callable[[bytes], Awaitable[None]]

#: How long teardown waits for a reply that is already in flight. The caller's audio ending
#: does not mean the persona has finished its sentence, and cutting it off mid-word is worse
#: than a slightly later close.
DRAIN_TIMEOUT_SECONDS = 30.0

#: Called when playback must stop immediately, so the transport can drop whatever it has
#: queued. Without this a browser with 2 seconds buffered keeps talking over the scammer
#: for 2 seconds after we decided to stop.
FlushSink = Callable[[], Awaitable[None]]


@dataclass(slots=True)
class CallConfig:
    """Everything about a call that is not a provider or a transport.

    The model names and the synthesis language come in from settings rather than being
    written here. Two reasons, and the second one cost a debugging session: a vendor model
    name in a service module breaks the rule that vendors are named only inside
    `app.providers`, and hardcoding the STT config here silently ignored `STT_FORMAT`, so a
    call transcribed in Devanagari instead of the transliterated Hinglish that DECISIONS
    0009 chose on measured evidence.
    """

    call_id: str
    stt_config: SttConfig
    tts_model: str
    tts_language: str
    sample_rate_in: int = 16000
    sample_rate_out: int = 24000
    #: Floor for the family relay. The real trigger is hang-up risk, which arrives with the
    #: analysis layer; this guarantees the handoff happens at all in a calm demo call.
    handoff_after_persona_turns: int = 4
    turn_budget: TurnBudget = field(default_factory=TurnBudget)
    max_reply_sentences: int = 3
    source: str = "browser"


@dataclass(slots=True)
class TurnLatency:
    """Milliseconds from the caller stopping speaking to each leg completing."""

    stt_final_ms: int | None = None
    llm_first_token_ms: int | None = None
    tts_first_byte_ms: int | None = None
    filler_played: bool = False

    def as_dict(self) -> dict[str, int | None]:
        return {
            "stt_final_ms": self.stt_final_ms,
            "llm_first_token_ms": self.llm_first_token_ms,
            "tts_first_byte_ms": self.tts_first_byte_ms,
        }


class CallOrchestrator:
    """Runs one baited call from answer to hang-up."""

    def __init__(
        self,
        config: CallConfig,
        *,
        stt: STTProvider,
        tts: TTSProvider,
        llm: LLMProvider,
        recorder: CallRecorder,
        voices: dict[PersonaId, str],
        sink: AudioSink,
        flush: FlushSink | None = None,
        vad: Vad | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._config = config
        self._stt = stt
        self._llm = llm
        self._recorder = recorder
        self._voices = voices
        self._sink = sink
        self._flush = flush
        self._vad = vad or EnergyVad()
        self._clock = clock or asyncio.get_event_loop().time

        self._speaker = PersonaSpeaker(tts, sink, flush=flush, clock=self._clock)
        self._state = CallState.RINGING
        self._persona: PersonaProfile = LEAD_PERSONA
        self._turns: list[Turn] = []
        self._turn_seq = 0
        self._started_at = self._clock()
        self._caller_stopped_at: float | None = None
        self._session: RotatingSttSession | None = None
        self._was_speaking = False
        self._rotations: dict[PersonaId, FillerRotation] = {}
        self.ended_reason: str | None = None
        self.persona_sequence: list[PersonaId] = [LEAD_PERSONA.persona_id]

    # ---- accessors used by the transport and by tests -------------------------------

    @property
    def state(self) -> CallState:
        return self._state

    @property
    def persona(self) -> PersonaProfile:
        return self._persona

    @property
    def turns(self) -> Sequence[Turn]:
        return tuple(self._turns)

    @property
    def speaking(self) -> bool:
        return self._speaker.speaking

    def profile(self) -> CallProfile:
        return PROFILES_BY_NAME[self._persona.call_profile]

    def elapsed_ms(self) -> int:
        return int((self._clock() - self._started_at) * 1000)

    # ---- state -----------------------------------------------------------------------

    async def _set_state(self, to_state: CallState, reason: str) -> None:
        previous = self._state
        self._state = transition(previous, to_state)

        def build(call_id: str, seq: int, ts: datetime) -> StateChangeEvent:
            return StateChangeEvent(
                call_id=call_id,
                seq=seq,
                ts=ts,
                from_state=previous,
                to_state=to_state,
                reason=reason,
            )

        await self._recorder.emit(build)

    # ---- the loop --------------------------------------------------------------------

    async def run(self, inbound: AsyncIterator[bytes]) -> None:
        """Answer, converse until the caller stops sending audio, then end."""
        session = await RotatingSttSession.open(self._stt, self._config.stt_config)
        self._session = session

        await self._set_state(CallState.HUMAN_CHECK, "answered")
        # Feature 6 (human vs robocall) decides this properly; until then we assume a human
        # so the persona loop is exercised. The state machine already models the other path.
        await self._set_state(CallState.PERSONA_ACTIVE, "assumed human pending human-check")

        events = asyncio.create_task(self._consume_events(session.events()))
        try:
            await self._pump_audio(inbound, session)
        finally:
            self.ended_reason = self.ended_reason or "caller audio ended"
            # Close the session first: that ends its event stream, which lets the consumer
            # finish the transcripts already queued instead of losing them. Cancelling here
            # instead would discard the caller's last utterance and the reply to it.
            await session.aclose()
            await self._drain_events(events)
            await self._speaker.stop()
            if self._state is not CallState.ENDED:
                await self._set_state(CallState.ENDED, self.ended_reason)
            await self._recorder.drain()

    async def _drain_events(self, events: asyncio.Task[None]) -> None:
        """Let queued transcripts and any in-flight reply finish, then stop waiting."""
        try:
            await asyncio.wait_for(events, timeout=DRAIN_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.warning(
                "timed out draining transcripts at the end of the call",
                extra={"call_id": self._config.call_id},
            )
            events.cancel()
            await asyncio.gather(events, return_exceptions=True)

    async def _pump_audio(self, inbound: AsyncIterator[bytes], session: RotatingSttSession) -> None:
        """Forward caller audio to the STT session, watching for barge-in and end of speech."""
        buffer = FrameBuffer(session.frame_bytes)
        async for data in inbound:
            for frame in buffer.push(data):
                speaking_now = self._vad.is_speech(frame)
                # Hold the STT rotation while the caller is mid-sentence: a seam inside an
                # utterance costs a transcript, a seam in a silence costs nothing.
                session.hold_rotation = speaking_now

                if speaking_now and not self._was_speaking and self._speaker.speaking:
                    await self._barge_in()
                if self._was_speaking and not speaking_now:
                    # The only end-of-speech signal there is: Prisma does not emit
                    # `processing` (DECISIONS 0012), so this is what starts the filler.
                    self._caller_stopped_at = self._clock()
                self._was_speaking = speaking_now
                await session.send_audio(frame)

    async def _barge_in(self) -> None:
        """The caller started talking while the persona was. Stop, immediately."""
        await self._speaker.stop()
        logger.info("barge-in", extra={"call_id": self._config.call_id})

    async def _consume_events(self, events: AsyncIterator[SttEvent]) -> None:
        """React to transcription events until the session ends."""
        async for event in events:
            if isinstance(event, SttReady):
                logger.info(
                    "stt ready",
                    extra={"sample_rate": event.sample_rate, "frame_bytes": event.frame_bytes},
                )
            elif isinstance(event, SttSpeechStart):
                self._caller_stopped_at = None
            elif isinstance(event, SttTranscript):
                if event.text.strip():
                    await self._on_transcript(event)
            elif isinstance(event, SttStreamError):
                logger.warning("stt error", extra={"message": event.message, "fatal": event.fatal})
                if event.fatal:
                    self.ended_reason = f"stt failed: {event.message}"
                    return

    async def _on_transcript(self, event: SttTranscript) -> None:
        """One complete caller utterance: record it, then reply."""
        latency = TurnLatency()
        stopped = self._caller_stopped_at
        self._caller_stopped_at = None
        if stopped is not None:
            latency.stt_final_ms = int((self._clock() - stopped) * 1000)

        caller_turn = Turn(
            seq=self._next_turn_seq(),
            speaker=Speaker.CALLER,
            text=event.text.strip(),
            language_code=event.detected_language_code,
            started_at_ms=max(0, self.elapsed_ms() - (event.audio_duration_ms or 0)),
            duration_ms=event.audio_duration_ms,
        )
        self._turns.append(caller_turn)
        await self._emit_transcript(caller_turn)
        self._recorder.record_turn(caller_turn)

        await self._maybe_hand_over()
        await self._reply(latency)

    async def _emit_transcript(self, turn: Turn) -> None:
        def build(call_id: str, seq: int, ts: datetime) -> TranscriptEvent:
            return TranscriptEvent(
                call_id=call_id,
                seq=seq,
                ts=ts,
                turn_seq=turn.seq,
                speaker=turn.speaker,
                text=turn.text,
                persona=turn.persona,
                language_code=turn.language_code,
                audio_duration_ms=turn.duration_ms,
            )

        await self._recorder.emit(build)

    def _next_turn_seq(self) -> int:
        seq = self._turn_seq
        self._turn_seq += 1
        return seq

    async def _maybe_hand_over(self) -> None:
        """Pass the phone to the next persona once the current one has said enough."""
        if not is_handoff_due(
            self._turns, after_persona_turns=self._config.handoff_after_persona_turns
        ):
            return
        successor = next_in_relay(self._persona.persona_id)
        if successor is None:
            return  # end of the relay; the current persona keeps the phone
        previous = self._persona
        self._persona = successor
        self.persona_sequence.append(successor.persona_id)

        def build(call_id: str, seq: int, ts: datetime) -> PersonaHandoffEvent:
            return PersonaHandoffEvent(
                call_id=call_id,
                seq=seq,
                ts=ts,
                from_persona=previous.persona_id,
                to_persona=successor.persona_id,
                reason=f"{previous.display_name} spoke "
                f"{self._config.handoff_after_persona_turns} turns",
            )

        await self._recorder.emit(build)

    # ---- speaking --------------------------------------------------------------------

    def _messages(self) -> list[ChatMessage]:
        """The dialogue request: constraints, persona, conversation, then who is speaking.

        The relay shares one history, which is the point - the grandson has to know what his
        grandfather was told. But the previous persona's replies sit in that history as
        `assistant` turns, so to the model they read as *its own* earlier words, and it
        continues them: on a real call the grandson's first line was Dada's line about
        looking for his glasses.

        So a handoff appends a note naming whose words those were. It goes last because the
        instruction that has to win is the one closest to the generation.
        """
        messages = [ChatMessage(role=Role.SYSTEM, content=persona_system_prompt(self._persona))]
        window = recent_turns(self._turns, self._config.turn_budget)
        for turn in window:
            role = Role.ASSISTANT if turn.speaker is Speaker.PERSONA else Role.USER
            messages.append(ChatMessage(role=role, content=turn.text))

        if (note := self._handoff_note(window)) is not None:
            messages.append(ChatMessage(role=Role.SYSTEM, content=note))
        return messages

    def _handoff_note(self, window: Sequence[Turn]) -> str | None:
        """Tell the current persona which of the replies above were somebody else's."""
        others = [
            turn.persona
            for turn in window
            if turn.speaker is Speaker.PERSONA
            and turn.persona is not None
            and turn.persona != self._persona.persona_id
        ]
        if not others:
            return None
        previous = profile_for(others[-1])
        return (
            f"The phone has just been handed to you. Any reply above that looks like yours "
            f"was spoken by {previous.display_name}, not by you. Do not repeat his lines, do "
            f"not continue his sentences, and do not borrow his mannerisms. Say briefly who "
            f"you are and carry on in your own voice."
        )

    async def _reply(self, latency: TurnLatency) -> None:
        """Generate and speak one persona turn."""
        filler = asyncio.create_task(self._play_filler(latency))
        spoken: list[str] = []
        buffer = ""
        sentences = 0
        started = self._clock()
        first_token_at: float | None = None

        try:
            async for piece in self._llm.stream_chat(self._messages()):
                if first_token_at is None:
                    first_token_at = self._clock()
                    latency.llm_first_token_ms = int((first_token_at - started) * 1000)
                buffer += piece
                while sentences < self._config.max_reply_sentences:
                    split = take_sentence(buffer)
                    if split is None:
                        break
                    sentence, buffer = split
                    await self._await_filler(filler)
                    await self._speak(sentence, latency, spoken)
                    sentences += 1
                if sentences >= self._config.max_reply_sentences:
                    break
        except ProviderError as exc:
            # The filler has already bought a couple of seconds and is the graceful
            # degradation: the persona has stalled rather than gone silent.
            logger.warning("dialogue generation failed", extra={"error": str(exc)})

        await self._await_filler(filler)
        if sentences < self._config.max_reply_sentences and (tail := flush_tail(buffer)):
            await self._speak(tail, latency, spoken)

        if spoken:
            await self._record_persona_turn(" ".join(spoken), latency)
        await self._emit_latency(latency)

    async def _play_filler(self, latency: TurnLatency) -> None:
        """Speak a cached stall phrase while the model is still generating."""
        rotation = self._rotation_for(self._persona)
        try:
            await self._speaker.speak(self._tts_request(rotation.next()), profile=self.profile())
        except ProviderError as exc:
            # A filler is a nicety. Losing it costs a second of silence, not the call.
            logger.info("filler playback failed", extra={"error": str(exc)})
            return
        latency.filler_played = True

    async def _await_filler(self, filler: asyncio.Task[None]) -> None:
        """Let the filler finish before the real reply starts, so they do not overlap."""
        if not filler.done():
            await asyncio.gather(filler, return_exceptions=True)

    def _rotation_for(self, persona: PersonaProfile) -> FillerRotation:
        if persona.persona_id not in self._rotations:
            self._rotations[persona.persona_id] = FillerRotation(
                persona_fillers(persona), seed=self._config.call_id
            )
        return self._rotations[persona.persona_id]

    def _tts_request(self, text: str) -> TtsRequest:
        cleaned, dropped = sanitize_tags(tidy_for_speech(text), self._persona.allowed_tags)
        if dropped:
            # Billed and then discarded by the provider, so this is real money on silence.
            logger.info(
                "stripped tags the persona may not use",
                extra={"call_id": self._config.call_id, "dropped": list(dropped)},
            )
        voice = self._voices.get(self._persona.persona_id, "")
        if not voice:
            raise ValueError(f"no voice configured for {self._persona.persona_id}")
        return TtsRequest(
            text=cleaned or text,
            voice=voice,
            language=self._config.tts_language,
            model=self._config.tts_model,
            sample_rate=self._config.sample_rate_out,
            call_id=self._config.call_id,
        )

    async def _speak(self, text: str, latency: TurnLatency, spoken: list[str]) -> None:
        """Synthesise and play one sentence."""
        try:
            first_byte = await self._speaker.speak(self._tts_request(text), profile=self.profile())
        except ProviderError as exc:
            logger.warning("synthesis failed", extra={"error": str(exc), "text": text[:40]})
            return
        if latency.tts_first_byte_ms is None and first_byte is not None:
            latency.tts_first_byte_ms = int(first_byte * 1000)
        spoken.append(text)

    async def _record_persona_turn(self, text: str, latency: TurnLatency) -> None:
        turn = Turn(
            seq=self._next_turn_seq(),
            speaker=Speaker.PERSONA,
            text=text,
            persona=self._persona.persona_id,
            started_at_ms=self.elapsed_ms(),
        )
        self._turns.append(turn)
        await self._emit_transcript(turn)
        self._recorder.record_turn(turn, latency=latency.as_dict())

    async def _emit_latency(self, latency: TurnLatency) -> None:
        turn_seq = max(0, self._turn_seq - 1)

        def build(call_id: str, seq: int, ts: datetime) -> LatencyEvent:
            return LatencyEvent(
                call_id=call_id,
                seq=seq,
                ts=ts,
                turn_seq=turn_seq,
                stt_final_ms=latency.stt_final_ms,
                llm_first_token_ms=latency.llm_first_token_ms,
                tts_first_byte_ms=latency.tts_first_byte_ms,
                filler_played=latency.filler_played,
            )

        await self._recorder.emit(build)
