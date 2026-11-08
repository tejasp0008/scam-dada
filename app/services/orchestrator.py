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
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar

from app.core.audio.call_profile import PROFILES_BY_NAME, CallProfile
from app.core.audio.framing import FrameBuffer
from app.core.audio.vad import EnergyVad, Vad, frame_rms
from app.core.canary import CanaryMatch
from app.core.events import (
    CanaryMatchEvent,
    EntityEvent,
    LatencyEvent,
    PersonaHandoffEvent,
    RiskEvent,
    StageEvent,
    StateChangeEvent,
    TranscriptEvent,
)
from app.core.models import Entity, PersonaId, ScamStage, Speaker, Turn
from app.core.personas import (
    LEAD_PERSONA,
    REPLY_STYLE_NOTE,
    PersonaProfile,
    hand_back,
    next_in_relay,
    profile_for,
)
from app.core.risk import PAYMENT_ELICITATION, BaitHook, choose_hook
from app.core.state import CallState, transition
from app.core.tags import has_speech, sanitize_tags, tidy_for_speech
from app.core.turn import (
    FillerRotation,
    TurnBudget,
    asks_for_elder,
    asks_for_younger,
    flush_tail,
    is_handoff_due,
    recent_turns,
    take_sentence,
)
from app.providers.base import (
    ChatMessage,
    LLMProvider,
    ProviderError,
    RateLimited,
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
from app.services.analysis import CallAnalyst, TurnAnalysis, deterministic_analysis
from app.services.canary_registry import CanaryRegistry
from app.services.prompts import dialogue_messages, persona_fillers
from app.services.recorder import CallRecorder
from app.services.speaker import PersonaSpeaker
from app.services.stt_rotation import RotatingSttSession

logger = logging.getLogger(__name__)

#: Audio out to the caller. Returns nothing; a sink that cannot keep up must buffer or drop
#: internally rather than applying backpressure to the dialogue path.
AudioSink = Callable[[bytes], Awaitable[None]]

#: Most stall phrases queued while waiting for one reply. A man who says "ek minute" three
#: times is still an old man; one who says it eight times is a loop.
MAX_FILLERS_PER_TURN = 3

#: How many times a rate-limited turn is asked for again. One. A second retry would hold the
#: line past what a caller sits through, and by then the honest answer is that the account has
#: no capacity left this minute (DECISIONS 0034).
RATE_LIMIT_RETRIES = 1

#: How long to stall before that retry, on top of letting the filler already playing finish.
#: Capped rather than taken from the server's `retry-after`, which is a whole minute: the
#: retry exists to ride out the second or two while a token window rolls over, not to wait
#: one out.
RATE_LIMIT_RETRY_WAIT = 1.5

#: Queue the next filler this long before the current one runs out, so the browser's buffer
#: never actually empties. Smaller than one filler and larger than a round trip to Timbre.
FILLER_LEAD_SECONDS = 0.45

#: How long teardown waits for a reply that is already in flight. The caller's audio ending
#: does not mean the persona has finished its sentence, and cutting it off mid-word is worse
#: than a slightly later close.
DRAIN_TIMEOUT_SECONDS = 30.0

#: How long teardown waits for analysis tasks. Shorter than the dialogue drain: an analysis
#: that has not finished by now is detail the evidence pack can do without, whereas a reply
#: cut off mid-word is something the scammer hears.
ANALYSIS_DRAIN_SECONDS = 10.0

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
    #: Floor for the family relay: hand the phone over after this many turns by the lead
    #: persona, if nothing else has triggered it. Raised from 4 after a live call where the
    #: grandson took over at turn 9 and still had it at turn 21 - the elder does the baiting
    #: and the grandson is meant to be an interlude.
    handoff_after_persona_turns: int = 6
    #: How many turns the grandson keeps the phone before handing it back unprompted.
    grandson_turns_before_handback: int = 2
    turn_budget: TurnBudget = field(default_factory=TurnBudget)
    #: How long one turn may run. What the caller experiences as latency is not the pipeline
    #: - first token is ~276 ms - but how long he waits before he can speak again.
    #:
    #: The sentence count stays at three: cutting to two produced forty-character stubs, and
    #: the prompt is right that a two-word reply reads as a broken machine rather than a slow
    #: man. Brevity is asked for in the turn note instead, where it can shorten a sentence
    #: rather than remove one. The character bound is a safety valve for a runaway reply, set
    #: well above the ~110 characters a normal turn produces (DECISIONS 0032).
    max_reply_sentences: int = 3
    max_reply_chars: int = 220
    source: str = "browser"
    #: Stream room tone between utterances so the line is never digitally silent. The
    #: operator can switch it off mid-call; see `set_room_tone`.
    room_tone: bool = True
    #: Barge-in while the persona is audible needs a higher bar than ordinary speech
    #: detection, because on speakers the microphone hears the persona too.
    #:
    #: The energy VAD opens on 3 frames above 0.015 - right for "is the caller talking" and
    #: far too eager for "is the caller talking *over us*", where our own voice is coming
    #: back through the room. Sustained and loud: 8 frames is 256 ms, which no echo of a
    #: single word survives, and which a real interruption clears easily.
    barge_in_enabled: bool = True
    barge_in_rms: float = 0.06
    barge_in_frames: int = 8

    analysis_enabled: bool = True
    analysis_model: str | None = None
    #: Analyse every Nth caller turn rather than all of them.
    #:
    #: Not a cost saving so much as a capacity one. The cap is 8000 tokens a minute per
    #: model, and analysis is the only request on the call that nobody is waiting for, so it
    #: is the one that gives way. Stage and risk move slowly - the stage never moves
    #: backwards at all - so a reading every fourth caller turn arrives before anything a
    #: human watching the dashboard would notice had changed.
    #:
    #: Turn one is always analysed, and so is every turn at the payment stage: see
    #: `_analysis_due`, which is where the exceptions and the reason for them live.
    analysis_every_n_turns: int = 3


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
        analysis_llm: LLMProvider | None = None,
        recorder: CallRecorder,
        voices: dict[PersonaId, str],
        sink: AudioSink,
        flush: FlushSink | None = None,
        vad: Vad | None = None,
        clock: Callable[[], float] | None = None,
        canary: CanaryRegistry | None = None,
    ) -> None:
        self._config = config
        self._stt = stt
        self._llm = llm
        self._recorder = recorder
        self._voices = voices
        self._sink = sink
        self._flush = flush
        self._vad = vad or EnergyVad()
        # Monotonic rather than the loop's clock, for the same reason as the speaker: an
        # orchestrator should be constructible outside a running loop.
        self._clock = clock or time.monotonic

        self._speaker = PersonaSpeaker(tts, sink, flush=flush, clock=self._clock)
        self._state = CallState.RINGING
        self._persona: PersonaProfile = LEAD_PERSONA
        self._turns: list[Turn] = []
        self._turn_seq = 0
        self._started_at = self._clock()
        self._caller_stopped_at: float | None = None
        self._session: RotatingSttSession | None = None
        self._was_speaking = False
        self._loud_run = 0
        self._rotations: dict[PersonaId, FillerRotation] = {}
        # A dedicated analysis provider already has its model and its reasoning effort
        # baked in, so there is nothing left to override. Without one the analyst shares the
        # dialogue provider and names the model per request, which is what the tests do.
        self._analyst = CallAnalyst(
            analysis_llm or llm, model=None if analysis_llm else config.analysis_model
        )
        self._canary = canary or CanaryRegistry(config.call_id)
        self._analyses: set[asyncio.Task[None]] = set()
        self._latest: TurnAnalysis | None = None
        self._used_hooks: list[str] = []
        self._handoff_reason = ""
        self._pending_hook: BaitHook | None = None
        self._pending_trigger = ""
        self._reported_entities: set[tuple[str, str]] = set()
        self._caller_turns_seen = 0
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

    @property
    def room_tone_enabled(self) -> bool:
        """Whether the line carries room tone between utterances."""
        return self._speaker.room_tone_enabled

    def set_room_tone(self, enabled: bool) -> None:
        """Turn the background room tone on or off mid-call."""
        self._speaker.set_room_tone(enabled)
        logger.info(
            "room tone toggled",
            extra={"call_id": self._config.call_id, "enabled": enabled},
        )

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
        await self._canary.start()
        if self._config.room_tone:
            self._speaker.start_room_tone(self.profile(), self._config.sample_rate_out)
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
            await self._drain_analyses()
            # `abandon`, not `stop`: the call ending is not the caller cutting in, and
            # counting it would add one phantom barge-in to every call ever made.
            await self._speaker.abandon()
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

                if await self._should_barge_in(frame, speaking_now):
                    await self._barge_in()
                if self._was_speaking and not speaking_now:
                    # The only end-of-speech signal there is: Prisma does not emit
                    # `processing` (DECISIONS 0012), so this is what starts the filler.
                    self._caller_stopped_at = self._clock()
                self._was_speaking = speaking_now
                await session.send_audio(frame)

    async def _should_barge_in(self, frame: bytes, speaking_now: bool) -> bool:
        """Whether this frame is the caller genuinely talking over the persona.

        Not simply "the VAD opened while we were speaking". On speakers the microphone hears
        the persona, so that test fires on our own voice coming back through the room - and
        because a barge-in flushes everything queued, the caller then hears almost nothing
        at all. A live call produced four of them in forty seconds and no audible reply.

        So while the persona is audible the bar is raised: sustained speech, well above the
        level a room echo returns. Headphones remove the problem entirely, which is what the
        page says; this makes speakers survivable rather than good.
        """
        if not self._config.barge_in_enabled or not self._persona_audible():
            self._loud_run = 0
            return False
        if frame_rms(frame) >= self._config.barge_in_rms and speaking_now:
            self._loud_run += 1
        else:
            self._loud_run = 0
        if self._loud_run < self._config.barge_in_frames:
            return False
        self._loud_run = 0
        return True

    @property
    def barge_in_enabled(self) -> bool:
        """Whether the caller can interrupt the persona."""
        return self._config.barge_in_enabled

    def set_barge_in(self, enabled: bool) -> None:
        """Turn barge-in on or off mid-call."""
        self._config.barge_in_enabled = enabled
        self._loud_run = 0
        logger.info(
            "barge-in toggled",
            extra={"call_id": self._config.call_id, "enabled": enabled},
        )

    def _persona_audible(self) -> bool:
        """Whether the caller can currently hear the persona.

        `speaking` alone is not enough: it covers synthesis and handing bytes to the
        transport, which together take a fraction of the seconds the audio then takes to
        play. Checking only that meant barge-in fired during synthesis and never during
        playback - the part the caller is actually talking over.
        """
        return self._speaker.speaking or self._speaker.transport_busy

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

        await self._check_canary(caller_turn)
        self._start_analysis(caller_turn.seq)
        await self._maybe_hand_over(caller_turn.text)
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

    async def _maybe_hand_over(self, heard: str = "") -> None:
        """Decide who should be holding the phone, and say so if it changes.

        Three triggers, in priority order, because the caller's own words beat any counter:

        1. **He asked for someone.** "Aap apne dada ji se baat karwaiye" is the clearest
           handoff signal there is, and on a live call the system could not comply - it had
           no way to go backwards. Ignoring a direct request is conspicuous.
        2. **The grandson has had his turns.** He is an interlude; the elder does the baiting
           and the phone belongs with him by default.
        3. **The lead has spoken enough** to make the relay worth showing at all.
        """
        successor = self._who_should_hold(heard)
        if successor is None or successor.persona_id is self._persona.persona_id:
            return
        previous = self._persona
        reason = self._handoff_reason
        self._persona = successor
        self.persona_sequence.append(successor.persona_id)

        def build(call_id: str, seq: int, ts: datetime) -> PersonaHandoffEvent:
            return PersonaHandoffEvent(
                call_id=call_id,
                seq=seq,
                ts=ts,
                from_persona=previous.persona_id,
                to_persona=successor.persona_id,
                reason=reason,
            )

        await self._recorder.emit(build)

    def _who_should_hold(self, heard: str) -> PersonaProfile | None:
        """Which persona the next reply belongs to. None means no change."""
        current = self._persona.persona_id
        is_lead = current is LEAD_PERSONA.persona_id

        if heard:
            if is_lead and asks_for_younger(heard):
                self._handoff_reason = "the caller asked for someone else"
                return next_in_relay(current)
            if not is_lead and asks_for_elder(heard):
                self._handoff_reason = "the caller asked for the elder back"
                return hand_back(current)

        persona_turns = sum(
            1 for turn in self._turns if turn.speaker is Speaker.PERSONA and turn.persona is current
        )
        if not is_lead and persona_turns >= self._config.grandson_turns_before_handback:
            self._handoff_reason = (
                f"{self._persona.display_name} has asked his questions and hands back"
            )
            return hand_back(current)

        if is_lead and is_handoff_due(
            self._turns, after_persona_turns=self._config.handoff_after_persona_turns
        ):
            self._handoff_reason = (
                f"{self._persona.display_name} spoke "
                f"{self._config.handoff_after_persona_turns} turns"
            )
            return next_in_relay(current)
        return None

    # ---- speaking --------------------------------------------------------------------

    def _messages(self) -> list[ChatMessage]:
        """The dialogue request: constraints, persona, conversation, then a note for this turn.

        The relay shares one history, which is the point - the grandson has to know what his
        grandfather was told. But the previous persona's replies sit in that history as
        `assistant` turns, so to the model they read as *its own* earlier words, and it
        continues them: on a real call the grandson's first line was Dada's line about
        looking for his glasses.

        So anything that must hold for *this* reply - whose words those were, a bait hook, the
        language and length to answer in - goes into one system note after the conversation.
        Three separate failures taught the same lesson: an instruction in the system prompt is
        the furthest message from the generation, and the one closest to it is the one wins.
        """
        window = recent_turns(self._turns, self._config.turn_budget)
        # Language and length last of all: form rules are the ones the model drops first.
        notes = [self._handoff_note(window), self._hook_note(), REPLY_STYLE_NOTE]
        return dialogue_messages(self._persona, window, notes)

    #: Why a hook was armed, which decides how it is introduced to the model. A delay hook
    #: needs the model to know he is leaving; telling it that in front of an elicitation hook
    #: would make the persona hurry, and hurrying is what loses the details.
    _HOOK_LEAD_IN: ClassVar[dict[str, str]] = {
        "hangup_risk": "The caller is close to ending the call.",
        "payment_stage": "He is asking for money, which is the one moment he will give up "
        "his own account details. Stay willing and slow; do not refuse and do not hurry.",
    }

    def _hook_note(self) -> str | None:
        """The armed bait hook as an instruction for this reply, if one is armed."""
        taken = self._take_hook()
        if taken is None:
            return None
        hook, trigger = taken
        lead_in = self._HOOK_LEAD_IN.get(trigger, "")
        return (
            f"{lead_in} Work the following into this reply, in character and without "
            f"hurrying: {hook.instruction}"
        ).strip()

    def _take_hook(self) -> tuple[BaitHook, str] | None:
        """Consume the armed hook, if there is one. Each hook is used at most once."""
        hook = self._pending_hook
        if hook is None:
            return None
        trigger = self._pending_trigger
        self._pending_hook = None
        self._pending_trigger = ""
        self._used_hooks.append(hook.name)
        return hook, trigger

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
        # Set the moment the first real sentence exists, which is what stops the fillers.
        ready = asyncio.Event()
        filler = asyncio.create_task(self._cover_silence(latency, ready))
        spoken: list[str] = []
        buffer = ""
        sentences = 0
        interrupted = False
        started = self._clock()
        first_token_at: float | None = None

        for attempt in range(1 + RATE_LIMIT_RETRIES):
            try:
                async for piece in self._llm.stream_chat(self._messages()):
                    if first_token_at is None:
                        first_token_at = self._clock()
                        latency.llm_first_token_ms = int((first_token_at - started) * 1000)
                    buffer += piece
                    while not self._reply_full(spoken, sentences):
                        split = take_sentence(buffer)
                        if split is None:
                            break
                        sentence, buffer = split
                        if not has_speech(sentence):
                            # Only audio tags. Billable, and it arrives as a breath with no
                            # words attached - so it is dropped rather than counted or spoken.
                            logger.info(
                                "dropped a sentence with no words in it",
                                extra={"call_id": self._config.call_id, "text": sentence[:40]},
                            )
                            continue
                        ready.set()
                        await self._await_filler(filler)
                        if not await self._speak(sentence, latency, spoken):
                            # Cut off, or the synthesiser refused. Either way the rest of this
                            # reply is stale: the caller has moved on and is already talking.
                            interrupted = True
                            break
                        sentences += 1
                    if interrupted:
                        break
                    if self._reply_full(spoken, sentences):
                        break
                break
            except RateLimited as exc:
                if spoken or attempt == RATE_LIMIT_RETRIES:
                    logger.warning("dialogue rate limited", extra={"error": str(exc)})
                    break
                # Every model on the account is capped per minute and the cap is per model,
                # so this means the dialogue model *and* its fallback are both out. The one
                # thing that must not happen is silence, so the persona goes on stalling -
                # a fresh run of filler phrases, which is in character for a slow old man -
                # and the turn is asked for again. The wait is bounded well under what a
                # scammer tolerates: better a stall he has heard before than dead air.
                wait = min(exc.retry_after or RATE_LIMIT_RETRY_WAIT, RATE_LIMIT_RETRY_WAIT)
                logger.warning(
                    "dialogue rate limited; stalling and retrying",
                    extra={"call_id": self._config.call_id, "wait_s": round(wait, 2)},
                )
                buffer = ""
                # Let the stall already in flight finish before starting another, or two
                # filler tasks speak over each other. Waiting for it *is* most of the delay
                # this retry needs, which is why the sleep after it is short.
                await self._await_filler(filler)
                filler = asyncio.create_task(self._cover_silence(latency, ready))
                await asyncio.sleep(wait)
                # `started` is deliberately not reset. What goes on the dashboard is how long
                # the caller waited, and he waited through both attempts.
                continue
            except ProviderError as exc:
                # The filler has already bought a couple of seconds and is the graceful
                # degradation: the persona has stalled rather than gone silent.
                logger.warning("dialogue generation failed", extra={"error": str(exc)})
                break

        ready.set()
        await self._await_filler(filler)
        if (
            not interrupted
            and not self._reply_full(spoken, sentences)
            and (tail := flush_tail(buffer))
            and has_speech(tail)
        ):
            await self._speak(tail, latency, spoken)

        if spoken:
            await self._record_persona_turn(" ".join(spoken), latency)
            await self._emit_latency(latency)
        else:
            # Nothing reached the caller - the model produced nothing, synthesis failed, or
            # the whole reply was cut off. Reporting "first token 300 ms" for a turn he
            # never heard would put a timing on the dashboard for speech that does not
            # exist, which is the same mistake as recording the text.
            logger.info(
                "turn produced no audible reply",
                extra={"call_id": self._config.call_id, "interrupted": interrupted},
            )

    def _reply_full(self, spoken: Sequence[str], sentences: int) -> bool:
        """Has this turn said enough to stop reading the stream?

        The sentence count is what the persona prompt asks for. The character count is what
        actually bounds how long he talks: one long sentence passes the first test and still
        holds the line for eight seconds. Whatever is left of the stream is dropped unspoken
        and unrecorded, the same way an interrupted sentence is, so the transcript continues
        to hold only what the caller heard.
        """
        if sentences >= self._config.max_reply_sentences:
            return True
        return sum(len(text) for text in spoken) >= self._config.max_reply_chars

    async def _cover_silence(self, latency: TurnLatency, ready: asyncio.Event) -> None:
        """Keep stall phrases going until the first real sentence exists.

        One filler is not enough. A cached filler is served from disk and `speak` returns as
        soon as the bytes are handed to the transport, so the audio is *queued*, not played -
        the browser's ring buffer does the pacing. Queue three at once and the persona stalls
        three times and then answers; queue one and wait, and the line goes silent the moment
        that one finishes if the model is slower than about two seconds.

        So this paces itself by the audio it just queued: hand over a filler, wait almost that
        long, and only reach for another if the reply still is not ready. The browser's buffer
        never empties and never holds more than one spare phrase.
        """
        rotation = self._rotation_for(self._persona)
        for _ in range(MAX_FILLERS_PER_TURN):
            if ready.is_set():
                return
            try:
                await self._speaker.speak(
                    self._tts_request(rotation.next()), profile=self.profile()
                )
            except ProviderError as exc:
                # A filler is a nicety. Losing it costs a second of silence, not the call.
                logger.info("filler playback failed", extra={"error": str(exc)})
                return
            latency.filler_played = True

            queued = max(0.0, self._speaker.last_audio_seconds - FILLER_LEAD_SECONDS)
            try:
                await asyncio.wait_for(ready.wait(), timeout=queued)
            except TimeoutError:
                continue  # still nothing to say; stall again
            return

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

    async def _speak(self, text: str, latency: TurnLatency, spoken: list[str]) -> bool:
        """Synthesise and play one sentence. Returns whether it reached the caller.

        A sentence cut off by barge-in is **not** recorded as spoken. It used to be: `speak`
        returned only a latency, so an interrupted utterance looked identical to a completed
        one and the full sentence went into the transcript regardless. On a live call that
        put words in Dada's mouth that the caller never heard - which is the evidence pack
        telling a story the audio does not support.
        """
        request = self._tts_request(text)
        try:
            result = await self._speaker.speak(request, profile=self.profile())
        except ProviderError as exc:
            logger.warning("synthesis failed", extra={"error": str(exc), "text": text[:40]})
            return False
        if latency.tts_first_byte_ms is None and result.first_byte_seconds is not None:
            latency.tts_first_byte_ms = int(result.first_byte_seconds * 1000)
        if not result.completed:
            logger.info(
                "utterance cut short by the caller",
                extra={
                    "call_id": self._config.call_id,
                    "text": request.text[:60],
                    "delivered_seconds": round(result.delivered_seconds, 2),
                },
            )
            return False
        # What was actually sent to the synthesiser, not the model's raw text: the recorded
        # transcript and the evidence pack should say what the persona said, and the raw text
        # still carries the line breaks and the stripped tags that never reached the caller.
        spoken.append(request.text)
        return True

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

    # ---- analysis ---------------------------------------------------------------------

    def _start_analysis(self, turn_seq: int) -> None:
        """Run the analysis pass as its own task.

        Not awaited, and that is the whole design: the persona's reply must not wait for a
        second model call, must not change because of its result, and must not fail when it
        fails. A model that is slow, rate-limited or down costs the dashboard detail and
        nothing else.
        """
        if not self._config.analysis_enabled:
            return
        self._caller_turns_seen += 1
        if not self._analysis_due():
            return
        self._track(asyncio.create_task(self._analyse(turn_seq)))

    def _analysis_due(self) -> bool:
        """Whether this caller turn is worth a model call.

        An interval, with two exceptions that are both about what the pack ends up holding.

        The **first turn** always counts, so the dashboard is populated before the first
        reply rather than staying blank through the opening exchange.

        So does **any turn at the payment stage**, interval or not. That is where the account
        number, the IFSC and the payee name are said, each exactly once, and a skipped turn
        there is an identifier that never reaches the evidence pack - the thing the call is
        for. Measured: at a flat interval of four a scripted call came back with six entities
        where the same call analysed on every payment turn had nine. The stage is read by the
        deterministic pass, which is pure and local and costs nothing (DECISIONS 0034).
        """
        every = max(1, self._config.analysis_every_n_turns)
        if self._caller_turns_seen == 1 or self._caller_turns_seen % every == 1:
            return True
        return deterministic_analysis(self._turns, turn_seq=0).stage is ScamStage.PAYMENT

    async def _analyse(self, turn_seq: int) -> None:
        """Analyse the call so far and emit what it found."""
        try:
            analysis = await self._analyst.analyse(tuple(self._turns), turn_seq=turn_seq)
        except Exception:
            # `analyse` is documented not to raise, so reaching here is a bug rather than a
            # provider failure - logged loudly, and still never allowed near the call.
            logger.exception("analysis failed", extra={"call_id": self._config.call_id})
            return

        self._latest = analysis
        await self._emit_stage(analysis)
        await self._emit_entities(analysis)
        await self._emit_risk(analysis)
        self._arm_hook(analysis)

    async def _emit_stage(self, analysis: TurnAnalysis) -> None:
        def build(call_id: str, seq: int, ts: datetime) -> StageEvent:
            return StageEvent(
                call_id=call_id,
                seq=seq,
                ts=ts,
                stage=analysis.stage,
                confidence=analysis.stage_confidence,
                pressure_score=analysis.pressure_score,
            )

        await self._recorder.emit(build)

    async def _emit_entities(self, analysis: TurnAnalysis) -> None:
        """Emit entities not already reported, so the dashboard list only grows."""
        for entity in analysis.entities:
            key = (entity.entity_type.value, entity.value.casefold())
            if key in self._reported_entities:
                continue
            self._reported_entities.add(key)
            await self._emit_entity(entity)
            self._recorder.record_entity(entity)

    async def _emit_entity(self, entity: Entity) -> None:
        def build(call_id: str, seq: int, ts: datetime) -> EntityEvent:
            return EntityEvent(
                call_id=call_id,
                seq=seq,
                ts=ts,
                entity_type=entity.entity_type,
                value=entity.value,
                source=entity.source,
                turn_seq=entity.turn_seq,
                confidence=entity.confidence,
            )

        await self._recorder.emit(build)

    async def _emit_risk(self, analysis: TurnAnalysis) -> None:
        tactic = self._pending_hook.name if self._pending_hook else None

        def build(call_id: str, seq: int, ts: datetime) -> RiskEvent:
            return RiskEvent(
                call_id=call_id,
                seq=seq,
                ts=ts,
                hangup_risk=analysis.hangup_risk,
                signals=analysis.risk_signals,
                tactic=tactic,
            )

        await self._recorder.emit(build)

    def _arm_hook(self, analysis: TurnAnalysis) -> None:
        """Pick the tactic for the next reply.

        Armed here and consumed by the next reply rather than applied retroactively: the
        turn whose analysis triggered the hook has already been spoken.

        Two different triggers, and the stage one is checked first. A delay hook is armed by
        hang-up risk, and it is recoverable - if the moment passes, a forgotten bank account
        is just as good four turns later. An elicitation hook is armed by *reaching the
        payment stage* regardless of risk, because it is not recoverable: the only window in
        which a scammer will read out a destination account is while he is still expecting
        money to arrive in it. A high risk score at the payment stage is a reason to elicit
        sooner rather than to stall - he stays for the same reason he called.
        """
        if self._pending_hook is not None:
            return
        if analysis.stage is ScamStage.PAYMENT and (
            hook := choose_hook(analysis.stage, self._used_hooks, hooks=PAYMENT_ELICITATION)
        ):
            self._arm(hook, analysis, trigger="payment_stage")
            return
        if not analysis.needs_hook:
            return
        hook = choose_hook(analysis.stage, self._used_hooks)
        if hook is None:
            logger.info("hang-up risk is high but every hook is spent")
            return
        self._arm(hook, analysis, trigger="hangup_risk")

    def _arm(self, hook: BaitHook, analysis: TurnAnalysis, *, trigger: str) -> None:
        self._pending_hook = hook
        self._pending_trigger = trigger
        logger.info(
            "arming a bait hook",
            extra={
                "call_id": self._config.call_id,
                "hook": hook.name,
                "trigger": trigger,
                "stage": analysis.stage.value,
                "hangup_risk": analysis.hangup_risk,
                "signals": list(analysis.risk_signals),
            },
        )

    async def _check_canary(self, turn: Turn) -> None:
        """Look for a canary issued to an earlier call in what the caller just said.

        A match means two calls are working from the same lead list, which is the strongest
        claim this system makes, so it is logged at warning level and persisted even though
        the dashboard already has the event.
        """
        matches = self._canary.check(turn.text)
        if not matches:
            return

        for match in matches:

            def build(
                call_id: str, seq: int, ts: datetime, found: CanaryMatch = match
            ) -> CanaryMatchEvent:
                return CanaryMatchEvent(
                    call_id=call_id,
                    seq=seq,
                    ts=ts,
                    identity_id=found.identity_id,
                    issued_call_id=found.issued_call_id,
                    matched_field=found.matched_field,
                    matched_text=found.matched_text,
                    score=found.score,
                )

            await self._recorder.emit(build)

        logger.warning(
            "canary identity from an earlier call appeared in this one",
            extra={
                "call_id": self._config.call_id,
                "matched_fields": [m.matched_field for m in matches],
                "issued_call_ids": [m.issued_call_id for m in matches],
            },
        )
        self._track(asyncio.create_task(self._canary.record(matches)))

    def _track(self, task: asyncio.Task[None]) -> None:
        """Keep a reference to a background task; asyncio holds only a weak one."""
        self._analyses.add(task)
        task.add_done_callback(self._analyses.discard)

    async def _drain_analyses(self) -> None:
        """Let outstanding analysis tasks finish so their events reach the evidence pack."""
        if not self._analyses:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*tuple(self._analyses), return_exceptions=True),
                timeout=ANALYSIS_DRAIN_SECONDS,
            )
        except TimeoutError:
            logger.warning("timed out draining analysis", extra={"call_id": self._config.call_id})

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
