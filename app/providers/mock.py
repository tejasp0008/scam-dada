"""Deterministic offline implementations of all three ports.

These exist so the orchestrator, the dashboard and the whole integration suite can be
developed and tested without a network call or a single credit spent. They are deliberately
boring: no randomness, no wall-clock dependence beyond explicit delays, so an asserted
event sequence stays asserted.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from pydantic import ValidationError

from app.core.audio.framing import PRISMA_FRAME_BYTES
from app.providers.base import (
    AudioChunk,
    BadRequest,
    ChatMessage,
    SchemaT,
    SttConfig,
    SttEvent,
    SttProcessing,
    SttReady,
    SttSpeechStart,
    SttTranscript,
    TtsRequest,
)

#: A synthetic scammer script. Obviously fictional: a fake officer, a reserved-range
#: number and a @demo UPI handle, per the project's synthetic-data-only rule.
DEFAULT_SCAMMER_SCRIPT: tuple[str, ...] = (
    "Hello madam, main Mumbai police cyber cell se bol raha hoon.",
    "Aapke naam par ek parcel aaya hai jisme illegal items hain.",
    "Case number FAKE-CYB-0001 register ho gaya hai, abhi verify karna padega.",
    "Aap kisi ko bataiye mat, warrant nikal jayega.",
    "Verification ke liye fake.officer@demo par chhota amount bhejiye.",
)


class MockSttSession:
    """Emits scripted utterances in response to audio.

    An utterance fires either after `frames_per_utterance` frames have been pushed or on an
    explicit `commit()`, which makes turn boundaries fully controllable from a test.
    """

    def __init__(
        self,
        script: Sequence[str],
        *,
        frames_per_utterance: int = 10,
        language_code: str = "hi-IN",
        sample_rate: int = 16000,
    ) -> None:
        self._script = list(script)
        self._frames_per_utterance = frames_per_utterance
        self._language_code = language_code
        self._queue: asyncio.Queue[SttEvent | None] = asyncio.Queue()
        self._frames_seen = 0
        self._utterance_index = 0
        self._elapsed_ms = 0
        self._closed = False
        self._queue.put_nowait(
            SttReady(sample_rate=sample_rate, frame_bytes=PRISMA_FRAME_BYTES, request_id="mock-1")
        )

    @property
    def frame_bytes(self) -> int:
        return PRISMA_FRAME_BYTES

    async def send_audio(self, frame: bytes) -> None:
        if self._closed:
            raise BadRequest("session is closed")
        if len(frame) != PRISMA_FRAME_BYTES:
            raise BadRequest(f"frame must be exactly {PRISMA_FRAME_BYTES} bytes, got {len(frame)}")
        self._frames_seen += 1
        # 512 samples at 16 kHz is 32 ms of audio.
        self._elapsed_ms += 32
        if self._frames_seen == 1:
            await self._queue.put(self._speech_start())
        if self._frames_seen % self._frames_per_utterance == 0:
            await self._emit_utterance()

    async def commit(self) -> None:
        await self._emit_utterance()

    async def _emit_utterance(self) -> None:
        if self._utterance_index >= len(self._script):
            return
        text = self._script[self._utterance_index]
        self._utterance_index += 1
        self._frames_seen = 0
        await self._queue.put(SttProcessing(at_ms=self._elapsed_ms, utterance_id=self._uid()))
        await self._queue.put(
            SttTranscript(
                text=text,
                is_final=True,
                at_ms=self._elapsed_ms,
                segment_id=f"mock-seg-{self._utterance_index}",
                segment_index=self._utterance_index,
                utterance_id=self._uid(),
                detected_language_code=self._language_code,
                audio_duration_ms=self._frames_per_utterance * 32,
                provider_latency_ms=120,
            )
        )
        if self._utterance_index < len(self._script):
            await self._queue.put(self._speech_start())

    def _speech_start(self) -> SttSpeechStart:
        return SttSpeechStart(at_ms=self._elapsed_ms, utterance_id=self._uid())

    def _uid(self) -> str:
        return f"mock-utt-{self._utterance_index}"

    async def events(self) -> AsyncIterator[SttEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def aclose(self) -> None:
        self._closed = True
        await self._queue.put(None)


class MockSTTProvider:
    """Hands out `MockSttSession`s driven by a fixed script."""

    name = "mock-stt"

    def __init__(
        self,
        script: Sequence[str] = DEFAULT_SCAMMER_SCRIPT,
        *,
        frames_per_utterance: int = 10,
    ) -> None:
        self._script = script
        self._frames_per_utterance = frames_per_utterance

    async def open_session(self, config: SttConfig) -> MockSttSession:
        return MockSttSession(
            self._script,
            frames_per_utterance=self._frames_per_utterance,
            language_code=config.language_code,
            sample_rate=config.sample_rate,
        )


class MockTTSProvider:
    """Returns silence whose length is proportional to the text.

    Proportional length matters: barge-in and the minutes-wasted counter are both timing
    logic, and they need audio that takes a plausible amount of time to play.
    """

    name = "mock-tts"

    def __init__(self, *, ms_per_char: int = 60, chunk_ms: int = 200, delay_s: float = 0.0) -> None:
        self._ms_per_char = ms_per_char
        self._chunk_ms = chunk_ms
        self._delay_s = delay_s
        self.requests: list[TtsRequest] = []

    async def synthesize(self, request: TtsRequest) -> AsyncIterator[AudioChunk]:
        self.requests.append(request)
        total_ms = max(self._chunk_ms, len(request.text) * self._ms_per_char)
        chunk_count = max(1, total_ms // self._chunk_ms)
        bytes_per_chunk = int(request.sample_rate * (self._chunk_ms / 1000.0)) * 2
        for index in range(chunk_count):
            if self._delay_s:
                await asyncio.sleep(self._delay_s)
            yield AudioChunk(
                index=index,
                pcm=b"\x00" * bytes_per_chunk,
                is_final=index == chunk_count - 1,
            )


#: Canned in-character replies. Hinglish, non-threatening, and stalling by design.
DEFAULT_PERSONA_REPLIES: tuple[str, ...] = (
    "Haan beta, kaun bol raha hai? <pauses> Thoda zor se boliye.",
    "Arre police? <sigh> Mera chashma kahan rakha, ek minute.",
    "Case number phir se boliye beta, main likh rahi hoon dheere dheere.",
    "Kisi ko nahi bataungi. <whispers> Par bahu ghar par hai.",
    "Paisa? Mujhe phone chalana nahi aata beta, aap samjhaiye.",
)


class MockLLMProvider:
    """Cycles canned replies, and validates queued JSON payloads for `complete_json`."""

    name = "mock-llm"

    def __init__(
        self,
        replies: Sequence[str] = DEFAULT_PERSONA_REPLIES,
        *,
        token_delay_s: float = 0.0,
    ) -> None:
        self._replies = list(replies)
        self._token_delay_s = token_delay_s
        self._reply_index = 0
        self._json_queue: list[dict[str, object]] = []
        self.chat_calls: list[list[ChatMessage]] = []

    def queue_json(self, payload: dict[str, object]) -> None:
        """Make `complete_json` return this payload on its next call."""
        self._json_queue.append(payload)

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.8,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        self.chat_calls.append(list(messages))
        reply = self._replies[self._reply_index % len(self._replies)]
        self._reply_index += 1
        # Word-at-a-time so sentence-boundary chunking has something to chunk.
        for word in reply.split(" "):
            if self._token_delay_s:
                await asyncio.sleep(self._token_delay_s)
            yield word + " "

    async def complete_json(
        self,
        messages: Sequence[ChatMessage],
        schema: type[SchemaT],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> SchemaT:
        if self._json_queue:
            return schema.model_validate(self._json_queue.pop(0))
        try:
            # Schemas whose fields are all defaulted validate from empty, which is the
            # degraded-analysis case the orchestrator must tolerate anyway.
            return schema.model_validate({})
        except ValidationError as exc:
            raise BadRequest(
                f"mock LLM has no queued payload for {schema.__name__}; call queue_json first"
            ) from exc


def mock_schema_instance(schema: type[SchemaT]) -> SchemaT:
    """Build a default instance of `schema`, for fixtures that need one."""
    return schema.model_validate({})


__all__ = [
    "DEFAULT_PERSONA_REPLIES",
    "DEFAULT_SCAMMER_SCRIPT",
    "MockLLMProvider",
    "MockSTTProvider",
    "MockSttSession",
    "MockTTSProvider",
    "mock_schema_instance",
]
