"""Provider ports: the interfaces every speech/LLM vendor is adapted to.

Nothing outside this package may name a vendor. Adding Sarvam or BharatGen later means a
new module here plus a config value, with no change anywhere else.

Two shapes here are driven by what Gnani's API actually does, not by preference:

* `SttEvent` is a union rather than a bare transcript stream. Prisma streaming returns
  `is_final=true` on every transcript (it has no partial hypotheses) and signals turn
  boundaries through separate `speech_start` / `processing` events. Collapsing those into a
  transcript-only iterator would throw away the only endpointing signal we get.
* `AudioChunk` carries raw PCM, not encoded audio. Timbre can return WAV, MP3 or OGG, but
  the browser needs to be able to drop queued audio the instant the scammer interrupts, and
  that is only cheap with unframed PCM in a ring buffer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from enum import StrEnum
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Base class for every provider failure, so callers never see vendor exceptions."""

    retryable: bool = False


class AuthError(ProviderError):
    """Missing or rejected credentials (HTTP 401/403)."""


class OutOfCredit(ProviderError):
    """The account has no credit left (HTTP 402). Never retried."""


class RateLimited(ProviderError):
    """HTTP 429. Retryable with backoff."""

    retryable = True


class UpstreamTimeout(ProviderError):
    """HTTP 504 or a client-side timeout. Retryable with backoff."""

    retryable = True


class ProviderUnavailable(ProviderError):
    """HTTP 5xx or a connection failure. Retryable with backoff."""

    retryable = True


class BadRequest(ProviderError):
    """HTTP 4xx other than auth/credit/rate. Not retryable: it fails the same way."""


class TranscriptionFailed(ProviderError):
    """A single utterance could not be transcribed; the session survives."""


class SynthesisFailed(ProviderError):
    """A single synthesis request failed; the connection may survive."""


class BudgetExceeded(ProviderError):
    """Raised by the usage recorder before a call goes out, not by the vendor."""


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------


class UnitType(StrEnum):
    """Billable units we can actually count.

    Gnani publishes no credit conversion, so the budget guard caps raw units rather than
    pretending to know the rupee or credit cost.
    """

    TTS_CHARS = "tts_chars"
    STT_SECONDS = "stt_seconds"
    LLM_TOKENS = "llm_tokens"


class UsageRecorder(Protocol):
    """Port implemented by the budget guard in `app.infra`.

    Declared here so providers depend on an interface they own rather than on infra.
    """

    async def check(self, provider: str, unit: UnitType, units: int) -> None:
        """Raise `BudgetExceeded` if spending `units` would breach the configured ceiling."""
        ...

    async def record(
        self,
        provider: str,
        unit: UnitType,
        units: int,
        *,
        call_id: str | None = None,
        cache_hit: bool = False,
    ) -> None:
        """Persist actual consumption. `cache_hit` rows cost nothing and prove savings."""
        ...


# ---------------------------------------------------------------------------
# Speech to text
# ---------------------------------------------------------------------------


class SttConfig(BaseModel):
    """Per-session STT settings.

    Field names are ours; each adapter maps them onto its vendor's wire format.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    language_code: str = Field(
        description="A vendor language code, a comma-separated list, or an auto value."
    )
    sample_rate: int = 16000
    output_format: str | None = Field(
        default=None, description="Vendor transcript style, e.g. verbatim vs transcribe."
    )
    keyterms: Sequence[str] = Field(
        default=(), description="Domain vocabulary to bias recognition towards."
    )
    idle_timeout_sec: int | None = Field(
        default=None,
        ge=1,
        description="How long a silent session is held open. Scammers pause to think.",
    )
    min_silence_ms: int | None = Field(default=None, ge=1)
    vad_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class SttEventKind(StrEnum):
    READY = "ready"
    SPEECH_START = "speech_start"
    PROCESSING = "processing"
    TRANSCRIPT = "transcript"
    ERROR = "error"


class SttReady(BaseModel):
    """The session is open and accepting audio."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[SttEventKind.READY] = SttEventKind.READY
    sample_rate: int
    frame_bytes: int
    request_id: str | None = None


class SttSpeechStart(BaseModel):
    """The provider's own VAD detected speech. Confirms, but does not drive, barge-in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[SttEventKind.SPEECH_START] = SttEventKind.SPEECH_START
    at_ms: int = Field(ge=0, description="Monotonic ms since the session opened.")
    utterance_id: str | None = None


class SttProcessing(BaseModel):
    """Speech ended; a transcript is being produced. The cue to start a filler phrase."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[SttEventKind.PROCESSING] = SttEventKind.PROCESSING
    at_ms: int = Field(ge=0)
    utterance_id: str | None = None


class SttTranscript(BaseModel):
    """A transcript segment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[SttEventKind.TRANSCRIPT] = SttEventKind.TRANSCRIPT
    text: str
    is_final: bool
    at_ms: int = Field(ge=0)
    segment_id: str | None = None
    segment_index: int | None = None
    utterance_id: str | None = None
    detected_language_code: str | None = None
    audio_duration_ms: int | None = Field(default=None, ge=0)
    provider_latency_ms: int | None = Field(default=None, ge=0)


class SttStreamError(BaseModel):
    """A recoverable error. The session stays open unless `fatal` is set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[SttEventKind.ERROR] = SttEventKind.ERROR
    message: str
    at_ms: int = Field(ge=0)
    fatal: bool = False


SttEvent = SttReady | SttSpeechStart | SttProcessing | SttTranscript | SttStreamError


class SttSession(Protocol):
    """A live transcription session.

    Audio is pushed in and events are pulled out, rather than the provider consuming an
    iterator, because the orchestrator must keep feeding frames while it is also reacting
    to events, and because barge-in needs to inspect frames on the way past.
    """

    @property
    def frame_bytes(self) -> int:
        """Exact byte length the provider requires for each audio frame."""
        ...

    async def send_audio(self, frame: bytes) -> None:
        """Send one PCM frame of exactly `frame_bytes` bytes."""
        ...

    async def commit(self) -> None:
        """Close the open utterance immediately instead of waiting for the VAD."""
        ...

    def events(self) -> AsyncIterator[SttEvent]:
        """Yield session events until the session closes."""
        ...

    async def aclose(self) -> None:
        """Close the session and release the connection."""
        ...


class STTProvider(Protocol):
    """Opens transcription sessions."""

    name: str

    async def open_session(self, config: SttConfig) -> SttSession:
        """Connect and return a session ready for audio."""
        ...


# ---------------------------------------------------------------------------
# Text to speech
# ---------------------------------------------------------------------------


class TtsRequest(BaseModel):
    """One synthesis request.

    `text` may contain the provider's inline expression tags. Tag characters are billable,
    so the budget guard counts the string as sent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    voice: str
    language: str
    model: str
    speed: float = Field(default=1.0, gt=0.0)
    sample_rate: int = 24000


class AudioChunk(BaseModel):
    """A slice of synthesised audio as raw PCM s16le mono."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    pcm: bytes
    is_final: bool


class TTSProvider(Protocol):
    """Streams synthesised speech."""

    name: str

    def synthesize(self, request: TtsRequest) -> AsyncIterator[AudioChunk]:
        """Yield PCM chunks as they are produced. First chunk latency is what matters."""
        ...


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    content: str


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMProvider(Protocol):
    """Dialogue generation and structured extraction.

    The two methods exist separately because the dialogue path must stream (the persona has
    to start speaking before the reply is finished) while the analysis path must not emit
    anything until it validates.
    """

    name: str

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.8,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield reply text incrementally."""
        ...

    async def complete_json(
        self,
        messages: Sequence[ChatMessage],
        schema: type[SchemaT],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> SchemaT:
        """Return a validated instance of `schema`, raising `BadRequest` if it never validates."""
        ...
