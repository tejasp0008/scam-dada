"""Getting a persona's words out of the TTS provider and onto the line.

Three jobs the orchestrator should not have to hold at once: synthesising, rendering
through the call profile, and being interrupted.

**Why the whole sentence is synthesised before any of it is sent.** The WSOLA time-stretch
in the ageing chain needs lookahead, so it cannot run on a 200 ms chunk in isolation. That
costs latency — first audio out is the full synthesis plus the render, not the provider's
first byte — and the mitigation is upstream: `take_sentence` keeps each request to one
sentence, so "the whole thing" is half a second of audio rather than ten seconds of it. The
measured render cost is about 1% of the audio's duration. A chunk-wise stretch with one
frame of lookahead is the fix if this ever dominates; see DECISIONS 0019.

**Why the bed offset is held here.** Room tone has to be continuous across utterances, so
one monotonic offset advances with everything ever sent. A bed restarted per utterance makes
the fan stop and start exactly when the persona does, which points at the machine.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import aclosing

import numpy as np

from app.core.audio.ambience import AmbienceBed
from app.core.audio.call_profile import CallProfile
from app.core.audio.framing import float32_to_pcm, pcm_to_float32
from app.providers.base import ProviderError, TTSProvider, TtsRequest

logger = logging.getLogger(__name__)

#: Outgoing audio is handed to the transport in slices this long, so a barge-in is acted on
#: within one slice rather than at the end of a sentence.
CHUNK_MS = 60


class PersonaSpeaker:
    """Synthesises, renders and plays one persona utterance at a time."""

    def __init__(
        self,
        tts: TTSProvider,
        sink: Callable[[bytes], Awaitable[None]],
        *,
        flush: Callable[[], Awaitable[None]] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._tts = tts
        self._sink = sink
        self._flush = flush
        self._clock = clock or asyncio.get_event_loop().time
        self._current: asyncio.Task[float | None] | None = None
        self._interrupted = False
        self._beds: dict[str, AmbienceBed] = {}
        self._bed_offset = 0
        self.interruptions = 0
        #: Seconds of audio the last completed utterance queued. The caller needs this to
        #: know how long the transport will be busy, because `speak` returns as soon as the
        #: bytes are handed over - the browser's ring buffer does the actual pacing.
        self.last_audio_seconds = 0.0

    @property
    def speaking(self) -> bool:
        """Whether an utterance is in flight."""
        return self._current is not None and not self._current.done()

    @property
    def bed_offset(self) -> int:
        """Samples of room tone consumed so far. Continuous for the whole call."""
        return self._bed_offset

    def _bed_for(self, profile: CallProfile, sample_rate: int) -> AmbienceBed | None:
        if profile.ambience is None:
            return None
        key = f"{profile.ambience.name}:{sample_rate}"
        if key not in self._beds:
            # Built once per call: generating a 12-second loop costs 7-15 ms, which is
            # nothing once and real money per utterance.
            self._beds[key] = AmbienceBed(profile.ambience, sample_rate)
        return self._beds[key]

    async def speak(self, request: TtsRequest, *, profile: CallProfile) -> float | None:
        """Say one utterance. Returns seconds to first audio out, or None if interrupted."""
        await self.stop()
        self._interrupted = False
        task = asyncio.create_task(self._play(request, profile))
        self._current = task
        try:
            return await task
        except asyncio.CancelledError:
            # Only swallow the cancellation that `stop` caused. An outer cancellation - the
            # call ending, the transport dropping - must keep propagating.
            if task.cancelled() and self._interrupted:
                return None
            raise

    async def _play(self, request: TtsRequest, profile: CallProfile) -> float | None:
        started = self._clock()
        pcm = bytearray()
        # `aclosing` rather than a bare `async for`: on a barge-in this task is cancelled
        # mid-stream, and the generator's cleanup has to run *now* rather than whenever the
        # garbage collector gets to it. Timbre allows one active request per key, so a stream
        # left dangling makes the persona's next sentence fail with a 429 - which is exactly
        # what happened on a real call before this was here.
        async with aclosing(self._tts.synthesize(request)) as stream:
            async for chunk in stream:
                pcm.extend(chunk.pcm)
        if not pcm:
            raise ProviderError("synthesis produced no audio")

        voice = pcm_to_float32(bytes(pcm))
        result = await asyncio.to_thread(
            profile.render,
            voice,
            request.sample_rate,
            bed=self._bed_for(profile, request.sample_rate),
            bed_offset=self._bed_offset,
        )
        self._bed_offset += result.audio.size

        out = float32_to_pcm(result.audio)
        self.last_audio_seconds = result.seconds
        slice_bytes = max(2, int(CHUNK_MS / 1000.0 * request.sample_rate) * 2)
        first_at: float | None = None
        for start in range(0, len(out), slice_bytes):
            await self._sink(out[start : start + slice_bytes])
            if first_at is None:
                first_at = self._clock()
        logger.info(
            "persona spoke",
            extra={
                "call_id": request.call_id,
                "chars": len(request.text),
                "audio_ms": int(result.seconds * 1000),
                "render_ms": int(result.total_ms),
                "dropped_packets": result.dropped_packets,
            },
        )
        return None if first_at is None else first_at - started

    async def stop(self) -> None:
        """Interrupt playback and tell the transport to drop what it has queued.

        Timbre documents no cancel on either streaming transport, so an interrupted
        synthesis is abandoned client-side; the audio already generated has been billed
        whatever we do with it.
        """
        task = self._current
        self._current = None
        if task is None or task.done():
            return
        self._interrupted = True
        self.interruptions += 1
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if self._flush is not None:
            await self._flush()

    def advance_room_tone(self, samples: int) -> None:
        """Account for room tone emitted by something other than an utterance.

        The comfort-noise loop sends bed between utterances; without this the offset would
        rewind every time the persona spoke and the fan would audibly jump.
        """
        if samples < 0:
            raise ValueError("samples must be non-negative")
        self._bed_offset += samples

    def room_tone(self, profile: CallProfile, sample_rate: int, samples: int) -> bytes:
        """Bed-only audio for the gap between utterances, advancing the shared offset.

        A line that goes digitally silent between turns is the single loudest tell that
        nobody is holding the handset. Real rooms do not stop.
        """
        bed = self._bed_for(profile, sample_rate)
        if bed is None or samples <= 0:
            return b""
        tone = bed.take(samples, self._bed_offset) * _comfort_gain(profile)
        self._bed_offset += samples
        return float32_to_pcm(tone.astype(np.float32))


def _comfort_gain(profile: CallProfile) -> float:
    """Absolute scale for bed-only audio, matched to the bed inside a spoken utterance.

    Inside an utterance the bed is set `level_db` under the speech and the whole mix is then
    peak-normalised to `output_peak`, so the bed's absolute level depends on the voice's
    crest factor. Rather than reproduce that chain, this uses a nominal speech peak-to-RMS
    of 4 (12 dB), which is typical of band-limited speech, and a test asserts the result
    lands within 4 dB of the bed measured in a real rendered utterance.
    """
    if profile.ambience is None:
        return 0.0
    nominal_speech_rms = profile.output_peak / 4.0
    gain: float = nominal_speech_rms * (10.0 ** (profile.ambience.level_db / 20.0))
    return gain
