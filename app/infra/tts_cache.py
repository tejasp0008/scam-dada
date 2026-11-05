"""Content-addressed cache in front of any `TTSProvider`.

Synthesis is the dominant credit cost and the persona repeats itself constantly: filler
phrases, greetings, retention hooks and "ek minute beta" come up in every call. Anything
synthesised once is never paid for twice.

Implemented as a decorator rather than inside an adapter so it applies to every vendor, and
so a cache hit never reaches the network at all.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Final

from app.providers.base import AudioChunk, TTSProvider, TtsRequest, UsageRecorder

logger = logging.getLogger(__name__)

#: Chunk size when replaying from cache. 100 ms at 24 kHz keeps the first-byte latency of a
#: cache hit indistinguishable from a live stream, which matters because fillers are played
#: precisely to cover latency.
REPLAY_CHUNK_MS: Final[int] = 100


def cache_key(request: TtsRequest) -> str:
    """A stable hash of everything that changes the audio.

    `call_id` is deliberately excluded: it does not affect the waveform, and including it
    would give every call a cold cache.
    """
    payload = json.dumps(
        {
            "text": request.text,
            "voice": request.voice,
            "language": request.language,
            "model": request.model,
            "speed": request.speed,
            "sample_rate": request.sample_rate,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachingTTS:
    """Wraps a `TTSProvider`, serving repeated text from disk."""

    def __init__(
        self,
        inner: TTSProvider,
        cache_dir: Path,
        *,
        usage: UsageRecorder | None = None,
        enabled: bool = True,
    ) -> None:
        self._inner = inner
        self._dir = cache_dir
        self._usage = usage
        self._enabled = enabled
        self.hits = 0
        self.misses = 0
        if enabled:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        """Report the wrapped provider's name: callers care about the vendor, not the cache."""
        return self._inner.name

    def path_for(self, request: TtsRequest) -> Path:
        """Where this request's audio is or would be stored."""
        return self._dir / f"{cache_key(request)}.pcm"

    async def synthesize(self, request: TtsRequest) -> AsyncGenerator[AudioChunk, None]:
        """Yield PCM from disk if we have it, otherwise from the provider, storing as we go."""
        if not self._enabled:
            async for chunk in self._inner.synthesize(request):
                yield chunk
            return

        path = self.path_for(request)
        if path.exists():
            self.hits += 1
            pcm = await asyncio.to_thread(path.read_bytes)
            await self._record_hit(request)
            logger.info(
                "tts cache hit",
                extra={"chars": len(request.text), "voice": request.voice, "bytes": len(pcm)},
            )
            for chunk in _slice(pcm, request.sample_rate):
                yield chunk
            return

        self.misses += 1
        collected = bytearray()
        completed = False
        try:
            async for chunk in self._inner.synthesize(request):
                collected.extend(chunk.pcm)
                yield chunk
            completed = True
        finally:
            # Only a stream that ran to completion is cacheable. Storing a partial result
            # would make a barge-in truncate that phrase forever.
            if completed and collected:
                await asyncio.to_thread(_write_atomic, path, bytes(collected))

    async def _record_hit(self, request: TtsRequest) -> None:
        """Log the hit at zero cost, with what it would have cost in `avoided_credits`."""
        if self._usage is None:
            return
        charge = self._usage.cost.price_tts(len(request.text))
        await self._usage.record(self.name, charge, call_id=request.call_id, cache_hit=True)


def _write_atomic(path: Path, data: bytes) -> None:
    """Write via a temporary file so a crash cannot leave a truncated cache entry."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _slice(pcm: bytes, sample_rate: int) -> list[AudioChunk]:
    """Cut cached PCM into chunks the size a live stream would have produced."""
    step = max(2, int(sample_rate * REPLAY_CHUNK_MS / 1000) * 2)
    chunks: list[AudioChunk] = []
    for index, start in enumerate(range(0, len(pcm), step)):
        piece = pcm[start : start + step]
        chunks.append(AudioChunk(index=index, pcm=piece, is_final=start + step >= len(pcm)))
    return chunks
