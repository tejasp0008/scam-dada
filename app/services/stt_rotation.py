"""One continuous transcription session built from several short upstream ones.

Prisma closes a streaming session at 15 minutes with websocket code 1001. The whole point
of this project is to keep a scammer talking for longer than that, so the cap cannot be
allowed to end a call — but it also cannot be allowed to leak upwards, because an
orchestrator that has to know about it would need reconnection logic tangled through its
turn-taking.

So this wrapper implements `SttSession` in terms of a sequence of real sessions. It rotates
*before* the cap rather than reacting to the close, because reacting means the frames in
flight at the moment of closure are lost, and those frames are the middle of a sentence.

Rotation is deferred while the caller is speaking and forced at a hard deadline, so a
scammer who monologues for fourteen minutes still gets a seam — just one placed as well as
the circumstances allow.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from app.providers.base import (
    ProviderError,
    SttConfig,
    SttEvent,
    STTProvider,
    SttReady,
    SttSession,
    SttStreamError,
)

logger = logging.getLogger(__name__)

#: Prisma's documented cap is 15 minutes. Rotating at 13 leaves room for a deferral while
#: the caller finishes a sentence, plus the round trip to open the replacement.
ROTATE_AFTER_SECONDS = 13 * 60.0

#: How long a deferral may hold rotation open past the soft deadline before it is forced.
MAX_DEFERRAL_SECONDS = 90.0

#: Events are buffered between the upstream session and the orchestrator. Deep enough to
#: absorb a rotation, shallow enough that a stalled consumer is noticed rather than hidden.
QUEUE_SIZE = 256


class RotatingSttSession:
    """A `SttSession` that survives the provider's session cap."""

    def __init__(
        self,
        provider: STTProvider,
        config: SttConfig,
        session: SttSession,
        *,
        rotate_after_seconds: float = ROTATE_AFTER_SECONDS,
        max_deferral_seconds: float = MAX_DEFERRAL_SECONDS,
    ) -> None:
        self._provider = provider
        self._config = config
        self._session = session
        self._rotate_after = rotate_after_seconds
        self._max_deferral = max_deferral_seconds
        self._loop = asyncio.get_running_loop()
        self._opened_at = self._loop.time()
        self._queue: asyncio.Queue[SttEvent | None] = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._pump: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._ready_seen = False
        self._rotating = False
        self.rotations = 0
        #: Set by the orchestrator while the caller is mid-utterance. A seam inside a
        #: sentence costs a transcript; a seam in a silence costs nothing.
        self.hold_rotation = False

    @classmethod
    async def open(
        cls,
        provider: STTProvider,
        config: SttConfig,
        *,
        rotate_after_seconds: float = ROTATE_AFTER_SECONDS,
    ) -> RotatingSttSession:
        """Open the first upstream session and start pumping its events."""
        session = await provider.open_session(config)
        wrapper = cls(provider, config, session, rotate_after_seconds=rotate_after_seconds)
        wrapper._start_pump()
        return wrapper

    @property
    def frame_bytes(self) -> int:
        return self._session.frame_bytes

    @property
    def age_seconds(self) -> float:
        """How long the current upstream session has been open."""
        return self._loop.time() - self._opened_at

    def _start_pump(self) -> None:
        self._pump = asyncio.create_task(self._drain(self._session))

    async def _drain(self, session: SttSession) -> None:
        """Forward one upstream session's events into the queue until it ends.

        What happens when the upstream ends is the whole point of this class, so the three
        cases are spelled out rather than collapsed:

        * **We closed it.** Pass the sentinel on; the call is over.
        * **It ended at or after the soft deadline.** That is the provider's cap. Replace it
          here, immediately, rather than waiting for the next frame - a caller who happens
          to be pausing at the fifteen-minute mark would otherwise end the call.
        * **It ended early.** Something is actually wrong. Surface it as fatal, and do *not*
          rotate: a provider handing out sessions that die on arrival would spin forever.
        """
        reason: str | None = None
        try:
            async for event in session.events():
                if isinstance(event, SttReady):
                    # The orchestrator is given exactly one `ready`, for the first session.
                    # A second one would look like the call restarting.
                    if self._ready_seen:
                        logger.info("stt session replaced", extra={"rotations": self.rotations})
                        continue
                    self._ready_seen = True
                await self._queue.put(event)
        except ProviderError as exc:
            reason = f"{type(exc).__name__}: {exc}"
        except asyncio.CancelledError:
            raise

        if self._closed:
            await self._queue.put(None)
            return
        if self.age_seconds < self._rotate_after:
            await self._queue.put(
                SttStreamError(message=reason or "session ended early", at_ms=0, fatal=True)
            )
            await self._queue.put(None)
            return
        logger.info(
            "upstream stt session ended at the cap, replacing it",
            extra={"reason": reason, "age_seconds": round(self.age_seconds, 1)},
        )
        # Scheduled rather than awaited: this coroutine *is* the pump task that `_rotate`
        # cancels, and a task cannot tidily cancel itself part-way through.
        asyncio.get_running_loop().create_task(self._rotate())

    async def _rotate(self) -> None:
        """Replace the upstream session, opening the new one before closing the old."""
        async with self._lock:
            if self._closed or self._rotating:
                return
            self._rotating = True
            previous = self._session
            previous_pump = self._pump
            try:
                replacement = await self._provider.open_session(self._config)
            except ProviderError as exc:
                # Keep the old session: it is within its cap for a little longer, and a
                # degraded session beats no session. The next attempt will try again.
                logger.warning(
                    "stt rotation failed, keeping the current session", extra={"error": str(exc)}
                )
                self._opened_at = self._loop.time() - self._rotate_after + 15.0
                self._rotating = False
                return

            self._session = replacement
            self._opened_at = self._loop.time()
            self.rotations += 1
            self._start_pump()
            self._rotating = False
            logger.info("stt session rotated", extra={"rotations": self.rotations})

        if previous_pump is not None:
            previous_pump.cancel()
        try:
            await previous.aclose()
        except ProviderError as exc:  # the replacement is already live; this is cleanup
            logger.info("closing the replaced stt session failed", extra={"error": str(exc)})

    async def _rotate_if_due(self) -> None:
        age = self.age_seconds
        if age < self._rotate_after:
            return
        if self.hold_rotation and age < self._rotate_after + self._max_deferral:
            return
        await self._rotate()

    async def send_audio(self, frame: bytes) -> None:
        """Send one frame, rotating first if the session is near its cap."""
        await self._rotate_if_due()
        await self._session.send_audio(frame)

    async def commit(self) -> None:
        await self._session.commit()

    async def events(self) -> AsyncIterator[SttEvent]:
        """One continuous event stream across every upstream session."""
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def aclose(self) -> None:
        self._closed = True
        if self._pump is not None:
            self._pump.cancel()
        await self._session.aclose()
        await self._queue.put(None)
