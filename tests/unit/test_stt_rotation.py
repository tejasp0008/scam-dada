"""Session rotation. Prisma closes at 15 minutes; a scambait is supposed to beat that."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from app.providers.base import (
    ProviderUnavailable,
    SttConfig,
    SttEvent,
    SttReady,
    SttStreamError,
    SttTranscript,
)
from app.services.stt_rotation import RotatingSttSession

CONFIG = SttConfig(model="prisma-v3.0", language_code="indic-auto", sample_rate=16000)


class FakeSession:
    """One upstream session. Ends its event stream when closed, or when told to die."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.frames: list[bytes] = []
        self.closed = False
        self._queue: asyncio.Queue[SttEvent | None] = asyncio.Queue()
        self._queue.put_nowait(SttReady(sample_rate=16000, frame_bytes=1024))

    @property
    def frame_bytes(self) -> int:
        return 1024

    async def send_audio(self, frame: bytes) -> None:
        if self.closed:
            raise ProviderUnavailable("session is closed")
        self.frames.append(frame)

    async def commit(self) -> None:
        return None

    async def push(self, event: SttEvent) -> None:
        await self._queue.put(event)

    async def die(self, message: str) -> None:
        """Simulate the upstream closing the connection."""
        self._dying = message
        await self._queue.put(None)

    _dying: str | None = None

    async def events(self) -> AsyncIterator[SttEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                if self._dying is not None:
                    raise ProviderUnavailable(self._dying)
                return
            yield event

    async def aclose(self) -> None:
        self.closed = True
        await self._queue.put(None)


class FakeProvider:
    """Hands out `FakeSession`s and counts how many were asked for."""

    name = "fake-stt"

    def __init__(self, *, fail_after: int | None = None) -> None:
        self.sessions: list[FakeSession] = []
        self._fail_after = fail_after

    async def open_session(self, config: SttConfig) -> FakeSession:
        if self._fail_after is not None and len(self.sessions) >= self._fail_after:
            raise ProviderUnavailable("no capacity")
        session = FakeSession(len(self.sessions))
        self.sessions.append(session)
        return session


async def collect(session: RotatingSttSession, count: int) -> list[SttEvent]:
    """Take exactly `count` events, so a test never waits on a stream that stays open."""
    events: list[SttEvent] = []
    stream = session.events()
    for _ in range(count):
        events.append(await asyncio.wait_for(anext(stream), timeout=2.0))
    return events


class TestPassThrough:
    async def test_it_forwards_audio_and_events(self) -> None:
        provider = FakeProvider()
        session = await RotatingSttSession.open(provider, CONFIG)
        await session.send_audio(b"\x00" * 1024)
        await provider.sessions[0].push(SttTranscript(text="haan", is_final=True, at_ms=10))

        events = await collect(session, 2)
        assert isinstance(events[0], SttReady)
        assert isinstance(events[1], SttTranscript)
        assert provider.sessions[0].frames == [b"\x00" * 1024]
        await session.aclose()

    async def test_frame_bytes_comes_from_the_live_session(self) -> None:
        session = await RotatingSttSession.open(FakeProvider(), CONFIG)
        assert session.frame_bytes == 1024
        await session.aclose()

    async def test_closing_ends_the_event_stream(self) -> None:
        session = await RotatingSttSession.open(FakeProvider(), CONFIG)
        await session.aclose()
        stream = session.events()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(stream), timeout=2.0)


class TestRotation:
    async def test_it_rotates_before_the_cap_and_keeps_accepting_audio(self) -> None:
        provider = FakeProvider()
        session = await RotatingSttSession.open(provider, CONFIG, rotate_after_seconds=0.0)
        await session.send_audio(b"\x01" * 1024)
        await asyncio.sleep(0)

        assert session.rotations == 1
        assert len(provider.sessions) == 2
        # The frame went to the replacement, not to the session that was about to close.
        assert provider.sessions[1].frames == [b"\x01" * 1024]
        assert provider.sessions[0].closed
        await session.aclose()

    async def test_the_orchestrator_sees_only_one_ready_event(self) -> None:
        """A second `ready` would look like the call restarting."""
        provider = FakeProvider()
        session = await RotatingSttSession.open(provider, CONFIG, rotate_after_seconds=0.0)
        await session.send_audio(b"\x01" * 1024)
        await provider.sessions[1].push(SttTranscript(text="after", is_final=True, at_ms=1))

        events = await collect(session, 2)
        assert sum(isinstance(e, SttReady) for e in events) == 1
        assert isinstance(events[1], SttTranscript)
        await session.aclose()

    async def test_a_held_rotation_is_deferred_while_the_caller_speaks(self) -> None:
        """A seam inside an utterance costs a transcript; one in a silence costs nothing."""
        provider = FakeProvider()
        session = await RotatingSttSession.open(provider, CONFIG, rotate_after_seconds=0.0)
        session.hold_rotation = True
        await session.send_audio(b"\x01" * 1024)
        assert session.rotations == 0

        session.hold_rotation = False
        await session.send_audio(b"\x02" * 1024)
        assert session.rotations == 1
        await session.aclose()

    async def test_a_hold_cannot_defer_past_the_hard_deadline(self) -> None:
        """Otherwise a fourteen-minute monologue would run into the provider's cap."""
        provider = FakeProvider()
        session = RotatingSttSession(
            provider,
            CONFIG,
            await FakeProvider().open_session(CONFIG),
            rotate_after_seconds=0.0,
            max_deferral_seconds=0.0,
        )
        session._start_pump()
        session.hold_rotation = True
        await session.send_audio(b"\x01" * 1024)
        assert session.rotations == 1
        await session.aclose()

    async def test_a_failed_rotation_keeps_the_current_session(self) -> None:
        """A session inside its cap for a little longer beats no session at all."""
        provider = FakeProvider(fail_after=1)
        session = await RotatingSttSession.open(provider, CONFIG, rotate_after_seconds=0.0)
        await session.send_audio(b"\x01" * 1024)

        assert session.rotations == 0
        assert not provider.sessions[0].closed
        assert provider.sessions[0].frames == [b"\x01" * 1024]
        await session.aclose()


class TestFailures:
    async def test_a_death_before_the_cap_is_surfaced_as_fatal(self) -> None:
        provider = FakeProvider()
        session = await RotatingSttSession.open(provider, CONFIG, rotate_after_seconds=600.0)
        await provider.sessions[0].die("connection reset")

        events = await collect(session, 2)
        error = events[1]
        assert isinstance(error, SttStreamError)
        assert error.fatal
        assert "connection reset" in error.message
        await session.aclose()

    async def test_a_death_at_the_cap_is_not_an_error(self) -> None:
        """Hitting the documented cap is the thing this class exists for, not a failure."""
        provider = FakeProvider()
        session = await RotatingSttSession.open(provider, CONFIG, rotate_after_seconds=0.0)
        await provider.sessions[0].die("1001 going away")
        await asyncio.sleep(0)

        stream = session.events()
        first = await asyncio.wait_for(anext(stream), timeout=2.0)
        assert isinstance(first, SttReady)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(anext(stream), timeout=0.05)
        await session.aclose()


class TestReactiveRotation:
    async def test_a_session_dying_at_the_cap_is_replaced_without_ending_the_call(self) -> None:
        """The cap must not reach the orchestrator. This is the whole purpose of the class."""
        provider = FakeProvider()
        session = await RotatingSttSession.open(provider, CONFIG, rotate_after_seconds=0.0)
        await provider.sessions[0].die("1001 going away")

        for _ in range(50):
            await asyncio.sleep(0.001)
            if session.rotations:
                break
        assert session.rotations == 1
        assert len(provider.sessions) == 2

        # And the replacement's events still reach the consumer on the same stream.
        await provider.sessions[1].push(SttTranscript(text="after", is_final=True, at_ms=1))
        events = await collect(session, 2)
        assert isinstance(events[1], SttTranscript)
        assert events[1].text == "after"
        await session.aclose()

    async def test_the_consumer_never_sees_the_stream_end_at_the_cap(self) -> None:
        provider = FakeProvider()
        session = await RotatingSttSession.open(provider, CONFIG, rotate_after_seconds=0.0)
        stream = session.events()
        assert isinstance(await asyncio.wait_for(anext(stream), timeout=2.0), SttReady)

        await provider.sessions[0].die("1001 going away")
        # Nothing arrives, but the stream stays open - which is the difference between a
        # rotated session and a dropped call.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(anext(stream), timeout=0.1)
        await session.aclose()

    async def test_an_early_death_ends_the_stream_rather_than_spinning(self) -> None:
        """A provider handing out sessions that die on arrival must not rotate forever."""
        provider = FakeProvider()
        session = await RotatingSttSession.open(provider, CONFIG, rotate_after_seconds=600.0)
        await provider.sessions[0].die("connection reset")

        stream = session.events()
        assert isinstance(await asyncio.wait_for(anext(stream), timeout=2.0), SttReady)
        error = await asyncio.wait_for(anext(stream), timeout=2.0)
        assert isinstance(error, SttStreamError)
        assert error.fatal
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(stream), timeout=2.0)
        assert len(provider.sessions) == 1
        await session.aclose()
