"""The speaker: what actually reached the caller, and the room tone between utterances."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.core.audio.call_profile import PROFILES_BY_NAME
from app.core.audio.framing import pcm_to_float32
from app.providers.base import TtsRequest
from app.providers.mock import MockTTSProvider
from app.services.speaker import PersonaSpeaker

PROFILE = PROFILES_BY_NAME["landline"]
RATE = 24000


def request(text: str = "Haan beta, ek minute.") -> TtsRequest:
    return TtsRequest(
        text=text, voice="Devansh", language="hi-en", model="timbre-v3.0", sample_rate=RATE
    )


def build(*, delay_s: float = 0.0) -> tuple[PersonaSpeaker, list[bytes]]:
    out: list[bytes] = []

    async def sink(chunk: bytes) -> None:
        if delay_s:
            await asyncio.sleep(delay_s)
        out.append(chunk)

    return PersonaSpeaker(MockTTSProvider(ms_per_char=4, chunk_ms=40), sink), out


class TestWhatReachedTheCaller:
    async def test_a_completed_utterance_says_so(self) -> None:
        speaker, out = build()
        result = await speaker.speak(request(), profile=PROFILE)
        assert result.completed
        assert result.audio_seconds > 0
        assert out

    async def test_an_interrupted_utterance_is_not_reported_as_completed(self) -> None:
        """This is the bug that put words in Dada's mouth he never said: `speak` returned
        only a latency, so a cut-off sentence looked identical to a finished one and the
        orchestrator recorded it either way."""
        speaker, _ = build(delay_s=0.02)
        task = asyncio.create_task(speaker.speak(request("a" * 400), profile=PROFILE))
        for _ in range(200):
            await asyncio.sleep(0.005)
            if speaker.speaking:
                break
        await speaker.stop()
        result = await task
        assert not result.completed
        assert speaker.interruptions == 1

    async def test_it_reports_how_much_was_delivered(self) -> None:
        speaker, _ = build()
        result = await speaker.speak(request(), profile=PROFILE)
        assert result.delivered_seconds == pytest.approx(result.audio_seconds, rel=0.05)


class TestRoomTone:
    """A line that goes digitally silent between utterances is the loudest tell left."""

    def test_it_produces_audio(self) -> None:
        speaker, _ = build()
        assert speaker.room_tone(PROFILE, RATE, RATE // 2)

    def test_it_advances_the_shared_offset_so_the_fan_does_not_restart(self) -> None:
        speaker, _ = build()
        before = speaker.bed_offset
        speaker.room_tone(PROFILE, RATE, RATE // 2)
        assert speaker.bed_offset == before + RATE // 2

    def test_consecutive_chunks_differ(self) -> None:
        """Repeating one chunk would be an audible loop rather than a room."""
        speaker, _ = build()
        first = speaker.room_tone(PROFILE, RATE, RATE // 2)
        second = speaker.room_tone(PROFILE, RATE, RATE // 2)
        assert first != second

    def test_a_profile_without_a_room_produces_nothing(self) -> None:
        speaker, _ = build()
        assert speaker.room_tone(PROFILES_BY_NAME["studio"], RATE, RATE) == b""

    def test_it_is_quieter_than_speech(self) -> None:
        """Otherwise it is not a background, it is a second speaker."""
        speaker, _ = build()
        bed = pcm_to_float32(speaker.room_tone(PROFILE, RATE, RATE))
        bed_rms = float(np.sqrt(np.mean(bed**2)))
        assert 0.0 < bed_rms < 0.05


class TestRoomToneLoop:
    async def test_it_streams_while_nobody_is_speaking(self) -> None:
        speaker, out = build()
        speaker.start_room_tone(PROFILE, RATE)
        await asyncio.sleep(1.4)
        speaker.stop_room_tone()
        assert out, "the line was digitally silent"

    async def test_switching_it_off_stops_the_audio(self) -> None:
        speaker, out = build()
        speaker.start_room_tone(PROFILE, RATE)
        await asyncio.sleep(1.2)
        before = len(out)
        speaker.set_room_tone(False)
        await asyncio.sleep(1.2)
        speaker.stop_room_tone()
        assert len(out) == before

    async def test_switching_it_back_on_resumes(self) -> None:
        speaker, out = build()
        speaker.start_room_tone(PROFILE, RATE)
        speaker.set_room_tone(False)
        await asyncio.sleep(1.2)
        paused = len(out)
        speaker.set_room_tone(True)
        await asyncio.sleep(1.2)
        speaker.stop_room_tone()
        assert len(out) > paused

    async def test_stopping_it_is_safe_when_it_never_started(self) -> None:
        speaker, _ = build()
        speaker.stop_room_tone()

    async def test_a_profile_without_a_room_starts_no_loop(self) -> None:
        speaker, out = build()
        speaker.start_room_tone(PROFILES_BY_NAME["studio"], RATE)
        await asyncio.sleep(1.2)
        speaker.stop_room_tone()
        assert not out
