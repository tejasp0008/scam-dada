"""Energy VAD behaviour.

The hysteresis is the point: barge-in that triggers on a single loud frame cancels the
persona mid-word every time the line clicks.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.audio.framing import PRISMA_FRAME_SAMPLES, float32_to_pcm
from app.core.audio.vad import EnergyVad


def tone_frame(amplitude: float) -> bytes:
    t = np.arange(PRISMA_FRAME_SAMPLES, dtype=np.float32) / 16000.0
    return float32_to_pcm((np.sin(2 * np.pi * 300.0 * t) * amplitude).astype(np.float32))


SILENCE = b"\x00" * (PRISMA_FRAME_SAMPLES * 2)
LOUD = tone_frame(0.5)
QUIET = tone_frame(0.001)


def test_silence_is_never_speech() -> None:
    vad = EnergyVad()
    assert not any(vad.is_speech(SILENCE) for _ in range(50))
    assert vad.rms(SILENCE) == 0.0


def test_speech_needs_consecutive_loud_frames_to_open() -> None:
    vad = EnergyVad(speech_frames_to_open=3)
    assert not vad.is_speech(LOUD)
    assert not vad.is_speech(LOUD)
    assert vad.is_speech(LOUD)
    assert vad.active


def test_an_isolated_loud_frame_does_not_open_the_gate() -> None:
    """A line click must not cut the persona off."""
    vad = EnergyVad(speech_frames_to_open=3)
    for _ in range(10):
        assert not vad.is_speech(LOUD)
        assert not vad.is_speech(SILENCE)
        assert not vad.is_speech(SILENCE)


def test_the_gate_stays_open_through_short_pauses() -> None:
    """A speaker pausing between words is still speaking."""
    vad = EnergyVad(speech_frames_to_open=2, silence_frames_to_close=12)
    for _ in range(3):
        vad.is_speech(LOUD)
    assert vad.active
    for _ in range(11):
        assert vad.is_speech(SILENCE)
    assert not vad.is_speech(SILENCE)
    assert not vad.active


def test_quiet_frames_fall_below_the_threshold() -> None:
    vad = EnergyVad(threshold_rms=0.015)
    assert vad.rms(QUIET) < 0.015
    assert not any(vad.is_speech(QUIET) for _ in range(20))


def test_reset_closes_the_gate() -> None:
    vad = EnergyVad(speech_frames_to_open=1)
    assert vad.is_speech(LOUD)
    vad.reset()
    assert not vad.active
    assert not vad.is_speech(LOUD) or vad.active  # a fresh run must build up again


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 2.0])
def test_out_of_range_thresholds_are_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold_rms"):
        EnergyVad(threshold_rms=threshold)
