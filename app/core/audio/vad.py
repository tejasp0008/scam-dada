"""Voice activity detection behind an interface.

Prisma ships its own server-side VAD and reports `speech_start`, but barge-in cannot wait
for a network round trip: by the time the server tells us the scammer started talking, the
persona has already talked over them. So a cheap local energy VAD runs on the inbound
frames purely to cut playback, and Prisma's events remain the authority on turn boundaries.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from app.core.audio.framing import pcm_to_float32


def frame_rms(frame: bytes) -> float:
    """Root-mean-square amplitude of one PCM frame, in [0, 1].

    A free function rather than only a VAD method: the barge-in gate needs a level without
    needing a detector, and adding `rms` to the `Vad` protocol would oblige every future
    implementation to expose one - a Silero VAD has no natural notion of it.
    """
    samples = pcm_to_float32(frame)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


class Vad(Protocol):
    """Frame-by-frame speech detection."""

    def is_speech(self, frame: bytes) -> bool:
        """Return whether this PCM frame contains speech."""
        ...

    def reset(self) -> None:
        """Forget accumulated state, e.g. between utterances."""
        ...


class EnergyVad:
    """RMS-threshold VAD with hysteresis.

    Hysteresis (requiring N consecutive speech frames to open and M to close) is what stops
    a cough or a line click from cancelling the persona mid-sentence, which is the failure
    mode that makes barge-in feel broken.
    """

    def __init__(
        self,
        threshold_rms: float = 0.015,
        speech_frames_to_open: int = 3,
        silence_frames_to_close: int = 12,
    ) -> None:
        if not 0.0 < threshold_rms < 1.0:
            raise ValueError("threshold_rms must be in (0, 1)")
        self._threshold = threshold_rms
        self._open_after = speech_frames_to_open
        self._close_after = silence_frames_to_close
        self._speech_run = 0
        self._silence_run = 0
        self._active = False

    @property
    def active(self) -> bool:
        """Whether the VAD currently believes someone is speaking."""
        return self._active

    def rms(self, frame: bytes) -> float:
        """Root-mean-square amplitude of a frame, in [0, 1]."""
        return frame_rms(frame)

    def is_speech(self, frame: bytes) -> bool:
        loud = self.rms(frame) >= self._threshold
        if loud:
            self._speech_run += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
            self._speech_run = 0

        if not self._active and self._speech_run >= self._open_after:
            self._active = True
        elif self._active and self._silence_run >= self._close_after:
            self._active = False
        return self._active

    def reset(self) -> None:
        self._speech_run = 0
        self._silence_run = 0
        self._active = False
