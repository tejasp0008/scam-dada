"""Frame slicing and resampling.

Prisma rejects any frame that is not exactly 1024 bytes, so these are the tests that stop
a silently garbled transcript later.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.audio.framing import (
    PRISMA_FRAME_BYTES,
    FrameBuffer,
    float32_to_pcm,
    pcm_to_float32,
    resample_linear,
)


def test_frames_are_emitted_only_when_whole() -> None:
    buf = FrameBuffer()
    assert buf.push(b"\x00" * (PRISMA_FRAME_BYTES - 2)) == []
    assert buf.pending_bytes == PRISMA_FRAME_BYTES - 2
    frames = buf.push(b"\x00" * 4)
    assert len(frames) == 1
    assert len(frames[0]) == PRISMA_FRAME_BYTES
    assert buf.pending_bytes == 2


def test_a_large_push_emits_several_frames_and_keeps_the_remainder() -> None:
    buf = FrameBuffer()
    frames = buf.push(b"\x01" * (PRISMA_FRAME_BYTES * 3 + 7))
    assert len(frames) == 3
    assert all(len(f) == PRISMA_FRAME_BYTES for f in frames)
    assert buf.pending_bytes == 7


def test_no_byte_is_lost_or_duplicated_across_ragged_pushes() -> None:
    """Browser chunk sizes have no relationship to the frame size."""
    payload = bytes(range(256)) * 20
    buf = FrameBuffer()
    emitted = bytearray()
    for start in range(0, len(payload), 37):
        for frame in buf.push(payload[start : start + 37]):
            emitted.extend(frame)
    tail = buf.flush(pad=False)
    if tail:
        emitted.extend(tail)
    assert bytes(emitted) == payload


def test_flush_pads_the_final_partial_frame() -> None:
    buf = FrameBuffer()
    buf.push(b"\x7f" * 10)
    tail = buf.flush()
    assert tail is not None
    assert len(tail) == PRISMA_FRAME_BYTES
    assert tail[:10] == b"\x7f" * 10
    assert tail[10:] == b"\x00" * (PRISMA_FRAME_BYTES - 10)
    assert buf.flush() is None


def test_odd_frame_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="even number"):
        FrameBuffer(frame_bytes=1023)


def test_pcm_round_trip_is_within_one_quantisation_step() -> None:
    original = np.array([0.0, 0.5, -0.5, 0.999, -0.999], dtype=np.float32)
    restored = pcm_to_float32(float32_to_pcm(original))
    assert np.allclose(original, restored, atol=1 / 32768)


def test_encoding_clips_instead_of_wrapping() -> None:
    """A wrapped sample is an audible click; a clipped one is not."""
    loud = np.array([2.0, -2.0], dtype=np.float32)
    decoded = pcm_to_float32(float32_to_pcm(loud))
    assert decoded[0] > 0.99
    assert decoded[1] < -0.99


def test_downsampling_48k_to_16k_gives_a_third_of_the_samples() -> None:
    samples = np.zeros(4800, dtype=np.float32)
    out = resample_linear(samples, 48000, 16000)
    assert out.size == 1600


def test_resampling_preserves_a_tone_frequency() -> None:
    """A 440 Hz tone must still peak at 440 Hz after 48k -> 16k."""
    duration_s = 0.5
    src_rate, dst_rate, freq = 48000, 16000, 440.0
    t = np.arange(int(src_rate * duration_s), dtype=np.float32) / src_rate
    tone = np.sin(2 * np.pi * freq * t).astype(np.float32)
    out = resample_linear(tone, src_rate, dst_rate)
    spectrum = np.abs(np.fft.rfft(out))
    peak_hz = float(np.fft.rfftfreq(out.size, 1 / dst_rate)[int(np.argmax(spectrum))])
    assert abs(peak_hz - freq) < 5.0


def test_resampling_is_a_no_op_at_the_same_rate() -> None:
    samples = np.array([0.1, 0.2], dtype=np.float32)
    assert resample_linear(samples, 16000, 16000) is samples


def test_empty_input_resamples_to_empty() -> None:
    assert resample_linear(np.zeros(0, dtype=np.float32), 48000, 16000).size == 0


def test_non_positive_rates_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        resample_linear(np.zeros(4, dtype=np.float32), 0, 16000)
