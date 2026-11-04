"""Frame slicing and resampling for the Prisma streaming contract.

Prisma requires raw PCM s16le mono in frames of *exactly* 1024 bytes (512 samples) sent at
real-time cadence. Browsers capture at the device rate (usually 48 kHz), so the pipeline
has to resample and re-frame. Both operations are pure functions over numpy arrays and
live in `core` for that reason: frame-boundary bugs are the kind that only show up as
garbled transcripts hours later, so they get tested directly.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: Prisma's fixed frame size. Not configurable: the server rejects anything else.
PRISMA_FRAME_BYTES: int = 1024
PRISMA_FRAME_SAMPLES: int = PRISMA_FRAME_BYTES // 2  # s16le


class FrameBuffer:
    """Accumulates PCM bytes and emits only whole 1024-byte frames.

    Audio arrives from a browser in chunks that have no relationship to Prisma's frame
    size, so a remainder must be carried across calls rather than padded or dropped.
    """

    def __init__(self, frame_bytes: int = PRISMA_FRAME_BYTES) -> None:
        if frame_bytes <= 0 or frame_bytes % 2:
            raise ValueError("frame_bytes must be a positive even number of bytes")
        self._frame_bytes = frame_bytes
        self._buf = bytearray()

    @property
    def pending_bytes(self) -> int:
        """Bytes held back because they do not fill a frame."""
        return len(self._buf)

    def push(self, data: bytes) -> list[bytes]:
        """Add `data` and return every whole frame now available."""
        self._buf.extend(data)
        frames: list[bytes] = []
        while len(self._buf) >= self._frame_bytes:
            frames.append(bytes(self._buf[: self._frame_bytes]))
            del self._buf[: self._frame_bytes]
        return frames

    def flush(self, pad: bool = True) -> bytes | None:
        """Return the remainder as a final frame, zero-padded, and clear the buffer.

        Used at end of utterance so a trailing word is not lost. Returns None if empty.
        """
        if not self._buf:
            return None
        tail = bytes(self._buf)
        self._buf.clear()
        if pad and len(tail) < self._frame_bytes:
            tail = tail + b"\x00" * (self._frame_bytes - len(tail))
        return tail


def pcm_to_float32(data: bytes) -> npt.NDArray[np.float32]:
    """Decode s16le bytes to float32 samples in [-1, 1)."""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm(samples: npt.NDArray[np.float32]) -> bytes:
    """Encode float32 samples in [-1, 1] to s16le bytes, clipping out-of-range values."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def resample_linear(
    samples: npt.NDArray[np.float32], src_rate: int, dst_rate: int
) -> npt.NDArray[np.float32]:
    """Resample with linear interpolation.

    Linear interpolation is not a good anti-aliasing filter, but downsampling 48k speech to
    16k for an ASR that was trained on telephone audio is dominated by other losses, and it
    keeps the hot path allocation-free and dependency-free. If transcript quality turns out
    to be the bottleneck we revisit this with a proper polyphase filter.
    """
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be positive")
    if src_rate == dst_rate or samples.size == 0:
        return samples
    duration = samples.size / src_rate
    out_count = int(duration * dst_rate)
    if out_count == 0:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.arange(samples.size, dtype=np.float32)
    dst_idx = np.linspace(0.0, samples.size - 1, out_count, dtype=np.float32)
    return np.interp(dst_idx, src_idx, samples).astype(np.float32)
