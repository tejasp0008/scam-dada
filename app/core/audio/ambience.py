"""Synthetic room tone, so the line does not sound like a recording booth.

Digital silence between words is the loudest tell that a voice is synthetic. A real
grandfather on a real phone comes with a ceiling fan, a mains hum the handset picked up, and
a street two floors down. This module generates that bed from noise and sinusoids rather
than from recordings: nothing has to be sourced, licensed or committed, and — because the
generators are seeded — the same preset always produces the same bytes, which the
content-hash TTS cache depends on.

This module also holds the room's *acoustics* — a short reverb — because that is the other
half of sounding like a place. Without it the voice is dry and the bed sits behind it, which
reads as "a recording with noise added" rather than as one room. Reverb is what glues them.

**Where it sits in the chain matters.** The bed is mixed in *before* the telephone band-pass,
not after, because in reality the microphone hears the room and then the carrier filters the
voice and the room together. Mixing after the band-pass would put 50 Hz hum and full-band
fan noise on a line that physically cannot carry them, which is audible as "noise laid over
a phone call" rather than "a phone call from a noisy room". A pleasant consequence: the
50 Hz hum fundamental is removed by the 300 Hz corner exactly as a real line removes it, and
only its upper harmonics survive, faintly.

**The bed is a loop, not a per-utterance effect.** `AmbienceBed.take` is positional so the
caller keeps one running offset for the whole call. Room noise that starts and stops with
each reply is worse than no room noise at all — it tells the listener precisely when the
machine is and is not speaking.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from app.core.audio.aging import band_pass

Signal = npt.NDArray[np.float32]

#: Mains frequency in India. The handset picks up its harmonics, not its fundamental.
MAINS_HZ = 50.0


def _white(rng: np.random.Generator, count: int) -> npt.NDArray[np.float64]:
    return rng.standard_normal(count)


def _fft_convolve(signal: Signal, kernel: Signal) -> Signal:
    """Convolve via FFT.

    A 300 ms impulse response at 24 kHz is 7200 taps, and `np.convolve` against a 12-second
    clip is some two billion multiply-adds. There is no scipy here, so the transform is
    written out: two forward transforms, a multiply and an inverse.
    """
    total = signal.size + kernel.size - 1
    size = 1 << (total - 1).bit_length()
    spectrum = np.fft.rfft(signal, size) * np.fft.rfft(kernel, size)
    return np.fft.irfft(spectrum, size)[:total].astype(np.float32)


class RoomAcoustics(BaseModel):
    """A small room's impulse response: a few early reflections and a decaying tail."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reverb_ms: float = Field(
        default=280.0,
        gt=0.0,
        le=2000.0,
        description="Roughly RT60. A furnished Indian living room is 250-400 ms; much more "
        "than that and it stops sounding like a home.",
    )
    mix: float = Field(
        default=0.16,
        ge=0.0,
        le=1.0,
        description="Wet level. Kept low because a phone mic is held close to the mouth, so "
        "the direct sound dominates even in a live room.",
    )
    reflections: int = Field(default=7, ge=0, le=40)
    seed: int = Field(default=20261105)

    def impulse_response(self, sample_rate: int) -> Signal:
        """Direct sound, then discrete early reflections, then a diffuse exponential tail."""
        count = max(2, int(self.reverb_ms / 1000.0 * sample_rate))
        rng = np.random.default_rng(self.seed)
        response = np.zeros(count, dtype=np.float32)
        response[0] = 1.0

        # Early reflections are what tell a listener the size of the room; the tail only
        # tells them how soft the furniture is.
        for index in range(self.reflections):
            delay = int((0.004 + 0.021 * rng.random()) * sample_rate * (1 + index * 0.6))
            if delay >= count:
                break
            response[delay] += float(0.6 * np.exp(-index * 0.45) * rng.choice([-1.0, 1.0]))

        decay = np.exp(-np.arange(count, dtype=np.float64) / (count / 4.5))
        response += (_white(rng, count) * decay * 0.09).astype(np.float32)
        energy = float(np.sqrt(np.sum(response**2)))
        return (response / energy).astype(np.float32) if energy > 0.0 else response


def room_reverb(samples: Signal, sample_rate: int, acoustics: RoomAcoustics) -> Signal:
    """Place a dry voice in a room, keeping its length and its level."""
    if samples.size == 0 or acoustics.mix <= 0.0:
        return samples
    response = acoustics.impulse_response(sample_rate)
    wet = _fft_convolve(samples, response)[: samples.size]
    dry_rms = _rms(samples)
    wet_rms = _rms(wet)
    if wet_rms > 0.0 and dry_rms > 0.0:
        wet = wet * (dry_rms / wet_rms)
    mixed: Signal = ((1.0 - acoustics.mix) * samples + acoustics.mix * wet).astype(np.float32)
    return mixed


def _rms(samples: npt.NDArray[np.float64] | Signal) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def _normalise(samples: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Scale a layer to unit RMS so preset weights mean the same thing for every layer."""
    level = _rms(samples)
    return samples / level if level > 0.0 else samples


def ceiling_fan(
    count: int, sample_rate: int, *, rpm: float, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Low-passed noise chopped at the blade-pass rate.

    A three-blade fan at 280 rpm passes a blade every 1/14 s, and that 14 Hz chop on a
    broadband whoosh is what makes a ceiling fan recognisable rather than just hiss. A slow
    0.3 Hz drift keeps it from sounding like a tremolo.
    """
    blade_hz = max(0.1, rpm / 60.0 * 3.0)
    t = np.arange(count, dtype=np.float64) / sample_rate
    body = band_pass(_white(rng, count).astype(np.float32), 0.0, 900.0, sample_rate)
    chop = 1.0 + 0.45 * np.sin(2.0 * np.pi * blade_hz * t)
    drift = 1.0 + 0.18 * np.sin(2.0 * np.pi * 0.3 * t + 0.7)
    return _normalise(body.astype(np.float64) * chop * drift)


def mains_hum(count: int, sample_rate: int) -> npt.NDArray[np.float64]:
    """50 Hz and its harmonics, at 1/n amplitude. No noise source, so no seed needed."""
    t = np.arange(count, dtype=np.float64) / sample_rate
    out = np.zeros(count, dtype=np.float64)
    for harmonic in range(1, 12):
        frequency = MAINS_HZ * harmonic
        if frequency >= sample_rate / 2:
            break
        # The phase offset per harmonic stops them all aligning into a click at t=0.
        out += np.sin(2.0 * np.pi * frequency * t + harmonic * 0.9) / harmonic
    return _normalise(out)


def street(count: int, sample_rate: int, *, rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """Distant traffic: low-passed noise with slow swells, as vehicles pass."""
    t = np.arange(count, dtype=np.float64) / sample_rate
    body = band_pass(_white(rng, count).astype(np.float32), 0.0, 500.0, sample_rate)
    swell = (
        1.0 + 0.55 * np.sin(2.0 * np.pi * 0.08 * t) + 0.25 * np.sin(2.0 * np.pi * 0.19 * t + 2.0)
    )
    return _normalise(body.astype(np.float64) * np.maximum(swell, 0.1))


def room_floor(
    count: int, sample_rate: int, *, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """The broadband floor every room has. Band-limited to where a phone can carry it."""
    return _normalise(
        band_pass(_white(rng, count).astype(np.float32), 200.0, 3000.0, sample_rate).astype(
            np.float64
        )
    )


class AmbiencePreset(BaseModel):
    """A mix of room-tone layers, and how loud the whole bed sits under the voice.

    Layer weights are relative RMS, so they are comparable to each other; `level_db` is the
    bed's RMS relative to the voice's and is the only knob that changes how obvious it is.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    fan: float = Field(default=0.0, ge=0.0, le=1.0)
    fan_rpm: float = Field(default=280.0, gt=0.0)
    hum: float = Field(default=0.0, ge=0.0, le=1.0)
    street: float = Field(default=0.0, ge=0.0, le=1.0)
    floor: float = Field(default=0.0, ge=0.0, le=1.0)
    level_db: float = Field(
        default=-26.0,
        le=0.0,
        description="Bed RMS relative to voice RMS. -26 dB is audible but does not fight "
        "the voice; past about -14 dB it starts to cost transcription accuracy if the "
        "scammer's side is also being transcribed.",
    )
    seed: int = Field(default=20261105, description="Fixed, so the bed is reproducible.")
    loop_seconds: float = Field(
        default=12.0,
        gt=1.0,
        description="Length of the generated loop. Long enough that the fan's 0.3 Hz drift "
        "does not repeat audibly, short enough to hold in memory for a whole call.",
    )

    def layers(self) -> tuple[tuple[str, float], ...]:
        """Non-zero layers, for logging and for the audition's output."""
        named = (
            ("fan", self.fan),
            ("hum", self.hum),
            ("street", self.street),
            ("floor", self.floor),
        )
        return tuple((name, weight) for name, weight in named if weight > 0.0)


class AmbienceBed:
    """A generated loop of room tone, read positionally.

    Built once per call and then read with a running offset, so the bed is continuous across
    utterances and across the silences between them.
    """

    def __init__(self, preset: AmbiencePreset, sample_rate: int) -> None:
        self.preset = preset
        self.sample_rate = sample_rate
        count = max(1, int(preset.loop_seconds * sample_rate))
        rng = np.random.default_rng(preset.seed)

        mix = np.zeros(count, dtype=np.float64)
        if preset.fan > 0.0:
            mix += preset.fan * ceiling_fan(count, sample_rate, rpm=preset.fan_rpm, rng=rng)
        if preset.hum > 0.0:
            mix += preset.hum * mains_hum(count, sample_rate)
        if preset.street > 0.0:
            mix += preset.street * street(count, sample_rate, rng=rng)
        if preset.floor > 0.0:
            mix += preset.floor * room_floor(count, sample_rate, rng=rng)

        # Cross-fade the loop into itself so the seam is not a click once per 12 seconds.
        fade = min(count // 4, int(0.25 * sample_rate))
        if fade > 1:
            ramp = np.linspace(0.0, 1.0, fade)
            head = mix[:fade] * ramp
            tail = mix[-fade:] * (1.0 - ramp)
            mix = np.concatenate([head + tail, mix[fade:-fade]])

        self._loop = _normalise(mix).astype(np.float32)

    @property
    def loop(self) -> Signal:
        """The unit-RMS loop itself, exposed for tests and for a continuous live mixer."""
        return self._loop

    def take(self, count: int, offset: int = 0) -> Signal:
        """`count` samples of bed starting at `offset`, wrapping around the loop."""
        if count <= 0:
            return np.zeros(0, dtype=np.float32)
        indices = (np.arange(count, dtype=np.int64) + int(offset)) % self._loop.size
        return self._loop[indices]

    def mix_into(self, voice: Signal, offset: int = 0) -> Signal:
        """Add the bed under `voice` at the preset's level, relative to the voice's RMS.

        Relative to RMS rather than to peak: peak tracks the loudest plosive, so a
        peak-referenced bed gets quieter the more expressive the line is.
        """
        if voice.size == 0:
            return voice
        voice_rms = _rms(voice)
        if voice_rms <= 0.0:
            return voice
        gain = voice_rms * (10.0 ** (self.preset.level_db / 20.0))
        mixed: Signal = (voice + self.take(voice.size, offset) * gain).astype(np.float32)
        return mixed


#: An Indian living room on a weekday afternoon, in three degrees of busy.
#:
#: There is deliberately no "television in the next room" preset: a TV carries speech, and
#: speech in our outgoing audio would be transcribed by the scammer's side as ours and, in
#: the browser simulator where a speaker feeds a microphone, by our own STT as the caller's.
AMBIENCE_PRESETS: tuple[AmbiencePreset, ...] = (
    AmbiencePreset(name="quiet-room", floor=1.0, hum=0.25, level_db=-32.0),
    AmbiencePreset(name="ceiling-fan", fan=1.0, floor=0.45, hum=0.3, level_db=-26.0),
    AmbiencePreset(name="fan-and-street", fan=0.9, street=0.7, floor=0.4, hum=0.25, level_db=-22.0),
)

AMBIENCE_BY_NAME: dict[str, AmbiencePreset] = {p.name: p for p in AMBIENCE_PRESETS}
