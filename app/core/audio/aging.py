"""Make a synthesised voice sound old, and sound like it came down a phone line.

Timbre's catalogue has no elderly voice. Gnani publishes gender and language per voice but
not age, and the male voices all read as 30-45. The gap between "a man" and "a retired
railway clerk" is closed here, in DSP, rather than by asking the TTS model for something it
does not have.

Four effects, in this order:

1. **Pitch down** — resampling scales pitch and duration together.
2. **Tempo** — a WSOLA time-stretch puts duration back where we want it, independently of
   pitch. Without this step the only reachable combination is "lower *and* slower by the
   same factor", which overshoots: a voice slowed 15% is noticeably lower than an old man.
3. **Tremble** — amplitude and pitch modulation around 5 Hz. Age shows up in a voice as
   unsteadiness more than as pitch, which is why `<slow>` alone does not read as elderly.
4. **Room tone** — mixed in here, between the voice processing and the band-pass, because a
   real microphone hears the room and the carrier then filters voice and room together. See
   `app.core.audio.ambience`.
5. **Telephone band** — 300-3400 Hz, the G.712 channel. This does most of the perceptual
   work: it removes the studio-clean low end that makes TTS sound like TTS.

Everything is deterministic. The same PCM and the same preset always produce the same bytes,
because the TTS cache is keyed by content hash and a modulator seeded from a clock would
make every cache entry a different voice.

Pure maths on numpy arrays. Nothing here opens a file or a socket.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

Signal = npt.NDArray[np.float32]

#: The ITU telephone channel. Gnani's STT is trained on call audio, and the scammer will
#: hear our output over a real carrier, so this band is the honest one to demo in.
TELEPHONE_LOW_HZ = 300.0
TELEPHONE_HIGH_HZ = 3400.0

#: Odd, so the kernel is symmetric and the filter is linear-phase (no smearing of the
#: plosives the band-pass is already attenuating).
_FIR_TAPS = 255


class BedMixer(Protocol):
    """Something that can lay room tone under a voice.

    Declared here and implemented in `app.core.audio.ambience` so that the ageing chain can
    mix a bed in at the right point without importing the generators — ambience needs this
    module's band-pass, so the dependency can only run one way.
    """

    def mix_into(self, voice: Signal, offset: int = 0) -> Signal:
        """Return `voice` with the bed added, starting `offset` samples into the loop."""
        ...


def band_pass_kernel(
    low_hz: float, high_hz: float, sample_rate: int, taps: int = _FIR_TAPS
) -> Signal:
    """Windowed-sinc band-pass kernel, normalised to unit gain at the band centre.

    FIR rather than a biquad cascade on purpose: `np.convolve` is one vectorised call, so
    this runs in microseconds and can move into the live path later, whereas a recursive
    biquad in pure Python is a per-sample loop. There is no scipy in this project.
    """
    if taps % 2 == 0:
        raise ValueError("taps must be odd for a symmetric kernel")
    if not 0.0 <= low_hz < high_hz < sample_rate / 2:
        raise ValueError("need 0 <= low < high < nyquist")

    n = np.arange(taps, dtype=np.float64) - (taps - 1) / 2.0
    # Difference of two low-passes is a band-pass. np.sinc already carries the pi.
    high_cut = (2.0 * high_hz / sample_rate) * np.sinc(2.0 * high_hz * n / sample_rate)
    low_cut = (2.0 * low_hz / sample_rate) * np.sinc(2.0 * low_hz * n / sample_rate)
    kernel = (high_cut - low_cut) * np.hamming(taps)

    centre_hz = np.sqrt(low_hz * high_hz)  # geometric centre: the band is ~3.5 octaves
    gain = np.abs(np.sum(kernel * np.exp(-2j * np.pi * centre_hz * n / sample_rate)))
    if gain > 0.0:
        kernel = kernel / gain
    return kernel.astype(np.float32)


def band_pass(samples: Signal, low_hz: float, high_hz: float, sample_rate: int) -> Signal:
    """Apply a telephone-style band-pass."""
    if samples.size == 0:
        return samples
    kernel = band_pass_kernel(low_hz, high_hz, sample_rate)
    return np.convolve(samples, kernel, mode="same").astype(np.float32)


def resample_rate(samples: Signal, rate: float) -> Signal:
    """Replay `samples` at `rate` times speed, scaling pitch and duration together.

    `rate` 0.94 gives a voice about one semitone lower that also takes 6% longer to say the
    same words. `time_stretch` is what separates those two again.
    """
    if rate <= 0.0:
        raise ValueError("rate must be positive")
    if rate == 1.0 or samples.size == 0:
        return samples.copy()
    out_len = max(1, round(samples.size / rate))
    positions = np.arange(out_len, dtype=np.float64) * rate
    source = np.arange(samples.size, dtype=np.float64)
    return np.interp(positions, source, samples).astype(np.float32)


def _best_offset(samples: Signal, target: Signal, low: int, high: int) -> int:
    """Offset in [low, high] whose frame best continues `target`.

    This is the W in WSOLA. Plain overlap-add splices waveforms at arbitrary phase, which
    on a voiced vowel cancels the fundamental and produces the warbling that makes naive
    time-stretching unusable. Aligning to the best correlation keeps the pitch periods lined
    up across the splice.
    """
    if high <= low:
        return low
    width = target.size
    candidates = np.lib.stride_tricks.sliding_window_view(samples[low : high + width], width)
    candidates = candidates[: high - low + 1]
    scores = candidates @ target
    energy = np.sqrt(np.sum(candidates * candidates, axis=1)) + 1e-9
    return low + int(np.argmax(scores / energy))


def time_stretch(
    samples: Signal, factor: float, *, frame_size: int = 1024, search: int = 96
) -> Signal:
    """Stretch duration by `factor` without moving pitch (WSOLA).

    `factor` 1.1 makes the clip 10% longer at the same pitch. Kept to small factors by the
    presets: beyond roughly 1.25 even pitch-synchronous overlap-add starts to sound like a
    stutter, and an elderly voice reads as slow at far less than that.
    """
    if factor <= 0.0:
        raise ValueError("factor must be positive")
    if factor == 1.0 or samples.size < frame_size * 2:
        return samples.copy()

    hop_syn = frame_size // 2
    hop_ana = max(1, round(hop_syn / factor))
    window = np.hanning(frame_size).astype(np.float32)

    out_len = int(samples.size * factor) + frame_size
    out = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)

    analysis = 0
    synthesis = 0
    target = samples[:frame_size]
    last = samples.size - frame_size - search - 1

    while analysis < last and synthesis + frame_size < out_len:
        low = max(0, analysis - search)
        high = min(last, analysis + search)
        chosen = analysis if synthesis == 0 else _best_offset(samples, target, low, high)

        out[synthesis : synthesis + frame_size] += samples[chosen : chosen + frame_size] * window
        norm[synthesis : synthesis + frame_size] += window

        # What would have come next in the source if we had not jumped: the next synthesis
        # frame is chosen to continue this, not to continue wherever the hop lands.
        target = samples[chosen + hop_syn : chosen + hop_syn + frame_size]
        synthesis += hop_syn
        analysis = chosen + hop_ana

    used = synthesis + frame_size
    return (out[:used] / np.maximum(norm[:used], 1e-6)).astype(np.float32)


def tremble(
    samples: Signal,
    sample_rate: int,
    *,
    rate_hz: float,
    am_depth: float,
    vibrato_cents: float,
) -> Signal:
    """Add the unsteadiness of an old voice: slow amplitude and pitch modulation.

    Two modulators at incommensurate rates rather than one, and no random source at all.
    A single sinusoid sounds like a tremolo pedal, while an RNG would make the output
    depend on a seed and quietly break the content-hash TTS cache.
    """
    if samples.size == 0:
        return samples
    if not 0.0 <= am_depth <= 1.0:
        raise ValueError("am_depth must be in [0, 1]")

    t = np.arange(samples.size, dtype=np.float64) / sample_rate
    slow = np.sin(2.0 * np.pi * rate_hz * t)
    # 0.37 is irrational enough over a few seconds that the pair never repeats audibly.
    fast = np.sin(2.0 * np.pi * rate_hz * 0.37 * t + 1.1)
    modulator = 0.72 * slow + 0.28 * fast

    out = samples.astype(np.float64)
    if vibrato_cents > 0.0:
        # Integrate the pitch ratio to get a read position: modulating position directly
        # would modulate delay, not pitch.
        ratio = np.power(2.0, vibrato_cents * modulator / 1200.0)
        position = np.cumsum(ratio)
        position *= (samples.size - 1) / position[-1]
        out = np.interp(position, np.arange(samples.size, dtype=np.float64), out)
    if am_depth > 0.0:
        # Stays at or below unity so the band-pass stage is the only thing that can clip.
        out = out * (1.0 - am_depth * 0.5 * (1.0 - modulator))
    return out.astype(np.float32)


def peak_normalise(samples: Signal, target: float = 0.89) -> Signal:
    """Scale to a fixed peak, so presets can be compared at equal loudness.

    Without this, the quieter preset always sounds worse in an audition and the choice gets
    made on gain rather than on age.
    """
    if samples.size == 0:
        return samples
    peak = float(np.max(np.abs(samples)))
    if peak <= 0.0:
        return samples
    return (samples * (target / peak)).astype(np.float32)


class AgingPreset(BaseModel):
    """One recipe for turning a young synthetic voice into an elderly one.

    Frozen and named: the chosen preset becomes part of the persona's voice identity and has
    to go into the TTS cache key beside the voice name, or two different Dadas collide.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    pitch: float = Field(gt=0.5, le=1.0, description="Pitch scale. 0.94 is ~1 semitone down.")
    speed: float = Field(gt=0.5, le=1.5, description="Playback speed. 0.9 is 10% slower.")
    tremble_hz: float = Field(default=5.0, ge=0.0, le=20.0)
    tremble_depth: float = Field(default=0.0, ge=0.0, le=1.0, description="AM depth.")
    vibrato_cents: float = Field(default=0.0, ge=0.0, le=80.0, description="Pitch wobble.")
    band_low_hz: float = Field(default=TELEPHONE_LOW_HZ, ge=0.0)
    band_high_hz: float = Field(default=TELEPHONE_HIGH_HZ, gt=0.0)

    def apply(
        self,
        samples: Signal,
        sample_rate: int,
        *,
        bed: BedMixer | None = None,
        bed_offset: int = 0,
        lead_ms: float = 0.0,
    ) -> Signal:
        """Run the full chain. Input and output are float32 in [-1, 1].

        `lead_ms` pads silence around the voice so a bed is heard on its own before the first
        word, which is what makes an audition clip sound like a room rather than like a file.
        Leave it at zero in the live path: there the bed is already running continuously, and
        padding would be pure added latency before every reply.
        """
        out = samples.astype(np.float32, copy=True)
        out = resample_rate(out, self.pitch)
        # resample_rate already slowed the clip to `pitch` speed; this restores the asked-for
        # tempo without undoing the pitch change.
        out = time_stretch(out, self.pitch / self.speed)
        if self.tremble_depth > 0.0 or self.vibrato_cents > 0.0:
            out = tremble(
                out,
                sample_rate,
                rate_hz=self.tremble_hz,
                am_depth=self.tremble_depth,
                vibrato_cents=self.vibrato_cents,
            )
        # Normalise before mixing: the bed level is set relative to the voice, so the voice
        # has to be at a known level first or the bed is louder on quieter voices.
        out = peak_normalise(out)
        if lead_ms > 0.0:
            pad = np.zeros(int(lead_ms / 1000.0 * sample_rate), dtype=np.float32)
            out = np.concatenate([pad, out, pad])
        if bed is not None:
            out = bed.mix_into(out, bed_offset)
        out = band_pass(out, self.band_low_hz, self.band_high_hz, sample_rate)
        return peak_normalise(out)


#: Presets for the audition, mildest first. These are post-processing, so all of them are
#: derived from one synthesis per voice — the whole grid costs the credits of a single call.
AGING_PRESETS: tuple[AgingPreset, ...] = (
    # The control. No ageing at all, just the phone channel, so the audition shows how much
    # of the effect is the band-pass rather than the pitch work.
    AgingPreset(name="phone-only", pitch=1.0, speed=1.0),
    AgingPreset(name="mild", pitch=0.97, speed=0.95, tremble_depth=0.07, vibrato_cents=12.0),
    AgingPreset(name="classic", pitch=0.94, speed=0.90, tremble_depth=0.14, vibrato_cents=22.0),
    AgingPreset(name="frail", pitch=0.91, speed=0.85, tremble_depth=0.22, vibrato_cents=34.0),
)

PRESETS_BY_NAME: dict[str, AgingPreset] = {preset.name: preset for preset in AGING_PRESETS}
