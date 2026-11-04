"""The channel between Dada's mouth and the scammer's ear.

The ageing chain makes the voice old and the ambience makes the room real, but a
300-3400 Hz band-pass is only the first of the things a phone call does to audio. What is
left still sounds like a clean studio recording with its top and bottom trimmed. The missing
cues are all in here, and they are what make the difference between "filtered TTS" and
"someone on a phone":

* **Handset response** — a cheap electret capsule has a presence bump around 1-2 kHz, not
  a flat response.
* **Carrier AGC** — telephony compresses hard. Loud syllables and quiet ones arrive at
  nearly the same level, which is why phone speech sounds flat and close.
* **Narrowband sampling** — a real call is sampled at 8 kHz, not 24 kHz. Going down and
  back up is not the same as band-limiting at 24 kHz: it adds the aliasing a real codec adds.
* **mu-law companding** — G.711 quantises to 8 bits on a logarithmic curve, so quantisation
  noise rides up with the signal. That level-dependent grain is the single most recognisable
  artefact of a landline.
* **Packet loss concealment** — when a VoIP frame goes missing the decoder repeats the
  previous one, fading. Those tiny stutters say "mobile call" more clearly than any filter.

Everything is deterministic: loss positions come from a seeded generator, so the same preset
always produces the same bytes and the content-hash TTS cache stays meaningful.

Pure maths on numpy arrays. Nothing here opens a file or a socket.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from app.core.audio.aging import (
    TELEPHONE_HIGH_HZ,
    TELEPHONE_LOW_HZ,
    Signal,
    band_pass,
    peak_normalise,
)
from app.core.audio.framing import resample_linear

#: G.711 narrowband. Both the sampling rate and the companding law come from the same spec.
NARROWBAND_HZ = 8000
MULAW_MU = 255.0


def handset_colour(samples: Signal, sample_rate: int, presence_db: float) -> Signal:
    """Add the presence bump of a cheap handset capsule.

    Implemented as a scaled band-passed copy rather than a peaking biquad: one more
    `np.convolve` is cheaper than a per-sample recursion and the shape is close enough at
    the few dB that reads as "a phone" rather than "an EQ".
    """
    if presence_db == 0.0 or samples.size == 0:
        return samples
    gain = 10.0 ** (presence_db / 20.0) - 1.0
    coloured: Signal = (samples + gain * band_pass(samples, 1200.0, 2600.0, sample_rate)).astype(
        np.float32
    )
    return coloured


def compress(
    samples: Signal,
    sample_rate: int,
    *,
    amount: float,
    window_ms: float = 25.0,
) -> Signal:
    """Flatten the dynamic range the way a carrier's AGC does.

    `amount` 0 leaves the signal alone and 1 pushes every voiced frame to the same level.
    A short-time envelope with an interpolated gain curve rather than a sample-recursive
    follower: it stays vectorised, and at 25 ms the difference is inaudible on speech.
    """
    if amount <= 0.0 or samples.size == 0:
        return samples
    if not 0.0 <= amount <= 1.0:
        raise ValueError("amount must be in [0, 1]")

    hop = max(1, int(window_ms / 1000.0 * sample_rate))
    frames = samples.size // hop
    if frames < 2:
        return samples

    usable = frames * hop
    blocks = samples[:usable].reshape(frames, hop).astype(np.float64)
    envelope = np.sqrt(np.mean(np.square(blocks), axis=1))

    # Only chase frames that carry speech. Without this floor the AGC hauls the room tone
    # up between words, which is the opposite of the intended effect.
    voiced = envelope > envelope.max() * 0.08
    if not voiced.any():
        return samples
    target = float(np.median(envelope[voiced]))
    gain = np.where(envelope > 1e-6, target / np.maximum(envelope, 1e-6), 1.0)
    gain = np.clip(gain, 0.25, 4.0)
    gain = 1.0 + amount * (gain - 1.0)

    # Interpolate the per-frame gain to per-sample so there is no step at a frame boundary.
    centres = np.arange(frames, dtype=np.float64) * hop + hop / 2.0
    smooth = np.interp(np.arange(samples.size, dtype=np.float64), centres, gain)
    return (samples * smooth).astype(np.float32)


def mulaw_round_trip(samples: Signal) -> Signal:
    """Encode to 8-bit G.711 mu-law and back, keeping the quantisation noise.

    The point is the loss, not the bytes: companding then quantising uniformly to 256 levels
    puts more quantisation noise on loud samples than on quiet ones, and that level-dependent
    grain is what a landline sounds like.
    """
    if samples.size == 0:
        return samples
    x = np.clip(samples.astype(np.float64), -1.0, 1.0)
    companded = np.sign(x) * np.log1p(MULAW_MU * np.abs(x)) / np.log1p(MULAW_MU)
    quantised = np.round(companded * 127.0) / 127.0
    expanded = np.sign(quantised) * (np.expm1(np.abs(quantised) * np.log1p(MULAW_MU)) / MULAW_MU)
    decoded: Signal = expanded.astype(np.float32)
    return decoded


def drop_packets(
    samples: Signal,
    sample_rate: int,
    *,
    loss_rate: float,
    packet_ms: float,
    rng: np.random.Generator,
) -> tuple[Signal, int]:
    """Lose frames and conceal them the way a VoIP decoder does. Returns (audio, dropped).

    Concealment repeats the previous frame at reduced gain rather than inserting silence,
    because that is what the decoder actually does and because a hole of true silence reads
    as an edit rather than as a bad line.
    """
    if loss_rate <= 0.0 or samples.size == 0:
        return samples, 0
    if not 0.0 <= loss_rate < 0.5:
        raise ValueError("loss_rate must be in [0, 0.5)")

    size = max(1, int(packet_ms / 1000.0 * sample_rate))
    count = samples.size // size
    if count < 3:
        return samples, 0

    out = samples.copy()
    lost = rng.random(count) < loss_rate
    lost[0] = False  # nothing to conceal with before the first frame
    dropped = 0
    fade = np.linspace(1.0, 0.35, size, dtype=np.float32)
    for index in np.flatnonzero(lost):
        start = int(index) * size
        previous = out[start - size : start]
        out[start : start + size] = previous * fade
        dropped += 1
    return out, dropped


class TelephonyPreset(BaseModel):
    """One kind of phone line, from a clean landline to a bad mobile connection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    presence_db: float = Field(default=0.0, ge=-12.0, le=12.0)
    compression: float = Field(default=0.0, ge=0.0, le=1.0)
    narrowband: bool = Field(default=False, description="Resample through 8 kHz and back.")
    mulaw: bool = Field(default=False, description="G.711 companding.")
    loss_rate: float = Field(default=0.0, ge=0.0, lt=0.5)
    packet_ms: float = Field(default=20.0, gt=0.0)
    band: bool = Field(
        default=True,
        description="Apply the carrier's 300-3400 Hz filter. False only for the studio "
        "control, so the audition can show what the channel contributes.",
    )
    band_low_hz: float = Field(default=TELEPHONE_LOW_HZ, ge=0.0)
    band_high_hz: float = Field(default=TELEPHONE_HIGH_HZ, gt=0.0)
    seed: int = Field(default=20261105, description="Fixed, so loss positions are reproducible.")

    def apply(self, samples: Signal, sample_rate: int) -> tuple[Signal, int]:
        """Run the channel. Returns the audio and how many packets were dropped.

        Order follows the signal path: capsule, then the encoder's AGC, then the codec's
        sampling and companding, then the network. Reordering these is audible — compressing
        after the codec amplifies quantisation noise in the gaps instead of hiding it under
        speech.
        """
        out = handset_colour(samples, sample_rate, self.presence_db)
        out = compress(out, sample_rate, amount=self.compression)
        if self.band:
            # At the full rate, where a 255-tap kernel has the resolution to put the corner
            # where it is asked for; downsampling first would smear it.
            out = band_pass(out, self.band_low_hz, self.band_high_hz, sample_rate)
        if self.narrowband:
            out = resample_linear(out, sample_rate, NARROWBAND_HZ)
        if self.mulaw:
            out = mulaw_round_trip(out)
        working_rate = NARROWBAND_HZ if self.narrowband else sample_rate
        dropped = 0
        if self.loss_rate > 0.0:
            out, dropped = drop_packets(
                out,
                working_rate,
                loss_rate=self.loss_rate,
                packet_ms=self.packet_ms,
                rng=np.random.default_rng(self.seed),
            )
        if self.narrowband:
            out = resample_linear(out, NARROWBAND_HZ, sample_rate)
        return peak_normalise(out), dropped


#: Channels, cleanest first. `studio` exists so the audition can show what the channel is
#: contributing, the same way `phone-only` isolates the ageing.
TELEPHONY_PRESETS: tuple[TelephonyPreset, ...] = (
    TelephonyPreset(name="studio", band=False),
    TelephonyPreset(name="landline", presence_db=3.0, compression=0.5, narrowband=True, mulaw=True),
    TelephonyPreset(
        name="mobile",
        presence_db=4.5,
        compression=0.7,
        narrowband=True,
        mulaw=True,
        loss_rate=0.004,
    ),
    TelephonyPreset(
        name="mobile-poor",
        presence_db=5.0,
        compression=0.85,
        narrowband=True,
        mulaw=True,
        loss_rate=0.02,
    ),
)

TELEPHONY_BY_NAME: dict[str, TelephonyPreset] = {p.name: p for p in TELEPHONY_PRESETS}
