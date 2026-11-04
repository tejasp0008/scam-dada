"""The phone line: does each channel effect do what it says, and stay reproducible."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.audio.telephony import (
    NARROWBAND_HZ,
    TELEPHONY_BY_NAME,
    TELEPHONY_PRESETS,
    TelephonyPreset,
    compress,
    drop_packets,
    handset_colour,
    mulaw_round_trip,
)

RATE = 24000


def sine(hz: float, seconds: float = 1.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * RATE), dtype=np.float32) / RATE
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def band_energy(x: np.ndarray, lo: float, hi: float, rate: int = RATE) -> float:
    spectrum = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(x.size, 1 / rate)
    return float(spectrum[(freqs >= lo) & (freqs < hi)].sum())


class TestHandsetColour:
    def test_it_lifts_the_presence_band_and_leaves_the_rest(self) -> None:
        before_presence = rms(handset_colour(sine(1800.0), RATE, 0.0))
        after_presence = rms(handset_colour(sine(1800.0), RATE, 6.0))
        after_low = rms(handset_colour(sine(400.0), RATE, 6.0))
        assert after_presence > before_presence * 1.3
        assert after_low == pytest.approx(rms(sine(400.0)), rel=0.25)

    def test_zero_is_a_no_op(self) -> None:
        tone = sine(1800.0)
        assert np.array_equal(handset_colour(tone, RATE, 0.0), tone)


class TestCompress:
    def test_it_flattens_a_loud_passage_against_a_quiet_one(self) -> None:
        quiet = sine(300.0, 0.5, amplitude=0.05)
        loud = sine(300.0, 0.5, amplitude=0.8)
        signal = np.concatenate([quiet, loud])
        out = compress(signal, RATE, amount=1.0)
        half = out.size // 2
        before = rms(loud) / rms(quiet)
        after = rms(out[half:]) / rms(out[:half])
        assert after < before / 2

    def test_zero_amount_is_a_no_op(self) -> None:
        tone = sine(300.0)
        assert np.array_equal(compress(tone, RATE, amount=0.0), tone)

    def test_silence_is_left_alone(self) -> None:
        silence = np.zeros(RATE, dtype=np.float32)
        assert np.array_equal(compress(silence, RATE, amount=1.0), silence)

    def test_a_clip_shorter_than_two_windows_is_passed_through(self) -> None:
        short = sine(300.0, 0.01)
        assert np.array_equal(compress(short, RATE, amount=1.0), short)

    def test_an_out_of_range_amount_is_refused(self) -> None:
        with pytest.raises(ValueError, match="amount"):
            compress(sine(300.0, 0.5), RATE, amount=2.0)


class TestMulaw:
    def test_it_preserves_the_waveform_roughly(self) -> None:
        tone = sine(400.0)
        out = mulaw_round_trip(tone)
        assert out.size == tone.size
        assert rms(out - tone) < rms(tone) * 0.05

    def test_quantisation_noise_rides_up_with_the_signal(self) -> None:
        """The defining property of mu-law, and what makes it sound like a landline."""
        quiet, loud = sine(400.0, 1.0, 0.02), sine(400.0, 1.0, 0.9)
        quiet_err = rms(mulaw_round_trip(quiet) - quiet)
        loud_err = rms(mulaw_round_trip(loud) - loud)
        assert loud_err > quiet_err * 5

    def test_it_is_lossy_at_all(self) -> None:
        tone = sine(400.0)
        assert not np.array_equal(mulaw_round_trip(tone), tone)

    def test_it_does_not_overflow_at_full_scale(self) -> None:
        out = mulaw_round_trip(sine(400.0, 0.2, amplitude=1.0))
        assert float(np.max(np.abs(out))) <= 1.001

    def test_an_empty_signal_survives(self) -> None:
        assert mulaw_round_trip(np.zeros(0, dtype=np.float32)).size == 0


class TestPacketLoss:
    def test_it_drops_roughly_the_requested_fraction(self) -> None:
        tone = sine(400.0, 10.0)
        _, dropped = drop_packets(
            tone, RATE, loss_rate=0.05, packet_ms=20.0, rng=np.random.default_rng(1)
        )
        packets = tone.size // int(0.02 * RATE)
        assert 0.02 < dropped / packets < 0.1

    def test_concealment_repeats_rather_than_silencing(self) -> None:
        """A hole of true silence reads as an edit; a repeat reads as a bad line."""
        tone = sine(400.0, 2.0)
        out, dropped = drop_packets(
            tone, RATE, loss_rate=0.2, packet_ms=20.0, rng=np.random.default_rng(2)
        )
        assert dropped > 0
        # No gap of dead air anywhere, despite a fifth of the packets being lost.
        size = int(0.02 * RATE)
        blocks = out[: (out.size // size) * size].reshape(-1, size)
        assert float(np.sqrt(np.mean(blocks**2, axis=1)).min()) > 0.01

    def test_the_length_never_changes(self) -> None:
        tone = sine(400.0, 2.0)
        out, _ = drop_packets(
            tone, RATE, loss_rate=0.1, packet_ms=20.0, rng=np.random.default_rng(3)
        )
        assert out.size == tone.size

    def test_the_first_packet_is_never_dropped(self) -> None:
        """There is nothing before it to conceal with."""
        tone = sine(400.0, 2.0)
        out, _ = drop_packets(
            tone, RATE, loss_rate=0.49, packet_ms=20.0, rng=np.random.default_rng(4)
        )
        size = int(0.02 * RATE)
        assert np.array_equal(out[:size], tone[:size])

    def test_zero_loss_is_a_no_op(self) -> None:
        tone = sine(400.0, 1.0)
        out, dropped = drop_packets(
            tone, RATE, loss_rate=0.0, packet_ms=20.0, rng=np.random.default_rng(5)
        )
        assert dropped == 0
        assert np.array_equal(out, tone)

    def test_an_out_of_range_loss_rate_is_refused(self) -> None:
        with pytest.raises(ValueError, match="loss_rate"):
            drop_packets(
                sine(400.0, 1.0), RATE, loss_rate=0.8, packet_ms=20.0, rng=np.random.default_rng(1)
            )


class TestPresets:
    def test_names_are_unique_and_indexed(self) -> None:
        names = [preset.name for preset in TELEPHONY_PRESETS]
        assert len(names) == len(set(names))
        assert {p.name: p for p in TELEPHONY_PRESETS} == TELEPHONY_BY_NAME

    def test_studio_is_a_true_control(self) -> None:
        studio = TELEPHONY_BY_NAME["studio"]
        assert not studio.band
        assert not studio.narrowband
        assert not studio.mulaw
        assert studio.loss_rate == 0.0
        assert studio.compression == 0.0

    @pytest.mark.parametrize("preset", TELEPHONY_PRESETS, ids=lambda p: p.name)
    def test_applying_a_preset_produces_usable_audio(self, preset: TelephonyPreset) -> None:
        voice = (sine(300.0) + sine(900.0) * 0.5 + sine(2200.0) * 0.2).astype(np.float32)
        out, dropped = preset.apply(voice, RATE)
        assert out.size > 0
        assert np.all(np.isfinite(out))
        assert float(np.max(np.abs(out))) <= 1.0
        assert dropped >= 0

    @pytest.mark.parametrize("preset", TELEPHONY_PRESETS, ids=lambda p: p.name)
    def test_applying_a_preset_is_deterministic(self, preset: TelephonyPreset) -> None:
        voice = sine(500.0, 2.0)
        first, first_dropped = preset.apply(voice, RATE)
        second, second_dropped = preset.apply(voice, RATE)
        assert np.array_equal(first, second)
        assert first_dropped == second_dropped

    @pytest.mark.parametrize("name", ["landline", "mobile", "mobile-poor"])
    def test_every_real_line_band_limits(self, name: str) -> None:
        voice = (sine(120.0) + sine(1000.0) + sine(6000.0)).astype(np.float32)
        out, _ = TELEPHONY_BY_NAME[name].apply(voice, RATE)
        total = band_energy(out, 0.0, RATE / 2) + 1e-30
        assert band_energy(out, 0.0, 250.0) / total < 0.05
        assert band_energy(out, 3600.0, RATE / 2) / total < 0.05

    def test_narrowband_does_not_change_the_output_rate(self) -> None:
        """The pipeline runs at one rate; 8 kHz is internal to the codec stage."""
        voice = sine(500.0, 1.0)
        out, _ = TELEPHONY_BY_NAME["landline"].apply(voice, RATE)
        assert out.size == pytest.approx(voice.size, rel=0.01)

    def test_a_poor_line_actually_drops_packets(self) -> None:
        out, dropped = TELEPHONY_BY_NAME["mobile-poor"].apply(sine(500.0, 8.0), RATE)
        assert dropped > 0
        assert out.size > 0

    def test_the_narrowband_rate_is_the_g711_rate(self) -> None:
        assert NARROWBAND_HZ == 8000
