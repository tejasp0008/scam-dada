"""The ageing chain: does it do the thing it claims, and does it do it the same way twice."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.core.audio.aging import (
    AGING_PRESETS,
    PRESETS_BY_NAME,
    TELEPHONE_HIGH_HZ,
    TELEPHONE_LOW_HZ,
    AgingPreset,
    band_pass,
    band_pass_kernel,
    inhale,
    peak_normalise,
    resample_rate,
    time_stretch,
    tremble,
    wander,
)

RATE = 24000


def sine(hz: float, seconds: float = 1.0, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def dominant_hz(samples: np.ndarray, rate: int = RATE) -> float:
    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
    return float(np.fft.rfftfreq(samples.size, 1 / rate)[int(np.argmax(spectrum))])


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


class TestBandPass:
    def test_the_passband_is_left_roughly_alone(self) -> None:
        tone = sine(1000.0)
        filtered = band_pass(tone, TELEPHONE_LOW_HZ, TELEPHONE_HIGH_HZ, RATE)
        assert rms(filtered) == pytest.approx(rms(tone), rel=0.15)

    @pytest.mark.parametrize("hz", [50.0, 120.0, 6000.0, 9000.0])
    def test_everything_outside_the_telephone_band_is_attenuated(self, hz: float) -> None:
        filtered = band_pass(sine(hz), TELEPHONE_LOW_HZ, TELEPHONE_HIGH_HZ, RATE)
        # 20 dB down is enough to take a mains hum or a studio-clean top end out of the way.
        assert rms(filtered) < rms(sine(hz)) * 0.1

    def test_a_zero_low_corner_gives_a_low_pass(self) -> None:
        """Ambience relies on this: `band_pass(x, 0, f)` is how its layers are shaped."""
        assert rms(band_pass(sine(200.0), 0.0, 900.0, RATE)) > 0.2
        assert rms(band_pass(sine(4000.0), 0.0, 900.0, RATE)) < 0.05

    def test_the_kernel_is_symmetric_so_the_filter_is_linear_phase(self) -> None:
        kernel = band_pass_kernel(TELEPHONE_LOW_HZ, TELEPHONE_HIGH_HZ, RATE)
        assert np.allclose(kernel, kernel[::-1], atol=1e-7)

    def test_an_even_tap_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            band_pass_kernel(300.0, 3400.0, RATE, taps=256)

    def test_a_corner_above_nyquist_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nyquist"):
            band_pass_kernel(300.0, 20000.0, RATE)

    def test_an_empty_signal_survives(self) -> None:
        assert band_pass(np.zeros(0, dtype=np.float32), 300.0, 3400.0, RATE).size == 0


class TestResample:
    def test_halving_the_rate_halves_the_pitch_and_doubles_the_length(self) -> None:
        out = resample_rate(sine(800.0), 0.5)
        assert out.size == pytest.approx(RATE * 2, rel=0.01)
        assert dominant_hz(out) == pytest.approx(400.0, rel=0.02)

    def test_unity_is_a_copy_not_the_same_array(self) -> None:
        source = sine(440.0, 0.1)
        out = resample_rate(source, 1.0)
        assert np.array_equal(out, source)
        assert out is not source

    def test_a_non_positive_rate_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            resample_rate(sine(440.0, 0.1), 0.0)


class TestTimeStretch:
    @pytest.mark.parametrize("factor", [0.85, 1.1, 1.25])
    def test_length_follows_the_factor(self, factor: float) -> None:
        out = time_stretch(sine(300.0), factor)
        assert out.size == pytest.approx(RATE * factor, rel=0.06)

    @pytest.mark.parametrize("factor", [0.9, 1.15])
    def test_pitch_does_not_move(self, factor: float) -> None:
        """The whole reason for WSOLA rather than resampling: duration without pitch."""
        out = time_stretch(sine(440.0), factor)
        assert dominant_hz(out) == pytest.approx(440.0, rel=0.03)

    def test_amplitude_is_preserved_across_the_overlap_add(self) -> None:
        out = time_stretch(sine(440.0), 1.1)
        # A normalisation bug in the overlap-add shows up here as a halved or doubled RMS.
        assert rms(out) == pytest.approx(rms(sine(440.0)), rel=0.15)

    def test_a_signal_shorter_than_a_frame_is_passed_through(self) -> None:
        short = sine(440.0, 0.01)
        assert np.array_equal(time_stretch(short, 1.2), short)


class TestTremble:
    def test_it_is_deterministic(self) -> None:
        """The TTS cache is keyed by content hash, so an RNG here would break it."""
        source = sine(440.0, 0.5)
        first = tremble(source, RATE, rate_hz=5.0, am_depth=0.2, vibrato_cents=30.0)
        second = tremble(source, RATE, rate_hz=5.0, am_depth=0.2, vibrato_cents=30.0)
        assert np.array_equal(first, second)

    def test_amplitude_modulation_never_amplifies(self) -> None:
        source = sine(440.0, 0.5)
        out = tremble(source, RATE, rate_hz=5.0, am_depth=0.5, vibrato_cents=0.0)
        assert float(np.max(np.abs(out))) <= float(np.max(np.abs(source))) + 1e-6

    def test_vibrato_spreads_the_spectrum_without_moving_the_centre(self) -> None:
        out = tremble(sine(440.0), RATE, rate_hz=5.0, am_depth=0.0, vibrato_cents=40.0)
        assert dominant_hz(out) == pytest.approx(440.0, abs=6.0)
        assert out.size == RATE

    def test_zero_depth_and_zero_vibrato_is_a_no_op(self) -> None:
        source = sine(440.0, 0.2)
        out = tremble(source, RATE, rate_hz=5.0, am_depth=0.0, vibrato_cents=0.0)
        assert np.allclose(out, source, atol=1e-6)

    def test_an_out_of_range_depth_is_refused(self) -> None:
        with pytest.raises(ValueError, match="am_depth"):
            tremble(sine(440.0, 0.1), RATE, rate_hz=5.0, am_depth=1.5, vibrato_cents=0.0)


class TestPeakNormalise:
    def test_it_hits_the_target_peak(self) -> None:
        out = peak_normalise(sine(440.0, 0.1) * 0.01, target=0.9)
        assert float(np.max(np.abs(out))) == pytest.approx(0.9, rel=1e-5)

    def test_silence_is_left_as_silence(self) -> None:
        silence = np.zeros(100, dtype=np.float32)
        assert np.array_equal(peak_normalise(silence), silence)


class TestPresets:
    def test_every_preset_has_a_unique_name_and_is_in_the_index(self) -> None:
        names = [preset.name for preset in AGING_PRESETS]
        assert len(names) == len(set(names))
        assert {p.name: p for p in AGING_PRESETS} == PRESETS_BY_NAME

    @pytest.mark.parametrize("preset", AGING_PRESETS, ids=lambda p: p.name)
    def test_applying_a_preset_produces_usable_audio(self, preset: AgingPreset) -> None:
        voice = (sine(220.0, 1.0) + sine(700.0, 1.0) * 0.4).astype(np.float32)
        out = preset.apply(voice, RATE)
        assert out.size > 0
        assert np.all(np.isfinite(out))
        assert float(np.max(np.abs(out))) <= 1.0

    @pytest.mark.parametrize("preset", AGING_PRESETS, ids=lambda p: p.name)
    def test_applying_a_preset_is_deterministic(self, preset: AgingPreset) -> None:
        voice = sine(220.0, 0.6)
        assert np.array_equal(preset.apply(voice, RATE), preset.apply(voice, RATE))

    def test_the_speed_knob_changes_duration(self) -> None:
        voice = sine(220.0, 1.0)
        slow = PRESETS_BY_NAME["frail"].apply(voice, RATE)
        plain = PRESETS_BY_NAME["none"].apply(voice, RATE)
        assert slow.size > plain.size

    def test_the_pitch_knob_lowers_the_fundamental(self) -> None:
        voice = sine(220.0, 1.0)
        aged = PRESETS_BY_NAME["frail"].apply(voice, RATE)
        # 0.91 of 220 Hz is ~200 Hz, and the band-pass leaves a 220 Hz tone audible.
        assert dominant_hz(aged) < 215.0

    def test_none_is_a_true_control(self) -> None:
        """The audition needs one preset that leaves the voice exactly as Timbre made it."""
        preset = PRESETS_BY_NAME["none"]
        assert preset.pitch == 1.0
        assert preset.speed == 1.0
        assert preset.tremble_depth == 0.0
        assert preset.vibrato_cents == 0.0
        assert preset.inhale_ms == 0.0

    def test_the_control_preset_only_normalises(self) -> None:
        voice = sine(220.0, 0.5) * 0.3
        out = PRESETS_BY_NAME["none"].apply(voice, RATE)
        assert out.size == voice.size
        assert dominant_hz(out) == pytest.approx(220.0, rel=0.01)

    def test_ageing_does_not_band_limit(self) -> None:
        """The carrier filter belongs to telephony now; ageing must not pre-empt it."""
        out = PRESETS_BY_NAME["classic"].apply(sine(150.0, 1.0), RATE)
        # A 150 Hz tone would be gone if a 300 Hz high-pass were still in this stage.
        assert rms(out) > 0.05

    @pytest.mark.parametrize("name", ["mild", "classic", "frail"])
    def test_every_aged_preset_breathes_before_speaking(self, name: str) -> None:
        """Synthetic speech starts from absolute stillness; nobody does that."""
        preset = PRESETS_BY_NAME[name]
        assert preset.inhale_ms > 0.0
        voice = sine(220.0, 0.5)
        out = preset.apply(voice, RATE)
        head = out[: int(0.05 * RATE)]
        assert rms(head) > 0.0

    @pytest.mark.parametrize("name", ["mild", "classic", "frail"])
    def test_every_aged_preset_wanders_rather_than_oscillating(self, name: str) -> None:
        """A purely sinusoidal tremble reads as a tremolo pedal, not as age."""
        assert PRESETS_BY_NAME[name].jitter > 0.0

    def test_a_preset_outside_the_declared_ranges_is_refused(self) -> None:
        with pytest.raises(ValueError):
            AgingPreset(name="silly", pitch=0.1, speed=1.0)


class TestWander:
    def test_it_stays_in_range_and_is_seeded(self) -> None:
        first = wander(RATE, RATE, 4.0, np.random.default_rng(7))
        second = wander(RATE, RATE, 4.0, np.random.default_rng(7))
        assert np.array_equal(first, second)
        assert float(np.max(np.abs(first))) == pytest.approx(1.0, rel=1e-5)

    def test_a_different_seed_gives_a_different_drift(self) -> None:
        assert not np.array_equal(
            wander(RATE, RATE, 4.0, np.random.default_rng(1)),
            wander(RATE, RATE, 4.0, np.random.default_rng(2)),
        )

    def test_the_drift_is_slow(self) -> None:
        """It modulates a voice, so energy above a few Hz would be heard as noise."""
        drift = wander(RATE * 2, RATE, 4.0, np.random.default_rng(3))
        assert dominant_hz(drift) < 12.0

    def test_nothing_requested_is_nothing_returned(self) -> None:
        assert wander(0, RATE, 4.0, np.random.default_rng(1)).size == 0


class TestInhale:
    def test_it_has_the_requested_length_and_fades_in_and_out(self) -> None:
        breath = inhale(RATE, duration_ms=300.0, rng=np.random.default_rng(1))
        assert breath.size == pytest.approx(0.3 * RATE, rel=0.01)
        edges = max(abs(float(breath[0])), abs(float(breath[-1])))
        assert edges < float(np.max(np.abs(breath))) * 0.05

    def test_it_sits_in_the_band_a_phone_can_carry(self) -> None:
        breath = inhale(RATE, duration_ms=300.0, rng=np.random.default_rng(1))
        assert 200.0 < dominant_hz(breath) < 2400.0

    def test_it_is_seeded(self) -> None:
        assert np.array_equal(
            inhale(RATE, duration_ms=200.0, rng=np.random.default_rng(5)),
            inhale(RATE, duration_ms=200.0, rng=np.random.default_rng(5)),
        )

    def test_zero_duration_is_no_breath(self) -> None:
        assert inhale(RATE, duration_ms=0.0, rng=np.random.default_rng(1)).size == 0


class TestJitter:
    def test_jitter_changes_the_result(self) -> None:
        voice = sine(440.0, 1.0)
        even = tremble(voice, RATE, rate_hz=5.0, am_depth=0.2, vibrato_cents=20.0, jitter=0.0)
        noisy = tremble(voice, RATE, rate_hz=5.0, am_depth=0.2, vibrato_cents=20.0, jitter=0.8)
        assert not np.array_equal(even, noisy)

    def test_jitter_is_reproducible_from_the_seed(self) -> None:
        voice = sine(440.0, 0.5)
        kwargs = {"rate_hz": 5.0, "am_depth": 0.2, "vibrato_cents": 20.0, "jitter": 0.7}
        assert np.array_equal(
            tremble(voice, RATE, seed=11, **kwargs),  # type: ignore[arg-type]
            tremble(voice, RATE, seed=11, **kwargs),  # type: ignore[arg-type]
        )
        assert not np.array_equal(
            tremble(voice, RATE, seed=11, **kwargs),  # type: ignore[arg-type]
            tremble(voice, RATE, seed=12, **kwargs),  # type: ignore[arg-type]
        )

    def test_an_out_of_range_jitter_is_refused(self) -> None:
        with pytest.raises(ValueError, match="jitter"):
            tremble(
                sine(440.0, 0.1), RATE, rate_hz=5.0, am_depth=0.1, vibrato_cents=0.0, jitter=2.0
            )


class TestYoungPreset:
    """Not an ageing preset. It exists so two personas can share one Timbre voice."""

    def test_it_pushes_pitch_and_rate_the_other_way(self) -> None:
        young = PRESETS_BY_NAME["young"]
        assert young.pitch > 1.0
        assert young.speed > 1.0

    def test_it_does_not_tremble_or_breathe(self) -> None:
        """He is 27 and he is the crisp one; that contrast is the point of the relay."""
        young = PRESETS_BY_NAME["young"]
        assert young.tremble_depth == 0.0
        assert young.vibrato_cents == 0.0
        assert young.inhale_ms == 0.0

    def test_it_raises_the_fundamental(self) -> None:
        out = PRESETS_BY_NAME["young"].apply(sine(200.0, 1.0), RATE)
        assert dominant_hz(out) > 205.0

    def test_it_shortens_the_clip(self) -> None:
        voice = sine(200.0, 1.0)
        assert PRESETS_BY_NAME["young"].apply(voice, RATE).size < voice.size

    def test_it_is_more_than_two_semitones_from_the_elder(self) -> None:
        """Roughly where two adult male voices stop sounding like one voice.

        Measured on the real Devansh render: 133.3 Hz against 151.9 Hz, +2.26 semitones,
        with the grandson 20% faster. Against `classic` or bare `none` the gap was 1.6
        semitones, which is inside one speaker's own range.
        """
        gap = 12 * math.log2(PRESETS_BY_NAME["young"].pitch / PRESETS_BY_NAME["frail"].pitch)
        assert gap > 2.0

    def test_a_pitch_beyond_the_declared_ceiling_is_refused(self) -> None:
        with pytest.raises(ValueError):
            AgingPreset(name="chipmunk", pitch=1.6, speed=1.0)
