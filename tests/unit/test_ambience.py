"""Room tone: the right level, the right band, and the same bytes every time."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.audio.ambience import (
    AMBIENCE_BY_NAME,
    AMBIENCE_PRESETS,
    MAINS_HZ,
    AmbienceBed,
    AmbiencePreset,
    RoomAcoustics,
    ceiling_fan,
    mains_hum,
    room_floor,
    room_reverb,
    street,
)

RATE = 24000


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def band_energy(samples: np.ndarray, low: float, high: float, rate: int = RATE) -> float:
    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
    freqs = np.fft.rfftfreq(samples.size, 1 / rate)
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(spectrum[mask] ** 2))


def voice(seconds: float = 1.0, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (0.4 * np.sin(2 * np.pi * 240 * t)).astype(np.float32)


class TestLayers:
    def test_every_layer_comes_out_at_unit_rms(self) -> None:
        """Preset weights are only comparable if the layers arrive at the same level."""
        rng = np.random.default_rng(1)
        for layer in (
            ceiling_fan(RATE, RATE, rpm=280.0, rng=rng),
            mains_hum(RATE, RATE),
            street(RATE, RATE, rng=rng),
            room_floor(RATE, RATE, rng=rng),
        ):
            assert rms(layer) == pytest.approx(1.0, rel=1e-6)

    def test_the_fan_is_chopped_at_the_blade_rate(self) -> None:
        fan = ceiling_fan(RATE * 2, RATE, rpm=280.0, rng=np.random.default_rng(2))
        # The chop lives in the envelope, not the spectrum, so look at the envelope.
        envelope = np.abs(fan).astype(np.float64)
        envelope -= envelope.mean()
        spectrum = np.abs(np.fft.rfft(envelope))
        freqs = np.fft.rfftfreq(envelope.size, 1 / RATE)
        peak = freqs[int(np.argmax(spectrum[freqs < 40.0]))]
        assert peak == pytest.approx(280.0 / 60.0 * 3.0, rel=0.08)

    def test_the_fan_is_low_passed(self) -> None:
        fan = ceiling_fan(RATE, RATE, rpm=280.0, rng=np.random.default_rng(3))
        assert band_energy(fan, 0.0, 1000.0) > band_energy(fan, 2000.0, 12000.0) * 20

    def test_the_hum_sits_on_mains_harmonics(self) -> None:
        hum = mains_hum(RATE, RATE)
        spectrum = np.abs(np.fft.rfft(hum.astype(np.float64)))
        freqs = np.fft.rfftfreq(hum.size, 1 / RATE)
        peak = float(freqs[int(np.argmax(spectrum))])
        assert peak == pytest.approx(MAINS_HZ, abs=2.0)

    def test_the_hum_needs_no_seed(self) -> None:
        assert np.array_equal(mains_hum(1000, RATE), mains_hum(1000, RATE))


class TestBed:
    def test_the_loop_is_unit_rms_and_the_declared_length(self) -> None:
        bed = AmbienceBed(AMBIENCE_BY_NAME["ceiling-fan"], RATE)
        assert rms(bed.loop) == pytest.approx(1.0, rel=1e-5)
        # The cross-fade folds one fade length back into the head, so the loop is shorter
        # than loop_seconds by exactly that much.
        assert bed.loop.size == pytest.approx(12.0 * RATE, rel=0.05)

    def test_take_wraps_around_the_loop(self) -> None:
        bed = AmbienceBed(AmbiencePreset(name="t", floor=1.0, loop_seconds=1.5), RATE)
        size = bed.loop.size
        wrapped = bed.take(100, offset=size - 50)
        assert np.array_equal(wrapped[:50], bed.loop[size - 50 :])
        assert np.array_equal(wrapped[50:], bed.loop[:50])

    def test_take_of_nothing_is_empty(self) -> None:
        bed = AmbienceBed(AmbiencePreset(name="t", floor=1.0, loop_seconds=1.5), RATE)
        assert bed.take(0).size == 0

    def test_the_loop_seam_does_not_click(self) -> None:
        """Without the cross-fade this is a tick once every loop_seconds, forever."""
        bed = AmbienceBed(AMBIENCE_BY_NAME["ceiling-fan"], RATE)
        two = bed.take(bed.loop.size * 2)
        steps = np.abs(np.diff(two.astype(np.float64)))
        seam = steps[bed.loop.size - 2 : bed.loop.size + 2].max()
        assert seam <= np.percentile(steps, 99.9)

    def test_a_bed_is_deterministic_across_instances(self) -> None:
        preset = AMBIENCE_BY_NAME["fan-and-street"]
        assert np.array_equal(AmbienceBed(preset, RATE).loop, AmbienceBed(preset, RATE).loop)

    def test_a_different_seed_gives_a_different_room(self) -> None:
        base = AMBIENCE_BY_NAME["ceiling-fan"]
        other = base.model_copy(update={"seed": base.seed + 1})
        assert not np.array_equal(AmbienceBed(base, RATE).loop, AmbienceBed(other, RATE).loop)


class TestMixing:
    @pytest.mark.parametrize("level_db", [-32.0, -26.0, -18.0])
    def test_the_bed_arrives_at_the_requested_level_under_the_voice(self, level_db: float) -> None:
        preset = AmbiencePreset(name="t", floor=1.0, level_db=level_db, loop_seconds=2.0)
        bed = AmbienceBed(preset, RATE)
        source = voice()
        mixed = bed.mix_into(source)
        added = mixed - source
        assert 20 * np.log10(rms(added) / rms(source)) == pytest.approx(level_db, abs=0.5)

    def test_mixing_is_positional_so_a_call_can_run_continuously(self) -> None:
        bed = AmbienceBed(AMBIENCE_BY_NAME["ceiling-fan"], RATE)
        source = voice(0.5)
        assert not np.array_equal(bed.mix_into(source, 0), bed.mix_into(source, 5000))

    def test_silence_is_left_alone(self) -> None:
        """A persona that said nothing must not emit a burst of room tone on its own."""
        bed = AmbienceBed(AMBIENCE_BY_NAME["ceiling-fan"], RATE)
        silence = np.zeros(1000, dtype=np.float32)
        assert np.array_equal(bed.mix_into(silence), silence)

    def test_an_empty_signal_survives(self) -> None:
        bed = AmbienceBed(AMBIENCE_BY_NAME["quiet-room"], RATE)
        assert bed.mix_into(np.zeros(0, dtype=np.float32)).size == 0


class TestPresets:
    def test_names_are_unique_and_indexed(self) -> None:
        names = [preset.name for preset in AMBIENCE_PRESETS]
        assert len(names) == len(set(names))
        assert {p.name: p for p in AMBIENCE_PRESETS} == AMBIENCE_BY_NAME

    @pytest.mark.parametrize("preset", AMBIENCE_PRESETS, ids=lambda p: p.name)
    def test_every_preset_has_at_least_one_layer(self, preset: AmbiencePreset) -> None:
        assert preset.layers()

    @pytest.mark.parametrize("preset", AMBIENCE_PRESETS, ids=lambda p: p.name)
    def test_every_preset_sits_below_the_voice(self, preset: AmbiencePreset) -> None:
        """A bed at or above the voice would cost transcription accuracy, not add realism."""
        assert preset.level_db <= -14.0

    def test_layers_reports_only_what_is_switched_on(self) -> None:
        preset = AmbiencePreset(name="t", fan=0.5, floor=0.25)
        assert dict(preset.layers()) == {"fan": 0.5, "floor": 0.25}

    def test_a_positive_level_is_refused(self) -> None:
        with pytest.raises(ValueError):
            AmbiencePreset(name="t", floor=1.0, level_db=6.0)


class TestRoomAcoustics:
    def test_the_impulse_response_starts_with_the_direct_sound(self) -> None:
        response = RoomAcoustics().impulse_response(RATE)
        assert float(np.abs(response[0])) == float(np.max(np.abs(response)))

    def test_the_response_decays(self) -> None:
        response = RoomAcoustics(reverb_ms=300.0).impulse_response(RATE)
        third = response.size // 3
        assert rms(response[:third]) > rms(response[-third:]) * 3

    def test_the_response_is_the_requested_length(self) -> None:
        response = RoomAcoustics(reverb_ms=250.0).impulse_response(RATE)
        assert response.size == pytest.approx(0.25 * RATE, rel=0.01)

    def test_it_is_seeded(self) -> None:
        acoustics = RoomAcoustics()
        assert np.array_equal(acoustics.impulse_response(RATE), acoustics.impulse_response(RATE))

    def test_reverb_keeps_the_length_and_the_level(self) -> None:
        """A reverb that changes either would break the bed level and the clip timing."""
        source = voice(1.0)
        wet = room_reverb(source, RATE, RoomAcoustics())
        assert wet.size == source.size
        assert rms(wet) == pytest.approx(rms(source), rel=0.2)

    def test_reverb_actually_changes_the_signal(self) -> None:
        source = voice(0.5)
        assert not np.array_equal(room_reverb(source, RATE, RoomAcoustics()), source)

    def test_reverb_smears_a_click_forwards_in_time(self) -> None:
        """The audible point of reverb: energy arrives after the sound that caused it."""
        click = np.zeros(RATE, dtype=np.float32)
        click[100] = 1.0
        wet = room_reverb(click, RATE, RoomAcoustics(mix=0.9))
        assert rms(wet[200:3000]) > 0.0
        assert rms(click[200:3000]) == 0.0

    def test_a_zero_mix_is_a_no_op(self) -> None:
        source = voice(0.3)
        assert np.array_equal(room_reverb(source, RATE, RoomAcoustics(mix=0.0)), source)

    def test_an_empty_signal_survives(self) -> None:
        empty = np.zeros(0, dtype=np.float32)
        assert room_reverb(empty, RATE, RoomAcoustics()).size == 0

    def test_a_room_longer_than_a_hall_is_refused(self) -> None:
        with pytest.raises(ValueError):
            RoomAcoustics(reverb_ms=5000.0)
