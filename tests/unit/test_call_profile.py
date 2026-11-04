"""The composed chain: voice, then room, then line — and that order specifically."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.audio.ambience import AMBIENCE_BY_NAME, AmbienceBed, RoomAcoustics
from app.core.audio.call_profile import (
    CALL_PROFILES,
    DEFAULT_ELDER_PROFILE,
    GRANDSON_PROFILE,
    PROFILES_BY_NAME,
    CallProfile,
)

RATE = 24000


def voice(seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE), dtype=np.float32) / RATE
    envelope = (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)).astype(np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 240 * t) + 0.2 * np.sin(2 * np.pi * 1400 * t)
    return (tone * envelope).astype(np.float32)


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def band_energy(x: np.ndarray, lo: float, hi: float, rate: int = RATE) -> float:
    spectrum = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(x.size, 1 / rate)
    return float(spectrum[(freqs >= lo) & (freqs < hi)].sum())


class TestRegistry:
    def test_names_are_unique_and_every_profile_is_addressable(self) -> None:
        everything = (*CALL_PROFILES, GRANDSON_PROFILE)
        names = [profile.name for profile in everything]
        assert len(names) == len(set(names))
        assert {p.name: p for p in everything} == PROFILES_BY_NAME

    def test_the_ladder_is_the_listening_order_and_excludes_the_grandson(self) -> None:
        """CALL_PROFILES exists to be played through in order; he is not a rung on it."""
        assert GRANDSON_PROFILE not in CALL_PROFILES
        assert [p.name for p in CALL_PROFILES] == [
            "studio",
            "aged-dry",
            "landline",
            "living-room",
            "mobile",
            "bad-line",
        ]

    def test_the_default_elder_profile_exists(self) -> None:
        assert DEFAULT_ELDER_PROFILE in PROFILES_BY_NAME

    def test_the_ladder_starts_from_an_untouched_control(self) -> None:
        """The audition has to be able to show what each stage contributes."""
        studio = PROFILES_BY_NAME["studio"]
        assert studio.acoustics is None
        assert studio.ambience is None
        assert studio.aging.name == "none"
        assert not studio.telephony.band

    def test_the_finished_profiles_have_all_three_stages(self) -> None:
        for name in ("living-room", "mobile", "bad-line"):
            profile = PROFILES_BY_NAME[name]
            assert profile.aging.name != "none"
            assert profile.acoustics is not None
            assert profile.ambience is not None
            assert profile.telephony.band

    def test_the_grandson_shares_the_room_but_is_not_aged(self) -> None:
        elder = PROFILES_BY_NAME[DEFAULT_ELDER_PROFILE]
        assert GRANDSON_PROFILE.aging.name == "none"
        assert GRANDSON_PROFILE.ambience == elder.ambience
        assert GRANDSON_PROFILE.telephony == elder.telephony


class TestRender:
    @pytest.mark.parametrize("profile", CALL_PROFILES, ids=lambda p: p.name)
    def test_every_profile_renders_usable_audio(self, profile: CallProfile) -> None:
        result = profile.render(voice(), RATE)
        assert result.audio.size > 0
        assert np.all(np.isfinite(result.audio))
        assert float(np.max(np.abs(result.audio))) <= 1.0
        assert result.sample_rate == RATE
        assert result.seconds > 0.0

    @pytest.mark.parametrize("profile", CALL_PROFILES, ids=lambda p: p.name)
    def test_every_profile_is_deterministic(self, profile: CallProfile) -> None:
        """The TTS cache is content-hashed; a profile that varied would void it."""
        first = profile.render(voice(0.6), RATE)
        second = profile.render(voice(0.6), RATE)
        assert np.array_equal(first.audio, second.audio)
        assert first.dropped_packets == second.dropped_packets

    def test_it_reports_a_timing_for_every_stage(self) -> None:
        result = PROFILES_BY_NAME["mobile"].render(voice(), RATE)
        assert set(result.stage_ms) == {"voice", "room", "line"}
        assert all(value >= 0.0 for value in result.stage_ms.values())
        assert result.total_ms == pytest.approx(sum(result.stage_ms.values()))

    def test_lead_padding_is_added_around_the_voice(self) -> None:
        profile = PROFILES_BY_NAME["studio"]
        plain = profile.render(voice(0.5), RATE)
        padded = profile.render(voice(0.5), RATE, lead_ms=200.0)
        assert padded.audio.size == pytest.approx(plain.audio.size + 2 * 0.2 * RATE, rel=0.02)

    def test_the_bed_is_audible_in_the_lead_before_any_speech(self) -> None:
        result = PROFILES_BY_NAME["mobile"].render(voice(0.5), RATE, lead_ms=300.0)
        assert rms(result.audio[: int(0.2 * RATE)]) > 0.0

    def test_a_profile_without_ambience_leaves_the_lead_silent(self) -> None:
        result = PROFILES_BY_NAME["landline"].render(voice(0.5), RATE, lead_ms=300.0)
        assert rms(result.audio[: int(0.2 * RATE)]) == pytest.approx(0.0, abs=1e-6)

    def test_the_bed_offset_changes_the_room_but_not_the_voice(self) -> None:
        profile = PROFILES_BY_NAME["mobile"]
        assert not np.array_equal(
            profile.render(voice(0.5), RATE, bed_offset=0).audio,
            profile.render(voice(0.5), RATE, bed_offset=7000).audio,
        )

    def test_a_prebuilt_bed_is_reused(self) -> None:
        """Building a bed costs real time; a live call builds one and keeps it."""
        profile = PROFILES_BY_NAME["mobile"]
        assert profile.ambience is not None
        bed = AmbienceBed(profile.ambience, RATE)
        with_bed = profile.render(voice(0.5), RATE, bed=bed)
        without = profile.render(voice(0.5), RATE)
        assert np.array_equal(with_bed.audio, without.audio)

    def test_a_bed_built_for_another_room_is_rejected_not_used(self) -> None:
        """Silently accepting it would put the wrong room behind the voice."""
        profile = PROFILES_BY_NAME["mobile"]
        wrong = AmbienceBed(AMBIENCE_BY_NAME["quiet-room"], RATE)
        assert np.array_equal(
            profile.render(voice(0.5), RATE, bed=wrong).audio,
            profile.render(voice(0.5), RATE).audio,
        )

    def test_a_real_line_band_limits_the_finished_mix(self) -> None:
        result = PROFILES_BY_NAME["mobile"].render(voice(), RATE, lead_ms=200.0)
        total = band_energy(result.audio, 0.0, RATE / 2) + 1e-30
        # The mains hum in the bed lives below 300 Hz and must not survive the carrier.
        assert band_energy(result.audio, 0.0, 250.0) / total < 0.05

    def test_the_room_is_applied_before_the_line_not_after(self) -> None:
        """If the bed were mixed after the carrier filter, its sub-300 Hz content would
        survive into the output. This is the ordering bug the module exists to prevent."""
        profile = PROFILES_BY_NAME["mobile"]
        assert profile.ambience is not None and profile.ambience.hum > 0.0
        result = profile.render(voice(), RATE, lead_ms=400.0)
        lead = result.audio[: int(0.3 * RATE)]
        total = band_energy(lead, 0.0, RATE / 2) + 1e-30
        assert band_energy(lead, 0.0, 250.0) / total < 0.1

    def test_a_poor_line_drops_packets_and_a_clean_one_does_not(self) -> None:
        long_voice = voice(8.0)
        assert PROFILES_BY_NAME["bad-line"].render(long_voice, RATE).dropped_packets > 0
        assert PROFILES_BY_NAME["landline"].render(long_voice, RATE).dropped_packets == 0

    def test_overriding_the_room_keeps_everything_else(self) -> None:
        """How a different room is auditioned: copy the profile, not edit the registry."""
        profile = PROFILES_BY_NAME["mobile"]
        quiet = profile.model_copy(update={"ambience": AMBIENCE_BY_NAME["quiet-room"]})
        assert quiet.aging == profile.aging
        assert quiet.telephony == profile.telephony
        assert not np.array_equal(
            quiet.render(voice(0.5), RATE, lead_ms=300.0).audio,
            profile.render(voice(0.5), RATE, lead_ms=300.0).audio,
        )

    def test_a_dry_profile_differs_from_a_reverberant_one(self) -> None:
        profile = PROFILES_BY_NAME["landline"]
        wet = profile.model_copy(update={"acoustics": RoomAcoustics(mix=0.5)})
        assert not np.array_equal(
            wet.render(voice(0.5), RATE).audio, profile.render(voice(0.5), RATE).audio
        )

    def test_an_empty_utterance_does_not_crash(self) -> None:
        result = PROFILES_BY_NAME["mobile"].render(np.zeros(0, dtype=np.float32), RATE)
        assert result.audio.size == 0
