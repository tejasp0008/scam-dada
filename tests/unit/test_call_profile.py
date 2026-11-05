"""The composed chain: voice, then room, then line — and that order specifically."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.audio.ambience import AMBIENCE_BY_NAME, AmbienceBed, RoomAcoustics
from app.core.audio.call_profile import (
    BAD_LINE_BED_DB,
    CALL_PROFILES,
    DEFAULT_ELDER_PROFILE,
    GRANDSON_PROFILE,
    PERSONA_BED_DB,
    PERSONA_OUTPUT_PEAK,
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
            "landline-bare",
            "landline",
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
        for name in ("landline", "mobile", "bad-line"):
            profile = PROFILES_BY_NAME[name]
            assert profile.aging.name != "none"
            assert profile.acoustics is not None
            assert profile.ambience is not None
            assert profile.telephony.band

    def test_the_grandson_differs_from_the_elder_in_exactly_one_respect(self) -> None:
        """He is handed the same handset in the same room: only the larynx changes.

        Written as a whole-object comparison rather than field by field, so a stage added to
        CallProfile later cannot quietly diverge between the two personas.
        """
        elder = PROFILES_BY_NAME[DEFAULT_ELDER_PROFILE]
        assert GRANDSON_PROFILE.aging != elder.aging
        rejoined = GRANDSON_PROFILE.model_copy(update={"name": elder.name, "aging": elder.aging})
        assert rejoined == elder


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
        result = PROFILES_BY_NAME["landline-bare"].render(voice(0.5), RATE, lead_ms=300.0)
        assert rms(result.audio[: int(0.2 * RATE)]) == pytest.approx(0.0, abs=1e-6)

    def test_the_default_elder_voice_has_a_room(self) -> None:
        """It was chosen by ear as the line *plus* the fan and the street, not the bare line."""
        profile = PROFILES_BY_NAME[DEFAULT_ELDER_PROFILE]
        assert profile.ambience is not None
        assert profile.acoustics is not None
        assert {name for name, _ in profile.ambience.layers()} >= {"fan", "street", "floor"}

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
        quiet = profile.with_room(AMBIENCE_BY_NAME["quiet-room"])
        assert quiet.aging == profile.aging
        assert quiet.telephony == profile.telephony
        assert quiet.acoustics == profile.acoustics
        assert not np.array_equal(
            quiet.render(voice(0.5), RATE, lead_ms=300.0).audio,
            profile.render(voice(0.5), RATE, lead_ms=300.0).audio,
        )

    def test_raising_the_bed_level_makes_the_room_louder(self) -> None:
        profile = PROFILES_BY_NAME["landline"]
        assert profile.ambience is not None
        quiet = profile.with_room(profile.ambience, level_db=-30.0)
        loud = profile.with_room(profile.ambience, level_db=-12.0)
        lead = int(0.4 * RATE)
        quiet_lead = rms(quiet.render(voice(0.5), RATE, lead_ms=600.0).audio[:lead])
        loud_lead = rms(loud.render(voice(0.5), RATE, lead_ms=600.0).audio[:lead])
        assert loud_lead > quiet_lead * 2

    def test_a_level_override_changes_nothing_but_the_level(self) -> None:
        profile = PROFILES_BY_NAME["landline"]
        assert profile.ambience is not None
        louder = profile.with_room(profile.ambience, level_db=-14.0)
        assert louder.ambience is not None
        assert louder.ambience.level_db == -14.0
        assert louder.ambience.model_copy(update={"level_db": profile.ambience.level_db}) == (
            profile.ambience
        )

    def test_dropping_the_room_is_possible(self) -> None:
        profile = PROFILES_BY_NAME["landline"]
        assert profile.with_room(None).ambience is None

    def test_a_dry_profile_differs_from_a_reverberant_one(self) -> None:
        profile = PROFILES_BY_NAME["landline"]
        wet = profile.model_copy(update={"acoustics": RoomAcoustics(mix=0.5)})
        assert not np.array_equal(
            wet.render(voice(0.5), RATE).audio, profile.render(voice(0.5), RATE).audio
        )

    def test_an_empty_utterance_does_not_crash(self) -> None:
        result = PROFILES_BY_NAME["mobile"].render(np.zeros(0, dtype=np.float32), RATE)
        assert result.audio.size == 0


class TestVoiceLoudness:
    def test_the_persona_profiles_sit_the_voice_low_against_the_room(self) -> None:
        """A voice low against a noisy room is harder to examine for synthesis artefacts."""
        for name in ("landline", "mobile"):
            profile = PROFILES_BY_NAME[name]
            assert profile.ambience is not None
            assert profile.ambience.level_db == PERSONA_BED_DB
            assert profile.output_peak == PERSONA_OUTPUT_PEAK

    def test_the_poor_line_carries_its_bed_further_back(self) -> None:
        """Not a fudge: its own AGC compresses at 0.85 and lifts the bed ~5 dB further than
        the cleaner lines do, so an equal nominal level is not an equal delivered level."""
        profile = PROFILES_BY_NAME["bad-line"]
        assert profile.ambience is not None
        assert profile.ambience.level_db == BAD_LINE_BED_DB
        assert BAD_LINE_BED_DB < PERSONA_BED_DB

    def test_the_persona_bed_is_closer_than_the_preset_default(self) -> None:
        assert AMBIENCE_BY_NAME["fan-and-street"].level_db < PERSONA_BED_DB

    def test_the_persona_bed_stops_short_of_drowning_the_voice(self) -> None:
        """Past about -10 dB the scammer says he cannot hear, which raises hang-up risk."""
        assert PERSONA_BED_DB <= -10.0

    def test_output_peak_is_honoured(self) -> None:
        profile = PROFILES_BY_NAME["landline"]
        result = profile.render(voice(), RATE, lead_ms=300.0)
        assert float(np.max(np.abs(result.audio))) == pytest.approx(profile.output_peak, rel=1e-4)

    def test_a_lower_output_peak_makes_the_whole_call_quieter(self) -> None:
        profile = PROFILES_BY_NAME["landline"]
        loud = profile.model_copy(update={"output_peak": 0.9})
        quiet = profile.model_copy(update={"output_peak": 0.3})
        assert rms(quiet.render(voice(), RATE).audio) < rms(loud.render(voice(), RATE).audio)

    def test_output_peak_does_not_change_the_voice_to_room_ratio(self) -> None:
        """Which is why it is cosmetic: the receiver's AGC lifts both back together."""
        profile = PROFILES_BY_NAME["landline"]
        ratios = []
        for peak in (0.3, 0.9):
            result = profile.model_copy(update={"output_peak": peak}).render(
                voice(), RATE, lead_ms=600.0
            )
            lead = rms(result.audio[: int(0.4 * RATE)])
            whole = rms(result.audio)
            ratios.append(lead / whole)
        assert ratios[0] == pytest.approx(ratios[1], rel=0.02)

    def test_an_output_peak_above_full_scale_is_refused(self) -> None:
        base = PROFILES_BY_NAME["landline"]
        with pytest.raises(ValueError):
            CallProfile(
                name="too-loud",
                aging=base.aging,
                telephony=base.telephony,
                output_peak=1.4,
            )


class TestBedReuse:
    def test_a_bed_is_reused_across_levels(self) -> None:
        """The loop does not depend on the level, so a level sweep rebuilds nothing."""
        profile = PROFILES_BY_NAME["landline"]
        assert profile.ambience is not None
        bed = AmbienceBed(profile.ambience, RATE)
        louder = profile.with_room(profile.ambience, level_db=-10.0)
        assert louder.ambience is not None
        assert bed.preset.generator_key() == louder.ambience.generator_key()
        # Reusing the bed must give the same answer as letting render build its own.
        assert np.array_equal(
            louder.render(voice(0.5), RATE, bed=bed, lead_ms=300.0).audio,
            louder.render(voice(0.5), RATE, lead_ms=300.0).audio,
        )

    def test_the_generator_key_ignores_the_level_but_not_the_layers(self) -> None:
        base = AMBIENCE_BY_NAME["fan-and-street"]
        assert base.generator_key() == base.model_copy(update={"level_db": -5.0}).generator_key()
        assert base.generator_key() != base.model_copy(update={"fan": 0.1}).generator_key()
        assert base.generator_key() != base.model_copy(update={"seed": 99}).generator_key()


class TestDeliveredRatio:
    """What comes out, as opposed to what the dial says.

    `level_db` sets the bed against the voice *before* the line. The carrier's AGC then lifts
    quiet passages - which are bed-only - toward its target, so the ratio that actually
    reaches the listener is several dB tighter than the nominal figure. That is what a real
    AGC does and why phone calls sound noisy, but it means the dial is not calibrated to the
    output and only a measurement is worth quoting.
    """

    @staticmethod
    def _delivered_snr(profile: CallProfile) -> float:
        result = profile.render(voice(3.0), RATE, lead_ms=800.0)
        audio = result.audio
        bed = audio[: int(0.4 * RATE)]
        body = audio[int(1.0 * RATE) : -int(1.0 * RATE)]
        hop = int(0.025 * RATE)
        frames = body[: (body.size // hop) * hop].reshape(-1, hop)
        energies = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        speech = float(energies[energies >= np.quantile(energies, 0.6)].mean())
        return float(20 * np.log10(speech / rms(bed)))

    def test_the_agc_tightens_the_ratio_below_the_nominal_level(self) -> None:
        profile = PROFILES_BY_NAME["landline"]
        assert profile.ambience is not None
        nominal = -profile.ambience.level_db
        assert self._delivered_snr(profile) < nominal

    def test_a_closer_bed_still_means_a_tighter_delivered_ratio(self) -> None:
        """Monotonic, even though not calibrated - the dial still points the right way."""
        profile = PROFILES_BY_NAME["landline"]
        assert profile.ambience is not None
        far = self._delivered_snr(profile.with_room(profile.ambience, level_db=-22.0))
        near = self._delivered_snr(profile.with_room(profile.ambience, level_db=-9.0))
        assert near < far

    def test_the_delivered_ratio_is_comparable_across_the_persona_profiles(self) -> None:
        """The thing worth holding constant is what the listener gets, not the dial.

        Each line's AGC lifts the bed by a different amount, so the nominal levels differ in
        order to make the delivered ratios land in the same region. Without this the choice of
        line would silently change how buried the voice is.
        """
        delivered = [
            self._delivered_snr(PROFILES_BY_NAME[name])
            for name in ("landline", "mobile", "bad-line")
        ]
        assert max(delivered) - min(delivered) < 6.0

    def test_the_voice_still_sits_above_the_room_on_the_chosen_profile(self) -> None:
        """`bad-line` was chosen deliberately and is the aggressive end of the ladder.

        The guard left here is only that the voice is still the loudest thing on the line. A
        negative ratio would mean the room is louder than the speaker, at which point no
        amount of scammer patience helps. On this synthetic tone the default measures ~2 dB;
        on the real Devansh render it measures 6.7 dB, because a tone with an envelope is not
        speech. Neither number is comfortable, and that is the recorded trade, not an
        accident — see DECISIONS 0021.
        """
        assert self._delivered_snr(PROFILES_BY_NAME[DEFAULT_ELDER_PROFILE]) > 0.0
