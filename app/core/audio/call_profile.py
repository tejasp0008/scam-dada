"""What the scammer actually hears, assembled in the order physics assembles it.

Three stages, each in its own module, and the only thing that matters here is that they run
in this order:

1. **The voice** (`aging`) — the larynx. Pitch, tempo, tremble, breath.
2. **The room** (`ambience`) — reverb puts the voice in a place, then the bed adds the other
   things in that place: the fan, the mains hum, the street.
3. **The line** (`telephony`) — the handset capsule, the carrier's AGC and 300-3400 Hz
   filter, 8 kHz sampling, mu-law companding, lost packets.

Getting the order wrong is audible, which is the whole reason this module exists rather than
a chain of calls at each call site:

* Room tone mixed *after* the carrier filter carries 50 Hz hum down a line that physically
  cannot pass it — it sounds like noise laid over a call rather than a call from a noisy room.
* Reverb applied *after* the bed reverberates the fan, which is already a diffuse recording
  of a whole room. The fan does not echo; it is the echo.
* Compression applied *after* companding pulls mu-law quantisation noise up into the gaps
  between words instead of leaving it buried under speech.

Pure maths on numpy arrays. Nothing here opens a file or a socket.
"""

from __future__ import annotations

import time

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from app.core.audio.aging import PRESETS_BY_NAME, AgingPreset, Signal, peak_normalise
from app.core.audio.ambience import (
    AMBIENCE_BY_NAME,
    AmbienceBed,
    AmbiencePreset,
    RoomAcoustics,
    room_reverb,
)
from app.core.audio.telephony import TELEPHONY_BY_NAME, TelephonyPreset


class RenderResult(BaseModel):
    """Rendered audio, plus what happened on the way.

    The per-stage timings are here because "can this run in the live path" is a question
    about one stage (the WSOLA stretch) rather than about the chain, and answering it from a
    real render beats answering it from an estimate.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    audio: Signal
    sample_rate: int
    dropped_packets: int
    stage_ms: dict[str, float]

    @property
    def seconds(self) -> float:
        return self.audio.size / self.sample_rate

    @property
    def total_ms(self) -> float:
        return sum(self.stage_ms.values())


class CallProfile(BaseModel):
    """A complete voice: a character, a room and a phone line."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    aging: AgingPreset
    telephony: TelephonyPreset
    acoustics: RoomAcoustics | None = Field(
        default=None, description="Room reverb. None leaves the voice dry."
    )
    ambience: AmbiencePreset | None = Field(
        default=None, description="Room tone bed. None leaves the line silent between words."
    )
    output_peak: float = Field(
        default=0.89,
        gt=0.0,
        le=1.0,
        description="Final peak. Lowering it makes the whole call quieter, but note that the "
        "receiving handset's AGC will largely undo that - it lifts voice and room together, "
        "so the ratio survives the channel and the absolute level does not. The knob that "
        "actually buries synthesis artefacts is the bed's level_db.",
    )

    def with_room(
        self, ambience: AmbiencePreset | None, *, level_db: float | None = None
    ) -> CallProfile:
        """A copy with a different room, or a different room *level*, and nothing else changed.

        How a room is auditioned: the registry above stays the single statement of what each
        profile is, and variations are copies rather than edits.
        """
        room = ambience
        if room is not None and level_db is not None:
            room = room.model_copy(update={"level_db": level_db})
        return self.model_copy(update={"ambience": room})

    def render(
        self,
        voice: Signal,
        sample_rate: int,
        *,
        bed: AmbienceBed | None = None,
        bed_offset: int = 0,
        lead_ms: float = 0.0,
    ) -> RenderResult:
        """Run the whole chain on one utterance.

        `bed` lets a caller pass a bed it already built: generating one costs a few hundred
        milliseconds, and in a live call it is built once and read positionally for the whole
        call, not rebuilt per reply.

        `lead_ms` pads silence around the voice so the bed is audible on its own before the
        first word. Audition clips want it; the live path does not, because there the bed
        already runs continuously between utterances and padding is pure added latency.
        """
        timings: dict[str, float] = {"voice": 0.0, "room": 0.0, "line": 0.0}
        if voice.size == 0:
            # No utterance, no output. A silent reply must not become a burst of room tone;
            # between utterances the live mixer emits the bed on its own.
            return RenderResult(
                audio=voice, sample_rate=sample_rate, dropped_packets=0, stage_ms=timings
            )

        began = time.perf_counter()
        out = self.aging.apply(voice, sample_rate)
        timings["voice"] = (time.perf_counter() - began) * 1000

        began = time.perf_counter()
        if self.acoustics is not None:
            out = room_reverb(out, sample_rate, self.acoustics)
        if lead_ms > 0.0:
            pad = np.zeros(int(lead_ms / 1000.0 * sample_rate), dtype=np.float32)
            out = np.concatenate([pad, out, pad])
        if self.ambience is not None:
            # Build one if the caller did not, but never silently ignore a bed from a
            # different room. The comparison is on the generator fields only, so a bed is
            # reused across levels - the level is applied here, at mix time.
            if bed is None or bed.preset.generator_key() != self.ambience.generator_key():
                bed = AmbienceBed(self.ambience, sample_rate)
            out = bed.mix_into(out, bed_offset, level_db=self.ambience.level_db)
        timings["room"] = (time.perf_counter() - began) * 1000

        began = time.perf_counter()
        out, dropped = self.telephony.apply(out, sample_rate)
        out = peak_normalise(out, self.output_peak)
        timings["line"] = (time.perf_counter() - began) * 1000

        return RenderResult(
            audio=out, sample_rate=sample_rate, dropped_packets=dropped, stage_ms=timings
        )


def _aging(name: str) -> AgingPreset:
    return PRESETS_BY_NAME[name]


def _ambience(name: str, *, level_db: float | None = None) -> AmbiencePreset:
    preset = AMBIENCE_BY_NAME[name]
    return preset if level_db is None else preset.model_copy(update={"level_db": level_db})


#: How far the room sits under the voice on the persona profiles.
#:
#: Closer than the -22 dB the `fan-and-street` preset carries by default, and deliberately so:
#: a voice that sits low against a noisy room is harder to examine, and synthesis artefacts
#: that survive the codec are masked by the fan rather than exposed by silence. This is the
#: knob that does that work. Lowering `output_peak` instead does almost nothing, because the
#: receiving handset's AGC lifts the voice and the room together.
#:
#: The limit is the scammer, not the audio: past roughly -10 dB he starts saying "hello, I
#: cannot hear you", which raises hang-up risk rather than lowering suspicion.
PERSONA_BED_DB = -15.0

#: A quiet talker. Mostly cosmetic for the reason in `output_peak`'s description, but it also
#: leaves headroom so the room tone is never the thing that clips.
PERSONA_OUTPUT_PEAK = 0.62


def _line(name: str) -> TelephonyPreset:
    return TELEPHONY_BY_NAME[name]


#: A ladder from "untouched TTS" to "bad mobile line", each step adding one thing, so the
#: audition shows what every stage contributes instead of only the finished result.
#:
#: `fan-and-street` is the room because that is the one that was chosen by ear. Override it
#: per render with `model_copy(update={"ambience": ...})` rather than editing this list.
CALL_PROFILES: tuple[CallProfile, ...] = (
    CallProfile(name="studio", aging=_aging("none"), telephony=_line("studio")),
    CallProfile(name="aged-dry", aging=_aging("classic"), telephony=_line("studio")),
    # The line on its own. Kept as a rung so the ladder still isolates one change at a time,
    # even though nothing uses it as a persona voice.
    CallProfile(name="landline-bare", aging=_aging("classic"), telephony=_line("landline")),
    CallProfile(
        name="landline",
        aging=_aging("classic"),
        telephony=_line("landline"),
        acoustics=RoomAcoustics(),
        ambience=_ambience("fan-and-street", level_db=PERSONA_BED_DB),
        output_peak=PERSONA_OUTPUT_PEAK,
    ),
    CallProfile(
        name="mobile",
        aging=_aging("classic"),
        telephony=_line("mobile"),
        acoustics=RoomAcoustics(),
        ambience=_ambience("fan-and-street", level_db=PERSONA_BED_DB),
        output_peak=PERSONA_OUTPUT_PEAK,
    ),
    CallProfile(
        name="bad-line",
        aging=_aging("frail"),
        telephony=_line("mobile-poor"),
        acoustics=RoomAcoustics(reverb_ms=340.0, mix=0.2),
        ambience=_ambience("fan-and-street", level_db=PERSONA_BED_DB),
        output_peak=PERSONA_OUTPUT_PEAK,
    ),
)

#: What Dada sounds like unless something says otherwise. Chosen by ear: the landline
#: channel, in a room with a fan and a street outside.
DEFAULT_ELDER_PROFILE = "landline"

#: The grandson is 27 and in the same room, so he shares the room and the line but is not
#: aged. He is not part of the ladder above — that exists to be listened through in order —
#: but he is addressable by name like any other profile.
GRANDSON_PROFILE = CallProfile(
    name="grandson",
    aging=_aging("none"),
    telephony=_line("landline"),
    acoustics=RoomAcoustics(),
    ambience=_ambience("fan-and-street", level_db=PERSONA_BED_DB),
    output_peak=PERSONA_OUTPUT_PEAK,
)

PROFILES_BY_NAME: dict[str, CallProfile] = {
    profile.name: profile for profile in (*CALL_PROFILES, GRANDSON_PROFILE)
}
