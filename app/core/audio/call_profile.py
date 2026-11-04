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

from app.core.audio.aging import PRESETS_BY_NAME, AgingPreset, Signal
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
            # Build one if the caller did not, but never silently ignore a mismatched bed:
            # a bed generated for another preset would be a different room.
            if bed is None or bed.preset != self.ambience:
                bed = AmbienceBed(self.ambience, sample_rate)
            out = bed.mix_into(out, bed_offset)
        timings["room"] = (time.perf_counter() - began) * 1000

        began = time.perf_counter()
        out, dropped = self.telephony.apply(out, sample_rate)
        timings["line"] = (time.perf_counter() - began) * 1000

        return RenderResult(
            audio=out, sample_rate=sample_rate, dropped_packets=dropped, stage_ms=timings
        )


def _aging(name: str) -> AgingPreset:
    return PRESETS_BY_NAME[name]


def _ambience(name: str) -> AmbiencePreset:
    return AMBIENCE_BY_NAME[name]


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
    CallProfile(name="landline", aging=_aging("classic"), telephony=_line("landline")),
    CallProfile(
        name="living-room",
        aging=_aging("classic"),
        telephony=_line("landline"),
        acoustics=RoomAcoustics(),
        ambience=_ambience("fan-and-street"),
    ),
    CallProfile(
        name="mobile",
        aging=_aging("classic"),
        telephony=_line("mobile"),
        acoustics=RoomAcoustics(),
        ambience=_ambience("fan-and-street"),
    ),
    CallProfile(
        name="bad-line",
        aging=_aging("frail"),
        telephony=_line("mobile-poor"),
        acoustics=RoomAcoustics(reverb_ms=340.0, mix=0.2),
        ambience=_ambience("fan-and-street"),
    ),
)

#: What Dada sounds like unless something says otherwise.
DEFAULT_ELDER_PROFILE = "mobile"

#: The grandson is 27 and in the same room, so he shares the room and the line but is not
#: aged. He is not part of the ladder above — that exists to be listened through in order —
#: but he is addressable by name like any other profile.
GRANDSON_PROFILE = CallProfile(
    name="grandson",
    aging=_aging("none"),
    telephony=_line("mobile"),
    acoustics=RoomAcoustics(),
    ambience=_ambience("fan-and-street"),
)

PROFILES_BY_NAME: dict[str, CallProfile] = {
    profile.name: profile for profile in (*CALL_PROFILES, GRANDSON_PROFILE)
}
