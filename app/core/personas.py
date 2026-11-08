"""Who can take the phone, in what order, and with what voice.

The relay is a list, not a chain of special cases. The orchestrator asks for the persona
after the current one and gets either a profile or `None`; it never names a persona. That is
what makes "add a second elder later" a three-line change — a new `PersonaId`, a new prompt
file, a new entry in `RELAY` — rather than a rewrite of the handoff logic.

Pure data. Prompt files are named here but read elsewhere, so `app.core` stays I/O-free.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.core.audio.call_profile import DEFAULT_ELDER_PROFILE, GRANDSON_PROFILE, PROFILES_BY_NAME
from app.core.models import PersonaId
from app.core.tags import ELDER_TAGS, GRANDSON_TAGS, TIMBRE_V3_TAGS

#: Prepended to every persona prompt. Kept in its own file so the ethics constraints are
#: literally the same bytes for every persona and a test can pin their digest.
ETHICS_PROMPT_FILE = "ethics.v1.md"


class PersonaProfile(BaseModel):
    """Everything needed to put one character on the call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    persona_id: PersonaId
    display_name: str = Field(description="Shown in the dashboard transcript.")
    prompt_file: str = Field(description="Filename under app/prompts/, read by the service layer.")
    filler_file: str = Field(
        description="Filename under app/prompts/ holding this persona's stall phrases, one "
        "per line. Separate from the prompt because these are pre-synthesised verbatim "
        "rather than sent to a model."
    )
    allowed_tags: frozenset[str] = Field(
        description="Timbre tags this persona may use. Anything else is stripped before "
        "synthesis, because unknown tags are billed and then silently discarded."
    )
    call_profile: str = Field(
        description="Name in app.core.audio.call_profile.PROFILES_BY_NAME. Bundles the voice "
        "ageing, the room and the phone line, because all three are part of how a character "
        "sounds and picking them separately produces mismatched combinations."
    )
    voice_setting: str = Field(
        description="Settings attribute holding the Timbre voice name. The voice is chosen "
        "by ear from an audition, so it is configuration, not a constant."
    )

    def model_post_init(self, _context: object) -> None:
        unknown = self.allowed_tags - TIMBRE_V3_TAGS
        if unknown:
            raise ValueError(f"{self.persona_id}: not Timbre v3.0 tags: {sorted(unknown)}")


#: Sent as a system note in last position on every dialogue turn.
#:
#: Two rules live here rather than in the prompts, for the same reason. The prompts name
#: Hinglish in their first paragraph and the model still answered a pure-English caller in
#: pure English, every turn, in both personas; and adding that section pushed the brevity
#: rule in `## Speech` far enough from the generation that replies grew 43%, from 94 to 134
#: characters, which on a live call is ten seconds before the caller gets a turn.
#:
#: Both are *form* rules, which is why they drift: the model is attending to what to say.
#: Stated last, immediately before the generation, they hold (DECISIONS 0031, 0032).
#:
#: The length is given as a *range*, not a maximum. A maximum alone collapsed the call: the
#: model trends to the floor, a clipped reply is recorded into the history, and from there it
#: imitates its own terse style - median reply length fell to 33 characters, which the
#: persona prompt rightly calls a broken machine rather than a slow man.
REPLY_STYLE_NOTE = (
    "Reply in Hinglish written in the English alphabet, whatever language the caller just "
    "used. If he spoke only English, you still answer in Hinglish. Never Devanagari. "
    "Two sentences, fifteen to twenty-five words - fewer sounds like a dropped line, more "
    "and he never gets a turn to speak."
)


#: Handoff order. The scammer meets these in sequence, and each handoff resets his patience
#: without resetting the shared call history.
RELAY: tuple[PersonaProfile, ...] = (
    PersonaProfile(
        persona_id=PersonaId.DADA,
        display_name="Dada",
        prompt_file="dada.v1.md",
        filler_file="fillers.dada.v1.txt",
        allowed_tags=ELDER_TAGS,
        call_profile=DEFAULT_ELDER_PROFILE,
        voice_setting="persona_elder_voice",
    ),
    PersonaProfile(
        persona_id=PersonaId.GRANDSON,
        display_name="Grandson",
        prompt_file="grandson.v1.md",
        filler_file="fillers.grandson.v1.txt",
        allowed_tags=GRANDSON_TAGS,
        call_profile=GRANDSON_PROFILE.name,
        voice_setting="persona_grandson_voice",
    ),
)

_BY_ID: dict[PersonaId, PersonaProfile] = {profile.persona_id: profile for profile in RELAY}

#: Who answers the phone.
LEAD_PERSONA: PersonaProfile = RELAY[0]


def profile_for(persona_id: PersonaId) -> PersonaProfile:
    """Look up a persona, raising if it is not in the relay."""
    try:
        return _BY_ID[persona_id]
    except KeyError:
        raise ValueError(f"no profile registered for {persona_id}") from None


def next_in_relay(current: PersonaId | None) -> PersonaProfile | None:
    """The persona after `current`, or `None` at the end of the relay.

    `None` in means "nobody has the phone yet", so the lead persona is returned. `None` out
    means the family has run out of people to pass the phone *forward* to.
    """
    if current is None:
        return LEAD_PERSONA
    order = [profile.persona_id for profile in RELAY]
    try:
        index = order.index(current)
    except ValueError:
        raise ValueError(f"no profile registered for {current}") from None
    if index + 1 >= len(RELAY):
        return None
    return RELAY[index + 1]


def hand_back(current: PersonaId | None) -> PersonaProfile | None:
    """Give the phone back to the persona before `current`, or None if there is none.

    An earlier version refused to go backwards, on the theory that returning the scammer to
    a voice he has already heard breaks the illusion. A live call showed the opposite: the
    grandson took the phone at turn 9 and still had it at turn 21, the scammer eventually
    said "aap apne dada ji se baat karwaiye" — put me on to your grandfather — and the system
    could not comply. Refusing the most natural thing in a family is what broke the illusion.

    It is also what the grandson's own prompt already promises: that he puts his grandfather
    back on when he runs out of questions. The lead persona is the one doing the baiting, so
    the phone belongs with him by default and the grandson is an interlude.
    """
    if current is None:
        return None
    order = [profile.persona_id for profile in RELAY]
    try:
        index = order.index(current)
    except ValueError:
        raise ValueError(f"no profile registered for {current}") from None
    return RELAY[index - 1] if index > 0 else None


def relay_collision(voices: dict[PersonaId, str]) -> str | None:
    """Explain why a voice assignment would make the relay inaudible, or return None.

    The relay only works because the scammer hears somebody else pick up the phone, so two
    personas have to be told apart. But "different Timbre voice" is the wrong test for that.

    **The same voice for both is legitimate** when their call profiles age them differently.
    Relatives share vocal timbre - a grandfather and grandson sounding related is more
    plausible than two unrelated strangers in one house - and Dada is pitched down 9%, slowed
    15%, given a tremble and a breath, which is a larger change than the gap between most
    pairs of voices in the catalogue.

    **What is never allowed is the same voice with the same ageing.** That is one person
    pretending to hand the phone to himself, and no prompt difference covers it.
    """
    rendered: dict[tuple[str, str], PersonaId] = {}
    for persona_id, voice in voices.items():
        name = voice.strip().casefold()
        if not name:
            continue  # voices are chosen by ear over several sessions; half-configured is normal
        profile = PROFILES_BY_NAME[profile_for(persona_id).call_profile]
        key = (name, profile.aging.name)
        if (clash := rendered.get(key)) is not None:
            return (
                f"{persona_id.value} and {clash.value} would both be {voice!r} with the "
                f"{profile.aging.name!r} ageing, so the relay handoff is inaudible"
            )
        rendered[key] = persona_id
    return None
