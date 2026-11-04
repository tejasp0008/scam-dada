"""The family relay, and the persona prompt files it points at.

The prompt tests are the mechanism behind the README's claim that the ethics constraints are
present and unmodified: the block's digest is pinned, so changing it is a deliberate act
that fails this suite until someone updates the constant on purpose.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from app.core.audio.call_profile import PROFILES_BY_NAME
from app.core.models import PersonaId
from app.core.personas import (
    ETHICS_PROMPT_FILE,
    LEAD_PERSONA,
    RELAY,
    PersonaProfile,
    next_in_relay,
    profile_for,
)
from app.core.tags import TIMBRE_V3_TAGS

PROMPT_DIR = Path(__file__).resolve().parents[2] / "app" / "prompts"

#: sha256 of app/prompts/ethics.v1.md. Update only alongside a deliberate ethics change.
ETHICS_DIGEST = "f41e1dc8f2f864c44ea38a0dd89622d1df5f9bd0b5bc30aa907f205874fce025"


class TestRelay:
    def test_dada_answers_the_phone(self) -> None:
        assert LEAD_PERSONA.persona_id is PersonaId.DADA
        assert next_in_relay(None) is LEAD_PERSONA

    def test_the_relay_runs_dada_then_grandson(self) -> None:
        assert [profile.persona_id for profile in RELAY] == [PersonaId.DADA, PersonaId.GRANDSON]

    def test_the_relay_ends_rather_than_wrapping(self) -> None:
        """Handing the scammer back to a voice he already heard breaks the illusion."""
        assert next_in_relay(PersonaId.DADA) is not None
        assert next_in_relay(PersonaId.GRANDSON) is None

    def test_walking_the_relay_visits_every_persona_once(self) -> None:
        seen: list[PersonaId] = []
        current: PersonaId | None = None
        while (profile := next_in_relay(current)) is not None:
            seen.append(profile.persona_id)
            current = profile.persona_id
        assert seen == [profile.persona_id for profile in RELAY]
        assert len(seen) == len(set(seen))

    def test_every_persona_id_has_a_profile(self) -> None:
        """Adding a PersonaId without a profile would fail at runtime, not here."""
        for persona_id in PersonaId:
            assert profile_for(persona_id).persona_id is persona_id

    def test_an_unregistered_persona_is_an_error_not_a_silent_none(self) -> None:
        with pytest.raises(ValueError, match="no profile registered"):
            profile_for("nani")  # type: ignore[arg-type]


class TestProfiles:
    @pytest.mark.parametrize("profile", RELAY, ids=lambda p: p.persona_id.value)
    def test_the_prompt_file_exists(self, profile: PersonaProfile) -> None:
        assert (PROMPT_DIR / profile.prompt_file).is_file()

    @pytest.mark.parametrize("profile", RELAY, ids=lambda p: p.persona_id.value)
    def test_the_tag_vocabulary_is_real(self, profile: PersonaProfile) -> None:
        assert profile.allowed_tags
        assert profile.allowed_tags <= TIMBRE_V3_TAGS

    @pytest.mark.parametrize("profile", RELAY, ids=lambda p: p.persona_id.value)
    def test_the_call_profile_named_actually_exists(self, profile: PersonaProfile) -> None:
        assert profile.call_profile in PROFILES_BY_NAME

    def test_the_elder_is_aged_and_the_grandson_is_not(self) -> None:
        """Timbre has no elderly voice, and a 27-year-old does not need one."""
        elder = PROFILES_BY_NAME[profile_for(PersonaId.DADA).call_profile]
        grandson = PROFILES_BY_NAME[profile_for(PersonaId.GRANDSON).call_profile]
        assert elder.aging.name != "none"
        assert grandson.aging.name == "none"

    def test_both_personas_are_in_the_same_room_on_the_same_line(self) -> None:
        """They are passing one handset between them, so the room cannot change."""
        elder = PROFILES_BY_NAME[profile_for(PersonaId.DADA).call_profile]
        grandson = PROFILES_BY_NAME[profile_for(PersonaId.GRANDSON).call_profile]
        assert elder.ambience == grandson.ambience
        assert elder.telephony == grandson.telephony

    def test_personas_do_not_share_a_voice_setting(self) -> None:
        settings = [profile.voice_setting for profile in RELAY]
        assert len(settings) == len(set(settings))

    def test_a_profile_with_an_invented_tag_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"not Timbre v3\.0 tags"):
            PersonaProfile(
                persona_id=PersonaId.DADA,
                display_name="x",
                prompt_file="dada.v1.md",
                allowed_tags=frozenset({"winks"}),
                call_profile="mobile",
                voice_setting="persona_elder_voice",
            )


class TestEthicsBlock:
    def test_the_constraint_block_is_unmodified(self) -> None:
        digest = hashlib.sha256((PROMPT_DIR / ETHICS_PROMPT_FILE).read_bytes()).hexdigest()
        assert digest == ETHICS_DIGEST, (
            "app/prompts/ethics.v1.md changed. If that was deliberate, update ETHICS_DIGEST "
            "in this test and say what changed in the commit message."
        )

    @pytest.mark.parametrize(
        "constraint",
        [
            "Never threaten",
            "Never insult",
            "Never claim to be police",
            "Never offer to call anyone back",
            "Never try to identify, locate, trace or look up any real person",
            "Never give a real identifier",
        ],
    )
    def test_each_promised_constraint_is_stated(self, constraint: str) -> None:
        """These six are the guardrails the project committed to in its README."""
        assert constraint in (PROMPT_DIR / ETHICS_PROMPT_FILE).read_text()


class TestPromptHygiene:
    @pytest.mark.parametrize("profile", RELAY, ids=lambda p: p.persona_id.value)
    def test_a_persona_only_asks_for_tags_it_is_allowed(self, profile: PersonaProfile) -> None:
        """A prompt that teaches a forbidden tag wastes billable characters on every reply."""
        text = (PROMPT_DIR / profile.prompt_file).read_text()
        backticked = set(re.findall(r"`<([^<>`]{1,24})>`", text))
        assert backticked <= profile.allowed_tags, backticked - profile.allowed_tags

    @pytest.mark.parametrize("profile", RELAY, ids=lambda p: p.persona_id.value)
    def test_a_persona_is_told_to_hide_that_it_is_a_machine(self, profile: PersonaProfile) -> None:
        assert "AI" in (PROMPT_DIR / profile.prompt_file).read_text()

    @pytest.mark.parametrize(
        "name", [profile.prompt_file for profile in RELAY] + [ETHICS_PROMPT_FILE]
    )
    def test_no_prompt_contains_anything_that_looks_like_a_real_identifier(self, name: str) -> None:
        """Synthetic data only, and a prompt is as much a fixture as a seed file is."""
        text = (PROMPT_DIR / name).read_text()
        assert not re.search(r"\b[6-9]\d{9}\b", text), "looks like an Indian mobile number"
        assert not re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", text), "looks like an Aadhaar number"
        assert not re.search(r"\b[A-Z]{5}\d{4}[A-Z]\b", text), "looks like a PAN"
        assert not re.search(r"\b[\w.]+@(?!demo\b)(ok\w+|paytm|ybl|upi|axl)\b", text), "live UPI"
