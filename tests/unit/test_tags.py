"""Audio tag filtering. Unknown tags are billed and then silently dropped by Timbre."""

from __future__ import annotations

import pytest

from app.core.tags import (
    ELDER_TAGS,
    GRANDSON_TAGS,
    TIMBRE_V3_TAGS,
    billable_characters,
    sanitize_tags,
    tidy_for_speech,
)


class TestCatalogue:
    def test_the_catalogue_is_the_documented_thirteen(self) -> None:
        assert len(TIMBRE_V3_TAGS) == 13

    @pytest.mark.parametrize("allowed", [ELDER_TAGS, GRANDSON_TAGS])
    def test_persona_vocabularies_are_subsets_of_the_catalogue(self, allowed: frozenset) -> None:
        assert allowed <= TIMBRE_V3_TAGS

    def test_no_persona_may_laugh_at_the_caller(self) -> None:
        """The ethics constraints rule out mockery, and a laugh mid-scam-call is mockery."""
        for allowed in (ELDER_TAGS, GRANDSON_TAGS):
            assert not allowed & {"laugh", "giggle", "shouting", "excited"}


class TestSanitize:
    def test_allowed_tags_survive(self) -> None:
        text, dropped = sanitize_tags("Haan beta <sigh> ek minute <pauses> ji", ELDER_TAGS)
        assert text == "Haan beta <sigh> ek minute <pauses> ji"
        assert dropped == ()

    def test_a_misspelled_tag_is_removed_and_reported(self) -> None:
        """`<cough>` is not `<coughs>`; Timbre would bill eight characters for silence."""
        text, dropped = sanitize_tags("Arre <cough> suno", ELDER_TAGS)
        assert text == "Arre suno"
        assert dropped == ("cough",)

    def test_a_real_tag_outside_the_persona_vocabulary_is_removed(self) -> None:
        text, dropped = sanitize_tags("Bahut accha <laugh> beta", ELDER_TAGS)
        assert text == "Bahut accha beta"
        assert dropped == ("laugh",)

    def test_several_drops_are_all_reported_in_order(self) -> None:
        _, dropped = sanitize_tags("<laugh> a <whispers> b <cough> c", ELDER_TAGS)
        assert dropped == ("laugh", "whispers", "cough")

    def test_removing_a_tag_does_not_leave_billable_whitespace(self) -> None:
        text, _ = sanitize_tags("ek <laugh> do", ELDER_TAGS)
        assert text == "ek do"
        assert "  " not in text

    def test_surrounding_whitespace_is_trimmed(self) -> None:
        text, _ = sanitize_tags("  <laugh> haan  ", ELDER_TAGS)
        assert text == "haan"

    def test_text_with_no_tags_is_untouched(self) -> None:
        text, dropped = sanitize_tags("Haan beta kaun bol raha hai", ELDER_TAGS)
        assert text == "Haan beta kaun bol raha hai"
        assert dropped == ()

    def test_a_tag_with_a_space_in_its_name_is_matched_exactly(self) -> None:
        text, dropped = sanitize_tags("Ek minute <clears throat> haan", ELDER_TAGS)
        assert text == "Ek minute <clears throat> haan"
        assert dropped == ()

    def test_tag_matching_is_case_sensitive_as_the_api_documents(self) -> None:
        text, dropped = sanitize_tags("<Sigh> haan", ELDER_TAGS)
        assert text == "haan"
        assert dropped == ("Sigh",)

    def test_an_allowed_set_containing_a_non_timbre_tag_is_a_programming_error(self) -> None:
        with pytest.raises(ValueError, match=r"not Timbre v3\.0 tags"):
            sanitize_tags("haan", {"sigh", "winks"})

    def test_markdown_emphasis_is_not_mistaken_for_a_tag(self) -> None:
        text, dropped = sanitize_tags("haan *beta* <sigh>", ELDER_TAGS)
        assert text == "haan *beta* <sigh>"
        assert dropped == ()


class TestBilling:
    def test_the_tag_and_its_brackets_are_billed(self) -> None:
        assert billable_characters("<sigh>") == 6
        assert billable_characters("haan <sigh> beta") == 16

    def test_the_documented_example_does_not_match_its_own_character_count(self) -> None:
        """A discrepancy in Gnani's docs, pinned here so it is not rediscovered by accident.

        The audio-tags page says `Hello! <giggle> Great to hear from you.` "is billed as 42
        characters - the full string, tag included". The full string is 39 characters. Either
        the example is wrong or Timbre adds ~3 characters per request, which would make our
        TTS estimates about 8% low on short tagged utterances. Resolving it needs one real
        invoice line: the Cost column of a TTS request next to the character count the
        dashboard logs for it. Until then `billable_characters` is plain `len` and every
        charge is flagged estimated.
        """
        documented = "Hello! <giggle> Great to hear from you."
        assert billable_characters(documented) == 39
        assert billable_characters(documented) != 42


class TestTidyForSpeech:
    """Both of these were seen on a real call, and both cost characters for nothing."""

    def test_line_breaks_become_spaces(self) -> None:
        assert tidy_for_speech("Haan haan.\n\nSabse pehle naam bataiye.") == (
            "Haan haan. Sabse pehle naam bataiye."
        )

    def test_runs_of_whitespace_collapse(self) -> None:
        assert tidy_for_speech("Haan   beta\t\tek  minute") == "Haan beta ek minute"

    def test_a_leading_silence_tag_is_dropped(self) -> None:
        """The filler already covered that beat; this would be dead air on top of it."""
        assert tidy_for_speech("<pauses> Haan beta kaun?") == "Haan beta kaun?"

    def test_several_leading_silence_tags_are_all_dropped(self) -> None:
        assert tidy_for_speech("<pauses> <pauses> Haan beta") == "Haan beta"

    def test_a_silence_tag_in_the_middle_is_kept(self) -> None:
        """Mid-utterance it is doing real work: an old man losing his thread."""
        text = "Haan beta ek minute. <pauses> Kya bola aapne?"
        assert tidy_for_speech(text) == text

    def test_a_different_leading_tag_is_kept(self) -> None:
        """`<slow>` at the front is the cheapest elderly-voice effect available."""
        assert tidy_for_speech("<slow> Haan beta ek minute") == "<slow> Haan beta ek minute"

    def test_a_leading_silence_tag_before_a_useful_tag_is_dropped(self) -> None:
        assert tidy_for_speech("<pauses> <slow> Haan beta") == "<slow> Haan beta"

    def test_plain_text_is_untouched(self) -> None:
        assert tidy_for_speech("Haan beta kaun bol raha hai") == "Haan beta kaun bol raha hai"

    def test_empty_text_stays_empty(self) -> None:
        assert tidy_for_speech("   \n  ") == ""

    def test_it_reduces_the_billable_count(self) -> None:
        raw = "<pauses> Haan haan.\n\nSabse pehle naam bataiye."
        assert billable_characters(tidy_for_speech(raw)) < billable_characters(raw)
