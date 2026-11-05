"""Turn-taking arithmetic. All pure, so all of it is here rather than in an async test."""

from __future__ import annotations

import pytest

from app.core.models import PersonaId, Speaker, Turn
from app.core.turn import (
    MAX_SENTENCE_CHARS,
    FillerRotation,
    TurnBudget,
    flush_tail,
    is_handoff_due,
    recent_turns,
    take_sentence,
)


def turn(seq: int, speaker: Speaker, text: str = "x", persona: PersonaId | None = None) -> Turn:
    return Turn(seq=seq, speaker=speaker, text=text, persona=persona, started_at_ms=seq * 1000)


class TestTakeSentence:
    def test_it_splits_at_the_first_full_stop(self) -> None:
        assert take_sentence("Haan beta, ek minute. Aage boliye") == (
            "Haan beta, ek minute.",
            "Aage boliye",
        )

    @pytest.mark.parametrize("mark", [".", "?", "!", "।"])
    def test_every_terminator_works(self, mark: str) -> None:
        split = take_sentence(f"Aap kaun bol rahe hain{mark} Haan")
        assert split is not None
        assert split[0].endswith(mark)

    def test_nothing_is_returned_until_a_sentence_exists(self) -> None:
        assert take_sentence("Haan beta") is None

    def test_a_fragment_shorter_than_the_minimum_is_not_worth_a_request(self) -> None:
        """ "Haan." as a first chunk sounds like a glitch, and each request has overhead."""
        assert take_sentence("Haan. Beta aap kaun bol rahe hain?") == (
            "Haan. Beta aap kaun bol rahe hain?",
            "",
        )

    def test_a_decimal_point_is_not_a_sentence_end(self) -> None:
        """Scammers say amounts constantly; "2.5 lakh" must not become two utterances."""
        assert take_sentence("Woh 2.5 lakh rupaye maang raha hai") is None

    def test_a_decimal_inside_a_longer_sentence_still_does_not_split(self) -> None:
        split = take_sentence("Amount 1.5 lakh hai beta, samajh nahi aaya. Phir se boliye")
        assert split is not None
        assert split[0] == "Amount 1.5 lakh hai beta, samajh nahi aaya."

    def test_a_terminator_inside_an_audio_tag_is_ignored(self) -> None:
        """Splitting a tag in half bills for it and speaks nothing."""
        split = take_sentence("Haan beta <clears throat> ek minute zara. Boliye")
        assert split is not None
        assert "<clears throat>" in split[0]

    def test_a_run_of_terminators_is_kept_with_the_sentence(self) -> None:
        split = take_sentence("Aap kaun bol rahe hain?! Mujhe samajh nahi aaya")
        assert split is not None
        assert split[0].endswith("?!")

    def test_a_long_run_on_is_cut_at_a_word_boundary(self) -> None:
        """An LLM that never punctuates must not hold the whole reply back."""
        words = "beta " * 60
        split = take_sentence(words)
        assert split is not None
        sentence, rest = split
        assert len(sentence) <= MAX_SENTENCE_CHARS
        assert not sentence.endswith("bet")
        assert rest

    def test_the_remainder_has_no_leading_space(self) -> None:
        split = take_sentence("Haan beta ek minute.    Boliye na")
        assert split is not None
        assert split[1] == "Boliye na"

    def test_an_impossible_bound_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_chars"):
            take_sentence("whatever", min_chars=50, max_chars=10)

    def test_an_empty_buffer_yields_nothing(self) -> None:
        assert take_sentence("") is None


class TestFlushTail:
    def test_a_real_fragment_is_returned(self) -> None:
        assert flush_tail("Haan beta") == "Haan beta"

    def test_whitespace_and_stray_punctuation_are_dropped(self) -> None:
        """Synthesising these costs characters and produces a click."""
        assert flush_tail("   ") is None
        assert flush_tail(".") is None
        assert flush_tail("") is None


class TestFillerRotation:
    def test_it_cycles_through_every_phrase(self) -> None:
        rotation = FillerRotation(["a", "b", "c"], seed="call-1")
        picked = {rotation.next() for _ in range(3)}
        assert picked == {"a", "b", "c"}

    def test_it_never_repeats_until_the_cycle_restarts(self) -> None:
        rotation = FillerRotation(["a", "b", "c"], seed="call-1")
        assert len({rotation.next() for _ in range(3)}) == 3

    def test_the_same_call_id_gives_the_same_order(self) -> None:
        """A replayed call has to produce the fillers the live one did."""
        first = FillerRotation(["a", "b", "c", "d"], seed="call-1")
        second = FillerRotation(["a", "b", "c", "d"], seed="call-1")
        assert [first.next() for _ in range(4)] == [second.next() for _ in range(4)]

    def test_a_different_call_gets_a_different_order(self) -> None:
        """A scammer who gets through twice should not hear the same openings in order."""
        orders = {
            tuple(FillerRotation(list("abcdefgh"), seed=f"call-{i}").phrases) for i in range(6)
        }
        assert len(orders) > 1

    def test_phrases_exposes_the_whole_set_for_cache_warming(self) -> None:
        rotation = FillerRotation(["a", "b", "c"], seed="x")
        assert sorted(rotation.phrases) == ["a", "b", "c"]

    def test_an_empty_catalogue_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            FillerRotation([])


class TestRecentTurns:
    def test_it_keeps_the_newest_turns_oldest_first(self) -> None:
        turns = [turn(i, Speaker.CALLER, f"t{i}") for i in range(10)]
        window = recent_turns(turns, TurnBudget(max_turns=3))
        assert [t.text for t in window] == ["t7", "t8", "t9"]

    def test_the_character_ceiling_wins_over_the_turn_count(self) -> None:
        turns = [turn(i, Speaker.CALLER, "x" * 500) for i in range(10)]
        window = recent_turns(turns, TurnBudget(max_turns=10, max_chars=1200))
        assert len(window) < 10

    def test_at_least_one_turn_survives_an_impossible_budget(self) -> None:
        """Otherwise the model is asked to reply to nothing at all."""
        turns = [turn(0, Speaker.CALLER, "x" * 10_000)]
        assert len(recent_turns(turns, TurnBudget(max_chars=200))) == 1

    def test_an_empty_history_is_empty(self) -> None:
        assert recent_turns([], TurnBudget()) == ()


class TestHandoff:
    def test_it_fires_once_the_persona_has_spoken_enough(self) -> None:
        turns = [
            turn(0, Speaker.CALLER),
            turn(1, Speaker.PERSONA, persona=PersonaId.DADA),
            turn(2, Speaker.CALLER),
            turn(3, Speaker.PERSONA, persona=PersonaId.DADA),
        ]
        assert is_handoff_due(turns, after_persona_turns=2)
        assert not is_handoff_due(turns, after_persona_turns=3)

    def test_the_count_resets_when_the_persona_changes(self) -> None:
        """Otherwise the grandson would hand over on his first word."""
        turns = [
            turn(0, Speaker.PERSONA, persona=PersonaId.DADA),
            turn(1, Speaker.PERSONA, persona=PersonaId.DADA),
            turn(2, Speaker.PERSONA, persona=PersonaId.GRANDSON),
        ]
        assert not is_handoff_due(turns, after_persona_turns=2)

    def test_caller_turns_do_not_count(self) -> None:
        turns = [turn(i, Speaker.CALLER) for i in range(10)]
        assert not is_handoff_due(turns, after_persona_turns=2)

    def test_a_non_positive_threshold_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            is_handoff_due([], after_persona_turns=0)


class TestEllipsisIsHesitationNotPunctuation:
    """From a real call. Trailing off is most of what this persona does.

    Treating "..." as a sentence boundary made the orchestrator speak "Inspector..." on its
    own, and once one terse reply was in the conversation history the model copied it for
    the rest of the call - four turns of two-word fragments.
    """

    def test_an_ellipsis_does_not_end_a_sentence(self) -> None:
        assert take_sentence("Inspector... haan beta kaun bol raha hai") is None

    def test_the_sentence_after_an_ellipsis_still_ends_normally(self) -> None:
        split = take_sentence("Inspector... haan beta kaun? Phir se boliye")
        assert split is not None
        assert split[0] == "Inspector... haan beta kaun?"

    def test_a_unicode_ellipsis_behaves_the_same(self) -> None:
        assert take_sentence("Inspector… haan beta kaun bol raha hai") is None

    def test_a_single_full_stop_still_ends_a_sentence(self) -> None:
        split = take_sentence("Haan beta ek minute. Boliye")
        assert split is not None
        assert split[0] == "Haan beta ek minute."

    def test_a_run_of_other_terminators_still_breaks_once(self) -> None:
        split = take_sentence("Aap kaun bol rahe hain?! Mujhe samajh nahi aaya")
        assert split is not None
        assert split[0].endswith("?!")

    def test_a_trailing_ellipsis_is_flushed_at_the_end_of_the_stream(self) -> None:
        """So an utterance that only trails off is still spoken, just not mid-stream."""
        assert flush_tail("Inspector...") == "Inspector..."
