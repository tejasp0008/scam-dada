"""Digits as a person reads them aloud, shared by the canary matcher and the extractor."""

from __future__ import annotations

from app.core.spoken import digit_run, digit_run_after


class TestDigitRun:
    def test_english_digit_words_become_digits(self) -> None:
        assert digit_run("zero zero one two") == "0012"

    def test_hinglish_digit_words_become_digits(self) -> None:
        assert digit_run("ek do teen chaar") == "1234"

    def test_a_repeater_multiplies_the_next_digit(self) -> None:
        assert digit_run("triple zero one") == "0001"

    def test_written_digits_pass_through(self) -> None:
        assert digit_run("5010 0234") == "50100234"

    def test_the_run_stops_at_the_first_word_that_is_not_a_number(self) -> None:
        """Without this the keyword would swallow the rest of the sentence."""
        assert digit_run("one one two two hai, IFSC FAKE0000777") == "1122"

    def test_text_that_does_not_start_with_a_number_yields_nothing(self) -> None:
        assert digit_run("Suresh Kumar ke naam") == ""


class TestDigitRunAfter:
    def test_a_few_words_before_the_number_are_stepped_over(self) -> None:
        assert digit_run_after(" likh lijiye, zero zero one one") == "0011"

    def test_a_number_far_past_the_keyword_is_not_claimed(self) -> None:
        """Reaching into the next clause is the failure the keyword pattern exists to avoid."""
        assert digit_run_after(" holder ka naam Rajesh hai aur phone 9876543210 hai") == ""
