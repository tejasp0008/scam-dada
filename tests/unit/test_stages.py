"""Scam stage detection and the pressure gauge. Pure, so every dashboard number is here."""

from __future__ import annotations

import pytest

from app.core.models import ScamStage
from app.core.stages import (
    MIN_STAGE_SCORE,
    STAGE_ORDER,
    assess,
    furthest_stage,
    pressure,
    stage_scores,
)

HOOK = "Hello sir, aapke naam par ek parcel aaya hai."
AUTHORITY = "Main Delhi cyber crime branch se inspector Sharma bol raha hoon."
FEAR = "Case number register ho gaya hai, warrant nikal jayega aapke naam par."
ISOLATION = "Aap kisi ko bataiye mat, yeh confidential hai."
PAYMENT = "Do lakh fake.officer@demo par UPI se bhej dijiye."


class TestStageDetection:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (HOOK, ScamStage.HOOK),
            (AUTHORITY, ScamStage.AUTHORITY),
            (FEAR, ScamStage.FEAR),
            (ISOLATION, ScamStage.ISOLATION),
            (PAYMENT, ScamStage.PAYMENT),
        ],
    )
    def test_each_stage_is_recognised_from_its_own_language(
        self, text: str, expected: ScamStage
    ) -> None:
        assert assess([text]).stage is expected

    def test_a_full_script_ends_at_payment(self) -> None:
        assert assess([HOOK, AUTHORITY, FEAR, ISOLATION, PAYMENT]).stage is ScamStage.PAYMENT

    def test_an_opening_line_alone_does_not_claim_payment(self) -> None:
        assert assess([HOOK]).stage is ScamStage.HOOK

    def test_weak_evidence_does_not_move_the_badge(self) -> None:
        """One glancing keyword in a long call is noise, and a flickering badge is unread."""
        assert assess(["Haan ji, department se baat ho gayi thi."]).stage is ScamStage.HOOK

    def test_an_empty_transcript_is_the_hook_with_low_confidence(self) -> None:
        assessment = assess([])
        assert assessment.stage is ScamStage.HOOK
        assert assessment.confidence <= 0.2


class TestMonotonicity:
    def test_the_stage_does_not_retreat_when_a_scammer_circles_back(self) -> None:
        """Returning to authority after payment is reinforcement, not regression."""
        assessment = assess([PAYMENT, AUTHORITY], furthest=ScamStage.PAYMENT)
        assert assessment.stage is ScamStage.PAYMENT

    def test_without_the_previous_stage_it_can_retreat(self) -> None:
        """Which is exactly why the orchestrator passes it back in."""
        assert assess([AUTHORITY]).stage is ScamStage.AUTHORITY

    @pytest.mark.parametrize("stage", STAGE_ORDER)
    def test_furthest_stage_never_goes_backwards(self, stage: ScamStage) -> None:
        assert furthest_stage(ScamStage.PAYMENT, stage) is ScamStage.PAYMENT
        assert furthest_stage(None, stage) is stage

    def test_furthest_stage_advances(self) -> None:
        assert furthest_stage(ScamStage.HOOK, ScamStage.FEAR) is ScamStage.FEAR


class TestPressure:
    def test_a_calm_opening_scores_low(self) -> None:
        score, signals = pressure([HOOK])
        assert score < 0.2
        assert signals == ()

    def test_threats_and_urgency_accumulate(self) -> None:
        score, signals = pressure([FEAR, "Abhi turant bhejiye, last warning hai."])
        assert score > 0.5
        assert "threat_of_arrest" in signals
        assert "urgency" in signals
        assert "final_warning" in signals

    def test_a_repeated_signal_counts_once(self) -> None:
        """A scammer saying "abhi" nine times is urgent, not nine times as urgent - and
        letting repeats accumulate would peg the gauge at 1.0 on every call."""
        once, _ = pressure(["Abhi bhejiye."])
        many, _ = pressure(["Abhi bhejiye."] * 9)
        assert once == many

    def test_it_is_capped_at_one(self) -> None:
        everything = [text for _, _, text in []] + [
            "Abhi turant, last warning, warrant nikal jayega, kisi ko bataiye mat, "
            "phone mat kaatiye, OTP bataiye, AnyDesk install kijiye, do minute mein!!"
        ]
        score, _ = pressure(everything)
        assert score == 1.0

    def test_remote_access_requests_are_flagged(self) -> None:
        """The single most damaging thing a victim can be talked into."""
        _, signals = pressure(["Aap AnyDesk download kar lijiye."])
        assert "remote_access" in signals

    def test_credential_requests_are_flagged(self) -> None:
        _, signals = pressure(["OTP bataiye jo aaya hai."])
        assert "credential_request" in signals

    def test_signals_are_sorted_so_the_gauge_is_stable(self) -> None:
        _, signals = pressure([FEAR, ISOLATION, "abhi jaldi"])
        assert list(signals) == sorted(signals)


class TestAssessment:
    def test_it_reports_per_stage_evidence_for_the_tooltip(self) -> None:
        assessment = assess([AUTHORITY, FEAR])
        assert set(assessment.scores) == set(STAGE_ORDER)
        assert assessment.scores[ScamStage.AUTHORITY] >= MIN_STAGE_SCORE

    def test_confidence_rises_with_evidence_and_never_exceeds_one(self) -> None:
        weak = assess([AUTHORITY])
        strong = assess([AUTHORITY, "inspector Sharma, cyber cell, crime branch, police"])
        assert weak.confidence < strong.confidence
        assert strong.confidence <= 1.0

    def test_pressure_and_signals_agree(self) -> None:
        assessment = assess([FEAR, "abhi turant"])
        expected_score, expected_signals = pressure([FEAR, "abhi turant"])
        assert assessment.pressure_score == pytest.approx(round(expected_score, 3))
        assert assessment.signals == expected_signals

    def test_everything_is_in_range(self) -> None:
        assessment = assess([HOOK, AUTHORITY, FEAR, ISOLATION, PAYMENT])
        assert 0.0 <= assessment.confidence <= 1.0
        assert 0.0 <= assessment.pressure_score <= 1.0

    def test_it_is_deterministic(self) -> None:
        """Every number on the dashboard has to be reproducible from the transcript."""
        transcript = [HOOK, AUTHORITY, FEAR]
        assert assess(transcript) == assess(transcript)

    def test_scores_are_cumulative_over_turns(self) -> None:
        one = stage_scores([AUTHORITY])
        two = stage_scores([AUTHORITY, AUTHORITY])
        assert two[ScamStage.AUTHORITY] > one[ScamStage.AUTHORITY]


class TestStageSelectionRule:
    """The latest stage with evidence, not the strongest. This was a real bug.

    A scammer who invokes the police in every sentence accumulates authority evidence
    indefinitely. Picking the maximum left the badge stuck on `authority` while he was
    asking for a UPI transfer - the one moment the dashboard most needs to be right.
    """

    def test_payment_wins_over_louder_authority_evidence(self) -> None:
        heavy_authority = [AUTHORITY] * 4
        assessment = assess([*heavy_authority, PAYMENT])
        assert assessment.scores[ScamStage.AUTHORITY] > assessment.scores[ScamStage.PAYMENT]
        assert assessment.stage is ScamStage.PAYMENT

    def test_an_earlier_stage_with_more_evidence_does_not_win(self) -> None:
        assert assess([HOOK] * 5 + [FEAR]).stage is ScamStage.FEAR

    def test_confidence_never_reaches_certainty(self) -> None:
        """Keyword matching does not justify "definitely", however many keywords match."""
        assessment = assess([AUTHORITY] * 20)
        assert assessment.confidence < 1.0
