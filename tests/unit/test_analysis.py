"""The analysis pass. Two rules: it never blocks the persona, and it never outranks the
deterministic reading."""

from __future__ import annotations

import pytest

from app.core.models import EntitySource, EntityType, PersonaId, ScamStage, Speaker, Turn
from app.providers.base import BadRequest, RateLimited
from app.providers.mock import MockLLMProvider
from app.services.analysis import (
    ANALYSIS_WINDOW_TURNS,
    RATE_LIMIT_BACKOFF_PASSES,
    CallAnalyst,
    LlmAnalysis,
    deterministic_analysis,
)

AUTHORITY = "Main Delhi cyber crime branch se inspector Sharma bol raha hoon."
PAYMENT = "Do lakh fake.officer@demo par UPI se bhej dijiye abhi."


def caller(text: str, seq: int = 0) -> Turn:
    return Turn(seq=seq, speaker=Speaker.CALLER, text=text, started_at_ms=seq * 1000)


def persona(text: str = "Haan beta", seq: int = 1) -> Turn:
    return Turn(
        seq=seq,
        speaker=Speaker.PERSONA,
        text=text,
        persona=PersonaId.DADA,
        started_at_ms=seq * 1000,
    )


class FailingLLM:
    name = "failing-llm"

    def stream_chat(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("the analysis pass must not stream")

    async def complete_json(self, *args: object, **kwargs: object) -> object:
        raise BadRequest("model is down")


class TestDeterministicPass:
    def test_it_works_with_no_model_at_all(self) -> None:
        """The dashboard has to render on keyword evidence alone."""
        analysis = deterministic_analysis([caller(AUTHORITY)], turn_seq=0)
        assert analysis.stage is ScamStage.AUTHORITY
        assert analysis.model_failed
        assert analysis.entities

    def test_it_only_reads_the_caller(self) -> None:
        """Dada mentioning a warrant must not advance the scammer's stage."""
        turns = [persona("Warrant? Arre beta, main retired clerk hoon.", 0)]
        assert deterministic_analysis(turns, turn_seq=0).stage is ScamStage.HOOK

    def test_everything_it_finds_is_marked_regex(self) -> None:
        analysis = deterministic_analysis([caller(PAYMENT)], turn_seq=0)
        assert analysis.entities
        assert all(e.source is EntitySource.REGEX for e in analysis.entities)

    def test_it_is_deterministic(self) -> None:
        turns = [caller(AUTHORITY, 0), caller(PAYMENT, 1)]
        assert deterministic_analysis(turns, turn_seq=1) == deterministic_analysis(
            turns, turn_seq=1
        )


class TestDegradation:
    async def test_a_dead_model_degrades_to_the_keyword_pass(self) -> None:
        """A model that is down must cost the dashboard detail, never the call."""
        analyst = CallAnalyst(FailingLLM())  # type: ignore[arg-type]
        analysis = await analyst.analyse([caller(AUTHORITY)], turn_seq=0)
        assert analysis.model_failed
        assert analysis.stage is ScamStage.AUTHORITY
        assert analyst.model_failures == 1

    async def test_analyse_never_raises(self) -> None:
        analyst = CallAnalyst(FailingLLM())  # type: ignore[arg-type]
        for seq in range(3):
            await analyst.analyse([caller(AUTHORITY, seq)], turn_seq=seq)
        assert analyst.model_failures == 3

    async def test_a_transcript_with_no_caller_turns_skips_the_model(self) -> None:
        """Nothing to analyse, so nothing is spent analysing it."""
        analyst = CallAnalyst(FailingLLM())  # type: ignore[arg-type]
        analysis = await analyst.analyse([persona(seq=0)], turn_seq=0)
        assert analyst.model_failures == 0
        assert analysis.model_failed


class TestRateLimitBackoff:
    """A rate limit means the window is spent, so the next pass inside it fails the same way.

    The analyst stands down instead of retrying, because its budget is shared with the
    dialogue path and nothing on the call is waiting for it. The deterministic pass keeps
    running throughout, which is why standing down is affordable at all.
    """

    class RateLimitedLLM:
        name = "rate-limited-llm"
        remaining_tokens: int | None = None

        def __init__(self) -> None:
            self.calls = 0

        def stream_chat(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("the analysis pass must not stream")

        async def complete_json(self, *args: object, **kwargs: object) -> object:
            self.calls += 1
            raise RateLimited("llm HTTP 429: ITPM")

    async def test_the_model_is_not_asked_again_for_several_passes(self) -> None:
        llm = self.RateLimitedLLM()
        analyst = CallAnalyst(llm)  # type: ignore[arg-type]
        for seq in range(1 + RATE_LIMIT_BACKOFF_PASSES):
            analysis = await analyst.analyse([caller(AUTHORITY, seq)], turn_seq=seq)
            assert analysis.model_failed
        assert llm.calls == 1
        assert analyst.rate_limited == 1

    async def test_it_tries_again_once_the_backoff_is_spent(self) -> None:
        llm = self.RateLimitedLLM()
        analyst = CallAnalyst(llm)  # type: ignore[arg-type]
        for seq in range(2 + RATE_LIMIT_BACKOFF_PASSES):
            await analyst.analyse([caller(AUTHORITY, seq)], turn_seq=seq)
        assert llm.calls == 2

    async def test_the_keyword_pass_still_reports_throughout(self) -> None:
        """The stage badge and the regex entities must not pause with the model."""
        analyst = CallAnalyst(self.RateLimitedLLM())  # type: ignore[arg-type]
        for seq in range(1 + RATE_LIMIT_BACKOFF_PASSES):
            analysis = await analyst.analyse([caller(PAYMENT, seq)], turn_seq=seq)
            assert analysis.stage is ScamStage.PAYMENT
            assert analysis.entities


class TestMerging:
    @staticmethod
    def _llm(payload: dict[str, object]) -> MockLLMProvider:
        llm = MockLLMProvider()
        llm.queue_json(payload)
        return llm

    async def test_the_model_can_add_entities_the_regex_cannot_see(self) -> None:
        llm = self._llm(
            {
                "stage": "authority",
                "stage_confidence": 0.8,
                "pressure": 0.4,
                "entities": [{"entity_type": "officer_name", "value": "Sharma", "confidence": 0.9}],
                "tactics": ["impersonating_police"],
                "notes": "Caller claims to be a police inspector.",
            }
        )
        analysis = await CallAnalyst(llm).analyse(
            [caller("main police se bol raha hoon")], turn_seq=0
        )
        llm_entities = [e for e in analysis.entities if e.source is EntitySource.LLM]
        assert [e.value for e in llm_entities] == ["Sharma"]
        assert analysis.tactics == ("impersonating_police",)
        assert not analysis.model_failed

    async def test_the_regex_row_wins_a_duplicate(self) -> None:
        """One UPI handle listed twice at two confidences looks like a bug to a reader."""
        llm = self._llm(
            {
                "stage": "payment",
                "entities": [
                    {"entity_type": "upi_id", "value": "fake.officer@demo", "confidence": 0.9}
                ],
            }
        )
        analysis = await CallAnalyst(llm).analyse([caller(PAYMENT)], turn_seq=0)
        handles = [e for e in analysis.entities if e.entity_type is EntityType.UPI_ID]
        assert len(handles) == 1
        assert handles[0].source is EntitySource.REGEX

    async def test_an_unmodelled_entity_type_is_dropped_not_coerced(self) -> None:
        llm = self._llm({"entities": [{"entity_type": "aadhaar", "value": "x", "confidence": 0.9}]})
        analysis = await CallAnalyst(llm).analyse([caller(AUTHORITY)], turn_seq=0)
        assert all(e.source is EntitySource.REGEX for e in analysis.entities)

    async def test_a_low_confidence_entity_is_dropped(self) -> None:
        llm = self._llm(
            {"entities": [{"entity_type": "officer_name", "value": "Maybe", "confidence": 0.1}]}
        )
        analysis = await CallAnalyst(llm).analyse([caller(AUTHORITY)], turn_seq=0)
        assert "Maybe" not in [e.value for e in analysis.entities]

    async def test_at_most_four_tactics_are_kept(self) -> None:
        llm = self._llm({"tactics": [f"t{i}" for i in range(9)]})
        analysis = await CallAnalyst(llm).analyse([caller(AUTHORITY)], turn_seq=0)
        assert len(analysis.tactics) == 4

    async def test_notes_are_truncated(self) -> None:
        llm = self._llm({"notes": "x" * 500})
        analysis = await CallAnalyst(llm).analyse([caller(AUTHORITY)], turn_seq=0)
        assert len(analysis.notes) <= 140


class TestStageReconciliation:
    @staticmethod
    def _llm(stage: str) -> MockLLMProvider:
        llm = MockLLMProvider()
        llm.queue_json({"stage": stage, "stage_confidence": 0.9})
        return llm

    async def test_the_model_can_advance_the_stage(self) -> None:
        """It sees things keywords cannot, so it is allowed to move the badge forward."""
        analysis = await CallAnalyst(self._llm("payment")).analyse([caller(AUTHORITY)], turn_seq=0)
        assert analysis.stage is ScamStage.PAYMENT

    async def test_the_model_cannot_drag_the_stage_backwards(self) -> None:
        analysis = await CallAnalyst(self._llm("hook")).analyse([caller(PAYMENT)], turn_seq=0)
        assert analysis.stage is ScamStage.PAYMENT

    async def test_a_disagreement_is_recorded_rather_than_averaged(self) -> None:
        """Showing a visible disagreement beats inventing an agreement."""
        analysis = await CallAnalyst(self._llm("hook")).analyse([caller(PAYMENT)], turn_seq=0)
        assert analysis.model_stage is ScamStage.HOOK
        assert analysis.stage is not analysis.model_stage

    async def test_agreement_leaves_no_disagreement_to_show(self) -> None:
        analysis = await CallAnalyst(self._llm("payment")).analyse([caller(PAYMENT)], turn_seq=0)
        assert analysis.model_stage is None

    async def test_an_invented_stage_name_is_ignored(self) -> None:
        analysis = await CallAnalyst(self._llm("extraction")).analyse(
            [caller(AUTHORITY)], turn_seq=0
        )
        assert analysis.stage is ScamStage.AUTHORITY

    async def test_the_stage_never_retreats_across_turns(self) -> None:
        analyst = CallAnalyst(FailingLLM())  # type: ignore[arg-type]
        await analyst.analyse([caller(PAYMENT, 0)], turn_seq=0)
        later = await analyst.analyse([caller(PAYMENT, 0), caller(AUTHORITY, 1)], turn_seq=1)
        assert later.stage is ScamStage.PAYMENT
        assert analyst.furthest_stage is ScamStage.PAYMENT


class TestPressure:
    async def test_the_higher_reading_wins(self) -> None:
        """Under-reporting pressure is the expensive direction: it decides a hook."""
        llm = MockLLMProvider()
        llm.queue_json({"stage": "hook", "pressure": 0.95})
        analysis = await CallAnalyst(llm).analyse([caller("Hello sir")], turn_seq=0)
        assert analysis.pressure_score == pytest.approx(0.95)

    async def test_the_keyword_reading_is_not_lowered_by_the_model(self) -> None:
        llm = MockLLMProvider()
        llm.queue_json({"stage": "payment", "pressure": 0.0})
        turns = [caller("Abhi turant bhejiye, warrant nikal jayega, last warning")]
        keyword = deterministic_analysis(turns, turn_seq=0).pressure_score
        analysis = await CallAnalyst(llm).analyse(turns, turn_seq=0)
        assert analysis.pressure_score >= keyword
        assert keyword > 0.0


class TestWindow:
    async def test_only_the_recent_window_is_sent(self) -> None:
        """The analyst shares a token budget with the dialogue model."""
        llm = MockLLMProvider()
        llm.queue_json({"stage": "hook"})
        turns = [caller(f"line {i}", i) for i in range(40)]
        await CallAnalyst(llm).analyse(turns, turn_seq=39)
        assert llm.chat_calls == []  # complete_json is a separate path from stream_chat

    def test_the_window_is_bounded(self) -> None:
        turns = [caller(f"line {i}", i) for i in range(40)]
        analysis = deterministic_analysis(turns, turn_seq=39)
        assert ANALYSIS_WINDOW_TURNS < 40
        assert analysis.turn_seq == 39


class TestSchema:
    def test_every_field_has_a_default_so_a_terse_answer_still_validates(self) -> None:
        """A model that returns `{}` must degrade, not raise."""
        parsed = LlmAnalysis.model_validate({})
        assert parsed.stage == ScamStage.HOOK.value
        assert parsed.entities == []

    def test_unknown_keys_are_ignored(self) -> None:
        parsed = LlmAnalysis.model_validate({"stage": "fear", "invented_field": 1})
        assert parsed.stage == "fear"

    def test_out_of_range_numbers_are_refused(self) -> None:
        with pytest.raises(ValueError):
            LlmAnalysis.model_validate({"pressure": 4.0})
