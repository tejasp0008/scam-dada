"""Report arithmetic. Every number a judge will see, computed where a test can reach it."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.core.models import Entity, EntitySource, EntityType, PersonaId, ScamStage, Speaker, Turn
from app.core.report import (
    CallReport,
    CallSummary,
    LeakReport,
    count_turns,
    latency_legs,
    percentile,
    personas_in_order,
    seconds_wasted,
)
from app.core.state import CallState
from app.services.evidence import render_html, render_pdf

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)


def turn(
    seq: int,
    speaker: Speaker,
    *,
    at_ms: int = 0,
    duration_ms: int | None = None,
    persona: PersonaId | None = None,
) -> Turn:
    return Turn(
        seq=seq,
        speaker=speaker,
        text=f"turn {seq}",
        persona=persona,
        started_at_ms=at_ms,
        duration_ms=duration_ms,
    )


class TestPercentile:
    def test_the_median_of_an_odd_sample(self) -> None:
        assert percentile([10, 20, 30], 0.5) == 20

    def test_it_returns_a_value_that_actually_occurred(self) -> None:
        """Nearest-rank, not interpolated: an invented number corresponds to no turn."""
        assert percentile([10, 20], 0.5) in (10, 20)

    def test_p95_of_a_small_sample_is_the_worst_case(self) -> None:
        assert percentile([5, 10, 100], 0.95) == 100

    def test_an_empty_sample_has_no_percentile(self) -> None:
        assert percentile([], 0.5) is None

    def test_a_single_sample_is_its_own_percentile(self) -> None:
        assert percentile([42], 0.5) == 42
        assert percentile([42], 0.95) == 42

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
    def test_an_impossible_fraction_is_refused(self, fraction: float) -> None:
        with pytest.raises(ValueError, match="fraction"):
            percentile([1, 2, 3], fraction)


class TestLatencyLegs:
    def test_each_leg_reports_its_sample_size(self) -> None:
        legs = latency_legs({"llm_first_token_ms": [200, 300, 400]})
        assert legs[0].samples == 3
        assert legs[0].median_ms == 300

    def test_an_unmeasured_leg_is_reported_not_omitted(self) -> None:
        """A missing leg reads as "fast" when it means "never measured"."""
        legs = latency_legs({"stt_final_ms": []})
        assert legs[0].samples == 0
        assert legs[0].median_ms is None
        assert legs[0].worst_ms is None

    def test_the_worst_case_is_the_maximum(self) -> None:
        legs = latency_legs({"tts_first_byte_ms": [100, 900, 200]})
        assert legs[0].worst_ms == 900

    def test_leg_order_is_preserved(self) -> None:
        legs = latency_legs({"a": [1], "b": [2], "c": [3]})
        assert [leg.name for leg in legs] == ["a", "b", "c"]


class TestTurnCounting:
    def test_caller_and_persona_turns_are_counted_separately(self) -> None:
        turns = [
            turn(0, Speaker.CALLER),
            turn(1, Speaker.PERSONA, persona=PersonaId.DADA),
            turn(2, Speaker.CALLER),
        ]
        assert count_turns(turns) == (2, 1)

    def test_system_turns_count_as_neither(self) -> None:
        assert count_turns([turn(0, Speaker.SYSTEM)]) == (0, 0)

    def test_an_empty_call_counts_nothing(self) -> None:
        assert count_turns([]) == (0, 0)


class TestPersonaOrder:
    def test_personas_appear_in_the_order_they_first_spoke(self) -> None:
        turns = [
            turn(0, Speaker.PERSONA, persona=PersonaId.DADA),
            turn(1, Speaker.PERSONA, persona=PersonaId.DADA),
            turn(2, Speaker.PERSONA, persona=PersonaId.GRANDSON),
        ]
        assert personas_in_order(turns) == (PersonaId.DADA, PersonaId.GRANDSON)

    def test_a_persona_is_listed_once(self) -> None:
        turns = [turn(i, Speaker.PERSONA, persona=PersonaId.DADA) for i in range(5)]
        assert personas_in_order(turns) == (PersonaId.DADA,)

    def test_caller_turns_contribute_nothing(self) -> None:
        assert personas_in_order([turn(0, Speaker.CALLER)]) == ()


class TestSecondsWasted:
    def test_it_spans_the_whole_call_including_the_silences(self) -> None:
        """The headline number. Silences between turns are the scammer's time too, and they
        are most of what a stalling persona produces."""
        turns = [
            turn(0, Speaker.CALLER, at_ms=0, duration_ms=3000),
            turn(1, Speaker.PERSONA, at_ms=20_000, persona=PersonaId.DADA),
        ]
        assert seconds_wasted(turns) == 20.0

    def test_it_is_not_the_sum_of_turn_durations(self) -> None:
        turns = [
            turn(0, Speaker.CALLER, at_ms=0, duration_ms=1000),
            turn(1, Speaker.CALLER, at_ms=60_000, duration_ms=1000),
        ]
        assert seconds_wasted(turns) == 61.0

    def test_the_calls_own_end_extends_it(self) -> None:
        """A persona talking after the last recorded turn still wasted that time."""
        turns = [turn(0, Speaker.CALLER, at_ms=0, duration_ms=1000)]
        assert seconds_wasted(turns, ended_at_ms=45_000) == 45.0

    def test_an_earlier_end_does_not_shorten_it(self) -> None:
        turns = [turn(0, Speaker.CALLER, at_ms=0, duration_ms=30_000)]
        assert seconds_wasted(turns, ended_at_ms=5_000) == 30.0

    def test_an_empty_call_wasted_nothing(self) -> None:
        assert seconds_wasted([]) == 0.0

    def test_it_is_never_negative(self) -> None:
        assert seconds_wasted([turn(0, Speaker.CALLER, at_ms=10_000)], ended_at_ms=0) >= 0.0


class TestSummary:
    def test_minutes_are_derived_from_seconds(self) -> None:
        summary = CallSummary(
            call_id="c1", started_at=NOW, state=CallState.ENDED, seconds_wasted=150.0
        )
        assert summary.minutes_wasted == 2.5

    def test_a_call_with_no_end_is_live(self) -> None:
        live = CallSummary(call_id="c1", started_at=NOW, state=CallState.PERSONA_ACTIVE)
        assert live.is_live
        done = live.model_copy(update={"ended_at": NOW})
        assert not done.is_live

    def test_out_of_range_scores_are_refused(self) -> None:
        with pytest.raises(ValueError):
            CallSummary(call_id="c1", started_at=NOW, state=CallState.ENDED, peak_pressure=1.4)


class TestReport:
    def test_entities_are_grouped_in_a_stable_order(self) -> None:
        report = CallReport(
            summary=CallSummary(call_id="c1", started_at=NOW, state=CallState.ENDED),
            entities=(
                Entity(
                    entity_type=EntityType.UPI_ID,
                    value="a@demo",
                    source=EntitySource.REGEX,
                    turn_seq=0,
                ),
                Entity(
                    entity_type=EntityType.AMOUNT,
                    value="INR 100",
                    source=EntitySource.REGEX,
                    turn_seq=1,
                ),
                Entity(
                    entity_type=EntityType.UPI_ID,
                    value="b@demo",
                    source=EntitySource.LLM,
                    turn_seq=2,
                    confidence=0.8,
                ),
            ),
        )
        grouped = report.entities_by_type
        assert list(grouped) == ["amount", "upi_id"]
        assert len(grouped["upi_id"]) == 2

    def test_a_report_with_nothing_in_it_is_valid(self) -> None:
        """A call that was hung up on immediately still has to render."""
        report = CallReport(
            summary=CallSummary(call_id="c1", started_at=NOW, state=CallState.ENDED)
        )
        assert report.entities_by_type == {}
        assert report.turns == ()

    def test_a_leak_carries_the_call_that_issued_the_identity(self) -> None:
        leak = LeakReport(
            identity_id="canary-1",
            issued_call_id="earlier",
            matched_field="upi_handle",
            matched_text="x@demo",
            score=1.0,
        )
        assert leak.issued_call_id == "earlier"

    def test_the_stage_timeline_is_ordered_pairs(self) -> None:
        report = CallReport(
            summary=CallSummary(call_id="c1", started_at=NOW, state=CallState.ENDED),
            stage_timeline=((NOW, ScamStage.HOOK), (NOW, ScamStage.PAYMENT)),
        )
        assert report.stage_timeline[-1][1] is ScamStage.PAYMENT


class TestEvidenceRendering:
    """The pack renders from the report and computes nothing of its own."""

    @staticmethod
    def _report() -> CallReport:
        return CallReport(
            summary=CallSummary(
                call_id="call-1",
                started_at=NOW,
                state=CallState.ENDED,
                seconds_wasted=125.0,
                caller_turns=4,
                persona_turns=4,
                personas=(PersonaId.DADA, PersonaId.GRANDSON),
                stage_reached=ScamStage.PAYMENT,
                peak_pressure=0.8,
                entity_count=1,
                leak_count=1,
            ),
            turns=(turn(0, Speaker.CALLER),),
            entities=(
                Entity(
                    entity_type=EntityType.UPI_ID,
                    value="fake.officer@demo",
                    source=EntitySource.REGEX,
                    turn_seq=0,
                ),
            ),
            leaks=(
                LeakReport(
                    identity_id="canary-1",
                    issued_call_id="earlier-call",
                    matched_field="upi_handle",
                    matched_text="kamla.1@demo",
                    score=1.0,
                ),
            ),
            latency=latency_legs({"llm_first_token_ms": [300], "stt_final_ms": []}),
        )

    def test_the_headline_comes_from_the_report(self) -> None:
        assert "2.1 minutes" in render_html(self._report())

    def test_a_leak_is_attributed_to_the_call_that_issued_it(self) -> None:
        page = render_html(self._report())
        assert "Identity leak" in page
        assert "earlier-call"[:8] in page

    def test_an_unmeasured_leg_is_shown_as_absent(self) -> None:
        """A blank cell reads as fast; the pack says so in words underneath."""
        page = render_html(self._report())
        assert "never measurable" in page

    def test_the_pack_states_its_identifiers_are_fictional(self) -> None:
        """It is a document people forward, so it has to say so on its own face."""
        page = render_html(self._report())
        assert "fictional" in page
        assert "cannot be dialled" in page

    def test_transcript_text_is_escaped(self) -> None:
        report = self._report().model_copy(
            update={
                "turns": (
                    Turn(
                        seq=0,
                        speaker=Speaker.CALLER,
                        text="<img src=x onerror=alert(1)>",
                        started_at_ms=0,
                    ),
                )
            }
        )
        page = render_html(report)
        assert "<img src=x" not in page
        assert "&lt;img" in page

    def test_an_empty_call_still_renders(self) -> None:
        empty = CallReport(summary=CallSummary(call_id="c", started_at=NOW, state=CallState.ENDED))
        page = render_html(empty)
        assert "No transcript was recorded" in page
        assert "Nothing was extracted" in page

    def test_the_pdf_path_explains_itself_when_it_cannot_run(self) -> None:
        """WeasyPrint fails with OSError, not ImportError, when its system libraries are
        missing - which turned a setup problem into a 500 until it was caught."""
        try:
            pdf = render_pdf(self._report())
        except RuntimeError as exc:
            assert "pdf" in str(exc).lower()
            assert "evidence.html" in str(exc)
        else:
            assert pdf[:4] == b"%PDF"
