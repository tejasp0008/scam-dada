"""Hang-up risk and hook selection. The feedback loop the whole project turns on."""

from __future__ import annotations

import re

import pytest

from app.core.models import PersonaId, ScamStage, Speaker, Turn
from app.core.risk import (
    BAIT_HOOKS,
    HOOK_THRESHOLD,
    BaitHook,
    assess,
    choose_hook,
)


def caller(text: str, seq: int = 0) -> Turn:
    return Turn(seq=seq, speaker=Speaker.CALLER, text=text, started_at_ms=seq * 1000)


def persona(text: str = "Haan beta", seq: int = 0) -> Turn:
    return Turn(
        seq=seq,
        speaker=Speaker.PERSONA,
        text=text,
        persona=PersonaId.DADA,
        started_at_ms=seq * 1000,
    )


class TestRiskScoring:
    def test_a_fresh_call_has_no_risk(self) -> None:
        assert assess([]).hangup_risk == 0.0

    def test_a_patient_scammer_scores_low(self) -> None:
        turns = [caller("Haan sir, main cyber cell se bol raha hoon", 0), persona(seq=1)]
        assessment = assess(turns)
        assert assessment.hangup_risk < 0.2
        assert not assessment.needs_hook

    @pytest.mark.parametrize(
        ("text", "signal"),
        [
            ("Main phone rakh raha hoon", "threatening_to_hang_up"),
            ("Aap mera time waste kar rahe hain", "accusing_time_waste"),
            ("Kitni baar bataun aapko", "repetition"),
            ("Kal baat karte hain phir", "deferring_the_call"),
            ("Aap koi bot ho? Recording hai?", "suspecting_a_bot"),
            ("Aap bahre ho kya", "commenting_on_age"),
        ],
    )
    def test_each_impatience_signal_is_named(self, text: str, signal: str) -> None:
        """A number the operator cannot explain is a number they will not trust."""
        assert signal in assess([caller(text)]).signals

    def test_suspecting_a_bot_is_the_heaviest_single_signal(self) -> None:
        """Once he thinks he is talking to a machine, nothing else matters."""
        bot = assess([caller("Yeh koi recording hai, bot hai")]).hangup_risk
        mild = assess([caller("Samajh nahi aa rahe aap")]).hangup_risk
        assert bot > mild

    def test_signals_accumulate_towards_a_hook(self) -> None:
        turns = [
            caller("Aap mera time waste kar rahe hain", 0),
            caller("Main phone rakh raha hoon", 1),
        ]
        assessment = assess(turns)
        assert assessment.hangup_risk >= HOOK_THRESHOLD
        assert assessment.needs_hook

    def test_a_repeated_signal_counts_once(self) -> None:
        once = assess([caller("Main phone rakh raha hoon")]).hangup_risk
        twice = assess([caller("Main phone rakh raha hoon", i) for i in range(2)]).hangup_risk
        assert once == twice

    def test_it_is_capped_at_one(self) -> None:
        text = (
            "Bot hai? Recording? Main phone rakh raha hoon, bye, time waste, kitni baar, "
            "kal baat karenge, aap bahre ho, supervisor se baat karo, samajh nahi aa rahe"
        )
        assert assess([caller(text)]).hangup_risk == 1.0

    def test_only_recent_turns_are_considered(self) -> None:
        """Impatience ten turns ago that he got over is not predictive of a hang-up now."""
        old = [caller("Main phone rakh raha hoon", 0)]
        later = [caller(f"Haan ji theek hai, sab samajh gaya number {i}", i) for i in range(1, 7)]
        assert assess(old + later).hangup_risk < assess(old).hangup_risk

    def test_persona_turns_are_not_scored(self) -> None:
        """Dada saying he will put the phone down is in character, not a hang-up risk."""
        assert assess([persona("Main phone rakh raha hoon beta")]).hangup_risk == 0.0

    def test_shortening_replies_are_a_signal_a_phrase_list_cannot_see(self) -> None:
        turns = [
            caller("Main Delhi cyber crime branch se inspector Sharma bol raha hoon", 0),
            caller("Aapke naam par parcel pakda gaya hai usme illegal items hain", 1),
            caller("Haan", 2),
            caller("Theek", 3),
        ]
        assert "replies_shortening" in assess(turns).signals

    def test_short_replies_early_are_not_a_signal(self) -> None:
        """Two short turns at the start of a call is just an opening, not disengagement."""
        assert "replies_shortening" not in assess([caller("Haan", 0), caller("Ji", 1)]).signals

    def test_signals_are_sorted_so_the_gauge_is_stable(self) -> None:
        turns = [caller("Time waste, bot hai, phone rakh raha hoon, kitni baar")]
        signals = assess(turns).signals
        assert list(signals) == sorted(signals)

    def test_the_threshold_is_configurable(self) -> None:
        turns = [caller("Samajh nahi aa rahe aap")]
        assert not assess(turns).needs_hook
        assert assess(turns, threshold=0.05).needs_hook

    def test_it_is_deterministic(self) -> None:
        turns = [caller("Time waste kar rahe hain", 0), caller("Phone rakh raha hoon", 1)]
        assert assess(turns) == assess(turns)


class TestHooks:
    def test_a_payment_stage_hook_is_preferred_at_payment(self) -> None:
        hook = choose_hook(ScamStage.PAYMENT)
        assert hook is not None
        assert hook.stage is ScamStage.PAYMENT

    def test_a_general_hook_is_used_when_no_stage_hook_fits(self) -> None:
        hook = choose_hook(ScamStage.HOOK)
        assert hook is not None
        assert hook.stage is None

    def test_a_used_hook_is_never_offered_again(self) -> None:
        """A persona that dangles the same forgotten account twice is a loop, not a man."""
        first = choose_hook(ScamStage.PAYMENT)
        assert first is not None
        second = choose_hook(ScamStage.PAYMENT, used=[first.name])
        assert second is not None
        assert second.name != first.name

    def test_when_every_hook_is_spent_none_is_returned(self) -> None:
        assert choose_hook(ScamStage.PAYMENT, used=[h.name for h in BAIT_HOOKS]) is None

    def test_hook_names_are_unique(self) -> None:
        names = [hook.name for hook in BAIT_HOOKS]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("hook", BAIT_HOOKS, ids=lambda h: h.name)
    def test_no_hook_pleads(self, hook: BaitHook) -> None:
        """ "Please don't go" confirms we are wasting his time. A hook is about his incentive."""
        lowered = hook.instruction.lower()
        for plea in ("please don't", "please do not", "stay on", "don't hang up"):
            assert plea not in lowered

    @pytest.mark.parametrize("hook", BAIT_HOOKS, ids=lambda h: h.name)
    def test_no_hook_promises_a_payment(self, hook: BaitHook) -> None:
        """A promise that is never kept is where a scammer gives up; a hook stays almost-ready."""
        lowered = hook.instruction.lower()
        assert "i will send" not in lowered
        assert "will transfer" not in lowered
        assert "bhej dunga" not in lowered

    @pytest.mark.parametrize("hook", BAIT_HOOKS, ids=lambda h: h.name)
    def test_every_hook_is_an_instruction_not_a_script(self, hook: BaitHook) -> None:
        """A verbatim line spoken twice stops sounding like the persona."""
        assert len(hook.instruction) > 40
        assert not hook.instruction.startswith('"')

    @pytest.mark.parametrize("hook", BAIT_HOOKS, ids=lambda h: h.name)
    def test_no_hook_invents_a_real_identifier(self, hook: BaitHook) -> None:
        assert not re.search(r"\b[6-9]\d{9}\b", hook.instruction)
        assert not re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", hook.instruction)


class TestFalsePositivesFromRealCalls:
    def test_the_word_fake_in_a_case_number_is_not_bot_suspicion(self) -> None:
        """A real call scored 0.4 for bot suspicion on "case number fake C Y B triple zero
        one" - our own synthetic case-number prefix, from a caller who suspected nothing."""
        turns = [caller("Case number fake C Y B triple zero one register ho gaya hai")]
        assert "suspecting_a_bot" not in assess(turns).signals

    @pytest.mark.parametrize(
        "text",
        [
            "Aap koi bot ho kya",
            "Yeh recording hai?",
            "Aap computer se baat kar rahe ho",
            "Natak kar rahe hain aap",
        ],
    )
    def test_a_real_accusation_still_scores(self, text: str) -> None:
        assert "suspecting_a_bot" in assess([caller(text)]).signals
