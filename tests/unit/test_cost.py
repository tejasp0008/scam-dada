"""The cost model.

Anchored on the only real datapoint available: six requests on the usage dashboard that
totalled ₹0.17, in which each ~2.7 s STT call was billed as 0.05 minutes.
"""

from __future__ import annotations

import pytest

from app.core.cost import (
    BilledUnit,
    BillingPool,
    CostModel,
    UnitType,
    round_up_to_increment,
)

MODEL = CostModel()


def test_a_short_call_is_billed_at_the_minimum_increment() -> None:
    """The observed case: 2.7 s of audio appeared on the dashboard as 0.05 minutes."""
    charge = MODEL.price_stt(2.7)
    assert charge.billed_units == 0.05
    assert charge.billed_unit is BilledUnit.STT_MINUTES
    assert charge.raw_unit is UnitType.STT_SECONDS
    assert charge.raw_units == 2.7


@pytest.mark.parametrize(
    ("seconds", "expected_minutes"),
    [
        (0.1, 0.05),
        (2.7, 0.05),
        (3.0, 0.05),
        (3.1, 0.10),
        (6.0, 0.10),
        (6.1, 0.15),
        (60.0, 1.00),
        (61.0, 1.05),
        (599.9, 10.00),
    ],
)
def test_stt_rounds_up_to_the_billing_increment(seconds: float, expected_minutes: float) -> None:
    assert MODEL.price_stt(seconds).billed_units == pytest.approx(expected_minutes)


def test_an_exact_multiple_is_not_pushed_to_the_next_increment() -> None:
    """Float division makes 3.0 s / 60 / 0.05 evaluate above 1.0; it must still bill 0.05."""
    assert MODEL.price_stt(3.0).billed_units == 0.05
    assert MODEL.price_stt(180.0).billed_units == pytest.approx(3.0)


def test_rounding_overhead_is_visible_in_the_record() -> None:
    """Billing a 2.7 s call as 3 s is an 11% overhead we want to be able to quantify."""
    charge = MODEL.price_stt(2.7)
    assert charge.billed_units * 60.0 > charge.raw_units


def test_zero_audio_is_not_billed() -> None:
    charge = MODEL.price_stt(0.0)
    assert charge.billed_units == 0.0
    assert charge.credits == 0.0


def test_stt_credits_follow_the_rate() -> None:
    model = CostModel(stt_rate_per_minute=2.0, rates_confirmed=True)
    charge = model.price_stt(60.0)
    assert charge.billed_units == pytest.approx(1.0)
    assert charge.credits == pytest.approx(2.0)
    assert charge.estimated is False


def test_tts_is_billed_per_character_with_no_rounding() -> None:
    charge = MODEL.price_tts(1000)
    assert charge.billed_units == 1000.0
    assert charge.credits == pytest.approx(MODEL.tts_rate_per_1k_chars)
    assert charge.pool is BillingPool.GNANI


def test_tts_counts_the_characters_it_is_given_including_tags() -> None:
    """Tag characters are billable, so the caller passes the string exactly as sent."""
    text = "Haan beta, <sigh> kaun bol raha hai?"
    assert MODEL.price_tts(len(text)).raw_units == 36.0


def test_the_llm_is_a_separate_pool() -> None:
    """Groq does not spend Gnani credits; mixing the two would hide a real ceiling."""
    assert MODEL.price_llm(1000).pool is BillingPool.LLM
    assert MODEL.price_stt(10).pool is BillingPool.GNANI


def test_an_unknown_llm_rate_still_records_the_tokens() -> None:
    charge = MODEL.price_llm(50_000)
    assert charge.raw_units == 50_000.0
    assert charge.credits == 0.0


def test_llm_credits_follow_the_rate_when_known() -> None:
    model = CostModel(llm_rate_per_1m_tokens=60.0)
    assert model.price_llm(1_000_000).credits == pytest.approx(60.0)


def test_placeholder_rates_mark_every_charge_as_estimated() -> None:
    """No number derived from a guessed rate may ever be presented as fact."""
    assert MODEL.rates_confirmed is False
    for charge in (MODEL.price_stt(5), MODEL.price_tts(10), MODEL.price_llm(10)):
        assert charge.estimated is True


def test_confirming_rates_clears_the_estimated_flag() -> None:
    model = CostModel(rates_confirmed=True)
    assert model.price_tts(10).estimated is False


@pytest.mark.parametrize(
    ("value", "increment", "expected"),
    [
        (0.0, 0.05, 0.0),
        (0.001, 0.05, 0.05),
        (0.05, 0.05, 0.05),
        (0.051, 0.05, 0.10),
        (1.0, 1.0, 1.0),
    ],
)
def test_round_up_to_increment(value: float, increment: float, expected: float) -> None:
    assert round_up_to_increment(value, increment) == pytest.approx(expected)


def test_a_non_positive_increment_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        round_up_to_increment(1.0, 0.0)


@pytest.mark.parametrize("method", ["price_stt", "price_tts", "price_llm"])
def test_negative_usage_is_rejected(method: str) -> None:
    with pytest.raises(ValueError, match="negative"):
        getattr(MODEL, method)(-1)


def test_the_observed_six_request_sample_is_in_the_right_ballpark() -> None:
    """Sanity check against the dashboard: 5 STT calls of 2.7 s plus 36 TTS chars ≈ ₹0.17.

    The placeholder rates were inferred from this figure, so this asserts internal
    consistency, not accuracy. It exists to fail loudly if someone edits a default rate
    without re-deriving it.
    """
    stt = sum(MODEL.price_stt(2.7).credits for _ in range(5))
    tts = MODEL.price_tts(36).credits
    assert stt + tts == pytest.approx(0.17, abs=0.02)
