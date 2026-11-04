"""What a provider call costs, as pure arithmetic.

Separate from the budget guard on purpose: the guard does I/O (Redis counters, usage rows)
while the conversion from "2.7 seconds of audio" to "0.05 billed minutes" to "credits" is
arithmetic that should be provable in a unit test. Gnani bills STT in rounded-up minutes,
not seconds, which is the kind of rule that quietly costs 10x if it is implemented as a
division.

Two separate pools are tracked because they are separate wallets: Gnani's programme credits
(denominated in rupees, 1 credit = ₹1) and the LLM vendor's own billing, which does not
touch Gnani credits at all.
"""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class BillingPool(StrEnum):
    """A wallet. Spending from one never affects the other."""

    GNANI = "gnani"
    LLM = "llm"


class UnitType(StrEnum):
    """What we measured, before any billing rule is applied."""

    STT_SECONDS = "stt_seconds"
    TTS_CHARS = "tts_chars"
    LLM_TOKENS = "llm_tokens"


class BilledUnit(StrEnum):
    """What the vendor actually charges for, after rounding."""

    STT_MINUTES = "stt_minutes"
    TTS_CHARS = "tts_chars"
    LLM_TOKENS = "llm_tokens"


class Charge(BaseModel):
    """One priced provider call. Raw and billed units are both kept.

    Keeping both is what makes the rate conversion auditable: the dashboard's credit figure
    can be re-derived from the raw measurement at any time, and a wrong rate can be
    corrected retroactively without having lost the underlying seconds.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pool: BillingPool
    raw_unit: UnitType
    raw_units: float = Field(ge=0.0)
    billed_unit: BilledUnit
    billed_units: float = Field(ge=0.0)
    credits: float = Field(ge=0.0, description="Rupees, since 1 Gnani credit = ₹1.")
    estimated: bool = Field(
        description="True while the rate behind this charge is an unconfirmed placeholder."
    )


def round_up_to_increment(value: float, increment: float) -> float:
    """Round `value` up to the next whole multiple of `increment`.

    The epsilon stops a value that is already an exact multiple from being pushed to the
    next one by float representation error (0.05 / 0.05 can evaluate to 1.0000000000000002).
    """
    if increment <= 0.0:
        raise ValueError("increment must be positive")
    steps = math.ceil(value / increment - 1e-9)
    return round(steps * increment, 6)


class CostModel(BaseModel):
    """Converts measured usage into billed units and credits.

    Rates default to placeholders inferred from a six-request sample read off the usage
    dashboard. Every charge produced while `rates_confirmed` is false is flagged
    `estimated=True`, so no number derived from a guessed rate can be presented as fact.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Gnani bills STT per minute, rounded up. A 2.7 s call appeared on the dashboard as
    # 0.05 minutes, which implies a 3-second increment (0.05 min) and the same as a floor.
    stt_rate_per_minute: float = Field(default=0.45, ge=0.0)
    stt_billing_increment_minutes: float = Field(default=0.05, gt=0.0)
    stt_minimum_billed_minutes: float = Field(default=0.05, ge=0.0)

    # 36 characters of TTS cost roughly a third of ₹0.17, read off a bar chart.
    tts_rate_per_1k_chars: float = Field(default=1.58, ge=0.0)

    # The LLM is billed by its own vendor, not from Gnani credits. Zero means "unknown",
    # which yields zero-credit charges while still recording the tokens.
    llm_rate_per_1m_tokens: float = Field(default=0.0, ge=0.0)

    rates_confirmed: bool = Field(
        default=False,
        description="Set true only once the rates are copied from the vendor's pricing page.",
    )

    def price_stt(self, audio_seconds: float) -> Charge:
        """Price transcription of `audio_seconds` of audio."""
        if audio_seconds < 0.0:
            raise ValueError("audio_seconds must not be negative")
        if audio_seconds == 0.0:
            billed_minutes = 0.0
        else:
            billed_minutes = max(
                self.stt_minimum_billed_minutes,
                round_up_to_increment(audio_seconds / 60.0, self.stt_billing_increment_minutes),
            )
        return Charge(
            pool=BillingPool.GNANI,
            raw_unit=UnitType.STT_SECONDS,
            raw_units=audio_seconds,
            billed_unit=BilledUnit.STT_MINUTES,
            billed_units=billed_minutes,
            credits=round(billed_minutes * self.stt_rate_per_minute, 6),
            estimated=not self.rates_confirmed,
        )

    def price_tts(self, characters: int) -> Charge:
        """Price synthesis of `characters` characters.

        Audio tag characters are included: the docs state tags count as billable, so the
        caller passes the string exactly as it will be sent.
        """
        if characters < 0:
            raise ValueError("characters must not be negative")
        return Charge(
            pool=BillingPool.GNANI,
            raw_unit=UnitType.TTS_CHARS,
            raw_units=float(characters),
            billed_unit=BilledUnit.TTS_CHARS,
            billed_units=float(characters),
            credits=round(characters / 1000.0 * self.tts_rate_per_1k_chars, 6),
            estimated=not self.rates_confirmed,
        )

    def price_llm(self, tokens: int) -> Charge:
        """Price `tokens` of LLM usage, against the LLM's own pool."""
        if tokens < 0:
            raise ValueError("tokens must not be negative")
        return Charge(
            pool=BillingPool.LLM,
            raw_unit=UnitType.LLM_TOKENS,
            raw_units=float(tokens),
            billed_unit=BilledUnit.LLM_TOKENS,
            billed_units=float(tokens),
            credits=round(tokens / 1_000_000.0 * self.llm_rate_per_1m_tokens, 6),
            estimated=not self.rates_confirmed,
        )
