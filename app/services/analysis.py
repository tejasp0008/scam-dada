"""The analysis pass: a second opinion on the transcript, off the dialogue path.

Two rules shape this module, and both are about what must never happen to a live call.

**It must never block the persona.** Analysis runs as its own task per turn. The persona's
reply does not wait for it, is not affected by its result, and does not fail when it fails.
A model that is slow, rate-limited or down degrades the dashboard, never the call.

**It must never be believed over the deterministic pass.** `app.core.stages`,
`app.core.extract` and `app.core.risk` are reproducible; a model is not. So the keyword
reading is computed first and the model is asked separately, and where they disagree about
the stage the disagreement is recorded rather than resolved silently. Entities the model
reports arrive with `source=LLM` and its own confidence, so the evidence pack always says
which machine found what.

One more thing worth stating plainly: the transcript is **untrusted input**. It is a
recording of a stranger trying to manipulate whoever is listening, and some of what he says
is phrased as instructions. The analysis prompt says to ignore instructions found in the
transcript, and the output is a validated schema rather than free text, so a transcript that
tries to redirect the analyst produces a schema violation rather than an action.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.core import extract, risk, stages
from app.core.models import Entity, EntitySource, EntityType, ScamStage, Speaker, Turn
from app.providers.base import ChatMessage, LLMProvider, ProviderError, Role
from app.services.prompts import load_prompt_file

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT_FILE = "analysis.v1.md"

#: How much transcript the analyst sees. Enough to judge the arc, short enough to keep the
#: request inside a token budget that is shared with the dialogue model.
ANALYSIS_WINDOW_TURNS = 12

#: Entities below this are noise in an evidence pack a human will read.
MIN_LLM_CONFIDENCE = 0.5


class LlmEntity(BaseModel):
    """An entity as the model reports it, before it is trusted."""

    model_config = ConfigDict(extra="ignore")

    entity_type: str
    value: str
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class LlmAnalysis(BaseModel):
    """The analyst's reply. Validated, so a malformed answer is an error not a surprise."""

    model_config = ConfigDict(extra="ignore")

    stage: str = ScamStage.HOOK.value
    stage_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    entities: list[LlmEntity] = Field(default_factory=list)
    tactics: list[str] = Field(default_factory=list)
    notes: str = ""


class TurnAnalysis(BaseModel):
    """What the deterministic pass and the model each concluded, side by side."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    turn_seq: int
    stage: ScamStage
    stage_confidence: float = Field(ge=0.0, le=1.0)
    pressure_score: float = Field(ge=0.0, le=1.0)
    pressure_signals: tuple[str, ...] = ()
    hangup_risk: float = Field(ge=0.0, le=1.0)
    risk_signals: tuple[str, ...] = ()
    needs_hook: bool = False
    entities: tuple[Entity, ...] = ()
    tactics: tuple[str, ...] = ()
    notes: str = ""
    model_stage: ScamStage | None = Field(
        default=None, description="What the model said, when it differs from the keyword pass."
    )
    model_failed: bool = Field(
        default=False, description="True when only the deterministic pass contributed."
    )


def _window(turns: Sequence[Turn]) -> list[Turn]:
    return list(turns[-ANALYSIS_WINDOW_TURNS:])


def _transcript_text(turns: Sequence[Turn]) -> str:
    lines = []
    for turn in turns:
        who = "CALLER" if turn.speaker is Speaker.CALLER else "PERSONA"
        lines.append(f"{who}: {turn.text}")
    return "\n".join(lines)


def _caller_texts(turns: Sequence[Turn]) -> list[str]:
    return [turn.text for turn in turns if turn.speaker is Speaker.CALLER]


def _coerce_stage(value: str) -> ScamStage | None:
    try:
        return ScamStage(value.strip().casefold())
    except ValueError:
        return None


def _coerce_entities(reported: Sequence[LlmEntity], turn_seq: int) -> tuple[Entity, ...]:
    """Keep only entities with a type we model and a confidence worth reading."""
    kept: list[Entity] = []
    for item in reported:
        try:
            entity_type = EntityType(item.entity_type.strip().casefold())
        except ValueError:
            # A type outside the schema is the model inventing a category. Dropped rather
            # than coerced: an evidence pack with an "aadhaar" row we never modelled would
            # be worse than one row short.
            logger.info("dropping an unmodelled entity type", extra={"type": item.entity_type})
            continue
        value = item.value.strip()
        if not value or item.confidence < MIN_LLM_CONFIDENCE:
            continue
        kept.append(
            Entity(
                entity_type=entity_type,
                value=value,
                source=EntitySource.LLM,
                turn_seq=turn_seq,
                confidence=item.confidence,
            )
        )
    return tuple(kept)


def deterministic_analysis(
    turns: Sequence[Turn], *, turn_seq: int, furthest: ScamStage | None = None
) -> TurnAnalysis:
    """The reproducible reading. Available even with no LLM at all."""
    window = _window(turns)
    caller = _caller_texts(window)
    assessment = stages.assess(caller, furthest=furthest)
    risk_assessment = risk.assess(window)
    found = extract.extract_all(caller)
    return TurnAnalysis(
        turn_seq=turn_seq,
        stage=assessment.stage,
        stage_confidence=assessment.confidence,
        pressure_score=assessment.pressure_score,
        pressure_signals=assessment.signals,
        hangup_risk=risk_assessment.hangup_risk,
        risk_signals=risk_assessment.signals,
        needs_hook=risk_assessment.needs_hook,
        entities=found,
        model_failed=True,
    )


class CallAnalyst:
    """Runs the analysis pass for one call, keeping what must not go backwards."""

    def __init__(self, llm: LLMProvider, *, model: str | None = None) -> None:
        self._llm = llm
        self._model = model
        self._furthest: ScamStage | None = None
        self.model_failures = 0

    @property
    def furthest_stage(self) -> ScamStage | None:
        """The furthest stage seen so far, so the badge never retreats."""
        return self._furthest

    def _messages(self, turns: Sequence[Turn]) -> list[ChatMessage]:
        return [
            ChatMessage(role=Role.SYSTEM, content=load_prompt_file(ANALYSIS_PROMPT_FILE)),
            # The transcript is quoted as user content, which is where untrusted input
            # belongs: it is a recording of a stranger trying to manipulate a listener, and
            # some of it is phrased as instructions.
            ChatMessage(role=Role.USER, content=_transcript_text(turns)),
        ]

    async def analyse(self, turns: Sequence[Turn], *, turn_seq: int) -> TurnAnalysis:
        """Analyse the call so far. Never raises: a failure degrades to the keyword pass."""
        base = deterministic_analysis(turns, turn_seq=turn_seq, furthest=self._furthest)
        self._furthest = stages.furthest_stage(self._furthest, base.stage)

        window = _window(turns)
        if not _caller_texts(window):
            return base

        try:
            reported = await self._llm.complete_json(
                self._messages(window), LlmAnalysis, model=self._model, temperature=0.0
            )
        except ProviderError as exc:
            self.model_failures += 1
            logger.info("analysis model failed; keyword pass only", extra={"error": str(exc)})
            return base

        model_stage = _coerce_stage(reported.stage)
        if model_stage is not None:
            self._furthest = stages.furthest_stage(self._furthest, model_stage)

        # The deterministic stage wins. The model's reading is carried alongside so the
        # dashboard can show a disagreement rather than hiding it behind an average.
        stage = stages.furthest_stage(base.stage, model_stage) if model_stage else base.stage
        merged = _merge_entities(base.entities, _coerce_entities(reported.entities, turn_seq))

        return base.model_copy(
            update={
                "stage": stage,
                # The higher pressure reading, because under-reporting pressure is the
                # expensive direction: it is what decides whether a hook is thrown.
                "pressure_score": max(base.pressure_score, round(reported.pressure, 3)),
                "entities": merged,
                "tactics": tuple(t.strip() for t in reported.tactics[:4] if t.strip()),
                "notes": reported.notes.strip()[:140],
                "model_stage": model_stage if model_stage is not base.stage else None,
                "model_failed": False,
            }
        )


def _merge_entities(
    deterministic: Sequence[Entity], from_model: Sequence[Entity]
) -> tuple[Entity, ...]:
    """Keep both sources, but never two rows for the same identifier.

    Where both found the same value the regex row wins: it is reproducible and certain, and
    an evidence pack listing one UPI handle twice at two confidences looks like a bug to the
    person reading it.
    """
    merged = list(deterministic)
    seen = {(e.entity_type, e.value.casefold()) for e in deterministic}
    for entity in from_model:
        key = (entity.entity_type, entity.value.casefold())
        if key not in seen:
            seen.add(key)
            merged.append(entity)
    return tuple(merged)
