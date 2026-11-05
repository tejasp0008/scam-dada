"""Where in the script a scammer currently is, and how hard he is pushing.

A social-engineering call follows a shape: establish authority, manufacture fear, isolate
the victim from anyone who would talk them out of it, then extract payment. Naming the stage
live is what makes the dashboard legible to someone who has never heard one of these calls,
and it is also what feature 3 keys its retention tactics off.

**Deterministic first.** Keyword evidence is computed here, in pure code, so every stage
badge and every pressure number on the dashboard can be reproduced from the transcript in a
test. `app.services.analysis` then asks the model for a second opinion and reconciles the
two; where they disagree, the dashboard shows both, because an invented agreement would be
worse than a visible disagreement.

**Stages do not go backwards.** A scammer who has reached payment and circles back to
authority has not regressed - he is reinforcing. Tracking the furthest stage reached rather
than the last one matched is what stops the badge flickering between two stages for the rest
of the call.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.core.models import ScamStage

#: Phrase evidence per stage, as a scammer actually says it in Hinglish and English. Each
#: pattern carries a weight, because "warrant" is far stronger evidence of the fear stage
#: than "problem" is.
_STAGE_PATTERNS: dict[ScamStage, tuple[tuple[str, float], ...]] = {
    ScamStage.HOOK: (
        (r"\b(?:hello|hallo)\b.{0,40}\b(?:sir|madam|ji)\b", 0.5),
        (r"\bbol\s+raha\s+hoon\b", 0.6),
        (r"\b(?:parcel|courier|packet|consignment)\b", 0.8),
        (r"\b(?:kyc|verification|verify)\b", 0.5),
        (r"\baapke\s+naam\s+par\b", 0.8),
        (r"\b(?:calling|speaking)\s+from\b", 0.6),
    ),
    ScamStage.AUTHORITY: (
        (r"\b(?:inspector|officer|commissioner|constable|dcp|acp|sho)\b", 0.9),
        (r"\bcyber\s*(?:crime|cell)\b", 0.9),
        (r"\b(?:police|crime\s+branch|cbi|trai|customs|narcotics)\b", 0.8),
        (r"\b(?:income\s*tax|enforcement|rbi|reserve\s+bank)\b", 0.7),
        (r"\b(?:department|vibhag)\b", 0.3),
        (r"\b(?:badge|staff|employee)\s*(?:number|code|id)\b", 0.4),
    ),
    ScamStage.FEAR: (
        (r"\b(?:warrant|arrest|giraftar|jail|hirasat)\b", 1.0),
        (r"\b(?:case|fir|complaint)\s*(?:number|no\.?|darj|register)\b", 0.8),
        (r"\b(?:illegal|drugs|narcotics|contraband|money\s+laundering)\b", 0.8),
        (r"\b(?:court|adalat|summons|notice)\b", 0.6),
        (r"\b(?:freeze|block|band)\s*(?:ho|kar|karna)?\b.{0,20}\baccount\b", 0.8),
        (r"\b(?:serious|gambhir|bada)\s+(?:matter|maamla|problem)\b", 0.5),
        (r"\b(?:name|naam)\s+(?:par|mein)\b.{0,30}\b(?:case|fir|warrant)\b", 0.8),
    ),
    ScamStage.ISOLATION: (
        (r"\bkisi\s+ko\s+(?:bataiye|bataye|batana)\s*mat\b", 1.0),
        (r"\b(?:do\s*not|don'?t|mat)\s+(?:tell|disclose|share)\b", 0.9),
        (r"\bphone\s+(?:mat|na)\s+(?:kaatiye|rakhiye|cut)\b", 0.8),
        (r"\b(?:confidential|secret|gupt)\b", 0.7),
        (r"\b(?:akele|alone|privately)\b", 0.5),
        (r"\b(?:family|parivar|beta|bete|bahu|ghar)\s*(?:ko|mein)?\b.{0,20}\bmat\b", 0.7),
        (r"\bline\s+(?:pe|par)\s+(?:rahiye|rehna|bane)\b", 0.6),
    ),
    ScamStage.PAYMENT: (
        (r"\b(?:upi|paytm|gpay|phonepe|google\s+pay)\b", 0.9),
        # Either order. A real call said "chhota amount bhej dijiye", which a
        # verb-then-noun pattern missed entirely - and that was the line that should have
        # moved the badge to payment.
        (r"\b(?:transfer|bhej\w*|send|deposit)\b.{0,30}\b(?:amount|paisa|rupaye|rs)\b", 0.9),
        (r"\b(?:amount|paisa|rupaye|payment)\b.{0,30}\b(?:transfer|bhej\w*|send|kar\s+d)\w*", 0.9),
        (r"\b(?:lakh|hazaar|hazar|crore)\b", 0.7),
        (r"\b(?:account|khata)\s*(?:number|no\.?)\b", 0.8),
        (r"\b(?:ifsc|neft|imps|rtgs)\b", 0.9),
        (r"\b(?:otp|pin|cvv)\b", 0.9),
        (r"\b(?:refundable|security\s+deposit|processing\s+fee|charge)\b", 0.7),
        (r"@\w{2,20}\b", 0.6),
    ),
}

#: Pressure signals: not what the scammer wants, but how hard he is pushing for it. These
#: are the ones that correlate with a victim being rushed past their own judgement.
_PRESSURE_PATTERNS: tuple[tuple[str, float, str], ...] = (
    (r"\b(?:abhi|turant|immediately|right\s+now|jaldi)\b", 0.18, "urgency"),
    (r"\b(?:\d+|ek|do|paanch|das)\s*(?:minute|minut|ghante|hour)\b", 0.15, "deadline"),
    (r"\b(?:last|final|aakhri)\s+(?:chance|warning|mauka)\b", 0.25, "final_warning"),
    (r"\b(?:warrant|arrest|giraftar|jail)\b", 0.25, "threat_of_arrest"),
    (r"\bkisi\s+ko\s+(?:bataiye|bataye|batana)\s*mat\b", 0.2, "secrecy_demand"),
    (r"\b(?:phone|line)\s+(?:mat|na)\s+(?:kaatiye|rakhiye|cut)\b", 0.2, "do_not_hang_up"),
    (r"\b(?:otp|pin|cvv|password)\b", 0.2, "credential_request"),
    (r"\b(?:install|download|anydesk|teamviewer|quick\s*support)\b", 0.25, "remote_access"),
    (r"!{2,}|\b(?:suniye|listen)\b.{0,10}\b(?:suniye|listen)\b", 0.1, "insistence"),
)

#: Order of the script. Used to decide whether a match is progress or reinforcement.
STAGE_ORDER: tuple[ScamStage, ...] = (
    ScamStage.HOOK,
    ScamStage.AUTHORITY,
    ScamStage.FEAR,
    ScamStage.ISOLATION,
    ScamStage.PAYMENT,
)

#: The most certainty keyword matching is allowed to claim. Deliberately short of one.
MAX_KEYWORD_CONFIDENCE = 0.95

#: Evidence below this is not worth moving the badge for. One weak keyword match in a long
#: call is noise, and a badge that flickers is a badge nobody reads.
MIN_STAGE_SCORE = 0.6


class StageAssessment(BaseModel):
    """What the deterministic pass believes, and on what evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: ScamStage
    confidence: float = Field(ge=0.0, le=1.0)
    pressure_score: float = Field(ge=0.0, le=1.0)
    signals: tuple[str, ...] = Field(
        default=(), description="Named pressure signals, so the gauge is auditable."
    )
    scores: dict[ScamStage, float] = Field(
        default_factory=dict, description="Raw evidence per stage, for the dashboard tooltip."
    )


def stage_scores(texts: Sequence[str]) -> dict[ScamStage, float]:
    """Accumulated keyword evidence per stage over a whole transcript."""
    totals = dict.fromkeys(STAGE_ORDER, 0.0)
    for text in texts:
        for stage, patterns in _STAGE_PATTERNS.items():
            for pattern, weight in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    totals[stage] += weight
    return totals


def pressure(texts: Sequence[str]) -> tuple[float, tuple[str, ...]]:
    """How hard the caller is pushing, in [0, 1], with the signals that contributed.

    Signals are counted once each however often they recur: a scammer who says "abhi" nine
    times is urgent, not nine times as urgent, and letting repeats accumulate would peg the
    gauge at 1.0 on every call and make it useless.
    """
    found: dict[str, float] = {}
    for text in texts:
        for pattern, weight, name in _PRESSURE_PATTERNS:
            if name not in found and re.search(pattern, text, re.IGNORECASE):
                found[name] = weight
    return min(1.0, sum(found.values())), tuple(sorted(found))


def assess(texts: Sequence[str], *, furthest: ScamStage | None = None) -> StageAssessment:
    """The deterministic reading of a transcript so far.

    `furthest` is the stage previously reached, passed back in so the badge never retreats.
    """
    scores = stage_scores(texts)
    score, signals = pressure(texts)

    # The *latest* stage with real evidence, not the highest-scoring one. A scammer who
    # invokes the police in every sentence accumulates authority evidence indefinitely, so
    # picking the maximum leaves the badge stuck on `authority` while he is asking for a UPI
    # transfer. Once he is asking for money he is at payment, whatever else he is also saying.
    best = ScamStage.HOOK
    best_score = 0.0
    for stage in STAGE_ORDER:
        if scores[stage] >= MIN_STAGE_SCORE:
            best, best_score = stage, scores[stage]

    if furthest is not None and STAGE_ORDER.index(furthest) > STAGE_ORDER.index(best):
        best = furthest
        best_score = max(best_score, scores[furthest])

    # Evidence saturating towards one, never reaching it. Keyword matching does not justify
    # certainty however many keywords match, so the curve is asymptotic rather than clipped:
    # a clipped score reads as "definitely" on the dashboard, which would be a lie.
    confidence = 1.0 - math.exp(-best_score / 2.0) if best_score else 0.2
    # Clamped below one, not merely asymptotic: at twenty matching turns the curve rounds to
    # 1.0 at three decimal places, and a dashboard reading "100%" from keyword matching is
    # a claim this method cannot support.
    confidence = min(confidence, MAX_KEYWORD_CONFIDENCE)
    return StageAssessment(
        stage=best,
        confidence=round(confidence, 3),
        pressure_score=round(score, 3),
        signals=signals,
        scores={stage: round(value, 3) for stage, value in scores.items()},
    )


def furthest_stage(previous: ScamStage | None, current: ScamStage) -> ScamStage:
    """The later of two stages. A scammer circling back is reinforcing, not regressing."""
    if previous is None:
        return current
    return max(previous, current, key=STAGE_ORDER.index)
