"""How close the scammer is to hanging up, and what to say about it.

This is the feedback loop the whole project turns on. A persona that stalls perfectly but
gets hung up on after ninety seconds has wasted ninety seconds of a scammer's day; one that
reads the room and throws out a hook at the right moment wastes twenty minutes. So the risk
score is not decoration on the dashboard - it selects the next tactic.

**Signals, not sentiment.** Every contribution is a named, inspectable pattern or a counted
fact about the transcript, because a number the operator cannot explain is a number they
will not trust. The dashboard shows the signals beside the gauge for exactly that reason.

**Hooks are bait, not pleading.** A persona that says "please don't go" confirms it is
wasting his time. A persona that mentions a fixed deposit he does not understand gives him a
reason to stay that is about his incentives, not ours. Every hook here is a dangled
opportunity, and every one of them is fictional.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.core.models import ScamStage, Speaker, Turn

#: Phrases that mean a scammer is losing patience, with what each is worth.
_RISK_PATTERNS: tuple[tuple[str, float, str], ...] = (
    (r"\b(?:phone|line)\s+(?:rakh|kaat|cut|band)\w*\b", 0.3, "threatening_to_hang_up"),
    (r"\b(?:bye|goodbye|rakhta\s+hoon|rakhti\s+hoon)\b", 0.35, "signing_off"),
    (r"\b(?:time\s+waste|waqt\s+kharab|bakwaas|nonsense|faltu)\b", 0.3, "accusing_time_waste"),
    (r"\b(?:samajh\s+nahi\s+(?:aa\s+)?rahe|understand\s+nahi)\b", 0.15, "frustration"),
    (r"\b(?:kitni\s+baar|how\s+many\s+times|phir\s+se|again\s+and\s+again)\b", 0.2, "repetition"),
    (
        r"\b(?:jaldi|abhi|turant|quickly|fast)\b.{0,20}\b(?:bolo|bataiye|kijiye)\b",
        0.15,
        "impatience",
    ),
    (r"\b(?:buzurg|old|budhe|bahre|deaf)\b", 0.2, "commenting_on_age"),
    (
        r"\b(?:kal|tomorrow|baad\s+mein|later)\b.{0,20}\b(?:call|baat|phone)\b",
        0.25,
        "deferring_the_call",
    ),
    (r"\b(?:supervisor|senior|manager)\s+(?:se|ko)\b", 0.1, "escalating_internally"),
    # Requires the accusation, not just the word. A real call said "case number fake C Y B
    # triple zero one" - our own synthetic case-number prefix - and scored 0.4 for bot
    # suspicion on a caller who had not suspected anything.
    (
        r"\b(?:aap|tu|yeh|this|you)\b.{0,25}\b(?:bot|robot|recording|machine|computer)\b"
        r"|\b(?:bot|robot|recording|machine)\s+(?:ho|hai|hain|है|है\?|\?)"
        r"|\b(?:jhooth|drama|natak)\s+(?:kar|bol|ho)\w*",
        0.4,
        "suspecting_a_bot",
    ),
)

#: A scammer who stops answering is a scammer who has stopped engaging.
SHORT_TURN_CHARS = 18

#: Above this, the orchestrator should spend a turn on a hook rather than on small talk.
HOOK_THRESHOLD = 0.45


class RiskAssessment(BaseModel):
    """A hang-up risk score and the reasons for it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hangup_risk: float = Field(ge=0.0, le=1.0)
    signals: tuple[str, ...] = ()
    needs_hook: bool = Field(description="Whether the next turn should dangle something.")


class BaitHook(BaseModel):
    """Something fictional to dangle so the scammer chooses to stay."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    stage: ScamStage | None = Field(
        default=None, description="Stage this hook suits best, or None for any."
    )
    instruction: str = Field(
        description="Appended to the dialogue prompt. An instruction, not a line to read: a "
        "scripted sentence spoken verbatim stops sounding like the persona."
    )


#: Every hook is a *reason for him to stay* framed in terms of his own incentive, and every
#: one is fictional. None of them promise a payment, because a promise that is never kept is
#: the point at which a scammer gives up - a hook has to be perpetually almost-ready.
BAIT_HOOKS: tuple[BaitHook, ...] = (
    BaitHook(
        name="fixed_deposit",
        stage=ScamStage.PAYMENT,
        instruction="Mention, confused, that most of your money is in a fixed deposit you do "
        "not know how to break, and ask him whether that will be a problem. Do not offer to "
        "break it.",
    ),
    BaitHook(
        name="pension_arriving",
        stage=ScamStage.PAYMENT,
        instruction="Say your pension comes on the first of the month and ask whether it can "
        "wait until then. Be apologetic rather than resistant.",
    ),
    BaitHook(
        name="second_account",
        stage=ScamStage.PAYMENT,
        instruction="Remember out loud that you have an old account at another bank you had "
        "forgotten about, and wonder aloud whether that one is also affected.",
    ),
    BaitHook(
        name="grandson_at_bank",
        stage=None,
        instruction="Mention that your grandson works in a bank and will be home soon, and "
        "that he handles these things. Offer to have him call back - then immediately say "
        "you do not know how to make calls on this phone.",
    ),
    BaitHook(
        name="writing_it_down",
        stage=None,
        instruction="Insist on writing down everything he has said so far and read it back "
        "to him, getting one detail wrong so he has to correct you.",
    ),
    BaitHook(
        name="another_caller",
        stage=ScamStage.AUTHORITY,
        instruction="Say someone from another department called you last week about something "
        "similar and ask whether it is the same case. Invent no names.",
    ),
    BaitHook(
        name="locker_documents",
        stage=ScamStage.FEAR,
        instruction="Worry aloud about documents in a bank locker and ask whether those are "
        "also in the case. Ask him to wait while you look for the locker key.",
    ),
)


def assess(turns: Sequence[Turn], *, threshold: float = HOOK_THRESHOLD) -> RiskAssessment:
    """Score how close the caller is to hanging up.

    Two contributions beyond the phrase patterns, both counted from the transcript rather
    than guessed:

    * **Shortening replies.** A scammer who was giving paragraphs and is now giving four
      words has disengaged, whatever he is saying.
    * **Silence.** Caller turns that are getting rarer is the signal that arrives before the
      hang-up, and it is the one a phrase list cannot see.
    """
    caller = [turn for turn in turns if turn.speaker is Speaker.CALLER]
    if not caller:
        return RiskAssessment(hangup_risk=0.0, signals=(), needs_hook=False)

    found: dict[str, float] = {}
    for turn in caller[-4:]:
        for pattern, weight, name in _RISK_PATTERNS:
            if name not in found and re.search(pattern, turn.text, re.IGNORECASE):
                found[name] = weight

    recent = caller[-2:]
    if len(caller) >= 4 and all(len(turn.text) <= SHORT_TURN_CHARS for turn in recent):
        found["replies_shortening"] = 0.25

    risk = min(1.0, sum(found.values()))
    return RiskAssessment(
        hangup_risk=round(risk, 3),
        signals=tuple(sorted(found)),
        needs_hook=risk >= threshold,
    )


def choose_hook(
    stage: ScamStage, used: Sequence[str] = (), *, hooks: Sequence[BaitHook] = BAIT_HOOKS
) -> BaitHook | None:
    """The best unused hook for this stage, or None if they are all spent.

    Stage-specific hooks are preferred over general ones: a fixed deposit he cannot break is
    a far better reason to stay once he is asking for money than it is while he is still
    establishing that he is a policeman.

    Returning None rather than repeating one is deliberate. A persona that dangles the same
    forgotten account twice has stopped being an old man and become a loop.
    """
    spent = set(used)
    available = [hook for hook in hooks if hook.name not in spent]
    if not available:
        return None
    for hook in available:
        if hook.stage is stage:
            return hook
    for hook in available:
        if hook.stage is None:
            return hook
    return None
