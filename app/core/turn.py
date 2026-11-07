"""Turn-taking arithmetic: when to speak, what to say first, and what to remember.

All pure. The orchestrator owns the sockets; this module owns the decisions, so every one
of them is a unit test rather than a timing-dependent integration test.

The thing that matters most here is `take_sentence`. A scammer hangs up on dead air, and the
largest latency win available is to stop waiting for the LLM to finish: the moment the first
sentence of a reply exists, it goes to TTS while the rest is still being generated. On a
measured 174-380 ms time-to-first-token that turns a two-second wait into about half of one.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.core.models import PersonaId, Speaker, Turn

#: What ends a sentence. `।` is the Devanagari danda: the persona writes Latin-script
#: Hinglish, but the LLM emits it often enough to be worth handling.
SENTENCE_ENDINGS = frozenset(".?!।…")

#: Below this, a fragment is not worth a synthesis request of its own — "Haan." as a first
#: chunk sounds like a glitch rather than a word, and each request has fixed overhead.
MIN_SENTENCE_CHARS = 12

#: Above this, stop waiting for punctuation. An LLM that runs on without a full stop would
#: otherwise hold the whole reply back, which is the exact failure this module exists to fix.
MAX_SENTENCE_CHARS = 180


def take_sentence(
    buffer: str,
    *,
    min_chars: int = MIN_SENTENCE_CHARS,
    max_chars: int = MAX_SENTENCE_CHARS,
) -> tuple[str, str] | None:
    """Split off the first speakable sentence, or return None if there is not one yet.

    Returns `(sentence, remainder)`. The sentence keeps its punctuation, because the TTS
    model's prosody depends on it.

    Three cases that a naive split on `.` gets wrong, all of which were cheaper to handle
    than to debug later:

    * **Inside an audio tag.** `<clears throat>` contains no terminator today, but a tag is
      a unit and splitting one in half would send `<clears` to the synthesiser, which bills
      for it and speaks nothing.
    * **A decimal point.** "2.5 lakh" and "₹1.5" are amounts a scammer says constantly.
      A terminator between two digits is not a sentence end.
    * **An ellipsis.** A run of two or more dots is a *hesitation*, not a sentence end -
      which matters enormously for this persona, because trailing off is most of what he
      does. Treating "..." as a boundary produced replies like "Inspector..." spoken on
      their own, and once one terse reply was in the history the model copied it for the
      rest of the call. A run of other terminators ("?!") still breaks, once, after the run.
    """
    if max_chars < min_chars:
        raise ValueError("max_chars must be >= min_chars")

    depth = 0
    for index, char in enumerate(buffer):
        if char == "<":
            depth += 1
            continue
        if char == ">":
            depth = max(0, depth - 1)
            continue
        if depth > 0 or char not in SENTENCE_ENDINGS:
            continue
        if _is_decimal_point(buffer, index):
            continue
        if _is_ellipsis(buffer, index):
            continue
        end = _end_of_terminator_run(buffer, index)
        if len(buffer[:end].strip()) < min_chars:
            continue
        return buffer[:end].strip(), buffer[end:].lstrip()

    if len(buffer.strip()) >= max_chars:
        # No punctuation in sight. Break at the last word boundary rather than mid-word.
        cut = buffer.rfind(" ", 0, max_chars)
        if cut > min_chars:
            return buffer[:cut].strip(), buffer[cut:].lstrip()
    return None


def _is_decimal_point(buffer: str, index: int) -> bool:
    """Whether `buffer[index]` is a decimal point rather than a full stop."""
    if buffer[index] != ".":
        return False
    before = buffer[index - 1] if index > 0 else ""
    after = buffer[index + 1] if index + 1 < len(buffer) else ""
    return before.isdigit() and after.isdigit()


def _is_ellipsis(buffer: str, index: int) -> bool:
    """Whether this dot belongs to a run of two or more - a trailing-off, not a full stop."""
    if buffer[index] not in ".…":
        return False
    if buffer[index] == "…":
        return True
    before = buffer[index - 1] == "." if index > 0 else False
    after = buffer[index + 1] == "." if index + 1 < len(buffer) else False
    return before or after


def _end_of_terminator_run(buffer: str, index: int) -> int:
    """Index just past a run of terminators starting at `index`."""
    end = index + 1
    while end < len(buffer) and buffer[end] in SENTENCE_ENDINGS:
        end += 1
    return end


def flush_tail(buffer: str, *, min_chars: int = 2) -> str | None:
    """Whatever is left when the stream ends, if it is worth speaking.

    A trailing fragment shorter than this is punctuation or whitespace the model emitted
    after its last sentence; synthesising it would cost characters and produce a click.
    """
    tail = buffer.strip()
    return tail if len(tail) >= min_chars else None


class FillerRotation:
    """Picks the stall phrase that plays while the LLM is still thinking.

    Round-robin over an order shuffled once per call. Round-robin rather than random so
    every phrase is used (and therefore every phrase is worth pre-caching), shuffled per
    call so a scammer who gets through to us twice does not hear the same six openings in
    the same six positions.

    Seeded from the call id, so a replayed call produces the same fillers as the live one.
    """

    def __init__(self, phrases: Sequence[str], *, seed: str = "") -> None:
        if not phrases:
            raise ValueError("need at least one filler phrase")
        self._phrases = list(phrases)
        random.Random(seed).shuffle(self._phrases)
        self._index = 0

    @property
    def phrases(self) -> tuple[str, ...]:
        """Every phrase in this call's order. Used to warm the TTS cache."""
        return tuple(self._phrases)

    def next(self) -> str:
        """The next phrase, cycling."""
        phrase = self._phrases[self._index % len(self._phrases)]
        self._index += 1
        return phrase


class TurnBudget(BaseModel):
    """How much conversation history goes to the dialogue model.

    Trimmed by turn count rather than tokens: the personas speak in two or three short
    sentences, so turns are a good enough proxy, and counting tokens would mean shipping a
    tokeniser for a vendor whose tokeniser we do not know.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_turns: int = Field(default=10, ge=2, description="Most recent turns kept verbatim.")
    max_chars: int = Field(
        default=1800,
        ge=200,
        description="Hard ceiling on the window. Sized against the token budget rather than "
        "against what the model could take: history is as large a per-turn cost as the "
        "system prompt, and dialogue and analysis share one 8000-tokens-per-minute account. "
        "Five exchanges is ample for staying in character and not repeating oneself; the "
        "scammer's opening pitch has already been extracted by the analysis layer by then.",
    )


def recent_turns(turns: Sequence[Turn], budget: TurnBudget) -> tuple[Turn, ...]:
    """The tail of the conversation that fits inside `budget`, oldest first.

    Dropping the oldest turns loses the scammer's opening pitch, which is exactly the part
    the analysis layer has already extracted and stored. The dialogue model only needs
    enough to stay in character and not repeat itself.
    """
    window: list[Turn] = []
    total = 0
    for turn in reversed(turns[-budget.max_turns :]):
        total += len(turn.text)
        if window and total > budget.max_chars:
            break
        window.append(turn)
    return tuple(reversed(window))


def is_handoff_due(turns: Sequence[Turn], *, after_persona_turns: int) -> bool:
    """Whether the current persona has said enough and should hand the phone over.

    Counts only the persona's own turns since the last handoff. The real trigger in the
    brief is hang-up risk, which arrives with the analysis layer; this is the floor that
    guarantees the relay happens at all in a demo where the scammer stays calm.
    """
    if after_persona_turns <= 0:
        raise ValueError("after_persona_turns must be positive")
    current: PersonaId | None = None
    count = 0
    for turn in turns:
        if turn.speaker is not Speaker.PERSONA:
            continue
        if turn.persona != current:
            current = turn.persona
            count = 0
        count += 1
    return count >= after_persona_turns


#: Things a caller says when he wants the other person back. Matched on the caller's last
#: turn, because a request to speak to someone is the clearest handoff signal there is - far
#: clearer than any turn count - and ignoring it is conspicuous.
_ASKS_FOR_ELDER = re.compile(
    r"\b(?:dada|dadaji|dada\s*ji|grandfather|uncle|buzurg|budhe?\s*(?:aadmi|sahab)?)\b"
    r".{0,30}\b(?:se|ko|baat|bula|karwa|do|dijiye|give|put|speak|talk)\w*"
    r"|\b(?:baat\s*karwa|phone\s*(?:do|dijiye|de)|put\s+(?:him|them)\s+(?:on|back))\w*",
    re.IGNORECASE,
)

#: And the reverse: a caller who wants the younger one, or says the elder cannot follow.
_ASKS_FOR_YOUNGER = re.compile(
    r"\b(?:pot[aei]|grandson|beta|young|ladke?)\b.{0,30}\b(?:se|ko|baat|bula|karwa|do|dijiye)\w*"
    r"|\b(?:koi\s+aur|someone\s+else|samajh\s+nahi\s+(?:aa\s+)?rah[ae])\b",
    re.IGNORECASE,
)


def asks_for_elder(text: str) -> bool:
    """Whether the caller just asked to speak to the older person."""
    return bool(_ASKS_FOR_ELDER.search(text))


def asks_for_younger(text: str) -> bool:
    """Whether the caller just asked for someone other than the person holding the phone."""
    return bool(_ASKS_FOR_YOUNGER.search(text))
