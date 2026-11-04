"""Timbre v3.0 inline audio tags: the catalogue, and a filter for LLM output.

Two documented facts make this worth a module rather than a regex at the call site
(docs.gnani.ai/api/TTS/tts-audio-tags):

1. **Unknown tags are stripped silently but still billed.** "Inline tags count as billable
   characters... the tag text (including the angle brackets) is included in the character
   count." So an LLM that writes `<cough>` instead of `<coughs>` costs us eight characters
   and produces nothing. Over a long call that is real credit spent on silence.
2. **A tag applies from its position to the end of the input,** or until the next tag.
   There is no closing tag. One `<slow>` at the front of an utterance therefore slows the
   whole utterance for six billable characters, which is the cheapest elderly-voice effect
   available — but it also means a stray `<excited>` poisons everything after it.

Filtering here, in core, rather than trusting the prompt: the persona prompt asks for a
small tag vocabulary, and this is what makes the ask enforceable.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

#: The 13 tags Timbre v3.0 accepts, verbatim from the catalogue. Names are case-sensitive
#: and must match exactly, spaces included.
TIMBRE_V3_TAGS: frozenset[str] = frozenset(
    {
        "laugh",
        "giggle",
        "sigh",
        "exhales",
        "sniffles",
        "yawns",
        "coughs",
        "clears throat",
        "whispers",
        "shouting",
        "excited",
        "slow",
        "pauses",
    }
)

#: What an elderly male persona is allowed to do with his voice. `laugh`, `giggle`,
#: `shouting` and `excited` are excluded because the persona must never sound like it is
#: mocking the caller — the ethics constraints rule out taunting, and a laugh in the middle
#: of a scam call is a taunt. `whispers` is excluded because it is inaudible over a phone
#: band-pass, so it buys nothing and risks the scammer hanging up on dead air.
ELDER_TAGS: frozenset[str] = frozenset(
    {"sigh", "exhales", "coughs", "clears throat", "slow", "pauses"}
)

#: The grandson is younger and sharper, so he gets the sceptical end of the catalogue but
#: still no laughter: the relay is meant to waste a scammer's time, not to bait a reaction.
GRANDSON_TAGS: frozenset[str] = frozenset({"sigh", "exhales", "clears throat", "pauses"})

_TAG = re.compile(r"<([^<>]{1,24})>")


def sanitize_tags(text: str, allowed: Iterable[str]) -> tuple[str, tuple[str, ...]]:
    """Drop every tag not in `allowed`, returning the cleaned text and what was dropped.

    The dropped list is returned rather than logged here so that `app.core` stays free of
    I/O; the caller logs it with the `call_id`. A rising count of drops means the persona
    prompt has drifted and is burning characters.
    """
    permitted = {tag.strip() for tag in allowed}
    unknown = permitted - TIMBRE_V3_TAGS
    if unknown:
        raise ValueError(f"not Timbre v3.0 tags: {sorted(unknown)}")

    dropped: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in permitted:
            return match.group(0)
        dropped.append(name)
        return ""

    cleaned = _TAG.sub(replace, text)
    # Removing a tag leaves the spaces that surrounded it, and Timbre bills whitespace too.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, tuple(dropped)


def billable_characters(text: str) -> int:
    """What Timbre will charge for this text, tags and angle brackets included.

    A separate function from `len` so the billing assumption is stated in one place and can
    be corrected if the documented rule turns out to differ from the invoice.
    """
    return len(text)
