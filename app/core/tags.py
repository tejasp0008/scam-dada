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


#: A tag that only inserts silence. Useful mid-utterance, dead weight at the very start.
_SILENCE_TAG = "pauses"


def tidy_for_speech(text: str) -> str:
    """Normalise a model's reply into something worth sending to the synthesiser.

    Two things observed on real calls, both of which cost billable characters and give
    nothing back:

    * **Line breaks.** The dialogue model emits paragraph breaks even when told to produce
      plain speech. They are billed, they are not spoken, and they are invisible in a log,
      which is the worst combination.
    * **A leading `<pauses>`.** The model latched onto opening every reply with it. Mid
      sentence that tag is useful; as the first thing in an utterance it is silence played
      immediately after the filler phrase already covered that beat - so it is dead air
      bolted onto the one place in the call where dead air is most expensive.
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    while True:
        match = re.match(r"<([^<>]{1,24})>\s*", cleaned)
        if match is None or match.group(1) != _SILENCE_TAG:
            break
        cleaned = cleaned[match.end() :]
    return cleaned.strip()


def has_speech(text: str) -> bool:
    """Is there anything here a listener would hear as a word?

    A reply that is only audio tags is billed in full and lands as a breath or a silence
    with no words around it. One scripted call ended with the persona's whole turn being
    `<pauses>`, which reads on the transcript as a reply and sounds like a dropped line.
    """
    return bool(re.sub(r"<[^<>]{1,24}>", " ", text).strip())


def billable_characters(text: str) -> int:
    """What Timbre will charge for this text, tags and angle brackets included.

    A separate function from `len` so the billing assumption is stated in one place and can
    be corrected if the documented rule turns out to differ from the invoice.
    """
    return len(text)
