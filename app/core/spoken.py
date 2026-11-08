"""Digits as a person reads them aloud.

Prisma transcribes a number read out digit by digit as words, not digits: an account number
dictated over the phone arrives as "zero zero one one two two". Two parts of this project
need to undo that - the canary matcher, so a leaked identifier is recognised however it was
spoken, and the extractor, so a destination account read aloud is still captured - and
neither should own the table. `app.core.canary` owned it first; it lives here because an
account number is not a canary concern.

Hinglish and English alternates for the same digit, because the caller switches mid-number.
"""

from __future__ import annotations

import re

#: Including the "oh" that stands in for zero and the "o" a transcriber sometimes produces.
SPOKEN_DIGITS: dict[str, str] = {
    "zero": "0",
    "oh": "0",
    "shunya": "0",
    "one": "1",
    "ek": "1",
    "two": "2",
    "do": "2",
    "three": "3",
    "teen": "3",
    "four": "4",
    "char": "4",
    "chaar": "4",
    "five": "5",
    "panch": "5",
    "paanch": "5",
    "six": "6",
    "chhe": "6",
    "seven": "7",
    "saat": "7",
    "eight": "8",
    "aath": "8",
    "nine": "9",
    "nau": "9",
}

#: Said before a digit to repeat it: "triple zero one".
REPEATERS: dict[str, int] = {"double": 2, "triple": 3, "thrice": 3}

_WORD = re.compile(r"[a-z]+|\d+")
_TOKEN = re.compile(r"[a-z]+|\d+", re.IGNORECASE)


def digit_run(text: str) -> str:
    """The digits in `text`, reading number words as digits and stopping at anything else.

    Built for a number being dictated, so it is deliberately unforgiving: the run ends at
    the first token that is neither a digit, a digit word nor a repeater. That is what keeps
    "one one two two hai, IFSC FAKE0000777" from swallowing the sentence after it, and what
    makes "do lakh" stop at "lakh" instead of yielding "2".

    Returns an empty string if the text does not start with a number at all.
    """
    out: list[str] = []
    repeat = 1
    for token in _WORD.findall(text.casefold()):
        if token in REPEATERS:
            repeat = REPEATERS[token]
            continue
        if token.isdigit():
            out.append(token * repeat)
        elif (digit := SPOKEN_DIGITS.get(token)) is not None:
            out.append(digit * repeat)
        else:
            break
        repeat = 1
    return "".join(out)


def digit_run_after(text: str, *, skip_words: int = 3) -> str:
    """The first dictated number in `text`, allowing a few words before it starts.

    A caller does not go straight from "account number" to the digits: he says "account
    number likh lijiye, zero zero..." or "account number yeh hai". `digit_run` stops at the
    first word that is not part of a number, which is what keeps it from running on - so the
    words *before* the number have to be stepped over here instead.

    `skip_words` is small on purpose. Allowing many words would let the keyword reach into
    the next clause and claim a number that belongs to something else, which is the failure
    this whole pattern exists to avoid.
    """
    for index, token in enumerate(_TOKEN.finditer(text)):
        if index > skip_words:
            return ""
        if run := digit_run(text[token.start() :]):
            return run
    return ""
