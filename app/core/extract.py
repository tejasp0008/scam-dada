"""Pulling identifiers out of a scammer's words, deterministically.

Regex only, and that is a deliberate division of labour: anything this module reports is
reproducible, auditable and free, and it lands in the evidence pack with
`source=EntitySource.REGEX` and full confidence. The LLM pass in `app.services.analysis`
adds what regex cannot see — a name spoken as "Sharma bol raha hoon", an organisation
implied rather than stated — and lands with `source=LLM` and a confidence below one, so a
reader of the evidence pack can always tell which machine said what.

Two things make this harder than a bank of patterns:

* **Hinglish amounts.** A scammer says "do lakh", "paanch hazaar", "2.5 lakh" and
  "₹50,000" interchangeably, often in the same sentence. The number words matter as much as
  the digits, and "lakh" and "crore" are multipliers that a plain number regex misses by
  five and seven orders of magnitude.
* **Transliteration variance.** Prisma's `transliterate` output is not spelling-stable:
  "paanch", "panch" and "pach" all occur. The word lists below are alternates for the same
  value rather than a tidy dictionary, which is what makes them look redundant.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from app.core.models import Entity, EntitySource, EntityType

#: Hindi/Hinglish number words, including the transliteration variants Prisma produces.
#: Values up to twenty plus the tens, which is as far as a spoken amount realistically goes
#: before it switches to "hazaar" or "lakh".
_NUMBER_WORDS: dict[str, int] = {
    "ek": 1,
    "do": 2,
    "teen": 3,
    "tin": 3,
    "char": 4,
    "chaar": 4,
    "panch": 5,
    "paanch": 5,
    "pach": 5,
    "chh": 6,
    "chhe": 6,
    "che": 6,
    "chay": 6,
    "saat": 7,
    "sat": 7,
    "aath": 8,
    "ath": 8,
    "nau": 9,
    "no": 9,
    "das": 10,
    "dus": 10,
    "gyarah": 11,
    "gyaarah": 11,
    "barah": 12,
    "baarah": 12,
    "terah": 13,
    "chaudah": 14,
    "pandrah": 15,
    "solah": 16,
    "satrah": 17,
    "atharah": 18,
    "unnis": 19,
    "bees": 20,
    "bis": 20,
    "pachees": 25,
    "pachis": 25,
    "tees": 30,
    "tis": 30,
    "chalees": 40,
    "chalis": 40,
    "pachaas": 50,
    "pachas": 50,
    "pachhas": 50,
    "saath": 60,
    "sattar": 70,
    "assi": 80,
    "nabbe": 90,
    "sau": 100,
}

#: Multipliers. `hazaar` is a thousand, `lakh` a hundred thousand, `crore` ten million.
_MULTIPLIERS: dict[str, int] = {
    "hazaar": 1_000,
    "hazar": 1_000,
    "hajar": 1_000,
    "thousand": 1_000,
    "k": 1_000,
    "lakh": 100_000,
    "lac": 100_000,
    "lakhs": 100_000,
    "crore": 10_000_000,
    "cr": 10_000_000,
    "karod": 10_000_000,
}

_MULTIPLIER_ALTERNATION = "|".join(sorted(_MULTIPLIERS, key=len, reverse=True))
_NUMBER_WORD_ALTERNATION = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))

#: UPI handles: `name@provider`. Deliberately not anchored to a provider allow-list - a
#: scammer's handle is on whatever PSP he could register, and the point is to capture it.
#:
#: The lookahead is what separates a handle from an email address: a UPI provider is a bare
#: token, so anything followed by a dot and more letters is a domain. Without it
#: `officer@gmail.com` is captured as a handle on the provider "gmail".
_UPI = re.compile(r"\b([a-z0-9][a-z0-9._-]{1,48})@([a-z]{2,20})(?!\.[a-z])\b", re.IGNORECASE)

#: IFSC: four letters, a zero, then six alphanumerics. The format is fixed by RBI.
_IFSC = re.compile(r"\b([A-Z]{4}0[A-Z0-9]{6})\b")

#: Indian mobile numbers: ten digits starting 6-9, optionally with +91 and separators.
#: Written as a single alternation rather than one loose pattern because a bare ten-digit
#: run also matches account numbers, case numbers and dates.
_PHONE = re.compile(r"(?<![\d])(?:\+?91[\s-]?)?([6-9]\d{4}[\s-]?\d{5}|[6-9]\d{9})(?![\d])")

#: Digits with an optional currency mark and an optional multiplier word.
_AMOUNT_DIGITS = re.compile(
    rf"(?:₹|rs\.?|inr)?\s?(\d+(?:,\d{{2,3}})*(?:\.\d+)?)\s*({_MULTIPLIER_ALTERNATION})?\b",
    re.IGNORECASE,
)

#: A spoken amount: one or two number words, then a multiplier. The multiplier is required
#: here - "do" on its own is the word "two", not a sum of money.
_AMOUNT_WORDS = re.compile(
    rf"\b((?:{_NUMBER_WORD_ALTERNATION})(?:\s+(?:{_NUMBER_WORD_ALTERNATION}))?)"
    rf"\s+({_MULTIPLIER_ALTERNATION})\b",
    re.IGNORECASE,
)

#: Case and reference numbers, as a scammer recites them.
#:
#: Two requirements, both learned from false positives on a real call. A keyword, because a
#: bare alphanumeric run matches far too much. And **at least one digit**, because without
#: it "warrant nikal jayega" yielded the case number "NIKAL" and "case number fake C Y B..."
#: yielded "FAKE". Every real case or FIR number contains a digit; an all-letters token
#: after the keyword is the next word of the sentence.
_CASE_NUMBER = re.compile(
    r"\b(?:case|fir|complaint|reference|ref|docket|warrant)"
    r"[\s.:#-]*(?:number|no\.?|num|#)?[\s.:#-]*"
    r"((?=[A-Z0-9/-]{4,25}\b)[A-Z0-9][A-Z0-9/-]*\d[A-Z0-9/-]*)\b",
    re.IGNORECASE,
)

#: Organisations a scammer claims to call from. Matched as phrases because the individual
#: words are far too common.
_ORGANISATIONS: tuple[tuple[str, str], ...] = (
    (r"cyber\s*(?:crime|cell)", "cyber crime cell"),
    (r"(?:delhi|mumbai|chennai|kolkata|bengaluru|bangalore|hyderabad)\s+police", "state police"),
    (r"crime\s+branch", "crime branch"),
    (r"income\s*tax(?:\s+department)?", "income tax department"),
    (r"narcotics?(?:\s+(?:bureau|department|control))?", "narcotics bureau"),
    (r"customs?(?:\s+department)?", "customs"),
    (r"\btrai\b", "TRAI"),
    (r"\bcbi\b", "CBI"),
    (r"\bed\b(?=\s+(?:department|office))", "Enforcement Directorate"),
    (r"(?:reserve\s+bank|\brbi\b)", "RBI"),
    (r"(?:state\s+bank|\bsbi\b)", "State Bank of India"),
    (r"(?:punjab\s+national\s+bank|\bpnb\b)", "Punjab National Bank"),
    (r"\bhdfc\b", "HDFC Bank"),
    (r"\bicici\b", "ICICI Bank"),
    (r"\baxis\s+bank\b", "Axis Bank"),
    (r"courier\s+(?:company|service)", "courier company"),
    (r"\b(?:fedex|dhl|bluedart|blue\s*dart)\b", "courier company"),
)

#: Where one organisation label makes another redundant. A scammer who says "cyber crime
#: branch" has made one claim, not two.
_ORGANISATION_SUBSUMES: tuple[tuple[str, str], ...] = (
    ("cyber crime cell", "crime branch"),
    ("cyber crime cell", "state police"),
)

#: An officer's claimed name: a rank, an optional honorific, then a capitalised surname.
#:
#: The rank and honorific are case-insensitive through scoped `(?i:)` groups, while the
#: surname stays case-sensitive. That asymmetry is the whole pattern: requiring the rank is
#: what stops every capitalised word becoming an officer, and requiring the capital is what
#: stops "inspector ne bola" yielding the officer "ne".
_OFFICER = re.compile(
    r"\b(?i:inspector|sub-?inspector|si|asi|officer|constable|head\s+constable|"
    r"commissioner|dcp|acp|sho|agent|manager|executive)\s+"
    r"(?i:mr\.?\s+|mrs\.?\s+|shri\s+|smt\.?\s+)?([A-Z][a-z]{2,20})\b"
)

#: Minimum amount worth recording. A scammer saying "do minute" would otherwise be logged as
#: a sum of two rupees.
MIN_AMOUNT = 100


def _dedupe(entities: Iterable[Entity]) -> tuple[Entity, ...]:
    """Collapse repeats of the same type and value, keeping the earliest turn."""
    seen: dict[tuple[EntityType, str], Entity] = {}
    for entity in entities:
        key = (entity.entity_type, entity.value.casefold())
        if key not in seen:
            seen[key] = entity
    return tuple(seen.values())


def words_to_number(phrase: str) -> int | None:
    """Convert a spoken number phrase to an integer, or None if it is not one.

    Handles the additive forms a speaker actually uses: "do" is 2, "pachaas" is 50, and
    "ek sau" is 100 - the last because "sau" is in the word table as 100 and the leading
    "ek" multiplies it rather than adding to it.
    """
    parts = [part for part in re.split(r"\s+", phrase.strip().casefold()) if part]
    if not parts or any(part not in _NUMBER_WORDS for part in parts):
        return None
    if len(parts) == 1:
        return _NUMBER_WORDS[parts[0]]
    if len(parts) == 2:
        first, second = (_NUMBER_WORDS[part] for part in parts)
        # "ek sau" is one hundred; "bees paanch" is twenty-five.
        return first * second if second >= 100 else first + second
    return None


def extract_amounts(text: str, turn_seq: int) -> list[Entity]:
    """Rupee amounts, written as digits or spoken as words."""
    found: list[Entity] = []

    for match in _AMOUNT_WORDS.finditer(text):
        value = words_to_number(match.group(1))
        if value is None:
            continue
        total = value * _MULTIPLIERS[match.group(2).casefold()]
        if total >= MIN_AMOUNT:
            found.append(_amount(total, turn_seq))

    for match in _AMOUNT_DIGITS.finditer(text):
        digits = match.group(1).replace(",", "")
        try:
            number = float(digits)
        except ValueError:
            continue
        multiplier = _MULTIPLIERS[match.group(2).casefold()] if match.group(2) else 1
        total = int(number * multiplier)
        # A bare number with no currency mark and no multiplier is usually not money: it is
        # a case number, a digit being read out, or a year.
        if multiplier == 1 and not re.search(r"(?:₹|rs\.?|inr)\s?$", text[: match.start(1)], re.I):
            continue
        if total >= MIN_AMOUNT:
            found.append(_amount(total, turn_seq))

    return found


def _amount(value: int, turn_seq: int) -> Entity:
    return Entity(
        entity_type=EntityType.AMOUNT,
        value=f"INR {value}",
        source=EntitySource.REGEX,
        turn_seq=turn_seq,
    )


def extract_from_turn(text: str, turn_seq: int) -> tuple[Entity, ...]:
    """Every identifier this module can find in one utterance."""
    found: list[Entity] = []

    for match in _UPI.finditer(text):
        handle = match.group(0)
        # An email address is not a UPI handle, and a scammer gives out both.
        if "." in match.group(2):
            continue
        found.append(
            Entity(
                entity_type=EntityType.UPI_ID,
                value=handle.casefold(),
                source=EntitySource.REGEX,
                turn_seq=turn_seq,
            )
        )

    for match in _IFSC.finditer(text.upper()):
        found.append(
            Entity(
                entity_type=EntityType.IFSC,
                value=match.group(1),
                source=EntitySource.REGEX,
                turn_seq=turn_seq,
            )
        )

    for match in _PHONE.finditer(text):
        digits = re.sub(r"\D", "", match.group(1))
        found.append(
            Entity(
                entity_type=EntityType.PHONE,
                value=digits,
                source=EntitySource.REGEX,
                turn_seq=turn_seq,
            )
        )

    found.extend(extract_amounts(text, turn_seq))

    for match in _CASE_NUMBER.finditer(text):
        found.append(
            Entity(
                entity_type=EntityType.CASE_NUMBER,
                value=match.group(1).upper(),
                source=EntitySource.REGEX,
                turn_seq=turn_seq,
            )
        )

    organisations = {
        label for pattern, label in _ORGANISATIONS if re.search(pattern, text, re.IGNORECASE)
    }
    # "cyber crime branch" matches both patterns, and two rows for one claim reads as a bug
    # to whoever is looking at the evidence pack. The more specific label wins.
    for specific, general in _ORGANISATION_SUBSUMES:
        if specific in organisations:
            organisations.discard(general)
    for label in sorted(organisations):
        found.append(
            Entity(
                entity_type=EntityType.ORGANISATION,
                value=label,
                source=EntitySource.REGEX,
                turn_seq=turn_seq,
            )
        )

    for match in _OFFICER.finditer(text):
        found.append(
            Entity(
                entity_type=EntityType.OFFICER_NAME,
                value=match.group(1),
                source=EntitySource.REGEX,
                turn_seq=turn_seq,
            )
        )

    return _dedupe(found)


def extract_all(texts: Sequence[str]) -> tuple[Entity, ...]:
    """Run over a whole transcript, deduplicated across turns."""
    found: list[Entity] = []
    for turn_seq, text in enumerate(texts):
        found.extend(extract_from_turn(text, turn_seq))
    return _dedupe(found)
