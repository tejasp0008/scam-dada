"""One fictional victim identity per call, and the detector that catches it resurfacing.

The idea is simple and the payoff is the most interesting thing in the project: give every
call a unique, fictional set of victim details. If details issued to call A turn up in the
mouth of the caller on call B, the two calls are reading from the same lead list. That is a
claim no single call can make, and it is visible in the dashboard the moment it happens.

**Every field has to be provably unusable, not merely unlikely.** The project's rule is
synthetic data only, and a canary is the one place the system *generates* identifiers, so it
is the easiest place to break that rule by accident. So:

* **Phone numbers start with `00`.** An Indian mobile must begin 6-9, so a number beginning
  `00` cannot be dialled and cannot belong to anyone. A randomly generated 10-digit number
  starting with 9 would be somebody's.
* **UPI handles are on `@demo`.** Not a real PSP handle suffix.
* **IFSC codes use the bank code `FAKE`.** The format is valid - it has to be, or the
  extractor would not catch it coming back - but `FAKE` is not an allocated bank.
* **Names are drawn from fixed lists** and combined with the identity id, so no generated
  name is a plausible full name belonging to a findable person.
* **Localities are invented**, and each contains a word no Indian locality uses.

**Matching is token-level and deliberately conservative.** A false positive here is a claim
that two criminals share infrastructure, which is the strongest thing the dashboard says.
The identifier fields are matched exactly after normalisation; names are matched on surname
plus given name rather than fuzzily, because "Kamla" alone is a common name and matching it
loosely would fire on half of all calls.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.core.models import CanaryIdentity

#: Given names for the fictional victim. Common enough to be plausible to a scammer reading
#: a lead list, and always paired with an invented surname.
_GIVEN_NAMES: tuple[str, ...] = (
    "Kamla",
    "Shanti",
    "Prem",
    "Hariprasad",
    "Vimla",
    "Ramesh",
    "Sushila",
    "Mohanlal",
    "Saroj",
    "Devilal",
)

#: Surnames that are not surnames. Each is a recognisable Hindi word that nobody is called,
#: so a full name from this generator cannot collide with a real person's.
_SURNAMES: tuple[str, ...] = (
    "Khidki",
    "Chabutra",
    "Peetal",
    "Lohakhand",
    "Tijori",
    "Bartan",
    "Chimni",
    "Angeethi",
    "Palanghar",
    "Chaukhat",
)

#: Localities that do not exist. Each contains a word no Indian locality uses.
_LOCALITIES: tuple[str, ...] = (
    "Demo Nagar",
    "Testpur Colony",
    "Sample Vihar",
    "Fixture Bagh",
    "Placeholder Ganj",
    "Mocknagar East",
    "Dummy Enclave",
    "Stub Chowk",
)

#: The UPI suffix. Not a real payment service provider.
CANARY_UPI_SUFFIX = "demo"

#: Non-dialable prefix. An Indian mobile must start 6-9, so this cannot be anyone's number.
CANARY_PHONE_PREFIX = "00"

#: Bank code for canary IFSC values. Valid shape, unallocated code.
CANARY_IFSC_BANK = "FAKE"


class CanaryMatch(BaseModel):
    """An identity issued to an earlier call, found in this one's transcript."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    identity_id: str
    issued_call_id: str
    matched_field: str
    matched_text: str
    score: float = Field(ge=0.0, le=1.0)


def _digits(seed: str, count: int) -> str:
    """Deterministic digits from a seed, so a replayed call issues the same identity."""
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return "".join(str(int(char, 16) % 10) for char in digest[:count])


def _pick(seed: str, options: Sequence[str], salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{seed}".encode()).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def issue(call_id: str, *, issued_at_ms: int = 0) -> CanaryIdentity:
    """Mint the identity for one call. Deterministic in `call_id`.

    Deterministic rather than random so a replayed call issues the identity the live call
    issued - otherwise the evidence pack for a replay would contradict the one for the
    original, and the leak detector would never match across the two.
    """
    if not call_id:
        raise ValueError("a call id is required")
    given = _pick(call_id, _GIVEN_NAMES, "given")
    surname = _pick(call_id, _SURNAMES, "surname")
    locality = _pick(call_id, _LOCALITIES, "locality")
    # Four digits of the call id in the handle, so two calls that happen to draw the same
    # name still get distinguishable handles.
    tag = _digits(call_id, 4)
    return CanaryIdentity(
        identity_id=f"canary-{_digits(call_id, 8)}",
        full_name=f"{given} {surname}",
        upi_handle=f"{given.lower()}.{tag}@{CANARY_UPI_SUFFIX}",
        locality=locality,
        phone=f"{CANARY_PHONE_PREFIX}{_digits(call_id, 8)}",
        issued_at_ms=issued_at_ms,
    )


def canary_ifsc(call_id: str) -> str:
    """A valid-shaped, unallocated IFSC for this call."""
    return f"{CANARY_IFSC_BANK}0{_digits(call_id, 6)}"


def is_visibly_fake(identity: CanaryIdentity) -> bool:
    """Whether every field of an identity is provably unusable.

    Asserted by a test over many generated identities. This is the project's
    synthetic-data-only rule expressed as code rather than as a promise in a README.
    """
    return (
        identity.phone.startswith(CANARY_PHONE_PREFIX)
        and not re.match(r"^[6-9]", identity.phone)
        and identity.upi_handle.endswith(f"@{CANARY_UPI_SUFFIX}")
        and any(word in identity.full_name for word in _SURNAMES)
        and identity.locality in _LOCALITIES
    )


def _normalise(text: str) -> str:
    return re.sub(r"[\s.-]+", "", text).casefold()


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"[A-Za-z]{3,}", text)}


def find_matches(
    text: str, issued: Sequence[tuple[str, CanaryIdentity]]
) -> tuple[CanaryMatch, ...]:
    """Canary details from earlier calls appearing in this transcript.

    `issued` pairs each earlier call id with the identity it was given. Exact-after-
    normalisation for the identifier fields; for the name, both the given name and the
    invented surname must appear, because the surnames are unique but the given names are
    not and matching on one alone would fire constantly.
    """
    haystack = _normalise(text)
    words = _tokens(text)
    matches: list[CanaryMatch] = []

    for call_id, identity in issued:
        for field, value, score in (
            ("upi_handle", identity.upi_handle, 1.0),
            ("phone", identity.phone, 1.0),
            ("identity_id", identity.identity_id, 1.0),
        ):
            if _normalise(value) in haystack:
                matches.append(
                    CanaryMatch(
                        identity_id=identity.identity_id,
                        issued_call_id=call_id,
                        matched_field=field,
                        matched_text=value,
                        score=score,
                    )
                )

        given, surname = identity.full_name.split(" ", 1)
        if given.casefold() in words and surname.casefold() in words:
            matches.append(
                CanaryMatch(
                    identity_id=identity.identity_id,
                    issued_call_id=call_id,
                    matched_field="full_name",
                    matched_text=identity.full_name,
                    # Below one: a name is strong evidence but not an identifier, and the
                    # dashboard shows the score beside the claim.
                    score=0.9,
                )
            )

        if _normalise(identity.locality) in haystack:
            matches.append(
                CanaryMatch(
                    identity_id=identity.identity_id,
                    issued_call_id=call_id,
                    matched_field="locality",
                    matched_text=identity.locality,
                    score=0.7,
                )
            )

    return tuple(matches)
