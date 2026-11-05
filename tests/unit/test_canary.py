"""Canary identities. The one place the system generates identifiers, so the one place the
synthetic-data-only rule is easiest to break by accident."""

from __future__ import annotations

import re

import pytest

from app.core.canary import (
    CANARY_IFSC_BANK,
    CANARY_PHONE_PREFIX,
    CANARY_UPI_SUFFIX,
    canary_ifsc,
    despoken,
    find_matches,
    is_visibly_fake,
    issue,
)
from app.core.extract import extract_from_turn
from app.core.models import CanaryIdentity, EntityType

CALLS = [f"call-{i:04d}" for i in range(300)]


class TestEveryFieldIsProvablyUnusable:
    """Not "unlikely to be real" - provably not dialable, not a real PSP, not a real bank."""

    @pytest.mark.parametrize("call_id", CALLS[:60])
    def test_a_generated_identity_is_visibly_fake(self, call_id: str) -> None:
        assert is_visibly_fake(issue(call_id))

    @pytest.mark.parametrize("call_id", CALLS[:60])
    def test_the_phone_cannot_be_an_indian_mobile(self, call_id: str) -> None:
        """An Indian mobile must begin 6-9. A random 10-digit number would be somebody's."""
        phone = issue(call_id).phone
        assert phone.startswith(CANARY_PHONE_PREFIX)
        assert not re.match(r"^[6-9]", phone)
        # And the project's own extractor agrees it is not a phone number.
        assert not [e for e in extract_from_turn(phone, 0) if e.entity_type is EntityType.PHONE]

    @pytest.mark.parametrize("call_id", CALLS[:60])
    def test_the_upi_handle_is_on_a_non_existent_provider(self, call_id: str) -> None:
        assert issue(call_id).upi_handle.endswith(f"@{CANARY_UPI_SUFFIX}")

    @pytest.mark.parametrize("call_id", CALLS[:40])
    def test_the_ifsc_has_a_valid_shape_but_an_unallocated_bank_code(self, call_id: str) -> None:
        """Valid shape is required, or the extractor would not catch it coming back."""
        code = canary_ifsc(call_id)
        assert code.startswith(f"{CANARY_IFSC_BANK}0")
        found = extract_from_turn(f"IFSC {code}", 0)
        assert [e.value for e in found if e.entity_type is EntityType.IFSC] == [code]

    @pytest.mark.parametrize("call_id", CALLS[:60])
    def test_the_locality_does_not_exist(self, call_id: str) -> None:
        locality = issue(call_id).locality
        assert any(
            word in locality
            for word in (
                "Demo",
                "Test",
                "Sample",
                "Fixture",
                "Placeholder",
                "Mock",
                "Dummy",
                "Stub",
            )
        )

    def test_no_generated_identity_contains_an_aadhaar_or_pan_shape(self) -> None:
        for call_id in CALLS[:80]:
            identity = issue(call_id)
            text = " ".join(str(v) for v in identity.model_dump().values())
            assert not re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", text)
            assert not re.search(r"\b[A-Z]{5}\d{4}[A-Z]\b", text)


class TestDeterminism:
    def test_the_same_call_gets_the_same_identity(self) -> None:
        """A replayed call must issue what the live call issued, or the evidence packs
        contradict each other and the leak detector never matches across them."""
        assert issue("call-1") == issue("call-1")

    def test_different_calls_get_different_identities(self) -> None:
        handles = {issue(call_id).upi_handle for call_id in CALLS}
        # Not necessarily all 300 distinct, but the collision rate must be low enough that
        # a leak match means something.
        assert len(handles) > len(CALLS) * 0.9

    def test_identity_ids_are_effectively_unique(self) -> None:
        assert len({issue(call_id).identity_id for call_id in CALLS}) > len(CALLS) * 0.95

    def test_an_empty_call_id_is_refused(self) -> None:
        with pytest.raises(ValueError, match="call id"):
            issue("")


class TestLeakDetection:
    def test_a_upi_handle_from_an_earlier_call_is_caught(self) -> None:
        """The whole point: two calls reading from one lead list."""
        identity = issue("call-A")
        text = f"Aapka account {identity.upi_handle} verify karna hai"
        matches = find_matches(text, [("call-A", identity)])
        assert len(matches) == 1
        assert matches[0].matched_field == "upi_handle"
        assert matches[0].issued_call_id == "call-A"
        assert matches[0].score == 1.0

    def test_a_phone_number_is_caught_through_separators(self) -> None:
        identity = issue("call-A")
        spaced = f"{identity.phone[:5]} {identity.phone[5:]}"
        matches = find_matches(f"number {spaced} hai", [("call-A", identity)])
        assert [m.matched_field for m in matches] == ["phone"]

    def test_a_full_name_needs_both_parts(self) -> None:
        """ "Kamla" alone is a common name; matching it loosely would fire on half of all calls."""
        identity = issue("call-A")
        given, surname = identity.full_name.split(" ", 1)
        assert find_matches(f"{given} ji aap hain?", [("call-A", identity)]) == ()
        matches = find_matches(f"{given} {surname} ji?", [("call-A", identity)])
        assert [m.matched_field for m in matches] == ["full_name"]

    def test_a_name_match_scores_below_an_identifier_match(self) -> None:
        identity = issue("call-A")
        name = find_matches(identity.full_name, [("call-A", identity)])[0]
        handle = find_matches(identity.upi_handle, [("call-A", identity)])[0]
        assert name.score < handle.score

    def test_an_unrelated_transcript_matches_nothing(self) -> None:
        """A false positive claims two criminals share infrastructure. It must be rare."""
        identity = issue("call-A")
        text = (
            "Hello sir, main Delhi cyber crime branch se inspector Sharma bol raha hoon. "
            "Aapke naam par parcel pakda gaya hai. Do lakh fake.officer@demo par bhejiye."
        )
        assert find_matches(text, [("call-A", identity)]) == ()

    def test_identities_from_many_calls_are_all_checked(self) -> None:
        issued = [(call_id, issue(call_id)) for call_id in CALLS[:20]]
        target = issued[7]
        matches = find_matches(f"handle {target[1].upi_handle}", issued)
        assert len(matches) == 1
        assert matches[0].issued_call_id == target[0]

    def test_matching_is_case_insensitive(self) -> None:
        identity = issue("call-A")
        assert find_matches(identity.upi_handle.upper(), [("call-A", identity)])

    def test_an_empty_history_matches_nothing(self) -> None:
        assert find_matches("anything at all", []) == ()

    def test_an_empty_transcript_matches_nothing(self) -> None:
        assert find_matches("", [("call-A", issue("call-A"))]) == ()


class TestReadAloudMatching:
    """From a real call that leaked a canary by construction and detected nothing.

    A scammer reading a handle off a lead list says it out loud, and transcription gives it
    back as words: `susila.9030@demo` arrives as "susila dot nine zero three zero at d e m o".
    Without this the detector would essentially never fire, and feature 4 would be decorative.
    """

    #: The exact transcript Prisma returned on the call that failed to detect its own leak.
    REAL_TRANSCRIPT = (
        "humare record mein aapka handle susila dot nine zero three zero at d e m o hai "
        "wahi confirm kijiye"
    )

    def test_the_transcript_that_was_missed_now_matches(self) -> None:
        identity = CanaryIdentity(
            identity_id="canary-00000000",
            full_name="Sushila Khidki",
            upi_handle="susila.9030@demo",
            locality="Demo Nagar",
            phone="0012345678",
            issued_at_ms=0,
        )
        matches = find_matches(self.REAL_TRANSCRIPT, [("call-earlier", identity)])
        assert [m.matched_field for m in matches] == ["upi_handle"]
        assert matches[0].score == 1.0

    @pytest.mark.parametrize(
        ("spoken", "expected"),
        [
            ("at", "@"),
            ("dot", "."),
            ("nine zero three zero", "9030"),
            ("one two three", "123"),
            ("double zero one", "001"),
            ("triple zero one", "0001"),
            ("oh nine", "09"),
            ("ek do teen", "123"),
            ("underscore", "_"),
        ],
    )
    def test_spoken_forms_are_rewritten(self, spoken: str, expected: str) -> None:
        assert despoken(spoken) == expected

    def test_a_phone_number_read_out_digit_by_digit_matches(self) -> None:
        identity = CanaryIdentity(
            identity_id="canary-00000000",
            full_name="Prem Tijori",
            upi_handle="prem.1111@demo",
            locality="Demo Nagar",
            phone="0012345678",
            issued_at_ms=0,
        )
        spoken = "number zero zero one two three four five six seven eight hai"
        matches = find_matches(spoken, [("call-earlier", identity)])
        assert "phone" in [m.matched_field for m in matches]

    def test_the_written_form_still_matches(self) -> None:
        identity = issue("call-A")
        matches = find_matches(f"handle {identity.upi_handle} hai", [("call-A", identity)])
        assert [m.matched_field for m in matches] == ["upi_handle"]

    def test_an_ordinary_sentence_does_not_match_after_rewriting(self) -> None:
        """The rewriting is aggressive, so this is the test that it has not become a
        false-positive machine: "at" and "do" appear constantly in Hinglish."""
        identity = issue("call-A")
        text = (
            "Haan beta, do minute ruko. Main ghar at home hoon, teen baje aana. "
            "Ek do teen char, sab theek hai."
        )
        assert find_matches(text, [("call-A", identity)]) == ()

    def test_despoken_is_only_used_for_matching_not_reported(self) -> None:
        """A de-spoken string is lossy; the match reports the identity's real value."""
        identity = CanaryIdentity(
            identity_id="canary-00000000",
            full_name="Sushila Khidki",
            upi_handle="susila.9030@demo",
            locality="Demo Nagar",
            phone="0012345678",
            issued_at_ms=0,
        )
        match = find_matches(self.REAL_TRANSCRIPT, [("call-earlier", identity)])[0]
        assert match.matched_text == "susila.9030@demo"
