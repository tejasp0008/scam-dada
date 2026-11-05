"""Entity extraction. Regex only, so every case here is a fixed input and a fixed output."""

from __future__ import annotations

import pytest

from app.core.extract import (
    MIN_AMOUNT,
    extract_all,
    extract_from_turn,
    words_to_number,
)
from app.core.models import EntitySource, EntityType


def values(text: str, kind: EntityType) -> list[str]:
    return [e.value for e in extract_from_turn(text, 0) if e.entity_type is kind]


class TestUpi:
    def test_a_handle_is_captured(self) -> None:
        assert values("fake.officer@demo par bhejiye", EntityType.UPI_ID) == ["fake.officer@demo"]

    @pytest.mark.parametrize(
        "text",
        [
            "paisa 9876543210@ybl par",
            "amount scammer-01@paytm bhejo",
            "transfer to fake_kamla@upi",
        ],
    )
    def test_real_world_handle_shapes(self, text: str) -> None:
        assert len(values(text, EntityType.UPI_ID)) == 1

    def test_an_email_address_is_not_a_upi_handle(self) -> None:
        """A scammer gives out both, and conflating them pollutes the evidence pack."""
        assert values("mail kijiye officer@gmail.com par", EntityType.UPI_ID) == []

    def test_handles_are_lowercased_so_repeats_collapse(self) -> None:
        entities = extract_all(["Fake.Officer@Demo", "fake.officer@demo"])
        assert len([e for e in entities if e.entity_type is EntityType.UPI_ID]) == 1


class TestPhone:
    @pytest.mark.parametrize(
        "text",
        ["call 9876543210", "number +91 9876543210", "+919876543210 par", "98765 43210 likhiye"],
    )
    def test_indian_mobile_shapes(self, text: str) -> None:
        assert values(text, EntityType.PHONE) == ["9876543210"]

    def test_a_number_not_starting_six_to_nine_is_not_a_mobile(self) -> None:
        assert values("account 1234567890 hai", EntityType.PHONE) == []

    def test_a_longer_digit_run_is_not_a_mobile(self) -> None:
        """Account numbers are longer, and matching them as phones would be wrong."""
        assert values("account 987654321012 hai", EntityType.PHONE) == []


class TestIfsc:
    def test_an_ifsc_is_captured(self) -> None:
        assert values("IFSC SBIN0001234 hai", EntityType.IFSC) == ["SBIN0001234"]

    def test_a_fictional_bank_code_still_matches_the_format(self) -> None:
        """The canary uses FAKE0..., and it has to be extractable for the leak test."""
        assert values("ifsc fake0000001", EntityType.IFSC) == ["FAKE0000001"]

    def test_a_string_without_the_mandatory_zero_is_not_an_ifsc(self) -> None:
        assert values("code SBIN1001234", EntityType.IFSC) == []


class TestAmounts:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("do lakh bhejiye", "INR 200000"),
            ("paanch hazaar rupaye", "INR 5000"),
            ("panch hazar", "INR 5000"),
            ("pachaas hazaar ka case", "INR 50000"),
            ("ek crore", "INR 10000000"),
            ("teen lakh", "INR 300000"),
            ("bees hazaar", "INR 20000"),
        ],
    )
    def test_spoken_hinglish_amounts(self, text: str, expected: str) -> None:
        """A scammer says these far more often than he reads out digits."""
        assert expected in values(text, EntityType.AMOUNT)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("₹50,000 bhejiye", "INR 50000"),
            ("Rs. 2500 ka charge", "INR 2500"),
            ("INR 1,20,000", "INR 120000"),
            ("2.5 lakh", "INR 250000"),
            ("50 hazaar", "INR 50000"),
        ],
    )
    def test_written_amounts(self, text: str, expected: str) -> None:
        assert expected in values(text, EntityType.AMOUNT)

    def test_transliteration_variants_agree(self) -> None:
        """Prisma's transliterate output is not spelling-stable."""
        for spelling in ("paanch hazaar", "panch hazar", "pach hajar"):
            assert "INR 5000" in values(spelling, EntityType.AMOUNT)

    def test_a_bare_number_is_not_money(self) -> None:
        """ "do minute" must not become two rupees, and a year is not an amount."""
        assert values("do minute ruko", EntityType.AMOUNT) == []
        assert values("1987 se kaam kar raha hoon", EntityType.AMOUNT) == []

    def test_a_number_word_without_a_multiplier_is_not_money(self) -> None:
        assert values("paanch baje aana", EntityType.AMOUNT) == []

    def test_trivial_amounts_are_ignored(self) -> None:
        assert values("Rs. 5 ka", EntityType.AMOUNT) == []
        assert MIN_AMOUNT == 100

    def test_two_amounts_in_one_sentence_are_both_found(self) -> None:
        found = values("pehle do hazaar, phir ₹50,000 bhejiye", EntityType.AMOUNT)
        assert "INR 2000" in found
        assert "INR 50000" in found


class TestWordsToNumber:
    @pytest.mark.parametrize(
        ("phrase", "expected"),
        [("ek", 1), ("do", 2), ("paanch", 5), ("das", 10), ("pachaas", 50), ("ek sau", 100)],
    )
    def test_known_phrases(self, phrase: str, expected: int) -> None:
        assert words_to_number(phrase) == expected

    def test_an_additive_pair(self) -> None:
        assert words_to_number("bees paanch") == 25

    @pytest.mark.parametrize("phrase", ["", "banana", "do banana", "ek do teen chaar"])
    def test_anything_else_is_not_a_number(self, phrase: str) -> None:
        assert words_to_number(phrase) is None


class TestCaseNumbers:
    @pytest.mark.parametrize(
        "text",
        [
            "case number FAKE-CYB-0001 register ho gaya",
            "FIR no. FAKE/2026/0001 hai",
            "reference number ABCD1234",
        ],
    )
    def test_a_case_number_needs_its_keyword(self, text: str) -> None:
        assert values(text, EntityType.CASE_NUMBER)

    def test_a_bare_alphanumeric_run_is_not_a_case_number(self) -> None:
        assert values("mera naam ABCD1234 nahi hai", EntityType.CASE_NUMBER) == []


class TestOrganisations:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("main cyber crime branch se bol raha hoon", "cyber crime cell"),
            ("Delhi police se", "state police"),
            ("income tax department se", "income tax department"),
            ("TRAI se bol raha hoon", "TRAI"),
            ("SBI ka account", "State Bank of India"),
            ("FedEx courier se", "courier company"),
        ],
    )
    def test_claimed_organisations(self, text: str, expected: str) -> None:
        assert expected in values(text, EntityType.ORGANISATION)

    def test_a_label_is_normalised_so_repeats_collapse(self) -> None:
        """ "cyber cell" and "cyber crime" are the same claim and must not count twice."""
        entities = extract_all(["cyber cell se", "cyber crime se"])
        orgs = [e for e in entities if e.entity_type is EntityType.ORGANISATION]
        assert len(orgs) == 1


class TestOfficerNames:
    def test_a_rank_and_a_name(self) -> None:
        assert values("main inspector Sharma bol raha hoon", EntityType.OFFICER_NAME) == ["Sharma"]

    def test_an_honorific_inside_the_phrase(self) -> None:
        assert values("officer Mr. Verma speaking", EntityType.OFFICER_NAME) == ["Verma"]

    def test_a_capitalised_word_without_a_rank_is_not_a_name(self) -> None:
        """Otherwise every sentence's first word becomes an officer."""
        assert values("Sharma ji ne bola tha", EntityType.OFFICER_NAME) == []


class TestProvenance:
    def test_everything_regex_finds_is_marked_regex_and_certain(self) -> None:
        """A reader of the evidence pack has to be able to tell which machine said what."""
        entities = extract_from_turn("inspector Sharma, fake.officer@demo, do lakh", 3)
        assert entities
        for entity in entities:
            assert entity.source is EntitySource.REGEX
            assert entity.confidence == 1.0
            assert entity.turn_seq == 3


class TestWholeTranscript:
    def test_a_realistic_call_yields_the_expected_set(self) -> None:
        transcript = [
            "Hello sir, main Delhi cyber crime branch se inspector Sharma bol raha hoon.",
            "Aapke naam par parcel pakda gaya hai, usme illegal items hain.",
            "Case number FAKE-CYB-0001 register ho gaya hai.",
            "Verification ke liye do lakh fake.officer@demo par bhej dijiye.",
        ]
        kinds = {e.entity_type for e in extract_all(transcript)}
        assert kinds == {
            EntityType.ORGANISATION,
            EntityType.OFFICER_NAME,
            EntityType.CASE_NUMBER,
            EntityType.AMOUNT,
            EntityType.UPI_ID,
        }

    def test_the_earliest_turn_is_kept_for_a_repeated_value(self) -> None:
        entities = extract_all(["do lakh bhejiye", "haan do lakh"])
        amounts = [e for e in entities if e.entity_type is EntityType.AMOUNT]
        assert len(amounts) == 1
        assert amounts[0].turn_seq == 0

    def test_an_empty_transcript_yields_nothing(self) -> None:
        assert extract_all([]) == ()
        assert extract_from_turn("", 0) == ()


class TestFalsePositivesFromRealCalls:
    """Every case here was a wrong extraction on an actual live call.

    These matter more than the happy path: the evidence pack is the deliverable, and a
    fabricated case number in it is worse than a missing one.
    """

    def test_a_keyword_followed_by_an_ordinary_word_is_not_a_case_number(self) -> None:
        """ "warrant nikal jayega" yielded the case number "NIKAL"."""
        assert values("warrant nikal jayega aapke naam par", EntityType.CASE_NUMBER) == []

    def test_a_spelled_out_case_number_is_left_to_the_model(self) -> None:
        """ "case number fake C Y B triple zero one" yielded "FAKE". Regex cannot reassemble
        a number spoken letter by letter, so reporting nothing is the correct answer."""
        found = values(
            "Case number fake C Y B triple zero one register ho gaya", EntityType.CASE_NUMBER
        )
        assert "FAKE" not in found

    @pytest.mark.parametrize(
        "text",
        [
            "case number FAKE-CYB-0001 register ho gaya",
            "FIR no. FAKE/2026/0001",
            "reference number ABCD1234",
        ],
    )
    def test_real_case_numbers_still_match(self, text: str) -> None:
        assert values(text, EntityType.CASE_NUMBER)

    def test_a_case_number_must_contain_a_digit(self) -> None:
        assert values("case number ABCDEFGH", EntityType.CASE_NUMBER) == []

    def test_one_organisation_claim_is_one_row(self) -> None:
        """ "Delhi cyber crime branch" produced three rows: cell, branch and police."""
        orgs = values("main Delhi cyber crime branch se bol raha hoon", EntityType.ORGANISATION)
        assert orgs == ["cyber crime cell"]

    def test_a_plain_police_claim_is_still_captured(self) -> None:
        assert values("Delhi police se bol raha hoon", EntityType.ORGANISATION) == ["state police"]
