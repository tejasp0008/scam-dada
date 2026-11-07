"""The read API, the dashboard feed and the evidence pack, over seeded calls."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.models import PersonaId, ScamStage, Speaker
from app.core.state import CallState
from app.infra.models_sql import Call as CallRow
from app.infra.models_sql import CanaryIdentity as CanaryRow
from app.infra.models_sql import CanaryMatch as MatchRow
from app.infra.models_sql import Entity as EntityRow
from app.infra.models_sql import Event as EventRow
from app.infra.models_sql import Turn as TurnRow

START = datetime(2026, 11, 6, 10, 0, tzinfo=UTC)
CALL = "11111111-1111-1111-1111-111111111111"
EARLIER = "22222222-2222-2222-2222-222222222222"


def event_payload(kind: str, seq: int, **extra: object) -> dict[str, object]:
    return {
        "type": kind,
        "call_id": CALL,
        "seq": seq,
        "ts": (START + timedelta(seconds=seq)).isoformat(),
        **extra,
    }


async def seed(sessions: async_sessionmaker[AsyncSession]) -> None:
    """A finished call with everything a report can contain, plus the call that leaked."""
    async with sessions() as session, session.begin():
        session.add(
            CallRow(
                id=EARLIER,
                started_at=START - timedelta(minutes=10),
                ended_at=START - timedelta(minutes=9),
                state=CallState.ENDED.value,
                source="browser",
            )
        )
        session.add(
            CallRow(
                id=CALL,
                started_at=START,
                ended_at=START + timedelta(seconds=90),
                state=CallState.ENDED.value,
                final_state=CallState.ENDED.value,
                ended_reason="caller hung up",
                source="browser",
                persona_sequence=["dada", "grandson"],
            )
        )
        session.add(
            CanaryRow(
                id="canary-earlier",
                issued_call_id=EARLIER,
                identity={
                    "identity_id": "canary-earlier",
                    "full_name": "Kamla Khidki",
                    "upi_handle": "kamla.1234@demo",
                    "locality": "Demo Nagar",
                    "phone": "0011223344",
                    "issued_at_ms": 0,
                },
            )
        )
        session.add(
            CanaryRow(
                id="canary-this",
                issued_call_id=CALL,
                identity={
                    "identity_id": "canary-this",
                    "full_name": "Prem Tijori",
                    "upi_handle": "prem.5678@demo",
                    "locality": "Testpur Colony",
                    "phone": "0055667788",
                    "issued_at_ms": 0,
                },
            )
        )
        session.add_all(
            [
                TurnRow(
                    call_id=CALL,
                    seq=0,
                    speaker=Speaker.CALLER.value,
                    text="Main cyber crime se inspector Sharma bol raha hoon",
                    started_at_ms=0,
                    duration_ms=4000,
                ),
                TurnRow(
                    call_id=CALL,
                    seq=1,
                    speaker=Speaker.PERSONA.value,
                    persona=PersonaId.DADA.value,
                    text="Haan beta, kaun bol raha hai?",
                    started_at_ms=6000,
                    stt_final_ms=480,
                    llm_first_token_ms=300,
                    tts_first_byte_ms=600,
                ),
                TurnRow(
                    call_id=CALL,
                    seq=2,
                    speaker=Speaker.CALLER.value,
                    text="kamla.1234@demo par bhejiye",
                    started_at_ms=40000,
                    duration_ms=3000,
                ),
                TurnRow(
                    call_id=CALL,
                    seq=3,
                    speaker=Speaker.PERSONA.value,
                    persona=PersonaId.GRANDSON.value,
                    text="Ek second, main dekhta hoon",
                    started_at_ms=50000,
                    llm_first_token_ms=260,
                    tts_first_byte_ms=410,
                ),
                # A caller who tries to put markup into the record. The evidence pack is a
                # document people forward, so this has to come out escaped.
                TurnRow(
                    call_id=CALL,
                    seq=4,
                    speaker=Speaker.CALLER.value,
                    text="<script>alert('x')</script>",
                    started_at_ms=60000,
                ),
            ]
        )
        session.add_all(
            [
                EntityRow(
                    call_id=CALL,
                    entity_type="organisation",
                    value="cyber crime cell",
                    source="regex",
                    confidence=1.0,
                ),
                EntityRow(
                    call_id=CALL,
                    entity_type="upi_id",
                    value="kamla.1234@demo",
                    source="regex",
                    confidence=1.0,
                ),
                EntityRow(
                    call_id=CALL,
                    entity_type="officer_name",
                    value="Inspector Sharma",
                    source="llm",
                    confidence=0.85,
                ),
            ]
        )
        session.add(
            MatchRow(
                identity_id="canary-earlier",
                matched_call_id=CALL,
                matched_field="upi_handle",
                matched_text="kamla.1234@demo",
                score=1.0,
            )
        )
        for seq, payload in enumerate(
            [
                event_payload(
                    "state_change",
                    0,
                    from_state="ringing",
                    to_state="human_check",
                    reason="answered",
                ),
                event_payload("stage", 1, stage="authority", confidence=0.7, pressure_score=0.2),
                event_payload("risk", 2, hangup_risk=0.1, signals=[], tactic=None),
                event_payload(
                    "latency",
                    3,
                    turn_seq=1,
                    stt_final_ms=480,
                    llm_first_token_ms=300,
                    tts_first_byte_ms=600,
                    filler_played=True,
                ),
                event_payload("stage", 4, stage="payment", confidence=0.6, pressure_score=0.75),
                event_payload(
                    "risk", 5, hangup_risk=0.55, signals=["impatience"], tactic="fixed_deposit"
                ),
                event_payload(
                    "latency",
                    6,
                    turn_seq=3,
                    stt_final_ms=None,
                    llm_first_token_ms=260,
                    tts_first_byte_ms=410,
                    filler_played=True,
                ),
            ]
        ):
            session.add(
                EventRow(
                    call_id=CALL,
                    seq=seq,
                    type=str(payload["type"]),
                    ts=START + timedelta(seconds=seq),
                    payload=payload,
                )
            )


@pytest.fixture
async def seeded(call_client: TestClient, sqlite_sessions: async_sessionmaker[AsyncSession]):
    await seed(sqlite_sessions)
    return call_client


class TestListCalls:
    def test_calls_are_listed_newest_first(self, seeded: TestClient) -> None:
        calls = seeded.get("/api/calls").json()
        assert [c["call_id"] for c in calls] == [CALL, EARLIER]

    def test_the_summary_carries_the_headline_number(self, seeded: TestClient) -> None:
        call = seeded.get("/api/calls").json()[0]
        assert call["seconds_wasted"] == pytest.approx(90.0)
        assert call["caller_turns"] == 3
        assert call["persona_turns"] == 2

    def test_the_furthest_stage_is_reported_not_the_last(self, seeded: TestClient) -> None:
        """A scammer circling back has not regressed."""
        assert seeded.get("/api/calls").json()[0]["stage_reached"] == ScamStage.PAYMENT.value

    def test_peaks_are_peaks(self, seeded: TestClient) -> None:
        call = seeded.get("/api/calls").json()[0]
        assert call["peak_pressure"] == pytest.approx(0.75)
        assert call["peak_hangup_risk"] == pytest.approx(0.55)

    def test_counts_come_from_the_tables(self, seeded: TestClient) -> None:
        call = seeded.get("/api/calls").json()[0]
        assert call["entity_count"] == 3
        assert call["leak_count"] == 1

    def test_personas_are_listed_in_handoff_order(self, seeded: TestClient) -> None:
        assert seeded.get("/api/calls").json()[0]["personas"] == ["dada", "grandson"]

    def test_the_limit_is_honoured(self, seeded: TestClient) -> None:
        assert len(seeded.get("/api/calls?limit=1").json()) == 1

    def test_an_absurd_limit_is_refused(self, seeded: TestClient) -> None:
        assert seeded.get("/api/calls?limit=9999").status_code == 422


class TestGetCall:
    def test_the_report_has_the_transcript_in_order(self, seeded: TestClient) -> None:
        report = seeded.get(f"/api/calls/{CALL}").json()
        assert [t["seq"] for t in report["turns"]] == [0, 1, 2, 3, 4]

    def test_entities_carry_their_provenance(self, seeded: TestClient) -> None:
        """A reader has to be able to tell which machine found what."""
        report = seeded.get(f"/api/calls/{CALL}").json()
        sources = {e["value"]: e["source"] for e in report["entities"]}
        assert sources["kamla.1234@demo"] == "regex"
        assert sources["Inspector Sharma"] == "llm"

    def test_the_leak_names_the_call_that_issued_the_identity(self, seeded: TestClient) -> None:
        leak = seeded.get(f"/api/calls/{CALL}").json()["leaks"][0]
        assert leak["issued_call_id"] == EARLIER
        assert leak["matched_text"] == "kamla.1234@demo"

    def test_an_unmeasurable_latency_leg_is_reported_as_absent(self, seeded: TestClient) -> None:
        """Not as zero, and not omitted: omitted reads as fast."""
        raw = seeded.get(f"/api/calls/{CALL}").json()["latency"]
        legs = {leg["name"]: leg for leg in raw}
        assert legs["stt_final_ms"]["samples"] == 1
        assert legs["llm_first_token_ms"]["median_ms"] in (260, 300)
        assert set(legs) == {"stt_final_ms", "llm_first_token_ms", "tts_first_byte_ms"}

    def test_the_stage_timeline_records_changes_not_ticks(self, seeded: TestClient) -> None:
        timeline = seeded.get(f"/api/calls/{CALL}").json()["stage_timeline"]
        assert [entry[1] for entry in timeline] == ["authority", "payment"]

    def test_the_canary_issued_to_this_call_is_included(self, seeded: TestClient) -> None:
        issued = seeded.get(f"/api/calls/{CALL}").json()["canary_issued"]
        assert issued["upi_handle"] == "prem.5678@demo"

    def test_retention_tactics_are_listed(self, seeded: TestClient) -> None:
        assert seeded.get(f"/api/calls/{CALL}").json()["tactics_used"] == ["fixed_deposit"]

    def test_an_unknown_call_is_a_404(self, seeded: TestClient) -> None:
        assert seeded.get("/api/calls/does-not-exist").status_code == 404


class TestEvidencePack:
    def test_the_html_pack_renders(self, seeded: TestClient) -> None:
        page = seeded.get(f"/api/calls/{CALL}/evidence.html")
        assert page.status_code == 200
        assert "Evidence pack" in page.text

    def test_it_states_the_headline_number(self, seeded: TestClient) -> None:
        assert "1.5 minutes" in seeded.get(f"/api/calls/{CALL}/evidence.html").text

    def test_it_includes_the_transcript_and_the_extracted_values(self, seeded: TestClient) -> None:
        page = seeded.get(f"/api/calls/{CALL}/evidence.html").text
        assert "inspector Sharma bol raha hoon" in page
        assert "kamla.1234@demo" in page

    def test_it_distinguishes_regex_from_model_findings(self, seeded: TestClient) -> None:
        page = seeded.get(f"/api/calls/{CALL}/evidence.html").text
        assert "regex" in page
        assert "llm" in page

    def test_it_reports_the_leak_prominently(self, seeded: TestClient) -> None:
        page = seeded.get(f"/api/calls/{CALL}/evidence.html").text
        assert "Identity leak" in page
        assert EARLIER[:8] in page

    def test_it_states_that_every_issued_identifier_is_fictional(self, seeded: TestClient) -> None:
        """The pack gets forwarded, so it has to say so on its own face."""
        page = seeded.get(f"/api/calls/{CALL}/evidence.html").text
        assert "fictional" in page
        assert "cannot be dialled" in page

    def test_transcript_text_is_escaped(
        self, seeded: TestClient, sqlite_sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        """Every string here came out of a stranger's mouth, and the pack gets forwarded."""
        import asyncio

        async def add_nasty() -> None:
            async with sqlite_sessions() as session, session.begin():
                session.add(
                    TurnRow(
                        call_id=CALL,
                        seq=9,
                        speaker=Speaker.CALLER.value,
                        text="<script>alert('x')</script>",
                        started_at_ms=1000,
                    )
                )

        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(add_nasty())
        page = seeded.get(f"/api/calls/{CALL}/evidence.html").text
        assert "<script>alert" not in page
        assert "&lt;script&gt;" in page

    def test_an_unknown_call_is_a_404(self, seeded: TestClient) -> None:
        assert seeded.get("/api/calls/nope/evidence.html").status_code == 404

    def test_the_pdf_is_either_produced_or_explained(self, seeded: TestClient) -> None:
        """WeasyPrint is an optional extra with system libraries behind it, so a machine
        without it must get operator information rather than a stack trace."""
        response = seeded.get(f"/api/calls/{CALL}/evidence.pdf")
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            assert response.content[:4] == b"%PDF"
        else:
            assert "pdf" in response.json()["detail"].lower()


class TestDashboardFeed:
    """The reconnect path.

    `bus.history` is the thing worth asserting here and it is reachable over HTTP, so that
    is where the backlog is tested. The socket itself is only checked for accepting and
    closing cleanly: blocking on `receive` would hang whenever the seeded events live in
    the database rather than in Redis, which is always, because seeding writes rows. The
    live feed with real events is covered by `test_call_socket.py`, which drives a whole
    call through the orchestrator.
    """

    def test_the_socket_accepts_and_closes_cleanly(self, seeded: TestClient) -> None:
        with seeded.websocket_connect(f"/ws/dashboard/{CALL}?after_seq=-1"):
            pass

    def test_the_socket_accepts_a_call_that_does_not_exist(self, seeded: TestClient) -> None:
        """A dashboard asking about a call Redis has forgotten must not get a 500."""
        with seeded.websocket_connect("/ws/dashboard/unknown-call"):
            pass

    def test_the_backlog_is_available_over_http(self, seeded: TestClient) -> None:
        body = seeded.get(f"/api/calls/{CALL}/events?after_seq=-1").json()
        assert body["call_id"] == CALL
        assert body["after_seq"] == -1
        assert isinstance(body["events"], list)

    def test_a_sequence_filter_is_honoured(self, seeded: TestClient) -> None:
        body = seeded.get(f"/api/calls/{CALL}/events?after_seq=1000").json()
        assert body["events"] == []


class TestDashboardPage:
    def test_the_page_is_served(self, call_client: TestClient) -> None:
        page = call_client.get("/dashboard")
        assert page.status_code == 200
        assert "control room" in page.text

    def test_it_renders_every_stage_in_order(self, call_client: TestClient) -> None:
        page = call_client.get("/dashboard").text
        assert '"hook", "authority", "fear", "isolation", "payment"' in page

    def test_it_handles_every_event_type(self, call_client: TestClient) -> None:
        page = call_client.get("/dashboard").text
        for kind in (
            "transcript",
            "stage",
            "risk",
            "entity",
            "canary_match",
            "latency",
            "persona_handoff",
        ):
            assert f'"{kind}"' in page, kind

    def test_it_links_to_both_evidence_formats(self, call_client: TestClient) -> None:
        page = call_client.get("/dashboard").text
        assert "evidence.html" in page
        assert "evidence.pdf" in page

    def test_an_unmeasurable_leg_is_labelled_not_blank(self, call_client: TestClient) -> None:
        assert "not measurable" in call_client.get("/dashboard").text


class TestNoFabricatedNumbers:
    """The project's honesty rule, as a test over the API surface."""

    def test_the_summary_does_not_carry_a_barge_in_count(self, seeded: TestClient) -> None:
        """Barge-in is not an event, so a field for it could only ever report zero - which
        reads as "never happened" rather than "not measured"."""
        assert "barge_ins" not in seeded.get("/api/calls").json()[0]

    def test_an_unmeasured_latency_leg_is_null_not_zero(self, seeded: TestClient) -> None:
        legs = {leg["name"]: leg for leg in seeded.get(f"/api/calls/{CALL}").json()["latency"]}
        for leg in legs.values():
            if leg["samples"] == 0:
                assert leg["median_ms"] is None
                assert leg["worst_ms"] is None

    def test_a_call_with_no_stage_events_reports_no_stage(self, seeded: TestClient) -> None:
        """Rather than defaulting to `hook`, which would be a claim about the call."""
        earlier = next(c for c in seeded.get("/api/calls").json() if c["call_id"] == EARLIER)
        assert earlier["stage_reached"] is None
        assert earlier["entity_count"] == 0


class TestDemoPage:
    """The page the demo video films. Mostly a contract check: a panel that silently stops
    handling an event type is an event nobody can see, which is worse than no event."""

    def test_the_page_is_served(self, call_client: TestClient) -> None:
        page = call_client.get("/demo")
        assert page.status_code == 200
        assert "live control room" in page.text

    def test_its_assets_are_served(self, call_client: TestClient) -> None:
        for path in ("/static/demo.css", "/static/demo.js"):
            assert call_client.get(path).status_code == 200, path

    def test_it_handles_every_event_type_the_orchestrator_emits(
        self, call_client: TestClient
    ) -> None:
        js = call_client.get("/static/demo.js").text
        for kind in (
            "transcript",
            "stage",
            "risk",
            "entity",
            "canary_match",
            "latency",
            "persona_handoff",
        ):
            assert f'"{kind}"' in js, kind

    def test_it_warns_about_echo(self, call_client: TestClient) -> None:
        """On speakers the persona's own room tone trips the barge-in detector."""
        page = call_client.get("/demo").text
        assert "headphones" in page.lower()
        assert "checkEcho" in call_client.get("/static/demo.js").text

    def test_it_shows_which_machine_found_an_entity(self, call_client: TestClient) -> None:
        js = call_client.get("/static/demo.js").text
        assert "event.source" in js
        assert "regex" in js

    def test_it_labels_an_unmeasurable_latency_leg_rather_than_showing_zero(
        self, call_client: TestClient
    ) -> None:
        js = call_client.get("/static/demo.js").text
        assert "fmtMs" in js

    def test_it_states_that_issued_identifiers_are_fictional(self, call_client: TestClient) -> None:
        """It is the frame that ends up on LinkedIn, so it has to say so on its own face."""
        page = call_client.get("/demo").text
        assert "fictional" in page
        assert "cannot be dialled" in page

    def test_it_names_the_gnani_models(self, call_client: TestClient) -> None:
        page = call_client.get("/demo").text
        assert "Prisma v3.0" in page
        assert "Timbre v3.0" in page

    def test_motion_can_be_switched_off(self, call_client: TestClient) -> None:
        """For reduced-motion users, and so a headless screenshot is deterministic."""
        css = call_client.get("/static/demo.css").text
        assert "prefers-reduced-motion" in css
        assert "body.still" in css

    def test_it_can_replay_a_recorded_call(self, seeded: TestClient) -> None:
        """Rehearsing the visuals without talking, and proof the panels are driven by the
        record rather than by a parallel code path."""
        js = seeded.get("/static/demo.js").text
        assert "function replay(" in js
        assert "/api/calls/" in js
        # The endpoints replay depends on must exist.
        assert seeded.get(f"/api/calls/{CALL}").status_code == 200
        assert seeded.get(f"/api/calls/{CALL}/events?after_seq=-1").status_code == 200
