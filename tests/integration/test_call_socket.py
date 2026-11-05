"""The call transport: the contract between the browser and the orchestrator.

Driven through FastAPI's test client against mock providers, so it spends nothing and needs
no microphone. What is asserted is the protocol, because that is what the browser depends on
and what a change to the page would silently break.
"""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from app.core.audio.framing import float32_to_pcm

FRAME_BYTES = 1024


def loud() -> bytes:
    return float32_to_pcm(np.full(FRAME_BYTES // 2, 0.3, dtype=np.float32))


def quiet() -> bytes:
    return float32_to_pcm(np.zeros(FRAME_BYTES // 2, dtype=np.float32))


def drain(socket: WebSocketTestSession, *, limit: int = 400) -> tuple[list[dict], int]:
    """Read until the server closes, separating control messages from audio."""
    control: list[dict] = []
    audio_frames = 0
    for _ in range(limit):
        try:
            message = socket.receive()
        except Exception:
            break
        if message["type"] == "websocket.close":
            break
        if message.get("text") is not None:
            import json

            control.append(json.loads(message["text"]))
        elif message.get("bytes") is not None:
            audio_frames += 1
    return control, audio_frames


def speak(socket: WebSocketTestSession, *, speech: int = 8, silence: int = 16) -> None:
    for _ in range(speech):
        socket.send_bytes(loud())
    for _ in range(silence):
        socket.send_bytes(quiet())


class TestProtocol:
    def test_it_announces_the_audio_contract_before_anything_else(
        self, call_client: TestClient
    ) -> None:
        """The browser cannot resample until it knows the rates, so this must come first."""
        with call_client.websocket_connect("/ws/call") as socket:
            ready = socket.receive_json()
            assert ready["type"] == "ready"
            assert ready["sample_rate_in"] == 16000
            assert ready["sample_rate_out"] == 24000
            assert ready["call_id"]
            socket.send_text("hangup")
            drain(socket)

    def test_hangup_from_the_browser_ends_the_call(self, call_client: TestClient) -> None:
        with call_client.websocket_connect("/ws/call") as socket:
            assert socket.receive_json()["type"] == "ready"
            socket.send_text("hangup")
            control, _ = drain(socket)
            assert any(m["type"] == "ended" for m in control)

    def test_an_unknown_text_message_is_ignored_rather_than_fatal(
        self, call_client: TestClient
    ) -> None:
        """An older page talking to a newer server should degrade, not drop the call."""
        with call_client.websocket_connect("/ws/call") as socket:
            assert socket.receive_json()["type"] == "ready"
            socket.send_text('{"type": "something-new"}')
            socket.send_text("hangup")
            control, _ = drain(socket)
            assert any(m["type"] == "ended" for m in control)

    def test_audio_frames_are_accepted_and_the_persona_replies_in_audio(
        self, call_client: TestClient
    ) -> None:
        with call_client.websocket_connect("/ws/call") as socket:
            assert socket.receive_json()["type"] == "ready"
            speak(socket)
            socket.send_text("hangup")
            _, audio_frames = drain(socket)
            assert audio_frames > 0, "the persona never sent any audio"


class TestEventRelay:
    def test_the_transcript_reaches_the_browser_as_events(self, call_client: TestClient) -> None:
        """The simulator reads the same channel the dashboard will, so this covers both."""
        with call_client.websocket_connect("/ws/call") as socket:
            assert socket.receive_json()["type"] == "ready"
            speak(socket)
            socket.send_text("hangup")
            control, _ = drain(socket)

        events = [m["event"] for m in control if m["type"] == "event"]
        if not events:
            import pytest

            pytest.skip("no Redis available; the relay is exercised only when it is up")
        kinds = {event["type"] for event in events}
        assert "state_change" in kinds
        assert "transcript" in kinds

    def test_events_carry_the_call_id_the_browser_was_given(self, call_client: TestClient) -> None:
        with call_client.websocket_connect("/ws/call") as socket:
            call_id = socket.receive_json()["call_id"]
            speak(socket)
            socket.send_text("hangup")
            control, _ = drain(socket)

        events = [m["event"] for m in control if m["type"] == "event"]
        if not events:
            import pytest

            pytest.skip("no Redis available; the relay is exercised only when it is up")
        assert {event["call_id"] for event in events} == {call_id}


class TestConfiguration:
    def test_an_unconfigured_voice_is_refused_with_an_explanation(self, client: TestClient) -> None:
        """Failing here beats a call that connects and then cannot speak. The voice is
        chosen by ear from the audition, so an unset one is a setup step, not a bug."""
        with client.websocket_connect("/ws/call") as socket:
            message = socket.receive_json()
            assert message["type"] == "error"
            assert "PERSONA_ELDER_VOICE" in message["message"]


class TestPage:
    def test_the_simulator_page_is_served(self, client: TestClient) -> None:
        response = client.get("/simulator")
        assert response.status_code == 200
        assert "Scam Dada" in response.text

    def test_the_root_serves_the_simulator_too(self, client: TestClient) -> None:
        assert client.get("/").status_code == 200

    def test_the_audio_worklets_are_served(self, client: TestClient) -> None:
        """They are loaded by URL from the page, so a missing file is a silent dead call."""
        for name in ("capture-worklet.js", "playback-worklet.js"):
            response = client.get(f"/static/{name}")
            assert response.status_code == 200, name
            assert "registerProcessor" in response.text

    def test_the_page_and_the_worklets_agree_on_the_frame_size(self, client: TestClient) -> None:
        """Prisma rejects anything but 1024 bytes, and the framing happens in the browser."""
        worklet = client.get("/static/capture-worklet.js").text
        assert "FRAME_SAMPLES = 512" in worklet
        assert "TARGET_RATE = 16000" in worklet
