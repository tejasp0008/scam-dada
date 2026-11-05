"""The call transport: PCM in, persona audio out, over one WebSocket.

Audio travels as binary frames and control travels as JSON text frames on the same socket.
Two sockets would need pairing logic and could not guarantee ordering between "stop talking"
and the audio it refers to.

**Why there is an explicit flush message.** Timbre has no cancel and the browser buffers
whatever it has been sent, so a barge-in that only stops the server from sending leaves the
browser playing for as long as its queue is deep. The orchestrator's flush hook sends
`{"type": "flush"}`, the playback worklet empties its ring buffer, and the persona goes quiet
within a frame instead of a second later.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import update

from app.api.deps import AppResources
from app.core.events import DomainEvent
from app.core.models import PersonaId
from app.core.state import CallState
from app.infra.models_sql import Call as CallRow
from app.providers.base import BudgetExceeded, ProviderError, SttConfig
from app.services.canary_registry import CanaryRegistry
from app.services.orchestrator import CallConfig, CallOrchestrator
from app.services.recorder import CallRecorder

logger = logging.getLogger(__name__)

router = APIRouter(tags=["call"])

#: Inbound audio contract. The capture worklet resamples to this and frames to 1024 bytes,
#: which is Prisma's required frame size (DECISIONS 0001).
INBOUND_SAMPLE_RATE = 16000

#: Outbound audio contract. Timbre's native rate; the browser resamples on playback.
OUTBOUND_SAMPLE_RATE = 24000


@router.websocket("/ws/call")
async def call_socket(socket: WebSocket) -> None:
    """Run one baited call for the lifetime of this socket."""
    resources: AppResources = socket.app.state.resources
    settings = resources.settings
    await socket.accept()

    call_id = str(uuid.uuid4())
    voices = {
        PersonaId.DADA: settings.persona_elder_voice,
        PersonaId.GRANDSON: settings.persona_grandson_voice,
    }
    missing = [persona.value for persona, voice in voices.items() if not voice]
    if missing:
        # Failing loudly here beats a call that connects and then cannot speak. The voice is
        # chosen by ear from the audition, so an unset one is a setup step, not a bug.
        await socket.send_json(
            {
                "type": "error",
                "message": f"no voice configured for: {', '.join(missing)}. "
                "Set PERSONA_ELDER_VOICE and PERSONA_GRANDSON_VOICE in .env",
            }
        )
        await socket.close(code=1011)
        return

    await _create_call_row(resources, call_id)
    recorder = CallRecorder(call_id, resources.bus, resources.session_factory)
    # The registry needs the database: a leak is by definition cross-call, so detecting one
    # means comparing against identities issued to calls this process never saw.
    canary = CanaryRegistry(call_id, resources.session_factory)

    async def sink(chunk: bytes) -> None:
        await socket.send_bytes(chunk)

    async def flush() -> None:
        await socket.send_json({"type": "flush"})

    orchestrator = CallOrchestrator(
        CallConfig(
            call_id=call_id,
            # Built here, from settings, so `STT_FORMAT` and the model names are honoured.
            # `format=transliterate` in particular is not optional: it is the only format
            # that returned Latin-script Hinglish intact (DECISIONS 0009), and without it
            # the transcript comes back in Devanagari and the dashboard is unreadable to
            # judges who do not read Hindi.
            stt_config=SttConfig(
                model=settings.stt_model,
                language_code=settings.stt_language_code,
                sample_rate=INBOUND_SAMPLE_RATE,
                output_format=settings.stt_format,
                call_id=call_id,
            ),
            tts_model=settings.tts_model,
            tts_language=settings.tts_language,
            sample_rate_in=INBOUND_SAMPLE_RATE,
            sample_rate_out=OUTBOUND_SAMPLE_RATE,
            analysis_model=settings.analysis_model,
        ),
        stt=resources.stt,
        tts=resources.tts,
        llm=resources.llm,
        recorder=recorder,
        voices=voices,
        sink=sink,
        flush=flush,
        canary=canary,
    )

    await socket.send_json(
        {
            "type": "ready",
            "call_id": call_id,
            # The operator sees which fictional identity this call will hand out, so the
            # demo can show a canary being issued and later caught.
            "canary": canary.identity.model_dump(mode="json"),
            "sample_rate_in": INBOUND_SAMPLE_RATE,
            "sample_rate_out": OUTBOUND_SAMPLE_RATE,
        }
    )
    logger.info("call started", extra={"call_id": call_id, "source": "browser"})

    # The simulator reads the transcript off the same pub/sub channel the dashboard will,
    # rather than through a second path from the orchestrator. One publisher, one schema:
    # if the simulator shows the right thing, so will the dashboard.
    relay = asyncio.create_task(_relay_events(socket, resources, call_id))

    try:
        await orchestrator.run(_inbound(socket))
    except BudgetExceeded as exc:
        # The one failure worth telling the operator about in words: it is not a bug, it is
        # the guard doing its job, and the demo needs to know before it records.
        orchestrator.ended_reason = f"budget exceeded: {exc}"
        await _say(socket, {"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        orchestrator.ended_reason = "caller hung up"
    except ProviderError as exc:
        orchestrator.ended_reason = f"{type(exc).__name__}: {exc}"
        logger.warning("call ended on a provider error", extra={"call_id": call_id})
        await _say(socket, {"type": "error", "message": str(exc)})
    finally:
        # Give the last events a moment to come back around through Redis before the socket
        # closes, or the transcript loses the persona's final turn.
        await asyncio.sleep(0.05)
        relay.cancel()
        await asyncio.gather(relay, return_exceptions=True)
        await _finish_call_row(resources, call_id, orchestrator, canary.identity.identity_id)
        logger.info(
            "call ended",
            extra={
                "call_id": call_id,
                "reason": orchestrator.ended_reason,
                "turns": len(orchestrator.turns),
                "personas": [p.value for p in orchestrator.persona_sequence],
            },
        )
        await _say(socket, {"type": "ended", "reason": orchestrator.ended_reason})
        await _close(socket)


async def _relay_events(socket: WebSocket, resources: AppResources, call_id: str) -> None:
    """Forward this call's published events to the browser as control messages."""
    try:
        async for event in resources.bus.subscribe(call_id):
            await _send_event(socket, event)
    except asyncio.CancelledError:
        raise
    except Exception:
        # The transcript panel going blank is a cosmetic failure; the call must survive it.
        logger.exception("event relay stopped", extra={"call_id": call_id})


async def _send_event(socket: WebSocket, event: DomainEvent) -> None:
    await _say(socket, {"type": "event", "event": event.model_dump(mode="json")})


async def _inbound(socket: WebSocket) -> AsyncIterator[bytes]:
    """Yield caller audio until the socket closes.

    A text frame from the browser is a control message, not audio. Only `hangup` is
    recognised; anything else is ignored rather than fatal, so an older page talking to a
    newer server degrades instead of dropping the call.
    """
    while True:
        try:
            message = await socket.receive()
        except (WebSocketDisconnect, RuntimeError):
            return
        if message["type"] == "websocket.disconnect":
            return
        if (data := message.get("bytes")) is not None:
            yield data
        elif (text := message.get("text")) is not None and text.strip() == "hangup":
            return


async def _say(socket: WebSocket, payload: dict[str, object]) -> None:
    """Send a control message, tolerating a socket the browser has already closed."""
    try:
        await socket.send_json(payload)
    except (WebSocketDisconnect, RuntimeError):
        logger.debug("control message dropped: socket already closed")


async def _close(socket: WebSocket) -> None:
    with contextlib.suppress(WebSocketDisconnect, RuntimeError):
        await socket.close()


async def _create_call_row(resources: AppResources, call_id: str) -> None:
    """Insert the call so turns and events have something to reference."""
    try:
        async with resources.session_factory() as session, session.begin():
            session.add(CallRow(id=call_id, state=CallState.RINGING.value, source="browser"))
    except Exception:
        # A call that cannot be recorded is still a call worth taking: the dashboard reads
        # from Redis, so the demo survives a database that is down.
        logger.exception("could not create the call row", extra={"call_id": call_id})


async def _finish_call_row(
    resources: AppResources,
    call_id: str,
    orchestrator: CallOrchestrator,
    canary_identity_id: str | None = None,
) -> None:
    """Write the outcome: final state, why it ended, and who held the phone."""
    try:
        async with resources.session_factory() as session, session.begin():
            await session.execute(
                update(CallRow)
                .where(CallRow.id == call_id)
                .values(
                    ended_at=datetime.now(UTC),
                    state=orchestrator.state.value,
                    final_state=orchestrator.state.value,
                    ended_reason=(orchestrator.ended_reason or "")[:64] or None,
                    persona_sequence=[p.value for p in orchestrator.persona_sequence],
                    canary_identity_id=canary_identity_id,
                    scammer_seconds_wasted=orchestrator.elapsed_ms() / 1000.0,
                )
            )
    except Exception:
        logger.exception("could not finalise the call row", extra={"call_id": call_id})
