"""The control room's live feed.

One socket per call being watched. The dashboard sends the last sequence number it has, gets
the backlog after that number, and then streams. That is the whole protocol, and it exists
because a dashboard that reconnects mid-call must not show a call that appears to have
started at the moment of reconnection.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.deps import AppResources

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dashboard"])


@router.websocket("/ws/dashboard/{call_id}")
async def dashboard_socket(socket: WebSocket, call_id: str, after_seq: int = -1) -> None:
    """Stream one call's events, starting with whatever the client has missed."""
    resources: AppResources = socket.app.state.resources
    await socket.accept()

    try:
        backlog = await resources.bus.history(call_id, after_seq=after_seq)
    except Exception:
        # A dashboard with no backlog is degraded; one that fails to open is useless.
        logger.exception("could not read the event backlog", extra={"call_id": call_id})
        backlog = []

    last_seq = after_seq
    for event in backlog:
        await socket.send_json(event.model_dump(mode="json"))
        last_seq = event.seq

    try:
        async for event in resources.bus.subscribe(call_id):
            # The backlog and the live stream overlap by however long the first read took,
            # so anything already sent is dropped rather than shown twice.
            if event.seq <= last_seq:
                continue
            last_seq = event.seq
            await socket.send_json(event.model_dump(mode="json"))
    except (WebSocketDisconnect, asyncio.CancelledError):
        raise
    except Exception:
        logger.exception("dashboard feed stopped", extra={"call_id": call_id})
    finally:
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await socket.close()
