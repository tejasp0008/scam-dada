"""Read-only endpoints over finished and in-progress calls.

The dashboard renders from these, and so does the evidence pack. There is nothing here that
mutates a call: a control room watches, and a call is driven by its own WebSocket.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import HTMLResponse, Response

from app.api.deps import Resources, SessionDep
from app.core.report import CallReport, CallSummary
from app.services import call_view
from app.services.evidence import render_html, render_pdf

router = APIRouter(prefix="/api", tags=["calls"])


@router.get("/calls", response_model=list[CallSummary])
async def list_calls(
    session: SessionDep, limit: int = Query(default=50, ge=1, le=200)
) -> list[CallSummary]:
    """Recent calls, newest first."""
    return await call_view.list_calls(session, limit=limit)


@router.get("/calls/{call_id}", response_model=CallReport)
async def get_call(call_id: str, session: SessionDep) -> CallReport:
    """Everything recorded about one call."""
    report = await call_view.get_report(session, call_id)
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such call")
    return report


@router.get("/calls/{call_id}/events")
async def get_events(call_id: str, resources: Resources, after_seq: int = -1) -> dict[str, object]:
    """Events from the live log, for a dashboard that reconnected mid-call.

    Served from Redis rather than Postgres because this is the reconnect path: the dashboard
    asks for everything after the last sequence number it saw, and the Redis list is already
    in that order and already the thing the live feed publishes to.
    """
    events = await resources.bus.history(call_id, after_seq=after_seq)
    return {
        "call_id": call_id,
        "after_seq": after_seq,
        "events": [event.model_dump(mode="json") for event in events],
    }


@router.get("/calls/{call_id}/evidence.html", response_class=HTMLResponse)
async def evidence_html(call_id: str, session: SessionDep) -> HTMLResponse:
    """The evidence pack as a page. Also what the PDF is rendered from."""
    report = await call_view.get_report(session, call_id)
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such call")
    return HTMLResponse(render_html(report))


@router.get("/calls/{call_id}/evidence.pdf")
async def evidence_pdf(call_id: str, session: SessionDep) -> Response:
    """The evidence pack as a PDF.

    503 rather than 500 when the renderer is missing: WeasyPrint is an optional dependency
    with system libraries behind it, and "install the extra" is operator information, not a
    bug report.
    """
    report = await call_view.get_report(session, call_id)
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such call")
    try:
        pdf = render_pdf(report)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="evidence-{call_id[:8]}.pdf"'},
    )
