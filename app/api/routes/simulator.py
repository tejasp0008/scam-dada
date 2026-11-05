"""The browser call simulator: a scammer's handset, in a page.

Serving a page from the API is a deliberate simplification for a four-day build. There is no
separate frontend to build, deploy or keep in sync with the event schema, and the demo is
recorded from this page.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"
TEMPLATES = Jinja2Templates(directory=str(WEB_DIR / "templates"))

router = APIRouter(tags=["simulator"])


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
@router.get("/simulator", response_class=HTMLResponse)
async def simulator(request: Request) -> HTMLResponse:
    """The call simulator page: a scammer's handset."""
    return TEMPLATES.TemplateResponse(request, "simulator.html")


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """The control room. This is the page the demo video films."""
    return TEMPLATES.TemplateResponse(request, "dashboard.html")
