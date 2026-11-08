"""The evidence pack: one call, rendered for a human who was not listening.

HTML first, PDF from that same HTML. Two reasons rather than one:

* The HTML is viewable without WeasyPrint installed, which matters because WeasyPrint needs
  system libraries and a demo machine that cannot produce the PDF should still produce the
  pack.
* A PDF rendered from the page the operator already looked at cannot disagree with it.

**Nothing here computes anything.** Every number comes from `CallReport`, which came from
`app.core.report`, which is pure and tested. A template that did its own arithmetic would be
a second implementation of the figures, and the two would eventually differ in the document
that is supposed to be the record.
"""

from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.core.report import CallReport

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "web" / "templates"

#: Autoescaped, and that is load-bearing rather than hygiene: every string in a report came
#: out of a scammer's mouth by way of a transcriber, and an evidence pack is a document that
#: gets forwarded. Unescaped transcript text is a stored injection.
_ENV = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(default_for_string=True, default=True),
    trim_blocks=True,
    lstrip_blocks=True,
)

#: Human labels for entity types, so the pack does not read like a database dump.
ENTITY_LABELS: dict[str, str] = {
    "upi_id": "UPI handle",
    "phone": "Phone number",
    "ifsc": "Bank IFSC",
    "account_number": "Destination account",
    "payee_name": "Destination account holder",
    "amount": "Amount demanded",
    "officer_name": "Claimed name",
    "organisation": "Claimed organisation",
    "case_number": "Case or reference number",
}

LEG_LABELS: dict[str, str] = {
    "stt_final_ms": "End of speech to transcript",
    "llm_first_token_ms": "Reply generation, first token",
    "tts_first_byte_ms": "Speech synthesis, first audio",
}


def render_html(report: CallReport) -> str:
    """The evidence pack as a self-contained page."""
    template = _ENV.get_template("evidence.html")
    return template.render(
        report=report,
        summary=report.summary,
        entities_by_type=report.entities_by_type,
        entity_labels=ENTITY_LABELS,
        leg_labels=LEG_LABELS,
    )


def render_pdf(report: CallReport) -> bytes:
    """The same pack as a PDF.

    Raises `RuntimeError` when WeasyPrint is not installed, so the caller can tell an
    operator to install the extra rather than reporting a bug.
    """
    try:
        # Imported here rather than at module scope: it is an optional extra, and a
        # missing system library must not stop the app importing.
        from weasyprint import HTML
    except (ImportError, OSError) as exc:  # pragma: no cover - depends on the install
        # OSError as well as ImportError, and that distinction is not theoretical: with the
        # package installed but Pango absent, the import fails with
        # `OSError: cannot load library 'libgobject-2.0-0'`. Catching only ImportError
        # turned a missing system library into a 500 instead of an explanation.
        raise RuntimeError(
            f"PDF rendering is unavailable: {exc}. "
            "It needs the 'pdf' extra (pip install -e '.[pdf]') and its system libraries "
            "(on macOS: brew install pango). The HTML pack at "
            "/api/calls/{call_id}/evidence.html needs neither."
        ) from exc

    document: bytes | None = HTML(string=render_html(report)).write_pdf()
    if document is None:  # pragma: no cover - WeasyPrint returns bytes when given no target
        raise RuntimeError("the PDF renderer produced nothing")
    return document
