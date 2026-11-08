#!/usr/bin/env python3
"""Price one call against the per-minute token limit, request by request.

Every model on this account is capped at the same **8000 tokens per minute**, and the cap is
per model rather than per account - the limit headers on a live response say so, and
`--live` prints them. So what matters is not the total a call spends but how much of it
lands on any one model inside any one minute.

The two paths are priced separately because they are billed separately:

* **dialogue** - the persona system prompt plus the conversation window plus this turn's
  notes, once per caller turn. Unavoidable: the caller is waiting for it.
* **analysis** - the analyst prompt plus a transcript window, every Nth caller turn. It can
  be made rarer, and it is the one that can be dropped, because nothing on the call waits
  for it.

Input tokens are counted with `cl100k_base`, which is not Groq's tokeniser for either model
family and lands within a few percent on Latin-script Hinglish - close enough to size a
budget, not close enough to quote. Completion tokens are the part that cannot be estimated
at all: a reasoning model spends most of its output on tokens nobody ever sees. So the
output side comes from a real response's `usage` block, under `--live`, or is left out and
labelled as left out.

    scripts/token_budget.py                      # input tokens only, no API calls
    scripts/token_budget.py --live               # adds real completion counts and headers
    scripts/token_budget.py --turns-per-minute 8 # a faster caller
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import tiktoken

from app.core.models import PersonaId, Speaker, Turn
from app.core.personas import LEAD_PERSONA, REPLY_STYLE_NOTE
from app.core.turn import TurnBudget, recent_turns
from app.infra.settings import get_settings
from app.providers.base import ChatMessage
from app.services.analysis import ANALYSIS_WINDOW_TURNS, LlmAnalysis, analysis_messages
from app.services.orchestrator import CallConfig
from app.services.prompts import dialogue_messages

#: The one limit that matters, read off `x-ratelimit-limit-tokens` for every model on this
#: account on 2026-11-08. Printed alongside the measurement rather than assumed: `--live`
#: reports what the header actually says today.
TOKENS_PER_MINUTE = 8000

#: What wrapping a message in the chat schema costs on top of its text. OpenAI's documented
#: figure for `cl100k_base`; it is small, and leaving it out would flatter the result.
PER_MESSAGE_OVERHEAD = 4

_encoding = tiktoken.get_encoding("cl100k_base")


def count(messages: Sequence[ChatMessage]) -> int:
    """Input tokens for one request, including per-message framing."""
    return sum(len(_encoding.encode(m.content)) + PER_MESSAGE_OVERHEAD for m in messages)


#: What the primary keeps in reserve before handing turns to the fallback. Mirrors
#: `MIN_HEADROOM_TOKENS` in the provider: below this it stops being asked, so this much of
#: its minute is never usable.
HEADROOM = 1800


@dataclass(frozen=True)
class RequestPath:
    """What one request path costs per call to it, and where it goes when the first is full."""

    label: str
    model: str
    calls_per_caller_turn: float
    input_tokens: int
    output_tokens: int | None
    #: Where turns go once `model` is out of budget, with that model's own cost per turn.
    #: Dialogue has one; analysis deliberately has none - it degrades to the keyword pass
    #: rather than spend a budget the caller is waiting on.
    spill: tuple[str, int] | None = None

    @property
    def total(self) -> int | None:
        return None if self.output_tokens is None else self.input_tokens + self.output_tokens

    def per_minute(self, turns_per_minute: float) -> float:
        billed = self.total if self.total is not None else self.input_tokens
        return billed * self.calls_per_caller_turn * turns_per_minute

    def allocate(self, turns_per_minute: float) -> dict[str, float]:
        """Spread a minute's spend over this path's models the way the router will.

        The primary takes turns until its usable budget is gone, then the fallback takes the
        rest. This is the arithmetic that decides whether a fast call runs clean or stalls,
        so it is reported rather than described.
        """
        billed = self.total if self.total is not None else self.input_tokens
        calls = self.calls_per_caller_turn * turns_per_minute
        usable = max(0.0, TOKENS_PER_MINUTE - HEADROOM)
        on_primary = min(calls, usable / billed) if billed else calls
        spend = {self.model: on_primary * billed}
        if self.spill is None:
            return spend
        spill_model, spill_cost = self.spill
        spend[spill_model] = (calls - on_primary) * spill_cost
        return spend


def load_turns(events: Path) -> list[Turn]:
    """The transcript of a recorded call, as the orchestrator held it."""
    payload = json.loads(events.read_text())
    turns: list[Turn] = []
    for event in payload:
        if event.get("type") != "transcript":
            continue
        caller = event["speaker"] == "caller"
        turns.append(
            Turn(
                seq=len(turns),
                speaker=Speaker.CALLER if caller else Speaker.PERSONA,
                text=event["text"],
                persona=None if caller else PersonaId.DADA,
                started_at_ms=0,
            )
        )
    if not turns:
        raise SystemExit(f"{events} holds no transcript events")
    return turns


def dialogue_cost(
    turns: Sequence[Turn], budget: TurnBudget
) -> tuple[int, list[ChatMessage]]:
    """The worst dialogue request in the call, and its token count.

    The worst rather than the mean: the window fills up within a few turns and then stays
    full, so the peak is what a steady fast call actually spends every turn.
    """
    worst = 0
    heaviest: list[ChatMessage] = []
    for index, turn in enumerate(turns):
        if turn.speaker is not Speaker.CALLER:
            continue
        window = recent_turns(turns[: index + 1], budget)
        messages = dialogue_messages(LEAD_PERSONA, window, [REPLY_STYLE_NOTE])
        tokens = count(messages)
        if tokens > worst:
            worst, heaviest = tokens, messages
    return worst, heaviest


def analysis_cost(turns: Sequence[Turn]) -> tuple[int, list[ChatMessage]]:
    """The worst analysis request in the call, and its token count."""
    worst = 0
    heaviest: list[ChatMessage] = []
    for index, turn in enumerate(turns):
        if turn.speaker is not Speaker.CALLER:
            continue
        messages = analysis_messages(turns[: index + 1][-ANALYSIS_WINDOW_TURNS:])
        tokens = count(messages)
        if tokens > worst:
            worst, heaviest = tokens, messages
    return worst, heaviest


async def live_usage(
    messages: Sequence[ChatMessage], model: str, *, json_mode: bool, effort: str | None = None
) -> tuple[dict[str, int], dict[str, str]]:
    """Send one real request and return its `usage` block and rate-limit headers."""
    settings = get_settings()
    body: dict[str, object] = {
        "model": model,
        "messages": [{"role": m.role.value, "content": m.content} for m in messages],
        "temperature": 0.0 if json_mode else 0.8,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
        body["reasoning_effort"] = settings.llm_analysis_reasoning_effort
    else:
        body["reasoning_effort"] = effort or settings.llm_reasoning_effort
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(
            settings.llm_base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            json=body,
        )
    headers = {k: v for k, v in response.headers.items() if "ratelimit-limit-tokens" in k.lower()}
    if response.status_code != 200:
        raise SystemExit(f"{model} returned HTTP {response.status_code}: {response.text[:200]}")
    return response.json().get("usage") or {}, headers


def report(paths: Sequence[RequestPath], rates: Sequence[float]) -> None:
    print(f"{'path':<10} {'model':<22} {'per call':>9} {'in':>7} {'out':>7} {'total':>7}")
    print("-" * 68)
    for path in paths:
        out = "n/a" if path.output_tokens is None else str(path.output_tokens)
        total = "n/a" if path.total is None else str(path.total)
        print(
            f"{path.label:<10} {path.model:<22} "
            f"{path.calls_per_caller_turn:>9.2f} {path.input_tokens:>7} {out:>7} {total:>7}"
        )
        if path.spill is not None:
            spill_model, spill_cost = path.spill
            print(f"{'  spills to':<10} {spill_model:<22} {'':>9} {'':>7} {'':>7} {spill_cost:>7}")

    for rate in rates:
        print(f"\nat {rate:g} caller turns per minute - limit {TOKENS_PER_MINUTE}/min per model")
        spend: dict[str, float] = {}
        sources: dict[str, set[str]] = {}
        for path in paths:
            for model, tokens in path.allocate(rate).items():
                spend[model] = spend.get(model, 0.0) + tokens
                sources.setdefault(model, set()).add(path.label)
        for model, tokens in spend.items():
            share = tokens / TOKENS_PER_MINUTE
            verdict = "OVER" if share > 1 else ("tight" if share > 0.8 else "ok")
            labels = "+".join(sorted(sources[model]))
            print(f"  {model:<22} {tokens:>8.0f}  {share:>5.0%} of limit  {verdict:<5} ({labels})")
        worst = max(spend.values()) / TOKENS_PER_MINUTE if spend else 0.0
        print(f"  -> {'clean' if worst <= 1 else 'STALLS: no model has room'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("var/calls/f0fe7592.events.json"),
        help="a recorded call to price; synthetic transcripts only",
    )
    parser.add_argument(
        "--turns-per-minute",
        type=float,
        nargs="*",
        default=[5, 7, 9],
        help="caller turn rates to report (5 is the scripted call, 7 a fast live one)",
    )
    parser.add_argument(
        "--live", action="store_true", help="send one real request per path to get usage"
    )
    args = parser.parse_args()

    settings = get_settings()
    config = CallConfig(call_id="budget", stt_config=None, tts_model="", tts_language="")  # type: ignore[arg-type]
    turns = load_turns(args.events)

    dialogue_tokens, dialogue_request = dialogue_cost(turns, config.turn_budget)
    analysis_tokens, analysis_request = analysis_cost(turns)

    dialogue_out: int | None = None
    analysis_out: int | None = None
    spill: tuple[str, int] | None = None
    if args.live:
        usage, headers = asyncio.run(
            live_usage(dialogue_request, settings.llm_model, json_mode=False)
        )
        dialogue_out = int(usage.get("completion_tokens") or 0)
        print(f"live dialogue  prompt={usage.get('prompt_tokens')} "
              f"completion={dialogue_out} headers={headers}")
        usage, headers = asyncio.run(
            live_usage(analysis_request, settings.analysis_model, json_mode=True)
        )
        analysis_out = int(usage.get("completion_tokens") or 0)
        print(f"live analysis  prompt={usage.get('prompt_tokens')} "
              f"completion={analysis_out} headers={headers}")
        if settings.llm_fallback_model:
            usage, headers = asyncio.run(
                live_usage(
                    dialogue_request,
                    settings.llm_fallback_model,
                    json_mode=False,
                    effort=settings.llm_fallback_reasoning_effort,
                )
            )
            spill = (
                settings.llm_fallback_model,
                int(usage.get("total_tokens") or 0),
            )
            print(f"live fallback  prompt={usage.get('prompt_tokens')} "
                  f"completion={usage.get('completion_tokens')} headers={headers}")
        print()

    every = max(1, config.analysis_every_n_turns)
    report(
        [
            RequestPath(
                "dialogue", settings.llm_model, 1.0, dialogue_tokens, dialogue_out, spill=spill
            ),
            RequestPath(
                "analysis", settings.analysis_model, 1 / every, analysis_tokens, analysis_out
            ),
        ],
        args.turns_per_minute,
    )
    print(
        f"\nwindow: {config.turn_budget.max_turns} turns / {config.turn_budget.max_chars} chars"
        f" | analysis: every {every} caller turns, {ANALYSIS_WINDOW_TURNS}-turn transcript"
    )
    if not args.live:
        print("completion tokens not measured; rerun with --live for the billed total")
    _ = LlmAnalysis  # the schema the analysis request is validated against


if __name__ == "__main__":
    main()
