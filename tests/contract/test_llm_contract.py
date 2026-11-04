"""The OpenAI-compatible adapter and the fallback wrapper, offline.

The two odd behaviours pinned here were both measured on Groq, not imagined: gpt-oss
returns HTTP 200 with no content when left on its defaults, and the same family emits
markdown fences into text that is about to be spoken aloud.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel, Field

from app.providers.base import (
    AuthError,
    BadRequest,
    ChatMessage,
    OutOfCredit,
    ProviderUnavailable,
    RateLimited,
    Role,
    UpstreamTimeout,
)
from app.providers.factory import PLAIN_TEXT_INSTRUCTION, build_llm_provider
from app.providers.names import ProviderName
from app.providers.openai_compatible import (
    FallbackLLM,
    OpenAICompatibleLLM,
    strip_markdown_fences,
)

pytestmark = pytest.mark.contract

MESSAGES = [
    ChatMessage(role=Role.SYSTEM, content="You are Dadi."),
    ChatMessage(role=Role.USER, content="Main cyber cell se bol raha hoon."),
]


class Stage(BaseModel):
    stage: str = Field(default="hook")
    pressure: float = Field(default=0.0, ge=0.0, le=1.0)


def llm(handler: object, **kwargs: object) -> OpenAICompatibleLLM:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    params: dict[str, object] = {
        "base_url": "https://example.invalid/openai/v1",
        "api_key": "test-key",
        "model": "test-model",
        "client": client,
    }
    params.update(kwargs)
    return OpenAICompatibleLLM(**params)  # type: ignore[arg-type]


def sse(body: str) -> object:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    return handler


async def test_a_recorded_stream_yields_text_pieces(fixture_text) -> None:
    pieces = [
        p async for p in llm(sse(fixture_text("openai_chat_stream.txt"))).stream_chat(MESSAGES)
    ]
    assert pieces == ["Arre ", "beta, ", "kaun?"]


async def test_the_stream_is_incremental_so_tts_can_start_mid_reply(fixture_text) -> None:
    pieces = [
        p async for p in llm(sse(fixture_text("openai_chat_stream.txt"))).stream_chat(MESSAGES)
    ]
    assert len(pieces) > 1


async def test_a_reasoning_only_completion_is_treated_as_a_failure(fixture_text) -> None:
    """Measured on gpt-oss defaults: HTTP 200, usage reported, no content. Silence is
    not a reply, and must be loud enough to trigger a fallback."""
    with pytest.raises(ProviderUnavailable, match="no content"):
        [
            p
            async for p in llm(sse(fixture_text("openai_chat_stream_empty.txt"))).stream_chat(
                MESSAGES
            )
        ]


async def test_markdown_fences_are_stripped_when_asked(fixture_text) -> None:
    body = fixture_text("openai_chat_stream_fenced.txt")
    raw = [p async for p in llm(sse(body)).stream_chat(MESSAGES)]
    assert "```" in "".join(raw)
    cleaned = [p async for p in llm(sse(body), strip_markdown=True).stream_chat(MESSAGES)]
    assert "`" not in "".join(cleaned)
    assert "".join(cleaned).strip() == "Arre beta"


async def test_a_reply_of_only_fences_still_counts_as_empty(fixture_text) -> None:
    body = (
        'data: {"choices": [{"delta": {"content": "```"}}]}\n\n'
        'data: {"choices": [], "usage": {"total_tokens": 10}}\n\n'
        "data: [DONE]\n\n"
    )
    with pytest.raises(ProviderUnavailable):
        [p async for p in llm(sse(body), strip_markdown=True).stream_chat(MESSAGES)]


def test_fence_stripping_leaves_hinglish_untouched() -> None:
    assert strip_markdown_fences("Haan beta, kaun?") == "Haan beta, kaun?"
    assert strip_markdown_fences("```Arre```") == "Arre"


async def test_token_usage_from_the_final_chunk_is_recorded(fixture_text) -> None:
    from app.core.cost import CostModel

    class Recorder:
        def __init__(self) -> None:
            self.cost = CostModel()
            self.tokens: list[float] = []

        async def check(self, provider: str, charge: object) -> None:
            pass

        async def record(self, provider: str, charge: object, **kw: object) -> None:
            self.tokens.append(charge.raw_units)  # type: ignore[attr-defined]

    recorder = Recorder()
    provider = llm(sse(fixture_text("openai_chat_stream.txt")), usage=recorder)
    [p async for p in provider.stream_chat(MESSAGES)]
    assert recorder.tokens == [102.0]


async def test_streaming_asks_for_usage_in_the_stream() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, text="data: [DONE]\n\n")

    with pytest.raises(ProviderUnavailable):
        [p async for p in llm(handler).stream_chat(MESSAGES)]
    assert seen["stream"] is True
    assert seen["stream_options"] == {"include_usage": True}


async def test_complete_json_validates_into_the_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"stage": "payment", "pressure": 0.9}'}}],
                "usage": {"total_tokens": 50},
            },
        )

    result = await llm(handler).complete_json(MESSAGES, Stage)
    assert result.stage == "payment"
    assert result.pressure == 0.9


async def test_json_wrapped_in_a_code_fence_is_still_parsed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '```json\n{"stage": "fear", "pressure": 0.5}\n```'}}
                ],
                "usage": {"total_tokens": 20},
            },
        )

    assert (await llm(handler).complete_json(MESSAGES, Stage)).stage == "fear"


async def test_invalid_json_is_re_asked_once_then_fails() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"pressure": 42}'}}],
                "usage": {"total_tokens": 10},
            },
        )

    with pytest.raises(BadRequest, match="Stage"):
        await llm(handler).complete_json(MESSAGES, Stage)
    assert attempts["n"] == 2


async def test_a_second_attempt_can_succeed() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        content = '{"pressure": 42}' if attempts["n"] == 1 else '{"stage": "hook", "pressure": 0.1}'
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 10}},
        )

    assert (await llm(handler).complete_json(MESSAGES, Stage)).pressure == 0.1
    assert attempts["n"] == 2


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, AuthError),
        (403, AuthError),
        (402, OutOfCredit),
        (429, RateLimited),
        (404, BadRequest),
        (500, ProviderUnavailable),
        (504, UpstreamTimeout),
    ],
)
async def test_error_statuses_map_to_our_exception_types(
    status: int, expected: type[Exception]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "nope"}})

    with pytest.raises(expected):
        [p async for p in llm(handler).stream_chat(MESSAGES)]


async def test_a_transport_failure_becomes_an_upstream_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(UpstreamTimeout):
        [p async for p in llm(handler).stream_chat(MESSAGES)]


async def test_a_system_suffix_is_appended_to_the_existing_system_message() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, text="data: [DONE]\n\n")

    with pytest.raises(ProviderUnavailable):
        [p async for p in llm(handler, system_suffix="NO MARKDOWN").stream_chat(MESSAGES)]
    messages = seen["messages"]
    assert len(messages) == 2  # type: ignore[arg-type]
    assert messages[0]["content"].endswith("NO MARKDOWN")  # type: ignore[index]


async def test_extra_body_params_reach_the_request() -> None:
    """`reasoning_effort` is what stops gpt-oss returning empty content."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, text="data: [DONE]\n\n")

    with pytest.raises(ProviderUnavailable):
        [
            p
            async for p in llm(handler, extra_body={"reasoning_effort": "low"}).stream_chat(
                MESSAGES
            )
        ]
    assert seen["reasoning_effort"] == "low"


# --- the fallback wrapper -------------------------------------------------------------


class ScriptedLLM:
    """A stand-in that fails, stalls or succeeds on demand."""

    def __init__(
        self,
        name: str,
        *,
        pieces: list[str] | None = None,
        error: Exception | None = None,
        stall: bool = False,
        fail_after: int | None = None,
    ) -> None:
        self.name = name
        self._pieces = ["ok "] if pieces is None else pieces
        self._error = error
        self._stall = stall
        self._fail_after = fail_after
        self.calls = 0

    async def stream_chat(self, messages, *, model=None, temperature=0.8, max_tokens=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self._error is not None:
            raise self._error
        if self._stall:
            import asyncio

            await asyncio.sleep(10)
        for index, piece in enumerate(self._pieces):
            if self._fail_after is not None and index == self._fail_after:
                raise ProviderUnavailable("died mid-stream")
            yield piece

    async def complete_json(self, messages, schema, *, model=None, temperature=0.0):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self._error is not None:
            raise self._error
        return schema()


async def test_the_fallback_takes_over_when_the_primary_errors() -> None:
    primary = ScriptedLLM("primary", error=BadRequest("llm HTTP 404: model withdrawn"))
    fallback = ScriptedLLM("fallback", pieces=["Arre ", "beta"])
    wrapper = FallbackLLM(primary, fallback)
    text = "".join([p async for p in wrapper.stream_chat(MESSAGES)])
    assert text == "Arre beta"
    assert wrapper.switches == 1
    assert fallback.calls == 1


async def test_the_fallback_takes_over_when_the_primary_stalls() -> None:
    """A preview model that hangs is worse than one that errors: the scammer hears silence."""
    primary = ScriptedLLM("primary", stall=True)
    fallback = ScriptedLLM("fallback", pieces=["Haan"])
    wrapper = FallbackLLM(primary, fallback, first_token_timeout=0.05)
    assert "".join([p async for p in wrapper.stream_chat(MESSAGES)]) == "Haan"
    assert wrapper.switches == 1


async def test_an_empty_primary_stream_triggers_the_fallback() -> None:
    primary = ScriptedLLM("primary", pieces=[])
    fallback = ScriptedLLM("fallback", pieces=["Haan"])
    wrapper = FallbackLLM(primary, fallback)
    assert "".join([p async for p in wrapper.stream_chat(MESSAGES)]) == "Haan"
    assert wrapper.switches == 1


async def test_a_healthy_primary_is_not_second_guessed() -> None:
    primary = ScriptedLLM("primary", pieces=["Haan ", "beta"])
    fallback = ScriptedLLM("fallback")
    wrapper = FallbackLLM(primary, fallback)
    assert "".join([p async for p in wrapper.stream_chat(MESSAGES)]) == "Haan beta"
    assert wrapper.switches == 0
    assert fallback.calls == 0


async def test_a_failure_after_the_first_token_propagates_instead_of_switching() -> None:
    """Switching mid-sentence would splice two different replies into one utterance."""
    primary = ScriptedLLM("primary", pieces=["Haan ", "beta ", "kaun"], fail_after=1)
    fallback = ScriptedLLM("fallback", pieces=["different reply"])
    wrapper = FallbackLLM(primary, fallback)
    collected: list[str] = []
    with pytest.raises(ProviderUnavailable, match="mid-stream"):
        async for piece in wrapper.stream_chat(MESSAGES):
            collected.append(piece)
    assert collected == ["Haan "]
    assert fallback.calls == 0


async def test_complete_json_falls_back_too() -> None:
    primary = ScriptedLLM("primary", error=ProviderUnavailable("down"))
    fallback = ScriptedLLM("fallback")
    wrapper = FallbackLLM(primary, fallback)
    assert isinstance(await wrapper.complete_json(MESSAGES, Stage), Stage)
    assert wrapper.switches == 1


# --- what the factory builds ----------------------------------------------------------


def test_the_factory_configures_the_fallback_against_empty_replies_and_markdown() -> None:
    provider = build_llm_provider(
        ProviderName.OPENAI,
        client=httpx.AsyncClient(),
        base_url="https://example.invalid/v1",
        api_key="k",
        model="qwen/qwen3.8-27b",
        fallback_model="openai/gpt-oss-120b",
    )
    assert isinstance(provider, FallbackLLM)
    secondary = provider._fallback
    assert secondary._extra_body == {"reasoning_effort": "low"}  # type: ignore[attr-defined]
    assert secondary._strip_markdown is True  # type: ignore[attr-defined]
    assert secondary._system_suffix == PLAIN_TEXT_INSTRUCTION  # type: ignore[attr-defined]


def test_no_fallback_is_wrapped_when_it_would_be_the_same_model() -> None:
    provider = build_llm_provider(
        ProviderName.OPENAI,
        client=httpx.AsyncClient(),
        base_url="https://example.invalid/v1",
        api_key="k",
        model="same",
        fallback_model="same",
    )
    assert isinstance(provider, OpenAICompatibleLLM)
