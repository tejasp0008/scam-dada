"""An OpenAI-compatible chat adapter, plus the fallback wrapper.

This is the whole LLM story for the project: Evon is not on Gnani's API platform, so the
only thing behind `LLMProvider` is an endpoint that speaks the OpenAI chat schema. Pointing
it at Evon later, self-hosted or hosted, is a change to `LLM_BASE_URL`.

Two behaviours here come from measurements rather than taste:

* `openai/gpt-oss-*` on default settings returns **empty content** — the entire token
  budget goes to reasoning — so an empty completion is treated as a failure to retry or
  fall back from, never as a valid reply.
* The same family emits stray markdown fences into what is supposed to be spoken dialogue.
  A persona that says "backtick backtick backtick" out loud is worse than a slow one, so
  the fallback strips them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, Final

import httpx
from pydantic import ValidationError

from app.providers.base import (
    AuthError,
    BadRequest,
    ChatMessage,
    LLMProvider,
    OutOfCredit,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
    Role,
    SchemaT,
    UpstreamTimeout,
    UsageRecorder,
)

logger = logging.getLogger(__name__)

#: How many times to re-ask when the model returns JSON that does not validate.
JSON_ATTEMPTS: Final[int] = 2


def strip_markdown_fences(text: str) -> str:
    """Remove backticks, which only ever arrive as stray markdown in spoken dialogue."""
    return text.replace("`", "")


def _strip_json_wrapper(text: str) -> str:
    """Pull the JSON object out of a reply that wrapped it in a fenced code block."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    return cleaned[start : end + 1] if 0 <= start < end else cleaned.strip()


class OpenAICompatibleLLM:
    """Chat completions against any endpoint speaking the OpenAI schema."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        client: httpx.AsyncClient,
        usage: UsageRecorder | None = None,
        name: str = "openai-compatible",
        extra_body: dict[str, Any] | None = None,
        system_suffix: str | None = None,
        strip_markdown: bool = False,
        request_timeout: float = 30.0,
    ) -> None:
        if not base_url:
            raise ValueError("an LLM base URL is required")
        if not model:
            raise ValueError("an LLM model name is required")
        self.name = name
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._model = model
        self._client = client
        self._usage = usage
        self._extra_body = dict(extra_body or {})
        self._system_suffix = system_suffix
        self._strip_markdown = strip_markdown
        self._timeout = httpx.Timeout(request_timeout, connect=5.0)
        self._remaining_tokens: int | None = None

    @property
    def model(self) -> str:
        """The model this instance calls."""
        return self._model

    @property
    def remaining_tokens(self) -> int | None:
        """What the vendor last said was left in this model's per-minute window."""
        return self._remaining_tokens

    def _read_meter(self, headers: httpx.Headers) -> None:
        """Record the rate-limit headers that come back with every response.

        Read on failures as well as successes, and on the stream's response head, which
        arrives before any token does. This is what lets a caller move the *next* request to
        another model instead of finding out by being refused.
        """
        raw = headers.get("x-ratelimit-remaining-tokens")
        if raw is None:
            return
        try:
            self._remaining_tokens = int(float(raw))
        except ValueError:
            logger.debug("unparseable rate-limit header", extra={"raw": raw[:40]})

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _payload(self, messages: Sequence[ChatMessage], **overrides: Any) -> dict[str, Any]:
        wire = [{"role": m.role.value, "content": m.content} for m in messages]
        if self._system_suffix:
            for message in wire:
                if message["role"] == Role.SYSTEM.value:
                    message["content"] = f"{message['content']}\n\n{self._system_suffix}"
                    break
            else:
                wire.insert(0, {"role": Role.SYSTEM.value, "content": self._system_suffix})
        return {"model": self._model, "messages": wire, **self._extra_body, **overrides}

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.8,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield reply text incrementally.

        `include_usage` is requested so the final chunk carries a token count and the
        budget guard can charge for what was actually spent rather than an estimate.
        """
        payload = self._payload(
            messages,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        if model:
            payload["model"] = model
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        emitted_text = False
        tokens = 0
        try:
            async with self._client.stream(
                "POST", self._url, headers=self._headers(), json=payload, timeout=self._timeout
            ) as response:
                self._read_meter(response.headers)
                if response.status_code != 200:
                    raise _error_for_status(
                        response.status_code, await response.aread(), response.headers
                    )
                async for data in _iter_data_lines(response):
                    tokens = int((data.get("usage") or {}).get("total_tokens") or tokens)
                    for choice in data.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content") or ""
                        if not piece:
                            continue
                        if self._strip_markdown:
                            piece = strip_markdown_fences(piece)
                            if not piece:
                                continue
                        emitted_text = True
                        yield piece
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise UpstreamTimeout(f"llm transport failure: {exc}") from exc
        finally:
            await self._record_tokens(tokens)

        if not emitted_text:
            # Measured on gpt-oss with default settings: a 200 response whose whole budget
            # went to reasoning tokens. Silence is not a reply.
            raise ProviderUnavailable(
                f"{self._model} returned no content (reasoning-only completion)"
            )

    async def complete_json(
        self,
        messages: Sequence[ChatMessage],
        schema: type[SchemaT],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> SchemaT:
        """Return a validated instance of `schema`, re-asking once if it does not validate."""
        conversation = list(messages)
        last_error: Exception | None = None

        for attempt in range(1, JSON_ATTEMPTS + 1):
            payload = self._payload(
                conversation,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            if model:
                payload["model"] = model
            try:
                response = await self._client.post(
                    self._url, headers=self._headers(), json=payload, timeout=self._timeout
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                raise UpstreamTimeout(f"llm transport failure: {exc}") from exc

            self._read_meter(response.headers)
            if response.status_code != 200:
                raise _error_for_status(response.status_code, response.content, response.headers)

            body = response.json()
            await self._record_tokens(int((body.get("usage") or {}).get("total_tokens") or 0))
            content = (body["choices"][0]["message"].get("content") or "").strip()
            if not content:
                last_error = ProviderUnavailable(f"{self._model} returned no JSON content")
            else:
                try:
                    return schema.model_validate_json(_strip_json_wrapper(content))
                except ValidationError as exc:
                    last_error = exc
                    logger.warning(
                        "llm json did not validate",
                        extra={"attempt": attempt, "schema": schema.__name__},
                    )
                    conversation = [
                        *messages,
                        ChatMessage(role=Role.ASSISTANT, content=content),
                        ChatMessage(
                            role=Role.USER,
                            content=(
                                "That did not match the required schema. Return only JSON "
                                f"matching: {json.dumps(schema.model_json_schema())}"
                            ),
                        ),
                    ]

        raise BadRequest(
            f"{self._model} did not produce valid {schema.__name__} in {JSON_ATTEMPTS} attempts"
        ) from last_error

    async def _record_tokens(self, tokens: int) -> None:
        if self._usage is None or tokens <= 0:
            return
        await self._usage.record(self.name, self._usage.cost.price_llm(tokens))


#: How long to leave a rate-limited model alone when it did not say. The cap is per minute,
#: so a minute is the honest guess; the server's own `retry-after` is preferred when present.
RATE_LIMIT_COOLDOWN: Final[float] = 60.0

#: Below this many tokens left in the window, stop sending the primary dialogue requests and
#: let the fallback take the turn. Sized at one whole request plus a margin: a measured
#: dialogue turn is about 1500 tokens in and 50 out, so anything under this cannot complete
#: and would come back as a refusal the caller hears as silence (DECISIONS 0034).
MIN_HEADROOM_TOKENS: Final[int] = 1800


class FallbackLLM:
    """Tries a primary model, then a second one.

    The primary here is a preview model, which means it can disappear or slow down without
    notice. Falling back is only safe before the first token has been handed to the caller:
    once the persona has started speaking a sentence, switching models mid-sentence would
    produce two different replies spliced together, so a late failure propagates instead.

    **A rate limit is remembered.** The per-minute token cap is per model, so once the
    primary has returned 429 it will keep returning 429 until its window rolls over - and
    asking it anyway costs a round trip on the one path where the caller is listening to
    silence. So the primary is skipped until its own `retry-after` has passed, and the
    fallback, which has a separate budget, carries the call in the meantime.
    """

    name = "llm-fallback"

    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider,
        *,
        first_token_timeout: float = 3.0,
        min_headroom: int = MIN_HEADROOM_TOKENS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._first_token_timeout = first_token_timeout
        self._min_headroom = min_headroom
        self._clock = clock or time.monotonic
        self._primary_rested_until = 0.0
        self.switches = 0
        self.rate_limited = 0
        self.pre_empted = 0

    @property
    def remaining_tokens(self) -> int | None:
        """Whichever model the next request would go to, and what it has left."""
        return (self._primary if self._primary_available() else self._fallback).remaining_tokens

    def _primary_available(self) -> bool:
        """Whether the next request should go to the primary at all.

        Two reasons it should not, and the second is the useful one. It has recently been
        refused, so asking again inside the same window buys a wasted round trip. Or its own
        meter says there is not enough left for a request this size - in which case moving
        now avoids the refusal altogether, which matters because the caller is listening to
        the gap it would open.
        """
        if self._clock() < self._primary_rested_until:
            return False
        remaining = self._primary.remaining_tokens
        if remaining is not None and remaining < self._min_headroom:
            self.pre_empted += 1
            logger.info(
                "primary llm is nearly out of budget; using the fallback for this turn",
                extra={"remaining": remaining, "needed": self._min_headroom},
            )
            return False
        return True

    def _rest_primary(self, exc: RateLimited) -> None:
        """Stand the primary down until its token window has rolled over."""
        wait = exc.retry_after if exc.retry_after is not None else RATE_LIMIT_COOLDOWN
        self._primary_rested_until = self._clock() + wait
        self.rate_limited += 1
        logger.warning(
            "primary llm is rate limited; resting it",
            extra={"seconds": round(wait, 1), "fallback": self._fallback.name},
        )

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.8,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream from the primary, switching to the fallback if it fails to get started."""
        if not self._primary_available():
            async for piece in self._fallback.stream_chat(
                messages, model=None, temperature=temperature, max_tokens=max_tokens
            ):
                yield piece
            return

        stream = self._primary.stream_chat(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )
        first: str | None = None
        try:
            first = await asyncio.wait_for(anext(stream), timeout=self._first_token_timeout)
        except StopAsyncIteration:
            reason = "primary produced an empty stream"
        except RateLimited as exc:
            self._rest_primary(exc)
            reason = f"RateLimited: {exc}"
        except (TimeoutError, ProviderError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
        else:
            reason = ""

        if reason:
            await _aclose(stream)
            self.switches += 1
            logger.warning(
                "falling back to the secondary llm",
                extra={"reason": reason, "fallback": self._fallback.name},
            )
            async for piece in self._fallback.stream_chat(
                messages, model=None, temperature=temperature, max_tokens=max_tokens
            ):
                yield piece
            return

        if first:
            yield first
        async for piece in stream:
            yield piece

    async def complete_json(
        self,
        messages: Sequence[ChatMessage],
        schema: type[SchemaT],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> SchemaT:
        """Try the primary, then the fallback. Nothing has been emitted, so this is safe."""
        if not self._primary_available():
            return await self._fallback.complete_json(
                messages, schema, model=None, temperature=temperature
            )
        try:
            return await self._primary.complete_json(
                messages, schema, model=model, temperature=temperature
            )
        except RateLimited as exc:
            self._rest_primary(exc)
            return await self._fallback.complete_json(
                messages, schema, model=None, temperature=temperature
            )
        except ProviderError as exc:
            self.switches += 1
            logger.warning(
                "falling back to the secondary llm for json",
                extra={"reason": str(exc), "fallback": self._fallback.name},
            )
            return await self._fallback.complete_json(
                messages, schema, model=None, temperature=temperature
            )


async def _aclose(stream: AsyncIterator[str]) -> None:
    """Close an abandoned stream so its connection is released promptly."""
    closer = getattr(stream, "aclose", None)
    if closer is not None:
        try:
            await closer()
        except Exception:  # a failing close must not mask the original failure
            logger.debug("failed to close abandoned llm stream", exc_info=True)


async def _iter_data_lines(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Yield parsed `data:` payloads from an OpenAI-style SSE stream."""
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if not chunk or chunk == "[DONE]":
            continue
        try:
            payload = json.loads(chunk)
        except json.JSONDecodeError:
            logger.warning("unparseable llm sse frame", extra={"raw": chunk[:200]})
            continue
        if isinstance(payload, dict):
            yield payload


def _retry_after(headers: httpx.Headers | None) -> float | None:
    """The server's own wait, in seconds, if it sent one."""
    if headers is None:
        return None
    raw = headers.get("retry-after") or headers.get("x-ratelimit-reset-tokens") or ""
    try:
        # Groq sends plain seconds on `retry-after` and a duration like "7.5s" on the reset
        # header. Anything else (an HTTP date) is left to the caller's own default.
        return float(raw.rstrip("s")) if raw else None
    except ValueError:
        return None


def _error_for_status(
    status: int, body: bytes, headers: httpx.Headers | None = None
) -> ProviderError:
    """Map an OpenAI-style error response onto our exception types."""
    message = body.decode(errors="replace")[:300]
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            message = str(parsed["error"].get("message", message))
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    detail = f"llm HTTP {status}: {message}"
    if status in (401, 403):
        return AuthError(detail)
    if status == 402:
        return OutOfCredit(detail)
    if status == 429:
        return RateLimited(detail, retry_after=_retry_after(headers))
    if status == 504:
        return UpstreamTimeout(detail)
    if status >= 500:
        return ProviderUnavailable(detail)
    # 404 from a model that was withdrawn is retryable only by switching models, which is
    # exactly what FallbackLLM does, so it must be a ProviderError subclass.
    return BadRequest(detail)
