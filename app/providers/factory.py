"""The single place provider implementations are constructed.

Selection is by name only, and every parameter arrives explicitly rather than by reading
settings, so this module stays free of any dependency on infra. The caller
(`app.main.build_resources`) is the composition root that reads configuration.

Because every adapter satisfies a Protocol in `base.py`, adding Sarvam or BharatGen later
means a new module here plus a config value, and no change anywhere else.
"""

from __future__ import annotations

import httpx

from app.providers.base import LLMProvider, STTProvider, TTSProvider, UsageRecorder
from app.providers.gnani_prisma import GnaniPrismaSTT
from app.providers.gnani_timbre import GnaniTimbreTTS
from app.providers.mock import MockLLMProvider, MockSTTProvider, MockTTSProvider
from app.providers.names import ProviderName
from app.providers.openai_compatible import FallbackLLM, OpenAICompatibleLLM

#: Appended to the fallback model's system prompt. Measured need: the gpt-oss family emits
#: markdown fences into text that is about to be spoken aloud.
PLAIN_TEXT_INSTRUCTION = (
    "Reply with plain spoken text only. Never use markdown, code fences, backticks, "
    "asterisks, bullet points or headings."
)


class ProviderNotImplemented(RuntimeError):
    """A configured provider has no adapter in this build."""


def build_stt_provider(
    name: ProviderName,
    *,
    api_key: str = "",
    ws_url: str = "",
    model: str = "",
    usage: UsageRecorder | None = None,
) -> STTProvider:
    """Return the configured speech-to-text provider."""
    if name is ProviderName.MOCK:
        return MockSTTProvider()
    if name is ProviderName.GNANI:
        return GnaniPrismaSTT(api_key=api_key, ws_url=ws_url, model=model, usage=usage)
    raise ProviderNotImplemented(f"no STT adapter for {name.value}")


def build_tts_provider(
    name: ProviderName,
    *,
    client: httpx.AsyncClient | None = None,
    api_key: str = "",
    sse_url: str = "",
    model: str = "",
    language: str = "",
    usage: UsageRecorder | None = None,
) -> TTSProvider:
    """Return the configured text-to-speech provider, uncached.

    Caching is applied by the caller, because it is disk I/O and therefore infra's job.
    """
    if name is ProviderName.MOCK:
        return MockTTSProvider()
    if name is ProviderName.GNANI:
        if client is None:
            raise ValueError("the Gnani TTS adapter needs an httpx client")
        return GnaniTimbreTTS(
            api_key=api_key,
            sse_url=sse_url,
            model=model,
            language=language,
            client=client,
            usage=usage,
        )
    raise ProviderNotImplemented(f"no TTS adapter for {name.value}")


def build_llm_provider(
    name: ProviderName,
    *,
    client: httpx.AsyncClient | None = None,
    base_url: str = "",
    api_key: str = "",
    model: str = "",
    fallback_model: str = "",
    fallback_reasoning_effort: str = "low",
    reasoning_effort: str = "",
    first_token_timeout: float = 3.0,
    request_timeout: float = 30.0,
    usage: UsageRecorder | None = None,
) -> LLMProvider:
    """Return the configured LLM provider, wrapped in a fallback when one is configured."""
    if name is ProviderName.MOCK:
        return MockLLMProvider()
    if name is not ProviderName.OPENAI:
        raise ProviderNotImplemented(f"no LLM adapter for {name.value}")
    if client is None:
        raise ValueError("the OpenAI-compatible LLM adapter needs an httpx client")

    primary = OpenAICompatibleLLM(
        base_url=base_url,
        api_key=api_key,
        model=model,
        client=client,
        usage=usage,
        name=f"llm:{model}",
        # The dialogue model needs this as much as the fallback does. Without it a reasoning
        # model given a dense persona prompt returns empty content, every such turn costs a
        # second model call, and the reply lands after the caller has started talking again.
        extra_body={"reasoning_effort": reasoning_effort} if reasoning_effort else None,
        request_timeout=request_timeout,
    )
    if not fallback_model or fallback_model == model:
        return primary

    fallback = OpenAICompatibleLLM(
        base_url=base_url,
        api_key=api_key,
        model=fallback_model,
        client=client,
        usage=usage,
        name=f"llm:{fallback_model}",
        # Without an explicit effort this family spends its whole budget on reasoning and
        # returns empty content; with it, replies arrive and stay short.
        extra_body={"reasoning_effort": fallback_reasoning_effort},
        system_suffix=PLAIN_TEXT_INSTRUCTION,
        strip_markdown=True,
        request_timeout=request_timeout,
    )
    return FallbackLLM(primary, fallback, first_token_timeout=first_token_timeout)
