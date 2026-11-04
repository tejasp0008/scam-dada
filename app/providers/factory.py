"""The single place provider implementations are chosen.

Selection is by config value only. Because every adapter satisfies a Protocol in
`base.py`, swapping Gnani for another vendor, or for the mocks, never reaches past this
module.
"""

from __future__ import annotations

from enum import StrEnum

from app.providers.base import LLMProvider, STTProvider, TTSProvider
from app.providers.mock import MockLLMProvider, MockSTTProvider, MockTTSProvider


class ProviderName(StrEnum):
    GNANI = "gnani"
    OPENAI = "openai"
    MOCK = "mock"


class ProviderNotImplemented(RuntimeError):
    """A configured provider has no adapter yet in this build."""


def build_stt_provider(name: ProviderName) -> STTProvider:
    """Return the configured speech-to-text provider."""
    if name is ProviderName.MOCK:
        return MockSTTProvider()
    raise ProviderNotImplemented(f"no STT adapter for {name.value} yet (arrives in phase 1)")


def build_tts_provider(name: ProviderName) -> TTSProvider:
    """Return the configured text-to-speech provider."""
    if name is ProviderName.MOCK:
        return MockTTSProvider()
    raise ProviderNotImplemented(f"no TTS adapter for {name.value} yet (arrives in phase 1)")


def build_llm_provider(name: ProviderName) -> LLMProvider:
    """Return the configured LLM provider."""
    if name is ProviderName.MOCK:
        return MockLLMProvider()
    raise ProviderNotImplemented(f"no LLM adapter for {name.value} yet (arrives in phase 1)")
