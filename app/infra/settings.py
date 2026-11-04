"""All configuration, from the environment, validated once at startup.

The only global in the project. Everything else is injected.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.providers.factory import ProviderName


class Settings(BaseSettings):
    """Environment-backed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True
    )

    app_env: str = "dev"
    log_level: str = "INFO"

    database_url: PostgresDsn = Field(
        default=PostgresDsn("postgresql+asyncpg://dadi:dadi@localhost:5432/scamdadi")
    )
    redis_url: RedisDsn = Field(default=RedisDsn("redis://localhost:6379/0"))

    stt_provider: ProviderName = ProviderName.MOCK
    tts_provider: ProviderName = ProviderName.MOCK
    llm_provider: ProviderName = ProviderName.MOCK

    # Gnani. One key serves Prisma and Timbre, sent as the X-API-Key-ID header.
    gnani_api_key: str = ""
    gnani_stt_ws_url: str = "wss://api.vachana.ai/stt/v3/stream"
    gnani_stt_rest_url: str = "https://api.vachana.ai/stt/v3"
    gnani_tts_sse_url: str = "https://api.vachana.ai/api/v1/tts/sse"
    gnani_tts_ws_url: str = "wss://api.vachana.ai/api/v1/tts"
    gnani_tts_rest_url: str = "https://api.vachana.ai/api/v1/tts/inference"

    # Prisma v3.0 is required for indic-auto and keyterms.
    stt_model: str = "prisma-v3.0"
    stt_language_code: str = "indic-auto"
    stt_sample_rate: int = 16000

    # Timbre v3.0 is required for inline audio tags; v2.5 strips them silently.
    tts_model: str = "timbre-v3.0"
    tts_language: str = "hi-en"

    # Evon is not on Gnani's API platform, so the LLM is always OpenAI-compatible.
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_analysis_model: str = ""

    # Gnani publishes no credit conversion, so ceilings are in raw units.
    budget_enabled: bool = True
    budget_max_tts_chars: int = 200_000
    budget_max_stt_seconds: int = 7_200
    budget_max_llm_tokens: int = 2_000_000

    @property
    def sqlalchemy_url(self) -> str:
        """The database URL as a plain string, which SQLAlchemy wants."""
        return str(self.database_url)

    @property
    def analysis_model(self) -> str:
        """Model for analysis calls, falling back to the dialogue model."""
        return self.llm_analysis_model or self.llm_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, parsed once."""
    return Settings()
