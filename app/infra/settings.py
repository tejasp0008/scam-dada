"""All configuration, from the environment, validated once at startup.

The only global in the project. Everything else is injected.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, PostgresDsn, RedisDsn, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.cost import CostModel
from app.providers.names import ProviderName


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
    # transliterate is the only format that returned Latin-script Hinglish intact.
    stt_format: str = "transliterate"
    stt_sample_rate: int = 16000

    # Timbre v3.0 is required for inline audio tags; v2.5 strips them silently.
    tts_model: str = "timbre-v3.0"
    tts_language: str = "hi-en"

    # ---- Personas ----
    # Voices are chosen by ear from `make audition-male`, not by a published age attribute,
    # because Gnani does not publish one. Empty until a human has listened; the ageing
    # preset that goes with the elder voice lives in app/core/personas.py, because it is
    # part of the character rather than part of the deployment.
    persona_elder_voice: str = ""
    persona_grandson_voice: str = ""

    @model_validator(mode="after")
    def _personas_must_sound_different(self) -> Settings:
        """Refuse to start with one voice for both personas.

        The family relay works because the scammer hears a different person take the phone.
        Two personas on one voice is not a degraded demo, it is a broken one, and it is an
        easy mistake to make by copying a line in `.env`. Better to fail at startup than
        halfway through a recorded call.
        """
        elder = self.persona_elder_voice.strip().casefold()
        grandson = self.persona_grandson_voice.strip().casefold()
        if elder and grandson and elder == grandson:
            raise ValueError(
                "PERSONA_ELDER_VOICE and PERSONA_GRANDSON_VOICE are both "
                f"{self.persona_elder_voice!r}; the relay needs two different voices"
            )
        return self

    # Evon is not on Gnani's API platform, so the LLM is always OpenAI-compatible.
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_analysis_model: str = ""
    # The dialogue model is a Groq preview model, which can be withdrawn or throttled
    # without notice, so a second model is always configured. Measured in DECISIONS 0010.
    llm_fallback_model: str = "openai/gpt-oss-120b"
    llm_fallback_reasoning_effort: str = "low"
    # If the primary has produced no token by now, switch. A scammer hears dead air.
    llm_first_token_timeout_s: float = 3.0
    llm_request_timeout_s: float = 30.0

    # ---- Budget ----
    budget_enabled: bool = True

    # Gnani programme credits, denominated in rupees (1 credit = ₹1). The hard stop sits
    # short of the balance so there is always enough left to re-record the demo.
    gnani_credit_total: float | None = 5_000.0
    gnani_credit_stop_at: float | None = 4_500.0

    # The LLM vendor bills separately and does not touch Gnani credits. Its total is
    # unknown, so the token ceiling below is what actually constrains it.
    llm_credit_total: float | None = None
    llm_credit_stop_at: float | None = None

    # Raw-unit ceilings: a second line of defence, and the only usable ceiling for a pool
    # whose rate is not yet known. None disables one.
    budget_max_tts_chars: int | None = 200_000
    budget_max_stt_seconds: int | None = 7_200
    budget_max_llm_tokens: int | None = 2_000_000

    # ---- Rates ----
    # Placeholders inferred from a six-request sample on the usage dashboard. Every charge
    # priced with these is flagged estimated=True until rates_confirmed is set.
    rates_confirmed: bool = False
    stt_rate_per_minute: float = 0.45
    stt_billing_increment_minutes: float = 0.05
    stt_minimum_billed_minutes: float = 0.05
    tts_rate_per_1k_chars: float = 1.58
    llm_rate_per_1m_tokens: float = 0.0

    # ---- Local storage ----
    var_dir: Path = Path("var")

    @property
    def tts_cache_dir(self) -> Path:
        """Where synthesised audio is cached by content hash."""
        return self.var_dir / "tts_cache"

    @property
    def sqlalchemy_url(self) -> str:
        """The database URL as a plain string, which SQLAlchemy wants."""
        return str(self.database_url)

    def cost_model(self) -> CostModel:
        """Build the cost model from the configured rates."""
        return CostModel(
            stt_rate_per_minute=self.stt_rate_per_minute,
            stt_billing_increment_minutes=self.stt_billing_increment_minutes,
            stt_minimum_billed_minutes=self.stt_minimum_billed_minutes,
            tts_rate_per_1k_chars=self.tts_rate_per_1k_chars,
            llm_rate_per_1m_tokens=self.llm_rate_per_1m_tokens,
            rates_confirmed=self.rates_confirmed,
        )

    @property
    def analysis_model(self) -> str:
        """Model for analysis calls, falling back to the dialogue model."""
        return self.llm_analysis_model or self.llm_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, parsed once."""
    return Settings()
