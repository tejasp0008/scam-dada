"""Provider identifiers.

A module of its own so `app.infra.settings` can name a provider without importing the
factory, and the factory can import settings-free configuration without a cycle.
"""

from __future__ import annotations

from enum import StrEnum


class ProviderName(StrEnum):
    """Which implementation to build. The only place a vendor is chosen."""

    GNANI = "gnani"
    OPENAI = "openai"
    MOCK = "mock"
