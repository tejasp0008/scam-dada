"""Contract-test helpers.

Every test in this directory is marked `contract` and runs fully offline against payloads
recorded in `tests/fixtures/`. Nothing here may touch the network: the marker is excluded
from the default `pytest` run so a live call can never sneak into CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def fixture_text() -> object:
    """Read a recorded payload by filename."""

    def _read(name: str) -> str:
        return (FIXTURES / name).read_text()

    return _read
