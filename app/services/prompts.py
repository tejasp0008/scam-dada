"""Reading the persona prompts and filler phrases off disk.

Separate from `app.core.personas`, which names the files but must not open them: core stays
I/O-free so the relay logic is testable without a filesystem.

Everything is read once and cached. These files do not change while the process runs, and a
disk read on the dialogue path would be latency spent for nothing.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from app.core.personas import ETHICS_PROMPT_FILE, PersonaProfile
from app.core.tags import sanitize_tags

logger = logging.getLogger(__name__)

#: Prompts ship inside the package, so they are found the same way in a container, in a
#: test and in an editable install. Not a configurable path: a deployment that could point
#: the ethics constraints at a different file would make the digest test meaningless.
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=32)
def load_prompt_file(name: str) -> str:
    """Read one file from the prompt directory."""
    path = PROMPT_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=8)
def system_prompt(prompt_file: str) -> str:
    """The ethics constraints followed by the persona, as one system message.

    Constraints first. Models weight earlier instructions more heavily, and a persona
    instruction that conflicts with a constraint must lose.
    """
    return f"{load_prompt_file(ETHICS_PROMPT_FILE)}\n\n---\n\n{load_prompt_file(prompt_file)}"


def persona_system_prompt(profile: PersonaProfile) -> str:
    """The system message for one persona."""
    return system_prompt(profile.prompt_file)


@lru_cache(maxsize=8)
def _load_fillers(filler_file: str, allowed: frozenset[str]) -> tuple[str, ...]:
    phrases: list[str] = []
    dropped_total = 0
    for line in load_prompt_file(filler_file).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cleaned, dropped = sanitize_tags(stripped, allowed)
        dropped_total += len(dropped)
        if cleaned:
            phrases.append(cleaned)
    if not phrases:
        raise ValueError(f"{filler_file} contains no usable phrases")
    if dropped_total:
        # Not fatal, but it means a phrase was written with a tag the persona cannot use,
        # and the file is the place to fix it rather than the filter.
        logger.warning(
            "filler phrases contained disallowed tags",
            extra={"filler_file": filler_file, "dropped": dropped_total},
        )
    return tuple(phrases)


def persona_fillers(profile: PersonaProfile) -> tuple[str, ...]:
    """This persona's stall phrases, with any tag it may not use already stripped."""
    return _load_fillers(profile.filler_file, frozenset(profile.allowed_tags))
