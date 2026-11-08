"""The per-minute token cap, as a test rather than as a thing someone remembers.

Every model on the account is capped at 8000 tokens a minute and the cap is **per model**.
A live call hit it: the dialogue request had grown to roughly 1800 input tokens, which is
only about four and a half turns a minute, and the analyst was sharing a model with the
dialogue fallback, so when one ran out both did.

So the sizes are pinned here. The numbers are not arbitrary - each one is a turn rate - and
a change that pushes a request over its ceiling is a change that shortens how fast a call
can go before it starts stalling. That belongs in a failing test, not in a recording.

`cl100k_base` is not the tokeniser either model family actually uses. It lands within a few
percent on Latin-script Hinglish, which is close enough to hold a ceiling and not close
enough to quote; `scripts/token_budget.py` prints the vendor's own count under `--live`.
"""

from __future__ import annotations

from dataclasses import fields

import tiktoken

from app.core.models import PersonaId, Speaker, Turn
from app.core.personas import RELAY, REPLY_STYLE_NOTE
from app.core.turn import TurnBudget, recent_turns
from app.services.analysis import ANALYSIS_WINDOW_TURNS, analysis_messages
from app.services.orchestrator import CallConfig
from app.services.prompts import dialogue_messages

#: `x-ratelimit-limit-tokens`, read off every model on the account on 2026-11-08.
TOKENS_PER_MINUTE = 8000

#: What the chat schema costs around each message. OpenAI's documented figure.
PER_MESSAGE_OVERHEAD = 4

#: Ceiling for one dialogue request. 1600 tokens in is five caller turns a minute with the
#: completion on top, which is what a fast live call actually runs at: the persona speaks for
#: six or seven seconds, the caller for four, and the pipeline adds a little over one. Above
#: this the model runs out mid-call and the fallback carries turns it should only be covering
#: for (DECISIONS 0034).
MAX_DIALOGUE_INPUT_TOKENS = 1600

#: Ceiling for one analysis request. It runs once every four caller turns, so this is a small
#: share of its own model's minute - which is the point of giving it one.
MAX_ANALYSIS_INPUT_TOKENS = 1300

_encoding = tiktoken.get_encoding("cl100k_base")


def _tokens(text: str) -> int:
    return len(_encoding.encode(text))


def _count(messages: object) -> int:
    assert isinstance(messages, list)
    return sum(_tokens(m.content) + PER_MESSAGE_OVERHEAD for m in messages)


def _default(cls: type, name: str) -> int:
    """A slots dataclass hides its defaults behind descriptors on the class."""
    field = next(f for f in fields(cls) if f.name == name)  # type: ignore[arg-type]
    assert isinstance(field.default, int)
    return field.default


def _long_call(exchanges: int = 20) -> list[Turn]:
    """A call long enough that the history window is full and has stopped growing.

    The turns are longer than real ones on purpose: the window is bounded by characters as
    well as by count, so filling it with verbose turns is what proves the character bound is
    doing the work.
    """
    turns: list[Turn] = []
    for index in range(exchanges):
        turns.append(
            Turn(
                seq=len(turns),
                speaker=Speaker.CALLER,
                text=(
                    f"Sir aapke naam par case number fake CYB triple zero {index} register ho "
                    "gaya hai, abhi verify karna padega warna warrant nikal jayega."
                ),
                started_at_ms=0,
            )
        )
        turns.append(
            Turn(
                seq=len(turns),
                speaker=Speaker.PERSONA,
                text=(
                    "Haan beta, main likh raha hoon, zara dheere bolo. Mera chashma kahan "
                    "gaya, ek minute ruko main dhoondh leta hoon."
                ),
                persona=PersonaId.DADA,
                started_at_ms=0,
            )
        )
    return turns


class TestDialogueRequests:
    def test_a_steady_dialogue_request_stays_inside_its_share_of_the_minute(self) -> None:
        turns = _long_call()
        window = recent_turns(turns, TurnBudget())
        for profile in RELAY:
            total = _count(dialogue_messages(profile, window, [REPLY_STYLE_NOTE]))
            assert total <= MAX_DIALOGUE_INPUT_TOKENS, (
                f"{profile.prompt_file} now costs {total} input tokens a turn, over the "
                f"{MAX_DIALOGUE_INPUT_TOKENS} ceiling. At that size a call starts stalling "
                "sooner; trim the prompt or the window rather than raising this."
            )

    def test_the_request_stops_growing_once_the_window_is_full(self) -> None:
        """The ceiling only means anything if a long call costs no more than a short one."""
        budget = TurnBudget()
        short = _count(
            dialogue_messages(RELAY[0], recent_turns(_long_call(4), budget), [REPLY_STYLE_NOTE])
        )
        long = _count(
            dialogue_messages(RELAY[0], recent_turns(_long_call(40), budget), [REPLY_STYLE_NOTE])
        )
        assert long == short

    def test_the_window_is_bounded_by_characters_and_not_only_by_turns(self) -> None:
        budget = TurnBudget()
        window = recent_turns(_long_call(), budget)
        assert len(window) <= budget.max_turns
        assert sum(len(turn.text) for turn in window) <= budget.max_chars + len(window[0].text)


class TestAnalysisRequests:
    def test_an_analysis_request_stays_inside_its_share_of_the_minute(self) -> None:
        total = _count(analysis_messages(_long_call()[-ANALYSIS_WINDOW_TURNS:]))
        assert total <= MAX_ANALYSIS_INPUT_TOKENS, (
            f"the analysis request now costs {total} input tokens, over the "
            f"{MAX_ANALYSIS_INPUT_TOKENS} ceiling"
        )

    def test_analysis_is_rare_enough_to_be_a_minority_of_its_own_model_s_minute(self) -> None:
        """At five caller turns a minute it must be well clear of its cap on its own."""
        every = _default(CallConfig, "analysis_every_n_turns")
        per_minute = (5 / every) * MAX_ANALYSIS_INPUT_TOKENS
        assert per_minute < TOKENS_PER_MINUTE / 2


class TestTheWholeCall:
    def test_dialogue_and_analysis_do_not_share_a_model(self) -> None:
        """The failure this whole budget exists because of: one bucket, two consumers.

        When the analyst and the dialogue fallback were the same model, a throttled dialogue
        model put every turn onto the bucket the analyst was already spending.
        """
        from app.infra.settings import Settings

        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            llm_model="qwen/qwen3.8-27b",
            llm_analysis_model="openai/gpt-oss-20b",
            llm_fallback_model="openai/gpt-oss-120b",
        )
        models = {settings.llm_model, settings.analysis_model, settings.llm_fallback_model}
        assert len(models) == 3

    def test_the_analyst_is_never_given_the_dialogue_model_s_reasoning_effort(self) -> None:
        """`none` is an HTTP 400 on the gpt-oss family, which made analysis fail every time."""
        from app.infra.settings import Settings

        settings = Settings(_env_file=None, llm_reasoning_effort="none")  # type: ignore[call-arg]
        assert settings.llm_analysis_reasoning_effort != "none"
        assert settings.llm_analysis_reasoning_effort
