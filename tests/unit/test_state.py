"""The transition table, exhaustively."""

from __future__ import annotations

import itertools

import pytest

from app.core.state import (
    TERMINAL_STATES,
    CallState,
    IllegalTransition,
    allowed_transitions,
    can_transition,
    transition,
)

LEGAL_EDGES = {
    (CallState.RINGING, CallState.GUARDIAN),
    (CallState.RINGING, CallState.HUMAN_CHECK),
    (CallState.RINGING, CallState.ENDED),
    (CallState.GUARDIAN, CallState.HANDOFF_PENDING),
    (CallState.GUARDIAN, CallState.ENDED),
    (CallState.HANDOFF_PENDING, CallState.HUMAN_CHECK),
    (CallState.HANDOFF_PENDING, CallState.ENDED),
    (CallState.HUMAN_CHECK, CallState.PERSONA_ACTIVE),
    (CallState.HUMAN_CHECK, CallState.ROBOCALL_DROPPED),
    (CallState.HUMAN_CHECK, CallState.ENDED),
    (CallState.PERSONA_ACTIVE, CallState.ENDED),
    (CallState.ROBOCALL_DROPPED, CallState.ENDED),
}


@pytest.mark.parametrize(("current", "requested"), sorted(LEGAL_EDGES))
def test_legal_edges_are_permitted(current: CallState, requested: CallState) -> None:
    assert transition(current, requested) is requested


@pytest.mark.parametrize(
    ("current", "requested"),
    sorted(set(itertools.product(CallState, CallState)) - LEGAL_EDGES),
)
def test_every_other_edge_raises(current: CallState, requested: CallState) -> None:
    """Including self-transitions: a state machine that silently no-ops hides bugs."""
    assert not can_transition(current, requested)
    with pytest.raises(IllegalTransition):
        transition(current, requested)


def test_illegal_transition_message_lists_the_alternatives() -> None:
    with pytest.raises(IllegalTransition) as excinfo:
        transition(CallState.RINGING, CallState.PERSONA_ACTIVE)
    message = str(excinfo.value)
    assert "ringing" in message
    assert "human_check" in message


def test_ended_is_a_sink() -> None:
    assert allowed_transitions(CallState.ENDED) == frozenset()


def test_robocall_dropped_can_only_be_closed_out() -> None:
    """A dropped robocall must still reach ENDED so the call row gets finalised."""
    assert allowed_transitions(CallState.ROBOCALL_DROPPED) == frozenset({CallState.ENDED})


def test_persona_active_is_not_reachable_without_a_human_check() -> None:
    """The human-vs-robocall gate cannot be skipped, by construction."""
    for state in CallState:
        if state is CallState.HUMAN_CHECK:
            continue
        assert CallState.PERSONA_ACTIVE not in allowed_transitions(state)


def test_terminal_states_are_consistent_with_the_table() -> None:
    assert {CallState.ENDED, CallState.ROBOCALL_DROPPED} == TERMINAL_STATES
    for state in TERMINAL_STATES:
        assert allowed_transitions(state) <= {CallState.ENDED}
