"""The call lifecycle as an explicit transition table.

Illegal transitions raise rather than being silently absorbed, because a call that is in a
state the orchestrator did not expect will otherwise fail later and further away.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class CallState(StrEnum):
    RINGING = "ringing"
    GUARDIAN = "guardian"
    HANDOFF_PENDING = "handoff_pending"
    HUMAN_CHECK = "human_check"
    PERSONA_ACTIVE = "persona_active"
    ROBOCALL_DROPPED = "robocall_dropped"
    ENDED = "ended"


#: Allowed successors for each state.
#:
#: RINGING can go straight to HUMAN_CHECK (we answer a call already flagged as a scam) or
#: into GUARDIAN (we are listening passively on the user's behalf). HANDOFF_PENDING is the
#: Guardian Mode trigger point: the user has asked us to take the call but we have not yet
#: established whether a human is on the line.
_TRANSITIONS: Final[Mapping[CallState, frozenset[CallState]]] = MappingProxyType(
    {
        CallState.RINGING: frozenset({CallState.GUARDIAN, CallState.HUMAN_CHECK, CallState.ENDED}),
        CallState.GUARDIAN: frozenset({CallState.HANDOFF_PENDING, CallState.ENDED}),
        CallState.HANDOFF_PENDING: frozenset({CallState.HUMAN_CHECK, CallState.ENDED}),
        CallState.HUMAN_CHECK: frozenset(
            {CallState.PERSONA_ACTIVE, CallState.ROBOCALL_DROPPED, CallState.ENDED}
        ),
        CallState.PERSONA_ACTIVE: frozenset({CallState.ENDED}),
        CallState.ROBOCALL_DROPPED: frozenset({CallState.ENDED}),
        CallState.ENDED: frozenset(),
    }
)

TERMINAL_STATES: Final[frozenset[CallState]] = frozenset(
    {CallState.ENDED, CallState.ROBOCALL_DROPPED}
)


class IllegalTransition(Exception):
    """Raised when a caller asks for a transition the table does not permit."""

    def __init__(self, current: CallState, requested: CallState) -> None:
        self.current = current
        self.requested = requested
        allowed = ", ".join(sorted(s.value for s in allowed_transitions(current))) or "none"
        super().__init__(
            f"cannot move from {current.value} to {requested.value}; allowed: {allowed}"
        )


def allowed_transitions(state: CallState) -> frozenset[CallState]:
    """Return the states reachable in one step from `state`."""
    return _TRANSITIONS[state]


def can_transition(current: CallState, requested: CallState) -> bool:
    """Return whether `requested` is reachable in one step from `current`."""
    return requested in _TRANSITIONS[current]


def transition(current: CallState, requested: CallState) -> CallState:
    """Return `requested` if the move is legal, else raise `IllegalTransition`."""
    if not can_transition(current, requested):
        raise IllegalTransition(current, requested)
    return requested
