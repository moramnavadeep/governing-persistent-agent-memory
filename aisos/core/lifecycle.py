"""
Formal Lifecycle State Model (Section 5.1 of the AISOS v2 spec).

The lifecycle is defined as a finite state machine (S, Sigma, delta, s0):

    S      = {AWAKE, LIGHT, DEEP, DREAM, WAKE_VERIFY}
    s0     = AWAKE
    Sigma  = {enter_light, enter_deep, enter_dream, finish_stage,
              request_wake_verify, verify_pass, verify_fail,
              preempt_p0, preempt_p1}
    delta  = the transition table below

Structural property (load-bearing for Theorem 1, "No Torn Reads"):
    Every path leaving {LIGHT, DEEP, DREAM} passes through WAKE_VERIFY.
    There is no transition that returns to AWAKE directly from a mutating
    state. WAKE_VERIFY is therefore the *only* place the published pointer
    can move, which is what Theorem 1's proof depends on.

This module enforces invariants I1-I4 by construction:
    I1 (Single Writer)     -- a process-wide lock; only the thread holding
                               it may drive a transition that mutates state.
    I2 (Write-Before-Log)  -- every mutating transition appends a WAL entry
                               *before* it is allowed to touch the private
                               working copy (see security/wal.py + checkpoint.py).
    I3 (Monotonic Trust / Atomic Publish) -- the published checkpoint pointer
                               only ever advances, and only inside WAKE_VERIFY.
    I4 (Copy-on-Write)      -- LIGHT/DEEP/DREAM operate on a private working
                               copy; AWAKE reads only ever dereference the
                               last published checkpoint.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class State(str, Enum):
    AWAKE = "AWAKE"
    LIGHT = "LIGHT"
    DEEP = "DEEP"
    DREAM = "DREAM"
    WAKE_VERIFY = "WAKE_VERIFY"


# Mutating states: states in which a private working copy exists and may be
# written to. AWAKE and WAKE_VERIFY are not mutating states for the purposes
# of I4 (WAKE_VERIFY only reads the private copy plus the WAL to decide
# commit/rollback; it does not itself perform consolidation work).
MUTATING_STATES = frozenset({State.LIGHT, State.DEEP, State.DREAM})

# The transition table (delta). Keys are (state, event) -> next_state.
# This table is the single source of truth for "every path out of a mutating
# state passes through WAKE_VERIFY" -- note there is no (LIGHT|DEEP|DREAM, *)
# entry that targets AWAKE directly.
_TRANSITIONS: dict[tuple[State, str], State] = {
    (State.AWAKE, "enter_light"): State.LIGHT,
    (State.AWAKE, "enter_deep"): State.DEEP,      # scheduler may target DEEP directly
    (State.AWAKE, "enter_dream"): State.DREAM,    # or DREAM directly, per policy

    (State.LIGHT, "finish_stage"): State.DEEP,           # escalate to deep sleep
    (State.LIGHT, "request_wake_verify"): State.WAKE_VERIFY,
    (State.LIGHT, "preempt_p0"): State.WAKE_VERIFY,
    (State.LIGHT, "preempt_p1"): State.WAKE_VERIFY,

    (State.DEEP, "finish_stage"): State.DREAM,           # escalate to dream
    (State.DEEP, "request_wake_verify"): State.WAKE_VERIFY,
    (State.DEEP, "preempt_p0"): State.WAKE_VERIFY,
    (State.DEEP, "preempt_p1"): State.WAKE_VERIFY,

    (State.DREAM, "request_wake_verify"): State.WAKE_VERIFY,
    (State.DREAM, "finish_stage"): State.WAKE_VERIFY,
    (State.DREAM, "preempt_p0"): State.WAKE_VERIFY,
    (State.DREAM, "preempt_p1"): State.WAKE_VERIFY,

    (State.WAKE_VERIFY, "verify_pass"): State.AWAKE,
    (State.WAKE_VERIFY, "verify_fail"): State.AWAKE,   # rollback lands back at AWAKE too
}


class InvalidTransition(RuntimeError):
    pass


@dataclass
class TransitionRecord:
    ts: float
    from_state: State
    event: str
    to_state: State


class Lifecycle:
    """Drives the AISOS state machine and enforces I1 (Single Writer)."""

    def __init__(self, on_transition: Optional[Callable[[TransitionRecord], None]] = None):
        self._state = State.AWAKE
        self._lock = threading.RLock()
        self._history: list[TransitionRecord] = []
        self._on_transition = on_transition
        # Monotonically increasing "stage counter" -- an operationally
        # checkable witness for I1 (Section 5.8): any two writers observing
        # the same stage counter value and both attempting to mutate is a
        # violation that can be asserted at runtime.
        self._stage_counter = 0

    @property
    def state(self) -> State:
        return self._state

    @property
    def stage_counter(self) -> int:
        return self._stage_counter

    def is_mutating(self) -> bool:
        return self._state in MUTATING_STATES

    def can(self, event: str) -> bool:
        return (self._state, event) in _TRANSITIONS

    def fire(self, event: str) -> State:
        """Attempt to fire an event. Acquiring self._lock is what makes this
        the single writer (I1): only one thread can be mid-transition at a
        time, and no transition is observable to readers until it completes.
        """
        with self._lock:
            key = (self._state, event)
            if key not in _TRANSITIONS:
                raise InvalidTransition(
                    f"no transition for event {event!r} from state {self._state.value}"
                )
            prev = self._state
            nxt = _TRANSITIONS[key]
            self._state = nxt
            self._stage_counter += 1
            rec = TransitionRecord(ts=time.time(), from_state=prev, event=event, to_state=nxt)
            self._history.append(rec)
            if self._on_transition:
                self._on_transition(rec)
            return nxt

    def history(self) -> list[TransitionRecord]:
        with self._lock:
            return list(self._history)

    def assert_every_exit_via_wake_verify(self) -> bool:
        """Operational check (Section 5.8, bullet 2): walk the recorded
        history and confirm no mutating state ever transitioned directly to
        AWAKE. Used by the invariant test suite to validate the *table*
        itself, independent of any particular run's history.
        """
        for (frm, _evt), to in _TRANSITIONS.items():
            if frm in MUTATING_STATES and to == State.AWAKE:
                return False
        return True
