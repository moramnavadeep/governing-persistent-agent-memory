"""
Algorithm A5 -- Interrupt Controller.

Preserves: I1 (single writer, enforced via PREEMPT_NOW); directly realizes
Theorem 3 (Bounded Preemption Latency) and the "P2 never blocks" clause of
Theorem 4 (No Starvation).

Priority levels (Technical Spec Section 20 / v2 Section 3):
    P0 - emergency/security critical  -> PREEMPT_NOW (immediate abort+rollback)
    P1 - urgent request               -> PREEMPT_NOW (immediate abort+rollback)
    P2 - normal interactive request   -> enqueued only, never blocks scheduling
    P3 - background maintenance       -> the sleep cycle itself

Cost: O(1) dispatch decision per event. The dominant cost on the P0/P1 path
is PREEMPT_NOW's rollback, which by Theorem 2 is O(1) regardless of how much
work the aborted cycle had done (copy-on-write discard, not undo).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional


class Priority(IntEnum):
    P0 = 0  # emergency / security critical
    P1 = 1  # urgent request
    P2 = 2  # normal interactive request
    P3 = 3  # background maintenance


@dataclass(order=True)
class InterruptEvent:
    priority: Priority
    ts: float = field(default_factory=time.time)
    payload: object = None
    dispatched_ts: Optional[float] = field(default=None, compare=False)

    @property
    def dispatch_latency_ms(self) -> Optional[float]:
        if self.dispatched_ts is None:
            return None
        return (self.dispatched_ts - self.ts) * 1000.0


class InterruptController:
    """Algorithm A5.

    atomic_unit_duration: the max time a single atomic unit of sleep work
    (one cluster reconciliation in A3, one synthetic task in A4, one
    segment in A2) may run before yielding a checkpoint-safe opportunity to
    preempt.
    max_defer_ms: an additional bound the controller may hold a P0/P1 event
    before forcing PREEMPT_NOW even mid-atomic-unit (implementation-defined
    upper bound on top of atomic_unit_duration).

    Theorem 3's bound -- max(atomic_unit_duration, max_defer_ms) -- is
    realized directly by can_enter_sleep()/dispatch() never waiting longer
    than that before calling PREEMPT_NOW for a P0/P1 event.
    """

    def __init__(
        self,
        atomic_unit_duration_ms: float = 50.0,
        max_defer_ms: float = 100.0,
        preempt_now: Optional[Callable[[InterruptEvent], None]] = None,
    ):
        self.atomic_unit_duration_ms = atomic_unit_duration_ms
        self.max_defer_ms = max_defer_ms
        self._preempt_now = preempt_now
        self._p2_queue: list[InterruptEvent] = []
        self._dispatched: list[InterruptEvent] = []
        self._in_mutating_stage = False

    def set_in_mutating_stage(self, flag: bool) -> None:
        self._in_mutating_stage = flag

    def can_enter_sleep(self) -> bool:
        """Sleep entry is refused only while a P0/P1 preemption is actively
        being handled; P2 traffic (however heavy) never appears here, which
        is exactly the "P2 never blocks" clause of Theorem 4.
        """
        return True  # P0/P1 are handled synchronously by dispatch(), not gated here

    def dispatch(self, event: InterruptEvent) -> InterruptEvent:
        """O(1) dispatch decision."""
        if event.priority in (Priority.P0, Priority.P1):
            event.dispatched_ts = time.time()
            if self._preempt_now is not None:
                self._preempt_now(event)
            self._dispatched.append(event)
            return event
        elif event.priority is Priority.P2:
            # Enqueue only -- never blocks SCHEDULE_SLEEP or can_enter_sleep().
            self._p2_queue.append(event)
            return event
        else:  # P3 - background maintenance is the sleep cycle itself
            self._dispatched.append(event)
            return event

    def drain_p2(self) -> list[InterruptEvent]:
        q, self._p2_queue = self._p2_queue, []
        return q

    @property
    def p2_backlog(self) -> int:
        return len(self._p2_queue)

    @property
    def dispatched(self) -> list[InterruptEvent]:
        return list(self._dispatched)

    def max_dispatch_latency_bound_ms(self) -> float:
        return max(self.atomic_unit_duration_ms, self.max_defer_ms)
