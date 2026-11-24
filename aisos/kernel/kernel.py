"""Cross-agent task scheduler ("Kernel") for the AISOS SDK.

This is the genuinely new piece of "broad scope" this SDK adds on top of the
existing research codebase. It answers a question nothing in aisos/core/,
aisos/fleet/, or aisos/multiagent/ answers: given N *independent*
AISOSAgent-backed sessions, which one gets the next turn?

What already exists, and why none of it is this:
  - aisos.core.scheduler.Scheduler decides WHEN one agent should pause
    serving reads and run its own maintenance cycle (sleep-pressure based).
  - aisos.core.interrupt.InterruptController preempts ONE agent's
    in-progress cycle for a P0/P1 event on that SAME agent.
  - aisos.fleet.coordinator.FleetCoordinator is a quorum limiting how many
    REPLICAS OF THE SAME AGENT may sleep at once (horizontal availability),
    not a dispatcher across different agents.
  - aisos.multiagent.gateway.MultiAgentGovernanceGateway funnels many
    producers' WRITE CANDIDATES into ONE shared agent's remember() through
    identity/authorization checks -- a write-governance gate, not a
    compute/turn scheduler.
None of the above decides turn order among distinct agents or context-
switches between them. The Kernel below does exactly that, and composes
freely with all four of the above -- an AgentSession's `agent` may itself be
fronted by a MultiAgentGovernanceGateway, and each session's own AISOSAgent
may itself participate in a FleetCoordinator-managed replica pool; the
Kernel is agnostic to both.

Because AISOSAgent holds no shared/global mutable state (confirmed by
reading aisos/core/adapter.py: every field is set in __init__, and the only
lock is a per-instance threading.RLock), advancing one session's turn never
touches another session's Lifecycle/staging state -- there is no explicit
save/restore "context switch" step to get wrong, which keeps this v0.1
Kernel's correctness argument simple.

Turn-scheduling policy is deliberately simple in v0.1 (round-robin or
priority-max) -- this is new, unvalidated engineering territory, and a
correct simple policy is worth more here than an elaborate one.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol

from aisos.core.adapter import AISOSAgent, CycleReport
from aisos.kernel.errors import DuplicateSessionError, InvalidWorkError, UnknownSessionError
from aisos.observability.emitter import DEFAULT_EMITTER, EventEmitter
from aisos.observability.events import EventType


class WorkState(str, Enum):
    """Lifecycle of one submitted work item. Deliberately minimal --
    a RETRYABLE state and automatic retry are ROADMAP, not v0.1, because
    we will not ship retry semantics we have not tested."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PendingWrite:
    subject: str
    value: str
    source: str = "user"
    confidence: float = 0.7
    credential: Optional[str] = None


@dataclass
class FailedWork:
    """A work item whose execution raised. Retained (never discarded) so a
    caller can inspect, log, or resubmit it -- Phase A found that the
    original implementation dropped these silently."""

    item: PendingWrite
    error: str
    error_type: str
    failed_at: float = field(default_factory=time.time)


@dataclass
class TurnResult:
    """What one call to Kernel.run_turn() did. run_turn() returns this
    instead of raising, so a caller's scheduling loop does not need a
    try/except around every turn."""

    session: str
    state: WorkState
    item: Optional[PendingWrite] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.state is WorkState.COMPLETED


@dataclass
class AgentSession:
    name: str
    agent: AISOSAgent
    priority: int = 0
    model: Optional[Any] = None  # an aisos.agent.model_backend.ModelBackend, kept generic here to avoid a hard import
    tools: Optional[Any] = None  # an aisos.tools.registry.ToolRegistry
    turns_taken: int = 0
    last_active_ts: float = field(default_factory=time.time)
    paused: bool = False
    failed: bool = False
    _pending: list = field(default_factory=list)
    _failed_items: list = field(default_factory=list)

    @property
    def pending_work(self) -> int:
        return len(self._pending)

    @property
    def failed_work(self) -> list:
        """Work items that raised during execution, in failure order."""
        return list(self._failed_items)

    @property
    def runnable(self) -> bool:
        """A session is runnable only if it has work, is not paused, and is
        not in a failed state. Excluding failed sessions here is what stops
        one repeatedly-failing agent from consuming every turn and starving
        healthy ones."""
        return bool(self._pending) and not self.paused and not self.failed


class SchedulingPolicy(Protocol):
    def next_session(self, sessions: list[AgentSession]) -> Optional[AgentSession]:
        ...


class RoundRobinPolicy:
    """Fair rotation among sessions that currently have pending work.
    Sessions with no pending work are skipped without consuming a turn in
    the rotation, so a busy session is not penalized by an idle one's slot.

    FAIRNESS DEFECT FIXED IN PHASE G
        The first implementation held an integer cursor INTO THE FILTERED
        `runnable` list. That list changes length as sessions run out of
        work, become failed, or are paused, so the same cursor value refers
        to a different session from one call to the next: after a session
        drops out, the rotation could skip a session entirely or serve one
        twice in a row. The bug is invisible with a fixed set of always-busy
        sessions, which is why the original tests did not catch it.

        The cursor is now the NAME of the last session served. The next turn
        goes to the first runnable session after that name in a stable sort
        order, wrapping around. That is well defined no matter how the
        runnable set changes between calls, and it makes scheduling
        reproducible: the same sequence of submissions always produces the
        same sequence of turns.
    """

    def __init__(self) -> None:
        self._last: Optional[str] = None

    def next_session(self, sessions: list[AgentSession]) -> Optional[AgentSession]:
        runnable = sorted((s for s in sessions if s.runnable), key=lambda s: s.name)
        if not runnable:
            return None
        chosen = runnable[0]
        if self._last is not None:
            for session in runnable:
                if session.name > self._last:
                    chosen = session
                    break
        self._last = chosen.name
        return chosen


class PriorityQueuePolicy:
    """The highest-`priority` runnable session goes next; ties broken by
    least-recently-active (so two equal-priority sessions still alternate
    fairly rather than one starving the other)."""

    def next_session(self, sessions: list[AgentSession]) -> Optional[AgentSession]:
        runnable = [s for s in sessions if s.runnable]
        if not runnable:
            return None
        # Sorted by name first so that sessions with identical priority AND
        # identical last_active_ts (entirely possible -- time.time() has
        # coarse resolution on Windows) resolve deterministically instead of
        # depending on dict insertion order.
        return max(sorted(runnable, key=lambda s: s.name),
                   key=lambda s: (s.priority, -s.last_active_ts))


class Kernel:
    """Registers named AgentSessions and advances them one turn at a time
    under a pluggable SchedulingPolicy. The Kernel owns turn ORDER only; it
    never second-guesses a session's own single-agent maintenance-cycle
    decision (that stays entirely with that session's own
    AISOSAgent.request_maintenance())."""

    def __init__(
        self,
        policy: Optional[SchedulingPolicy] = None,
        emitter: Optional[EventEmitter] = None,
    ) -> None:
        self.policy = policy or RoundRobinPolicy()
        self._sessions: dict[str, AgentSession] = {}
        self._lock = threading.RLock()
        # Defaults to the process-wide emitter, which has no sinks attached
        # unless the application added one -- so an unobserved Kernel pays
        # nothing for these calls (see EventEmitter.emit's fast path).
        self.emitter = emitter if emitter is not None else DEFAULT_EMITTER

    def register_agent(
        self,
        name: str,
        agent: AISOSAgent,
        priority: int = 0,
        model: Optional[Any] = None,
        tools: Optional[Any] = None,
    ) -> AgentSession:
        with self._lock:
            if name in self._sessions:
                raise DuplicateSessionError(name)
            session = AgentSession(name=name, agent=agent, priority=priority, model=model, tools=tools)
            self._sessions[name] = session
        self.emitter.emit(
            EventType.AGENT_REGISTERED, session=name, priority=priority,
            lifecycle_state=agent.lifecycle.state.value,
        )
        return session

    def unregister_agent(self, name: str) -> Optional[AgentSession]:
        """Removes a session and RETURNS it, so the caller can see what work
        was dropped rather than discovering later that it vanished.

        Deliberately idempotent (returns None for an unknown name instead of
        raising): cleanup paths run in `finally` blocks and during shutdown,
        where making removal-of-something-already-gone an error causes more
        bugs than it catches. Every other accessor raises UnknownSessionError.
        """
        with self._lock:
            removed = self._sessions.pop(name, None)
        if removed is not None:
            self.emitter.emit(
                EventType.AGENT_UNREGISTERED, session=name,
                turns_taken=removed.turns_taken,
                pending_work=removed.pending_work,
                failed_work=len(removed._failed_items),
            )
        return removed

    def submit(
        self,
        name: str,
        subject: str,
        value: str,
        source: str = "user",
        confidence: float = 0.7,
        credential: Optional[str] = None,
    ) -> None:
        """Enqueues a write for session `name`. The actual
        agent.remember() call happens later, on that session's own turn
        inside run_turn() -- this is what makes the context switch
        explicit rather than implicit: submitting work to a session never
        itself advances the scheduler."""
        if not subject:
            raise InvalidWorkError("subject must be a non-empty string")
        if value is None:
            raise InvalidWorkError("value must not be None")
        with self._lock:
            session = self._session(name)
            session._pending.append(
                PendingWrite(subject=subject, value=value, source=source, confidence=confidence, credential=credential)
            )

    def run_turn(self) -> Optional[TurnResult]:
        """Advances exactly one session, chosen by self.policy, by
        processing one of its pending writes.

        Returns None if no session is currently runnable, otherwise a
        TurnResult describing what happened. This method does NOT raise for
        an agent-side failure: if the underlying remember() call raises, the
        work item is moved to that session's failed list (never discarded),
        the session is marked failed so it stops consuming turns, and the
        error is returned in the TurnResult.
        """
        with self._lock:
            session = self.policy.next_session(list(self._sessions.values()))
            if session is None:
                return None
            write = session._pending.pop(0)
        self.emitter.emit(
            EventType.TURN_STARTED, session=session.name, subject=write.subject,
            lifecycle_state=session.agent.lifecycle.state.value,
        )
        # The remember() call itself happens OUTSIDE self._lock -- it only
        # touches this session's own AISOSAgent, which has its own
        # internal lock, so holding the Kernel's lock across it would
        # only serialize unrelated sessions' turns for no correctness
        # benefit.
        try:
            session.agent.remember(
                subject=write.subject,
                value=write.value,
                source=write.source,
                confidence=write.confidence,
                credential=write.credential,
            )
        except Exception as exc:  # noqa: BLE001 -- any agent-side failure is captured, never swallowed
            with self._lock:
                session._failed_items.append(
                    FailedWork(item=write, error=str(exc), error_type=type(exc).__name__)
                )
                session.failed = True
                session.last_active_ts = time.time()
            self.emitter.emit(
                EventType.TURN_FAILED, session=session.name, subject=write.subject,
                error_type=type(exc).__name__, error=str(exc),
            )
            return TurnResult(
                session=session.name,
                state=WorkState.FAILED,
                item=write,
                error=f"{type(exc).__name__}: {exc}",
            )

        with self._lock:
            session.turns_taken += 1
            session.last_active_ts = time.time()
        self.emitter.emit(
            EventType.TURN_COMPLETED, session=session.name, subject=write.subject,
            turns_taken=session.turns_taken,
        )
        self.emitter.emit(
            EventType.MEMORY_WRITTEN, session=session.name, subject=write.subject,
            source=write.source, confidence=write.confidence,
        )
        return TurnResult(session=session.name, state=WorkState.COMPLETED, item=write)

    def clear_failure(self, name: str, requeue: bool = False) -> int:
        """Clears a session's failed state so it becomes runnable again.

        requeue=True puts the failed work items back at the front of the
        pending queue (caller-driven retry -- the Kernel itself never
        retries automatically in v0.1). Returns how many items were
        requeued or discarded from the failed list.
        """
        with self._lock:
            session = self._session(name)
            failed_items = list(session._failed_items)
            session._failed_items.clear()
            session.failed = False
            if requeue:
                session._pending[:0] = [f.item for f in failed_items]
            return len(failed_items)

    def run_maintenance_if_due(self, name: str, run_dream: bool = True) -> Optional[CycleReport]:
        """Pure delegation to that session's own AISOSAgent -- the Kernel
        never overrides a single agent's own sleep-pressure decision."""
        with self._lock:
            session = self._session(name)
        return session.agent.request_maintenance(run_dream=run_dream)

    def pause(self, name: str) -> None:
        with self._lock:
            self._session(name).paused = True

    def resume(self, name: str) -> None:
        with self._lock:
            self._session(name).paused = False

    def _session(self, name: str) -> AgentSession:
        """Single lookup point, so every accessor reports an unknown session
        the same way. Callers hold self._lock."""
        try:
            return self._sessions[name]
        except KeyError:
            raise UnknownSessionError(name, self._sessions) from None

    def get_session(self, name: str) -> AgentSession:
        with self._lock:
            return self._session(name)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._sessions

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def status(self) -> dict:
        with self._lock:
            return {
                name: {
                    "priority": s.priority,
                    "pending_work": s.pending_work,
                    "turns_taken": s.turns_taken,
                    "paused": s.paused,
                    "failed": s.failed,
                    "failed_work": len(s._failed_items),
                    "runnable": s.runnable,
                    "agent_status": s.agent.status(),
                }
                for name, s in self._sessions.items()
            }
