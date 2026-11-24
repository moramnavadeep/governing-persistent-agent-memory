"""AgentRuntime: the SDK's operational facade over a research AISOSAgent.

WHAT THIS IS, AND WHAT IT IS NOT
    AISOSAgent is the research object: it owns the five-state lifecycle FSM,
    the sleep-pressure scheduler, the copy-on-write memory store, the
    promotion pipeline and WAKE_VERIFY. None of that is reimplemented here.

    AgentRuntime is the thing an application actually holds. It binds one
    AISOSAgent to durable storage and an event emitter, and gives the three
    operations an operator needs that the research object does not provide:
    come back up after a restart (wake), do maintenance and persist at the
    resulting consistency point (sleep), and stop cleanly (shutdown).

THE LIFECYCLE, HONESTLY
    The research FSM has exactly five states and this layer does not add to
    them:

        AWAKE -> LIGHT -> DEEP -> DREAM -> WAKE_VERIFY -> AWAKE

    sleep() is a thin, honest wrapper over AISOSAgent.request_maintenance():
    it runs one real cycle through those states and returns the research
    CycleReport unchanged. It is NOT a new lifecycle.

    wake() and shutdown() are NOT research primitives and this module does
    not pretend otherwise. There is no "wake" event in the FSM above -- a
    cycle returns to AWAKE by itself, synchronously, inside
    run_sleep_cycle(). wake() means "bring this agent_id back up from
    durable storage," which is a process-level operation the research core
    has no concept of because nothing in it was ever durable.

PERSISTENCE BOUNDARY
    A snapshot can only be taken between cycles, never during one
    (MemoryStore.snapshot() refuses while a private working copy is open).
    sleep(persist=True) therefore saves AFTER the cycle has committed or
    rolled back, at the point where the store is in a definite state. A
    failed cycle does not corrupt the stored snapshot: the previous snapshot
    stays in place until a new, complete one replaces it atomically.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

from aisos.core.adapter import AISOSAgent, CycleReport
from aisos.observability.emitter import DEFAULT_EMITTER, EventEmitter
from aisos.observability.events import EventType
from aisos.storage.base import StorageBackend
from aisos.storage.errors import SnapshotNotFoundError, StorageError
from aisos.storage.snapshot import create_snapshot, restore_snapshot

#: sleep() depth -> the research scheduler event it forces.
_DEPTHS = {
    "light": "enter_light",
    "deep": "enter_deep",
    "dream": "enter_dream",
}


class AgentRuntime:
    """One agent, its durable store, and its event stream.

        runtime = AgentRuntime(agent_id="assistant", storage=SQLiteStorage("agents.sqlite"))
        runtime.wake()                       # restores prior state if any
        runtime.remember("user.city", "Springfield")
        runtime.sleep(depth="deep")          # maintenance + persist
        runtime.shutdown()                   # final persist + close

    `storage=None` is supported and means "no durability" -- every
    persistence method then becomes an explicit no-op that returns False
    rather than silently pretending to have saved.
    """

    def __init__(
        self,
        agent: Optional[AISOSAgent] = None,
        agent_id: str = "default",
        storage: Optional[StorageBackend] = None,
        emitter: Optional[EventEmitter] = None,
        autosave: bool = True,
        metadata: Optional[dict] = None,
        **agent_kwargs: Any,
    ):
        self.agent_id = agent_id
        self.storage = storage
        self.emitter = emitter if emitter is not None else DEFAULT_EMITTER
        self.autosave = autosave
        self.metadata = dict(metadata or {})
        self._lock = threading.RLock()
        self._closed = False
        self.restored_from_snapshot = False
        self.last_saved_at: Optional[float] = None

        if agent is not None:
            if agent_kwargs:
                raise ValueError(
                    "pass either an existing `agent` or agent constructor kwargs, not both"
                )
            self.agent = agent
            # A caller-supplied agent already built its Lifecycle, and the
            # transition hook is a CONSTRUCTOR argument on the research
            # object. Rather than reach into Lifecycle's private state to
            # attach one, we report honestly that transitions are not
            # observed for this agent. Callers who want them construct the
            # agent through this runtime instead.
            self.lifecycle_observed = False
        else:
            self.agent = AISOSAgent(on_transition=self._on_transition, **agent_kwargs)
            self.lifecycle_observed = True

        self.emitter.emit(
            EventType.AGENT_CREATED,
            agent_id=self.agent_id,
            lifecycle_state=self.agent.lifecycle.state.value,
            storage=type(storage).__name__ if storage else None,
            lifecycle_observed=self.lifecycle_observed,
        )

    # -- observability hook ------------------------------------------------
    def _on_transition(self, record) -> None:
        """Forwarded to AISOSAgent's existing `on_transition` constructor
        argument -- a hook the research core already provided. This adds an
        event; it does not change any transition."""
        self.emitter.emit(
            EventType.LIFECYCLE_TRANSITION,
            agent_id=self.agent_id,
            lifecycle_state=getattr(record.to_state, "value", str(record.to_state)),
            from_state=getattr(record.from_state, "value", str(record.from_state)),
            event=record.event,
        )

    # -- recovery ----------------------------------------------------------
    def wake(self, strict: bool = False) -> bool:
        """Bring this agent up, restoring durable state if any exists.

        Returns True if a snapshot was found and restored, False if this is
        a cold start with nothing stored. A cold start is a normal outcome,
        not an error -- the first run of any agent has nothing to restore.

        `strict=True` turns "nothing stored" into a SnapshotNotFoundError,
        for callers who know a snapshot must exist and would rather fail
        than silently start with an empty memory.

        A corrupt or version-incompatible snapshot ALWAYS raises, in both
        modes: starting an agent on state we could not verify is worse than
        not starting at all.
        """
        if self.storage is None:
            if strict:
                raise StorageError("wake(strict=True) requires a storage backend")
            return False

        self.emitter.emit(EventType.RECOVERY_STARTED, agent_id=self.agent_id)
        try:
            snapshot = self.storage.load(self.agent_id)
        except SnapshotNotFoundError:
            self.emitter.emit(
                EventType.RECOVERY_COMPLETED, agent_id=self.agent_id,
                restored=False, reason="no snapshot stored (cold start)",
            )
            if strict:
                raise
            return False
        except StorageError as exc:
            self.emitter.emit(
                EventType.RECOVERY_FAILED, agent_id=self.agent_id,
                error_type=type(exc).__name__, error=str(exc),
            )
            raise

        with self._lock:
            restore_snapshot(self.agent, snapshot)
            self.restored_from_snapshot = True

        saved_state = snapshot.get("agent", {}).get("lifecycle_state")
        self.emitter.emit(
            EventType.SNAPSHOT_RESTORED, agent_id=self.agent_id,
            lifecycle_state=self.agent.lifecycle.state.value,
            saved_lifecycle_state=saved_state,
            live_memory=len(self.agent.store.read_live()),
            store_version=self.agent.store.version,
        )
        self.emitter.emit(
            EventType.RECOVERY_COMPLETED, agent_id=self.agent_id, restored=True,
            live_memory=len(self.agent.store.read_live()),
        )
        return True

    # -- normal operation ---------------------------------------------------
    def remember(self, subject: str, value: str, source: str = "user",
                 confidence: float = 0.7, credential: Optional[str] = None):
        """Delegates to AISOSAgent.remember() unchanged.

        Note what this does NOT do: it does not make the value live-readable.
        A write lands in staging as a candidate and must clear the Promotion
        Pipeline across maintenance cycles before a live read can see it.
        That is the research core's governance property and this facade does
        not shortcut it.
        """
        self._check_open()
        record = self.agent.remember(
            subject=subject, value=value, source=source,
            confidence=confidence, credential=credential,
        )
        self.emitter.emit(
            EventType.MEMORY_WRITTEN, agent_id=self.agent_id, subject=subject,
            source=source, confidence=confidence, record_id=record.id,
            lifecycle_state=self.agent.lifecycle.state.value,
        )
        return record

    def sleep(
        self,
        depth: Optional[str] = "deep",
        run_dream: bool = True,
        persist: Optional[bool] = None,
    ) -> Optional[CycleReport]:
        """Run one maintenance cycle, then (by default) persist.

        `depth=None` defers to the agent's own sleep-pressure scheduler,
        which may decide to stay awake and return None -- the research
        decision is respected, never overridden.

        `persist` defaults to self.autosave. Persisting happens strictly
        after the cycle finishes, because a snapshot cannot be taken while a
        private working copy is open.
        """
        self._check_open()
        if depth is not None and depth not in _DEPTHS:
            raise ValueError(f"depth must be one of {sorted(_DEPTHS)} or None, got {depth!r}")
        force = _DEPTHS[depth] if depth else None

        self.emitter.emit(
            EventType.SLEEP_STARTED, agent_id=self.agent_id, depth=depth,
            run_dream=run_dream, lifecycle_state=self.agent.lifecycle.state.value,
        )
        t0 = time.perf_counter()
        report = self.agent.request_maintenance(force=force, run_dream=run_dream)
        elapsed = time.perf_counter() - t0

        if report is None:
            # The scheduler decided to stay awake. Not a failure.
            self.emitter.emit(
                EventType.SLEEP_COMPLETED, agent_id=self.agent_id, ran=False,
                reason="scheduler decided to stay awake",
                lifecycle_state=self.agent.lifecycle.state.value,
            )
            return None

        event = EventType.SLEEP_ABORTED if report.aborted else EventType.SLEEP_COMPLETED
        self.emitter.emit(
            event, agent_id=self.agent_id, ran=True, depth=depth,
            stage_reached=report.stage_reached, aborted=report.aborted,
            duration_s=round(elapsed, 6),
            lifecycle_state=self.agent.lifecycle.state.value,
            store_version=self.agent.store.version,
        )

        should_persist = self.autosave if persist is None else persist
        if should_persist:
            self.save()
        return report

    def save(self, metadata: Optional[dict] = None) -> bool:
        """Persist current state. Returns False if no storage is configured.

        Raises if the agent has a cycle in flight -- persisting uncommitted
        maintenance work is refused rather than silently accepted.
        """
        self._check_open()
        if self.storage is None:
            return False
        combined = {**self.metadata, **(metadata or {})}
        with self._lock:
            snapshot = create_snapshot(self.agent, self.agent_id, metadata=combined)
            self.emitter.emit(
                EventType.SNAPSHOT_CREATED, agent_id=self.agent_id,
                store_version=snapshot["memory"]["version"],
                published=len(snapshot["memory"]["published"]),
                staging=len(snapshot["memory"]["staging"]),
            )
            self.storage.save(self.agent_id, snapshot)
            self.last_saved_at = time.time()
        self.emitter.emit(
            EventType.SNAPSHOT_SAVED, agent_id=self.agent_id,
            backend=type(self.storage).__name__,
        )
        return True

    def shutdown(self, persist: bool = True) -> None:
        """Persist (optionally) and release the storage backend.

        Idempotent: calling it twice is safe, and a runtime that failed to
        persist still closes its storage rather than leaking the handle.
        """
        with self._lock:
            if self._closed:
                return
            saved = False
            error = None
            if persist and self.storage is not None:
                try:
                    saved = self.save()
                except Exception as exc:  # noqa: BLE001 -- shutdown must still close storage
                    error = f"{type(exc).__name__}: {exc}"
            if self.storage is not None:
                self.storage.close()
            self._closed = True
        self.emitter.emit(
            EventType.SHUTDOWN, agent_id=self.agent_id, persisted=saved, error=error,
            lifecycle_state=self.agent.lifecycle.state.value,
        )
        if error is not None:
            raise StorageError(f"shutdown could not persist state: {error}")

    # -- introspection ------------------------------------------------------
    def status(self) -> dict:
        """Agent status plus runtime/storage facts.

        The nested "agent" block is AISOSAgent.status() verbatim, so nothing
        here reinterprets or relabels a research metric.
        """
        agent_status = self.agent.status()
        stored = None
        if self.storage is not None and hasattr(self.storage, "stats"):
            try:
                stored = self.storage.stats(self.agent_id)
            except Exception:  # noqa: BLE001 -- status must work against a broken store
                stored = {"error": "could not read snapshot metadata"}
        return {
            "agent_id": self.agent_id,
            "agent": agent_status,
            "lifecycle_state": agent_status["state"],
            "storage_backend": type(self.storage).__name__ if self.storage else None,
            "durable": self.storage is not None,
            "restored_from_snapshot": self.restored_from_snapshot,
            "last_saved_at": self.last_saved_at,
            "autosave": self.autosave,
            "lifecycle_observed": self.lifecycle_observed,
            "closed": self._closed,
            "stored_snapshot": stored,
        }

    def _check_open(self) -> None:
        if self._closed:
            raise StorageError(
                f"runtime for agent_id {self.agent_id!r} has been shut down; "
                "construct a new AgentRuntime to continue"
            )

    # -- context manager ----------------------------------------------------
    def __enter__(self) -> "AgentRuntime":
        self.wake()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # On an exception, still persist: losing acknowledged writes because
        # an unrelated error escaped is worse than storing state from a
        # process that was about to die anyway.
        try:
            self.shutdown(persist=True)
        except StorageError:
            if exc_type is None:
                raise
        return False

    def __repr__(self) -> str:
        return (
            f"AgentRuntime(agent_id={self.agent_id!r}, "
            f"storage={type(self.storage).__name__ if self.storage else None}, "
            f"state={self.agent.lifecycle.state.value})"
        )
