"""EventEmitter: the single emission point for SDK events.

THE TWO PROPERTIES THAT MATTER
    1. Zero cost when nobody is listening. emit() returns immediately if no
       sink is attached -- before building an Event, before redacting a
       payload. Observability you pay for whether or not you use it is
       observability people disable.
    2. It cannot break the caller. A sink that raises is caught, counted in
       `sink_errors`, and otherwise ignored. Agent code calls emit() in its
       normal path; a misbehaving log handler must not turn that into a
       failed turn.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

from aisos.observability.events import Event, EventType, redact
from aisos.observability.sinks import EventSink


class EventEmitter:
    def __init__(self, sinks: Optional[list] = None, enabled: bool = True):
        self._sinks: list = list(sinks or [])
        self._lock = threading.RLock()
        self.enabled = enabled
        #: (sink_repr, exception_repr) for every sink failure, bounded by
        #: nothing on purpose -- a failing sink is a bug to be seen, and in
        #: practice it fails on the first event or not at all.
        self.sink_errors: list[tuple[str, str]] = []
        self.emitted_count = 0

    # -- sink management --------------------------------------------------
    def add_sink(self, sink: EventSink) -> EventSink:
        with self._lock:
            self._sinks.append(sink)
        return sink

    def remove_sink(self, sink: EventSink) -> bool:
        with self._lock:
            try:
                self._sinks.remove(sink)
                return True
            except ValueError:
                return False

    def clear_sinks(self) -> None:
        with self._lock:
            self._sinks.clear()

    @property
    def sinks(self) -> list:
        with self._lock:
            return list(self._sinks)

    # -- emission ---------------------------------------------------------
    def emit(
        self,
        event_type: EventType,
        *,
        agent_id: Optional[str] = None,
        session: Optional[str] = None,
        lifecycle_state: Optional[str] = None,
        **data: Any,
    ) -> Optional[Event]:
        """Builds, redacts and dispatches one event.

        Returns the Event for callers that want it (tests, mostly), or None
        when emission was skipped because nothing is listening.
        """
        if not self.enabled:
            return None
        with self._lock:
            sinks = list(self._sinks)
        if not sinks:
            return None  # fast path: no Event allocated, no redaction run

        event = Event(
            type=event_type,
            ts=time.time(),
            agent_id=agent_id,
            session=session,
            lifecycle_state=lifecycle_state,
            data=redact(data),
        )
        for sink in sinks:
            try:
                sink.handle(event)
            except Exception as exc:  # noqa: BLE001 -- a sink must never break the caller
                with self._lock:
                    self.sink_errors.append((repr(sink), repr(exc)))
        with self._lock:
            self.emitted_count += 1
        return event


#: Process-wide default. It has NO sinks attached, so by default the SDK
#: emits nothing and costs nothing; attaching a sink is what turns
#: observability on.
DEFAULT_EMITTER = EventEmitter()
