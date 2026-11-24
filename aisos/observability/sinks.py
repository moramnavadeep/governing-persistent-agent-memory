"""Event sinks: where emitted events go.

A sink is anything with a `handle(event: Event) -> None` method. Three are
provided because three cover the real cases: keep them for inspection
(InMemorySink), write them to the application's logger (LoggingSink), or
hand them to arbitrary caller code (CallbackSink). Anything else -- OTLP,
Prometheus, a file rotator -- is a sink someone writes in ten lines, which
is why this module does not ship one.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Callable, Optional, Protocol

from aisos.observability.events import Event, EventType


class EventSink(Protocol):
    def handle(self, event: Event) -> None:
        ...


class InMemorySink:
    """Keeps the most recent events in a bounded deque.

    Bounded by default and deliberately so: an unbounded event buffer in a
    long-running agent is a memory leak with a nice name. `maxlen=None` is
    available for tests that want every event.
    """

    def __init__(self, maxlen: Optional[int] = 1000):
        self._events: deque[Event] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def handle(self, event: Event) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> list[Event]:
        with self._lock:
            return list(self._events)

    def of_type(self, event_type: EventType) -> list[Event]:
        return [e for e in self.events if e.type is event_type]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


class LoggingSink:
    """Writes events to a standard library logger.

    Uses the caller's logging configuration rather than configuring logging
    itself: a library that calls basicConfig() hijacks the whole
    application's logging, which is rude and hard to undo.
    """

    #: Event types logged at WARNING rather than INFO.
    WARNING_TYPES = frozenset({
        EventType.TURN_FAILED,
        EventType.TOOL_FAILED,
        EventType.PROVIDER_ERROR,
        EventType.SLEEP_ABORTED,
        EventType.RECOVERY_FAILED,
    })

    def __init__(self, logger: Optional[logging.Logger] = None, level: int = logging.INFO):
        self.logger = logger or logging.getLogger("aisos.events")
        self.level = level

    def handle(self, event: Event) -> None:
        level = logging.WARNING if event.type in self.WARNING_TYPES else self.level
        # %s on the Event, not an f-string: the formatting cost is skipped
        # entirely when the logger is below this level.
        self.logger.log(level, "%s", event)


class CallbackSink:
    """Adapts a plain callable into a sink."""

    def __init__(self, fn: Callable[[Event], None]):
        self.fn = fn

    def handle(self, event: Event) -> None:
        self.fn(event)
