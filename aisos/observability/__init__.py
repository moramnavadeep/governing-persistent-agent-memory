"""Lightweight observability for the AISOS SDK (Phase E).

    agent / kernel / tools / storage
              |
          EventEmitter  (no sinks attached -> emits nothing, costs nothing)
              |
      +-------+--------+-------------+
      |                |             |
  InMemorySink    LoggingSink   CallbackSink

This is ENGINEERING scope. It observes the SDK's own operations; it does not
instrument the research core, and it does not add states to the five-state
research lifecycle -- lifecycle state travels on events as reported data.
"""
from aisos.observability.emitter import DEFAULT_EMITTER, EventEmitter
from aisos.observability.events import REDACTED, Event, EventType, redact
from aisos.observability.sinks import CallbackSink, EventSink, InMemorySink, LoggingSink

__all__ = [
    "EventEmitter", "DEFAULT_EMITTER",
    "Event", "EventType", "redact", "REDACTED",
    "EventSink", "InMemorySink", "LoggingSink", "CallbackSink",
]
