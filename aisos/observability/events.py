"""The AISOS SDK event model (Phase E).

DESIGN CONSTRAINTS
    * Lightweight. This is not a monitoring platform. An Event is a frozen
      dataclass and a sink is any object with `handle(event)`. There is no
      agent daemon, no exporter, no metrics server.
    * Never breaks the caller. Emission is best-effort: a sink that raises
      is isolated and counted, never propagated into agent code. An
      observability layer that can take down the thing it observes is worse
      than no observability layer.
    * Never leaks secrets. Every event payload passes through redact()
      before it reaches any sink, so an API key cannot arrive in a log file
      because someone put a config dict in an event.
    * Separate from the research lifecycle. The five research states
      (AWAKE / LIGHT / DEEP / DREAM / WAKE_VERIFY) are reported as DATA on
      an event; this module does not add states to that FSM. The SDK-level
      notions of "failed" or "recovering" live in event types and in Kernel
      metadata, never in the research state machine.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

#: Substrings that mark a field name as secret-bearing. Matching is
#: case-insensitive on the key.
SECRET_KEY_MARKERS = (
    "api_key", "apikey", "key", "token", "secret", "password", "passwd",
    "authorization", "auth", "credential", "signing_key", "bearer",
)

#: Value prefixes that look like credential material regardless of the field
#: name they arrived under -- a key pasted into a `note` field is still a key.
SECRET_VALUE_PREFIXES = ("sk-", "sk-ant-", "gsk_", "hf_", "ghp_", "xoxb-", "Bearer ")

REDACTED = "***redacted***"

#: Keys that contain the word "key" but are ordinary, non-secret metadata.
#: Without this, `keyword`/`key_count` style fields would be redacted into
#: uselessness.
_SECRET_KEY_EXEMPTIONS = ("keyword", "key_count", "keys", "monkeypatch")


class EventType(str, Enum):
    """SDK-level events. Deliberately a closed set: an open string space
    makes downstream filtering unreliable, and every type here corresponds
    to a real emission site in this codebase, not an aspirational one."""

    AGENT_CREATED = "agent_created"
    AGENT_REGISTERED = "agent_registered"
    AGENT_UNREGISTERED = "agent_unregistered"

    TURN_STARTED = "turn_started"
    TURN_COMPLETED = "turn_completed"
    TURN_FAILED = "turn_failed"

    MEMORY_WRITTEN = "memory_written"

    SNAPSHOT_CREATED = "snapshot_created"
    SNAPSHOT_SAVED = "snapshot_saved"
    SNAPSHOT_RESTORED = "snapshot_restored"
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_COMPLETED = "recovery_completed"
    RECOVERY_FAILED = "recovery_failed"

    SLEEP_STARTED = "sleep_started"
    SLEEP_COMPLETED = "sleep_completed"
    SLEEP_ABORTED = "sleep_aborted"
    LIFECYCLE_TRANSITION = "lifecycle_transition"

    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"

    PROVIDER_ERROR = "provider_error"

    SHUTDOWN = "shutdown"


def _looks_secret_key(key: str) -> bool:
    lowered = str(key).lower()
    if any(exempt in lowered for exempt in _SECRET_KEY_EXEMPTIONS):
        return False
    return any(marker in lowered for marker in SECRET_KEY_MARKERS)


def _looks_secret_value(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(SECRET_VALUE_PREFIXES)


def redact(payload: Any, _depth: int = 0) -> Any:
    """Recursively replace anything that looks like credential material.

    Applied to EVERY event payload before it reaches a sink. It is
    deliberately aggressive: over-redacting a debug field is a minor
    annoyance, while writing one API key into a log file is a real incident.

    Scope, stated honestly: this catches secrets in event payloads by field
    name and by well-known value prefixes. It cannot recognize an arbitrary
    opaque credential that has neither a suggestive field name nor a known
    prefix. Do not put credentials in event payloads and rely on this to
    save you.
    """
    if _depth > 6:  # runaway/recursive structures
        return "..."
    if isinstance(payload, dict):
        out = {}
        for k, v in payload.items():
            if _looks_secret_key(k):
                out[k] = REDACTED
            else:
                out[k] = redact(v, _depth + 1)
        return out
    if isinstance(payload, (list, tuple)):
        return [redact(v, _depth + 1) for v in payload]
    if isinstance(payload, (bytes, bytearray)):
        # Raw bytes in an event are almost always key material (this
        # codebase's signing keys are bytes) -- never render them.
        return REDACTED
    if _looks_secret_value(payload):
        return REDACTED
    return payload


@dataclass(frozen=True)
class Event:
    """One structured observation.

    `lifecycle_state` carries the RESEARCH lifecycle state as a string when
    one is known. It is reported, never set, by this layer.
    """

    type: EventType
    ts: float = field(default_factory=time.time)
    agent_id: Optional[str] = None
    session: Optional[str] = None
    lifecycle_state: Optional[str] = None
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "ts": self.ts,
            "agent_id": self.agent_id,
            "session": self.session,
            "lifecycle_state": self.lifecycle_state,
            "data": self.data,
        }

    def __str__(self) -> str:
        bits = [f"{self.type.value}"]
        if self.agent_id:
            bits.append(f"agent={self.agent_id}")
        if self.session:
            bits.append(f"session={self.session}")
        if self.lifecycle_state:
            bits.append(f"state={self.lifecycle_state}")
        for k, v in self.data.items():
            bits.append(f"{k}={v}")
        return " ".join(bits)
