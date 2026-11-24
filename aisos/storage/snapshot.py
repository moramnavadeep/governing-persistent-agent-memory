"""The agent-level durable snapshot contract (SDK Phase D).

TWO KINDS OF STATE -- THE DISTINCTION THIS MODULE EXISTS TO MAKE
----------------------------------------------------------------
AISOS already had transactional protection *within* one maintenance cycle,
and it is easy to mistake that for durability. It is not:

  intra-cycle transactional state          durable process-restart state
  -------------------------------          -----------------------------
  aisos.security.wal.WriteAheadLog          this module
  MemoryStore._private (copy-on-write)      MemoryStore.snapshot()/restore()
  MemoryStore._checkpoints (deque)          a StorageBackend
  in memory only                            on disk (SQLiteStorage)
  RESET every cycle by WAKE_VERIFY,         survives process termination
    on both commit and rollback
  protects against a failed cycle           protects against a process exit

The WAL is NOT durable persistence and this codebase must never describe it
as such: it is an in-memory, single-cycle structure that WAKE_VERIFY clears
whether the cycle commits or rolls back. Nothing in the research core
survives `python` exiting except aisos/security/durable_nonce_ledger.py.

WHAT IS PERSISTED
-----------------
  memory   -- MemoryStore.snapshot(): published (M_V), staging, store
              version, promotion-hash ledger.
  agent    -- context_buffer (observations not yet consolidated -- dropping
              these would lose acknowledged user writes across a restart),
              theta (the dream-phase parameter vector), the rollback /
              verify-failure counters, and the lifecycle state.

WHAT IS DELIBERATELY NOT PERSISTED
----------------------------------
  signing_key -- SECURITY DECISION, not an oversight. AISOSAgent.signing_key
                 signs checkpoint bundles. Writing it into the same file as
                 the data it protects would mean that anyone who obtains the
                 snapshot file can forge a valid-looking signed checkpoint.
                 A restored agent therefore gets a FRESH signing key, and
                 signatures produced before the restart do not verify after
                 it. That is the correct trade: a restart is a new trust
                 epoch. Callers who genuinely need key continuity must supply
                 the key themselves through the normal AISOSAgent(signing_key=...)
                 constructor argument, sourced from their own secret store.
  wal         -- intra-cycle by construction (see table above).
  checkpoints -- bounded in-process forensic history, not committed state.
  cycle_reports -- diagnostic summaries; not required to resume operation.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Optional

# Bump ONLY together with a documented migration path. The storage layer
# refuses an unknown version rather than loading it partially.
SNAPSHOT_SCHEMA_VERSION = 1

# Fields of the snapshot envelope that the checksum covers. `checksum`
# itself is obviously excluded.
_CHECKSUMMED_KEYS = ("schema_version", "agent_id", "created_at", "memory", "agent", "metadata")


def _canonical_bytes(snapshot: dict) -> bytes:
    """Deterministic serialization for checksumming.

    sort_keys=True and a fixed separator make the encoding independent of
    dict insertion order, so the same logical state always produces the same
    checksum -- otherwise a snapshot would fail its own integrity check
    purely because Python iterated a dict differently.
    """
    payload = {k: snapshot.get(k) for k in _CHECKSUMMED_KEYS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def compute_checksum(snapshot: dict) -> str:
    """SHA-256 over the canonical encoding.

    Scope, stated plainly: this detects ACCIDENTAL corruption (truncated
    write, damaged file, manual edit). It is unkeyed, so it does not
    authenticate anything -- an attacker with write access to the store can
    recompute it. Do not describe it as tamper-proofing.
    """
    return "sha256:" + hashlib.sha256(_canonical_bytes(snapshot)).hexdigest()


def create_snapshot(agent, agent_id: str, metadata: Optional[dict] = None) -> dict:
    """Builds a complete, checksummed snapshot of `agent`.

    Raises RuntimeError (from MemoryStore.snapshot()) if the agent has a
    maintenance cycle in flight -- persisting uncommitted cycle state is
    refused, not silently accepted.
    """
    snapshot: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "agent_id": agent_id,
        "created_at": time.time(),
        "memory": agent.store.snapshot(),
        "agent": {
            "context_buffer": [dict(e) for e in agent.context_buffer],
            "theta": list(agent.theta),
            "rollback_count": agent.rollback_count,
            "verify_fail_count": agent.verify_fail_count,
            "lifecycle_state": agent.lifecycle.state.value,
            "cycle_count": len(agent.cycle_reports),
        },
        "metadata": dict(metadata or {}),
    }
    snapshot["checksum"] = compute_checksum(snapshot)
    return snapshot


def verify_snapshot(snapshot: dict) -> bool:
    """True if the snapshot carries a checksum that matches its content."""
    stored = snapshot.get("checksum")
    if not stored:
        return False
    return stored == compute_checksum(snapshot)


def restore_snapshot(agent, snapshot: dict) -> None:
    """Applies `snapshot` to a freshly-constructed `agent`, in place.

    The agent must be constructed the normal way first (so that its
    scheduler, interrupt controller, locks and configuration are the
    caller's, not the snapshot's) -- this restores STATE, not configuration.
    Configuration deliberately does not travel in a snapshot: silently
    resurrecting an old tau_light or promotion threshold on restart would
    make a running system's behavior depend on a file rather than on its
    own config.

    The lifecycle is NOT forced to the saved state. A restored agent starts
    AWAKE, which is the only state from which it is safe to begin serving
    again: a snapshot cannot be taken mid-cycle (MemoryStore.snapshot()
    refuses), so the saved state is always AWAKE in practice, and forcing a
    transition here would mean writing to the research FSM from outside it.
    The saved value is kept in the snapshot for diagnostics and is reported
    by the SDK's recovery events.
    """
    agent.store.restore(snapshot["memory"])
    saved = snapshot.get("agent", {})
    with agent._lock:
        agent.context_buffer = [dict(e) for e in saved.get("context_buffer", [])]
    theta = list(saved.get("theta", []))
    if theta:
        # Length must match this agent's configured theta_dims; a mismatch
        # means the snapshot came from a differently-configured agent and is
        # a caller error, not something to paper over by padding.
        if len(theta) != len(agent.theta):
            raise ValueError(
                f"snapshot theta has {len(theta)} dimensions but this agent is configured "
                f"for {len(agent.theta)} (theta_dims). Construct the agent with matching "
                "theta_dims before restoring."
            )
        agent.theta = theta
    agent.rollback_count = int(saved.get("rollback_count", 0))
    agent.verify_fail_count = int(saved.get("verify_fail_count", 0))
