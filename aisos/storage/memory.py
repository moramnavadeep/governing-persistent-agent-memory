"""In-memory StorageBackend: for tests, development and ephemeral runs.

This is NOT durable. It is included because a great many tests need a
storage backend's *semantics* (schema validation, corruption detection,
overwrite behavior) without touching the filesystem, and because testing
those semantics against a real file is slower and flakier. Anything that
claims to survive a process restart must be tested against SQLiteStorage --
and the recovery tests are, in a genuinely separate process.
"""
from __future__ import annotations

import copy
import threading

from aisos.storage.base import StorageBackend
from aisos.storage.errors import SnapshotNotFoundError


class InMemoryStorage(StorageBackend):
    def __init__(self) -> None:
        self._data: dict[str, dict] = {}
        self._lock = threading.RLock()

    def save(self, agent_id: str, snapshot: dict) -> None:
        # Deep-copied on the way in AND on the way out, so a caller that
        # keeps mutating its own agent cannot retroactively change what is
        # "stored" -- which would make this backend behave unlike a real
        # one and let a broken test pass.
        stored = copy.deepcopy(snapshot)
        with self._lock:
            self._data[agent_id] = stored

    def load(self, agent_id: str) -> dict:
        with self._lock:
            if agent_id not in self._data:
                raise SnapshotNotFoundError(agent_id)
            snapshot = copy.deepcopy(self._data[agent_id])
        self._validate_schema(snapshot)
        self._verify_integrity(agent_id, snapshot)
        return snapshot

    def exists(self, agent_id: str) -> bool:
        with self._lock:
            return agent_id in self._data

    def delete(self, agent_id: str) -> bool:
        with self._lock:
            return self._data.pop(agent_id, None) is not None

    def list_agents(self) -> list[str]:
        with self._lock:
            return sorted(self._data)

    def close(self) -> None:
        with self._lock:
            self._data.clear()
