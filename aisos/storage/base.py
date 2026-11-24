"""StorageBackend: the durable-persistence interface for the AISOS SDK.

A backend stores one current snapshot per agent_id. It is deliberately NOT
a general document database and not versioned history -- "what state should
this agent resume from" is the only question it answers, and keeping it that
narrow is what lets InMemoryStorage and SQLiteStorage be trivially
interchangeable in tests.

Every backend must:
  * round-trip a snapshot dict unchanged (save -> load is an identity for
    the snapshot's meaningful content);
  * detect accidental corruption on load and raise SnapshotCorruptError
    rather than returning damaged state;
  * refuse an unsupported schema_version with IncompatibleSchemaError
    rather than loading it partially;
  * make save() atomic with respect to concurrent readers -- an interrupted
    save must leave the PREVIOUS snapshot intact, never a half-written one.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from aisos.storage.errors import IncompatibleSchemaError
from aisos.storage.snapshot import SNAPSHOT_SCHEMA_VERSION, verify_snapshot


class StorageBackend(ABC):
    """Abstract durable store for agent snapshots."""

    #: Schema versions this backend will load. A backend subclass does not
    #: normally override this -- it is the SDK's snapshot contract version.
    supported_schema_versions: tuple[int, ...] = (SNAPSHOT_SCHEMA_VERSION,)

    @abstractmethod
    def save(self, agent_id: str, snapshot: dict) -> None:
        """Persist `snapshot` as the current state of `agent_id`, replacing
        any previous one atomically."""

    @abstractmethod
    def load(self, agent_id: str) -> dict:
        """Return the stored snapshot. Raises SnapshotNotFoundError,
        SnapshotCorruptError or IncompatibleSchemaError."""

    @abstractmethod
    def exists(self, agent_id: str) -> bool:
        ...

    @abstractmethod
    def delete(self, agent_id: str) -> bool:
        """Returns True if something was deleted, False if there was nothing
        to delete. Never raises for a missing agent_id."""

    @abstractmethod
    def list_agents(self) -> list[str]:
        ...

    def close(self) -> None:
        """Release any underlying resource. Safe to call more than once;
        the default is a no-op for backends that hold nothing."""
        return None

    # -- shared validation, used by every backend's load() ----------------
    def _validate_schema(self, snapshot: dict) -> None:
        found = snapshot.get("schema_version")
        if found not in self.supported_schema_versions:
            raise IncompatibleSchemaError(found, self.supported_schema_versions)

    def _verify_integrity(self, agent_id: str, snapshot: dict) -> None:
        from aisos.storage.errors import SnapshotCorruptError

        if not verify_snapshot(snapshot):
            raise SnapshotCorruptError(
                agent_id,
                "checksum mismatch -- the stored content does not match its recorded "
                "SHA-256, so it was damaged or edited after being written",
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
