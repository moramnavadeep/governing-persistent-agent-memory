"""Durable storage and recovery for the AISOS SDK (Phase D).

    AISOSAgent
        |
        +-- MemoryStore.snapshot() / .restore()      <- the serialization boundary
                 |
             create_snapshot() / restore_snapshot()  <- the agent-level contract
                 |
             StorageBackend
                 +-- InMemoryStorage   (tests/dev, NOT durable)
                 +-- SQLiteStorage     (durable, local file)

This layer is ENGINEERING scope, not research scope: nothing about it was
evaluated by the experiments behind the AISOS papers. It exists because the
research core has no durable persistence at all -- its write-ahead log and
checkpoints are in-memory and intra-cycle, and are cleared by WAKE_VERIFY
whether a cycle commits or rolls back.
"""
from aisos.storage.base import StorageBackend
from aisos.storage.errors import (
    IncompatibleSchemaError,
    SnapshotCorruptError,
    SnapshotNotFoundError,
    StorageBackendError,
    StorageError,
)
from aisos.storage.memory import InMemoryStorage
from aisos.storage.snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    compute_checksum,
    create_snapshot,
    restore_snapshot,
    verify_snapshot,
)
from aisos.storage.sqlite import SQLiteStorage

__all__ = [
    "StorageBackend",
    "InMemoryStorage",
    "SQLiteStorage",
    "SNAPSHOT_SCHEMA_VERSION",
    "create_snapshot",
    "restore_snapshot",
    "verify_snapshot",
    "compute_checksum",
    "StorageError",
    "StorageBackendError",
    "SnapshotNotFoundError",
    "SnapshotCorruptError",
    "IncompatibleSchemaError",
]
