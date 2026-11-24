"""Exceptions raised by the AISOS SDK storage layer (aisos/storage/).

Separate module so a caller can `from aisos.storage.errors import
SnapshotCorruptError` without importing sqlite3 machinery.
"""
from __future__ import annotations

from typing import Optional


class StorageError(Exception):
    """Base class for every error the storage layer raises."""


class SnapshotNotFoundError(StorageError):
    def __init__(self, agent_id: str):
        super().__init__(f"no snapshot stored for agent_id {agent_id!r}")
        self.agent_id = agent_id


class SnapshotCorruptError(StorageError):
    """The stored bytes did not survive their own integrity check, or could
    not be parsed back into a snapshot.

    IMPORTANT, and stated precisely because it is easy to overclaim: the
    checksum this detects is a plain SHA-256 of the serialized payload with
    NO secret key. It detects accidental corruption -- a truncated write, a
    damaged file, a bad edit. It does NOT authenticate the snapshot: anyone
    who can modify the database file can also recompute the checksum. It is
    an integrity check, not a signature. See docs/sdk/security.md.
    """

    def __init__(self, agent_id: str, detail: str, cause: Optional[BaseException] = None):
        super().__init__(f"snapshot for agent_id {agent_id!r} is corrupt: {detail}")
        self.agent_id = agent_id
        self.detail = detail
        if cause is not None:
            self.__cause__ = cause


class IncompatibleSchemaError(StorageError):
    """The snapshot was written by a build with a schema version this build
    does not understand. Refused rather than partially applied: silently
    loading a subset of an unknown format is how state gets quietly lost."""

    def __init__(self, found, supported):
        super().__init__(
            f"snapshot schema_version {found!r} is not supported by this build "
            f"(supported: {supported!r}). Refusing to load rather than guess."
        )
        self.found = found
        self.supported = supported


class StorageBackendError(StorageError):
    """The underlying store (e.g. sqlite3) failed. The original exception is
    always preserved as __cause__."""
