"""SQLite StorageBackend -- the SDK's real durable persistence.

This is the first mechanism in AISOS by which an agent's memory survives the
process exiting. (The only other file-touching code in the whole research
core is aisos/security/durable_nonce_ledger.py, whose own docstring calls
itself "the first persistence mechanism anywhere in this reference
architecture" -- and it persists replay-protection nonces, not memory.)

WHY SQLITE AND NOT A SERVER
    A single local agent's snapshot is one row. SQLite gives atomic commits,
    crash-safe writes and a single self-contained file with no daemon, no
    port and no credentials. Introducing Postgres here would add operational
    surface for zero capability the SDK actually needs.

DURABILITY, STATED HONESTLY
    save() runs inside one transaction with synchronous=FULL, so SQLite
    fsyncs before reporting success, and an interrupted write leaves the
    PREVIOUS row intact rather than a half-written one. That is SQLite's own
    atomicity guarantee, and it is the guarantee this class inherits -- no
    more. It does not protect against a failing disk, a filesystem that lies
    about fsync, or someone deleting the file.

JOURNAL MODE
    The default rollback journal is used rather than WAL mode, for two
    reasons: it keeps the store a single file with no -wal/-shm siblings to
    ship or clean up, and "WAL" already means something entirely different
    and load-bearing in this codebase (aisos.security.wal, the intra-cycle,
    in-memory write-ahead log). Not worth the ambiguity for a workload that
    writes one row per snapshot.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional, Union

from aisos.storage.base import StorageBackend
from aisos.storage.errors import (
    IncompatibleSchemaError,
    SnapshotCorruptError,
    SnapshotNotFoundError,
    StorageBackendError,
)

#: Version of the DATABASE layout (tables/columns) -- distinct from the
#: snapshot payload's own schema_version, which versions the state contract.
#: They move independently: the table layout can stay at 1 while the payload
#: contract advances, and vice versa.
DB_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS aisos_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_snapshots (
    agent_id        TEXT PRIMARY KEY,
    schema_version  INTEGER NOT NULL,
    checksum        TEXT NOT NULL,
    payload         TEXT NOT NULL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
"""


class SQLiteStorage(StorageBackend):
    """Durable snapshot storage in a local SQLite file.

    `path` may be ":memory:" for a non-durable database that still exercises
    every real SQLite code path -- useful in tests, but note that it does NOT
    test restart recovery, which needs a real file and a real second process.
    """

    def __init__(self, path: Union[str, Path] = "aisos.sqlite", timeout_s: float = 10.0):
        self.path = str(path)
        self._lock = threading.RLock()
        is_file = self.path != ":memory:"
        new_file = is_file and not os.path.exists(self.path)
        if is_file:
            parent = Path(self.path).parent
            if str(parent) and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
        try:
            # check_same_thread=False plus our own RLock: the connection is
            # shared across threads but every use is serialized by the lock,
            # which is the supported way to do this with sqlite3.
            self._conn = sqlite3.connect(self.path, timeout=timeout_s, check_same_thread=False)
            self._conn.execute("PRAGMA synchronous=FULL")
            with self._conn:
                self._conn.executescript(_SCHEMA)
                self._conn.execute(
                    "INSERT OR IGNORE INTO aisos_meta (key, value) VALUES (?, ?)",
                    ("db_schema_version", str(DB_SCHEMA_VERSION)),
                )
        except sqlite3.Error as exc:
            raise StorageBackendError(f"could not open SQLite store at {self.path!r}: {exc}") from exc

        if new_file:
            self._restrict_permissions()
        self._check_db_schema_version()

    # -- security ---------------------------------------------------------
    def _restrict_permissions(self) -> None:
        """Best-effort owner-only permissions on a newly created database.

        Honest scope: on POSIX this is a real 0600. On Windows, os.chmod only
        toggles the read-only attribute and does NOT restrict other users --
        the file inherits its directory's ACL. We therefore do not claim this
        makes the file private on Windows; callers storing sensitive memory
        there should place the database in a directory whose ACL they
        control. Failure here is never fatal: a store that works with default
        permissions is better than one that refuses to start.
        """
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def _check_db_schema_version(self) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM aisos_meta WHERE key = 'db_schema_version'"
            ).fetchone()
        found = int(row[0]) if row else None
        if found != DB_SCHEMA_VERSION:
            raise IncompatibleSchemaError(found, (DB_SCHEMA_VERSION,))

    # -- StorageBackend ---------------------------------------------------
    def save(self, agent_id: str, snapshot: dict) -> None:
        try:
            payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)
        except (TypeError, ValueError) as exc:
            raise StorageBackendError(
                f"snapshot for {agent_id!r} is not JSON-serializable: {exc}"
            ) from exc

        now = time.time()
        checksum = snapshot.get("checksum", "")
        schema_version = snapshot.get("schema_version")
        try:
            with self._lock, self._conn:
                # One statement, one transaction: either the new row replaces
                # the old one completely or the old row is untouched. There is
                # no window in which a reader observes a partial snapshot.
                self._conn.execute(
                    """
                    INSERT INTO agent_snapshots
                        (agent_id, schema_version, checksum, payload, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(agent_id) DO UPDATE SET
                        schema_version = excluded.schema_version,
                        checksum       = excluded.checksum,
                        payload        = excluded.payload,
                        updated_at     = excluded.updated_at
                    """,
                    (agent_id, schema_version, checksum, payload, now, now),
                )
        except sqlite3.Error as exc:
            raise StorageBackendError(f"failed to save snapshot for {agent_id!r}: {exc}") from exc

    def load(self, agent_id: str) -> dict:
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT payload FROM agent_snapshots WHERE agent_id = ?", (agent_id,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise StorageBackendError(f"failed to read snapshot for {agent_id!r}: {exc}") from exc
        if row is None:
            raise SnapshotNotFoundError(agent_id)

        try:
            snapshot = json.loads(row[0])
        except (ValueError, TypeError) as exc:
            raise SnapshotCorruptError(
                agent_id, f"stored payload is not valid JSON ({exc})", exc
            ) from exc
        if not isinstance(snapshot, dict):
            raise SnapshotCorruptError(agent_id, "stored payload is not a snapshot object")

        # Order matters: schema first. An unknown schema_version must be
        # reported as such, not as a checksum failure -- they call for
        # different responses (upgrade the reader vs. restore from backup).
        self._validate_schema(snapshot)
        self._verify_integrity(agent_id, snapshot)
        return snapshot

    def exists(self, agent_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM agent_snapshots WHERE agent_id = ?", (agent_id,)
            ).fetchone()
        return row is not None

    def delete(self, agent_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM agent_snapshots WHERE agent_id = ?", (agent_id,))
        return cur.rowcount > 0

    def list_agents(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT agent_id FROM agent_snapshots ORDER BY agent_id"
            ).fetchall()
        return [r[0] for r in rows]

    def stats(self, agent_id: str) -> Optional[dict]:
        """Metadata about a stored snapshot WITHOUT loading or verifying its
        payload -- used by `aisos status` / `aisos doctor`, which must be able
        to report on a corrupt store rather than blow up on it."""
        with self._lock:
            row = self._conn.execute(
                "SELECT schema_version, checksum, length(payload), created_at, updated_at "
                "FROM agent_snapshots WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "agent_id": agent_id,
            "schema_version": row[0],
            "checksum": row[1],
            "payload_bytes": row[2],
            "created_at": row[3],
            "updated_at": row[4],
        }

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
