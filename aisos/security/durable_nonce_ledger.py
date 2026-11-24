"""
Durable Nonce Ledger (AISOS_Restart_Replay_Closure_Milestone.md): the
first persistence mechanism anywhere in this reference architecture,
built specifically and narrowly to close the restart-replay vulnerability
-- a consumed credential's nonce must not become valid again after a
process restart or crash.

Design rationale (docs/restart_replay_design.md): of the five options
considered (durable nonce ledger; WAL-backed consumption events; a
checkpoint-backed registry; a fully durable registry; a WAL+reconstruction
hybrid), this is the durable-nonce-ledger option, chosen because:
  - the existing WriteAheadLog is owned by AISOSAgent, is IN-MEMORY only,
    and is reset every sleep cycle -- coupling a shared
    TrustedAuthorityRegistry (which may serve MULTIPLE agent replicas,
    per the trusted-authority milestone's own `registry=` sharing
    pattern) to one specific agent's transient WAL would be an awkward,
    incorrect dependency, not a reuse of existing infrastructure;
  - no other persistence primitive exists anywhere in this codebase to
    build on;
  - the actual required property (has this exact nonce ever been
    consumed?) is a simple append-only set, not a general-purpose
    transaction log -- a dedicated, minimal ledger is the smallest
    mechanism that satisfies it, reusing the WAL's own TAMPER-EVIDENCE
    *pattern* (SHA-256 hash chaining) even though it needed its own
    class, since nothing durable existed to literally import.

Format: one line per consumed nonce, `"{nonce}|{entry_hash}"`, where
`entry_hash = sha256(f"{nonce}|{prev_hash}")` -- the identical
hash-chaining scheme `aisos/security/wal.py::WriteAheadLog` already uses
for its own (in-memory, per-cycle) tamper evidence.

Fail-closed policy (Phase 8/J): ANY parse failure, hash-chain break, or
malformed trailing entry (e.g. a torn write from a crash mid-append)
causes construction to RAISE `LedgerCorruptionError` rather than silently
starting from an empty (and therefore falsely "nothing was ever
consumed") state. A research reference architecture does not need to
auto-recover from ambiguous security state; refusing to start is the
conservative, correct choice per this milestone's own explicit
instruction not to fail open.

Cross-process locking (AISOS_Simultaneous_MultiProcess_Security_Boundary_
Milestone.md): the deployment-boundary audit
(docs/deployment_security_boundary.md) determined that same-machine
simultaneous multi-process consumption of the SAME credential belongs
inside AISOS's security boundary (it is a narrow, consistent extension of
the already-accepted same-machine sequential cross-process case the
restart-replay milestone closed), while cross-machine/distributed access
remains explicitly out of scope. The root cause
(docs/multiprocess_race_analysis.md) was a process-local `threading.Lock`
providing zero mutual exclusion across OS processes, combined with each
process checking a STALE, private in-memory snapshot of consumed nonces
(a TOCTOU) rather than the authoritative on-disk file. `try_consume()`
now additionally acquires an OS-level advisory lock
(`_InterProcessFileLock`, `flock` on POSIX / `msvcrt.locking` on Windows
-- both standard library, no new dependency) around a MANDATORY re-read
of the ledger from disk before deciding, closing the same-machine
multi-process race while leaving the previously-established same-process
(threading.Lock) and sequential-cross-process (restart-replay) guarantees
unchanged. Scope: one local filesystem on one machine -- explicitly NOT
claimed to work correctly over a network filesystem (NFS/SMB advisory
locking is not reliably enforced), matching the deployment boundary this
mechanism is scoped to.
"""
from __future__ import annotations

import hashlib
import os
import sys
import threading
from pathlib import Path
from typing import Union

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


class LedgerCorruptionError(Exception):
    """Raised when the durable ledger's on-disk state is malformed or its
    hash chain is broken -- construction refuses to proceed (fail closed)
    rather than silently treating corrupted/ambiguous state as "nothing
    consumed yet."
    """


class _InterProcessFileLock:
    """Minimal OS-level advisory exclusive lock on a sidecar `<path>.lock`
    file. Portable across Windows (`msvcrt.locking`) and POSIX
    (`fcntl.flock`) using only the standard library. Blocking acquire;
    always released on `__exit__`, including when an exception propagates
    through the `with` block.

    Scope (docs/multiprocess_mechanism_design.md): a single local
    filesystem on a single machine. The OS releases the lock automatically
    if the holding process crashes or is killed while holding it -- no
    stale-lock cleanup mechanism is needed (verified empirically in
    aisos/tests/test_multiprocess_security_boundary.py). NOT claimed to
    provide correct mutual exclusion over a network filesystem.
    """

    def __init__(self, path: Union[str, Path]):
        self._lock_path = str(path) + ".lock"
        self._fh = None

    def __enter__(self) -> "_InterProcessFileLock":
        self._fh = open(self._lock_path, "a+b")
        if sys.platform == "win32":
            # LK_LOCK blocks until the lock is acquired or a fixed number
            # of internal retries elapse (raising OSError); loop to turn
            # that into an unconditional blocking wait, matching flock's
            # semantics on POSIX.
            while True:
                try:
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue
        else:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if sys.platform == "win32":
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()


class DurableNonceLedger:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._consumed: set[str] = set()
        self._last_hash = "genesis"
        self._load_and_verify()

    def _parse(self) -> tuple[set[str], str]:
        """Pure parse of the on-disk file into (consumed set, last hash).
        Raises LedgerCorruptionError on any malformed entry or broken hash
        chain. Does not mutate self -- shared by __init__ (initial load)
        and _reload_locked (mandatory re-read under the inter-process
        lock before every try_consume decision).
        """
        if not self.path.exists():
            return set(), "genesis"  # fresh ledger: nothing has ever been consumed
        text = self.path.read_text(encoding="utf-8")
        lines = text.splitlines()
        prev = "genesis"
        consumed: set[str] = set()
        for i, line in enumerate(lines):
            parts = line.split("|")
            if len(parts) != 2:
                raise LedgerCorruptionError(
                    f"malformed entry at line {i} in {self.path}: {line!r} -- "
                    f"refusing to start (fail closed) rather than silently discard consumed-nonce history"
                )
            nonce, entry_hash = parts
            expected = hashlib.sha256(f"{nonce}|{prev}".encode("utf-8")).hexdigest()
            if expected != entry_hash:
                raise LedgerCorruptionError(
                    f"hash chain broken at line {i} in {self.path} -- "
                    f"refusing to start (fail closed): a broken chain could mean an entry was "
                    f"deleted, reordered, or tampered with, and silently proceeding could let a "
                    f"previously-consumed credential become valid again"
                )
            consumed.add(nonce)
            prev = entry_hash
        return consumed, prev

    def _load_and_verify(self) -> None:
        self._consumed, self._last_hash = self._parse()

    def is_consumed(self, nonce: str) -> bool:
        with self._lock:
            return nonce in self._consumed

    def try_consume(self, nonce: str) -> bool:
        """Atomic check-and-durably-record, now safe across BOTH threads
        (self._lock, unchanged) AND OS processes (the new
        _InterProcessFileLock, AISOS_Simultaneous_MultiProcess_Security_
        Boundary_Milestone.md). Returns True if this call newly recorded
        the nonce as consumed; False if it was already consumed -- by
        this process, by a prior process (restart-replay), or by ANOTHER
        process racing this one at the same instant (the new guarantee).

        Sequencing per call: acquire the in-process lock (self._lock) ->
        acquire the inter-process lock (the OS-level advisory lock on the
        sidecar .lock file) -> MANDATORY re-read of the ledger from disk
        (closing the TOCTOU identified in
        docs/multiprocess_race_analysis.md: without this re-read, a
        process's decision would still be based on a stale snapshot taken
        before any other process's write) -> check -> write -> fsync ->
        release inter-process lock -> release in-process lock.

        Durability ordering (unchanged from the restart-replay milestone):
        the line is written, flushed, and fsync'd to the underlying
        storage device BEFORE the in-memory `_consumed` set is updated --
        if the process crashes between the fsync and the in-memory
        update, upon restart (or in another process acquiring the lock
        next) a fresh read of this same file will correctly see the nonce
        as consumed (I-R1 / I-MP4). If the process crashes BEFORE the
        fsync completes, the entry either was never written (torn writes
        are treated as corruption by _parse -- fail closed) or was not
        written at all, in which case the nonce remains legitimately
        unconsumed (I-R2 / I-MP3).

        LedgerCorruptionError from the mandatory re-read propagates
        uncaught (I-MP5, I-MP8): corruption -- whether from a single
        process's torn write or from an unprotected race BEFORE this
        fix -- must never be silently reinterpreted as "nothing has been
        consumed."
        """
        with self._lock:
            with _InterProcessFileLock(self.path):
                # In-place refresh (clear + update), not reassignment: preserves
                # the identity/type of self._consumed so a caller-injected
                # set subclass (e.g. a crash-simulation test overriding add())
                # is still the object the final self._consumed.add(nonce) call
                # below invokes.
                parsed_consumed, parsed_hash = self._parse()
                self._consumed.clear()
                self._consumed.update(parsed_consumed)
                self._last_hash = parsed_hash
                if nonce in self._consumed:
                    return False
                entry_hash = hashlib.sha256(f"{nonce}|{self._last_hash}".encode("utf-8")).hexdigest()
                line = f"{nonce}|{entry_hash}\n"
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
                self._consumed.add(nonce)
                self._last_hash = entry_hash
                return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._consumed)
