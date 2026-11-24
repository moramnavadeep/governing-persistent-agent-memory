"""
Write-ahead log (Technical Spec Section 14; v2 invariant I2, Write-Before-Log).

"Every mutating operation is logged before it is applied to the private
working copy." The log is per-cycle: it is opened when a mutating state is
entered and consumed (verified, then cleared) by WAKE_VERIFY.
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field


@dataclass
class WALEntry:
    seq: int
    op: str
    record_id: str
    payload: str          # a content hash, not the raw content -- WAL entries
                           # are cheap to sign/verify (Algorithm A6 cost model).
    ts: float = field(default_factory=time.time)
    prev_hash: str = ""
    entry_hash: str = ""


class WriteAheadLog:
    """Append-only within one sleep cycle. `reset()` is only ever called by
    WAKE_VERIFY, either after a successful commit (fresh log for the next
    cycle) or after a rollback (discarding the aborted cycle's log along
    with the discarded private copy).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: list[WALEntry] = []
        self._seq = 0
        # SECURITY CLOSURE (AISOS_Security_Closure_Milestone.md Phase G/H):
        # tracked independently of len(self._entries) specifically so that
        # TAIL TRUNCATION (dropping the last N entries from the list, which
        # leaves the REMAINING chain internally consistent -- is_chain_intact()
        # alone cannot detect this, confirmed in the comprehensive audit)
        # becomes detectable: an attacker who mutates `_entries` directly
        # would need to ALSO decrement this separate counter to stay
        # undetected, raising the bar from "one list slice" to "two
        # consistent mutations." This does not defend against a fully
        # capable T14 (direct object mutation) attacker who knows about
        # both fields -- that remains explicitly out of scope, as
        # everywhere else T14 is discussed in this project -- but it does
        # detect the class of truncation this milestone was asked to close
        # (a shortened-but-internally-valid chain produced by any means
        # that doesn't also patch this counter).
        self._committed_count = 0

    def append(self, op: str, record_id: str, payload: str) -> WALEntry:
        with self._lock:
            prev_hash = self._entries[-1].entry_hash if self._entries else "genesis"
            self._seq += 1
            entry = WALEntry(seq=self._seq, op=op, record_id=record_id, payload=payload, prev_hash=prev_hash)
            entry.entry_hash = hashlib.sha256(
                f"{entry.seq}|{entry.op}|{entry.record_id}|{entry.payload}|{entry.prev_hash}".encode()
            ).hexdigest()
            self._entries.append(entry)
            self._committed_count += 1
            return entry

    def entries(self) -> list[WALEntry]:
        with self._lock:
            return list(self._entries)

    def is_chain_intact(self) -> bool:
        """O(|wal_entries|) integrity check: recompute each entry's hash and
        confirm the prev_hash chain is unbroken. This is the "signature
        verification" cost term in Algorithm A6's cost model, simplified to
        a hash chain (see security/checkpoint.py for the HMAC signing layer
        applied on top of the whole log at commit time).
        """
        with self._lock:
            if len(self._entries) != self._committed_count:
                # Tail truncation (or any other length mismatch): the
                # entries actually present no longer match the number of
                # successful append() calls this cycle has recorded.
                return False
            prev = "genesis"
            for e in self._entries:
                expected = hashlib.sha256(
                    f"{e.seq}|{e.op}|{e.record_id}|{e.payload}|{prev}".encode()
                ).hexdigest()
                if expected != e.entry_hash or e.prev_hash != prev:
                    return False
                prev = e.entry_hash
            return True

    def verify_record_integrity(self, record) -> bool:
        """Active provenance-integrity re-verification
        (AISOS_Next_Milestone_Replay_Provenance.md Phase J): recomputes
        `record.content_hash()` and compares it against the `payload` of
        the MOST RECENT WAL entry logged for this record's id. Returns
        False (tampering / corruption detected) if either:
          - no WAL entry exists for this id at all (the record was never
            logged through an observed mutation path -- e.g. a direct
            object construction bypassing remember()/promotion_pipeline);
          - the most recent logged payload does not match the record's
            CURRENT content hash (something changed the record's fields
            -- including verified_authority, since content_hash() covers
            it -- after it was logged, without going through a code path
            that re-logs the mutation).
        This is the "active" half of provenance integrity: the pre-
        existing is_chain_intact() only verifies the WAL's own internal
        consistency (no entries inserted/removed/reordered); it does not,
        by itself, check that a live record still matches what was last
        logged about it. This method closes that specific gap, reusing
        the existing content_hash()/WAL infrastructure -- no new
        cryptographic primitive.
        """
        with self._lock:
            matching = [e for e in self._entries if e.record_id == record.id]
        if not matching:
            return False
        return matching[-1].payload == record.content_hash()

    def reset(self) -> None:
        with self._lock:
            self._entries = []
            self._seq = 0
            self._committed_count = 0

    def __len__(self) -> int:
        return len(self._entries)
