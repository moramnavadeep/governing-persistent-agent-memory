"""
Continuous MVCC background-consolidation baseline.

Section 3 of the v2 document and Section 9 (Phase 2) both require this
baseline to exist and be *competently implemented* -- "not a strawman" --
because H1 (AISOS outperforms continuous consolidation) is only a
meaningful claim if H0's champion is genuinely good. This module is the
condition (b) baseline used across Experiments E1, E3, and E4.

Design, deliberately not a strawman:
  - True multi-version concurrency control: every write gets a
    monotonically increasing global txn id; nothing is ever mutated in
    place, matching real MVCC engines (Postgres, etc).
  - `snapshot_read()` pins a txn id at the start of a multi-key read and
    only considers versions committed at or before it -- this gives proper
    snapshot isolation and, like AISOS, does *not* exhibit torn reads.
    E1 measures this path as the fair comparison point for Theorem 1.
  - `naive_read()` is included separately to show what an *incompetent*
    continuous implementation looks like (no pinned snapshot across a
    multi-key read) -- reported for context, not used as AISOS's real
    competitor.
  - Fine-grained per-subject locking replaces AISOS's single-writer lock,
    which is the source of the "reintroduces a scheduling problem in a
    worse form" cost named in Section 3.2: every write pays a per-subject
    lock acquisition instead of amortizing over a batched cycle.
  - No discrete checkpoint boundary: "rollback" here means walking every
    subject's version history back to before a target txn id and
    discarding every later version -- for *every* subject touched since
    then, not just the offending one -- which is the concrete mechanism
    behind the "no clean before-state to revert to without also reverting
    legitimate concurrent work" argument (Section 3.2). This is measured,
    not asserted, in Experiment E3.
  - Consolidation (dedup + contradiction resolution) runs per-write,
    incrementally, using only local (this write vs. the immediately prior
    version) information -- it cannot batch a whole cluster's evidence the
    way Algorithm A3 does, because there is no cycle boundary to batch
    within. This is what produces a real, measured difference in
    contradiction rate against AISOS's cluster-level, authority-aware
    resolution (memory/contradiction.py), rather than a difference
    asserted by construction.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from aisos.memory.record import MemoryRecord, TrustTier, Status


@dataclass
class Version:
    txn_id: int
    record: MemoryRecord


class ContinuousMVCCBaseline:
    def __init__(self):
        self._global_txn = 0
        self._global_lock = threading.Lock()
        self._subject_locks: dict[str, threading.Lock] = {}
        self._history: dict[str, list[Version]] = {}
        self.contradiction_events = 0
        self.write_count = 0
        self.rollback_records_discarded = 0
        self.rollback_count = 0

    def _lock_for(self, subject: str) -> threading.Lock:
        with self._global_lock:
            if subject not in self._subject_locks:
                self._subject_locks[subject] = threading.Lock()
            return self._subject_locks[subject]

    def _next_txn(self) -> int:
        with self._global_lock:
            self._global_txn += 1
            return self._global_txn

    def write(self, subject: str, value: str, source: str = "user", confidence: float = 0.7) -> MemoryRecord:
        lock = self._lock_for(subject)
        with lock:
            # txn-id generation and the `self._history[subject]` mutation
            # must be one critical section under `_global_lock`, matching
            # what `rollback_to()` holds while it reassigns
            # `self._history[subject]` to a filtered list: otherwise a write
            # can bind `hist` to the list object `rollback_to()` is about to
            # replace, then append to that now-orphaned list after the
            # replacement -- a write silently lost from history despite the
            # caller seeing success. (Inlined rather than calling
            # `_next_txn()`, since `_global_lock` is a plain non-reentrant
            # Lock.)
            with self._global_lock:
                self._global_txn += 1
                txn_id = self._global_txn
                rec = MemoryRecord(
                    subject=subject, value=value, source=source, confidence=confidence,
                    trust_tier=TrustTier.T4_VERIFIED, status=Status.TRUSTED,
                    provenance={"origin": "continuous_mvcc", "txn": txn_id},
                )
                hist = self._history.setdefault(subject, [])
                # Incremental, *local* contradiction check: only against the
                # immediately preceding version -- no cluster-level batching
                # is possible without a cycle boundary.
                if hist and hist[-1].record.value != value:
                    self.contradiction_events += 1
                hist.append(Version(txn_id=txn_id, record=rec))
            self.write_count += 1
            return rec

    def naive_read(self, subjects: Optional[list[str]] = None) -> dict[str, MemoryRecord]:
        """No pinned snapshot: each subject's 'latest' is read independently,
        so a concurrent multi-subject write burst can produce a globally
        inconsistent (torn) view across the returned dict, even though each
        individual record object is itself never partially-written.
        """
        subs = subjects if subjects is not None else list(self._history.keys())
        out = {}
        for s in subs:
            hist = self._history.get(s)
            if hist:
                out[s] = hist[-1].record
        return out

    def snapshot_read(self, subjects: Optional[list[str]] = None, as_of_txn: Optional[int] = None) -> dict[str, MemoryRecord]:
        """Proper MVCC snapshot isolation: pin as_of_txn once, then every
        subject is resolved to its latest version <= as_of_txn. This is the
        competent baseline's read path -- it does not exhibit torn reads,
        matching AISOS on that specific axis (E1 measures this honestly).
        """
        if as_of_txn is None:
            with self._global_lock:
                as_of_txn = self._global_txn
        subs = subjects if subjects is not None else list(self._history.keys())
        out = {}
        for s in subs:
            hist = self._history.get(s, [])
            candidates = [v for v in hist if v.txn_id <= as_of_txn]
            if candidates:
                out[s] = candidates[-1].record
        return out

    def current_txn(self) -> int:
        with self._global_lock:
            return self._global_txn

    def rollback_to(self, target_txn: int) -> int:
        """Discard every version with txn_id > target_txn, across *every*
        subject -- this is O(total versions since target_txn), and it
        necessarily discards legitimate writes interleaved with whatever
        triggered the rollback, unlike AISOS's O(1) single-private-copy
        discard (Theorem 2). Returns the number of record-versions
        discarded, i.e. the blast radius.
        """
        discarded = 0
        with self._global_lock:
            for subject, hist in self._history.items():
                kept = [v for v in hist if v.txn_id <= target_txn]
                discarded += len(hist) - len(kept)
                self._history[subject] = kept
        self.rollback_records_discarded += discarded
        self.rollback_count += 1
        return discarded
