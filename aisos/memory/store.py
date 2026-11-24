"""
Copy-on-write memory store (Technical Spec Section 13; v2 invariant I4).

    Trusted State ckpt*_n
        |
        +--> live inference   (dereferences `published` only)
        |
        +--> private working copy ckpt*_n + delta
                 |
                 +--> consolidate / dream / validate (mutates the copy only)

`published` is the only thing a live read ever sees. A sleep cycle mutates
`private` (created by begin_private_copy, a deep copy). WAKE_VERIFY either
publishes the private copy (atomic pointer swap -- the single write in
Algorithm A6) or discards it, in which case `published` is untouched.

This is also the mechanism Theorem 2 (Bounded Rollback) depends on:
rollback is O(1) because it is a discard, not an undo.
"""
from __future__ import annotations

import copy
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from aisos.memory.record import MemoryRecord, TrustTier, Status

# Checkpoint Accumulation & Long-Horizon Scaling Closure
# (AISOS_Checkpoint_Accumulation_Scaling_Closure_Prompt.md): the default
# retention window for MemoryStore._checkpoints (see docs/
# checkpoint_retention_design.md "Choosing N" for the full justification).
# Bounded, not unbounded -- the forensic audit
# (docs/checkpoint_accumulation_forensic_audit.md) traced checkpoint_at()'s
# ONLY real consumer (concurrent-read torn-read validation,
# aisos/sim/harness.py::ReadLoadGenerator + Theorem 1's own fuzz test) and
# confirmed the realistic version-gap between a reader's capture and its
# validation is 0-2 versions in every configuration this project runs;
# N=50 keeps a >=25x safety margin while still bounding total checkpoint
# memory to a small, fixed multiple of one current-state size, regardless
# of total run length.
DEFAULT_CHECKPOINT_RETENTION = 50

# ---------------------------------------------------------------------------
# Durable-snapshot serialization (SDK Phase D, approved Option B).
#
# WHY THIS LIVES HERE AND WHY IT IS ADDITIVE ONLY
# ------------------------------------------------
# Nothing in this store was durable before: `_published`, `_staging`,
# `_checkpoints` and `_promotion_hashes` are process-lifetime in-memory
# structures, and the WriteAheadLog is BOTH in-memory AND reset every cycle.
# The SDK needed a way to persist and reconstruct a store across a process
# restart. Three options were considered; the two rejected ones matter:
#
#   (A) let the SDK read/write `_published`/`_staging` directly. Rejected as
#       a permanent architecture: it welds the SDK to private attributes of a
#       research module that must stay free to evolve.
#   (C) restore by replaying records through ingest_raw(). Rejected as
#       INCORRECT: ingest_raw() is the AWAKE-time write path and lands
#       everything in `_staging` as a T1 candidate. Restoring a T4_VERIFIED,
#       live-readable record through it would silently DE-PROMOTE it, so a
#       process restart would quietly undo promotion decisions that the
#       Promotion Pipeline and WAKE_VERIFY had already made. That is exactly
#       the class of silent semantic change this codebase must not have.
#
# What is below is therefore (B): an explicit, versioned serialization
# boundary. It adds two methods and changes NO existing behavior -- no
# existing call path reaches them, and every field they touch keeps its
# meaning. Trust tier, status, version, provenance and verified_authority all
# round-trip exactly, so a restored store is promotion-equivalent to the one
# that was saved.
#
# WHAT IS DELIBERATELY *NOT* IN A SNAPSHOT
#   - `_private`: an open private working copy is intra-cycle, uncommitted
#     state. snapshot() refuses to run while one is open rather than
#     persisting a half-finished maintenance cycle (see I4).
#   - `_checkpoints`: bounded in-process rollback/forensic history. It is a
#     detection aid for in-place mutation within one process's lifetime, not
#     part of the committed state, and persisting 50 deep copies of full
#     state per agent would multiply snapshot size for no recovery benefit.
#     A restored store starts a fresh checkpoint history at the restored
#     version -- documented, not silent.
MEMORY_SNAPSHOT_SCHEMA_VERSION = 1


def _record_to_dict(rec: MemoryRecord) -> dict:
    """JSON-safe projection of one MemoryRecord.

    Written by hand rather than via dataclasses.asdict() so that the exact
    set of persisted fields is explicit and reviewable: if a field is later
    added to MemoryRecord, this function does not silently start persisting
    it under a schema version that never described it.
    """
    return {
        "id": rec.id,
        "subject": rec.subject,
        "value": rec.value,
        "source": rec.source,
        "created_at": rec.created_at,
        "updated_at": rec.updated_at,
        "trust_tier": int(rec.trust_tier),
        "status": str(rec.status),
        "confidence": rec.confidence,
        "version": rec.version,
        "validation_rounds": rec.validation_rounds,
        "related": list(rec.related),
        "provenance": dict(rec.provenance),
        "embedding": list(rec.embedding) if rec.embedding is not None else None,
        "verified_authority": rec.verified_authority,
    }


def _record_from_dict(d: dict) -> MemoryRecord:
    """Inverse of _record_to_dict.

    `embedding` is restored as a tuple (JSON round-trips it as a list) so a
    restored record is identical to the saved one, not merely equivalent;
    `trust_tier` is restored to the TrustTier IntEnum so that the
    `trust_tier >= LIVE_READABLE_FLOOR` comparisons in record.py and the
    promotion pipeline behave exactly as they did pre-restart.
    """
    embedding = d.get("embedding")
    return MemoryRecord(
        id=d["id"],
        subject=d["subject"],
        value=d["value"],
        source=d.get("source", "user"),
        created_at=d["created_at"],
        updated_at=d["updated_at"],
        trust_tier=TrustTier(d["trust_tier"]),
        status=d.get("status", Status.CANDIDATE),
        confidence=d.get("confidence", 0.5),
        version=d.get("version", 1),
        validation_rounds=d.get("validation_rounds", 0),
        related=list(d.get("related", [])),
        provenance=dict(d.get("provenance", {})),
        embedding=tuple(embedding) if embedding is not None else None,
        verified_authority=d.get("verified_authority"),
    )



@dataclass
class Checkpoint:
    version: int
    published: dict            # id -> MemoryRecord (M_V only, i.e. live-readable)
    staging: dict               # id -> MemoryRecord (M_D + M_C + rejected M_X, all non-live tiers)
    ts: float = field(default_factory=time.time)
    hash: str = ""


class MemoryStore:
    def __init__(self, checkpoint_retention: Optional[int] = DEFAULT_CHECKPOINT_RETENTION):
        """checkpoint_retention: maximum number of historical Checkpoint
        objects retained in self._checkpoints (docs/checkpoint_retention_design.md).
        None means unbounded (the ORIGINAL, pre-fix behavior) -- kept
        available, opt-in, for any caller that genuinely needs full
        history (e.g. a future diagnostic tool), but is NOT the default,
        since the forensic audit found no production code path requires
        it. Bounded via collections.deque(maxlen=...), which evicts the
        OLDEST checkpoint automatically and in O(1) once the window is
        full -- no separate pruning pass, no new lock ordering to reason
        about (the existing self._lock already serializes every mutation
        site, unchanged).
        """
        self._lock = threading.RLock()
        self._version = 0
        self._published: dict[str, MemoryRecord] = {}
        self._staging: dict[str, MemoryRecord] = {}
        self._private: Optional[dict] = None  # {"published": {...}, "staging": {...}}
        self._checkpoint_retention = checkpoint_retention
        self._checkpoints: deque[Checkpoint] = deque(maxlen=checkpoint_retention)
        # SECURITY CLOSURE (AISOS_Security_Closure_Milestone.md Phase K/L,
        # closing the "post-publication provenance integrity" gap): the
        # WAL is reset every cycle (its "promotion_finalized" entries do
        # NOT survive past the cycle that created them), so it cannot
        # serve as evidence for verifying an ALREADY-published record in a
        # LATER cycle. MemoryStore, unlike the WAL, already persists
        # across cycles (it is never reset) -- the natural, minimal home
        # for a durable (for the process's lifetime; not across restarts,
        # matching every other in-memory structure in this codebase)
        # record of each promoted record's content_hash at the moment of
        # its own promotion, checked continuously thereafter by
        # check_published_integrity() below.
        self._promotion_hashes: dict[str, str] = {}
        self._publish_ckpt(initial=True)

    def record_promotion_hash(self, record_id: str, content_hash: str) -> None:
        """Called once, at the moment a record reaches T4_VERIFIED
        (wake_verify.py), regardless of whether continuous checking is
        enabled -- recording is unconditional and O(1) per promotion, so
        enabling the check later still has historical evidence to compare
        against.
        """
        with self._lock:
            self._promotion_hashes[record_id] = content_hash

    def check_published_integrity(self) -> list[str]:
        """Returns the ids of currently-published records whose CURRENT
        content_hash() no longer matches what was recorded at their own
        promotion time. A record with no recorded hash (e.g. published
        before this mechanism existed, or never promoted through the
        normal path) is skipped -- "no evidence" is reported as such, not
        conflated with "definitely tampered." This is the continuous,
        post-publication counterpart to WriteAheadLog.verify_record_integrity(),
        which only ever covers the single cycle a record was promoted in.
        """
        with self._lock:
            published = list(self._published.values())
            hashes = dict(self._promotion_hashes)
        tampered = []
        for r in published:
            recorded = hashes.get(r.id)
            if recorded is not None and recorded != r.content_hash():
                tampered.append(r.id)
        return tampered

    # ---- I4: copy-on-write ----------------------------------------------
    def begin_private_copy(self) -> dict:
        with self._lock:
            if self._private is not None:
                raise RuntimeError("a private working copy is already open (I1 violation)")
            self._private = {
                "published": copy.deepcopy(self._published),
                "staging": copy.deepcopy(self._staging),
            }
            return self._private

    def private(self) -> dict:
        if self._private is None:
            raise RuntimeError("no private working copy is open")
        return self._private

    def discard_private_copy(self) -> None:
        """Theorem 2's O(1) rollback: just drop the reference. No undo log
        replay, no per-record revert -- this is what makes rollback latency
        independent of how much work the aborted cycle had done.
        """
        with self._lock:
            self._private = None

    def publish(self) -> int:
        """The single atomic pointer swap (Algorithm A6 step 6 / Theorem 1's
        proof). Only WAKE_VERIFY calls this, and only after signature +
        contradiction + drift checks all pass.
        """
        with self._lock:
            if self._private is None:
                raise RuntimeError("nothing to publish -- no private working copy")
            self._evict_expired_terminal_staging(self._private["staging"])
            self._published = self._private["published"]
            self._staging = self._private["staging"]
            self._private = None
            self._version += 1
            self._publish_ckpt()
            return self._version

    def _evict_expired_terminal_staging(self, new_staging: dict) -> None:
        # `_staging` Retention & Memory Scaling
        # (AISOS_staging_retention_next_milestone.md): forensic audit
        # (docs/staging_retention_forensic_audit.md) found no live decision
        # path -- promotion_pipeline(), deep_sleep(), the read() fallback --
        # ever re-examines a staging entry once it reaches a terminal
        # status (ARCHIVED, REJECTED, or TRUSTED-and-already-in-_published);
        # every real consumer that inspects such an entry via all_staging()
        # does so within the SAME cycle it became terminal, never later.
        # Retention invariant (docs/staging_retention_invariant.md):
        # a terminal entry is evicted once it has survived one full
        # committed cycle unchanged -- i.e. it was ALREADY that same
        # terminal status in the previously-committed self._staging. A
        # record that just became terminal THIS cycle is left alone (still
        # visible immediately after this publish()); it is swept on the
        # NEXT publish() instead. This is a generation bound, not an
        # invented numeric cap (docs/staging_retention_design.md Option 7).
        for rid, rec in list(new_staging.items()):
            if rec.status in (Status.ARCHIVED, Status.REJECTED, Status.TRUSTED):
                prior = self._staging.get(rid)
                if prior is not None and prior.status == rec.status:
                    del new_staging[rid]

    def _publish_ckpt(self, initial: bool = False) -> None:
        # Checkpoint Accumulation & Long-Horizon Scaling Closure
        # (AISOS_Checkpoint_Accumulation_Scaling_Closure_Prompt.md):
        # deep copy is RETAINED here deliberately -- an earlier version of
        # this fix switched to a shallow copy (dict(...) instead of
        # copy.deepcopy(...)), reasoning that MemoryRecord objects are
        # never mutated in place in PRODUCTION code (true -- every
        # legitimate transition goes through clone_bumped(), a new
        # object). That reasoning missed that the checkpoint audit trail
        # is ITSELF a validated detection mechanism against exactly the
        # kind of illegal in-place mutation clone_bumped() is designed to
        # avoid (test_theorem1_negative_control_mutating_published_in_place_is_detected,
        # aisos/tests/test_invariants.py): if a caller bypasses the
        # public API and mutates a MemoryRecord object directly (a
        # deliberate T14-class white-box attack simulation, matching
        # every other "detect, not necessarily prevent, direct object
        # mutation" disclosure throughout this project), a checkpoint
        # that SHARES that same object reference can no longer diverge
        # from the tampered live state, silently destroying this
        # detection capability. Caught by running the full regression
        # suite before finalizing this change -- exactly the "optimize
        # memory while silently breaking correctness" failure mode this
        # milestone's own rules explicitly forbid. The bounded-retention
        # window (self._checkpoints now a maxlen=N deque, see __init__)
        # is retained and is, on its own, sufficient to convert unbounded
        # growth into growth bounded by N x current-state-size --
        # measured directly in docs/checkpoint_scaling_report.md.
        #
        # `_published` Memory Scaling, Retention & Compaction
        # (AISOS_published_memory_scaling_next_milestone.md): checkpoint-
        # retained history, not live `_published`/`_staging` themselves,
        # was measured to dominate steady-state memory (a ~29x multiplier
        # at 3,000 events: 30.9MB of checkpoint-retained copies vs. 1.06MB
        # of live state -- docs/published_checkpoint_interaction.md).
        # "Checkpoint structural sharing" (reusing an already-made frozen
        # copy for a record unchanged since the last checkpoint, instead
        # of re-deep-copying it every cycle) was implemented and then
        # REVERTED after measurement showed it provides ZERO benefit:
        # `begin_private_copy()` above already deep-copies the ENTIRE
        # published/staging state on every cycle, unconditionally, to
        # create the private working copy (required for I4 isolation) --
        # meaning by the time THIS function runs, every record is already
        # a brand-new Python object (different identity) than whatever was
        # frozen into the previous checkpoint, REGARDLESS of whether its
        # content actually changed. An identity-based cache can therefore
        # never hit (measured: 0/274 published and 0/548 staging records
        # shared by identity between consecutive checkpoints in the
        # attempted implementation). A CONTENT-based comparison was
        # considered and rejected as unsafe: MemoryRecord.content_hash()
        # does not cover `status`/`confidence`/`validation_rounds`, so a
        # CANDIDATE->ARCHIVED transition with the same subject/value/tier
        # would hash identically and be wrongly treated as "unchanged,"
        # silently losing the status transition from a reused checkpoint
        # copy. A full dataclass equality check would be safe but costs
        # roughly the same order of work as the deepcopy it would
        # sometimes avoid (MemoryRecord carries a 16-float embedding
        # tuple), for uncertain net benefit -- not a clearly minimal,
        # clearly beneficial change. See
        # docs/published_memory_design_options.md's Option J entry for
        # the full evaluation. Deep copy is retained here, unchanged.
        ck = Checkpoint(
            version=self._version,
            published=copy.deepcopy(self._published),
            staging=copy.deepcopy(self._staging),
        )
        self._checkpoints.append(ck)

    # ---- reads (I4: always dereference the currently published pointer) --
    def read_live(self, subject: Optional[str] = None) -> list[MemoryRecord]:
        with self._lock:
            recs = list(self._published.values())
        if subject is not None:
            recs = [r for r in recs if r.subject == subject]
        return recs

    def get_live(self, record_id: str) -> Optional[MemoryRecord]:
        with self._lock:
            return self._published.get(record_id)

    def read_live_with_version(self, subject: Optional[str] = None) -> tuple[int, list[MemoryRecord]]:
        """Atomically snapshot (version, dict-reference) together under the
        lock, then release: because `_published` is only ever replaced by a
        brand-new dict (never mutated in place -- see publish()), the
        caller can safely iterate the returned list after the lock is
        released with the guarantee that every record in it belongs to
        exactly the returned version. Used by the Theorem 1 fuzz test to
        confirm every concurrent read corresponds to a single checkpoint.
        """
        with self._lock:
            v = self._version
            d = self._published
            recs = list(d.values())
        if subject is not None:
            recs = [r for r in recs if r.subject == subject]
        return v, recs

    @property
    def version(self) -> int:
        return self._version

    def checkpoint_at(self, version: int) -> Optional[Checkpoint]:
        for ck in self._checkpoints:
            if ck.version == version:
                return ck
        return None

    def all_staging(self) -> list[MemoryRecord]:
        with self._lock:
            return list(self._staging.values())

    def ingest_raw(self, record: MemoryRecord) -> None:
        """Live-time write path (AWAKE state): new content lands in staging
        (M_D/M_C territory), never directly in `published` (M_V) -- this is
        I6's "never written directly into M_V" clause applied uniformly to
        observed content as well as synthetic content.

        Always writes to the *published* copy's staging dict, under the
        lock, even while a sleep cycle has a private working copy open
        (I4): a write racing an in-progress cycle must never land in
        `private["staging"]`, which the sleep-stage algorithms are
        concurrently iterating and mutating unlocked. It is instead picked
        up by the next cycle's `begin_private_copy()` snapshot.
        """
        with self._lock:
            self._staging[record.id] = record

    # ---- durable snapshot boundary (SDK Phase D, additive) ---------------
    def snapshot(self) -> dict:
        """Returns a JSON-serializable, versioned projection of this store's
        COMMITTED state: the published (live-readable, M_V) set, the staging
        set, the store version, and the promotion-hash ledger that
        check_published_integrity() compares against.

        Raises RuntimeError if a private working copy is open. A snapshot
        taken mid-cycle would capture uncommitted maintenance work that
        WAKE_VERIFY has not yet accepted or discarded, so refusing is the
        only answer consistent with I4 -- the alternative would let a
        process restart resurrect a cycle that was never verified.

        This is a *contract*, not an internal dump: callers depend on
        schema_version, and the SDK's storage layer rejects a version it
        does not understand rather than guessing.
        """
        with self._lock:
            if self._private is not None:
                raise RuntimeError(
                    "cannot snapshot while a private working copy is open -- "
                    "finish or discard the in-progress maintenance cycle first (I4)"
                )
            return {
                "schema_version": MEMORY_SNAPSHOT_SCHEMA_VERSION,
                "version": self._version,
                "published": [_record_to_dict(r) for r in self._published.values()],
                "staging": [_record_to_dict(r) for r in self._staging.values()],
                "promotion_hashes": dict(self._promotion_hashes),
            }

    def restore(self, snapshot: dict) -> None:
        """Replaces this store's committed state with `snapshot`'s.

        Trust tiers and statuses are restored EXACTLY as they were saved: a
        record that was live-readable (T4_VERIFIED/trusted) before the
        restart is live-readable after it. This is the whole reason
        restore() exists as an explicit method instead of the SDK replaying
        records through ingest_raw(), which would have silently demoted
        every promoted record back to a T1 staging candidate.

        Checkpoint history is NOT restored (it is bounded in-process
        forensic state, not committed state). A fresh history is started at
        the restored version, so checkpoint_at(v) for a pre-restart v
        returns None -- by design, and asserted by the SDK's storage tests.

        Raises ValueError for an unknown schema version, and RuntimeError if
        a private working copy is open.
        """
        version = snapshot.get("schema_version")
        if version != MEMORY_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported memory snapshot schema_version {version!r} "
                f"(this build understands {MEMORY_SNAPSHOT_SCHEMA_VERSION})"
            )
        published = {}
        for d in snapshot.get("published", []):
            rec = _record_from_dict(d)
            published[rec.id] = rec
        staging = {}
        for d in snapshot.get("staging", []):
            rec = _record_from_dict(d)
            staging[rec.id] = rec

        with self._lock:
            if self._private is not None:
                raise RuntimeError(
                    "cannot restore while a private working copy is open -- "
                    "finish or discard the in-progress maintenance cycle first (I4)"
                )
            self._published = published
            self._staging = staging
            self._promotion_hashes = dict(snapshot.get("promotion_hashes", {}))
            self._version = int(snapshot.get("version", 0))
            self._checkpoints.clear()
            self._publish_ckpt(initial=True)

