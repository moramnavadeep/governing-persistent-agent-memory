"""
Algorithm A2 -- Light Sleep: Short-Term Context Consolidation.

Preserves: I2 (Write-Before-Log) via checkpointing before mutation; I4 via
private snapshot.

Cost: O(|C_t|) for segmentation and summarization, dominated by the number
of raw context entries in the active window. Step 4 checks for a wake
request before each segment rather than once per full context, so a single
segment is the atomic unit of preemption (this is what lets the Interrupt
Controller bound dispatch latency at the segment granularity, per Theorem 3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from aisos.memory.record import MemoryRecord, TrustTier, Status
from aisos.security.wal import WriteAheadLog


@dataclass
class LightSleepResult:
    segments_processed: int
    segments_total: int
    preempted: bool
    new_candidates: list[MemoryRecord]


def _segment(context_entries: list[dict], segment_size: int = 8) -> list[list[dict]]:
    return [context_entries[i:i + segment_size] for i in range(0, len(context_entries), segment_size)]


def light_sleep(
    private_staging: dict,
    context_entries: list[dict],
    wal: WriteAheadLog,
    should_preempt: Optional[Callable[[], bool]] = None,
    segment_size: int = 8,
) -> LightSleepResult:
    """
    private_staging: the `staging` dict of the store's *private* copy
                      (caller must have already called store.begin_private_copy()
                      -- I4).
    context_entries: raw short-term context C_t, list of {"subject","value","source",...}
    wal:              write-ahead log; every produced summary is logged
                      (I2) before it is inserted into the private copy.
    should_preempt:   polled once per segment; if it returns True, the
                      function stops and returns what it has done so far --
                      the segment is the atomic unit (nothing partial is
                      written for the segment in progress).
    """
    segments = _segment(context_entries, segment_size)
    new_candidates: list[MemoryRecord] = []
    preempted = False
    processed = 0

    for seg in segments:
        if should_preempt is not None and should_preempt():
            preempted = True
            break

        # "Summarization": symbolic reference implementation -- group the
        # segment's entries by subject and keep the most recent value per
        # subject as the segment's summary candidate(s). A real deployment
        # plugs in an LLM-based summarizer here (v2 Section 1.2: AISOS
        # orchestrates pluggable consolidation, it does not claim a novel one).
        by_subject: dict[str, dict] = {}
        for entry in seg:
            subj = entry["subject"]
            if subj not in by_subject or entry.get("ts", 0) >= by_subject[subj].get("ts", 0):
                by_subject[subj] = entry

        for subj, entry in by_subject.items():
            rec = MemoryRecord(
                subject=subj,
                value=entry["value"],
                source=entry.get("source", "user"),
                trust_tier=TrustTier.T1_CANDIDATE,
                status=Status.CANDIDATE,
                confidence=float(entry.get("confidence", 0.6)),
                provenance={"origin": "light_sleep", **entry.get("provenance", {})},
                # Preserves the trusted-authority mechanism's verified
                # claim across the context-buffer round-trip -- without
                # this, this reconstruction step silently drops
                # verified_authority (found during the trusted-authority
                # milestone's lifecycle-integrity check: dedup then keeps
                # THIS unverified duplicate over the original verified
                # record, since it has a more recent updated_at).
                verified_authority=entry.get("verified_authority"),
            )
            # I2: WAL entry appended before the record enters the private copy.
            wal.append(op="light_sleep_summarize", record_id=rec.id, payload=rec.content_hash())
            private_staging[rec.id] = rec
            new_candidates.append(rec)

        processed += 1

    return LightSleepResult(
        segments_processed=processed,
        segments_total=len(segments),
        preempted=preempted,
        new_candidates=new_candidates,
    )
