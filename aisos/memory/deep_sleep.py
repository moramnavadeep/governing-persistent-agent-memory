"""
Algorithm A3 -- Deep Sleep: Long-Term Memory Consolidation and Pruning.

Preserves: I2, I4; the atomic-unit structuring here (one cluster = one
atomic unit) is what Theorem 3 (Bounded Preemption Latency) depends on for
the DEEP state.

Cost: dominated by CLUSTER_SIMILAR over the private copy's staging entries
plus newly summarized entries E, O(n log n) with an approximate index
rather than O(n^2) exhaustive pairwise comparison. This reference
implementation clusters by `subject` (an O(n) hash-partition, which is the
degenerate/fast case of "approximate nearest neighbor" -- exact semantic
clustering is exactly the pluggable slot v2 Section 1.2 says AISOS does not
claim novelty over).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from aisos.memory.record import MemoryRecord, TrustTier, Status
from aisos.memory.contradiction import find_contradictions, _score
from aisos.security.wal import WriteAheadLog


def dedup_priority(r: MemoryRecord) -> tuple:
    """Shared priority ordering for collapsing same-(subject,value)
    duplicate candidates: prefer higher verified_authority, then
    recency. Unset authority is treated as 0.0 (UNVERIFIED_AUTHORITY's
    own fail-safe value) -- a no-op under the legacy, non-trusted-
    authority path, where every record ties at 0.0 and recency alone
    decides. Extracted to module scope (Adaptive Candidate Backlog
    Control, AISOS_adaptive_candidate_backlog_next_milestone.md) so
    dream_phase.py's own same-cycle coalescing step uses the IDENTICAL
    logic this module already trusts, rather than a second, potentially
    divergent implementation.
    """
    va = r.verified_authority if r.verified_authority is not None else 0.0
    return (va, r.updated_at)


@dataclass
class DeepSleepResult:
    clusters_processed: int
    clusters_total: int
    preempted: bool
    deduped: int
    contradictions_resolved: int
    archived: int
    consolidated: list[MemoryRecord]


def deep_sleep(
    private_staging: dict,
    wal: WriteAheadLog,
    should_preempt: Optional[Callable[[], bool]] = None,
    max_versions_per_subject: int = 3,
    contradiction_score_fn: Callable[[MemoryRecord], float] = _score,
) -> DeepSleepResult:
    """Mutates `private_staging` in place (I4: this is always the private
    copy, never the published dict). One "cluster" (all candidate records
    sharing a subject) is one atomic unit: it is either fully processed
    (dedup + contradiction-resolve + prune) or left untouched if a preempt
    is observed before it starts.

    contradiction_score_fn: passed through to find_contradictions();
    defaults to the real authority-aware _score(). Overridden only by
    aisos/agent/condition_backends.py's authority-ablated backend (see
    docs/authority_ablation_spec.md) -- every other caller is unaffected.
    """
    by_subject: dict[str, list[MemoryRecord]] = {}
    for rec in private_staging.values():
        if rec.status == Status.CANDIDATE:
            by_subject.setdefault(rec.subject, []).append(rec)

    clusters = list(by_subject.items())
    deduped = 0
    archived = 0
    contradictions_resolved = 0
    consolidated: list[MemoryRecord] = []
    processed = 0
    preempted = False

    for subject, recs in clusters:
        if should_preempt is not None and should_preempt():
            preempted = True
            break

        # --- deduplicate: identical (subject, value) collapse to one
        # record; every other id sharing that value is dropped from the
        # private copy and counted as deduped.
        #
        # SECURITY FINDING (AISOS_Comprehensive_Security_Audit_Next_
        # Milestone.md Phase P), fixed here: an earlier version of this
        # tie-break used pure recency (`r.updated_at > existing.updated_at`),
        # which meant two same-value candidates with DIFFERENT
        # verified_authority (e.g. a genuine high-authority record and a
        # coincidentally-identical-value, lower-authority one) resolved
        # purely by which was more recent -- silently discarding the
        # HIGHER-authority record's tracked authority in favor of a lower
        # one, purely because it happened to arrive later. Fixed by
        # preferring the higher verified_authority first, falling back to
        # recency only when authority ties (including the legacy case
        # where verified_authority is None for both -- unset is treated as
        # 0.0, matching UNVERIFIED_AUTHORITY's own fail-safe value, so
        # this is a NO-OP under the legacy, non-trusted-authority path:
        # every record ties at 0.0 there and recency alone decides,
        # exactly as before).
        seen_values: dict[str, MemoryRecord] = {}
        for r in recs:
            existing = seen_values.get(r.value)
            if existing is None:
                seen_values[r.value] = r
                continue
            newer, older = (r, existing) if dedup_priority(r) > dedup_priority(existing) else (existing, r)
            seen_values[r.value] = newer
            private_staging.pop(older.id, None)
            wal.append(op="deep_sleep_dedup_drop", record_id=older.id, payload=older.content_hash())
            deduped += 1
        remaining = list(seen_values.values())

        # --- contradiction detection + resolution ---
        contradictions = find_contradictions(remaining, score_fn=contradiction_score_fn)
        archived_ids: set[str] = set()
        for c in contradictions:
            for loser in c.losers:
                arch = loser.clone_bumped(status=Status.ARCHIVED)
                private_staging[loser.id] = arch
                wal.append(op="deep_sleep_archive_contradiction", record_id=arch.id, payload=arch.content_hash())
                archived_ids.add(loser.id)
                archived += 1
            contradictions_resolved += 1
        # Filter by id, not by `.status`: `clone_bumped` returns a new object
        # rather than mutating `loser` in place, so the original objects
        # still in `remaining` never observe the ARCHIVED status above.
        remaining = [r for r in remaining if r.id not in archived_ids]

        # --- prune: cap versions retained per subject, oldest first ---
        remaining.sort(key=lambda r: r.updated_at, reverse=True)
        keep, prune = remaining[:max_versions_per_subject], remaining[max_versions_per_subject:]
        for r in prune:
            arch = r.clone_bumped(status=Status.ARCHIVED)
            private_staging[r.id] = arch
            wal.append(op="deep_sleep_archive_prune", record_id=arch.id, payload=arch.content_hash())
            archived += 1

        for r in keep:
            wal.append(op="deep_sleep_consolidate", record_id=r.id, payload=r.content_hash())
            consolidated.append(r)

        processed += 1

    return DeepSleepResult(
        clusters_processed=processed,
        clusters_total=len(clusters),
        preempted=preempted,
        deduped=deduped,
        contradictions_resolved=contradictions_resolved,
        archived=archived,
        consolidated=consolidated,
    )
