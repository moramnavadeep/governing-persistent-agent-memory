"""
Algorithm A6 -- Wake-Up: Integrity Verification.

Preserves: I3 (Monotonic Trust); this is the single atomic commit point
Theorem 1 (No Torn Reads) is proved against.

Cost: O(|wal_entries|) for signature/chain verification plus O(|M_LT'|)
for contradiction scanning, run once per sleep cycle rather than per
request -- the single serialization point in the entire lifecycle, and by
I1 (Single Writer) it is guaranteed no other writer is active while this
runs (the Lifecycle's RLock is held by the same thread driving the cycle).

Order of checks (cheapest/most-disqualifying first, mirroring A7's own
ordering rationale):
    1. WAL chain integrity (I2 witness)
    2. HMAC signature over the WAL bundle (Appendix B's threat model:
       detects a maintenance-time write that didn't go through the
       Adapter API's committed path)
    3. Promotion pipeline (Algorithm A7) -- may raise some candidates to
       M_V; this is the only place M_V ever gains members (I6)
    4. Contradiction scan over what is about to become the new published
       set (defends against a promotion that would create a live
       contradiction)
    5. Drift bound -- caps how much of the store may change in one cycle,
       and how far theta may move, as a coarse poisoning circuit-breaker
       (Appendix B.3)

Any failure -> reject. Rejection and PREEMPT_NOW-triggered abort both take
the same path: discard the private working copy (O(1), Theorem 2) and
report a failure; `published` is left untouched either way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from aisos.memory.record import MemoryRecord, Status, TrustTier
from aisos.memory.contradiction import find_contradictions, _score
from aisos.memory.promotion import promotion_pipeline, PromotionResult, _authority_score
from aisos.memory.store import MemoryStore
from aisos.security.wal import WriteAheadLog
from aisos.security.checkpoint import sign, verify_sig, checkpoint_hash


@dataclass
class WakeVerifyResult:
    passed: bool
    reason: str = "ok"
    promoted: int = 0
    new_version: Optional[int] = None
    promotion: Optional[PromotionResult] = None
    checks: dict = field(default_factory=dict)


def wake_verify(
    store: MemoryStore,
    wal: WriteAheadLog,
    signing_key: bytes,
    *,
    aborted: bool = False,
    tamper_signature: bool = False,
    check_external_sources: Optional[Callable[[MemoryRecord], bool]] = None,
    consistency_tolerance: int = 0,
    max_allowed_drift: float = 0.6,
    drift_baseline_min: int = 20,
    max_promotions_per_cycle: Optional[int] = 25,
    confidence_threshold: float = 0.55,
    max_promotion_rounds: int = 4,
    contradiction_score_fn: Callable[[MemoryRecord], float] = _score,
    authority_score_fn: Callable[[MemoryRecord], tuple] = _authority_score,
    verify_provenance_integrity: bool = False,
    verify_continuous_integrity: bool = False,
) -> WakeVerifyResult:
    """
    contradiction_score_fn / authority_score_fn: passed through to
    find_contradictions() (step 4's safety scan) and promotion_pipeline()
    (the candidate-vs-live gate) respectively. Both default to the real
    authority-aware scorers; overridden only by aisos/agent/
    condition_backends.py's authority-ablated backend
    (docs/authority_ablation_spec.md). Every other caller unaffected.
    aborted:          True on the PREEMPT_NOW path (P0/P1). Skips the
                       (more expensive) checks entirely and rolls back --
                       this is precisely what keeps rollback O(1)
                       regardless of how much of the cycle had run
                       (Theorem 2): we never pay for verification work on
                       an aborted cycle.
    tamper_signature:  test hook for Experiment E3 (adversarial robustness)
                       -- simulates a maintenance-time write that bypassed
                       the signed commit path.
    consistency_tolerance: number of live contradictions tolerated before
                       rejecting (0 = strict, matching Theorem 1's claim).
    max_allowed_drift: fraction of max(drift_baseline_min, |published|)
                       that may newly become live-readable in a single
                       cycle before the drift breaker rejects (a coarse
                       defense described in Appendix B.3; it does *not*
                       catch a poisoning attempt that stays within
                       tolerance -- documented there as a limitation, not
                       hidden here).
    drift_baseline_min: floor on the denominator above, so an
                       (almost-)empty store isn't spuriously drift-limited
                       during cold start -- e.g. a store with 0-19 live
                       records is judged against a baseline of
                       drift_baseline_min, not against its own tiny size.
    verify_provenance_integrity: opt-in (AISOS_Next_Milestone_Replay_
                       Provenance.md Phase J), default False (fully
                       backward compatible). When True, every record
                       about to be promoted this cycle is re-checked via
                       wal.verify_record_integrity() -- its live
                       content_hash() must match the most recent WAL
                       entry logged for its id. Any mismatch (a record
                       mutated after being logged, without going through
                       a re-logging code path) rejects and rolls back the
                       WHOLE cycle, exactly like the other integrity
                       checks above -- this is the "reject/rollback/
                       quarantine" branch of that milestone's Phase J
                       diagram; there is no separate quarantine state in
                       this codebase, so rollback (discarding the private
                       copy, leaving `published` untouched) IS the
                       quarantine action.
    verify_continuous_integrity: opt-in (AISOS_Security_Closure_Milestone.md
                       Phase K/L), default False. When True, every
                       record ALREADY published (from before this cycle)
                       is re-checked against store.check_published_integrity()'s
                       durable, cross-cycle hash reference -- closing the
                       gap verify_provenance_integrity leaves open (a
                       record tampered in a LATER cycle than the one that
                       promoted it). Independent flag from
                       verify_provenance_integrity since this is an
                       O(|published|) scan, not O(1) per newly-promoted
                       record.
    """
    if aborted:
        store.discard_private_copy()
        wal.reset()
        return WakeVerifyResult(passed=False, reason="preempted")

    checks: dict = {}

    # 1. WAL chain integrity
    chain_ok = wal.is_chain_intact()
    checks["wal_chain_intact"] = chain_ok
    if not chain_ok:
        store.discard_private_copy()
        wal.reset()
        return WakeVerifyResult(passed=False, reason="wal_chain_broken", checks=checks)

    # 2. Signature over the WAL bundle
    bundle = "|".join(e.entry_hash for e in wal.entries())
    signature = sign(signing_key, bundle)
    if tamper_signature:
        signature = signature[:-1] + ("0" if signature[-1] != "0" else "1")
    sig_ok = verify_sig(signing_key, bundle, signature)
    checks["signature_valid"] = sig_ok
    if not sig_ok:
        store.discard_private_copy()
        wal.reset()
        return WakeVerifyResult(passed=False, reason="signature_invalid", checks=checks)

    private = store.private()
    published_before = private["published"]
    staging = private["staging"]

    # 3. Promotion pipeline (Algorithm A7) -- the only path into M_V (I6).
    # The per-cycle promotion cap is derived *from* the drift bound itself
    # (rather than configured independently) so the pipeline never hands
    # back a batch that step 5 would then have to reject wholesale: a
    # legitimate backlog (e.g. after several cycles of sparse scheduling)
    # is throttled to the drift-allowed rate instead of being computed in
    # full and then discarded every cycle, which would otherwise deadlock
    # (the same oversized batch recomputing identically and failing
    # identically forever).
    drift_baseline = max(drift_baseline_min, len(published_before))
    drift_derived_cap = max(1, int(max_allowed_drift * drift_baseline))
    effective_cap = drift_derived_cap if max_promotions_per_cycle is None else min(drift_derived_cap, max_promotions_per_cycle)

    promo = promotion_pipeline(
        staging, published_before, wal, check_external_sources=check_external_sources,
        max_promotions_per_cycle=effective_cap,
        confidence_threshold=confidence_threshold, max_rounds=max_promotion_rounds,
        authority_score_fn=authority_score_fn,
    )
    checks["promoted_this_round"] = len(promo.promoted_to_verified)
    checks["promotion_cap_this_cycle"] = effective_cap

    # Build the candidate new published set: retire anything the Promotion
    # Pipeline decided a newly-promoted, higher-authority record supersedes
    # (memory/promotion.py's supersede_target mechanism) *before* adding
    # the new arrivals, so the set handed to the contradiction scan below
    # never legitimately contains two live values for the same subject.
    new_published = dict(published_before)
    for old in promo.supersedes:
        new_published.pop(old.id, None)
    for rec in promo.promoted_to_verified:
        new_published[rec.id] = rec
    checks["superseded_this_round"] = len(promo.supersedes)

    # 4. Contradiction scan over the candidate new published set
    contradictions = find_contradictions(list(new_published.values()), score_fn=contradiction_score_fn)
    checks["live_contradictions"] = len(contradictions)
    if len(contradictions) > consistency_tolerance:
        store.discard_private_copy()
        wal.reset()
        return WakeVerifyResult(
            passed=False, reason="contradiction_would_go_live", promotion=promo, checks=checks
        )

    # 5. Drift bound -- coarse poisoning circuit-breaker (Appendix B.3).
    # With promotion already capped to drift_derived_cap above, this should
    # never legitimately fire; it remains as defense-in-depth against any
    # other path that might add entries to new_published (there is none in
    # this reference implementation, but Appendix B.2 does not assume the
    # Adapter API's privileged calls are misuse-proof).
    changed = len(promo.promoted_to_verified)
    drift = changed / drift_baseline
    checks["drift"] = drift
    if drift > max_allowed_drift:
        store.discard_private_copy()
        wal.reset()
        return WakeVerifyResult(
            passed=False, reason="drift_bound_exceeded", promotion=promo, checks=checks
        )

    # 6. Active provenance-integrity re-verification (opt-in, Phase J).
    # Covers only records newly promoted THIS cycle -- the smallest set
    # that matters for this specific check, rather than re-checking the
    # entire store every cycle.
    if verify_provenance_integrity:
        tampered = [r for r in promo.promoted_to_verified if not wal.verify_record_integrity(r)]
        checks["provenance_integrity_verified"] = len(tampered) == 0
        if tampered:
            store.discard_private_copy()
            wal.reset()
            return WakeVerifyResult(
                passed=False, reason="provenance_integrity_failed", promotion=promo, checks=checks
            )

    # 7. Continuous (post-publication) provenance-integrity re-verification
    # (opt-in, AISOS_Security_Closure_Milestone.md Phase K/L -- closes the
    # gap step 6 above does NOT cover: an ALREADY-published record,
    # tampered in a LATER cycle than the one that promoted it). Checks
    # every record CURRENTLY published (from before this cycle's own
    # changes) against store.check_published_integrity()'s durable,
    # cross-cycle hash reference -- unlike the WAL, which resets every
    # cycle and cannot answer this question. A separate opt-in flag from
    # step 6 (Phase L: "choose the smallest mechanism... without creating
    # unnecessary cost") since this is an O(|published|) scan, not O(1)
    # per newly-promoted record -- a caller who only needs step 6's
    # cheaper guarantee is not forced to pay for this one.
    if verify_continuous_integrity:
        tampered_ids = store.check_published_integrity()
        checks["continuous_integrity_verified"] = len(tampered_ids) == 0
        if tampered_ids:
            store.discard_private_copy()
            wal.reset()
            return WakeVerifyResult(
                passed=False, reason="continuous_provenance_integrity_failed", promotion=promo, checks=checks
            )

    # Record this cycle's promotions' hashes for FUTURE cycles' continuous
    # checks (step 7 above) -- unconditional, O(1) per promoted record, so
    # enabling verify_continuous_integrity later still has history to
    # compare against. Done only once the cycle is guaranteed to commit
    # (every check above has already passed), so a rejected cycle never
    # records hashes for records that never actually go live.
    for rec in promo.promoted_to_verified:
        store.record_promotion_hash(rec.id, rec.content_hash())

    # All checks passed: single atomic commit point (I3).
    private["published"] = new_published
    new_version = store.publish()
    wal.reset()
    return WakeVerifyResult(
        passed=True, promoted=len(promo.promoted_to_verified), new_version=new_version,
        promotion=promo, checks=checks,
    )
