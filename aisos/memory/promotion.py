"""
Algorithm A7 -- Promotion Pipeline (Trust-Tiered Memory Verification).

Preserves: I6 (Monotonic Trust Tiering); this is the mechanism Theorem 6
(Bounded Pollution) is proved about. Invoked from within WAKE_VERIFY
(Algorithm A6) before the checkpoint commit, once per pending candidate
rather than once per cycle -- a candidate that does not clear every check
in a given cycle simply stays at its current tier (or is rejected once it
exhausts `max_rounds`) and is re-evaluated on a subsequent cycle.

Cost: O(|M_D| + |M_C_pending|) per Wake-Up cycle, dominated by
CHECK_EXTERNAL_SOURCES, which is the most expensive check and is therefore
run only after the cheaper contradiction and confidence checks have already
filtered candidates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from aisos.memory.record import MemoryRecord, TrustTier, Status
from aisos.memory.contradiction import SOURCE_AUTHORITY
from aisos.security.wal import WriteAheadLog


@dataclass
class PromotionResult:
    promoted_to_verified: list[MemoryRecord]  # reached T4_VERIFIED == M_V this round
    advanced: list[MemoryRecord]              # tier bumped, still below M_V
    held: list[MemoryRecord]                  # unchanged this round, will retry
    rejected: list[MemoryRecord]              # moved to REJECTED == M_X
    external_checks_run: int
    supersedes: list[MemoryRecord] = field(default_factory=list)  # published ids to retire


def _authority_score(r: MemoryRecord) -> tuple:
    return (SOURCE_AUTHORITY.get(r.source, 0.4), float(r.trust_tier), r.updated_at)


def authority_score_without_authority(r: MemoryRecord) -> tuple:
    """Authority-ablation counterpart to _authority_score()
    (docs/authority_ablation_spec.md): identical tuple with the
    SOURCE_AUTHORITY element removed, trust_tier and updated_at retained
    in the same lexicographic priority. The project's own existing
    fallback signals, not an invented rule. Used only by
    aisos/agent/condition_backends.py's authority-ablated backend; the
    default behavior of promotion_pipeline() below is unchanged.
    """
    return (float(r.trust_tier), r.updated_at)


def authority_score_recency_only(r: MemoryRecord) -> tuple:
    """Trust-tier ablation counterpart (docs/mechanism_hypotheses.md
    H1/H2/H3), built on top of authority_score_without_authority: removes
    trust_tier too, leaving pure recency as the sole remaining element.
    This is what actually removes the incumbency effect the prior
    milestone's unit test identified (trust_tier lexicographically
    dominating an established T4 live record over a fresh T1 candidate
    regardless of recency) -- with this comparator, a candidate's recency
    alone decides against a live fact, tier difference notwithstanding.
    Used only by aisos/agent/condition_backends.py's trust-tier-ablated
    backend.
    """
    return (r.updated_at,)


def authority_score_trust_only(r: MemoryRecord) -> tuple:
    """Security-Retrieval-Freshness causal ablation
    (AISOS_v3_Claude_Security_Retrieval_Freshness_Novelty_Prompt.md
    Section 12, condition B): trust_tier alone, no authority, no
    recency tie-break."""
    return (float(r.trust_tier),)


def authority_score_authority_only(r: MemoryRecord) -> tuple:
    """Security-Retrieval-Freshness causal ablation, condition C:
    authority alone, no trust_tier, no recency tie-break."""
    return (SOURCE_AUTHORITY.get(r.source, 0.4),)


def authority_score_authority_and_trust(r: MemoryRecord) -> tuple:
    """Security-Retrieval-Freshness causal ablation, condition D:
    authority and trust_tier, no recency tie-break at all."""
    return (SOURCE_AUTHORITY.get(r.source, 0.4), float(r.trust_tier))


def verified_authority_score(r: MemoryRecord) -> tuple:
    """Trusted-authority-registry counterpart to _authority_score()
    (AISOS_Trusted_Authority_Provenance_Next_Milestone.md): identical
    tuple shape and priority order -- (authority, trust_tier, updated_at)
    -- but reads r.verified_authority (set once by AISOSAgent.remember()
    via aisos/security/authority_registry.py's TrustedAuthorityRegistry,
    never the caller's bare source-string claim) instead of
    SOURCE_AUTHORITY.get(r.source, 0.4). A record with
    verified_authority=None is treated as UNVERIFIED_AUTHORITY (0.0),
    the fail-safe default -- strictly below every registered source's
    authority, including 'dream' (0.1). This is what closes the
    self-declaration vulnerability documented in
    docs/authority_security_stress_test.md Test H: a candidate can no
    longer obtain privileged authority merely by claiming a high-authority
    source string with no verified credential. Used only by
    aisos/agent/condition_backends.py's trusted-authority backend; the
    default behavior of promotion_pipeline() below is unchanged.
    """
    from aisos.security.authority_registry import UNVERIFIED_AUTHORITY
    authority = r.verified_authority if r.verified_authority is not None else UNVERIFIED_AUTHORITY
    return (authority, float(r.trust_tier), r.updated_at)


def promotion_pipeline(
    private_staging: dict,
    published_snapshot: dict,
    wal: WriteAheadLog,
    check_external_sources: Optional[Callable[[MemoryRecord], bool]] = None,
    confidence_threshold: float = 0.55,
    max_rounds: int = 4,
    external_evidence_required_from: TrustTier = TrustTier.T2_VALIDATED,
    max_promotions_per_cycle: Optional[int] = 25,
    authority_score_fn: Callable[[MemoryRecord], tuple] = _authority_score,
) -> PromotionResult:
    """Mutates `private_staging` in place. `check_external_sources=None`
    models Experiment E5's ablation condition (c): external-evidence
    checking disabled -- any candidate that reaches the tier requiring it
    is auto-corroborated rather than checked, which is expected to *not*
    change the zero-pollution result (Theorem 6 depends only on I4+I6,
    not on the external-evidence check's accuracy) but *does* change
    promotion precision/recall, which is exactly what E5 measures.

    `max_promotions_per_cycle` throttles how many candidates may complete
    their *final* hop to T4_VERIFIED (M_V) in one Wake-Up cycle. This is
    what keeps a legitimate backlog (e.g. after several cycles of sparse
    scheduling) from arriving as one large simultaneous jump that would
    otherwise trip Wake-Up's drift-bound circuit breaker (Appendix B.3) and
    then, because a rejected cycle discards *all* of that cycle's
    progress, get stuck retrying the exact same oversized batch forever.
    Candidates beyond the cap simply are not advanced this round -- no
    validation-round penalty -- and are picked up on a subsequent cycle,
    prioritized by how close they already are to promotion (closest first),
    so the backlog drains at a bounded rate instead of deadlocking.
    """
    pending = [
        r for r in private_staging.values()
        if r.status == Status.CANDIDATE and r.trust_tier != TrustTier.REJECTED
    ]
    # Prioritize candidates closest to completion first, so a throttled
    # cycle still makes forward progress rather than round-robining evenly
    # across an ever-growing backlog.
    pending.sort(key=lambda r: (-int(r.trust_tier), -r.validation_rounds))

    promoted_to_verified: list[MemoryRecord] = []
    advanced: list[MemoryRecord] = []
    held: list[MemoryRecord] = []
    rejected: list[MemoryRecord] = []
    supersedes: list[MemoryRecord] = []
    external_checks_run = 0
    t4_this_round = 0
    supersede_target: dict[str, MemoryRecord] = {}  # r.id -> live record it would retire, if r reaches T4

    for r in pending:
        # --- cheap check 1: contradiction against the currently published
        # (M_V) view for this subject. Deep Sleep already resolves
        # contradictions *within* a cycle's candidates; this check handles
        # a candidate that contradicts what is already live -- e.g. a
        # correction that arrives (and gets promoted) *after* an earlier,
        # now-superseded fact already reached M_V. Resolution uses the same
        # authority+recency scoring as memory/contradiction.py (Technical
        # Spec Section 27) rather than a blanket "never override" rule,
        # which would otherwise let a low-authority fact that got promoted
        # early permanently block a later, better-sourced correction.
        conflicting_live = [
            o for o in published_snapshot.values()
            if o.subject == r.subject and o.value != r.value
        ]
        if conflicting_live:
            best_live = max(conflicting_live, key=authority_score_fn)
            if authority_score_fn(r) <= authority_score_fn(best_live):
                # The candidate does not outrank what's already live --
                # hold it (it may win later if its own confidence/authority
                # improves, or if the live fact is itself superseded).
                r2 = _hold_or_reject(r, private_staging, max_rounds, rejected, wal)
                held.append(r2) if r2.status != Status.REJECTED else None
                continue
            # Candidate outranks the live fact: it may proceed through the
            # remaining checks. It is only actually recorded as superseding
            # `best_live` if it goes on to reach T4_VERIFIED *this round*
            # (see the promotion block below) -- wake_verify then retires
            # `best_live` from `published` in the same atomic commit, so
            # there are never two live values for one subject at once.
            supersede_target[r.id] = best_live

        # --- cheap check 2: confidence threshold ---
        if r.confidence < confidence_threshold:
            r2 = _hold_or_reject(r, private_staging, max_rounds, rejected, wal)
            if r2.status != Status.REJECTED:
                held.append(r2)
            continue

        # --- expensive check 3: external corroboration, only once the
        # candidate is trying to advance past the required tier ---
        need_external = r.trust_tier >= external_evidence_required_from
        corroborated = True
        if need_external and check_external_sources is not None:
            external_checks_run += 1
            corroborated = check_external_sources(r)
        # check_external_sources is None -> ablation E5(c): auto-pass.

        if not corroborated:
            r2 = _hold_or_reject(r, private_staging, max_rounds, rejected, wal)
            if r2.status != Status.REJECTED:
                held.append(r2)
            continue

        # --- throttle: this candidate passed every check and is one hop
        # from T4_VERIFIED, but the per-cycle promotion budget is spent.
        # Leave it untouched (no round penalty) for a subsequent cycle.
        about_to_verify = (int(r.trust_tier) + 1) == int(TrustTier.T4_VERIFIED)
        if about_to_verify and max_promotions_per_cycle is not None and t4_this_round >= max_promotions_per_cycle:
            held.append(r)
            continue

        # --- all checks passed this round: advance exactly one tier ---
        new_tier_val = min(int(r.trust_tier) + 1, int(TrustTier.T4_VERIFIED))
        new_tier = TrustTier(new_tier_val)
        wal.append(op="promote", record_id=r.id, payload=f"{int(r.trust_tier)}->{new_tier_val}")

        if new_tier == TrustTier.T4_VERIFIED:
            t4_this_round += 1
            final = r.clone_bumped(
                trust_tier=new_tier, status=Status.TRUSTED, validation_rounds=r.validation_rounds + 1
            )
            private_staging[final.id] = final
            promoted_to_verified.append(final)
            # Separate from the "promote" entry above (whose payload is a
            # human-readable tier-transition string, not a hash, and is
            # left unchanged): this entry exists specifically so
            # wal.verify_record_integrity() (AISOS_Next_Milestone_Replay_
            # Provenance.md Phase J) has a genuine content_hash() to check
            # a freshly-promoted record against, logged at the exact
            # moment it reaches its final T4_VERIFIED state.
            wal.append(op="promotion_finalized", record_id=final.id, payload=final.content_hash())
            if r.id in supersede_target:
                supersedes.append(supersede_target[r.id])
        else:
            bumped = r.clone_bumped(trust_tier=new_tier, validation_rounds=r.validation_rounds + 1)
            private_staging[bumped.id] = bumped
            advanced.append(bumped)

    return PromotionResult(
        promoted_to_verified=promoted_to_verified,
        supersedes=supersedes,
        advanced=advanced,
        held=held,
        rejected=rejected,
        external_checks_run=external_checks_run,
    )


def _hold_or_reject(
    r: MemoryRecord, private_staging: dict, max_rounds: int, rejected: list, wal: WriteAheadLog
) -> MemoryRecord:
    if r.validation_rounds + 1 >= max_rounds:
        rej = r.clone_bumped(status=Status.REJECTED, trust_tier=TrustTier.REJECTED)
        private_staging[rej.id] = rej
        wal.append(op="promotion_reject", record_id=rej.id, payload=rej.content_hash())
        rejected.append(rej)
        return rej
    bumped = r.clone_bumped(validation_rounds=r.validation_rounds + 1)
    private_staging[bumped.id] = bumped
    wal.append(op="promotion_hold", record_id=bumped.id, payload=bumped.content_hash())
    return bumped
