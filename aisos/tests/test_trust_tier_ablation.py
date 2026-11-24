"""
Required tests for the trust-tier ablation (H1/H2/H3;
AISOS_Complete_Next_Research_Engineering_Master_Prompt.md Section 13,
docs/mechanism_hypotheses.md) -- built and run BEFORE the expensive
5x4x1000 experiment, mirroring test_authority_ablation.py's pattern.

The trust-tier ablation is built ON TOP OF the already-authority-ablated
comparator (docs/mechanism_hypotheses.md "Ablation Definition"), so its
tests are phrased relative to AuthorityAblatedPeriodicAISOSBackend, not the
authority-intact baseline: the key empirical claim is that once trust_tier
is ALSO removed, pure recency decides -- meaning the prior milestone's
incumbency-confound result (a fresh low-trust-tier candidate is blocked
regardless of recency) must FLIP to the candidate winning, since recency is
now the only remaining signal.
"""
from __future__ import annotations

from aisos.memory.record import MemoryRecord, TrustTier, Status
from aisos.memory.contradiction import find_contradictions, score_without_authority, score_recency_only
from aisos.memory.promotion import promotion_pipeline, authority_score_without_authority, authority_score_recency_only
from aisos.security.wal import WriteAheadLog
from aisos.agent.condition_backends import (
    PeriodicAISOSBackend,
    AuthorityAblatedPeriodicAISOSBackend,
    TrustTierAblatedPeriodicAISOSBackend,
    make_backend,
)
from aisos.agent.workload_1000 import build_workload, workload_hash


def _rec(subject, value, source, trust_tier, updated_at, confidence=0.9):
    r = MemoryRecord(subject=subject, value=value, source=source, confidence=confidence,
                      trust_tier=trust_tier, status=Status.CANDIDATE)
    r.updated_at = updated_at
    return r


# ---------------------------------------------------------------------------
# Test 1: the flip -- once trust_tier is ALSO removed, the fresh (T1) low-
# trust-tier candidate WINS the promotion gate against an established live
# T4 record, purely because it is more recent. This is the exact opposite
# outcome of test_authority_ablation.py's incumbency-confound test (which
# used the same live/candidate shape but with trust_tier still active), and
# is the direct empirical test of H1/H2's falsification condition.
# ---------------------------------------------------------------------------
def _run_until_settled(record, published_snapshot, authority_score_fn, max_calls=6):
    """promotion_pipeline() advances a passing candidate exactly ONE trust
    tier per call (per-cycle, one Wake-Up round at a time -- see
    promotion.py's docstring), so a T1_CANDIDATE needs three successful
    gate-passes (T1->T2->T3->T4) before it can appear in
    promoted_to_verified. Repeatedly re-submits the SAME staging dict (which
    promotion_pipeline mutates in place, preserving the record's id across
    clone_bumped calls) until it is promoted, rejected, or max_calls is
    exhausted (which would mean it is stuck being held every round).
    """
    staging = {record.id: record}
    result = None
    for _ in range(max_calls):
        result = promotion_pipeline(
            staging, published_snapshot, WriteAheadLog(),
            authority_score_fn=authority_score_fn, confidence_threshold=0.5,
        )
        if result.promoted_to_verified or result.rejected:
            return result
    return result


def test_1_recency_only_gate_lets_fresh_candidate_beat_established_live_record():
    live = _rec("x", "TRUE", "user", TrustTier.T4_VERIFIED, updated_at=100.0)
    live.status = Status.TRUSTED
    published_snapshot = {live.id: live}

    # Reconfirm the prior milestone's result still holds with trust_tier active
    # (authority removed, trust_tier intact): the candidate must never reach
    # T4_VERIFIED no matter how many rounds it is resubmitted, because the
    # gate holds it (never advancing its tier) every single round.
    candidate_a = _rec("x", "FALSE", "tool", TrustTier.T1_CANDIDATE, updated_at=99999.0)  # far more recent
    result_authority_ablated = _run_until_settled(
        candidate_a, published_snapshot, authority_score_without_authority,
    )
    assert result_authority_ablated.promoted_to_verified == [], (
        "sanity check: with trust_tier still active, the incumbency confound must still block "
        "the fresh candidate from ever reaching T4_VERIFIED, exactly as the prior milestone found"
    )

    # Now ALSO remove trust_tier: recency alone decides the gate every round,
    # so the far-more-recent candidate must climb all the way to T4_VERIFIED
    # despite starting at T1 against an established T4 live record.
    candidate_b = _rec("x", "FALSE", "tool", TrustTier.T1_CANDIDATE, updated_at=99999.0)
    result_recency_only = _run_until_settled(
        candidate_b, published_snapshot, authority_score_recency_only,
    )
    assert len(result_recency_only.promoted_to_verified) == 1, (
        "H1/H2's falsification condition: once trust_tier is removed from the gate in addition to "
        "authority, a fresh low-trust-tier candidate must eventually beat an established live "
        "record purely on recency -- confirming the incumbency effect isolated in the prior "
        "milestone was really caused by trust_tier specifically, not some other property of the "
        "live record"
    )
    assert result_recency_only.promoted_to_verified[0].value == "FALSE"


# ---------------------------------------------------------------------------
# Test 2: trust_tier is genuinely absent from the recency-only decision --
# varying trust_tier alone (recency held fixed) must not change the outcome.
# ---------------------------------------------------------------------------
def test_2_recency_only_decision_invariant_to_trust_tier():
    a = _rec("x", "A", "user", TrustTier.T4_VERIFIED, updated_at=100.0)   # high tier, older
    b = _rec("x", "B", "tool", TrustTier.T1_CANDIDATE, updated_at=200.0)  # low tier, newer
    winner_1 = find_contradictions([a, b], score_fn=score_recency_only)[0].winner.value

    a2 = _rec("x", "A", "user", TrustTier.T1_CANDIDATE, updated_at=100.0)  # tiers swapped
    b2 = _rec("x", "B", "tool", TrustTier.T4_VERIFIED, updated_at=200.0)
    winner_2 = find_contradictions([a2, b2], score_fn=score_recency_only)[0].winner.value

    assert winner_1 == winner_2 == "B", (
        "the recency-only scorer's decision must depend only on updated_at, not on which record "
        "has the higher trust_tier -- swapping tiers while keeping recency fixed must not change "
        "the winner"
    )


# ---------------------------------------------------------------------------
# Test 3: recency still functions as the sole active signal (not simply
# broken / always-tie).
# ---------------------------------------------------------------------------
def test_3_recency_still_functions_as_sole_signal():
    older = _rec("x", "OLD", "user", TrustTier.T4_VERIFIED, updated_at=100.0)
    newer = _rec("x", "NEW", "user", TrustTier.T4_VERIFIED, updated_at=999.0)
    winner = find_contradictions([older, newer], score_fn=score_recency_only)[0].winner
    assert winner.value == "NEW"


# ---------------------------------------------------------------------------
# Test 4: sleep/consolidation lifecycle equivalence (mirrors Test E of the
# authority ablation) -- H5's falsification condition depends on cycle
# count/lifecycle states being identical, so consolidation frequency is
# isolated from candidate-selection logic.
# ---------------------------------------------------------------------------
def test_4_trust_tier_ablated_backend_enters_same_lifecycle_states_as_normal():
    specs, ledger = build_workload(seed=0)
    small = specs[:120]

    normal = PeriodicAISOSBackend()
    ablated = TrustTierAblatedPeriodicAISOSBackend()
    for backend in (normal, ablated):
        for spec in small:
            if spec.context_reset_before:
                backend.apply_context_reset()
            if spec.kind == "teach":
                backend.write(spec.subject, spec.value, spec.source, spec.confidence, spec.interaction_id)
            backend.maybe_maintain(spec.interaction_id)

    assert normal.snapshot_metrics()["cycles_run"] == ablated.snapshot_metrics()["cycles_run"], (
        "the trust-tier-ablated backend must run exactly as many sleep cycles as the normal one, "
        "on the identical interaction sequence -- H5 predicts consolidation frequency itself is "
        "unaffected by this ablation"
    )
    normal_states = {t["to_state"] for t in normal.lifecycle_trace()}
    ablated_states = {t["to_state"] for t in ablated.lifecycle_trace()}
    assert normal_states == ablated_states


# ---------------------------------------------------------------------------
# Test 5: workload equality across all 5 registered conditions (via the
# shared hash mechanism) -- mirrors Test F of the authority ablation,
# extended to include the new condition.
# ---------------------------------------------------------------------------
def test_5_workload_hash_identical_across_all_five_conditions():
    specs, _ = build_workload(seed=0)
    h = workload_hash(specs)
    for condition in [
        "no_sleep_naive", "continuous_mvcc", "periodic_aisos",
        "periodic_aisos_without_authority", "periodic_aisos_without_trust_tier",
    ]:
        backend = make_backend(condition)
        for spec in specs[:30]:
            if spec.kind == "teach":
                backend.write(spec.subject, spec.value, spec.source, spec.confidence, spec.interaction_id)
        assert workload_hash(specs) == h, f"condition {condition!r} must not mutate the shared workload"


# ---------------------------------------------------------------------------
# Test 6: registry / wiring sanity -- make_backend() returns the correct
# class and it actually carries the recency-only functions through to the
# underlying AISOSAgent (not silently falling back to the defaults, which
# is exactly the failure mode the super()-call bug fixed in this milestone
# would have caused if left unfixed: a TypeError on construction).
# ---------------------------------------------------------------------------
def test_6_registry_wires_recency_only_functions_into_agent():
    backend = make_backend("periodic_aisos_without_trust_tier")
    assert isinstance(backend, TrustTierAblatedPeriodicAISOSBackend)
    assert backend.agent.contradiction_score_fn is score_recency_only
    assert backend.agent.authority_score_fn is authority_score_recency_only

    baseline = make_backend("periodic_aisos")
    assert baseline.agent.contradiction_score_fn is not score_recency_only
    assert baseline.agent.authority_score_fn is not authority_score_recency_only
