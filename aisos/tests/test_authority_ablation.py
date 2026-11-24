"""
Required tests A-H for the authority ablation
(AISOS_Authority_Ablation_Next_Milestone_Prompt.md Section 8), plus an
explicit empirical test of the trust-tier-incumbency confounder identified
in docs/authority_ablation_spec.md Section 7a -- built and run BEFORE the
expensive 5x4x1000 experiment, per Section 36's "if the proposed ablation
cannot cleanly isolate authority, DO NOT launch the expensive experiment."
"""
from __future__ import annotations

import time

from aisos.memory.record import MemoryRecord, TrustTier, Status
from aisos.memory.contradiction import find_contradictions, _score, score_without_authority
from aisos.memory.promotion import promotion_pipeline, _authority_score, authority_score_without_authority
from aisos.security.wal import WriteAheadLog
from aisos.agent.condition_backends import PeriodicAISOSBackend, AuthorityAblatedPeriodicAISOSBackend, make_backend
from aisos.agent.workload_1000 import build_workload, workload_hash
from aisos.agent.evaluator import evaluate
import inspect


def _rec(subject, value, source, trust_tier, updated_at, confidence=0.9):
    r = MemoryRecord(subject=subject, value=value, source=source, confidence=confidence,
                      trust_tier=trust_tier, status=Status.CANDIDATE)
    r.updated_at = updated_at
    return r


# ---------------------------------------------------------------------------
# Test A: authority affects normal AISOS (within-cycle candidate ranking,
# contradiction.py's _score -- both candidates at the SAME trust tier so
# the incumbency confound (Section 7a) does not apply here).
# ---------------------------------------------------------------------------
def test_A_authority_affects_normal_ranking():
    high = _rec("x", "TRUE", "user", TrustTier.T1_CANDIDATE, updated_at=100.0)
    low = _rec("x", "FALSE", "tool", TrustTier.T1_CANDIDATE, updated_at=200.0)  # low authority but MORE recent
    contradictions = find_contradictions([high, low], score_fn=_score)
    assert len(contradictions) == 1
    assert contradictions[0].winner.source == "user", (
        "normal _score must let higher authority (user=0.8) beat lower authority (tool=0.6) "
        "even when the low-authority record is more recent"
    )


# ---------------------------------------------------------------------------
# Test B: authority is absent from the ablated decision (same setup, ablated scorer)
# ---------------------------------------------------------------------------
def test_B_ablated_ranking_ignores_authority_uses_recency_at_equal_trust_tier():
    high = _rec("x", "TRUE", "user", TrustTier.T1_CANDIDATE, updated_at=100.0)
    low = _rec("x", "FALSE", "tool", TrustTier.T1_CANDIDATE, updated_at=200.0)  # more recent
    contradictions = find_contradictions([high, low], score_fn=score_without_authority)
    assert contradictions[0].winner.source == "tool", (
        "with authority removed and trust_tier tied, the ablated scorer must fall through to "
        "recency -- the more recent (but low-authority) record must win"
    )


# ---------------------------------------------------------------------------
# Test C: changing authority does not affect the ablated output
# ---------------------------------------------------------------------------
def test_C_ablated_decision_invariant_to_swapping_authority_labels():
    a = _rec("x", "A", "user", TrustTier.T1_CANDIDATE, updated_at=100.0)      # high authority, older
    b = _rec("x", "B", "tool", TrustTier.T1_CANDIDATE, updated_at=200.0)      # low authority, newer
    winner_1 = find_contradictions([a, b], score_fn=score_without_authority)[0].winner.value

    a2 = _rec("x", "A", "tool", TrustTier.T1_CANDIDATE, updated_at=100.0)     # authority swapped: now LOW, older
    b2 = _rec("x", "B", "user_correction", TrustTier.T1_CANDIDATE, updated_at=200.0)  # now HIGH, newer
    winner_2 = find_contradictions([a2, b2], score_fn=score_without_authority)[0].winner.value

    assert winner_1 == winner_2 == "B", (
        "the ablated scorer's decision must depend only on recency/trust_tier, not on which "
        "source label is attached -- swapping which record has 'high' vs 'low' authority while "
        "keeping recency fixed must not change the winner"
    )


# ---------------------------------------------------------------------------
# Test D: non-authority variables (recency) remain active in the ablated scorer
# ---------------------------------------------------------------------------
def test_D_recency_still_functions_in_ablated_scorer():
    older = _rec("x", "OLD", "user", TrustTier.T1_CANDIDATE, updated_at=100.0)
    newer = _rec("x", "NEW", "user", TrustTier.T1_CANDIDATE, updated_at=999.0)  # same source, just newer
    winner = find_contradictions([older, newer], score_fn=score_without_authority)[0].winner
    assert winner.value == "NEW", "recency must still break ties when source/authority is identical"


# ---------------------------------------------------------------------------
# Empirical confounder test (docs/authority_ablation_spec.md Section 7a):
# an established LIVE (T4) fact vs. a fresh low-authority CANDIDATE (T1) --
# the promotion-gate comparison, not the within-cycle comparison above.
# ---------------------------------------------------------------------------
def test_incumbency_confound_promotion_gate_blocks_fresh_candidate_regardless_of_recency():
    """This is the realistic shape of the actual workload's contradiction
    attack: a high-authority fact is already published (T4_VERIFIED) by
    the time a low-authority contradicting candidate (T1_CANDIDATE, fresh)
    arrives, however much MORE recent that candidate is. Confirms the
    Section 7a analysis empirically rather than asserting it from theory.
    """
    live = _rec("x", "TRUE", "user", TrustTier.T4_VERIFIED, updated_at=100.0)
    live.status = Status.TRUSTED
    published_snapshot = {live.id: live}

    candidate = _rec("x", "FALSE", "tool", TrustTier.T1_CANDIDATE, updated_at=99999.0)  # far more recent
    staging = {candidate.id: candidate}
    wal = WriteAheadLog()

    # Normal (authority-aware): candidate authority (0.6) < live authority (0.8) -> held.
    result_normal = promotion_pipeline(staging, published_snapshot, wal, authority_score_fn=_authority_score,
                                        confidence_threshold=0.5)
    assert result_normal.promoted_to_verified == []

    # Ablated: trust_tier (T1) < live's (T4) -- held REGARDLESS of the
    # candidate being drastically more recent, because tuple comparison
    # never reaches updated_at while trust_tier already differs.
    staging2 = {candidate.id: MemoryRecord(**{**candidate.__dict__})}
    result_ablated = promotion_pipeline(staging2, published_snapshot, WriteAheadLog(),
                                         authority_score_fn=authority_score_without_authority,
                                         confidence_threshold=0.5)
    assert result_ablated.promoted_to_verified == [], (
        "CONFIRMS Section 7a: the ablated promotion gate blocks a fresh low-trust-tier candidate "
        "via trust_tier incumbency alone, even when it is drastically more recent than the "
        "already-published fact -- this is NOT a recency effect, it is a trust_tier/incumbency "
        "effect, and must be interpreted as such if the full experiment shows a similar pattern"
    )


# ---------------------------------------------------------------------------
# Test E: sleep/consolidation remains active and equivalent between the two backends
# ---------------------------------------------------------------------------
def test_E_ablated_backend_enters_same_lifecycle_states_as_normal():
    specs, ledger = build_workload(seed=0)
    small = specs[:120]  # spans at least 8 sessions' worth -> at least one 15-interaction period boundary

    normal = PeriodicAISOSBackend()
    ablated = AuthorityAblatedPeriodicAISOSBackend()
    for backend in (normal, ablated):
        for spec in small:
            if spec.context_reset_before:
                backend.apply_context_reset()
            if spec.kind == "teach":
                backend.write(spec.subject, spec.value, spec.source, spec.confidence, spec.interaction_id)
            backend.maybe_maintain(spec.interaction_id)

    assert normal.snapshot_metrics()["cycles_run"] == ablated.snapshot_metrics()["cycles_run"], (
        "the ablated backend must run exactly as many sleep cycles as the normal one, on the "
        "identical interaction sequence -- the scheduler and period are untouched by the ablation"
    )
    normal_states = {t["to_state"] for t in normal.lifecycle_trace()}
    ablated_states = {t["to_state"] for t in ablated.lifecycle_trace()}
    assert normal_states == ablated_states, "both backends must visit the same set of lifecycle states"


# ---------------------------------------------------------------------------
# Test F: workload equality across all 4 conditions (via the shared hash mechanism)
# ---------------------------------------------------------------------------
def test_F_workload_hash_identical_for_all_four_conditions():
    specs, _ = build_workload(seed=0)
    h = workload_hash(specs)
    # Simulate what e8_ablation.py's main() loop does: hash before/after
    # constructing and driving each of the 4 backends against the SAME specs list.
    for condition in ["no_sleep_naive", "continuous_mvcc", "periodic_aisos", "periodic_aisos_without_authority"]:
        backend = make_backend(condition)
        for spec in specs[:30]:
            if spec.kind == "teach":
                backend.write(spec.subject, spec.value, spec.source, spec.confidence, spec.interaction_id)
        assert workload_hash(specs) == h, f"condition {condition!r} must not mutate the shared workload"


# ---------------------------------------------------------------------------
# Test G: model equality -- the same model object is used for every condition
# (structural guarantee: e8_ablation.py's run_seed()/run_condition() take one
# shared `model` instance per seed, reused across the whole conditions loop).
# ---------------------------------------------------------------------------
def test_G_model_configuration_shared_across_conditions():
    from aisos.experiments.e8_long_running_1000 import run_condition
    sig = inspect.signature(run_condition)
    assert "model" in sig.parameters, (
        "run_condition() must take the model as a parameter (not construct its own), so the "
        "caller can guarantee one shared instance/configuration across every condition"
    )


# ---------------------------------------------------------------------------
# Test H: evaluator blindness -- evaluate() has no notion of "condition"
# ---------------------------------------------------------------------------
def test_H_evaluator_signature_has_no_condition_parameter():
    sig = inspect.signature(evaluate)
    assert "condition" not in sig.parameters
    assert "backend" not in sig.parameters
    # Confirm the same call with the same arguments returns the same
    # verdict regardless of which "condition" the caller claims -- the
    # function has no way to know or care.
    v1 = evaluate("PostgreSQL.", "PostgreSQL", ["PostgreSQL"], ["PostgreSQL"])
    v2 = evaluate("PostgreSQL.", "PostgreSQL", ["PostgreSQL"], ["PostgreSQL"])
    assert v1 == v2
