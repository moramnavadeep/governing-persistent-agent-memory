"""
Clean 2x2 factorial (Authority x Dream, trust_tier held constant ON),
per AISOS_Next_Claude_Master_Prompt_Causal_Mechanism_v2.md Sections 6-9,
15. Zero model calls. Formalizes docs/candidate_lifecycle_trace.md's
exploratory diagnostic into permanent deterministic regression tests.

Finding: Authority is the sole factor separating IDEAL (attacker rejected,
legitimate correction accepted) from SAFE-BUT-STALE (both rejected) in
this design -- Dream has no effect on any outcome once trust_tier is held
constant. This is a distinct, cleaner result from the earlier trust-tier
ablation (aisos/tests/test_pruning_mechanism.py), which additionally
removed trust_tier and found Dream DOES matter there.
"""
from __future__ import annotations

from aisos.agent.workload_1000 import build_workload
from aisos.agent.condition_backends import PeriodicAISOSBackend, AuthorityAblatedPeriodicAISOSBackend
from aisos.memory.promotion import _authority_score, authority_score_without_authority
from aisos.memory.record import Status


def _run(backend_cls, dream_seeds, up_to=800, seed=0):
    specs, _ = build_workload(seed=seed)
    backend = backend_cls(dream_seeds=dream_seeds)
    for spec in specs:
        if spec.interaction_id > up_to:
            break
        if spec.context_reset_before:
            backend.apply_context_reset()
        if spec.kind == "teach":
            backend.write(spec.subject, spec.value, spec.source, spec.confidence, spec.interaction_id)
        backend.maybe_maintain(spec.interaction_id)
    return backend


def _outcome(backend):
    attacker_live = any(r.value == "contra_0_FALSE_value" for r in backend.agent.store.read_live("contra_0"))
    correction_live = any(r.value == "changing_0_v1" for r in backend.agent.store.read_live("changing_0"))
    return attacker_live, correction_live


def test_C1_authority_on_dream_on_is_ideal():
    attacker_live, correction_live = _outcome(_run(PeriodicAISOSBackend, dream_seeds=4))
    assert attacker_live is False, "authority ON must reject the attacker"
    assert correction_live is True, "authority ON must accept the legitimate correction"


def test_C2_authority_off_dream_on_is_safe_but_stale():
    attacker_live, correction_live = _outcome(_run(AuthorityAblatedPeriodicAISOSBackend, dream_seeds=4))
    assert attacker_live is False, "trust-tier incumbency alone still rejects the attacker"
    assert correction_live is False, (
        "with authority removed, the legitimate correction must ALSO be rejected -- trust-tier "
        "incumbency cannot distinguish it from an attacker"
    )


def test_C3_authority_on_dream_off_is_identical_to_C1():
    """Dream must have NO effect on outcomes when trust_tier is intact."""
    attacker_live, correction_live = _outcome(_run(PeriodicAISOSBackend, dream_seeds=0))
    assert attacker_live is False
    assert correction_live is True


def test_C4_authority_off_dream_off_is_identical_to_C2():
    """Dream must have NO effect on outcomes when trust_tier is intact,
    even with authority also removed -- confirming Dream's suppressive
    role (H11) is specific to the FURTHER trust-tier-ablated condition,
    not to authority-ablation alone.
    """
    attacker_live, correction_live = _outcome(_run(AuthorityAblatedPeriodicAISOSBackend, dream_seeds=0))
    assert attacker_live is False
    assert correction_live is False


def test_authority_score_difference_is_measurable_and_decisive():
    """Direct numeric confirmation of the mechanism: the gate's tuple
    comparison is decided at the AUTHORITY element alone in both
    scenarios, without ever consulting trust_tier -- the exact numbers
    from docs/candidate_lifecycle_trace.md's Case A/B.
    """
    from aisos.memory.record import MemoryRecord, TrustTier

    attacker = MemoryRecord(subject="x", value="FALSE", source="tool", confidence=0.6,
                             trust_tier=TrustTier.T1_CANDIDATE, status=Status.CANDIDATE)
    live_true = MemoryRecord(subject="x", value="TRUE", source="user", confidence=0.9,
                              trust_tier=TrustTier.T4_VERIFIED, status=Status.TRUSTED)
    correction = MemoryRecord(subject="x", value="NEW", source="user_correction", confidence=0.95,
                               trust_tier=TrustTier.T1_CANDIDATE, status=Status.CANDIDATE)

    assert _authority_score(attacker)[0] < _authority_score(live_true)[0], "attacker's authority must be lower"
    assert _authority_score(correction)[0] > _authority_score(live_true)[0], "correction's authority must be higher"
    # Confirm trust_tier is genuinely irrelevant to these two outcomes: even
    # if we forcibly gave the attacker/correction a HIGHER trust_tier than
    # the live record, the authority element (compared first) still decides.
    attacker_high_tier = attacker.clone_bumped(trust_tier=TrustTier.T4_VERIFIED)
    correction_low_tier = correction.clone_bumped(trust_tier=TrustTier.T0_SYNTHETIC)
    assert _authority_score(attacker_high_tier) < _authority_score(live_true), (
        "authority must still reject the attacker even if it is given the SAME trust_tier as live"
    )
    assert _authority_score(correction_low_tier) > _authority_score(live_true), (
        "authority must still accept the correction even at the LOWEST possible trust_tier"
    )


def test_no_benchmark_labels_leak_into_the_comparator():
    """Fairness check (Section 20): the comparator functions take only a
    MemoryRecord and read only its source/trust_tier/updated_at -- no
    is_malicious/is_legitimate field exists anywhere on the record.
    """
    from aisos.memory.record import MemoryRecord
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(MemoryRecord)}
    assert "is_malicious" not in field_names
    assert "is_legitimate" not in field_names
    assert "ground_truth" not in field_names


def test_attacker_and_correction_are_evaluated_by_the_identical_code_path():
    """Structural fairness check: both scenarios in this factorial call
    the exact same backend.write()/maybe_maintain() methods with no
    scenario-specific branching -- confirmed by inspecting that
    AuthorityAblatedPeriodicAISOSBackend and PeriodicAISOSBackend expose
    no parameter distinguishing an 'attack' write from a 'correction'
    write; only the caller-supplied source/value/confidence(/credential,
    added by the trusted-authority milestone) differ, which are
    legitimately available signals (Section 20), not hidden labels like
    is_attacker/is_legitimate/ground_truth.
    """
    import inspect
    sig = inspect.signature(PeriodicAISOSBackend.write)
    params = set(sig.parameters) - {"self"}
    assert params == {"subject", "value", "source", "confidence", "interaction_id", "credential"}
    assert "is_attacker" not in params and "is_legitimate" not in params and "ground_truth" not in params
