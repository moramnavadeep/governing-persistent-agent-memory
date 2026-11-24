"""
H11 (docs/mechanism_hypotheses.md): the actual mechanism behind the
trust-tier ablation's "0.0 dangerous-contradiction-error rate" result --
and its freshness cost -- deterministically, with zero model calls.

Corrected finding (superseding an earlier, wrong attribution to
deep_sleep.py's hardcoded version-pruning sort): the responsible code path
is deep_sleep.py's WITHIN-CYCLE contradiction resolution
(find_contradictions(), parameterized by contradiction_score_fn -- the
same function ablated by the trust-tier experiment). Dream
(dream_phase.py + adapter.py::_build_weakness_graph()) unconditionally
creates dream_seeds=4 fresh T0_SYNTHETIC candidates PER CYCLE for every
currently-live subject, each carrying that subject's CURRENT live value
with a brand-new updated_at. Because find_contradictions() keeps only ONE
winner per subject per cycle (archiving every other distinct value), this
becomes a tournament every cycle between whatever external candidate
exists for a subject (attacker or legitimate correction) and dream's own
same-subject rehearsal:

  - Under the REAL, authority-aware _score(): dream's SOURCE_AUTHORITY is
    0.1, the LOWEST of every registered source (tool=0.6, user=0.8,
    user_correction=1.0) -- so dream never wins this tournament against
    any externally-taught candidate, regardless of recency. This mechanism
    is effectively invisible under full periodic_aisos.
  - Under score_recency_only (the trust-tier ablation's comparator): dream
    candidates, refreshed every single cycle, are almost always the most
    recently created same-subject candidate -- so they usually WIN this
    tournament, archiving the external candidate within 1-2 cycles,
    REGARDLESS of whether that candidate was an attack or a legitimate
    correction. This is confirmed directly below: disabling dream entirely
    (dream_seeds=0) lets BOTH a legitimate correction AND an attacker
    reach T4_VERIFIED under the identical pure-recency comparator -- proof
    that dream-noise-suppression, not recency correctly resolving the
    conflict, was responsible for the previously reported "H3 supported"
    result.
"""
from __future__ import annotations

from aisos.agent.workload_1000 import build_workload
from aisos.agent.condition_backends import TrustTierAblatedPeriodicAISOSBackend, PeriodicAISOSBackend
from aisos.memory.record import Status


def _run(backend, specs, up_to_interaction: int, observed: dict | None = None):
    """`observed`, if given, accumulates {(subject, value): status} across
    EVERY cycle boundary, not just the final one -- required since
    `_staging` Retention & Memory Scaling
    (AISOS_staging_retention_next_milestone.md) now evicts a
    terminal-status (ARCHIVED/REJECTED/TRUSTED) staging entry once it has
    survived one full committed cycle unchanged (docs/
    staging_retention_design.md). A record's terminal disposition is
    still visible for the cycle it was decided in -- callers that need to
    observe a historical disposition must capture it then, not assume
    `all_staging()` retains it indefinitely afterward.
    """
    for spec in specs:
        if spec.interaction_id > up_to_interaction:
            break
        if spec.context_reset_before:
            backend.apply_context_reset()
        if spec.kind == "teach":
            backend.write(spec.subject, spec.value, spec.source, spec.confidence, spec.interaction_id)
        backend.maybe_maintain(spec.interaction_id)
        if observed is not None:
            for r in backend.agent.store.all_staging():
                observed[(r.subject, r.value)] = r.status
    return backend


def test_1_attacker_and_correction_are_both_archived_under_pure_recency_with_dream_enabled():
    """Baseline (dream enabled, default dream_seeds=4): confirms the
    previously-observed outcome -- both the contra_0 attacker and the
    changing_0 legitimate correction get archived under
    score_recency_only, never reaching T4_VERIFIED within a comparable
    number of cycles.
    """
    specs, _ = build_workload(seed=0)
    observed: dict = {}
    backend = _run(TrustTierAblatedPeriodicAISOSBackend(), specs, 800, observed=observed)

    assert observed.get(("contra_0", "contra_0_FALSE_value")) == Status.ARCHIVED
    assert observed.get(("changing_0", "changing_0_v1")) == Status.ARCHIVED
    live_contra = backend.agent.store.read_live("contra_0")
    assert all(r.value == "contra_0_true_value" for r in live_contra)


def test_2_disabling_dream_lets_the_legitimate_correction_reach_live():
    """Positive control isolating dream specifically (not the comparator):
    with dream_seeds=0, the identical pure-recency comparator lets the
    changing_0 correction climb all the way to T4_VERIFIED and become
    live -- confirming dream-rehearsal volume, not the comparator itself,
    was suppressing it in test 1.
    """
    specs, _ = build_workload(seed=0)
    backend = _run(TrustTierAblatedPeriodicAISOSBackend(dream_seeds=0), specs, 800)

    live = backend.agent.store.read_live("changing_0")
    assert any(r.value == "changing_0_v1" for r in live), (
        "with dream disabled, the legitimate correction must reach live status under the same "
        "pure-recency comparator that failed to promote it in test 1"
    )


def test_3_disabling_dream_ALSO_lets_the_attacker_reach_live_this_is_the_key_safety_finding():
    """THE key correction to the trust-tier ablation report: with dream
    disabled, the attacker's false value ALSO reaches T4_VERIFIED under
    the identical pure-recency comparator that the ablation report
    credited with a 0.0 dangerous-contradiction-error rate. This proves
    the previously reported safety result under
    periodic_aisos_without_trust_tier was an artifact of dream's noise
    indiscriminately suppressing external candidates (attacker and
    legitimate correction alike), not evidence that recency alone
    correctly resolves the conflict. H3 ("recency is sufficient for this
    attack pattern") is therefore FALSIFIED when dream's confound is
    controlled for, reversing the trust-tier ablation report's conclusion.
    """
    specs, _ = build_workload(seed=0)
    backend = _run(TrustTierAblatedPeriodicAISOSBackend(dream_seeds=0), specs, 800)

    live = backend.agent.store.read_live("contra_0")
    assert any(r.value == "contra_0_FALSE_value" for r in live), (
        "with dream disabled, pure recency must let the attacker's more-recent false value reach "
        "live status -- confirming recency alone does not resist this attack, and the previously "
        "reported 0.0 dangerous-error rate depended on dream's noise, not on recency being a "
        "genuine defense"
    )


def test_4_full_periodic_aisos_is_protected_by_authority_not_by_dream_noise():
    """Under the REAL authority-aware comparator, dream's own fixed
    SOURCE_AUTHORITY (0.1, the lowest of any registered source) means it
    never wins the within-cycle tournament against an externally-taught
    candidate (tool=0.6, user=0.8, user_correction=1.0) regardless of
    recency -- so disabling dream should NOT be needed for full
    periodic_aisos to resist the attack, and the attacker must still lose
    even with dream fully enabled.
    """
    specs, _ = build_workload(seed=0)
    backend = _run(PeriodicAISOSBackend(), specs, 800)

    live = backend.agent.store.read_live("contra_0")
    assert all(r.value == "contra_0_true_value" for r in live), (
        "full periodic_aisos's authority-aware comparator must resist the attack even with dream "
        "fully enabled, because dream's fixed 0.1 authority never outranks the attacker's own 0.6 "
        "-- confirming authority (not dream suppression) is what protects the FULL condition"
    )


def test_5_dream_records_carry_the_lowest_source_authority_in_the_table():
    """Structural confirmation of the mechanism explaining tests 3 vs 4:
    'dream' is registered in SOURCE_AUTHORITY at 0.1, strictly below every
    externally-taught source this workload uses.
    """
    from aisos.memory.contradiction import SOURCE_AUTHORITY
    assert SOURCE_AUTHORITY["dream"] == 0.1
    assert SOURCE_AUTHORITY["dream"] < SOURCE_AUTHORITY["tool"] < SOURCE_AUTHORITY["user"] < SOURCE_AUTHORITY["user_correction"]
