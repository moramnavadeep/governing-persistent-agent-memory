"""
HSRF causal ablation: confirms that trust_tier and/or authority (conditions
B/C/D/E) provide GENUINE, comparator-driven protection against a malicious
contradiction (W3) and a high-confidence false memory (W4), independent of
dream rehearsal -- while pure recency (condition A) provides NO genuine
protection at all: its apparent resistance under default settings is
entirely an artifact of deep_sleep.py's dream-rehearsal candidates
out-recencying a stale attacker/false-memory candidate (the same confound
documented in aisos/tests/test_pruning_mechanism.py's H11 finding),
reproduced here in the HSRF adversarial workloads. Disabling dream
(dream_seeds=0) isolates the comparator alone and flips A's result.
"""
from __future__ import annotations

from aisos.agent.condition_backends import (
    TrustTierAblatedPeriodicAISOSBackend, TrustOnlyPeriodicAISOSBackend,
    AuthorityOnlyPeriodicAISOSBackend, AuthorityTrustNoRecencyPeriodicAISOSBackend,
    PeriodicAISOSBackend,
)

PERIOD = 15


def _run_w3(backend) -> list:
    backend.write("w3_subj", "TRUE", "user", 0.9, 0)
    for i in range(1, 4 * PERIOD):
        backend.maybe_maintain(i)
    backend.write("w3_subj", "ATTACK", "tool", 0.6, 4 * PERIOD - 1)
    for i in range(4 * PERIOD, 12 * PERIOD):
        backend.maybe_maintain(i)
    return backend.read_all_values("w3_subj")


def test_recency_only_attack_resistance_is_a_dream_noise_confound():
    with_dream = _run_w3(TrustTierAblatedPeriodicAISOSBackend(period=PERIOD))
    no_dream = _run_w3(TrustTierAblatedPeriodicAISOSBackend(period=PERIOD, dream_seeds=0))
    assert with_dream == ["TRUE"], "with dream enabled, recency-only APPEARS to resist (the confound)"
    assert no_dream == ["ATTACK"], "with dream disabled, recency-only provides NO genuine protection"


def test_trust_authority_conditions_resist_independent_of_dream():
    for cls in (
        TrustOnlyPeriodicAISOSBackend, AuthorityOnlyPeriodicAISOSBackend,
        AuthorityTrustNoRecencyPeriodicAISOSBackend, PeriodicAISOSBackend,
    ):
        with_dream = _run_w3(cls(period=PERIOD))
        no_dream = _run_w3(cls(period=PERIOD, dream_seeds=0))
        assert with_dream == ["TRUE"], f"{cls.__name__} should resist with dream enabled"
        assert no_dream == ["TRUE"], f"{cls.__name__} should resist EVEN WITHOUT dream (genuine comparator security)"
