"""
Security-Retrieval-Freshness (HSRF) causal ablation
(AISOS_v3_Claude_Security_Retrieval_Freshness_Novelty_Prompt.md
Section 12): the 5-point A-E design required 3 new comparator variants
(trust-only, authority-only, authority+trust-no-recency) that did not
previously exist in this codebase (only recency-only and the full
comparator existed as isolated single/combined-term variants). These
are purely additive, dependency-injected functions following the exact
pattern already established by every prior ablation in this project --
no existing decision logic was modified.
"""
from __future__ import annotations

from aisos.memory.contradiction import (
    score_trust_only, score_authority_only, score_authority_and_trust_no_recency, _score,
)
from aisos.memory.promotion import (
    authority_score_trust_only, authority_score_authority_only, authority_score_authority_and_trust,
)
from aisos.memory.record import MemoryRecord, TrustTier, Status
from aisos.agent.condition_backends import (
    TrustOnlyPeriodicAISOSBackend, AuthorityOnlyPeriodicAISOSBackend,
    AuthorityTrustNoRecencyPeriodicAISOSBackend,
)

PERIOD = 15


def _rec(source="user", trust_tier=TrustTier.T4_VERIFIED, updated_at=1000.0):
    r = MemoryRecord(subject="s", value="v", source=source, trust_tier=trust_tier, status=Status.TRUSTED)
    r.updated_at = updated_at
    return r


def test_score_trust_only_ignores_authority_and_recency():
    low_authority_high_trust = _rec(source="dream", trust_tier=TrustTier.T4_VERIFIED, updated_at=1.0)
    high_authority_low_trust_recent = _rec(source="user_correction", trust_tier=TrustTier.T1_CANDIDATE, updated_at=999999.0)
    assert score_trust_only(low_authority_high_trust) > score_trust_only(high_authority_low_trust_recent)


def test_score_authority_only_ignores_trust_and_recency():
    high_authority_low_trust_old = _rec(source="user_correction", trust_tier=TrustTier.T1_CANDIDATE, updated_at=1.0)
    low_authority_high_trust_recent = _rec(source="dream", trust_tier=TrustTier.T4_VERIFIED, updated_at=999999.0)
    assert score_authority_only(high_authority_low_trust_old) > score_authority_only(low_authority_high_trust_recent)


def test_score_authority_and_trust_no_recency_ignores_only_recency():
    a = _rec(source="user_correction", trust_tier=TrustTier.T4_VERIFIED, updated_at=1.0)
    b = _rec(source="user_correction", trust_tier=TrustTier.T4_VERIFIED, updated_at=999999.0)
    assert score_authority_and_trust_no_recency(a) == score_authority_and_trust_no_recency(b)  # recency truly ignored
    c = _rec(source="dream", trust_tier=TrustTier.T1_CANDIDATE, updated_at=1.0)
    assert score_authority_and_trust_no_recency(a) > score_authority_and_trust_no_recency(c)  # authority+trust still decide


def test_authority_score_variants_match_contradiction_score_ranking():
    a = _rec(source="user_correction", trust_tier=TrustTier.T4_VERIFIED)
    b = _rec(source="dream", trust_tier=TrustTier.T1_CANDIDATE)
    assert authority_score_trust_only(a) > authority_score_trust_only(b)
    assert authority_score_authority_only(a) > authority_score_authority_only(b)
    assert authority_score_authority_and_trust(a) > authority_score_authority_and_trust(b)


def test_trust_only_backend_runs_end_to_end():
    backend = TrustOnlyPeriodicAISOSBackend(period=PERIOD)
    backend.write("s1", "V1", "user", 0.9, 0)
    for i in range(1, 8 * PERIOD):
        backend.maybe_maintain(i)
    assert backend.read("s1") == "V1"


def test_authority_only_backend_runs_end_to_end():
    backend = AuthorityOnlyPeriodicAISOSBackend(period=PERIOD)
    backend.write("s1", "V1", "user", 0.9, 0)
    for i in range(1, 8 * PERIOD):
        backend.maybe_maintain(i)
    assert backend.read("s1") == "V1"


def test_authority_trust_no_recency_backend_runs_end_to_end():
    backend = AuthorityTrustNoRecencyPeriodicAISOSBackend(period=PERIOD)
    backend.write("s1", "V1", "user", 0.9, 0)
    for i in range(1, 8 * PERIOD):
        backend.maybe_maintain(i)
    assert backend.read("s1") == "V1"


def test_authority_only_resists_low_authority_attacker():
    backend = AuthorityOnlyPeriodicAISOSBackend(period=PERIOD)
    backend.write("s1", "TRUE", "user", 0.9, 0)
    for i in range(1, 4 * PERIOD):
        backend.maybe_maintain(i)
    for cycle in range(4, 9):
        backend.write("s1", "ATTACK", "dream", 0.6, cycle * PERIOD - 1)
        backend.maybe_maintain(cycle * PERIOD)
    assert backend.read("s1") == "TRUE"


def test_default_full_comparator_unchanged_by_new_additions():
    """Confirms the default _score() function itself is byte-for-byte
    unaffected by the new sibling functions added alongside it."""
    r = _rec()
    assert _score(r) == SOURCE_AUTHORITY_SCORE_REFERENCE(r)


def SOURCE_AUTHORITY_SCORE_REFERENCE(r):
    from aisos.memory.contradiction import SOURCE_AUTHORITY
    authority = SOURCE_AUTHORITY.get(r.source, 0.4)
    trust = float(r.trust_tier) / 4.0
    recency = r.updated_at / 1e8
    return authority * 10_000 + trust * 1_000 + recency
