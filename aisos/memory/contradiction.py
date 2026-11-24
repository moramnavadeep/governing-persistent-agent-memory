"""
Contradiction detection and resolution (Technical Spec Section 27).

    A: "User lives in City X."
    B: "User moved to City Y."

    detect relation -> inspect source/time/trust -> resolve, retain with
    conditions, or flag -> record provenance.

Resolution uses recency, source authority, explicit user correction,
external evidence, or contextual coexistence -- here implemented as a
deterministic scoring function combining recency, a per-source authority
table, and trust tier, which is the pluggable slot's reference policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from aisos.memory.record import MemoryRecord

SOURCE_AUTHORITY = {
    "user_correction": 1.0,
    "user": 0.8,
    "tool": 0.6,
    "consolidation": 0.5,
    "dream": 0.1,
}


@dataclass
class Contradiction:
    subject: str
    records: list[MemoryRecord]
    winner: MemoryRecord
    losers: list[MemoryRecord]


def _score(r: MemoryRecord) -> float:
    authority = SOURCE_AUTHORITY.get(r.source, 0.4)
    trust = float(r.trust_tier) / 4.0
    # `updated_at` is a raw Unix epoch (~1.7e9): added directly it would
    # dwarf the authority/trust terms (max ~11,000) and let recency alone
    # decide any contradiction more than ~a few hours apart, overriding
    # authority entirely -- the opposite of the guarantee below. Scaled down
    # by 1e8, its contribution stays under the smallest gap between
    # authority/trust bands (250, from the 1000/4 trust-tier step), so it
    # only breaks ties *within* a band; authority and trust still dominate
    # and guard against a low-authority-but-very-recent record beating a
    # high-authority older one purely on timestamp.
    recency = r.updated_at / 1e8
    return authority * 10_000 + trust * 1_000 + recency


def score_without_authority(r: MemoryRecord) -> float:
    """Authority-ablation counterpart to _score() (docs/authority_ablation_spec.md):
    identical formula with the SOURCE_AUTHORITY term removed, trust_tier
    and recency retained at their original relative weight. This is the
    project's own existing fallback signal, not an invented rule -- see
    the ablation spec for why. Used only by
    aisos/agent/condition_backends.py's authority-ablated backend; the
    default behavior of find_contradictions() below is unchanged.
    """
    trust = float(r.trust_tier) / 4.0
    recency = r.updated_at / 1e8
    return trust * 1_000 + recency


def score_recency_only(r: MemoryRecord) -> float:
    """Trust-tier ablation counterpart (docs/mechanism_hypotheses.md
    H1/H2/H3), built on top of score_without_authority: removes trust_tier
    too, leaving pure recency -- the minimum experiment that tests
    'does trust_tier matter, independent of authority' and 'is pure
    recency sufficient' in a single comparator, since only one term
    (updated_at) remains once both others are stripped. Used only by
    aisos/agent/condition_backends.py's trust-tier-ablated backend.
    """
    return r.updated_at / 1e8


def score_trust_only(r: MemoryRecord) -> float:
    """Security-Retrieval-Freshness causal ablation
    (AISOS_v3_Claude_Security_Retrieval_Freshness_Novelty_Prompt.md
    Section 12, condition B): isolates trust_tier alone -- neither
    authority nor recency ever break a tie. Completes the 5-point A-E
    design alongside the already-existing score_recency_only (A) and
    _score (E, full); the project's prior ablations only ever removed
    ONE term at a time from the full comparator (authority, or
    authority+trust), never isolated trust_tier by itself.
    """
    return float(r.trust_tier)


def score_authority_only(r: MemoryRecord) -> float:
    """Security-Retrieval-Freshness causal ablation, condition C:
    isolates authority alone -- neither trust_tier nor recency ever
    break a tie."""
    return SOURCE_AUTHORITY.get(r.source, 0.4)


def score_authority_and_trust_no_recency(r: MemoryRecord) -> float:
    """Security-Retrieval-Freshness causal ablation, condition D:
    authority and trust_tier both weighted (same relative weight as the
    full comparator _score()), recency term removed entirely -- isolates
    whether authority+trust alone (no tie-breaking by time at all)
    changes outcomes relative to the full comparator E."""
    authority = SOURCE_AUTHORITY.get(r.source, 0.4)
    trust = float(r.trust_tier) / 4.0
    return authority * 10_000 + trust * 1_000


def verified_authority_contradiction_score(r: MemoryRecord) -> float:
    """Trusted-authority-registry counterpart to _score()
    (AISOS_Trusted_Authority_Provenance_Next_Milestone.md): identical
    formula and weighting, but reads r.verified_authority (set once by
    AISOSAgent.remember() via aisos/security/authority_registry.py, never
    the caller's own claim) instead of SOURCE_AUTHORITY.get(r.source, 0.4).
    A record with verified_authority=None (no registry was consulted for
    it) is treated as UNVERIFIED_AUTHORITY (0.0), NOT as trusting its
    source string -- this is the fail-safe policy, applied here exactly as
    in the promotion-gate counterpart in promotion.py. Used only by
    aisos/agent/condition_backends.py's trusted-authority backend; the
    default behavior of find_contradictions() below is unchanged.
    """
    from aisos.security.authority_registry import UNVERIFIED_AUTHORITY
    authority = r.verified_authority if r.verified_authority is not None else UNVERIFIED_AUTHORITY
    trust = float(r.trust_tier) / 4.0
    recency = r.updated_at / 1e8
    return authority * 10_000 + trust * 1_000 + recency


def find_contradictions(
    records: list[MemoryRecord], score_fn: Callable[[MemoryRecord], float] = _score,
) -> list[Contradiction]:
    by_subject: dict[str, list[MemoryRecord]] = {}
    for r in records:
        by_subject.setdefault(r.subject, []).append(r)

    out = []
    for subj, recs in by_subject.items():
        distinct_values = {r.value for r in recs}
        if len(distinct_values) <= 1:
            continue
        ranked = sorted(recs, key=score_fn, reverse=True)
        winner = ranked[0]
        losers = ranked[1:]
        out.append(Contradiction(subject=subj, records=recs, winner=winner, losers=losers))
    return out
