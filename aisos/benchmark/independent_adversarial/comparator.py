"""
Independent-benchmark authority comparator.

**A real bug, found and fixed via smoke-testing before any result was
accepted** -- the FOURTH recurrence of an exact bug class this project
has now hit four times (first in the semantic-RAG-benchmark milestone,
then the multi-agent benchmark, then again here): this package's own
source labels ("verified_correction", "official_record",
"department_memo", "informal_note", "unverified_submission") do not
exist in EITHER AISOS's production SOURCE_AUTHORITY table nor the
semantic benchmark's own SOURCE_TYPES table -- both silently collapse
every one of this package's source labels to the same default authority
(0.4), destroying the corpus's entire designed authority hierarchy.
Confirmed directly via smoke test: A1 (direct false injection from an
"unverified_submission" source) was accepted as the governed value by
S6 (AISOS) when first run against `_make_semantic_aisos_backend()`,
which reads the SEMANTIC benchmark's SOURCE_TYPES table, not this
package's own.

Fixed the same way every prior occurrence of this bug class was fixed:
an additive, dependency-injected comparator reading THIS package's own
source_authority() table, wired in via AISOS's existing
contradiction_score_fn/authority_score_fn constructor parameters.
AISOS's own production _score()/_authority_score()/SOURCE_AUTHORITY, and
the semantic benchmark's own comparator.py, are never touched.
"""
from __future__ import annotations

from aisos.benchmark.independent_adversarial.corpus import source_authority
from aisos.memory.record import MemoryRecord


def independent_contradiction_score(r: MemoryRecord) -> float:
    authority = source_authority(r.source)
    trust = float(r.trust_tier) / 4.0
    recency = r.updated_at / 1e8
    return authority * 10_000 + trust * 1_000 + recency


def independent_authority_score(r: MemoryRecord) -> tuple:
    return (source_authority(r.source), float(r.trust_tier), r.updated_at)


# ---------------------------------------------------------------------
# Ablation variants (Section 21, G1-G3) -- own versions of this project's
# established ablation pattern (aisos/memory/contradiction.py::score_without_authority/
# score_recency_only, aisos/memory/promotion.py::authority_score_without_authority/
# authority_score_recency_only), reading THIS package's own authority
# table rather than AISOS's production table, for the independent
# benchmark's own G1 (authority removed) / G2 (trust removed) / G3
# (freshness removed) ablations.
# ---------------------------------------------------------------------

def independent_score_without_authority(r: MemoryRecord) -> float:
    """G1: authority term removed, trust_tier and recency retained."""
    trust = float(r.trust_tier) / 4.0
    recency = r.updated_at / 1e8
    return trust * 1_000 + recency


def independent_authority_score_without_authority(r: MemoryRecord) -> tuple:
    return (float(r.trust_tier), r.updated_at)


def independent_score_without_trust(r: MemoryRecord) -> float:
    """G2: trust_tier term removed, authority and recency retained."""
    authority = source_authority(r.source)
    recency = r.updated_at / 1e8
    return authority * 10_000 + recency


def independent_authority_score_without_trust(r: MemoryRecord) -> tuple:
    return (source_authority(r.source), r.updated_at)


def independent_score_without_freshness(r: MemoryRecord) -> float:
    """G3: recency term removed entirely, authority and trust retained --
    ties broken arbitrarily (insertion order) when authority and trust
    are equal, since no other differentiating signal remains."""
    authority = source_authority(r.source)
    trust = float(r.trust_tier) / 4.0
    return authority * 10_000 + trust * 1_000


def independent_authority_score_without_freshness(r: MemoryRecord) -> tuple:
    return (source_authority(r.source), float(r.trust_tier))
