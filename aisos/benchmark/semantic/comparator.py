"""
Benchmark-specific authority comparator for the semantic/enterprise
corpus (AISOS_v3_Next_Semantic_RAG_Enterprise_Benchmark_Prompt.md).

**A real bug, found and fixed before any benchmark result was accepted**:
`aisos/benchmark/semantic/corpus.py`'s documents use enterprise-style
source strings ("policy_doc", "engineering_wiki", "anonymous_edit",
"ticket", "incident_report", "user_correction") with their OWN authority
table (`SOURCE_TYPES`). AISOS's production comparators
(`aisos.memory.contradiction._score`, `aisos.memory.promotion._authority_score`)
read a DIFFERENT, unrelated table (`aisos.memory.contradiction.SOURCE_AUTHORITY`,
keyed only by `"user_correction"`, `"user"`, `"tool"`, `"consolidation"`,
`"dream"`) -- confirmed directly that `SOURCE_AUTHORITY.get("policy_doc", 0.4)`
and `SOURCE_AUTHORITY.get("anonymous_edit", 0.4)` both silently returned
the SAME default (0.4), destroying every intended authority
differentiation in this benchmark's R6/R7/R8/R10 and the Section 11/13
adversarial experiments -- resolution fell through to recency/trust_tier
alone, and specific queries were confirmed to resolve to the WRONG
(attacker/conflicting) value as a direct, reproduced consequence (see
docs/semantic_rag_failure_analysis.md).

Fixed the same way every prior milestone in this project fixes an
authority-mapping mismatch: an ADDITIVE, dependency-injected comparator
function wired in via AISOS's existing `contradiction_score_fn`/
`authority_score_fn` constructor parameters -- AISOS's own production
`_score()`/`_authority_score()`/`SOURCE_AUTHORITY` are never touched. This
is a benchmark-corpus/harness bug, not an AISOS production bug (the
production `SOURCE_AUTHORITY` table works exactly as documented for its
own recognized source vocabulary; this benchmark simply needs its own).
"""
from __future__ import annotations

from aisos.benchmark.semantic.corpus import SOURCE_TYPES
from aisos.memory.record import MemoryRecord


def semantic_contradiction_score(r: MemoryRecord) -> float:
    """Identical formula/weighting to aisos.memory.contradiction._score(),
    reading this benchmark's own SOURCE_TYPES authority table instead of
    AISOS's production SOURCE_AUTHORITY table."""
    authority = SOURCE_TYPES.get(r.source, 0.4)
    trust = float(r.trust_tier) / 4.0
    recency = r.updated_at / 1e8
    return authority * 10_000 + trust * 1_000 + recency


def semantic_authority_score(r: MemoryRecord) -> tuple:
    """Identical shape to aisos.memory.promotion._authority_score()."""
    return (SOURCE_TYPES.get(r.source, 0.4), float(r.trust_tier), r.updated_at)
