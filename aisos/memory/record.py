"""
MemoryRecord (Technical Spec Section 25) and the trust-tiered memory model
(v2 Section 5.10, "Extension II"): M_LT is replaced by four tiers of
increasing trust, movement permitted in one direction only.

    M_D (T0_SYNTHETIC)   -- raw Dream Phase output. Never readable by live
                             requests (I6).
    M_C (T1_CANDIDATE)   -- passed internal-consistency checks; still not
                             readable by live requests.
    M_V (T2_VERIFIED)    -- promoted, externally-corroborated-or-repeatedly-
                             validated. This is the *only* tier live reads
                             may dereference.
    M_X (REJECTED)       -- terminal sink for anything that fails promotion.

This mirrors the coarser T0-T4 tiers in the Technical Spec (Section 18);
T0-T4 are exposed as an alias mapping for readability in reports.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class TrustTier(IntEnum):
    REJECTED = -1       # M_X
    T0_SYNTHETIC = 0    # M_D  -- raw dream/rehearsal output
    T1_CANDIDATE = 1    # M_C  -- internally consistent candidate
    T2_VALIDATED = 2    # survived >=1 additional promotion round
    T3_CORROBORATED = 3  # externally corroborated
    T4_VERIFIED = 4     # M_V  -- live-readable trusted operational memory


# Live requests may only dereference tiers >= this value (I6).
LIVE_READABLE_FLOOR = TrustTier.T4_VERIFIED


class Status(str):
    CANDIDATE = "candidate"
    TRUSTED = "trusted"
    ARCHIVED = "archived"
    REJECTED = "rejected"


def _pseudo_embedding(text: str, dims: int = 16) -> tuple[float, ...]:
    """Deterministic, dependency-free stand-in for a real embedding model.
    AISOS does not claim a novel embedding/consolidation algorithm (v2
    Section 1.2) -- this is the pluggable slot's reference implementation,
    good enough to drive real (not fabricated) clustering/similarity
    behavior in the simulation without requiring a live model.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vals = []
    for i in range(dims):
        byte = h[i % len(h)]
        vals.append((byte / 255.0) * 2 - 1)
    return tuple(vals)


def cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


@dataclass
class MemoryRecord:
    subject: str            # entity/fact slot, e.g. "user.city"
    value: str               # the content/fact value, e.g. "Springfield"
    source: str = "user"      # "user" | "tool" | "dream" | "consolidation"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    trust_tier: TrustTier = TrustTier.T1_CANDIDATE
    status: str = Status.CANDIDATE
    confidence: float = 0.5
    version: int = 1
    validation_rounds: int = 0
    related: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    embedding: tuple[float, ...] = field(default=None)  # set in __post_init__
    # Trusted-authority mechanism (AISOS_Trusted_Authority_Provenance_Next_Milestone.md):
    # set ONCE at creation by AISOSAgent.remember() when a TrustedAuthorityRegistry
    # is configured, from registry.verify(source, credential) -- never the
    # caller's own claim. None means "no registry was consulted" (the
    # legacy, pre-this-milestone path); verified_authority_score() in
    # contradiction.py/promotion.py treats None as UNVERIFIED_AUTHORITY,
    # NOT as "trust the source string" -- see authority_registry.py.
    verified_authority: Optional[float] = None

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = _pseudo_embedding(f"{self.subject}:{self.value}")

    def is_live_readable(self) -> bool:
        return self.trust_tier >= LIVE_READABLE_FLOOR and self.status == Status.TRUSTED

    def content_hash(self) -> str:
        # verified_authority is included (Trusted Authority Registry
        # milestone, Phase E "provenance binding"): once set at creation,
        # it must never silently change without being detectable by the
        # existing WAL-chain/HMAC-bundle integrity checks that already
        # consume this hash as their payload.
        payload = f"{self.id}|{self.subject}|{self.value}|{self.trust_tier}|{self.version}|{self.verified_authority}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def clone_bumped(self, **overrides) -> "MemoryRecord":
        """Copy-on-write helper: never mutate a published record in place."""
        data = {
            "subject": self.subject,
            "value": self.value,
            "source": self.source,
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": time.time(),
            "trust_tier": self.trust_tier,
            "status": self.status,
            "confidence": self.confidence,
            "version": self.version + 1,
            "validation_rounds": self.validation_rounds,
            "related": list(self.related),
            "provenance": dict(self.provenance),
            "embedding": self.embedding,
            "verified_authority": self.verified_authority,
        }
        data.update(overrides)
        return MemoryRecord(**data)
