"""
HSRF causal ablation (Section 12): pure trust-tier scoring (condition B,
score_trust_only / authority_score_trust_only, no authority term at all)
has a genuine, dream-independent structural pathology distinct from the
dream-noise confound documented in test_hsrf_dream_confound.py -- it makes
memory permanently immutable once any value reaches T4_VERIFIED for a
subject, REGARDLESS of how legitimate or high-confidence a later correction
is.

Mechanism: promotion_pipeline()'s conflicting-live check requires a
candidate's authority_score_fn(r) to exceed the live incumbent's
authority_score_fn(best_live) BEFORE it is allowed to advance any further
tier at all -- if it does not, it is held (and eventually rejected after
max_rounds). Under authority_score_trust_only, that score IS the trust_tier
itself: a fresh candidate always starts below T4 by construction, so it can
never out-trust-tier an already-T4 incumbent, and therefore can never climb
the very tiers that would let it do so. This is a genuine, permanent
incumbency lock (perfect resistance to ANY change, legitimate or not), not
an artifact of dream rehearsal -- confirmed here with dream disabled and a
generous 24-cycle budget (far more than the ~4 cycles a normal promotion
needs).

Trust_tier combined with ANY other differentiating term (authority in C/D/E,
or recency in the full comparator) does not have this pathology, because a
tie on trust_tier falls through to a term that CAN differ between a fresh
candidate and an old incumbent.
"""
from __future__ import annotations

from aisos.agent.condition_backends import (
    TrustOnlyPeriodicAISOSBackend, AuthorityOnlyPeriodicAISOSBackend,
    AuthorityTrustNoRecencyPeriodicAISOSBackend, PeriodicAISOSBackend,
)

PERIOD = 15


def _run_correction(cls, dream_seeds=None):
    kw = {"period": PERIOD}
    if dream_seeds is not None:
        kw["dream_seeds"] = dream_seeds
    backend = cls(**kw)
    backend.write("s", "A", "user", 0.9, 0)
    for i in range(1, 6 * PERIOD):
        backend.maybe_maintain(i)
    backend.write("s", "B", "user_correction", 0.95, 6 * PERIOD - 1)
    for i in range(6 * PERIOD, 30 * PERIOD):  # generous 24-cycle budget
        backend.maybe_maintain(i)
    return backend.read_all_values("s")


def test_trust_only_permanently_blocks_any_correction_even_with_generous_cycles_and_no_dream():
    assert _run_correction(TrustOnlyPeriodicAISOSBackend, dream_seeds=0) == ["A"], (
        "pure trust-tier scoring should never let a fresh candidate supersede "
        "an established T4 incumbent, even a high-confidence legitimate correction "
        "given 24 cycles -- this is a structural deadlock, not a dream artifact"
    )


def test_authority_only_and_authority_trust_do_not_have_this_deadlock():
    for cls in (AuthorityOnlyPeriodicAISOSBackend, AuthorityTrustNoRecencyPeriodicAISOSBackend, PeriodicAISOSBackend):
        assert _run_correction(cls, dream_seeds=0) == ["B"], (
            f"{cls.__name__} has an authority term differentiating 'user_correction' (1.0) "
            "from 'user' (0.8), so a legitimate correction should still succeed"
        )
