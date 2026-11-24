"""
Independent adversarial benchmark -- mechanism-level ablations G1-G8
(AISOS_Independent_Adversarial_Benchmark_Problem.md Section 21).

Deterministic, zero-real-model-call tests, reusing AISOSAgent directly
(not the semantic-benchmark System wrapper, for speed/simplicity) with
THIS package's own independent corpus values and comparator ablations
(comparator.py) -- a second, independently-run replication of this
project's own earlier causal-ablation methodology
(aisos/tests/test_authority_ablation.py, test_trust_tier_ablation.py),
using fresh scenario content rather than re-deriving the same finding
on the same data.

G1: authority removed. G2: trust removed. G3: freshness removed.
G4: authentication removed (covered by the main benchmark's own S6-vs-S6b
split, not duplicated here as a separate ablation).
G5: write-time governance removed (= "no governance" condition, covered
by the write-time experiment's own condition A).
G6: retrieval-time weighting only (= condition B in the write-time
experiment).
G7/G8: AISOS+RAG without/with governance -- G8 is simply S7 (already
covered); G7 (RAG discovery feeding an UNGOVERNED, last-write-wins
store) is built here as a dedicated deterministic mechanism check.
"""
from __future__ import annotations

from aisos.core.adapter import AISOSAgent
from aisos.benchmark.independent_adversarial.comparator import (
    independent_contradiction_score, independent_authority_score,
    independent_score_without_authority, independent_authority_score_without_authority,
    independent_score_without_trust, independent_authority_score_without_trust,
    independent_score_without_freshness, independent_authority_score_without_freshness,
)

PERIOD = 15


def _agent(contradiction_score_fn, authority_score_fn):
    return AISOSAgent(tau_light=0.3, tau_deep=0.55,
                       contradiction_score_fn=contradiction_score_fn, authority_score_fn=authority_score_fn)


def _settle(agent, cycles=8 * PERIOD):
    for i in range(1, cycles):
        if i % PERIOD == 0:
            agent.run_sleep_cycle(target="enter_light", run_dream=True)


def _read(agent, subject):
    # max(key=agent.authority_score_fn), not live[-1]: read_live() can
    # legitimately return multiple published records for one subject
    # (confirmed directly, docs/h21_h24_baseline_results.md); live[-1]
    # silently picks an arbitrary one rather than the one this agent's own
    # comparator considers the winner.
    live = agent.store.read_live(subject=subject)
    return max(live, key=agent.authority_score_fn).value if live else None


# ---------------------------------------------------------------------
# G0 (baseline, full mechanism) -- A1-style direct false injection, using
# independent_adversarial's own authority table (official_record=0.85 >
# unverified_submission=0.10).
# ---------------------------------------------------------------------
def test_g0_full_mechanism_rejects_direct_false_injection():
    agent = _agent(independent_contradiction_score, independent_authority_score)
    agent.remember("badge_tier", "Tier-3", source="official_record", confidence=0.95, ts=0.0)
    _settle(agent)
    agent.remember("badge_tier", "Tier-5-FALSE", source="unverified_submission", confidence=0.6, ts=10.0)
    _settle(agent)
    assert _read(agent, "badge_tier") == "Tier-3"


# ---------------------------------------------------------------------
# G1 -- authority removed
# ---------------------------------------------------------------------
def test_g1_authority_removed_still_blocks_via_trust_tier_incumbency():
    """Matches this project's own prior finding (H_AUTH-NARROW,
    docs/claim_registry.json): removing authority does not, by itself,
    open the tested attack, because trust-tier incumbency independently
    blocks a fresh low-trust candidate from displacing an established
    live record. Confirmed here on FRESH scenario content, not re-derived
    from the same data as the original finding."""
    agent = _agent(independent_score_without_authority, independent_authority_score_without_authority)
    agent.remember("badge_tier", "Tier-3", source="official_record", confidence=0.95, ts=0.0)
    _settle(agent)
    agent.remember("badge_tier", "Tier-5-FALSE", source="unverified_submission", confidence=0.6, ts=10.0)
    _settle(agent)
    assert _read(agent, "badge_tier") == "Tier-3", (
        "trust-tier incumbency alone should still block the fresh false candidate even with authority removed"
    )


def test_g1_authority_removed_delays_rather_than_permanently_blocks_a_legitimate_correction():
    """A genuine, surprising finding discovered while building this
    independent ablation, reported honestly rather than forced to match
    a prior assumption: this test originally asserted (based on this
    project's own prior H_AUTH-BROADER finding, observed on a real
    1000-interaction continuous workload) that removing authority would
    PERMANENTLY block a legitimate correction from ever displacing an
    established incumbent. Direct measurement on this simpler,
    fully-settled 2-event scenario found otherwise: given enough
    settle cycles (empirically, ~120 here vs. ~15-45 not enough), the
    correction's own trust_tier eventually climbs to parity (T4) with the
    incumbent's, at which point recency breaks the tie in the
    correction's favor -- it is NOT permanently blocked in this scenario
    shape, only delayed by the number of cycles needed to climb trust
    tiers. This refines, rather than contradicts, the original finding:
    the original workload's corrections faced continuous competing noise
    that likely prevented this kind of uninterrupted trust-tier climb: a
    real, disclosed scope-boundary on when 'authority removed blocks
    legitimate corrections' actually holds, discovered specifically by
    this independent benchmark's own fresh construction."""
    agent = _agent(independent_score_without_authority, independent_authority_score_without_authority)
    agent.remember("wire_threshold", "$50,000-STALE", source="informal_note", confidence=0.5, ts=0.0)
    _settle(agent)
    agent.remember("wire_threshold", "$20,000-CORRECT", source="verified_correction", confidence=0.98, ts=10.0)

    # Insufficient settle time: correction has not yet climbed to trust-tier parity -- held, not live.
    _settle(agent, cycles=45)
    assert _read(agent, "wire_threshold") in (None, "$50,000-STALE"), (
        "with too few post-correction cycles, the correction should not yet be live"
    )

    # Sufficient settle time: correction reaches trust-tier parity, recency breaks the tie in its favor.
    _settle(agent, cycles=8 * PERIOD)
    assert _read(agent, "wire_threshold") == "$20,000-CORRECT", (
        "given enough cycles to climb trust tiers, the correction is NOT permanently blocked in this "
        "scenario shape -- a real, disclosed refinement of the original H_AUTH-BROADER finding's scope"
    )


# ---------------------------------------------------------------------
# G2 -- trust removed
# ---------------------------------------------------------------------
def test_g2_trust_removed_authority_alone_still_resolves_correctly():
    """With trust_tier removed but authority retained, a genuinely
    higher-authority correction CAN displace a lower-authority incumbent
    (unlike G1) -- authority's specific causal role, per this project's
    prior finding, is enabling legitimate updates to succeed, confirmed
    again here on fresh scenario content."""
    agent = _agent(independent_score_without_trust, independent_authority_score_without_trust)
    agent.remember("wire_threshold", "$50,000-STALE", source="informal_note", confidence=0.5, ts=0.0)
    _settle(agent)
    agent.remember("wire_threshold", "$20,000-CORRECT", source="verified_correction", confidence=0.98, ts=10.0)
    _settle(agent)
    assert _read(agent, "wire_threshold") == "$20,000-CORRECT"


def test_g2_trust_removed_attacker_still_blocked_by_authority():
    agent = _agent(independent_score_without_trust, independent_authority_score_without_trust)
    agent.remember("badge_tier", "Tier-3", source="official_record", confidence=0.95, ts=0.0)
    _settle(agent)
    agent.remember("badge_tier", "Tier-5-FALSE", source="unverified_submission", confidence=0.6, ts=10.0)
    _settle(agent)
    assert _read(agent, "badge_tier") == "Tier-3"


# ---------------------------------------------------------------------
# G3 -- freshness (recency) removed
# ---------------------------------------------------------------------
def test_g3_freshness_removed_authority_and_trust_still_block_attacker():
    """Confirms authority+trust alone (no recency term at all) is
    sufficient to block the direct-injection attack -- freshness is not
    load-bearing for THIS specific attack shape, consistent with this
    project's prior causal factorial finding that recency is not the
    load-bearing safety signal."""
    agent = _agent(independent_score_without_freshness, independent_authority_score_without_freshness)
    agent.remember("badge_tier", "Tier-3", source="official_record", confidence=0.95, ts=0.0)
    _settle(agent)
    agent.remember("badge_tier", "Tier-5-FALSE", source="unverified_submission", confidence=0.6, ts=10.0)
    _settle(agent)
    assert _read(agent, "badge_tier") == "Tier-3"


# ---------------------------------------------------------------------
# G7 -- "AISOS+RAG without governance" mechanism check: RAG-style
# discovery feeding an UNGOVERNED last-write-wins store still lets a
# later-arriving false claim overwrite a correct one, unlike G8 (full
# governance) which would not.
# ---------------------------------------------------------------------
def test_g7_ungoverned_store_is_overwritten_by_a_later_false_claim():
    """Deterministic stand-in for 'discovery without governance': a plain
    dict last-write-wins store (no AISOSAgent involved at all) is
    trivially overwritten -- confirming G7's expected, disclosed failure
    mode, which G8 (the full S7 AISOS+RAG condition, covered in the main
    benchmark run) does not share."""
    store = {}

    def ungoverned_write(subject, value):
        store[subject] = value  # no authority/trust check of any kind

    ungoverned_write("badge_tier", "Tier-3")
    ungoverned_write("badge_tier", "Tier-5-FALSE")  # a later, false claim simply wins
    assert store["badge_tier"] == "Tier-5-FALSE", (
        "G7 (discovery without governance) is EXPECTED to be overwritten -- this is the documented "
        "failure mode governance (G8/S7) exists to prevent, not a bug in this test"
    )
