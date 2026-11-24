"""
Authority-first mechanism security stress test
(AISOS_Authority_Security_Stress_Test_Master_Prompt.md Phases D-J).

Deterministic, zero-model-call tests exercising promotion.py's
candidate-vs-live gate directly with precisely controlled
(authority, trust_tier, updated_at) tuples -- attacking the
authority-first mechanism identified in the causal-mechanism milestone
(docs/candidate_lifecycle_trace.md), not merely confirming it.

Full results and interpretation in docs/authority_security_stress_test.md.
"""
from __future__ import annotations

from aisos.memory.record import MemoryRecord, TrustTier, Status
from aisos.memory.promotion import promotion_pipeline, _authority_score
from aisos.memory.contradiction import SOURCE_AUTHORITY
from aisos.security.wal import WriteAheadLog


def _rec(subject, value, source, trust_tier, updated_at, confidence=0.9):
    r = MemoryRecord(subject=subject, value=value, source=source, confidence=confidence,
                      trust_tier=trust_tier, status=Status.CANDIDATE)
    r.updated_at = updated_at
    return r


def _live(subject, value, source, trust_tier, updated_at, confidence=0.9):
    r = _rec(subject, value, source, trust_tier, updated_at, confidence)
    r.status = Status.TRUSTED
    return r


def _run_until_settled(record, published_snapshot, max_calls=8, confidence_threshold=0.55):
    staging = {record.id: record}
    result = None
    for _ in range(max_calls):
        result = promotion_pipeline(staging, published_snapshot, WriteAheadLog(), confidence_threshold=confidence_threshold)
        if result.promoted_to_verified or result.rejected:
            return result
    return result


# ---------------------------------------------------------------------------
# Test D1/D2 -- Equal authority: does the system fall back safely to
# trust_tier, and if trust_tier also ties, to recency?
# ---------------------------------------------------------------------------
def test_D1_equal_authority_newer_attacker_falls_back_to_trust_tier_incumbency():
    """Equal authority (both 'user', 0.8), attacker far more recent: with
    authority tied, the comparison falls through to trust_tier -- the
    live record's T4 vs the attacker's fresh T1 -- so the attacker is
    still rejected, protected by the SAME incumbency backstop
    (docs/mechanism_hypotheses.md H5), not by authority (which is tied
    and therefore uninformative here).
    """
    live = _live("x", "TRUE", "user", TrustTier.T4_VERIFIED, updated_at=100.0)
    attacker = _rec("x", "FALSE", "user", TrustTier.T1_CANDIDATE, updated_at=99999.0)
    result = _run_until_settled(attacker, {live.id: live})
    assert result.promoted_to_verified == [], (
        "equal-authority attacker must still be rejected via trust_tier incumbency, "
        "even when far more recent"
    )


def test_D2_equal_authority_newer_legitimate_correction_ALSO_falls_back_to_trust_tier_incumbency():
    """This is the key pairing (Section 11): under equal authority, a
    LEGITIMATE correction is rejected via the identical mechanism as the
    attacker in D1 -- equal authority does NOT let recency rescue a
    legitimate update. This is a real, disclosed freshness cost of the
    authority-first design, not a bug specific to attacks.
    """
    live = _live("x", "OLD", "user", TrustTier.T4_VERIFIED, updated_at=100.0)
    correction = _rec("x", "NEW", "user", TrustTier.T1_CANDIDATE, updated_at=99999.0)
    result = _run_until_settled(correction, {live.id: live})
    assert result.promoted_to_verified == [], (
        "FINDING: under equal authority, a legitimate correction is rejected by the same "
        "trust-tier incumbency backstop that protects against an equal-authority attacker -- "
        "recency does NOT provide a safe fallback for freshness here"
    )


# ---------------------------------------------------------------------------
# Test E -- High-authority attacker: does a higher-authority attacker win?
# ---------------------------------------------------------------------------
def test_E_higher_authority_attacker_WINS_this_is_a_genuine_failure_boundary():
    """SECURITY FINDING (Section 12): the mechanism cannot distinguish a
    genuinely higher-authority correction from an attacker who has
    (successfully, per docs/authority_provenance.md) claimed a higher
    authority source string. A 'user_correction'-sourced attacker
    (authority 1.0) beats a 'user'-sourced live record (0.8) exactly as
    readily as the legitimate correction in the causal-mechanism
    milestone did -- because nothing in the comparator can tell them
    apart. This is NOT a bug; it is the direct, disclosed consequence of
    authority being self-declared (docs/authority_provenance.md) rather
    than independently verified.
    """
    live = _live("x", "TRUE", "user", TrustTier.T4_VERIFIED, updated_at=100.0)
    attacker = _rec("x", "FALSE", "user_correction", TrustTier.T1_CANDIDATE, updated_at=99999.0)
    result = _run_until_settled(attacker, {live.id: live})
    assert len(result.promoted_to_verified) == 1, (
        "SECURITY FINDING: a higher-(claimed)-authority attacker successfully replaces the "
        "live value -- the authority-first mechanism provides NO protection once the "
        "attacker's claimed source authority exceeds the incumbent's, confirming authority "
        "alone is NOT sufficient against an adversary who can choose their own source label"
    )
    assert result.promoted_to_verified[0].value == "FALSE"


# ---------------------------------------------------------------------------
# Test F -- Low-authority legitimate correction: rejected, same as an attack
# ---------------------------------------------------------------------------
def test_F_lower_authority_legitimate_correction_is_rejected_indistinguishably_from_an_attack():
    """A legitimate correction sourced at LOWER authority than the live
    record (e.g. a 'tool' correcting a 'user' fact) is rejected via the
    exact same mechanism as an attack -- this is an intentional
    conservative security/freshness trade-off (Section 13), not
    automatically a bug, but it means legitimately-lower-authority
    corrections can never propagate no matter how recent or well-evidenced.
    """
    live = _live("x", "OLD", "user", TrustTier.T4_VERIFIED, updated_at=100.0)
    correction = _rec("x", "NEW", "tool", TrustTier.T1_CANDIDATE, updated_at=99999.0, confidence=0.99)
    result = _run_until_settled(correction, {live.id: live})
    assert result.promoted_to_verified == [], (
        "a lower-authority legitimate correction, however recent or high-confidence, must be "
        "rejected -- confirming authority is treated as an absolute hierarchy, not a weighted "
        "signal that evidence/confidence/recency can ever outweigh"
    )
    assert result.rejected, "it must be explicitly REJECTED (not merely held forever) once max_rounds is exhausted"


# ---------------------------------------------------------------------------
# Test G -- Missing / unrecognized authority
# ---------------------------------------------------------------------------
def test_G_unrecognized_source_falls_back_to_the_default_authority_value():
    """SOURCE_AUTHORITY.get(source, 0.4): an unrecognized source string is
    silently treated as authority=0.4 -- BELOW 'tool' (0.6) but above
    'dream' (0.1). This is neither an error nor a conservative
    fail-closed/fail-open policy; it is a specific, undocumented numeric
    default. Direct structural confirmation.
    """
    assert SOURCE_AUTHORITY.get("totally_unrecognized_source_xyz", 0.4) == 0.4
    assert 0.4 < SOURCE_AUTHORITY["tool"]
    assert 0.4 > SOURCE_AUTHORITY["dream"]


def test_G2_candidate_with_missing_authority_vs_established_live_fact_is_rejected():
    live = _live("x", "TRUE", "user", TrustTier.T4_VERIFIED, updated_at=100.0)
    candidate = _rec("x", "UNKNOWN_SOURCE_VALUE", "totally_unrecognized_source_xyz",
                      TrustTier.T1_CANDIDATE, updated_at=99999.0)
    result = _run_until_settled(candidate, {live.id: live})
    assert result.promoted_to_verified == [], (
        "a candidate with an unrecognized ('missing') source authority (default 0.4) must be "
        "rejected against a 0.8-authority live record -- the default value happens to be "
        "conservative here, but only because 0.4 < 0.8, not from any explicit fail-safe policy"
    )


def test_G3_candidate_with_missing_authority_vs_a_low_authority_live_fact_is_ACCEPTED():
    """Confirms G2's rejection is NOT a fail-safe policy: against a
    lower-authority live record (e.g. 'tool', 0.6), the SAME
    unrecognized-source candidate (default 0.4) is still rejected here
    too (0.4 < 0.6) -- but against a 'dream'-sourced live record (0.1),
    it would be ACCEPTED (0.4 > 0.1), demonstrating there is no
    fail-closed behavior for missing authority, only whatever the
    arithmetic default happens to produce.
    """
    live = _live("x", "OLD_DREAM_VALUE", "dream", TrustTier.T4_VERIFIED, updated_at=100.0)
    candidate = _rec("x", "NEW_UNKNOWN_SOURCE_VALUE", "totally_unrecognized_source_xyz",
                      TrustTier.T1_CANDIDATE, updated_at=99999.0)
    result = _run_until_settled(candidate, {live.id: live})
    assert len(result.promoted_to_verified) == 1, (
        "FINDING: 'missing' authority is not fail-safe -- it is accepted or rejected purely "
        "based on where the arithmetic default (0.4) happens to fall relative to the specific "
        "live record's authority, with no explicit handling of the 'unknown' case at all"
    )


# ---------------------------------------------------------------------------
# Test H -- Authority spoofing: can an attacker simply declare high authority?
# ---------------------------------------------------------------------------
def test_H_spoofed_authority_is_accepted_identically_to_genuine_authority():
    """Direct empirical demonstration of docs/authority_provenance.md's
    central finding: nothing distinguishes a genuinely-issued
    source='user_correction' record from one where an attacker (anyone
    able to call remember()) simply supplies that same string. Both
    candidates constructed identically here except for `value` --
    confirming the runtime has no signal to tell them apart, per Section
    24's 'no is_malicious/is_legitimate leakage' rule (there is nothing to
    leak: the system genuinely does not know).
    """
    live = _live("x", "TRUE", "user", TrustTier.T4_VERIFIED, updated_at=100.0)

    genuine = _rec("x", "GENUINE_CORRECTION", "user_correction", TrustTier.T1_CANDIDATE, updated_at=99999.0)
    result_genuine = _run_until_settled(genuine, {live.id: live})

    live2 = _live("x", "TRUE", "user", TrustTier.T4_VERIFIED, updated_at=100.0)
    spoofed = _rec("x", "SPOOFED_VALUE", "user_correction", TrustTier.T1_CANDIDATE, updated_at=99999.0)
    result_spoofed = _run_until_settled(spoofed, {live2.id: live2})

    assert len(result_genuine.promoted_to_verified) == len(result_spoofed.promoted_to_verified) == 1, (
        "SECURITY FINDING: a record with a merely-CLAIMED source='user_correction' is promoted "
        "through the identical code path, in the identical number of rounds, as a genuinely-"
        "issued one -- confirming the mechanism provides no spoofing resistance whatsoever"
    )


# ---------------------------------------------------------------------------
# Test I -- Authority vs. evidence (confidence): can strong evidence
# override lower authority?
# ---------------------------------------------------------------------------
def test_I_high_confidence_cannot_rescue_a_lower_authority_candidate():
    """Confidence is checked ONLY as a separate binary gate (Section 2 of
    promotion.py's own docstring: 'cheap check 2: confidence threshold'),
    reached only AFTER the authority-tuple gate already passed. A
    lower-authority candidate, however high its confidence, never even
    reaches the confidence check -- it is held/rejected at the authority
    comparison first. This directly answers Section 17 (Test 7):
    evidence/confidence is strictly SUBORDINATE to authority in this
    architecture -- it can only ever REJECT an authority-winning
    candidate that fails corroboration, never RESCUE an authority-losing
    one.
    """
    live = _live("x", "TRUE", "user", TrustTier.T4_VERIFIED, updated_at=100.0)
    weak_evidence_low_authority = _rec("x", "FALSE", "tool", TrustTier.T1_CANDIDATE,
                                        updated_at=99999.0, confidence=0.99)
    result = _run_until_settled(weak_evidence_low_authority, {live.id: live})
    assert result.promoted_to_verified == [], (
        "confidence=0.99 (near-maximum evidence) must NOT rescue a lower-authority ('tool', "
        "0.6 vs live's 0.8) candidate -- authority strictly dominates evidence in this design"
    )


def test_I2_low_confidence_CAN_reject_a_higher_authority_candidate():
    """The converse: a higher-authority candidate that wins the tuple
    comparison can still be rejected by the SEPARATE confidence gate if
    its confidence is below threshold -- confirming confidence is a real,
    independent check, just one that is only ever reached after authority
    already decided the ranking.
    """
    live = _live("x", "TRUE", "user", TrustTier.T4_VERIFIED, updated_at=100.0)
    high_authority_low_confidence = _rec("x", "FALSE", "user_correction", TrustTier.T1_CANDIDATE,
                                          updated_at=99999.0, confidence=0.10)
    result = _run_until_settled(high_authority_low_confidence, {live.id: live}, confidence_threshold=0.55)
    assert result.promoted_to_verified == [], (
        "a higher-authority candidate with confidence (0.10) below threshold (0.55) must still "
        "be held/rejected -- confidence is a genuine, independent gate, reached only after "
        "authority already let the candidate through the first check"
    )


# ---------------------------------------------------------------------------
# Test J -- Full tuple boundary analysis (Section 20)
# ---------------------------------------------------------------------------
def test_J1_equal_authority_equal_trust_tier_recency_finally_decides():
    """Isolates the recency fallback precisely: only when BOTH authority
    and trust_tier tie does updated_at become the deciding factor.
    """
    a = MemoryRecord(subject="x", value="A", source="user", confidence=0.9,
                      trust_tier=TrustTier.T2_VALIDATED, status=Status.CANDIDATE)
    a.updated_at = 100.0
    b = MemoryRecord(subject="x", value="B", source="user", confidence=0.9,
                      trust_tier=TrustTier.T2_VALIDATED, status=Status.CANDIDATE)
    b.updated_at = 200.0
    from aisos.memory.contradiction import find_contradictions
    winner = find_contradictions([a, b])[0].winner
    assert winner.value == "B", "with authority AND trust_tier tied, recency (updated_at) must decide"


def test_J2_equal_authority_different_trust_tier_trust_tier_decides_before_recency():
    """Confirms trust_tier is consulted BEFORE recency whenever authority
    ties but trust_tier does not -- the second tuple position genuinely
    matters, independent of which record is more recent.
    """
    a = MemoryRecord(subject="x", value="A", source="user", confidence=0.9,
                      trust_tier=TrustTier.T4_VERIFIED, status=Status.CANDIDATE)
    a.updated_at = 100.0  # older
    b = MemoryRecord(subject="x", value="B", source="user", confidence=0.9,
                      trust_tier=TrustTier.T1_CANDIDATE, status=Status.CANDIDATE)
    b.updated_at = 99999.0  # far more recent
    assert _authority_score(a) > _authority_score(b), (
        "with authority tied, the higher-trust-tier record must win the tuple comparison "
        "even though it is far less recent -- trust_tier is consulted before updated_at"
    )
