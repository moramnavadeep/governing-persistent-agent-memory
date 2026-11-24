"""
Trusted Authority Registry mechanism validation
(AISOS_Trusted_Authority_Provenance_Next_Milestone.md Phase H).

Deterministic, zero-model-call tests. Covers the required 10 boundary
cases (Section 11), plus lifecycle-integrity checks (Phase K) that found
and fixed a real bug: light_sleep.py's context-buffer reconstruction was
silently dropping verified_authority (aisos/core/adapter.py::remember(),
aisos/memory/light_sleep.py -- see their docstrings for the fix).
"""
from __future__ import annotations

from aisos.memory.record import MemoryRecord, TrustTier, Status
from aisos.memory.promotion import promotion_pipeline, verified_authority_score
from aisos.memory.contradiction import SOURCE_AUTHORITY
from aisos.security.wal import WriteAheadLog
from aisos.security.authority_registry import TrustedAuthorityRegistry, UNVERIFIED_AUTHORITY
from aisos.security.checkpoint import new_signing_key
from aisos.agent.condition_backends import TrustedAuthorityPeriodicAISOSBackend, PeriodicAISOSBackend, make_backend


def _rec(subject, value, source, trust_tier, updated_at, confidence=0.9, verified_authority=None):
    r = MemoryRecord(subject=subject, value=value, source=source, confidence=confidence,
                      trust_tier=trust_tier, status=Status.CANDIDATE, verified_authority=verified_authority)
    r.updated_at = updated_at
    return r


def _live(subject, value, source, trust_tier, updated_at, verified_authority):
    r = _rec(subject, value, source, trust_tier, updated_at, verified_authority=verified_authority)
    r.status = Status.TRUSTED
    return r


def _run_until_settled(record, published_snapshot, max_calls=8):
    staging = {record.id: record}
    result = None
    for _ in range(max_calls):
        result = promotion_pipeline(staging, published_snapshot, WriteAheadLog(),
                                     authority_score_fn=verified_authority_score, confidence_threshold=0.55)
        if result.promoted_to_verified or result.rejected:
            return result
    return result


# ---------------------------------------------------------------------------
# 1-2. Baseline: genuine low-authority attacker still rejected, genuine
# high-authority correction still accepted, both with VALID credentials.
# ---------------------------------------------------------------------------
def test_1_genuine_low_authority_attacker_still_rejected():
    registry = TrustedAuthorityRegistry(new_signing_key())
    live = _live("x", "TRUE", "user", TrustTier.T4_VERIFIED, 100.0, verified_authority=registry.verify("user", registry.issue_credential("user")))
    attacker_cred = registry.issue_credential("tool")
    attacker = _rec("x", "FALSE", "tool", TrustTier.T1_CANDIDATE, 99999.0,
                     verified_authority=registry.verify("tool", attacker_cred))
    result = _run_until_settled(attacker, {live.id: live})
    assert result.promoted_to_verified == []


def test_2_genuine_high_authority_correction_still_accepted():
    registry = TrustedAuthorityRegistry(new_signing_key())
    live = _live("x", "OLD", "user", TrustTier.T4_VERIFIED, 100.0, verified_authority=registry.verify("user", registry.issue_credential("user")))
    correction_cred = registry.issue_credential("user_correction")
    correction = _rec("x", "NEW", "user_correction", TrustTier.T1_CANDIDATE, 99999.0,
                       verified_authority=registry.verify("user_correction", correction_cred))
    result = _run_until_settled(correction, {live.id: live})
    assert len(result.promoted_to_verified) == 1
    assert result.promoted_to_verified[0].value == "NEW"


# ---------------------------------------------------------------------------
# 3. THE KEY FIX CONFIRMATION: fake high-authority attacker (no credential)
# is now REJECTED -- reversing docs/authority_security_stress_test.md Test E.
# ---------------------------------------------------------------------------
def test_3_fake_high_authority_attacker_NOW_rejected():
    registry = TrustedAuthorityRegistry(new_signing_key())
    live = _live("x", "TRUE", "user", TrustTier.T4_VERIFIED, 100.0, verified_authority=registry.verify("user", registry.issue_credential("user")))
    # Attacker CLAIMS source="user_correction" (authority 1.0) but presents
    # NO credential -- exactly docs/authority_security_stress_test.md Test
    # E/H's scenario, which previously succeeded.
    fake = _rec("x", "FALSE", "user_correction", TrustTier.T1_CANDIDATE, 99999.0,
                verified_authority=registry.verify("user_correction", None))
    assert fake.verified_authority == UNVERIFIED_AUTHORITY
    result = _run_until_settled(fake, {live.id: live})
    assert result.promoted_to_verified == [], (
        "THE FIX: a candidate claiming a high-authority source with no valid credential must now "
        "be rejected -- this is the exact attack that succeeded under the legacy self-declared-"
        "authority mechanism (docs/authority_security_stress_test.md Test E)"
    )


# ---------------------------------------------------------------------------
# 4. Trusted-source spoof: presenting a credential issued for a DIFFERENT
# source than the one claimed.
# ---------------------------------------------------------------------------
def test_4_credential_issued_for_a_different_source_does_not_verify():
    registry = TrustedAuthorityRegistry(new_signing_key())
    live = _live("x", "TRUE", "user", TrustTier.T4_VERIFIED, 100.0, verified_authority=registry.verify("user", registry.issue_credential("user")))
    mismatched_cred = registry.issue_credential("tool")  # issued for "tool", not "user_correction"
    spoof = _rec("x", "FALSE", "user_correction", TrustTier.T1_CANDIDATE, 99999.0,
                 verified_authority=registry.verify("user_correction", mismatched_cred))
    assert spoof.verified_authority == UNVERIFIED_AUTHORITY, (
        "a credential valid for 'tool' must NOT verify a claim of 'user_correction'"
    )
    result = _run_until_settled(spoof, {live.id: live})
    assert result.promoted_to_verified == []


# ---------------------------------------------------------------------------
# 5. Tampered credential: a single-character mutation must fail verification.
# ---------------------------------------------------------------------------
def test_5_tampered_credential_fails_verification():
    registry = TrustedAuthorityRegistry(new_signing_key())
    cred = registry.issue_credential("user_correction")
    tampered = cred[:-1] + ("0" if cred[-1] != "0" else "1")
    assert registry.verify("user_correction", tampered) == UNVERIFIED_AUTHORITY


# ---------------------------------------------------------------------------
# 6. Missing provenance (no credential at all) fails safe.
# ---------------------------------------------------------------------------
def test_6_missing_credential_fails_safe_to_unverified():
    registry = TrustedAuthorityRegistry(new_signing_key())
    assert registry.verify("user_correction", None) == UNVERIFIED_AUTHORITY
    assert registry.verify("user", None) == UNVERIFIED_AUTHORITY
    # UNVERIFIED_AUTHORITY must be strictly below every registered source,
    # including the lowest legitimate one ("dream", 0.1) -- true fail-safe,
    # not merely "usually loses."
    assert UNVERIFIED_AUTHORITY < min(SOURCE_AUTHORITY.values())


# ---------------------------------------------------------------------------
# 7. Unrecognized source cannot be verified even with a validly-signed
# credential for that exact (nonsense) string -- nothing to verify it
# against.
# ---------------------------------------------------------------------------
def test_7_unrecognized_source_cannot_be_verified_even_with_a_signed_credential():
    registry = TrustedAuthorityRegistry(new_signing_key())
    cred = registry.issue_credential("totally_unregistered_source")
    assert registry.verify("totally_unregistered_source", cred) == UNVERIFIED_AUTHORITY


# ---------------------------------------------------------------------------
# 8. Legacy backend (no registry configured) is completely unaffected --
# regression check confirming full backward compatibility.
# ---------------------------------------------------------------------------
def test_8_legacy_backend_without_a_registry_is_unaffected():
    backend = PeriodicAISOSBackend()
    assert backend.agent.authority_registry is None
    rec = backend.agent.remember("x", "TRUE", source="user", confidence=0.9)
    assert rec.verified_authority is None, "no registry configured -> verified_authority must stay None (legacy path)"


# ---------------------------------------------------------------------------
# 9. THE BUG FOUND DURING THIS MILESTONE'S OWN LIFECYCLE-INTEGRITY CHECK:
# verified_authority must survive light_sleep's context-buffer round-trip.
# ---------------------------------------------------------------------------
def test_9_verified_authority_survives_light_sleep_and_dedup():
    backend = TrustedAuthorityPeriodicAISOSBackend()
    cred = backend.issue_credential("user")
    backend.write("x", "TRUE", "user", 0.9, 0, credential=cred)
    for i in range(1, 20):  # one full period (15) so light_sleep + deep_sleep actually run
        backend.maybe_maintain(i)
    staging = [r for r in backend.agent.store.all_staging() if r.subject == "x"]
    assert staging, "the candidate must still exist in staging after a light_sleep/deep_sleep cycle"
    assert any(r.verified_authority == 0.8 for r in staging), (
        "REGRESSION GUARD: verified_authority must survive light_sleep's reconstruction of the "
        "candidate from the raw context_buffer -- previously (bug found and fixed in this "
        "milestone) the reconstructed duplicate carried verified_authority=None and won dedup's "
        "keep-newer tie-break, silently destroying the verified claim"
    )


# ---------------------------------------------------------------------------
# 10. End-to-end: genuine correction propagates, spoofed claim does not,
# through the FULL pipeline (light_sleep, deep_sleep, dream, promotion,
# wake_verify) -- not just the isolated promotion_pipeline() tests above.
# ---------------------------------------------------------------------------
def test_10_end_to_end_genuine_correction_propagates_spoofed_claim_does_not():
    backend = TrustedAuthorityPeriodicAISOSBackend()
    backend.write("x", "TRUE", "user", 0.9, 0, credential=backend.issue_credential("user"))
    for i in range(1, 50):
        backend.maybe_maintain(i)
    assert [r.value for r in backend.agent.store.read_live("x")] == ["TRUE"]

    backend.write("x", "SPOOFED", "user_correction", 0.9, 50, credential=None)
    for i in range(51, 130):
        backend.maybe_maintain(i)
    assert [r.value for r in backend.agent.store.read_live("x")] == ["TRUE"], (
        "a spoofed claim (no credential) must never become live, end to end"
    )

    backend.write("x", "GENUINE_NEW", "user_correction", 0.95, 130, credential=backend.issue_credential("user_correction"))
    for i in range(131, 220):
        backend.maybe_maintain(i)
    assert [r.value for r in backend.agent.store.read_live("x")] == ["GENUINE_NEW"], (
        "a genuine, validly-credentialed correction must still become live, end to end"
    )


# ---------------------------------------------------------------------------
# Disclosed limitation: replaying a validly-issued credential succeeds --
# this mechanism defeats bare self-declaration, NOT credential theft/replay
# (P5, explicitly out of scope per authority_registry.py's own docstring).
# ---------------------------------------------------------------------------
def test_11_replaying_a_valid_credential_for_a_different_claim_still_verifies_disclosed_limitation():
    registry = TrustedAuthorityRegistry(new_signing_key())
    genuine_cred = registry.issue_credential("user_correction")
    # An attacker who has somehow observed/obtained this credential (out of
    # scope: HOW they obtained it, per the prior milestone's Section 16
    # "no real-world credential theft simulation") can reuse it for an
    # entirely different, later, unrelated claim -- since the credential is
    # bound only to the source label, not to a specific record/time.
    replayed = registry.verify("user_correction", genuine_cred)
    assert replayed == SOURCE_AUTHORITY["user_correction"], (
        "DISCLOSED LIMITATION (docs/trusted_authority_mechanism_report.md Section 19): this "
        "mechanism does not provide replay resistance -- a leaked valid credential remains usable "
        "indefinitely for any future claim of the same source"
    )


def test_12_registry_default_authority_values_match_the_existing_table_unchanged():
    registry = TrustedAuthorityRegistry(new_signing_key())
    for source, value in SOURCE_AUTHORITY.items():
        cred = registry.issue_credential(source)
        assert registry.verify(source, cred) == value, (
            f"the numeric authority meaning of {source!r} must be unchanged by this mechanism -- "
            "only WHO can claim it changes, not what the levels mean"
        )


def test_14_equal_verified_authority_attacker_rejected_via_trust_tier_incumbency():
    """Completes the required final table's 'equal-authority attacker'
    row: with a VALID credential (genuinely equal authority, not spoofed),
    the tuple ties at the authority element and falls through to
    trust_tier, exactly as the legacy mechanism's D1 test found -- this
    milestone does not change that structural fallback.
    """
    registry = TrustedAuthorityRegistry(new_signing_key())
    live = _live("x", "TRUE", "user", TrustTier.T4_VERIFIED, 100.0, verified_authority=registry.verify("user", registry.issue_credential("user")))
    attacker = _rec("x", "FALSE", "user", TrustTier.T1_CANDIDATE, 99999.0,
                     verified_authority=registry.verify("user", registry.issue_credential("user")))
    result = _run_until_settled(attacker, {live.id: live})
    assert result.promoted_to_verified == []


def test_15_equal_verified_authority_legitimate_correction_ALSO_rejected():
    """Symmetric pairing (Section 12): an equal-verified-authority
    legitimate correction is rejected via the identical mechanism -- this
    milestone does not change the pre-existing equal-authority freshness
    cost (docs/authority_security_stress_test.md D2), only the
    self-declaration vulnerability.
    """
    registry = TrustedAuthorityRegistry(new_signing_key())
    live = _live("x", "OLD", "user", TrustTier.T4_VERIFIED, 100.0, verified_authority=registry.verify("user", registry.issue_credential("user")))
    correction = _rec("x", "NEW", "user", TrustTier.T1_CANDIDATE, 99999.0,
                       verified_authority=registry.verify("user", registry.issue_credential("user")))
    result = _run_until_settled(correction, {live.id: live})
    assert result.promoted_to_verified == []


def test_16_missing_provenance_is_a_true_fail_safe_regardless_of_which_live_record_it_faces():
    """Improvement over the legacy mechanism's G2/G3 finding (no principled
    fail-safe policy -- outcome depended on which live record happened to
    exist). Under the trusted-authority mechanism, UNVERIFIED_AUTHORITY
    (0.0) is unconditionally below every registered source, so a
    missing-credential candidate is rejected against EVERY possible live
    record, not just some.
    """
    registry = TrustedAuthorityRegistry(new_signing_key())
    # Even against the LOWEST-authority legitimate live source ("dream", 0.1):
    live = _live("x", "OLD_DREAM_VALUE", "dream", TrustTier.T4_VERIFIED, 100.0, verified_authority=0.1)
    candidate = _rec("x", "NEW_UNVERIFIED", "totally_unregistered_source", TrustTier.T1_CANDIDATE, 99999.0,
                      verified_authority=registry.verify("totally_unregistered_source", None))
    result = _run_until_settled(candidate, {live.id: live})
    assert result.promoted_to_verified == [], (
        "unlike the legacy mechanism's G3 finding (missing authority ACCEPTED against a low-"
        "authority live record), the trusted-authority mechanism must reject it here too -- a "
        "genuine, unconditional fail-safe"
    )


def test_13_make_backend_registers_the_trusted_authority_condition():
    backend = make_backend("periodic_aisos_trusted_authority")
    assert isinstance(backend, TrustedAuthorityPeriodicAISOSBackend)
    assert backend.agent.authority_registry is not None
