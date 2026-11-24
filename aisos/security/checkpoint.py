"""
Signing / verification for checkpoints and WAL bundles (Technical Spec
Section 14; v2 Section 6.6, SIGN(...) / VERIFY_SIG(...)).

Reference implementation uses HMAC-SHA256 with a process-local maintenance
signing key. Section 10 (Limitations) and Appendix B.2 both state that key
management/compromise is explicitly out of scope for this model -- the key
here is generated once per AISOSAgent instance and held in memory, which is
sufficient to exercise the verify/reject/rollback logic under test, but is
*not* a production key-management design.
"""
from __future__ import annotations

import hashlib
import hmac
import os


def new_signing_key() -> bytes:
    return os.urandom(32)


def checkpoint_hash(published_ids_versions: list[tuple[str, int]]) -> str:
    payload = "|".join(f"{i}:{v}" for i, v in sorted(published_ids_versions))
    return hashlib.sha256(payload.encode()).hexdigest()


def sign(key: bytes, message: str) -> str:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_sig(key: bytes, message: str, signature: str) -> bool:
    """Fails safe (returns False) for ANY malformed `signature`, rather
    than raising. Found via the comprehensive-security-audit milestone's
    Phase S fuzzing (AISOS_Comprehensive_Security_Audit_Next_Milestone.md):
    hmac.compare_digest() raises TypeError when comparing two `str`
    operands if either contains non-ASCII characters -- so a credential
    carrying a non-ASCII "signature" portion previously CRASHED
    verification (an unhandled exception propagating out of
    AISOSAgent.remember()) instead of being rejected as invalid. Fixed by
    comparing as `bytes` (UTF-8-encoded), which hmac.compare_digest
    accepts for arbitrary byte content with no ASCII restriction, and by
    catching any other malformed-input exception defensively.
    """
    expected = sign(key, message)
    try:
        return hmac.compare_digest(expected.encode("ascii"), str(signature).encode("utf-8"))
    except (TypeError, ValueError, AttributeError):
        return False
