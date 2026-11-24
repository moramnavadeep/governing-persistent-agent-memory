"""
Trusted Authority Registry (AISOS_Trusted_Authority_Provenance_Next_Milestone.md
Phases C-D; extended by AISOS_Next_Milestone_Replay_Provenance.md Phases
D-F for replay resistance). Closes two vulnerabilities in sequence:

1. Authority-spoofing (docs/authority_security_stress_test.md Test H):
   previously, any caller of AISOSAgent.remember(source=...) received the
   authority SOURCE_AUTHORITY[source] purely because it claimed that
   source string, with zero verification. Closed by requiring an
   HMAC-signed credential (Model C, "authenticated provenance").

2. Credential replay (docs/credential_replay_report.md): the ORIGINAL
   credential design, sign(signing_key, source), was a pure deterministic
   function of `source` alone -- every credential ever issued for a given
   source was the IDENTICAL string, making "replay" trivial by
   construction (anyone who observed one credential had, in effect, THE
   credential for that source, forever). Closed by binding each issuance
   to a fresh, unique nonce and tracking which nonces have already been
   consumed (Model C + Model E hybrid, per that milestone's Phase D/F
   selection: nonce/context binding + an explicit credential-use
   registry) -- the smallest mechanism that makes replay detectable and
   rejectable without introducing expiry timers, sequence numbers, or any
   new cryptographic primitive.

Reuses aisos/security/checkpoint.py's existing HMAC-SHA256 sign/verify_sig
primitives throughout -- no new cryptographic infrastructure.

Scope, disclosed rather than hidden (docs/credential_replay_report.md
Section 17): replay protection is enforced by an in-memory used-nonce set
scoped to ONE TrustedAuthorityRegistry instance's lifetime. It does not
cover: multiple registry instances sharing a signing_key (cross-instance/
cross-"session" replay remains possible unless the used-nonce state
itself is externally shared -- explicitly out of scope, matching the
prior milestone's "do not implement distributed revocation" instruction);
signing-key compromise; or direct in-process mutation of an already-
verified MemoryRecord (a fundamentally different, stronger threat model,
also explicitly out of scope per that milestone's own Limitation 4).
"""
from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Optional, Union

from aisos.memory.contradiction import SOURCE_AUTHORITY
from aisos.security.checkpoint import sign, verify_sig
from aisos.security.durable_nonce_ledger import DurableNonceLedger

# A candidate with no credential, an invalid one, or a REPLAYED one is
# treated as strictly less authoritative than every registered source --
# including "dream" (0.1), the lowest value SOURCE_AUTHORITY assigns to a
# LEGITIMATE, internally-generated source. Fail-safe: unknown/replayed
# provenance never receives privileged authority, and can never win a
# comparison against ANY genuinely-registered live record. This is P-R4
# ("replay must never increase authority") by construction: a replayed
# credential's outcome is identical to a missing one.
UNVERIFIED_AUTHORITY: float = 0.0


class TrustedAuthorityRegistry:
    """Verifies that a claimed `source` label is entitled to the authority
    SOURCE_AUTHORITY assigns it, via a single-use, HMAC-signed credential
    bound to a fresh nonce -- the candidate itself never gets to assign
    its own authority by declaration alone, and a credential cannot be
    replayed once consumed.
    """

    def __init__(self, signing_key: bytes, trusted_sources: Optional[dict] = None,
                 persistence_path: Optional[Union[str, Path]] = None):
        self.signing_key = signing_key
        # Defaults to the existing SOURCE_AUTHORITY table: this mechanism
        # changes WHO can claim a given authority level, not WHAT the
        # levels mean -- preserving every existing numeric authority value
        # unmodified.
        self.trusted_sources = dict(trusted_sources) if trusted_sources is not None else dict(SOURCE_AUTHORITY)
        # persistence_path: opt-in (AISOS_Restart_Replay_Closure_Milestone.md),
        # default None = fully backward compatible with every prior
        # milestone's behavior (in-memory only, does not survive restart).
        # When set, a DurableNonceLedger is constructed at this path;
        # construction FAILS CLOSED (raises LedgerCorruptionError) if the
        # file exists but is corrupted/tampered, rather than silently
        # starting from an empty (falsely "nothing consumed yet") state --
        # see durable_nonce_ledger.py for the full design rationale.
        self._durable_ledger = DurableNonceLedger(persistence_path) if persistence_path is not None else None
        # Replay-prevention state (AISOS_Next_Milestone_Replay_Provenance.md
        # Model E, "explicit credential-use registry"): every nonce, once
        # successfully verified, is recorded here and can never verify
        # again -- single-use by construction, regardless of how much time
        # has passed (no expiry timer needed) or what different
        # subject/value/candidate the replay attempt targets (the nonce
        # alone is the unit of "context" this mechanism binds to -- see
        # docs/credential_replay_security_property.md P-R1). When a
        # durable ledger is configured, this in-memory set is seeded from
        # it at construction (recovery reconstruction) and is otherwise a
        # cache -- the ledger's own on-disk state is authoritative.
        self._used_nonces: set[str] = set(self._durable_ledger._consumed) if self._durable_ledger is not None else set()
        # Comprehensive-security-audit milestone (AISOS_Comprehensive_
        # Security_Audit_Next_Milestone.md Phase Q): the nonce
        # check-then-add below is a classic TOCTOU pattern and was found,
        # via a forced-race-window test, to allow the SAME single-use
        # credential to be accepted by MULTIPLE concurrent verify() calls
        # -- i.e. concurrent verification could defeat single-use
        # enforcement entirely. Fixed with the same threading.Lock
        # pattern MemoryStore and WriteAheadLog already use elsewhere in
        # this codebase (no new concurrency-control approach introduced).
        self._lock = threading.Lock()
        # Observability counter (answers Phase B Q17, "is replay currently
        # detectable?" -- yes, and countable): incremented every time
        # verify() rejects a credential specifically because its nonce was
        # already consumed, distinct from other rejection reasons.
        self.replay_attempts_detected: int = 0

    def issue_credential(self, source: str) -> str:
        """Simulates out-of-band credential issuance to a genuinely
        authorized caller. Each call returns a FRESH, unique credential
        (a random nonce plus its HMAC signature over `source|nonce`) --
        two calls for the identical source never produce the same
        credential, which is what makes single-use enforcement meaningful
        (previously, issue_credential(source) was a pure function of
        `source` alone, so every issuance for a given source was
        byte-for-byte identical).
        """
        nonce = uuid.uuid4().hex
        signature = sign(self.signing_key, f"{source}|{nonce}")
        return f"{nonce}:{signature}"

    def verify(self, source: str, credential: Optional[str]) -> float:
        """Returns the VERIFIED authority for this (source, credential)
        pair. Fails safe to UNVERIFIED_AUTHORITY whenever:
          - no credential is supplied (bare self-declaration);
          - the credential is malformed (not a `nonce:signature` pair);
          - `source` is not in this registry's trusted_sources table;
          - the credential does not verify against `source` under this
            registry's signing key (tampered, or issued for a different
            source -- the signed message binds both);
          - the credential's nonce has ALREADY been consumed by a prior
            verify() call (a replay attempt, of any kind: immediate,
            delayed, or targeting a different candidate/subject/value --
            the nonce alone determines reuse, not what it is being
            reused FOR).
        A successful verification consumes the nonce: the identical
        credential can never verify a second time.
        """
        if credential is None:
            return UNVERIFIED_AUTHORITY
        if source not in self.trusted_sources:
            return UNVERIFIED_AUTHORITY
        if ":" not in credential:
            return UNVERIFIED_AUTHORITY
        nonce, _, signature = credential.partition(":")
        if not verify_sig(self.signing_key, f"{source}|{nonce}", signature):
            return UNVERIFIED_AUTHORITY
        # The nonce check-then-add below is the entire security-critical
        # section: held under self._lock as a single atomic unit so that
        # concurrent verify() calls for the SAME credential cannot both
        # observe "not yet used" before either records it as used (the
        # exact race a forced-window test demonstrated pre-fix).
        with self._lock:
            if nonce in self._used_nonces:
                self.replay_attempts_detected += 1
                return UNVERIFIED_AUTHORITY
            if self._durable_ledger is not None:
                # Durable write happens BEFORE the in-memory set is
                # updated (AISOS_Restart_Replay_Closure_Milestone.md
                # Phase E, "atomic consumption"): if the process crashes
                # after this line returns but before the function
                # returns, a fresh registry reconstructing from this same
                # file will correctly see the nonce as consumed. If
                # try_consume() itself returns False here, the ledger
                # (the authoritative store when persistence is enabled)
                # already considered it consumed even though this
                # registry's own in-memory cache did not yet know --
                # fail closed by treating this identically to an
                # in-memory replay, rather than proceeding.
                newly_consumed = self._durable_ledger.try_consume(nonce)
                if not newly_consumed:
                    self.replay_attempts_detected += 1
                    return UNVERIFIED_AUTHORITY
            self._used_nonces.add(nonce)
        return self.trusted_sources[source]
