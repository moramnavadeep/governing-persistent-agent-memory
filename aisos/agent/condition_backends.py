"""
The four memory-maintenance conditions the 1,000-interaction experiment
compares (AISOS_1000_INTERACTION_EXPERIMENT_PROMPT.md Section 2), sharing
one interface so the long-running-agent harness can drive all four
identically. Each condition wraps the SAME underlying classes E1/E3/E4/E7
already use (NaiveKV-equivalent, ContinuousMVCCBaseline, AISOSAgent) rather
than reimplementing them, for the same reason E7 reused E3's
AuthenticatedContinuousWrapper: a fair comparison requires the fairness to
be structural (literally the same class), not asserted.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from aisos.core.adapter import AISOSAgent
from aisos.core.lifecycle import TransitionRecord
from aisos.baselines.continuous_mvcc import ContinuousMVCCBaseline
from aisos.dream.dream_phase import DreamBudget
from aisos.memory.record import Status


@dataclass
class MaintenanceEvent:
    interaction_id: int
    kind: str  # "sleep_entry" | "cycle_complete" | "rollback"
    detail: dict


class ConditionBackend(ABC):
    condition_name: str

    @abstractmethod
    def write(self, subject: str, value: str, source: str, confidence: float, interaction_id: int) -> None: ...

    @abstractmethod
    def read(self, subject: str) -> Optional[str]: ...

    @abstractmethod
    def read_all_values(self, subject: str) -> list[str]:
        """All currently-live/visible values for subject -- >1 distinct
        value here is a store-level contradiction, independent of what any
        model says about it."""
        ...

    @abstractmethod
    def maybe_maintain(self, interaction_id: int) -> None: ...

    @abstractmethod
    def snapshot_metrics(self) -> dict: ...

    def lifecycle_trace(self) -> list[dict]:
        return []

    def scheduler_trace(self) -> list[dict]:
        return []

    def maintenance_events(self) -> list[MaintenanceEvent]:
        return []

    def apply_context_reset(self) -> bool:
        """Returns whether a reset was actually applied -- False for
        conditions with no short-term tier to reset (no_sleep_naive,
        continuous_mvcc), which is itself a meaningful, recorded fact
        about those architectures rather than a no-op to hide."""
        return False


class NoSleepBackend(ConditionBackend):
    """Condition A: no maintenance discipline. Last-write-wins, no
    authority weighting -- the deliberate floor (E1's NaiveKV, reused)."""
    condition_name = "no_sleep_naive"

    def __init__(self):
        self._store: dict[str, str] = {}
        self.write_count = 0

    def write(self, subject, value, source, confidence, interaction_id):
        self._store[subject] = value
        self.write_count += 1

    def read(self, subject):
        return self._store.get(subject)

    def read_all_values(self, subject):
        v = self._store.get(subject)
        return [v] if v is not None else []

    def maybe_maintain(self, interaction_id):
        pass

    def snapshot_metrics(self):
        return {"store_size": len(self._store), "cycles_run": 0, "lifecycle_state": "N/A", "sleep_pressure": None}


class ContinuousMVCCBackend(ConditionBackend):
    """Condition B: the required competent continuous baseline (reused
    directly, unmodified, from aisos/baselines/continuous_mvcc.py)."""
    condition_name = "continuous_mvcc"

    def __init__(self):
        self.baseline = ContinuousMVCCBaseline()

    def write(self, subject, value, source, confidence, interaction_id):
        self.baseline.write(subject=subject, value=value, source=source, confidence=confidence)

    def read(self, subject):
        rec = self.baseline.snapshot_read([subject]).get(subject)
        return rec.value if rec else None

    def read_all_values(self, subject):
        rec = self.baseline.snapshot_read([subject]).get(subject)
        return [rec.value] if rec else []

    def maybe_maintain(self, interaction_id):
        pass  # continuous: consolidation happens per-write, inside write() itself

    def snapshot_metrics(self):
        n_versions = sum(len(h) for h in self.baseline._history.values())
        return {
            "store_size": n_versions, "cycles_run": 0, "lifecycle_state": "N/A",
            "sleep_pressure": None, "contradiction_events": self.baseline.contradiction_events,
        }


class _AISOSBackendBase(ConditionBackend):
    """Shared plumbing for the two AISOS conditions: lifecycle-transition
    and maintenance-event capture (Section 8), read via the same
    short-term/long-term two-tier strategy as
    aisos/agent/long_running_agent.py.
    """

    def __init__(
        self, tau_light: float = 0.3, tau_deep: float = 0.55, dream_seeds: int = 4,
        contradiction_score_fn=None, authority_score_fn=None, authority_registry=None,
        verify_provenance_integrity: bool = False, verify_continuous_integrity: bool = False,
    ):
        self._transitions: list[TransitionRecord] = []
        self._maint_events: list[MaintenanceEvent] = []
        agent_kwargs = dict(
            tau_light=tau_light, tau_deep=tau_deep, dream_budget=DreamBudget(n_seeds=dream_seeds),
            on_transition=self._on_transition,
        )
        # Only override AISOSAgent's own defaults (the real authority-aware
        # scorers) when a caller explicitly supplies an alternative -- this
        # is what keeps PeriodicAISOSBackend byte-for-byte identical to its
        # pre-ablation behavior (docs/authority_ablation_spec.md's
        # "everything else unchanged" requirement) while letting
        # AuthorityAblatedPeriodicAISOSBackend below override just these two.
        if contradiction_score_fn is not None:
            agent_kwargs["contradiction_score_fn"] = contradiction_score_fn
        if authority_score_fn is not None:
            agent_kwargs["authority_score_fn"] = authority_score_fn
        if authority_registry is not None:
            agent_kwargs["authority_registry"] = authority_registry
        if verify_provenance_integrity:
            agent_kwargs["verify_provenance_integrity"] = True
        if verify_continuous_integrity:
            agent_kwargs["verify_continuous_integrity"] = True
        self.agent = AISOSAgent(**agent_kwargs)
        self._prev_cycle_count = 0
        self._current_interaction_id = 0
        self._last_cycle_interaction_id = 0

    def _on_transition(self, rec: TransitionRecord) -> None:
        self._transitions.append(rec)

    def write(self, subject, value, source, confidence, interaction_id, credential=None):
        # credential: optional, ignored by every condition except
        # TrustedAuthorityPeriodicAISOSBackend below (whose agent has an
        # authority_registry configured) -- added as a trailing optional
        # kwarg so the existing 5-positional-arg call sites throughout the
        # codebase (experiments, tests) are entirely unaffected.
        self.agent.remember(subject=subject, value=value, source=source, confidence=confidence, credential=credential)

    def read(self, subject):
        """Transient Read-Window Closure Milestone
        (AISOS_Transient_Read_Window_Closure_Next_Milestone.md): the
        fallback below (used only when nothing for `subject` has reached
        T4_VERIFIED/`_published` yet) previously scanned `context_buffer`
        for the chronologically most recent raw write, with NO comparison
        logic at all -- a low-authority attack write landing after a
        higher-authority, not-yet-consolidated candidate for the same
        subject could win purely by recency (docs/read_window_causal_matrix.md
        H1, confirmed the root cause: 10/10 deterministic reproductions,
        closed by this exact change and nothing else).

        Fix (Option B, docs/read_window_fix_design.md): compare `_staging`
        CANDIDATE-status records for this subject using the SAME
        `authority_score_fn` comparator `promotion_pipeline()` and
        `find_contradictions()` already use -- not a new, independently
        derived rule (I-RW4). `_staging` (not `context_buffer`) is used
        because `AISOSAgent.remember()` writes a fully-typed `MemoryRecord`
        (trust_tier, status, verified_authority) into `_staging`
        SYNCHRONOUSLY, before any sleep cycle runs -- richer and simpler
        to compare than reconstructing metadata from `context_buffer`'s
        raw dicts, which are untouched by this fix and still serve their
        original role for `light_sleep`/`apply_context_reset()`.

        Filtering to `status == Status.CANDIDATE` enforces I-RW3/I-RW5:
        ARCHIVED/REJECTED candidates (already correctly resolved away)
        are never resurrected, and TRUSTED-status records never reach this
        branch at all (their subject would already have matched
        `read_live()` above, since promotion always adds a T4_VERIFIED
        record to `_published` in the same step it sets status=TRUSTED).
        """
        trusted = self.agent.store.read_live(subject=subject)
        if trusted:
            return trusted[-1].value
        candidates = [
            r for r in self.agent.store.all_staging()
            if r.subject == subject and r.status == Status.CANDIDATE
        ]
        if not candidates:
            return None
        best = max(candidates, key=self.agent.authority_score_fn)
        return best.value

    def read_all_values(self, subject):
        trusted = self.agent.store.read_live(subject=subject)
        return list({r.value for r in trusted})

    def _run_cycle_and_record(self, interaction_id: int, target: str, run_dream: bool) -> None:
        before = len(self.agent.cycle_reports)
        self._maint_events.append(MaintenanceEvent(interaction_id, "sleep_entry", {"target": target}))
        report = self.agent.run_sleep_cycle(target=target, run_dream=run_dream)
        after = len(self.agent.cycle_reports)
        self._last_cycle_interaction_id = interaction_id
        self._maint_events.append(MaintenanceEvent(
            interaction_id, "cycle_complete",
            {"aborted": report.aborted, "stage_reached": report.stage_reached,
             "verify_passed": report.verify.passed if report.verify else None},
        ))
        if report.verify is not None and not report.verify.passed:
            self._maint_events.append(MaintenanceEvent(interaction_id, "rollback", {"reason": report.verify.reason}))

    def snapshot_metrics(self):
        return {
            "store_size": len(self.agent.store.all_staging()) + len(self.agent.store.read_live()),
            "live_memory_count": len(self.agent.store.read_live()),
            "cycles_run": len(self.agent.cycle_reports),
            "lifecycle_state": self.agent.lifecycle.state.value,
            "sleep_pressure": self.agent.scheduler.last_pressure,
            "rollback_count": self.agent.rollback_count,
        }

    def lifecycle_trace(self):
        return [
            {"ts": r.ts, "from_state": r.from_state.value, "event": r.event, "to_state": r.to_state.value}
            for r in self._transitions
        ]

    def maintenance_events(self):
        return list(self._maint_events)

    def apply_context_reset(self) -> bool:
        # Publish-Race / Staging-Consistency Closure Milestone: protected
        # by the same agent-level lock remember()/run_sleep_cycle() now use
        # for every context_buffer mutation, for the identical reason --
        # an unlocked wholesale reassignment here could otherwise discard
        # a write landing in the same narrow window.
        with self.agent._lock:
            self.agent.context_buffer = []
        return True


class PeriodicAISOSBackend(_AISOSBackendBase):
    """Condition C: AISOS's own algorithms, forced onto a fixed schedule
    instead of pressure-driven -- isolates "discrete cycles" from
    "signal-driven scheduling" (matches E1's aisos_periodic condition)."""
    condition_name = "periodic_aisos"

    def __init__(self, period: int = 15, **kw):
        super().__init__(**kw)
        self.period = period
        self._scheduler_trace: list[dict] = []

    def maybe_maintain(self, interaction_id):
        if interaction_id > 0 and interaction_id % self.period == 0:
            self._scheduler_trace.append({"interaction_id": interaction_id, "decision": "enter_light (periodic)"})
            self._run_cycle_and_record(interaction_id, target="enter_light", run_dream=True)

    def scheduler_trace(self):
        return list(self._scheduler_trace)


class AuthorityAblatedPeriodicAISOSBackend(PeriodicAISOSBackend):
    """Condition C3 (AISOS_Authority_Ablation_Next_Milestone_Prompt.md):
    byte-for-byte identical to PeriodicAISOSBackend -- same scheduler
    (fixed period), same lifecycle, same sleep/consolidation/promotion
    pipeline, same workload, same model -- with ONLY the two authority-
    reading scoring functions replaced by their authority-free
    counterparts (docs/authority_ablation_spec.md). Achieved entirely by
    dependency injection into the unmodified AISOSAgent/deep_sleep/
    wake_verify/promotion_pipeline/find_contradictions code paths added
    for this milestone -- no duplicated logic, no forked implementation.
    """
    condition_name = "periodic_aisos_without_authority"

    def __init__(self, period: int = 15, **kw):
        from aisos.memory.contradiction import score_without_authority
        from aisos.memory.promotion import authority_score_without_authority
        kw.setdefault("contradiction_score_fn", score_without_authority)
        kw.setdefault("authority_score_fn", authority_score_without_authority)
        super().__init__(period=period, **kw)


class TrustTierAblatedPeriodicAISOSBackend(PeriodicAISOSBackend):
    """Condition C4 (AISOS_Complete_Next_Research_Engineering_Master_Prompt.md
    Section 10, docs/mechanism_hypotheses.md H1/H2/H3): builds on top of the
    authority ablation by ALSO removing trust_tier from both comparators,
    leaving pure recency. Same dependency-injection pattern, same
    unmodified AISOSAgent/deep_sleep/wake_verify/promotion_pipeline code
    paths -- only the two scoring functions differ from
    AuthorityAblatedPeriodicAISOSBackend.
    """
    condition_name = "periodic_aisos_without_trust_tier"

    def __init__(self, period: int = 15, **kw):
        from aisos.memory.contradiction import score_recency_only
        from aisos.memory.promotion import authority_score_recency_only
        kw.setdefault("contradiction_score_fn", score_recency_only)
        kw.setdefault("authority_score_fn", authority_score_recency_only)
        # NOTE: calls PeriodicAISOSBackend.__init__ (not _AISOSBackendBase
        # directly) -- its own kw.setdefault(...) calls for the authority-
        # ablated functions become no-ops since these keys are already set
        # above, so the recency-only functions set here are what actually
        # reach _AISOSBackendBase/AISOSAgent.
        super().__init__(period=period, **kw)


class TrustOnlyPeriodicAISOSBackend(PeriodicAISOSBackend):
    """Security-Retrieval-Freshness causal ablation
    (AISOS_v3_Claude_Security_Retrieval_Freshness_Novelty_Prompt.md
    Section 12, condition B): trust_tier alone decides every
    contradiction/promotion comparison -- neither authority nor recency
    ever break a tie. Same DI pattern as every other ablation backend."""
    condition_name = "periodic_aisos_trust_only"

    def __init__(self, period: int = 15, **kw):
        from aisos.memory.contradiction import score_trust_only
        from aisos.memory.promotion import authority_score_trust_only
        kw.setdefault("contradiction_score_fn", score_trust_only)
        kw.setdefault("authority_score_fn", authority_score_trust_only)
        super().__init__(period=period, **kw)


class AuthorityOnlyPeriodicAISOSBackend(PeriodicAISOSBackend):
    """Security-Retrieval-Freshness causal ablation, condition C:
    authority alone decides every comparison."""
    condition_name = "periodic_aisos_authority_only"

    def __init__(self, period: int = 15, **kw):
        from aisos.memory.contradiction import score_authority_only
        from aisos.memory.promotion import authority_score_authority_only
        kw.setdefault("contradiction_score_fn", score_authority_only)
        kw.setdefault("authority_score_fn", authority_score_authority_only)
        super().__init__(period=period, **kw)


class AuthorityTrustNoRecencyPeriodicAISOSBackend(PeriodicAISOSBackend):
    """Security-Retrieval-Freshness causal ablation, condition D:
    authority + trust_tier, recency term removed entirely."""
    condition_name = "periodic_aisos_authority_trust_no_recency"

    def __init__(self, period: int = 15, **kw):
        from aisos.memory.contradiction import score_authority_and_trust_no_recency
        from aisos.memory.promotion import authority_score_authority_and_trust
        kw.setdefault("contradiction_score_fn", score_authority_and_trust_no_recency)
        kw.setdefault("authority_score_fn", authority_score_authority_and_trust)
        super().__init__(period=period, **kw)


class TrustedAuthorityPeriodicAISOSBackend(PeriodicAISOSBackend):
    """AISOS_Trusted_Authority_Provenance_Next_Milestone.md: byte-for-byte
    identical to PeriodicAISOSBackend -- same scheduler, lifecycle, sleep/
    consolidation/promotion pipeline, same numeric authority meanings --
    with the two authority-reading comparators replaced by their
    verified-authority counterparts, AND a TrustedAuthorityRegistry wired
    into the agent so remember() actually verifies claims instead of
    trusting the caller's source string. Achieved entirely by dependency
    injection into the unmodified AISOSAgent/deep_sleep/wake_verify/
    promotion_pipeline/find_contradictions code paths -- no forked logic.
    """
    condition_name = "periodic_aisos_trusted_authority"

    def __init__(self, period: int = 15, signing_key: Optional[bytes] = None,
                 registry: Optional["TrustedAuthorityRegistry"] = None, **kw):
        from aisos.memory.contradiction import verified_authority_contradiction_score
        from aisos.memory.promotion import verified_authority_score
        from aisos.security.authority_registry import TrustedAuthorityRegistry
        from aisos.security.checkpoint import new_signing_key
        # registry: optional (AISOS_Next_Milestone_Replay_Provenance.md) --
        # lets a caller share ONE registry instance across multiple
        # backends/agents (e.g. to test cross-instance replay against a
        # SHARED used-nonce set) instead of always constructing a fresh
        # one. Defaults to constructing a new registry, unchanged from
        # the prior milestone's behavior.
        self.registry = registry if registry is not None else TrustedAuthorityRegistry(signing_key or new_signing_key())
        kw.setdefault("contradiction_score_fn", verified_authority_contradiction_score)
        kw.setdefault("authority_score_fn", verified_authority_score)
        kw.setdefault("authority_registry", self.registry)
        super().__init__(period=period, **kw)

    def issue_credential(self, source: str) -> str:
        """Convenience passthrough so callers (tests, experiments) don't
        need to import TrustedAuthorityRegistry separately."""
        return self.registry.issue_credential(source)


class FullAISOSBackend(_AISOSBackendBase):
    """Condition D: AISOS's full pressure-driven pipeline."""
    condition_name = "aisos_full"

    def __init__(self, poll_every: int = 10, **kw):
        super().__init__(**kw)
        self.poll_every = poll_every
        self._scheduler_trace: list[dict] = []
        self._register_signals()

    def _register_signals(self):
        """Registers all signal sources Algorithm A1's default weight table
        defines room for (aisos/core/scheduler.py::SleepPressureWeights),
        matching the pattern already validated in aisos/sim/harness.py::
        run_aisos_workload() -- used by E1. An earlier version of this
        method registered only memory_utilization/maintenance_backlog/
        verification_backlog and omitted uptime_since_last_sleep, which is
        specifically the signal that guarantees pressure keeps rising over
        time regardless of write volume (the mechanism Theorem 4 depends
        on). On the 1,000-interaction pilot's low write-volume workload,
        that omission meant pressure plateaued at 0.2775 (just under
        tau_light=0.3) and never triggered a single sleep cycle across all
        1,000 interactions -- a harness miscalibration, not a defect in
        AISOS's scheduler or algorithms, which computed the (correctly
        under-specified) weighted average exactly as coded. See
        docs/research_log.md's 2026-08-31 (night) entry for the full
        diagnosis and the corrected re-run this fix produced.
        """
        agent = self.agent

        def memory_utilization():
            return min(1.0, len(agent.store.all_staging()) / 200)

        def maintenance_backlog():
            return min(1.0, len(agent.context_buffer) / 100)

        def verification_backlog():
            return min(1.0, len(agent.wal) / 50)

        def retrieval_degradation():
            return min(1.0, len(agent.store.read_live()) / 200)

        def uptime_since_last_sleep():
            elapsed = self._current_interaction_id - self._last_cycle_interaction_id
            return min(1.0, elapsed / 100)

        agent.register_signal_source("memory_utilization", memory_utilization)
        agent.register_signal_source("maintenance_backlog", maintenance_backlog)
        agent.register_signal_source("verification_backlog", verification_backlog)
        agent.register_signal_source("retrieval_degradation", retrieval_degradation)
        agent.register_signal_source("uptime_since_last_sleep", uptime_since_last_sleep)

    def maybe_maintain(self, interaction_id):
        self._current_interaction_id = interaction_id
        if interaction_id % self.poll_every != 0:
            return
        decision = self.agent.scheduler.decide()
        self._scheduler_trace.append({
            "interaction_id": interaction_id, "decision": decision,
            "pressure": self.agent.scheduler.last_pressure,
        })
        if decision != "stay_awake":
            self._run_cycle_and_record(interaction_id, target=decision, run_dream=True)

    def scheduler_trace(self):
        return list(self._scheduler_trace)


def make_backend(condition: str) -> ConditionBackend:
    return {
        "no_sleep_naive": NoSleepBackend,
        "continuous_mvcc": ContinuousMVCCBackend,
        "periodic_aisos": PeriodicAISOSBackend,
        "periodic_aisos_without_authority": AuthorityAblatedPeriodicAISOSBackend,
        "periodic_aisos_without_trust_tier": TrustTierAblatedPeriodicAISOSBackend,
        "periodic_aisos_trusted_authority": TrustedAuthorityPeriodicAISOSBackend,
        "aisos_full": FullAISOSBackend,
    }[condition]()
