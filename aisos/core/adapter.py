"""
Agent Adapter API (Technical Spec Sections 22-24) and AISOSAgent, the
orchestrator that wires the lifecycle FSM, scheduler, interrupt controller,
memory store, WAL, and the four sleep-stage algorithms together into one
runnable object.

SDK API philosophy (Technical Spec Section 23):

    agent = AISOSAgent(model=model, memory=memory_backend, policy=policy)
    agent.remember(content="...", source="user", trust="observed")
    agent.run()
    agent.request_maintenance()
    agent.status()

"Exact APIs are not fixed. The SDK should be provider-agnostic." This
reference implementation keeps that spirit: `model`/`memory backend` are
not required (the store here is the reference memory backend), and every
extension point (scheduler weights, dream score function, external-source
checker, contradiction resolution) is a constructor argument or callable.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from aisos.core.lifecycle import Lifecycle, State, InvalidTransition
from aisos.core.scheduler import Scheduler, SleepPressureWeights
from aisos.core.interrupt import InterruptController, InterruptEvent, Priority

from aisos.memory.record import MemoryRecord, TrustTier, Status
from aisos.memory.store import MemoryStore
from aisos.memory.light_sleep import light_sleep, LightSleepResult
from aisos.memory.deep_sleep import deep_sleep, DeepSleepResult
from aisos.memory.contradiction import _score as _default_contradiction_score
from aisos.memory.promotion import _authority_score as _default_authority_score

from aisos.dream.dream_phase import dream_phase, DreamBudget, DreamResult

from aisos.security.wal import WriteAheadLog
from aisos.security.checkpoint import new_signing_key
from aisos.security.wake_verify import wake_verify, WakeVerifyResult


@dataclass
class CycleReport:
    target: str
    aborted: bool
    stage_reached: str
    duration_s: float
    light: Optional[LightSleepResult] = None
    deep: Optional[DeepSleepResult] = None
    dream: Optional[DreamResult] = None
    verify: Optional[WakeVerifyResult] = None


@dataclass
class CycleReportSummary:
    """`AISOSAgent.cycle_reports` retention (AISOS_v3_Single_Master_Research_Completion_Prompt.md,
    `cycle_reports` forensic audit): a forensic trace of every read site
    in this codebase (grep across aisos/, zero hits for
    `cycle_reports[...]` indexing) found that NOTHING ever reads a
    historical CycleReport's nested light/deep/dream/verify sub-results
    -- every consumer uses only `len(cycle_reports)` (a count) or
    `sum(r.duration_s for r in cycle_reports)` (aisos/experiments/
    e4_energy_latency.py, the one exception, which needs only the
    scalar `duration_s` field). The nested sub-results hold `list[MemoryRecord]`
    fields (LightSleepResult.new_candidates, DeepSleepResult.consolidated,
    DreamResult.synthetic_candidates, and -- via WakeVerifyResult.promotion
    -- PromotionResult's 5 separate MemoryRecord lists) that were
    retained, unbounded, for the agent's entire lifetime, with no
    production or diagnostic need ever exercising that retention.
    CycleReportSummary preserves every field an existing consumer reads
    (target, aborted, stage_reached, duration_s) plus lightweight COUNTS
    in place of the full lists -- enough for auditability/debugging
    ("how many candidates did light sleep produce this cycle") without
    retaining the actual MemoryRecord object graph, which remains fully
    available via `_published`/`_staging`/WAL at commit time regardless.
    `run_sleep_cycle()`'s own RETURN VALUE is UNCHANGED (still the full,
    rich `CycleReport`) -- only the long-term retention in
    `self.cycle_reports` uses this lighter summary.
    """
    target: str
    aborted: bool
    stage_reached: str
    duration_s: float
    light_new_candidates: int = 0
    deep_consolidated: int = 0
    deep_archived: int = 0
    deep_deduped: int = 0
    dream_synthetic_candidates: int = 0
    verify_passed: Optional[bool] = None
    verify_reason: str = ""
    promoted: int = 0


def _summarize_cycle_report(report: "CycleReport") -> CycleReportSummary:
    return CycleReportSummary(
        target=report.target, aborted=report.aborted, stage_reached=report.stage_reached,
        duration_s=report.duration_s,
        light_new_candidates=len(report.light.new_candidates) if report.light else 0,
        deep_consolidated=len(report.deep.consolidated) if report.deep else 0,
        deep_archived=report.deep.archived if report.deep else 0,
        deep_deduped=report.deep.deduped if report.deep else 0,
        dream_synthetic_candidates=len(report.dream.synthetic_candidates) if report.dream else 0,
        verify_passed=report.verify.passed if report.verify else None,
        verify_reason=report.verify.reason if report.verify else "",
        promoted=report.verify.promoted if report.verify else 0,
    )


class AgentAdapterAPI:
    """The minimal surface an agent framework touches (Technical Spec
    Section 22, `core/` : lifecycle, scheduler, events, policies)."""

    def register_signal_source(self, name: str, fn: Callable[[], float]) -> None:
        raise NotImplementedError

    def remember(self, subject: str, value: str, source: str = "user", confidence: float = 0.7) -> MemoryRecord:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def request_maintenance(self) -> Optional[CycleReport]:
        raise NotImplementedError

    def status(self) -> dict:
        raise NotImplementedError


class AISOSAgent(AgentAdapterAPI):
    def __init__(
        self,
        weights: Optional[SleepPressureWeights] = None,
        tau_light: float = 0.35,
        tau_deep: float = 0.65,
        atomic_unit_duration_ms: float = 50.0,
        max_defer_ms: float = 100.0,
        dream_budget: Optional[DreamBudget] = None,
        theta_dims: int = 8,
        signing_key: Optional[bytes] = None,
        segment_size: int = 8,
        max_versions_per_subject: int = 3,
        check_external_sources: Optional[Callable[[MemoryRecord], bool]] = None,
        max_allowed_drift: float = 0.6,
        drift_baseline_min: int = 20,
        max_promotions_per_cycle: Optional[int] = 25,
        confidence_threshold: float = 0.55,
        max_promotion_rounds: int = 4,
        consistency_tolerance: int = 0,
        fleet_coordinator=None,
        replica_id: int = 0,
        on_transition: Optional[Callable[[object], None]] = None,
        contradiction_score_fn: Callable[[MemoryRecord], float] = _default_contradiction_score,
        authority_score_fn: Callable[[MemoryRecord], tuple] = _default_authority_score,
        authority_registry: Optional["TrustedAuthorityRegistry"] = None,
        verify_provenance_integrity: bool = False,
        verify_continuous_integrity: bool = False,
    ):
        # on_transition: observability hook (1000-interaction experiment
        # prompt, Section 8: "every state transition must be observable").
        # Forwarded to Lifecycle unmodified -- it does not change any
        # transition-table semantics, only whether a caller is notified.
        self.lifecycle = Lifecycle(on_transition=on_transition)
        # contradiction_score_fn / authority_score_fn: passed through to
        # deep_sleep() and wake_verify() respectively on every cycle.
        # Default to the real authority-aware scorers -- overridden only by
        # aisos/agent/condition_backends.py's authority-ablated backend
        # (docs/authority_ablation_spec.md). Every other caller of
        # AISOSAgent is unaffected by these two new parameters.
        self.contradiction_score_fn = contradiction_score_fn
        self.authority_score_fn = authority_score_fn
        # authority_registry: optional (AISOS_Trusted_Authority_Provenance_
        # Next_Milestone.md). None (the default) means remember() never
        # consults it, so verified_authority is never set -- fully
        # backward compatible with every pre-existing caller/test. Set
        # only by aisos/agent/condition_backends.py's trusted-authority
        # backend.
        self.authority_registry = authority_registry
        # SECURITY FINDING (AISOS_Comprehensive_Security_Audit_Next_
        # Milestone.md Phase V, "configuration security"), fixed here:
        # configuring authority_registry alone does NOT protect anything
        # unless authority_score_fn/contradiction_score_fn are ALSO
        # switched to their verified_authority-reading counterparts --
        # otherwise verified_authority is correctly computed and stored,
        # but the comparator never reads it, and ranking silently falls
        # back to trusting the raw, unverified `source` claim (the exact
        # vulnerability the trusted-authority mechanism exists to close).
        # Demonstrated directly: AISOSAgent(authority_registry=registry)
        # with no other change still let a credential-less
        # source="user_correction" claim win via the legacy comparator.
        # Fail closed rather than silently do nothing: refuse to
        # construct an agent in this specific, almost-certainly-a-mistake
        # configuration, rather than warn (which could be missed/
        # suppressed) or silently proceed unprotected.
        if authority_registry is not None and (
            authority_score_fn is _default_authority_score or contradiction_score_fn is _default_contradiction_score
        ):
            raise ValueError(
                "authority_registry is configured but authority_score_fn/contradiction_score_fn "
                "are still the legacy defaults, which never read verified_authority -- this "
                "silently provides NO protection against authority spoofing. Pass "
                "authority_score_fn=verified_authority_score and "
                "contradiction_score_fn=verified_authority_contradiction_score (from "
                "aisos.memory.promotion / aisos.memory.contradiction) alongside authority_registry, "
                "or omit authority_registry if you intend to use the legacy, unverified path."
            )
        # AISOS_Next_Milestone_Replay_Provenance.md Phase J, opt-in
        # (default False = fully backward compatible).
        self.verify_provenance_integrity = verify_provenance_integrity
        # AISOS_Security_Closure_Milestone.md Phase K/L, opt-in (default
        # False = fully backward compatible).
        self.verify_continuous_integrity = verify_continuous_integrity
        self._emergency_pending = False
        self.scheduler = Scheduler(
            weights, tau_light, tau_deep,
            sec_alert_critical=self._consume_emergency_pending,
        )
        self.store = MemoryStore()
        self.wal = WriteAheadLog()
        self.signing_key = signing_key or new_signing_key()
        self.interrupt = InterruptController(
            atomic_unit_duration_ms, max_defer_ms, preempt_now=self._on_preempt_now
        )
        self.theta: list[float] = [0.0] * theta_dims
        self.context_buffer: list[dict] = []
        self.dream_budget = dream_budget or DreamBudget()
        self.segment_size = segment_size
        self.max_versions_per_subject = max_versions_per_subject
        self.check_external_sources = check_external_sources
        self.max_allowed_drift = max_allowed_drift
        self.drift_baseline_min = drift_baseline_min
        self.max_promotions_per_cycle = max_promotions_per_cycle
        self.confidence_threshold = confidence_threshold
        self.max_promotion_rounds = max_promotion_rounds
        self.consistency_tolerance = consistency_tolerance

        self.fleet_coordinator = fleet_coordinator
        self.replica_id = replica_id

        self._preempt_flag = False
        self._pending_preempt_event: Optional[InterruptEvent] = None
        self._lock = threading.RLock()

        self.rollback_count = 0
        self.verify_fail_count = 0
        self.cycle_reports: list[CycleReportSummary] = []
        self._weakness_scores: dict[str, float] = {}

    @classmethod
    def from_config(
        cls,
        default_path: str = "default.yaml",
        models_path: str = "models.yaml",
        **overrides,
    ) -> "AISOSAgent":
        """Construct an AISOSAgent from configs/default.yaml + models.yaml
        (Master Prompt Section 4/7: thresholds/weights must be configurable,
        not hard-coded). `overrides` takes precedence over the file values,
        e.g. AISOSAgent.from_config(tau_light=0.2) for a one-off sweep point.
        """
        from aisos.config import agent_kwargs_from_config
        kwargs = agent_kwargs_from_config(default_path, models_path, overrides)
        return cls(**kwargs)

    # ---- Agent Adapter API -------------------------------------------------
    def register_signal_source(self, name: str, fn: Callable[[], float]) -> None:
        self.scheduler.register_signal_source(name, fn)

    def remember(self, subject: str, value: str, source: str = "user", confidence: float = 0.7,
                 ts: Optional[float] = None, credential: Optional[str] = None) -> MemoryRecord:
        """Live-time write path (AWAKE). Lands in the (published copy's)
        staging area, never directly in `published`/M_V -- I6 applies to
        observed content too, so even a `source="user_correction"` write
        must clear the Promotion Pipeline on the next Wake-Up before it is
        live-readable.

        credential: optional (AISOS_Trusted_Authority_Provenance_Next_
        Milestone.md). Ignored entirely unless self.authority_registry is
        set -- when it is, verified_authority is derived from
        registry.verify(source, credential), NEVER from the caller's own
        `source` claim, closing the self-declared-authority vulnerability
        (docs/authority_security_stress_test.md Test H). When no registry
        is configured (the default), this parameter has no effect and
        verified_authority stays None, preserving every pre-existing
        caller's behavior exactly.
        """
        ts = ts if ts is not None else time.time()
        verified_authority = None
        if self.authority_registry is not None:
            verified_authority = self.authority_registry.verify(source, credential)
        # Publish-Race / Staging-Consistency Closure Milestone
        # (AISOS_Publish_Race_Staging_Consistency_Closure_Prompt.md):
        # context_buffer.append() below is now under self._lock -- see
        # run_sleep_cycle()'s LIGHT stage for the matching, snapshot-based
        # consumer-side fix and the full mechanism writeup.
        # verified_authority MUST travel with the context_buffer entry too:
        # light_sleep.py reconstructs a brand-new MemoryRecord from these
        # raw dicts (a pre-existing, unrelated code path -- see that
        # module's docstring), not from the MemoryRecord object created
        # below. Without this, light_sleep's reconstructed duplicate would
        # carry verified_authority=None, and deep_sleep's dedup step would
        # then discard the ORIGINAL verified record in favor of that
        # unverified duplicate (whichever has the more recent updated_at
        # wins dedup's tie-break) -- silently destroying the verified
        # authority before promotion ever sees it. Found and fixed during
        # the trusted-authority milestone's own lifecycle-integrity check
        # (Phase K); see aisos/tests/test_trusted_authority.py's
        # provenance-survives-sleep test.
        entry = {"subject": subject, "value": value, "source": source, "confidence": confidence, "ts": ts,
                 "verified_authority": verified_authority}
        with self._lock:
            self.context_buffer.append(entry)
        rec = MemoryRecord(
            subject=subject, value=value, source=source, confidence=confidence,
            trust_tier=TrustTier.T1_CANDIDATE, status=Status.CANDIDATE,
            provenance={"origin": "observed"},
            verified_authority=verified_authority,
        )
        self.store.ingest_raw(rec)
        return rec

    def run(self) -> None:
        """DEPRECATED and a no-op. Do not call.

        This method has always returned None without doing anything: an
        agent framework drives its own request loop and calls
        remember()/request_maintenance() as needed. It exists only for
        API-surface completeness with Technical Spec Section 23.

        It is deprecated rather than removed because removing it would
        break the documented Section 23 surface. It emits a
        DeprecationWarning (silent by default in Python, visible under
        `-W default` or pytest) so that a caller who believes it runs an
        agent loop finds out. Nothing in this repository calls it.

        What to use instead:
            aisos.runtime.AgentRuntime -- wake() / remember() / sleep() /
            shutdown(), with durable storage and events. sleep() is a thin
            wrapper over request_maintenance() and does not change any
            research semantics.

        The BEHAVIOR of this method is unchanged (still a no-op returning
        None); only the warning is new. SDK Phase F.
        """
        import warnings

        warnings.warn(
            "AISOSAgent.run() is a deprecated no-op and does nothing. Use "
            "aisos.runtime.AgentRuntime (wake/remember/sleep/shutdown), or call "
            "remember()/request_maintenance() directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return None

    def status(self) -> dict:
        return {
            "state": self.lifecycle.state.value,
            "sleep_pressure": self.scheduler.last_pressure,
            "store_version": self.store.version,
            "wal_len": len(self.wal),
            "p2_backlog": self.interrupt.p2_backlog,
            "rollback_count": self.rollback_count,
            "verify_fail_count": self.verify_fail_count,
            "live_memory_count": len(self.store.read_live()),
            "theta_norm": sum(t * t for t in self.theta) ** 0.5,
        }

    # ---- interrupts ---------------------------------------------------------
    def dispatch_interrupt(self, priority: Priority, payload=None) -> InterruptEvent:
        event = InterruptEvent(priority=priority, payload=payload)
        return self.interrupt.dispatch(event)

    def _on_preempt_now(self, event: InterruptEvent) -> None:
        with self._lock:
            self._preempt_flag = True
            self._pending_preempt_event = event
            if event.priority == Priority.P0:
                # Closes the EMERGENCY gap (docs/spec_reconciliation.md
                # Section 5): a P0 event dispatched while AWAKE previously
                # set _preempt_flag with nothing to consume it (no cycle is
                # running yet), so it was silently cleared, with no effect,
                # by the next run_sleep_cycle() call. This flag is instead
                # consumed by the Scheduler's sec_alert_critical() check
                # (core/scheduler.py), forcing the next decide() to return
                # 'enter_deep' regardless of ordinary sleep pressure.
                self._emergency_pending = True

    def _consume_emergency_pending(self) -> bool:
        with self._lock:
            pending = self._emergency_pending
            self._emergency_pending = False
            return pending

    # ---- maintenance ---------------------------------------------------------
    def request_maintenance(self, force: Optional[str] = None, run_dream: bool = True) -> Optional[CycleReport]:
        """force in {"enter_light", "enter_deep", "enter_dream", None}."""
        target = force or self.scheduler.decide()
        if target == "stay_awake":
            return None
        if target == "enter_deep":
            target = "enter_deep"
        return self.run_sleep_cycle(target=target, run_dream=run_dream)

    def _build_weakness_graph(self) -> list[dict]:
        # Weakness graph G: subjects with low recent confidence or a history
        # of contradiction get rehearsed preferentially in Dream.
        nodes = []
        for r in self.store.read_live():
            weakness = 1.0 - r.confidence
            nodes.append({"subject": r.subject, "value": r.value, "proposed_value": r.value, "weakness": weakness})
        if not nodes:
            nodes = [{"subject": "bootstrap", "value": "unknown", "proposed_value": "unknown", "weakness": 1.0}]
        return nodes

    def run_sleep_cycle(self, target: str = "enter_light", run_dream: bool = True) -> CycleReport:
        t_start = time.time()
        with self._lock:
            self._preempt_flag = False
            self._pending_preempt_event = None

        # Fleet Quorum (I5), if this agent is part of a coordinated fleet.
        if self.fleet_coordinator is not None:
            if not self.fleet_coordinator.try_acquire(self.replica_id):
                return CycleReport(
                    target=target, aborted=True, stage_reached="AWAKE (fleet token unavailable)",
                    duration_s=time.time() - t_start,
                )

        try:
            self.lifecycle.fire(target)
            self.interrupt.set_in_mutating_stage(True)
            self.store.begin_private_copy()
            private = self.store.private()
            self.wal.reset()

            light_res = deep_res = dream_res = None
            stage_reached = self.lifecycle.state.value

            if self.lifecycle.state == State.LIGHT:
                # Publish-Race / Staging-Consistency Closure Milestone: the
                # ORIGINAL code passed self.context_buffer directly to
                # light_sleep() and then unconditionally cleared it
                # (`= []`) or trimmed it by an OFFSET computed against its
                # length AT THE TIME light_sleep() returned. Neither
                # operation was lock-protected, and -- independent of any
                # low-level thread-safety concern -- the clear/trim step
                # itself assumed context_buffer had not grown since
                # light_sleep() started running. Under real concurrent
                # writers, ANY remember() call landing between the moment
                # light_sleep() took its own internal segment snapshot and
                # this clear/trim line executing was silently discarded --
                # not merely delayed, PERMANENTLY LOST (confirmed
                # empirically: up to 31% loss under sustained concurrent
                # load in docs/publish_race_reproduction.md's thread-stress
                # trials). The vulnerable window is proportional to
                # light_sleep()'s OWN execution time (more segments/records
                # = a wider window), not a rare single-instruction fluke.
                #
                # Fix: take an explicit, lock-protected SNAPSHOT of
                # context_buffer before calling light_sleep() (light_sleep
                # already internally re-slices its input into segments via
                # _segment(), so handing it an immutable snapshot changes
                # nothing about its own behavior), then afterward remove
                # ONLY the snapshot's own length from the front of the
                # (possibly since-grown) live context_buffer -- since
                # writers only ever append, the snapshot is guaranteed to
                # be exactly the current front of context_buffer, so this
                # correctly preserves anything appended during light_sleep's
                # execution instead of discarding it.
                with self._lock:
                    context_snapshot = list(self.context_buffer)
                light_res = light_sleep(
                    private["staging"], context_snapshot, self.wal,
                    should_preempt=lambda: self._preempt_flag, segment_size=self.segment_size,
                )
                n_consumed = (light_res.segments_processed * self.segment_size) if light_res.preempted else len(context_snapshot)
                with self._lock:
                    self.context_buffer = self.context_buffer[n_consumed:]
                if self._preempt_flag:
                    return self._abort(t_start, target, "LIGHT", light_res, None, None)
                self.lifecycle.fire("finish_stage")

            if self.lifecycle.state == State.DEEP:
                deep_res = deep_sleep(
                    private["staging"], self.wal,
                    should_preempt=lambda: self._preempt_flag,
                    max_versions_per_subject=self.max_versions_per_subject,
                    contradiction_score_fn=self.contradiction_score_fn,
                )
                if self._preempt_flag:
                    return self._abort(t_start, target, "DEEP", light_res, deep_res, None)
                if run_dream:
                    self.lifecycle.fire("finish_stage")
                else:
                    self.lifecycle.fire("request_wake_verify")

            if self.lifecycle.state == State.DREAM:
                if run_dream:
                    weakness_graph = self._build_weakness_graph()
                    dream_res = dream_phase(
                        private["staging"], weakness_graph, self.theta, self.dream_budget, self.wal,
                        should_preempt=lambda: self._preempt_flag,
                        seed=len(self.cycle_reports),
                    )
                    if self._preempt_flag:
                        return self._abort(t_start, target, "DREAM", light_res, deep_res, dream_res)
                # Fire request_wake_verify regardless of run_dream: DREAM is
                # reachable via target="enter_dream" with run_dream=False (no
                # dream processing wanted, just the state), and it must still
                # leave DREAM through the FSM's only exit rather than being
                # left stranded for the verify_pass/verify_fail fire() below,
                # which has no (DREAM, verify_pass/verify_fail) transition.
                self.lifecycle.fire("request_wake_verify")

            self.interrupt.set_in_mutating_stage(False)
            result = wake_verify(
                self.store, self.wal, self.signing_key,
                check_external_sources=self.check_external_sources,
                max_allowed_drift=self.max_allowed_drift,
                drift_baseline_min=self.drift_baseline_min,
                max_promotions_per_cycle=self.max_promotions_per_cycle,
                confidence_threshold=self.confidence_threshold,
                max_promotion_rounds=self.max_promotion_rounds,
                consistency_tolerance=self.consistency_tolerance,
                contradiction_score_fn=self.contradiction_score_fn,
                authority_score_fn=self.authority_score_fn,
                verify_provenance_integrity=self.verify_provenance_integrity,
                verify_continuous_integrity=self.verify_continuous_integrity,
            )
            if result.passed:
                self.lifecycle.fire("verify_pass")
                if dream_res is not None:
                    self.theta = dream_res.theta_after
            else:
                self.lifecycle.fire("verify_fail")
                self.verify_fail_count += 1
                self.rollback_count += 1

            report = CycleReport(
                target=target, aborted=False, stage_reached=stage_reached,
                duration_s=time.time() - t_start, light=light_res, deep=deep_res,
                dream=dream_res, verify=result,
            )
            self.cycle_reports.append(_summarize_cycle_report(report))
            return report
        except Exception:
            # Any exception past this point left the lifecycle in a
            # mutating state with no compensating transition -- without
            # this, the FSM's transition table (no (LIGHT|DEEP|DREAM, *)
            # entry targets AWAKE) permanently wedges every subsequent
            # run_sleep_cycle() call. Recover the same way PREEMPT_NOW does:
            # force a path to WAKE_VERIFY, discard the private copy, and
            # land back at AWAKE, then re-raise so the caller still sees
            # the original failure.
            self.interrupt.set_in_mutating_stage(False)
            if self.lifecycle.is_mutating():
                try:
                    self.lifecycle.fire("preempt_p0")
                except InvalidTransition:
                    pass
            if self.lifecycle.state == State.WAKE_VERIFY:
                wake_verify(self.store, self.wal, self.signing_key, aborted=True)
                try:
                    self.lifecycle.fire("verify_fail")
                except InvalidTransition:
                    pass
            self.rollback_count += 1
            raise
        finally:
            if self.fleet_coordinator is not None:
                self.fleet_coordinator.release(self.replica_id)

    def _abort(self, t_start, target, stage, light_res, deep_res, dream_res) -> CycleReport:
        """PREEMPT_NOW path (P0/P1): fire the FSM's preempt_* transition
        (mutating state -> WAKE_VERIFY, never directly to AWAKE -- Theorem
        1's structural precondition), then let wake_verify's aborted=True
        branch perform the O(1) discard (Theorem 2).
        """
        priority = "p0"
        if self._pending_preempt_event is not None:
            priority = "p0" if self._pending_preempt_event.priority == Priority.P0 else "p1"
        event_name = f"preempt_{priority}"
        try:
            self.lifecycle.fire(event_name)
        except InvalidTransition:
            pass
        self.interrupt.set_in_mutating_stage(False)
        result = wake_verify(self.store, self.wal, self.signing_key, aborted=True)
        try:
            self.lifecycle.fire("verify_fail")
        except InvalidTransition:
            pass
        self.rollback_count += 1
        report = CycleReport(
            target=target, aborted=True, stage_reached=stage,
            duration_s=time.time() - t_start, light=light_res, deep=deep_res,
            dream=dream_res, verify=result,
        )
        self.cycle_reports.append(_summarize_cycle_report(report))
        return report
