"""
Algorithm A1 -- Sleep Pressure Estimation and Scheduling.

Preserves: I1 (Single Writer); realizes the scheduler side of Theorem 4
(No Starvation).

sleep_pressure(t) = sum_i w_i * signal_i(t)   (Technical Spec Section 8)

Signals default to the composite listed in the spec: memory_utilization,
duplicate_ratio, contradiction_ratio, retrieval_degradation,
verification_backlog, maintenance_backlog, uptime_since_last_sleep.

Cost: O(k) per poll for k registered signal sources, run at fixed interval
delta_poll -- this is the mechanism Theorem 4 is proved about: as long as
P0/P1 events remain transient, the loop's decision eventually admits a
sleep cycle once s(t) crosses tau_light, regardless of P2 traffic (P2 is
only ever enqueued by the Interrupt Controller, never blocking here).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


SignalSource = Callable[[], float]


@dataclass
class SleepPressureWeights:
    memory_utilization: float = 0.20
    duplicate_ratio: float = 0.15
    contradiction_ratio: float = 0.20
    retrieval_degradation: float = 0.15
    verification_backlog: float = 0.10
    maintenance_backlog: float = 0.10
    uptime_since_last_sleep: float = 0.10

    def as_dict(self) -> dict:
        return {
            "memory_utilization": self.memory_utilization,
            "duplicate_ratio": self.duplicate_ratio,
            "contradiction_ratio": self.contradiction_ratio,
            "retrieval_degradation": self.retrieval_degradation,
            "verification_backlog": self.verification_backlog,
            "maintenance_backlog": self.maintenance_backlog,
            "uptime_since_last_sleep": self.uptime_since_last_sleep,
        }


class Scheduler:
    """Algorithm A1. Signal sources are registered by name; a missing or
    failing source degrades to the remaining signals (renormalized) rather
    than blocking scheduling -- Section 4.2's stated failure mode, and part
    of what keeps Theorem 4 (No Starvation) true even under partial signal
    failure.
    """

    def __init__(
        self,
        weights: Optional[SleepPressureWeights] = None,
        tau_light: float = 0.35,
        tau_deep: float = 0.65,
        sec_alert_critical: Optional[Callable[[], bool]] = None,
    ):
        self.weights = weights or SleepPressureWeights()
        self.tau_light = tau_light
        self.tau_deep = tau_deep
        self._sources: dict[str, SignalSource] = {}
        self._last_pressure: float = 0.0
        self._poll_count: int = 0
        # A1 line 2 (v3 Section 6.1): "if sec_alert.critical(): return
        # EMERGENCY_SLEEP" -- checked before the weighted signal formula,
        # independent of tau_light/tau_deep. This reference implementation
        # has no distinct EMERGENCY FSM state (see docs/spec_reconciliation.md
        # Section 5), so the emergency response is "enter_deep": the
        # strongest available protected state, which forces a full
        # Wake-Verify re-validation of the trusted checkpoint.
        self._sec_alert_critical = sec_alert_critical or (lambda: False)
        self.emergency_decisions: int = 0

    def register_signal_source(self, name: str, fn: SignalSource) -> None:
        if name not in self.weights.as_dict():
            raise ValueError(
                f"unknown signal {name!r}; register via a custom weights object "
                f"or use one of {list(self.weights.as_dict())}"
            )
        self._sources[name] = fn

    def unregister_signal_source(self, name: str) -> None:
        self._sources.pop(name, None)

    def poll(self) -> float:
        """O(k) for k registered sources. Missing sources are simply
        excluded and the remaining weights renormalized -- a single broken
        metric cannot itself cause starvation (Section 4.2).
        """
        self._poll_count += 1
        weights = self.weights.as_dict()
        total_w = 0.0
        acc = 0.0
        for name, fn in self._sources.items():
            w = weights.get(name, 0.0)
            try:
                val = float(fn())
            except Exception:
                # A failing source is treated exactly like a missing one.
                continue
            val = max(0.0, min(1.0, val))
            acc += w * val
            total_w += w
        pressure = (acc / total_w) if total_w > 0 else 0.0
        self._last_pressure = pressure
        return pressure

    def decide(self) -> str:
        """Returns one of 'stay_awake' | 'enter_light' | 'enter_deep'.

        A P0 security-critical event forces 'enter_deep' unconditionally,
        bypassing the weighted pressure formula entirely -- this is what
        closes the gap where a security alert arriving while AWAKE
        previously had no effect until ordinary pressure happened to cross
        threshold on its own (see docs/spec_reconciliation.md Section 5).
        """
        if self._sec_alert_critical():
            self.emergency_decisions += 1
            self._last_pressure = 1.0
            return "enter_deep"
        s = self.poll()
        if s >= self.tau_deep:
            return "enter_deep"
        if s >= self.tau_light:
            return "enter_light"
        return "stay_awake"

    @property
    def last_pressure(self) -> float:
        return self._last_pressure

    @property
    def poll_count(self) -> int:
        return self._poll_count
