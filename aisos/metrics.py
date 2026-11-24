"""
Metric definitions used across the experiments (v2 Section 7 / Appendix E).
Each metric here is defined once and applied identically across conditions,
per Appendix E's requirement that a metric "defined differently for AISOS
than for its baselines would invalidate the comparison."
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional

from aisos.memory.contradiction import SOURCE_AUTHORITY


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (k - f) * (s[c] - s[f])


def mean_ci95(values: list[float]) -> tuple[float, float]:
    """Mean and a normal-approximation 95% CI half-width across seeds/replays
    (v2 Section 7.6: 'mean over at least 3 seeds ... with 95% confidence
    intervals'). Falls back to (mean, 0.0) for a single sample.
    """
    if not values:
        return float("nan"), 0.0
    if len(values) == 1:
        return values[0], 0.0
    m = statistics.mean(values)
    sd = statistics.stdev(values)
    half = 1.96 * sd / math.sqrt(len(values))
    return m, half


def oracle_value(subject: str, events: list[dict], up_to_seq: Optional[int] = None) -> Optional[str]:
    """The 'ground truth' legitimate value for a subject: the highest
    (source-authority, recency) ranked *legitimate* (non-dream, non-
    duplicate-noise) write up to a point in the stream. Used as the
    reference point for the self-contradiction-rate and staleness metrics,
    since we control the full generative process and can therefore compute
    an actual ground truth rather than asserting one.
    """
    best = None
    best_score = None
    for e in events:
        if up_to_seq is not None and e["seq"] > up_to_seq:
            break
        if e["subject"] != subject:
            continue
        authority = SOURCE_AUTHORITY.get(e["source"], 0.4)
        score = (authority, e["ts"])
        if best_score is None or score >= best_score:
            best_score = score
            best = e
    return best["value"] if best else None


@dataclass
class RunMetrics:
    condition: str = "unknown"
    seed: int = 0
    n_events: int = 0
    contradiction_rate: Optional[float] = None
    coverage: Optional[float] = None
    fake_memory_rate: Optional[float] = None
    staleness_s: Optional[float] = None
    read_latency_p50_ms: Optional[float] = None
    read_latency_p95_ms: Optional[float] = None
    write_latency_p50_ms: Optional[float] = None
    write_latency_p95_ms: Optional[float] = None
    torn_reads: int = 0
    reads_total: int = 0
    rollback_count: int = 0
    rollback_records_discarded: Optional[int] = None
    cycles_run: int = 0
    wall_time_s: float = 0.0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "condition": self.condition, "seed": self.seed, "n_events": self.n_events,
            "contradiction_rate": self.contradiction_rate, "coverage": self.coverage,
            "fake_memory_rate": self.fake_memory_rate,
            "staleness_s": self.staleness_s,
            "read_latency_p50_ms": self.read_latency_p50_ms, "read_latency_p95_ms": self.read_latency_p95_ms,
            "write_latency_p50_ms": self.write_latency_p50_ms, "write_latency_p95_ms": self.write_latency_p95_ms,
            "torn_reads": self.torn_reads, "reads_total": self.reads_total,
            "rollback_count": self.rollback_count, "rollback_records_discarded": self.rollback_records_discarded,
            "cycles_run": self.cycles_run, "wall_time_s": self.wall_time_s, **self.extra,
        }
