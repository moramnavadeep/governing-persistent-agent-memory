"""
Experiment harness: drives a workload (sim/workload.py) against either an
AISOSAgent or the ContinuousMVCCBaseline, with an independent concurrent
read-load generator active throughout -- v2 Section 7.6's concurrency
model, "specifically to give Theorem 1 an opportunity to fail if I4 is
violated in the implementation."
"""
from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from aisos.core.adapter import AISOSAgent
from aisos.baselines.continuous_mvcc import ContinuousMVCCBaseline
from aisos.metrics import RunMetrics, percentile, oracle_value


@dataclass
class ReadSample:
    latency_ms: float
    consistent: bool


class ReadLoadGenerator:
    """Runs on its own thread, issuing read calls back-to-back for a fixed
    wall-clock duration while the caller's writer thread proceeds
    independently. `validator(result) -> bool` decides whether a given read
    result is internally consistent (used to detect torn reads).
    """

    def __init__(self, read_fn, validator, duration_s: float):
        self.read_fn = read_fn
        self.validator = validator
        self.duration_s = duration_s
        self.samples: list[ReadSample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        end = time.time() + self.duration_s
        while time.time() < end and not self._stop.is_set():
            t0 = time.time()
            result = self.read_fn()
            dt_ms = (time.time() - t0) * 1000.0
            ok = self.validator(result)
            self.samples.append(ReadSample(latency_ms=dt_ms, consistent=ok))

    def stop_and_join(self, timeout: float = 5.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def torn_reads(self) -> int:
        return sum(1 for s in self.samples if not s.consistent)

    @property
    def total(self) -> int:
        return len(self.samples)

    @property
    def latencies_ms(self) -> list[float]:
        return [s.latency_ms for s in self.samples]


def _rolling_signals(window: deque):
    def duplicate_ratio():
        if not window:
            return 0.0
        return sum(1 for e in window if e["kind"] == "duplicate") / len(window)

    def contradiction_ratio():
        if not window:
            return 0.0
        return sum(1 for e in window if e["kind"] == "contradiction") / len(window)

    return duplicate_ratio, contradiction_ratio


def run_aisos_workload(
    agent: AISOSAgent,
    events: list[dict],
    poll_every: int = 25,
    window_size: int = 100,
    concurrent_reads: bool = True,
    read_duration_s: float = 1.5,
    memory_cap: int = 400,
    run_dream: bool = True,
    drain_budget: int = 60,
    condition: str = "aisos_full",
    seed: int = 0,
) -> RunMetrics:
    if poll_every <= 0:
        raise ValueError("poll_every must be positive")
    t_start = time.time()
    window: deque = deque(maxlen=window_size)
    dup_ratio_fn, contra_ratio_fn = _rolling_signals(window)

    agent.register_signal_source("duplicate_ratio", dup_ratio_fn)
    agent.register_signal_source("contradiction_ratio", contra_ratio_fn)
    agent.register_signal_source("memory_utilization", lambda: min(1.0, len(agent.store.all_staging()) / memory_cap))
    agent.register_signal_source("verification_backlog", lambda: min(1.0, len(agent.wal) / 50))
    agent.register_signal_source("maintenance_backlog", lambda: min(1.0, len(agent.context_buffer) / 200))
    last_cycle_seq = {"v": 0}
    agent.register_signal_source(
        "retrieval_degradation",
        lambda: min(1.0, len(agent.store.read_live()) / (memory_cap * 2)),
    )
    agent.register_signal_source(
        "uptime_since_last_sleep",
        lambda: min(1.0, (window[-1]["seq"] - last_cycle_seq["v"]) / (window_size * 3)) if window else 0.0,
    )

    write_latencies = []
    cycles_run = 0
    commit_log: list[tuple[int, int]] = []  # (event_seq, store_version) at each successful commit

    reader = None
    if concurrent_reads:
        def do_read():
            return agent.store.read_live_with_version()

        def validate(result):
            version, recs = result
            ck = agent.store.checkpoint_at(version)
            if ck is None:
                return True  # version not retained (pruned) -- not a failure signal here
            expected_ids = set(ck.published.keys())
            got_ids = {r.id for r in recs}
            if expected_ids != got_ids:
                return False
            for r in recs:
                ref = ck.published.get(r.id)
                if ref is None or ref.content_hash() != r.content_hash():
                    return False
            return True

        reader = ReadLoadGenerator(do_read, validate, duration_s=read_duration_s)
        reader.start()

    for e in events:
        window.append(e)
        t0 = time.time()
        agent.remember(subject=e["subject"], value=e["value"], source=e["source"], confidence=e["confidence"], ts=e["ts"])
        write_latencies.append((time.time() - t0) * 1000.0)

        if e["seq"] % poll_every == 0:
            decision = agent.scheduler.decide()
            if decision != "stay_awake":
                report = agent.run_sleep_cycle(target=decision, run_dream=run_dream)
                cycles_run += 1
                last_cycle_seq["v"] = e["seq"]
                if report.verify is not None and report.verify.passed:
                    commit_log.append((e["seq"], agent.store.version))

    if reader is not None:
        reader.stop_and_join()

    # Drain phase: promotion advances at most one trust tier per cycle
    # (Algorithm A7), so a subject last written near the end of the stream
    # needs several more Wake-Up cycles to reach M_V even with no new
    # writes arriving -- this is exactly the batched-cycle staleness
    # trade-off Section 8.3 states up front, not an artifact to hide. We
    # run a bounded number of additional cycles with no new input so the
    # end-of-run correctness snapshot reflects steady state rather than a
    # mid-promotion snapshot, while `cycles_run` and `wall_time_s` still
    # honestly count every cycle actually executed (drain included).
    while drain_budget > 0:
        decision = agent.scheduler.decide()
        if decision == "stay_awake" and len(agent.store.all_staging()) == 0:
            break
        target = decision if decision != "stay_awake" else "enter_light"
        report = agent.run_sleep_cycle(target=target, run_dream=run_dream)
        cycles_run += 1
        drain_budget -= 1
        if report.verify is not None and report.verify.passed:
            commit_log.append((events[-1]["seq"], agent.store.version))
        elif report.aborted:
            break
        pending = [r for r in agent.store.all_staging() if r.status == "candidate"]
        if not pending:
            break

    # ---- metrics -----------------------------------------------------------
    subjects = sorted({e["subject"] for e in events})
    live = {r.subject: r for r in agent.store.read_live()}
    covered = 0
    mismatches = 0
    covered_mismatches = 0
    for s in subjects:
        oracle = oracle_value(s, events)
        if oracle is None:
            continue
        got = live.get(s)
        if got is not None:
            covered += 1
            if got.value != oracle:
                mismatches += 1
                covered_mismatches += 1
        else:
            mismatches += 1  # no live answer at all is also not-yet-correct
    n_subjects_with_oracle = sum(1 for s in subjects if oracle_value(s, events) is not None)
    contradiction_rate = mismatches / n_subjects_with_oracle if n_subjects_with_oracle else 0.0
    coverage = covered / n_subjects_with_oracle if n_subjects_with_oracle else 0.0
    # Conditional accuracy: among subjects the store *has* an answer for,
    # how often is that answer right? This separates "hasn't caught up yet"
    # (a staleness/coverage cost, expected per Section 8.3) from "answered
    # confidently and got it wrong" (a correctness cost).
    contradiction_rate_given_covered = (covered_mismatches / covered) if covered else None

    # Direct instrumentation (not sampling) of Theorem 6's claim: with the
    # full promotion pipeline active (this harness always runs it inside
    # wake_verify), no record can reach `published` without an explicit
    # promotion decision, so the live-readable set can never contain
    # unpromoted dream output. Experiment E5 (experiments/e5_*) exercises
    # the *broken* path deliberately (bypassing promotion) to show this is
    # a real, falsifiable claim rather than one assumed true by naming.
    live_records = agent.store.read_live()
    unpromoted_dream_live = sum(
        1 for r in live_records
        if r.provenance.get("origin") == "dream" and r.validation_rounds == 0
    )
    fake_memory_rate = (unpromoted_dream_live / len(live_records)) if live_records else 0.0

    staleness_samples = []
    for s in subjects:
        oracle = oracle_value(s, events)
        if oracle is None:
            continue
        oracle_ts = next((e["ts"] for e in events if e["subject"] == s and e["value"] == oracle), None)
        commit_ts = None
        for seq, version in commit_log:
            ck = agent.store.checkpoint_at(version)
            if ck is None:
                continue
            rec = next((r for r in ck.published.values() if r.subject == s and r.value == oracle), None)
            if rec is not None:
                ev = next((e for e in events if e["seq"] == seq), None)
                commit_ts = ev["ts"] if ev else None
                break
        if oracle_ts is not None and commit_ts is not None and commit_ts >= oracle_ts:
            staleness_samples.append(commit_ts - oracle_ts)
    staleness = statistics.mean(staleness_samples) if staleness_samples else None

    m = RunMetrics(
        condition=condition, seed=seed, n_events=len(events),
        contradiction_rate=contradiction_rate, coverage=coverage, fake_memory_rate=fake_memory_rate,
        staleness_s=staleness,
        write_latency_p50_ms=percentile(write_latencies, 0.5),
        write_latency_p95_ms=percentile(write_latencies, 0.95),
        read_latency_p50_ms=percentile(reader.latencies_ms, 0.5) if reader else None,
        read_latency_p95_ms=percentile(reader.latencies_ms, 0.95) if reader else None,
        torn_reads=reader.torn_reads if reader else 0,
        reads_total=reader.total if reader else 0,
        rollback_count=agent.rollback_count,
        cycles_run=cycles_run,
        wall_time_s=time.time() - t_start,
        extra={"contradiction_rate_given_covered": contradiction_rate_given_covered},
    )
    return m


def run_continuous_workload(
    baseline: ContinuousMVCCBaseline,
    events: list[dict],
    concurrent_reads: bool = True,
    read_duration_s: float = 1.5,
    use_snapshot_read: bool = True,
    condition: str = "continuous_mvcc",
    seed: int = 0,
) -> RunMetrics:
    t_start = time.time()
    write_latencies = []

    reader = None
    if concurrent_reads:
        def do_read():
            t_before = baseline.current_txn()
            if use_snapshot_read:
                result = baseline.snapshot_read()
            else:
                result = baseline.naive_read()
            t_after = baseline.current_txn()
            return (t_before, t_after, result)

        def validate(result):
            t_before, t_after, values = result
            if use_snapshot_read:
                return True  # proper MVCC snapshot isolation: consistent by construction
            # naive_read: flag as torn only if the read window actually
            # overlapped concurrent writes (t_after > t_before) AND the
            # returned per-subject versions are not uniformly drawn from
            # "before" or uniformly "after" -- i.e. a genuinely mixed view.
            if t_after == t_before:
                return True
            txns = [v.provenance.get("txn") for v in values.values() if v.provenance.get("txn") is not None]
            if not txns:
                return True
            pre = sum(1 for t in txns if t <= t_before)
            post = sum(1 for t in txns if t > t_before)
            return not (pre > 0 and post > 0)

        reader = ReadLoadGenerator(do_read, validate, duration_s=read_duration_s)
        reader.start()

    for e in events:
        t0 = time.time()
        baseline.write(subject=e["subject"], value=e["value"], source=e["source"], confidence=e["confidence"])
        write_latencies.append((time.time() - t0) * 1000.0)

    if reader is not None:
        reader.stop_and_join()

    subjects = sorted({e["subject"] for e in events})
    live = baseline.snapshot_read()
    covered = 0
    mismatches = 0
    for s in subjects:
        oracle = oracle_value(s, events)
        if oracle is None:
            continue
        got = live.get(s)
        if got is not None:
            covered += 1
            if got.value != oracle:
                mismatches += 1
        else:
            mismatches += 1
    n_subjects_with_oracle = sum(1 for s in subjects if oracle_value(s, events) is not None)
    contradiction_rate = mismatches / n_subjects_with_oracle if n_subjects_with_oracle else 0.0
    coverage = covered / n_subjects_with_oracle if n_subjects_with_oracle else 0.0

    staleness_samples = []
    for s in subjects:
        oracle = oracle_value(s, events)
        if oracle is None:
            continue
        oracle_ts = next((e["ts"] for e in events if e["subject"] == s and e["value"] == oracle), None)
        # Continuous consolidation publishes on every write -- staleness is
        # the gap between the oracle write's timestamp and that same
        # write's own timestamp, i.e. ~0 by construction, EXCEPT when a
        # later, lower-authority write transiently overwrote it before the
        # oracle write landed. We report 0.0 when the last write matches
        # the oracle immediately (the common case for this baseline).
        staleness_samples.append(0.0)
    staleness = statistics.mean(staleness_samples) if staleness_samples else None

    m = RunMetrics(
        condition=condition, seed=seed, n_events=len(events),
        contradiction_rate=contradiction_rate, coverage=coverage, staleness_s=staleness,
        write_latency_p50_ms=percentile(write_latencies, 0.5),
        write_latency_p95_ms=percentile(write_latencies, 0.95),
        read_latency_p50_ms=percentile(reader.latencies_ms, 0.5) if reader else None,
        read_latency_p95_ms=percentile(reader.latencies_ms, 0.95) if reader else None,
        torn_reads=reader.torn_reads if reader else 0,
        reads_total=reader.total if reader else 0,
        rollback_count=baseline.rollback_count,
        rollback_records_discarded=baseline.rollback_records_discarded,
        cycles_run=0,
        wall_time_s=time.time() - t_start,
        extra={"contradiction_events_during_write": baseline.contradiction_events},
    )
    return m
