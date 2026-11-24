"""
Algorithm A4 -- Dream Phase: Bounded Synthetic Rehearsal.

Preserves: I4 (private checkpoint before any theta update); the clip bound
on the update rule is what keeps Theorem 3's atomic-unit bound intact (each
synthetic task, including its possible update, is one atomic unit).

Cost: O(budget.n_seeds * |G|) synthetic tasks generated, each scored in a
single forward pass and, on failure to meet calibration_threshold, updated
with a single bounded, clipped gradient step.

Dream-generated memory candidates enter at TrustTier.T0_SYNTHETIC (M_D) and
status CANDIDATE -- by I6 they are never written into `published`/M_V here;
Algorithm A7 (promotion.py), run later inside WAKE_VERIFY, is the only path
by which they can ever become live-readable.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from aisos.memory.record import MemoryRecord, TrustTier, Status
from aisos.memory.deep_sleep import dedup_priority
from aisos.security.wal import WriteAheadLog


@dataclass
class DreamBudget:
    n_seeds: int = 10
    clip_bound: float = 0.05
    lr: float = 0.02


@dataclass
class DreamResult:
    tasks_generated: int
    updates_applied: int
    synthetic_candidates: list[MemoryRecord]
    theta_after: list[float]
    theta_delta_norm: float
    preempted: bool = False


ScoreFn = Callable[[dict, list[float], random.Random], float]


def _default_score_fn(node: dict, theta: list[float], rng: random.Random) -> float:
    # Deterministic-ish surrogate: nodes flagged as "known weakness" score
    # low more often, letting the calibration loop actually do work; this
    # is the pluggable slot a real deployment replaces with an LLM forward
    # pass against a held-out synthetic task.
    base = 0.8 - 0.5 * float(node.get("weakness", 0.0))
    noise = rng.uniform(-0.15, 0.15)
    return max(0.0, min(1.0, base + noise))


def dream_phase(
    private_staging: dict,
    weakness_graph: list[dict],
    theta: list[float],
    budget: DreamBudget,
    wal: WriteAheadLog,
    should_preempt: Optional[Callable[[], bool]] = None,
    score_fn: Optional[ScoreFn] = None,
    calibration_threshold: float = 0.6,
    seed: Optional[int] = None,
) -> DreamResult:
    rng = random.Random(seed)
    score_fn = score_fn or _default_score_fn
    theta_before = list(theta)
    theta_cur = list(theta)
    generated: list[MemoryRecord] = []
    updates = 0
    tasks = 0
    preempted = False

    for s in range(budget.n_seeds):
        if preempted:
            break
        for node in weakness_graph:
            # Checked per-task, not per-seed: each synthetic task (including
            # its possible theta update) is the atomic unit of preemption
            # (module docstring), so dispatch latency must be bounded per
            # task, not by the whole weakness graph for a seed.
            if should_preempt is not None and should_preempt():
                preempted = True
                break
            tasks += 1
            score = score_fn(node, theta_cur, rng)

            if score < calibration_threshold:
                grad = rng.uniform(-1.0, 1.0)
                step = max(-budget.clip_bound, min(budget.clip_bound, -budget.lr * grad))
                theta_cur = [t + step for t in theta_cur]
                updates += 1

            rec = MemoryRecord(
                subject=node["subject"],
                value=node.get("proposed_value", node.get("value", "?")),
                source="dream",
                trust_tier=TrustTier.T0_SYNTHETIC,
                status=Status.CANDIDATE,
                confidence=score,
                provenance={"origin": "dream", "seed": s, "weakness": node.get("weakness", 0.0)},
            )
            generated.append(rec)

    # Adaptive Candidate Backlog Control
    # (AISOS_adaptive_candidate_backlog_next_milestone.md): the loop above
    # produces `n_seeds` candidates per rehearsed subject every cycle, all
    # sharing the identical (subject, value) -- the calibration loop needs
    # every one of those `n_seeds` scoring/theta-update passes (unchanged
    # above), but inserting all of them into staging is pure duplication:
    # deep_sleep.py's own dedup would collapse them to one anyway, one
    # full cycle later, by which point a fresh duplicate batch has already
    # been generated (docs/candidate_backlog_model.md's root-cause
    # finding). Coalescing HERE, before any of them ever reaches
    # private_staging/promotion_pipeline, removes the duplication at its
    # source using the SAME priority ordering deep_sleep.py already
    # trusts (dedup_priority) -- not a new selection rule.
    survivors: dict[tuple, MemoryRecord] = {}
    for rec in generated:
        key = (rec.subject, rec.value)
        existing = survivors.get(key)
        if existing is None or dedup_priority(rec) > dedup_priority(existing):
            survivors[key] = rec
    synthetic = list(survivors.values())
    for rec in synthetic:
        wal.append(op="dream_candidate", record_id=rec.id, payload=rec.content_hash())
        private_staging[rec.id] = rec

    delta_norm = sum((a - b) ** 2 for a, b in zip(theta_cur, theta_before)) ** 0.5
    return DreamResult(
        tasks_generated=tasks,
        updates_applied=updates,
        synthetic_candidates=synthetic,
        theta_after=theta_cur,
        theta_delta_norm=delta_norm,
        preempted=preempted,
    )
