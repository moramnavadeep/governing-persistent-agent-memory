"""
Authority x Dream factorial, extended from the single constructed attack pattern in
aisos/tests/test_authority_dream_factorial.py to every workload family W1-W9
(aisos/sim/workload_families.py) x 5 seeds. Zero model calls, exactly like the original
factorial and the generalization sweep: backend.read() is the evaluated answer.

Cells (trust tier held ON in the first four, as in the original factorial):
    authority_on_dream_on    PeriodicAISOSBackend(dream_seeds=4)
    authority_off_dream_on   AuthorityAblatedPeriodicAISOSBackend(dream_seeds=4)
    authority_on_dream_off   PeriodicAISOSBackend(dream_seeds=0)
    authority_off_dream_off  AuthorityAblatedPeriodicAISOSBackend(dream_seeds=0)
plus the recency-only comparator (authority and trust tier both removed) with Dream on and off,
which is the Dream-confound isolation repeated across families, and the authenticated-credential
backend (authority from a verified credential) as the comparison for spoofed claims.

No mechanism code is changed; the generalization sweep's own runner and metric definitions are reused.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from aisos.agent.condition_backends import (
    AuthorityAblatedPeriodicAISOSBackend, PeriodicAISOSBackend, TrustedAuthorityPeriodicAISOSBackend,
    TrustTierAblatedPeriodicAISOSBackend,
)
from aisos.experiments import generalization_sweep as gs
from aisos.sim.workload_families import FAMILIES

CELLS = {
    "authority_on_dream_on": lambda: PeriodicAISOSBackend(dream_seeds=4),
    "authority_off_dream_on": lambda: AuthorityAblatedPeriodicAISOSBackend(dream_seeds=4),
    "authority_on_dream_off": lambda: PeriodicAISOSBackend(dream_seeds=0),
    "authority_off_dream_off": lambda: AuthorityAblatedPeriodicAISOSBackend(dream_seeds=0),
    "recency_only_dream_on": lambda: TrustTierAblatedPeriodicAISOSBackend(dream_seeds=4),
    "recency_only_dream_off": lambda: TrustTierAblatedPeriodicAISOSBackend(dream_seeds=0),
    # authority read from a verified credential instead of the claimed source; the sweep issues,
    # spoofs and replays credentials only for a condition with exactly this name
    "periodic_aisos_trusted_authority": lambda: TrustedAuthorityPeriodicAISOSBackend(dream_seeds=4),
}
SEEDS = [0, 1, 2, 3, 4]
ROOT = Path(__file__).resolve().parents[2]


def main() -> dict:
    gs.make_backend = lambda condition: CELLS[condition]()
    t0 = time.time()
    per_run = []
    for family in FAMILIES:
        for seed in SEEDS:
            out = gs.run_seed_workload(family, seed, conditions=list(CELLS))
            for cell, s in out["summaries"].items():
                per_run.append({
                    "family": family, "seed": seed, "cell": cell, "workload_hash": out["workload_hash"],
                    "n_contradictory": s["n_contradictory_category_queries"],
                    "n_dangerous": s["n_dangerous_contradiction_errors"],
                    "changing_accuracy": s["changing_category_accuracy"],
                    "stale_rate": s["stale_rate"],
                })
    pooled = {}
    for cell in CELLS:
        rows = [r for r in per_run if r["cell"] == cell]
        n_c = sum(r["n_contradictory"] for r in rows)
        n_d = sum(r["n_dangerous"] for r in rows)
        ch = [r["changing_accuracy"] for r in rows if r["changing_accuracy"] is not None]
        st = [r["stale_rate"] for r in rows if r["stale_rate"] is not None]
        pooled[cell] = {
            "runs": len(rows),
            "runs_with_any_dangerous": sum(1 for r in rows if r["n_dangerous"] > 0),
            "dangerous_errors": n_d, "contradictory_queries": n_c,
            "dangerous_rate": (n_d / n_c) if n_c else None,
            "mean_changing_accuracy": (sum(ch) / len(ch)) if ch else None,
            "mean_stale_rate": (sum(st) / len(st)) if st else None,
        }
    result = {
        "manifest": {"timestamp": time.time(), "families": list(FAMILIES), "seeds": SEEDS, "cells": list(CELLS),
                     "model_calls": 0, "wall_time_s": time.time() - t0},
        "pooled": pooled, "per_run": per_run,
    }
    out_path = ROOT / "results" / "authority_dream_factorial_9families.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(pooled, indent=2))
    print("->", out_path)
    return result


if __name__ == "__main__":
    main()
