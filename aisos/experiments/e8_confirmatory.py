"""
E8 Confirmatory Experiment: 5 seeds x 3 conditions x 1,000 interactions
(AISOS_Next_Confirmatory_Experiment_Master_Prompt.md).

Explicitly excludes aisos_full (Section 3: keeping "does AISOS's memory
mechanism help" separate from "is the full scheduler calibrated" as two
different research questions, per this prompt's own reasoning).

Each seed's workload is generated once and hashed; all three conditions
for that seed consume the identical `specs` list (enforced by
e8_long_running_1000.py::main()'s own workload_hash() assertions). Results
are written to results/raw/e8_confirmatory/seed_{N}/, never touching or
overwriting results/raw/e8_long_running_1000/ (the pilot).

Absolute rule (Section 4): this script does not tune the scheduler,
promotion, trust, evaluator, prompts, or workload. It is a pure
replication of the pilot's periodic_aisos/no_sleep_naive/continuous_mvcc
conditions across 5 independent seeds.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from aisos.experiments.e8_long_running_1000 import main as run_seed

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIRMATORY_ROOT = ROOT / "results" / "raw" / "e8_confirmatory"
CONDITIONS = ["no_sleep_naive", "continuous_mvcc", "periodic_aisos"]
SEEDS = [0, 1, 2, 3, 4]


def run(seeds: list[int] = SEEDS, conditions: list[str] = CONDITIONS, max_interactions: int | None = None) -> dict:
    t0 = time.time()
    per_seed_summaries = {}
    for seed in seeds:
        seed_dir = CONFIRMATORY_ROOT / f"seed_{seed}"
        print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")
        summaries = run_seed(
            seed=seed, conditions=conditions, out_dir=seed_dir,
            max_interactions=max_interactions, experiment_id="e8_confirmatory",
        )
        per_seed_summaries[seed] = summaries

    elapsed = time.time() - t0
    print(f"\nAll {len(seeds)} seeds x {len(conditions)} conditions complete in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min).")

    index = {
        "experiment_id": "e8_confirmatory", "seeds": seeds, "conditions": conditions,
        "max_interactions": max_interactions, "total_wall_time_s": elapsed,
        "seed_dirs": [str(CONFIRMATORY_ROOT / f"seed_{s}") for s in seeds],
    }
    (CONFIRMATORY_ROOT / "confirmatory_index.json").write_text(json.dumps(index, indent=2, default=str))
    return per_seed_summaries


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "dry-run":
        print("=== DRY RUN: 25 interactions x 3 conditions x 1 seed ===")
        run(seeds=[0], max_interactions=25)
    else:
        print("=== FULL CONFIRMATORY RUN: 1000 interactions x 3 conditions x 5 seeds ===")
        run()
