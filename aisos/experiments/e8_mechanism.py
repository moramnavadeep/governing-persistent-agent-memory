"""
E8 Trust-Tier Mechanism Ablation: 5 seeds x 4 conditions x 1,000 interactions
(AISOS_Complete_Next_Research_Engineering_Master_Prompt.md, H1/H2/H3;
docs/mechanism_hypotheses.md).

Compares periodic_aisos against periodic_aisos_without_trust_tier (built on
top of the already-authority-ablated comparator -- see
docs/mechanism_hypotheses.md's "Ablation Definition"), alongside the same
no_sleep_naive/continuous_mvcc baselines used by every prior E8 experiment,
so the primary metric's baseline comparison remains apples-to-apples.
Results written to results/raw/e8_mechanism/, never touching
results/raw/e8_long_running_1000/ (the pilot), results/raw/e8_confirmatory/
(the confirmatory run), or results/raw/e8_ablation/ (the authority
ablation) -- four separate, independently preserved experiments.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from aisos.experiments.e8_long_running_1000 import main as run_seed

ROOT = Path(__file__).resolve().parent.parent.parent
MECHANISM_ROOT = ROOT / "results" / "raw" / "e8_mechanism"
CONDITIONS = ["no_sleep_naive", "continuous_mvcc", "periodic_aisos", "periodic_aisos_without_trust_tier"]
SEEDS = [0, 1, 2, 3, 4]


def run(seeds: list[int] = SEEDS, conditions: list[str] = CONDITIONS, max_interactions: int | None = None) -> dict:
    t0 = time.time()
    per_seed_summaries = {}
    for seed in seeds:
        seed_dir = MECHANISM_ROOT / f"seed_{seed}"
        print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")
        summaries = run_seed(
            seed=seed, conditions=conditions, out_dir=seed_dir,
            max_interactions=max_interactions, experiment_id="e8_mechanism",
        )
        per_seed_summaries[seed] = summaries

    elapsed = time.time() - t0
    print(f"\nAll {len(seeds)} seeds x {len(conditions)} conditions complete in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min).")

    index = {
        "experiment_id": "e8_mechanism", "seeds": seeds, "conditions": conditions,
        "max_interactions": max_interactions, "total_wall_time_s": elapsed,
        "seed_dirs": [str(MECHANISM_ROOT / f"seed_{s}") for s in seeds],
    }
    (MECHANISM_ROOT / "mechanism_index.json").write_text(json.dumps(index, indent=2, default=str))
    return per_seed_summaries


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "dry-run":
        print("=== DRY RUN: 25 interactions x 4 conditions x 1 seed ===")
        run(seeds=[0], max_interactions=25)
    else:
        print("=== FULL TRUST-TIER MECHANISM RUN: 1000 interactions x 4 conditions x 5 seeds ===")
        run()
