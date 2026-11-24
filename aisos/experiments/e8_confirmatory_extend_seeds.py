"""
Statistical-power extension of E8 Confirmatory (reviewer feedback: n=5 is
small). Adds 25 MORE independent seeds (5-29) on top of the existing,
already-collected seeds 0-4 -- it does not rerun or alter seeds 0-4, and
does not touch the scheduler, promotion, trust, evaluator, prompts, or
workload generator (same "absolute rule" as e8_confirmatory.py itself).
Combined with the existing 5 seeds this gives n=30.
"""
from __future__ import annotations

import json
import time

from aisos.experiments.e8_confirmatory import run, CONFIRMATORY_ROOT, CONDITIONS

NEW_SEEDS = list(range(5, 30))  # 25 additional seeds -> 30 total with existing 0-4
ALL_SEEDS = list(range(30))


def main():
    t0 = time.time()
    print(f"=== EXTENDING E8 CONFIRMATORY: seeds 5-29 (25 new seeds), reusing existing seeds 0-4 unchanged ===")
    run(seeds=NEW_SEEDS, conditions=CONDITIONS)
    elapsed = time.time() - t0
    print(f"\nExtension complete: {len(NEW_SEEDS)} new seeds x {len(CONDITIONS)} conditions "
          f"in {elapsed:.1f}s ({elapsed/60:.1f} min).")

    index = {
        "experiment_id": "e8_confirmatory_extended", "seeds": ALL_SEEDS, "conditions": CONDITIONS,
        "new_seeds_this_run": NEW_SEEDS, "reused_seeds": [0, 1, 2, 3, 4],
        "extension_wall_time_s": elapsed,
        "seed_dirs": [str(CONFIRMATORY_ROOT / f"seed_{s}") for s in ALL_SEEDS],
    }
    (CONFIRMATORY_ROOT / "confirmatory_index_extended.json").write_text(json.dumps(index, indent=2, default=str))
    print(f"-> {CONFIRMATORY_ROOT / 'confirmatory_index_extended.json'}")


if __name__ == "__main__":
    main()
