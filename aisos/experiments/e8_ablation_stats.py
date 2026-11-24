"""
Statistical analysis for the authority ablation experiment
(AISOS_Authority_Ablation_Next_Milestone_Prompt.md Section 18).

Reuses the exact bootstrap/Wilcoxon methodology from
e8_confirmatory_stats.py (already independently unit-tested,
aisos/tests/test_confirmatory_stats.py) against the ablation run's own
raw data (results/raw/e8_ablation/), never touching the confirmatory
run's files.

Primary comparison (Section 16): periodic_aisos vs.
periodic_aisos_without_authority -- the direct test of H_AUTH.
"""
from __future__ import annotations

import json
from pathlib import Path

from aisos.experiments.e8_confirmatory_stats import bootstrap_ci, wilcoxon_signed_rank, paired_differences

ROOT = Path(__file__).resolve().parent.parent.parent
ABLATION_ROOT = ROOT / "results" / "raw" / "e8_ablation"
CONDITIONS = ["no_sleep_naive", "continuous_mvcc", "periodic_aisos", "periodic_aisos_without_authority"]
SEEDS = [0, 1, 2, 3, 4]


def load_seed_summaries(seeds: list[int] = SEEDS, conditions: list[str] = CONDITIONS) -> dict:
    out = {}
    for seed in seeds:
        seed_dir = ABLATION_ROOT / f"seed_{seed}"
        summary_files = sorted(seed_dir.glob("e8_ablation_seed*_summary.json"))
        if not summary_files:
            raise FileNotFoundError(f"no summary file for seed {seed} in {seed_dir}")
        data = json.loads(summary_files[-1].read_text())
        out[seed] = {c: data["summaries"][c] for c in conditions}
    return out


def analyze_comparison(seed_summaries: dict, metric: str, cond_a: str, cond_b: str) -> dict:
    diffs = paired_differences(seed_summaries, metric, cond_a, cond_b)
    per_seed = {
        seed: {
            cond_a: seed_summaries[seed][cond_a].get(metric),
            cond_b: seed_summaries[seed][cond_b].get(metric),
        }
        for seed in seed_summaries
    }
    boot = bootstrap_ci(diffs)
    wilcoxon = wilcoxon_signed_rank(diffs)
    import statistics
    vals_a = [v[cond_a] for v in per_seed.values() if v[cond_a] is not None]
    vals_b = [v[cond_b] for v in per_seed.values() if v[cond_b] is not None]
    mean_a = statistics.mean(vals_a) if vals_a else None
    mean_b = statistics.mean(vals_b) if vals_b else None
    return {
        "metric": metric, "condition_a": cond_a, "condition_b": cond_b,
        "per_seed": per_seed, "mean_a": mean_a, "mean_b": mean_b,
        "absolute_difference": boot["mean"], "bootstrap": boot, "wilcoxon": wilcoxon,
    }


def run_full_analysis(seeds: list[int] = SEEDS, conditions: list[str] = CONDITIONS) -> dict:
    seed_summaries = load_seed_summaries(seeds, conditions)
    primary = {
        "periodic_vs_ablated__dangerous_contradiction_error_rate": analyze_comparison(
            seed_summaries, "dangerous_contradiction_error_rate", "periodic_aisos", "periodic_aisos_without_authority"),
    }
    secondary = {}
    metrics = ["dangerous_contradiction_error_rate", "retrieval_accuracy", "stale_rate", "unknown_rate", "cycles_run"]
    pairs = [
        ("periodic_aisos_without_authority", "no_sleep_naive"),
        ("periodic_aisos_without_authority", "continuous_mvcc"),
        ("periodic_aisos", "no_sleep_naive"),
        ("periodic_aisos", "continuous_mvcc"),
    ]
    for metric in metrics:
        for a, b in pairs:
            secondary[f"{a}_vs_{b}__{metric}"] = analyze_comparison(seed_summaries, metric, a, b)
    return {
        "seeds": seeds, "conditions": conditions,
        "statistical_unit": "workload seed (n=5)",
        "primary_comparisons": primary,
        "secondary_comparisons": secondary,
        "raw_per_seed_summaries": seed_summaries,
    }


def main():
    from aisos.experiments.common import save_json
    result = run_full_analysis()
    path = save_json("e8_ablation_statistics", result)
    print(f"Saved -> {path}")
    comp = result["primary_comparisons"]["periodic_vs_ablated__dangerous_contradiction_error_rate"]
    print(f"\nPRIMARY (H_AUTH test): periodic_aisos vs periodic_aisos_without_authority, dangerous_contradiction_error_rate")
    print(f"  mean periodic={comp['mean_a']}, mean ablated={comp['mean_b']}")
    print(f"  absolute_difference={comp['absolute_difference']}")
    print(f"  bootstrap 95% CI: [{comp['bootstrap']['ci_low']}, {comp['bootstrap']['ci_high']}]")
    print(f"  wilcoxon: {comp['wilcoxon']}")
    print(f"  per_seed: {comp['per_seed']}")
    return result


if __name__ == "__main__":
    main()
