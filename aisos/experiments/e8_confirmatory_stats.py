"""
Statistical analysis for the E8 confirmatory experiment
(AISOS_Next_Confirmatory_Experiment_Master_Prompt.md Sections 28-31).

Statistical unit: the 5 workload seeds/runs, NOT the ~24,000 individual
interactions (Section 28 explicitly warns against this pseudoreplication).
Every comparison below is a PAIRED comparison across seeds -- each seed
contributes exactly one (periodic_aisos, no_sleep_naive, continuous_mvcc)
triple built from the identical workload realization for that seed.

Method: paired bootstrap over the 5 per-seed differences, not a t-test
(explicitly disallowed by Section 29 without justification; n=5 is far
too small for a t-test's normality assumption to be meaningfully
checked). A paired Wilcoxon signed-rank test is also reported where it is
not degenerate, with its power limitation at n=5 stated explicitly
(Section 46) -- the smallest achievable two-sided p-value at n=5 nonzero
paired differences is 0.0625, so "not significant" at n=5 is uninformative
on its own and must be read alongside the bootstrap CI and the raw
per-seed values, not instead of them.
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIRMATORY_ROOT = ROOT / "results" / "raw" / "e8_confirmatory"

N_BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 12345
CONFIDENCE_LEVEL = 0.95


def load_seed_summaries(seeds: list[int], conditions: list[str]) -> dict:
    """Returns {seed: {condition: summary_dict}}."""
    out = {}
    for seed in seeds:
        seed_dir = CONFIRMATORY_ROOT / f"seed_{seed}"
        summary_files = sorted(seed_dir.glob("e8_confirmatory_seed*_summary.json"))
        if not summary_files:
            raise FileNotFoundError(f"no summary file found for seed {seed} in {seed_dir}")
        data = json.loads(summary_files[-1].read_text())
        out[seed] = {c: data["summaries"][c] for c in conditions}
    return out


def paired_differences(seed_summaries: dict, metric: str, cond_a: str, cond_b: str) -> list[float]:
    """[metric(cond_a, seed) - metric(cond_b, seed) for each seed], skipping
    seeds where the metric is None for either condition (e.g. a metric
    like dangerous_contradiction_error_rate with a zero-query denominator)."""
    diffs = []
    for seed, conds in seed_summaries.items():
        a = conds[cond_a].get(metric)
        b = conds[cond_b].get(metric)
        if a is None or b is None:
            continue
        diffs.append(a - b)
    return diffs


def bootstrap_ci(values: list[float], n_resamples: int = N_BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
                  confidence: float = CONFIDENCE_LEVEL) -> dict:
    """Paired bootstrap: resample the per-seed DIFFERENCES with
    replacement (not the raw per-condition values), n_resamples times,
    compute the mean of each resample, and report the empirical
    percentile CI. This is the standard paired-bootstrap-of-differences
    procedure."""
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0, "n_resamples": n_resamples, "seed": seed}
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(resample))
    means.sort()
    alpha = 1 - confidence
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    return {
        "mean": statistics.mean(values), "median": statistics.median(values),
        "stdev": statistics.stdev(values) if n > 1 else 0.0,
        "ci_low": means[lo_idx], "ci_high": means[hi_idx],
        "confidence_level": confidence, "n": n, "n_resamples": n_resamples, "resample_seed": seed,
        "per_seed_values": values,
    }


def wilcoxon_signed_rank(diffs: list[float]) -> dict:
    """Minimal paired Wilcoxon signed-rank test (two-sided, no continuity
    correction, exact for small n via a lookup-free direct enumeration --
    appropriate given n<=5 here). Returns None-statistic if all diffs are
    zero (degenerate: no evidence either direction, not "no effect")."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return {"statistic": None, "p_value": None, "n_nonzero": 0,
                "note": "all paired differences are exactly zero -- degenerate, not evidence of no effect"}
    ranks = sorted(range(n), key=lambda i: abs(nz[i]))
    rank_of = {}
    for rank, i in enumerate(ranks, start=1):
        rank_of[i] = rank
    # average ranks for ties
    from collections import defaultdict
    by_abs = defaultdict(list)
    for i in range(n):
        by_abs[abs(nz[i])].append(i)
    for group in by_abs.values():
        if len(group) > 1:
            avg = sum(rank_of[i] for i in group) / len(group)
            for i in group:
                rank_of[i] = avg
    w_plus = sum(rank_of[i] for i in range(n) if nz[i] > 0)
    w_minus = sum(rank_of[i] for i in range(n) if nz[i] < 0)
    w = min(w_plus, w_minus)
    # Exact two-sided p-value via full enumeration of sign assignments (n<=5 -> 2^n<=32, trivial).
    from itertools import product
    all_w = []
    for signs in product([1, -1], repeat=n):
        wp = sum(rank_of[i] for i in range(n) if signs[i] > 0)
        all_w.append(wp)
    n_as_extreme = sum(1 for x in all_w if x <= w or x >= (sum(rank_of.values()) - w))
    p_value = n_as_extreme / len(all_w)
    return {
        "statistic": w, "w_plus": w_plus, "w_minus": w_minus, "n_nonzero": n, "p_value": p_value,
        "note": f"exact two-sided test via full sign enumeration (2^{n}={2**n} assignments); "
                f"minimum achievable p-value at n_nonzero={n} is {2/2**n if n else None} -- "
                f"treat 'not significant' as uninformative at this sample size, not as evidence of no effect",
    }


def analyze_primary_comparison(seed_summaries: dict, metric: str, cond_a: str, cond_b: str) -> dict:
    diffs = paired_differences(seed_summaries, metric, cond_a, cond_b)
    per_seed = {
        seed: {
            cond_a: seed_summaries[seed][cond_a].get(metric),
            cond_b: seed_summaries[seed][cond_b].get(metric),
            "diff": (seed_summaries[seed][cond_a].get(metric) - seed_summaries[seed][cond_b].get(metric))
                    if seed_summaries[seed][cond_a].get(metric) is not None
                    and seed_summaries[seed][cond_b].get(metric) is not None else None,
        }
        for seed in seed_summaries
    }
    boot = bootstrap_ci(diffs)
    wilcoxon = wilcoxon_signed_rank(diffs)
    mean_a = statistics.mean(v[cond_a] for v in per_seed.values() if v[cond_a] is not None) if any(v[cond_a] is not None for v in per_seed.values()) else None
    mean_b = statistics.mean(v[cond_b] for v in per_seed.values() if v[cond_b] is not None) if any(v[cond_b] is not None for v in per_seed.values()) else None
    relative_diff = ((mean_a - mean_b) / mean_b) if (mean_a is not None and mean_b and mean_b != 0) else None
    return {
        "metric": metric, "condition_a": cond_a, "condition_b": cond_b,
        "per_seed": per_seed, "mean_a": mean_a, "mean_b": mean_b,
        "absolute_difference": boot["mean"], "relative_difference": relative_diff,
        "bootstrap": boot, "wilcoxon": wilcoxon,
    }


def run_full_analysis(seeds: list[int], conditions: list[str]) -> dict:
    seed_summaries = load_seed_summaries(seeds, conditions)

    primary = {
        "periodic_vs_no_sleep__dangerous_contradiction_error_rate": analyze_primary_comparison(
            seed_summaries, "dangerous_contradiction_error_rate", "periodic_aisos", "no_sleep_naive"),
        "periodic_vs_continuous__dangerous_contradiction_error_rate": analyze_primary_comparison(
            seed_summaries, "dangerous_contradiction_error_rate", "periodic_aisos", "continuous_mvcc"),
    }
    secondary_metrics = ["retrieval_accuracy", "stale_rate", "unknown_rate", "correct_rate", "cycles_run"]
    secondary = {}
    for metric in secondary_metrics:
        secondary[f"periodic_vs_no_sleep__{metric}"] = analyze_primary_comparison(
            seed_summaries, metric, "periodic_aisos", "no_sleep_naive")
        secondary[f"periodic_vs_continuous__{metric}"] = analyze_primary_comparison(
            seed_summaries, metric, "periodic_aisos", "continuous_mvcc")

    return {
        "seeds": seeds, "conditions": conditions,
        "statistical_unit": "workload seed (n=5), not individual interaction -- see module docstring",
        "primary_comparisons": primary,
        "secondary_comparisons": secondary,
        "raw_per_seed_summaries": seed_summaries,
    }


def main():
    from aisos.experiments.e8_confirmatory import SEEDS, CONDITIONS
    from aisos.experiments.common import save_json
    result = run_full_analysis(SEEDS, CONDITIONS)
    path = save_json("e8_confirmatory_statistics", result)
    print(f"Saved -> {path}")
    for name, comp in result["primary_comparisons"].items():
        print(f"\n{name}:")
        print(f"  mean_a(periodic)={comp['mean_a']}, mean_b(baseline)={comp['mean_b']}")
        print(f"  absolute_difference={comp['absolute_difference']}, relative_difference={comp['relative_difference']}")
        print(f"  bootstrap 95% CI: [{comp['bootstrap']['ci_low']}, {comp['bootstrap']['ci_high']}]")
        print(f"  wilcoxon: {comp['wilcoxon']}")
    return result


if __name__ == "__main__":
    main()
