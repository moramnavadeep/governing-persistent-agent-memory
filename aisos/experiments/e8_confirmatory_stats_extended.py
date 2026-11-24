"""
Statistical analysis for the EXTENDED E8 confirmatory experiment (n=30
seeds, reviewer feedback on the original n=5 report). Reuses the paired
bootstrap of per-seed differences from e8_confirmatory_stats.py UNCHANGED,
and adds an interquartile range (IQR) per condition and a paired effect
size (Cohen's d_z: mean paired difference / stdev of paired differences).

IMPORTANT: this does NOT reuse e8_confirmatory_stats.py's
wilcoxon_signed_rank(), which computes an EXACT p-value by enumerating
all 2^n sign assignments -- explicitly documented there as "trivial" only
because the original script's n<=5 (2^5=32). At n=30 that enumeration is
2^30 (~1.07 billion) sign assignments; running it here first exhausted
several GB of RAM before being killed. For n=30, this module instead uses
the standard normal approximation to the Wilcoxon signed-rank statistic
(with both a tie correction and a continuity correction), which is the
textbook-correct choice once n is no longer small enough for exact
enumeration.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from itertools import product
from pathlib import Path

from aisos.experiments.e8_confirmatory_stats import load_seed_summaries, paired_differences, bootstrap_ci

ROOT = Path(__file__).resolve().parent.parent.parent

EXACT_ENUMERATION_MAX_N = 15  # 2^15 = 32,768 -- still cheap; above this, use the normal approximation


def wilcoxon_signed_rank_auto(diffs: list[float]) -> dict:
    """Wilcoxon signed-rank test that picks a tractable method by sample
    size: exact sign enumeration for n_nonzero <= EXACT_ENUMERATION_MAX_N
    (identical method to e8_confirmatory_stats.py's version), otherwise the
    normal approximation with tie and continuity correction. Never builds a
    2^n-sized list for n larger than the cutoff."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return {"statistic": None, "p_value": None, "n_nonzero": 0, "method": "degenerate",
                "note": "all paired differences are exactly zero -- degenerate, not evidence of no effect"}

    ranks = sorted(range(n), key=lambda i: abs(nz[i]))
    rank_of = {}
    for rank, i in enumerate(ranks, start=1):
        rank_of[i] = rank
    by_abs = defaultdict(list)
    for i in range(n):
        by_abs[abs(nz[i])].append(i)
    tie_group_sizes = []
    for group in by_abs.values():
        if len(group) > 1:
            avg = sum(rank_of[i] for i in group) / len(group)
            for i in group:
                rank_of[i] = avg
            tie_group_sizes.append(len(group))

    w_plus = sum(rank_of[i] for i in range(n) if nz[i] > 0)
    w_minus = sum(rank_of[i] for i in range(n) if nz[i] < 0)
    w = min(w_plus, w_minus)

    if n <= EXACT_ENUMERATION_MAX_N:
        all_w = []
        for signs in product([1, -1], repeat=n):
            wp = sum(rank_of[i] for i in range(n) if signs[i] > 0)
            all_w.append(wp)
        n_as_extreme = sum(1 for x in all_w if x <= w or x >= (sum(rank_of.values()) - w))
        p_value = n_as_extreme / len(all_w)
        return {
            "statistic": w, "w_plus": w_plus, "w_minus": w_minus, "n_nonzero": n, "p_value": p_value,
            "method": "exact_enumeration",
            "note": f"exact two-sided test via full sign enumeration (2^{n}={2**n} assignments)",
        }

    # Normal approximation: mean/variance of W+ under H0, with the standard
    # tie-correction term subtracted from the variance, and a continuity
    # correction of 0.5 applied toward the mean.
    mean_w = n * (n + 1) / 4.0
    tie_correction = sum(t ** 3 - t for t in tie_group_sizes) / 48.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction
    if var_w <= 0:
        return {"statistic": w, "w_plus": w_plus, "w_minus": w_minus, "n_nonzero": n, "p_value": None,
                "method": "normal_approximation", "note": "degenerate variance (all values tied)"}
    sd_w = math.sqrt(var_w)
    diff_from_mean = w_plus - mean_w
    cc = 0.5 if diff_from_mean != 0 else 0.0
    z = (diff_from_mean - math.copysign(cc, diff_from_mean)) / sd_w
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2))))
    p_value = min(1.0, p_value)
    return {
        "statistic": w, "w_plus": w_plus, "w_minus": w_minus, "n_nonzero": n,
        "z": z, "p_value": p_value, "method": "normal_approximation",
        "note": f"normal approximation to the Wilcoxon signed-rank statistic (n_nonzero={n} > "
                f"{EXACT_ENUMERATION_MAX_N}, exact 2^n enumeration is intractable at this n); "
                f"tie-corrected variance, continuity-corrected z",
    }


def iqr(values: list[float]) -> dict:
    if not values:
        return {"q1": None, "median": None, "q3": None, "iqr": None}
    s = sorted(values)
    n = len(s)
    def _pct(p):
        if n == 1:
            return s[0]
        idx = p * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        frac = idx - lo
        return s[lo] + (s[hi] - s[lo]) * frac
    q1, med, q3 = _pct(0.25), _pct(0.5), _pct(0.75)
    return {"q1": q1, "median": med, "q3": q3, "iqr": q3 - q1, "min": s[0], "max": s[-1], "n": n}


def cohens_dz(diffs: list[float]) -> float | None:
    if len(diffs) < 2:
        return None
    sd = statistics.stdev(diffs)
    if sd == 0:
        return None
    return statistics.mean(diffs) / sd


def analyze_extended(seed_summaries: dict, metric: str, cond_a: str, cond_b: str) -> dict:
    diffs = paired_differences(seed_summaries, metric, cond_a, cond_b)
    values_a = [seed_summaries[s][cond_a].get(metric) for s in seed_summaries if seed_summaries[s][cond_a].get(metric) is not None]
    values_b = [seed_summaries[s][cond_b].get(metric) for s in seed_summaries if seed_summaries[s][cond_b].get(metric) is not None]
    per_seed = {
        seed: {
            cond_a: seed_summaries[seed][cond_a].get(metric),
            cond_b: seed_summaries[seed][cond_b].get(metric),
        }
        for seed in sorted(seed_summaries)
    }
    return {
        "metric": metric, "condition_a": cond_a, "condition_b": cond_b, "n_seeds": len(seed_summaries),
        "per_seed": per_seed,
        f"{cond_a}_iqr": iqr(values_a), f"{cond_b}_iqr": iqr(values_b),
        "diffs_iqr": iqr(diffs),
        "bootstrap_95ci": bootstrap_ci(diffs),
        "wilcoxon": wilcoxon_signed_rank_auto(diffs),
        "cohens_dz": cohens_dz(diffs),
    }


def run_extended_analysis(seeds: list[int], conditions: list[str]) -> dict:
    seed_summaries = load_seed_summaries(seeds, conditions)
    primary = {
        "periodic_vs_no_sleep__dangerous_contradiction_error_rate": analyze_extended(
            seed_summaries, "dangerous_contradiction_error_rate", "periodic_aisos", "no_sleep_naive"),
        "periodic_vs_continuous__dangerous_contradiction_error_rate": analyze_extended(
            seed_summaries, "dangerous_contradiction_error_rate", "periodic_aisos", "continuous_mvcc"),
    }
    secondary_metrics = ["retrieval_accuracy", "stale_rate", "unknown_rate", "correct_rate", "cycles_run"]
    secondary = {}
    for metric in secondary_metrics:
        secondary[f"periodic_vs_no_sleep__{metric}"] = analyze_extended(
            seed_summaries, metric, "periodic_aisos", "no_sleep_naive")
        secondary[f"periodic_vs_continuous__{metric}"] = analyze_extended(
            seed_summaries, metric, "periodic_aisos", "continuous_mvcc")
    return {
        "seeds": seeds, "n_seeds": len(seeds), "conditions": conditions,
        "statistical_unit": "workload seed (n=30), not individual interaction",
        "primary_comparisons": primary, "secondary_comparisons": secondary,
    }


def main():
    from aisos.experiments.e8_confirmatory import CONDITIONS
    from aisos.experiments.common import save_json
    seeds = list(range(30))
    result = run_extended_analysis(seeds, CONDITIONS)
    path = save_json("e8_confirmatory_statistics_extended_n30", result)
    print(f"Saved -> {path}")
    for name, comp in result["primary_comparisons"].items():
        print(f"\n{name}:")
        print(f"  n_seeds={comp['n_seeds']}")
        print(f"  periodic_aisos IQR: {comp['periodic_aisos_iqr']}")
        print(f"  baseline ({comp['condition_b']}) IQR: {comp[comp['condition_b'] + '_iqr']}")
        print(f"  diffs median/IQR: {comp['diffs_iqr']}")
        print(f"  bootstrap 95% CI: [{comp['bootstrap_95ci']['ci_low']}, {comp['bootstrap_95ci']['ci_high']}]")
        print(f"  wilcoxon: p={comp['wilcoxon'].get('p_value')}")
        print(f"  cohen's d_z: {comp['cohens_dz']}")
    return result


if __name__ == "__main__":
    main()
