"""Tests for the confirmatory experiment's statistical analysis module
(paired bootstrap + exact Wilcoxon), using synthetic per-seed data -- no
Ollama dependency."""
from __future__ import annotations

from aisos.experiments.e8_confirmatory_stats import (
    paired_differences, bootstrap_ci, wilcoxon_signed_rank, analyze_primary_comparison,
)


def _synthetic_seed_summaries(metric_a_values: list[float], metric_b_values: list[float], metric: str) -> dict:
    return {
        i: {"a": {metric: a}, "b": {metric: b}}
        for i, (a, b) in enumerate(zip(metric_a_values, metric_b_values))
    }


def test_paired_differences_basic():
    seed_summaries = _synthetic_seed_summaries([0.1, 0.2, 0.0, 0.15, 0.05], [0.5, 0.6, 0.4, 0.55, 0.45], "m")
    diffs = paired_differences(seed_summaries, "m", "a", "b")
    assert len(diffs) == 5
    assert all(d < 0 for d in diffs), "a consistently lower than b in this synthetic example"


def test_paired_differences_skips_none():
    seed_summaries = {
        0: {"a": {"m": 0.1}, "b": {"m": None}},
        1: {"a": {"m": 0.2}, "b": {"m": 0.3}},
    }
    diffs = paired_differences(seed_summaries, "m", "a", "b")
    assert len(diffs) == 1


def test_bootstrap_ci_shape_and_determinism():
    values = [-0.4, -0.4, -0.4, -0.4, -0.4]
    result = bootstrap_ci(values, n_resamples=1000, seed=1)
    assert result["mean"] == -0.4
    assert result["n"] == 5
    # constant input -> zero-width CI
    assert result["ci_low"] == result["ci_high"] == -0.4

    result2 = bootstrap_ci(values, n_resamples=1000, seed=1)
    assert result == result2, "same seed must reproduce identical bootstrap output"


def test_bootstrap_ci_empty_input():
    result = bootstrap_ci([])
    assert result["mean"] is None
    assert result["n"] == 0


def test_wilcoxon_all_same_sign_gives_small_p_value():
    # All 5 differences negative and distinct in magnitude -> the most
    # extreme possible rank configuration for one-directional data at n=5.
    diffs = [-0.1, -0.2, -0.3, -0.4, -0.5]
    result = wilcoxon_signed_rank(diffs)
    assert result["n_nonzero"] == 5
    assert result["p_value"] == 2 / 2**5  # minimum achievable at n=5, per the module's own documented note
    assert result["w_minus"] > result["w_plus"]


def test_wilcoxon_degenerate_all_zero():
    result = wilcoxon_signed_rank([0.0, 0.0, 0.0])
    assert result["statistic"] is None
    assert result["n_nonzero"] == 0
    assert "degenerate" in result["note"]


def test_wilcoxon_mixed_signs_larger_p_value():
    diffs = [0.1, -0.1, 0.1, -0.1, 0.05]
    result = wilcoxon_signed_rank(diffs)
    assert result["p_value"] > 2 / 2**5, "mixed-sign differences must not achieve the extreme minimum p-value"


def test_analyze_primary_comparison_end_to_end():
    seed_summaries = {
        i: {
            "periodic_aisos": {"dangerous_contradiction_error_rate": 0.0},
            "no_sleep_naive": {"dangerous_contradiction_error_rate": 0.75},
        }
        for i in range(5)
    }
    result = analyze_primary_comparison(seed_summaries, "dangerous_contradiction_error_rate", "periodic_aisos", "no_sleep_naive")
    assert result["mean_a"] == 0.0
    assert result["mean_b"] == 0.75
    assert result["absolute_difference"] == -0.75
    assert result["relative_difference"] == -1.0
    assert result["bootstrap"]["ci_low"] == result["bootstrap"]["ci_high"] == -0.75
