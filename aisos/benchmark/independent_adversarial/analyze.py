"""
Paired statistical analysis for the independent adversarial benchmark
(AISOS_Independent_Adversarial_Benchmark_Problem.md Section 14).
McNemar's exact test via scipy.stats.binomtest, matching this project's
established convention throughout its history for paired binary
outcomes on shared queries -- explicitly not presented as independent-
seed replication (N=1 per scenario/system in this benchmark).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load_by_system(path: str) -> dict:
    d = json.load(open(ROOT / "results" / path))
    by_system = defaultdict(dict)
    for r in d["runs"]:
        if "eval" in r:
            by_system[r["system"]][r["attack_id"]] = r
    return by_system


def mcnemar(by_system: dict, sys_a: str, sys_b: str, metric: str = "answer_correctness") -> dict:
    """Paired exact test over shared attack_ids: sys_b correct where sys_a
    incorrect (b_only), sys_a correct where sys_b incorrect (a_only)."""
    a_rows, b_rows = by_system[sys_a], by_system[sys_b]
    shared = sorted(set(a_rows) & set(b_rows))
    a_only = b_only = both = neither = 0
    for aid in shared:
        av, bv = a_rows[aid]["eval"][metric], b_rows[aid]["eval"][metric]
        if av and bv:
            both += 1
        elif av and not bv:
            a_only += 1
        elif bv and not av:
            b_only += 1
        else:
            neither += 1
    n_discordant = a_only + b_only
    if n_discordant == 0:
        p = 1.0
    else:
        p = binomtest(min(a_only, b_only), n_discordant, 0.5, alternative="two-sided").pvalue
    return {
        "sys_a": sys_a, "sys_b": sys_b, "metric": metric, "n_shared": len(shared),
        "a_only_correct": a_only, "b_only_correct": b_only, "both_correct": both, "neither_correct": neither,
        "p_value": p,
    }


def full_report(result_file: str, primary_systems: list[str], baseline_systems: list[str]) -> list[dict]:
    by_system = load_by_system(result_file)
    out = []
    for metric in ("answer_correctness", "dangerous_contradiction"):
        for p in primary_systems:
            for b in baseline_systems:
                if p in by_system and b in by_system:
                    out.append(mcnemar(by_system, b, p, metric=metric))
    return out


if __name__ == "__main__":
    report = full_report(
        "independent_adversarial_benchmark_llama3_latest.json",
        primary_systems=["S6b_aisos_authenticated", "S8_smsr_style", "S6_aisos", "S7_aisos_plus_rag"],
        baseline_systems=["S1_no_sleep_naive", "S2_rag_standard", "S5_continuous_mvcc"],
    )
    for row in report:
        print(row)
