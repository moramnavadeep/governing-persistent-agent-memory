"""
Check every quantitative claim of the AIEI 2027 paper against the code and the result files in this repository.

    python verify_claims.py            # checks the stored result files and the code constants (a few seconds)
    python verify_claims.py --rerun    # also re-runs the zero-model-call experiments and compares them
                                       # with the stored files (about 2 minutes, no LLM needed)

Experiments that call a language model (llama3 / qwen3 through Ollama) are NOT re-run here; their stored
outputs, including every prompt and model response, are checked instead. See CLAIMS.md for how to re-run them.

Every check prints PASS or FAIL with the claim id used in CLAIMS.md. Exit code is 1 if any check fails.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
import sys
from collections import Counter
from fractions import Fraction
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
results: list[tuple[str, bool, str]] = []


def check(cid: str, ok: bool, text: str) -> None:
    results.append((cid, bool(ok), text))
    print(f"[{'PASS' if ok else 'FAIL'}] {cid:5s} {text}")


def load(name: str):
    return json.loads((RES / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- statistics (exact, no approximation)
def exact_wilcoxon_two_sided(diffs: list[float]) -> Fraction:
    """Exact two-sided Wilcoxon signed-rank p-value with average ranks for ties, by dynamic programming over
    all 2^n sign assignments (exact for any n; no normal approximation)."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    r2 = [int(round(2 * r)) for r in ranks]           # doubled ranks are integers
    w_plus_obs = sum(r for r, d in zip(r2, nz) if d > 0)
    dist = Counter({0: 1})
    for r in r2:
        nxt = Counter()
        for s, c in dist.items():
            nxt[s] += c
            nxt[s + r] += c
        dist = nxt
    total = 2 ** n
    lo = sum(c for s, c in dist.items() if s <= w_plus_obs)
    hi = sum(c for s, c in dist.items() if s >= w_plus_obs)
    return min(Fraction(1), 2 * Fraction(min(lo, hi), total))


def exact_mcnemar(b01: int, b10: int) -> float:
    n = b01 + b10
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b01, b10) + 1)) / 2 ** n) if n else 1.0


def bootstrap_mean_ci(values: list[float], n_resamples: int = 10_000, seed: int = 12345) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n_resamples))
    return means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples) - 1]


# ---------------------------------------------------------------- per-seed summaries of the long-running runs
def seed_summaries(experiment: str, seeds: range) -> dict[int, dict]:
    out = {}
    for s in seeds:
        files = sorted(glob.glob(str(RES / "raw" / experiment / f"seed_{s}" / f"{experiment}_seed{s}_summary.json")))
        out[s] = json.loads(Path(files[-1]).read_text(encoding="utf-8"))["summaries"]
    return out


def pooled(summ: dict[int, dict], cond: str) -> tuple[int, int]:
    return (sum(v[cond]["n_dangerous_contradiction_errors"] for v in summ.values()),
            sum(v[cond]["n_contradictory_category_queries"] for v in summ.values()))


def section_mechanism() -> None:
    print("\n== Section IV: the specified mechanism is the code that ran ==")
    from aisos.memory.contradiction import SOURCE_AUTHORITY
    from aisos.memory.record import TrustTier, LIVE_READABLE_FLOOR
    from aisos.memory import promotion
    from aisos.security.authority_registry import UNVERIFIED_AUTHORITY
    from aisos.memory.record import MemoryRecord
    import inspect

    check("M1", SOURCE_AUTHORITY == {"user_correction": 1.0, "user": 0.8, "tool": 0.6, "consolidation": 0.5,
                                      "dream": 0.1}, f"source authorities {SOURCE_AUTHORITY}")
    check("M2", "SOURCE_AUTHORITY.get(r.source, 0.4)" in inspect.getsource(promotion._authority_score),
          "unlisted source gets authority 0.4")
    check("M3", UNVERIFIED_AUTHORITY == 0.0, "missing/forged/replayed credential gives authority 0.0")
    check("M4", [t.name for t in TrustTier] == ["REJECTED", "T0_SYNTHETIC", "T1_CANDIDATE", "T2_VALIDATED",
                                                  "T3_CORROBORATED", "T4_VERIFIED"]
          and LIVE_READABLE_FLOOR == TrustTier.T4_VERIFIED, "tiers T0..T4, only T4 is readable")
    rec = MemoryRecord.__new__(MemoryRecord)
    rec.source, rec.trust_tier, rec.updated_at = "tool", TrustTier.T1_CANDIDATE, 7.0
    check("M5", promotion._authority_score(rec) == (0.6, 1.0, 7.0)
          and promotion.authority_score_without_authority(rec) == (1.0, 7.0)
          and promotion.authority_score_recency_only(rec) == (7.0,),
          "kappa = (authority, tier, time); -Authority = (tier, time); pure recency = (time)")
    sig = inspect.signature(promotion.promotion_pipeline).parameters
    check("M6", sig["confidence_threshold"].default == 0.55 and sig["max_rounds"].default == 4
          and sig["max_promotions_per_cycle"].default == 25,
          "Algorithm 1: confidence 0.55, reject after 4 rounds, at most 25 promotions to T4 per night")
    from aisos.agent.condition_backends import PeriodicAISOSBackend
    b = PeriodicAISOSBackend()
    check("M7", b.period == 15, "sleep every 15 interactions")
    src = inspect.getsource(PeriodicAISOSBackend.__mro__[1].__init__)
    check("M8", "dream_seeds: int = 4" in src, "Dream rehearsal on by default (dream_seeds=4)")


def section_confirmatory() -> None:
    print("\n== Section VI-A, Table II: confirmatory experiment ==")
    s5 = seed_summaries("e8_confirmatory", range(5))
    s30 = seed_summaries("e8_confirmatory", range(30))
    for cond, want in (("periodic_aisos", 0.0), ("no_sleep_naive", 1.0), ("continuous_mvcc", 1.0)):
        rates = [s5[s][cond]["dangerous_contradiction_error_rate"] for s in s5]
        check("C1", all(r == want for r in rates), f"5 seeds, {cond}: dangerous rate {want} in every seed {rates}")
    st5 = [s5[s]["periodic_aisos"]["stale_rate"] for s in s5]
    check("C2", all(0.01 <= x <= 0.03 for x in st5), f"5 seeds: staleness cost 1-3 pp {[round(x, 4) for x in st5]}")
    d5 = [s5[s]["periodic_aisos"]["dangerous_contradiction_error_rate"]
          - s5[s]["no_sleep_naive"]["dangerous_contradiction_error_rate"] for s in s5]
    p5 = exact_wilcoxon_two_sided(d5)
    stored = load("e8_confirmatory_statistics.json")["primary_comparisons"][
        "periodic_vs_no_sleep__dangerous_contradiction_error_rate"]["wilcoxon"]["p_value"]
    check("C3", p5 == Fraction(1, 16) and stored == 0.0625, f"5 seeds: exact Wilcoxon p = {float(p5)} (stored {stored})")
    k, n = pooled(s30, "periodic_aisos")
    kn, nn = pooled(s30, "no_sleep_naive")
    km, nm = pooled(s30, "continuous_mvcc")
    check("C4", (k, n, kn, nn, km, nm) == (0, 871, 871, 871, 871, 871),
          f"30 seeds: governed {k}/{n}, naive {kn}/{nn}, MVCC {km}/{nm} dangerous")
    for base in ("no_sleep_naive", "continuous_mvcc"):
        d30 = [s30[s]["periodic_aisos"]["dangerous_contradiction_error_rate"]
               - s30[s][base]["dangerous_contradiction_error_rate"] for s in s30]
        p30 = exact_wilcoxon_two_sided(d30)
        check("C5", p30 == Fraction(1, 2 ** 29), f"30 seeds vs {base}: exact Wilcoxon p = 2^-29 = {float(p30):.3e}")
    st30 = [s30[s]["periodic_aisos"]["stale_rate"] for s in s30]
    lo, hi = bootstrap_mean_ci(st30)
    m = statistics.fmean(st30)
    check("C6", round(100 * m, 1) == 2.0 and round(100 * lo, 1) == 1.9 and round(100 * hi, 1) == 2.1,
          f"30 seeds: staleness {100 * m:.2f} pp, 95% bootstrap CI {100 * lo:.2f}-{100 * hi:.2f}")
    check("C7", all(s30[s]["periodic_aisos"]["model_calls"] > 0 for s in s30),
          "the confirmatory runs used a real model (model_calls > 0 in every seed)")


def section_ablation() -> None:
    print("\n== Section VI-B, Table III, Fig. 1: causal ablation ==")
    ab = seed_summaries("e8_ablation", range(5))
    k, n = pooled(ab, "periodic_aisos_without_authority")
    st_full = statistics.fmean(ab[s]["periodic_aisos"]["stale_rate"] for s in ab)
    st_noauth = statistics.fmean(ab[s]["periodic_aisos_without_authority"]["stale_rate"] for s in ab)
    check("A1", k == 0 and n == 139, f"-Authority: {k}/{n} dangerous over 5 seeds (0.0 on this pattern)")
    check("A2", st_noauth > 10 * st_full,
          f"-Authority blocks legitimate updates: staleness {100 * st_noauth:.1f}% vs {100 * st_full:.1f}% with authority")
    me = seed_summaries("e8_mechanism", range(5))
    k, n = pooled(me, "periodic_aisos_without_trust_tier")
    check("A3", k == 0 and n == 139, f"-Authority -Trust (pure recency, Dream on): {k}/{n} dangerous over 5 seeds")
    dc = load("hsrf_w3_dream_confound_control.json")["W3_malicious_contradiction"]
    check("A4", len(dc) == 5, f"Dream-confound isolation: {len(dc)} deterministic conditions {sorted(dc)}")
    a = dc["A_recency_only"]
    check("A5", (not a["dream_seeds_4_default"]["attack_success"]) and a["dream_seeds_0_control"]["attack_success"]
          and a["protection_is_dream_noise_confound"],
          "pure recency: attack blocked with Dream on, succeeds with Dream off (rate 1.0) -> confound")
    others = [c for c in dc if c != "A_recency_only"]
    check("A6", all(dc[c]["genuine_comparator_driven_protection"] for c in others),
          f"conditions with authority or trust tier block the attack with Dream off too {others}")
    check("A7", not dc["B_trust_only"]["dream_seeds_0_control"]["attack_success"],
          "trust-tier only: 0.0 on the attack (deadlock on correction: test_hsrf_trust_only_incumbency_deadlock)")


def section_factorial() -> dict:
    print("\n== Section VI-B, 'Across workload families': authority x Dream factorial, 9 families ==")
    f = load("authority_dream_factorial_9families.json")
    p = f["pooled"]
    from aisos.sim.workload_families import FAMILIES
    check("F1", f["manifest"]["model_calls"] == 0 and len(f["manifest"]["families"]) == 9
          and all(v["runs"] == 45 and v["contradictory_queries"] == 832 for v in p.values()),
          "zero model calls, 9 families x 5 seeds = 45 runs, 832 contradiction questions per condition")
    same = lambda x, y: {k: v for k, v in p[x].items()} == {k: v for k, v in p[y].items()}
    check("F2", same("authority_on_dream_on", "authority_on_dream_off")
          and same("authority_off_dream_on", "authority_off_dream_off"),
          "Dream has no effect once trust tier is present (on/off cells identical)")
    check("F3", p["authority_off_dream_on"]["dangerous_rate"] == 0.0
          and round(100 * p["authority_off_dream_on"]["mean_changing_accuracy"], 1) == 40.8
          and round(100 * p["authority_on_dream_on"]["mean_changing_accuracy"], 1) == 89.5,
          f"removing authority: 0.00 dangerous, corrections correct "
          f"{100 * p['authority_off_dream_on']['mean_changing_accuracy']:.1f}% vs "
          f"{100 * p['authority_on_dream_on']['mean_changing_accuracy']:.1f}% with authority")
    check("F4", round(p["recency_only_dream_on"]["dangerous_rate"], 2) == 0.14
          and round(p["recency_only_dream_off"]["dangerous_rate"], 2) == 0.81,
          f"pure recency: {p['recency_only_dream_on']['dangerous_rate']:.3f} with Dream on, "
          f"{p['recency_only_dream_off']['dangerous_rate']:.3f} with Dream off")
    fam = Counter(r["family"] for r in f["per_run"] if r["cell"] == "authority_on_dream_on" and r["n_dangerous"] > 0)
    spoof_fams = {k for k, v in FAMILIES.items() if v.spoofed_fraction > 0}
    check("F5", round(p["authority_on_dream_on"]["dangerous_rate"], 2) == 0.06 and set(fam) == spoof_fams
          and len(spoof_fams) == 2,
          f"full mechanism {p['authority_on_dream_on']['dangerous_rate']:.4f}, only from the spoofing families "
          f"{sorted(fam)} (families with spoofed claims: {sorted(spoof_fams)})")
    check("F6", p["periodic_aisos_trusted_authority"]["dangerous_rate"] == 0.0,
          "authenticated credentials: 0.00 dangerous")
    return f


def section_benchmark() -> None:
    print("\n== Section VI-C, Table IV: independent adversarial benchmark ==")
    from aisos.benchmark.independent_adversarial import attacks as A, corpus as C
    docs = C.build_independent_corpus().documents
    check("B1", (len(docs), sum(d.hand_authored for d in docs)) == (70, 28),
          f"corpus: {len(docs)} documents, {sum(d.hand_authored for d in docs)} hand-written, "
          f"{sum(not d.hand_authored for d in docs)} template-generated (each flagged hand_authored=False)")
    scen = [f() for f in A.ALL_BASIC_ATTACKS] + [f() for f in A.ADAPTIVE_ATTACKS] + A.compositional_scenarios()
    types = {s.attack_id.split("-")[0].split("+")[0] for s in scen}
    check("B2", len(scen) == 27 and types == {f"A{i}" for i in range(1, 19)},
          f"{len(scen)} scenarios covering {len(types)} attack types A1-A18")
    want = {"llama3_latest": {"S1_no_sleep_naive": 21, "S5_continuous_mvcc": 21, "S2_rag_standard": 19,
                              "S6_aisos": 14, "S6b_aisos_authenticated": 4},
            "qwen3_8b": {"S1_no_sleep_naive": 21, "S5_continuous_mvcc": 21, "S2_rag_standard": 10,
                         "S6_aisos": 14, "S6b_aisos_authenticated": 4}}
    residual = {}
    for model, exp in want.items():
        d = load(f"independent_adversarial_benchmark_{model}.json")
        by: dict[str, list] = {}
        for r in d["runs"]:
            by.setdefault(r["system"], []).append(r)
        bad = lambda r: bool(r["eval"]["dangerous_contradiction"])
        got = {s: sum(bad(r) for r in by[s]) for s in exp}
        n = {s: len(by[s]) for s in exp}
        check("B3", got == exp and set(n.values()) == {27} and not any("error" in r for r in d["runs"]),
              f"{model}: dangerous rates " + ", ".join(f"{s.split('_', 1)[1]} {got[s] / 27:.2f}" for s in exp))
        a, b = by["S1_no_sleep_naive"], by["S6b_aisos_authenticated"]
        assert [r["attack_id"] for r in a] == [r["attack_id"] for r in b]
        b01 = sum(bad(x) and not bad(y) for x, y in zip(a, b))
        b10 = sum(not bad(x) and bad(y) for x, y in zip(a, b))
        pm = exact_mcnemar(b01, b10)
        check("B4", round(pm, 6) == 0.000076 and pm < 0.001,
              f"{model}: exact McNemar ungoverned vs authenticated, discordant {b01}/{b10}, p = {pm:.2e}")
        residual[model] = sorted(r["attack_id"] for r in b if bad(r))
    desc = {s.attack_id: s.description for s in scen}
    res = residual["llama3_latest"]
    check("B5", res == residual["qwen3_8b"] and len(res) == 4,
          f"the same 4 residual scenarios with both models {res}; 23 of 27 answered safely")
    for aid in res:
        print(f"        {aid}: {desc[aid]}")
    print("        (A5's wrong content comes from a genuine higher-authority document)")
    ta = load("trusted_authority_real_model_confirmation.json")
    steps = {s["step"]: s for s in ta["steps"]}
    check("B6", all(s["verdict"] == "CORRECT" for s in ta["steps"])
          and steps["after_spoofed_claim"]["store_values"] == ["GENUINE_INITIAL"]
          and steps["after_genuine_correction"]["store_values"] == ["GENUINE_CORRECTED"],
          "real-model confirmation (llama3): spoofed claim ignored, credentialed correction adopted")


def rerun() -> None:
    print("\n== --rerun: re-running the zero-model-call experiments ==")
    import time
    for name, mod, key in (("hsrf_w3_dream_confound_control.json", "aisos.experiments.hsrf_dream_confound_control", None),
                           ("authority_dream_factorial_9families.json", "aisos.experiments.factorial_generalization_sweep",
                            "pooled")):
        path = RES / name
        stored_bytes = path.read_bytes()
        stored = json.loads(stored_bytes)
        t0 = time.time()
        m = __import__(mod, fromlist=["x"])
        try:
            (m.main if hasattr(m, "main") else m.run)()
            fresh = json.loads(path.read_text(encoding="utf-8"))
        finally:
            path.write_bytes(stored_bytes)           # the stored file is left exactly as it was
        if not key:
            check("R1", stored == fresh, f"{name}: re-run ({time.time() - t0:.0f} s) reproduces the stored result exactly")
            continue
        # Safety numbers must match exactly. Correct/stale answer rates of the rapid-alternation family (W4) can
        # differ slightly between runs: two writes in the same session may receive the same wall-clock timestamp,
        # and the recency tie-break then depends on timing (see CLAIMS.md, "Known differences").
        sr = {(r["family"], r["seed"], r["cell"]): r for r in stored["per_run"]}
        fr = {(r["family"], r["seed"], r["cell"]): r for r in fresh["per_run"]}
        exact = sr.keys() == fr.keys() and all(
            sr[k][f] == fr[k][f] for k in sr for f in ("n_contradictory", "n_dangerous", "workload_hash"))
        check("R2", exact, f"{name}: re-run ({time.time() - t0:.0f} s) reproduces every dangerous-answer count "
                           f"of all {len(sr)} runs exactly")
        moved = sorted({k[:2] for k in sr if sr[k] != fr[k]})
        fp = fresh["pooled"]
        same_text = (round(100 * fp["authority_on_dream_on"]["mean_changing_accuracy"], 1) == 89.5
                     and round(100 * fp["authority_off_dream_on"]["mean_changing_accuracy"], 1) == 40.8)
        check("R3", same_text and all(k[0] == "W4" for k in moved),
              f"{name}: correct-answer rates in the paper (40.8%, 89.5%) also hold for the re-run; "
              f"timing-dependent runs this time: {moved or 'none'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()
    section_mechanism()
    section_confirmatory()
    section_ablation()
    section_factorial()
    section_benchmark()
    if args.rerun:
        rerun()
    n_fail = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - n_fail} of {len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
