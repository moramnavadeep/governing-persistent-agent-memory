# Claim-by-claim evidence map

For each statement in the paper *Governing Persistent Agent Memory: A Causal Study of Authority, Trust, and
Recency* (AIEI 2027, paper 1223), this table gives:

- the code that implements or produced it;
- the stored result in `results/` that backs it;
- the check in `verify_claims.py` or the test in `aisos/tests/` that confirms it.

Run all checks:

```
python verify_claims.py            # 39 checks on the stored results and the code constants
python verify_claims.py --rerun    # + re-runs the zero-model-call experiments (about 2 min)
python -m pytest aisos/tests       # 83 deterministic tests
```

"Model calls" says whether reproducing the claim needs a language model. No means it is deterministic, runs in
seconds to minutes, and needs nothing beyond `requirements.txt`. Yes means it needs [Ollama](https://ollama.com)
with `llama3:latest` (and `qwen3:8b` for the second-model rows).

## Section IV: governance mechanism

| Paper statement | Implementation | Check |
|---|---|---|
| Source authorities: user correction 1.0, user 0.8, tool 0.6, consolidation 0.5, dream 0.1; unlisted 0.4 | `aisos/memory/contradiction.py` (`SOURCE_AUTHORITY`), `aisos/memory/promotion.py` (`_authority_score`) | M1, M2 |
| Authenticated authority comes from an HMAC-signed, single-use credential; missing, forged or replayed gives 0.0 | `aisos/security/authority_registry.py` (`TrustedAuthorityRegistry`, `UNVERIFIED_AUTHORITY`), `aisos/security/durable_nonce_ledger.py` | M3; `aisos/tests/test_trusted_authority.py` |
| Trust tiers T0–T4; only T4 is readable | `aisos/memory/record.py` (`TrustTier`, `LIVE_READABLE_FLOOR`) | M4 |
| κ(r) = (authority, tier, timestamp); −Authority uses (tier, timestamp); pure recency uses (timestamp) | `aisos/memory/promotion.py` (`_authority_score`, `authority_score_without_authority`, `authority_score_recency_only`); backends in `aisos/agent/condition_backends.py` | M5; `aisos/tests/test_authority_ablation.py`, `aisos/tests/test_trust_tier_ablation.py` |
| Lifecycle: staged at T1, sleep every 15 interactions, light / deep / Dream / wake-verify, atomic publish or rollback | `aisos/agent/condition_backends.py` (`PeriodicAISOSBackend`), `aisos/memory/light_sleep.py`, `aisos/memory/deep_sleep.py`, `aisos/dream/dream_phase.py`, `aisos/security/wake_verify.py`, `aisos/security/wal.py` | M7, M8 |
| Algorithm 1: conflicting-value check, confidence 0.55, one tier per night, cap 25 promotions to T4, reject after 4 rounds | `aisos/memory/promotion.py` (`promotion_pipeline`, `_hold_or_reject`) | M6 |
| Quarantine and Abstain are not used in any experiment | The ATC gateway (`aisos/atc/`) is not part of this repository; no experiment here imports it | (by construction) |

## Section VI-A and Table II: confirmatory experiment (model calls: yes, llama3)

| Paper statement | Code | Result files | Check |
|---|---|---|---|
| 5 seeds: governed dangerous rate 0.0 in every seed; naive and MVCC 1.0 | `aisos/experiments/e8_confirmatory.py` → `e8_long_running_1000.py`, workload `aisos/agent/workload_1000.py`, evaluator `aisos/agent/evaluator.py` | `results/raw/e8_confirmatory/seed_0..4/` | C1 |
| Staleness cost 1–3 pp | same | same (`stale_rate`) | C2 |
| Exact Wilcoxon p = 0.0625, not significant on its own | `aisos/experiments/e8_confirmatory_stats.py` | `results/e8_confirmatory_statistics.json` | C3 (recomputed exactly) |
| 30 seeds: 0 of 871 dangerous for governed, 871 of 871 for each baseline | `aisos/experiments/e8_confirmatory_extend_seeds.py` | `results/raw/e8_confirmatory/seed_0..29/` | C4 |
| Exact two-sided Wilcoxon p = 2⁻²⁹ ≈ 1.9 × 10⁻⁹ | `verify_claims.py` (`exact_wilcoxon_two_sided`, exact for any n) | as above | C5 |
| Staleness cost 2.0 pp, 95% bootstrap CI 1.9–2.1 | `verify_claims.py` (`bootstrap_mean_ci`, 10,000 resamples, seed 12345) | as above | C6 |

Every seed directory holds the workload (`workload_seed*.json`), the run manifest, and one file per condition.
Each condition file records every prompt, model response and verdict.

`results/e8_confirmatory_statistics_extended_n30.json` reports p from the normal approximation (4.6 × 10⁻⁸),
which was the method of the earlier analysis script. The paper reports the exact value, computed by check C5.

## Section VI-B, Table III and Fig. 1: causal ablation

| Paper statement | Code | Result files | Model calls | Check |
|---|---|---|---|---|
| −Authority: 0.0 on this pattern | `aisos/experiments/e8_ablation.py`, `e8_ablation_stats.py` | `results/raw/e8_ablation/`, `results/e8_ablation_statistics.json` | yes | A1 |
| −Authority is needed for legitimate updates (staleness 25.1% vs 1.8%) | same | same | yes | A2 |
| −Authority −Trust (pure recency): 0.0 while Dream is on | `aisos/experiments/e8_mechanism.py`, `e8_mechanism_stats.py` | `results/raw/e8_mechanism/`, `results/e8_mechanism_statistics.json` | yes | A3 |
| Dream-confound isolation (5 deterministic conditions); recency alone gives 1.0 with Dream off | `aisos/experiments/hsrf_dream_confound_control.py` | `results/hsrf_w3_dream_confound_control.json` | no | A4, A5, A6, `--rerun` |
| Trust-tier only: 0.0 on the attack, deadlock on correction | `aisos/memory/promotion.py` (`authority_score_trust_only`) | `results/hsrf_w3_dream_confound_control.json` | no | A7; `aisos/tests/test_hsrf_trust_only_incumbency_deadlock.py` |
| Exact 2 × 2 authority × Dream factorial on one constructed attack | `aisos/tests/test_authority_dream_factorial.py` | (asserted in the test) | no | 7 tests |
| Fig. 1 bars 0, 0, 0, 1.0 | `paper/make_fig1.py` | values from A1, A3, A5 | no | — |

## Section VI-B, "Across workload families" (model calls: no)

| Paper statement | Code | Result file | Check |
|---|---|---|---|
| 9 families, 5 seeds each, 45 runs, 832 contradiction questions per condition, zero model calls | `aisos/experiments/factorial_generalization_sweep.py`, families in `aisos/sim/workload_families.py`, runner in `aisos/experiments/generalization_sweep.py` | `results/authority_dream_factorial_9families.json` | F1, `--rerun` |
| Dream has no effect once trust tier is present | same | same | F2 |
| Removing authority: 0.00 dangerous, 40.8% of corrections correct vs 89.5% with authority | same | same | F3 |
| Pure recency 0.14 with Dream on, 0.81 with Dream off | same | same | F4 |
| Full mechanism 0.06, only from the two families with spoofed claims (W3, W9) | same | same | F5 |
| Authenticated credentials: 0.00 | same | same | F6 |

## Section VI-C and Table IV: authentication boundary

| Paper statement | Code | Result file | Model calls | Check |
|---|---|---|---|---|
| 12-test authority-security stress test | `aisos/tests/test_authority_security_stress.py` | — | no | 12 tests |
| Authenticated credentials close the gap: 16 tests | `aisos/tests/test_trusted_authority.py` | — | no | 16 tests |
| … plus a real-model confirmation | `aisos/experiments/trusted_authority_real_model_confirmation.py` | `results/trusted_authority_real_model_confirmation.json` | yes | B6 |
| 70-document corpus, 28 hand-written, 42 template-generated and labelled | `aisos/benchmark/independent_adversarial/corpus.py` | — | no | B1 |
| 27 scenarios, 18 attack types | `aisos/benchmark/independent_adversarial/attacks.py` | — | no | B2 |
| llama3: ungoverned 0.78, RAG 0.70, unauthenticated 0.52, authenticated 0.15 | `aisos/benchmark/independent_adversarial/run_independent_benchmark.py`, systems in `aisos/benchmark/semantic/systems.py`, evaluator in `aisos/benchmark/independent_adversarial/evaluator.py` | `results/independent_adversarial_benchmark_llama3_latest.json` | yes | B3 |
| qwen3: identical governed rates; RAG 0.37 | `scripts/run_independent_benchmark_qwen3.py` | `results/independent_adversarial_benchmark_qwen3_8b.json` | yes | B3 |
| p < 0.001; exact McNemar p = 7.6 × 10⁻⁵ | `verify_claims.py` (`exact_mcnemar`) | both files | — | B4 |
| All 4 residual failures are authenticated-but-false (A3, A5, A13, A3+A8), the same with both models; 23 of 27 safe | as above | both files | — | B5 |

The 27 scenarios are 21 basic and adaptive scenarios plus 6 compositional ones. Two compositional scenarios
reuse the ids `A12` and `A18`, so those ids appear twice in the result files.

## Known differences between the stored results and this code

- **Read fallback.** Before the run of seeds 5–29, one change was made to how a subject with nothing published
  yet is answered: the κ-best staged candidate replaced the most recent staged write. This repository contains
  the code after that change. The stored files for seeds 0–4 and the 5-seed ablations were produced before it.
  Re-running those seeds with this code can therefore give slightly different correct and stale counts. The
  governed dangerous-answer rate is 0.0 under both versions, as the paper states.
- **Timestamp ties in one workload family.** Records are stamped with the wall clock (`time.time()`). In the
  rapid-alternation family W4, two writes in one session can get the same timestamp, and the recency
  tie-break then depends on timing. When the 9-family factorial is re-run, the correct and stale answer rates
  of a W4 run can therefore differ slightly; for W4, seed 2, the result was 89.2% or 91.4% across repeated runs
  of the same code. Dangerous-answer counts are identical in every run, and the pooled rates in the paper
  (40.8%, 89.5%) hold either way. `verify_claims.py --rerun` checks all three facts (R2, R3). Use Python 3.13
  or Linux: on Windows with Python 3.12 or older, `time.time()` advances only every 15.6 ms, which makes such
  ties far more common.
- **Absolute paths.** Paths in the stored run manifests were absolute paths on the authors' machine. They have
  been made relative to the repository root; no other value in any result file was changed.
