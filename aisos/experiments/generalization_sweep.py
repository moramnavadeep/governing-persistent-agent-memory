"""
Post-Security-Closure Scientific Generalization Milestone
(AISOS_Post_Security_Closure_Generalization_Milestone.md), Phases 3-10.

Design decision (docs/generalization_experiment_plan.md explains this in
full): the PRIMARY generalization campaign here is ZERO-MODEL-CALL, exactly
like every prior mechanism-level claim in this project (the authority
ablation, trust-tier ablation, causal factorial, and authority security
stress test were ALL zero-model-call -- see docs/implementation_traceability.md).
Instead of a real LLM's free-text answer, the backend's own deterministic
retrieval (`backend.read(subject)`) is fed AS the "model_output" into the
existing 6-label evaluator (aisos/agent/evaluator.py). This measures the
MEMORY MECHANISM's own correctness directly, structurally guaranteeing
H-G6 (mechanism attribution) for this campaign by construction rather than
requiring it to be inferred after the fact, and keeps a 5-seed x 9-workload
x 4-condition sweep (up to ~180 runs x ~300-560 interactions each) a
matter of seconds rather than the many hours a real-Ollama campaign at
this scale would cost -- directly satisfying Phase 8's "do not immediately
launch a large multi-model campaign" instruction. Phase 12's small
(~100-300 interaction) REAL-model confirmation is a separate, later script
(aisos/experiments/generalization_real_model_confirmation.py) that checks
a genuine LLM does not contradict this campaign's qualitative conclusions
-- it is the confirmation, not the primary evidence.

Conditions (Phase 7 + a necessary extension): no_sleep_naive, continuous_mvcc,
periodic_aisos (Phase 7's literal three baselines), PLUS
periodic_aisos_trusted_authority -- included because Phase 4/5's spoofing,
replay, and "authenticated vs. merely-claimed authority" items (H, I, J, K
in Phase 4; items 3-4 in Phase 5) are structurally untestable without a
condition that can distinguish an authenticated credential from a bare
claim at all; periodic_aisos alone cannot express that distinction (it
never checks a credential -- see condition_backends.py's own docstring).
aisos_full is excluded, per Phase 7's explicit instruction (its
scheduler-calibration issue was not separately re-resolved this milestone).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from aisos.agent.condition_backends import make_backend
from aisos.agent.evaluator import evaluate
from aisos.sim.workload_families import FAMILIES, FamilyConfig, build_family_workload, workload_hash

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "results" / "raw" / "generalization_sweep"
CONDITIONS = ["no_sleep_naive", "continuous_mvcc", "periodic_aisos", "periodic_aisos_trusted_authority"]
SEEDS = [0, 1, 2, 3, 4]

PHASE1_WORKLOADS = ["W3", "W4", "W9"]  # highest-information: pure-attack, tie-breaking stress, combined stress
PHASE2_WORKLOADS = ["W1", "W2", "W5", "W6", "W7", "W8"]


def _model_output_from_store(current_value) -> str:
    """The memory layer's own retrieval, standing in for a real model's
    answer (see module docstring). None (nothing in the store) is mapped
    to an explicit "no memory available" string, which the evaluator's own
    HEDGE_MARKERS list ('no memory' is one of them) correctly resolves to
    UNKNOWN rather than crashing on None.lower() or being unfairly scored
    INCORRECT for an honest absence of information.
    """
    return current_value if current_value is not None else "no memory available"


def run_family_condition(condition: str, specs, ledger) -> dict:
    backend = make_backend(condition)
    supports_credential = condition == "periodic_aisos_trusted_authority"
    issued_credentials: dict[int, str] = {}
    records = []
    t_start = time.time()

    for spec in specs:
        if spec.context_reset_before:
            backend.apply_context_reset()

        if spec.kind == "teach":
            credential = None
            if supports_credential:
                if spec.credential_action == "issue":
                    credential = backend.issue_credential(spec.source)
                    issued_credentials[spec.interaction_id] = credential
                elif spec.credential_action == "replay_prior":
                    credential = issued_credentials.get(spec.replay_source_interaction_id)
                # "spoof" / "none": credential stays None -- a bare, unauthenticated claim
            if supports_credential:
                backend.write(spec.subject, spec.value, spec.source, spec.confidence, spec.interaction_id, credential=credential)
            else:
                backend.write(spec.subject, spec.value, spec.source, spec.confidence, spec.interaction_id)
            backend.maybe_maintain(spec.interaction_id)
            continue

        # kind == "query"
        current_value = backend.read(spec.subject)
        store_values = backend.read_all_values(spec.subject)
        model_output = _model_output_from_store(current_value)

        gt_fact = ledger.current_fact(spec.subject, spec.predicate, spec.interaction_id)
        gt_value = gt_fact.value if gt_fact else None
        all_known = [f.value for f in ledger.all_facts(spec.subject, spec.predicate, up_to_interaction=spec.interaction_id)]
        verdict = evaluate(model_output, gt_value, all_known, store_values)

        metrics = backend.snapshot_metrics()
        backend.maybe_maintain(spec.interaction_id)

        records.append({
            "interaction_id": spec.interaction_id, "session_id": spec.session_id,
            "context_reset_applied": spec.context_reset_before, "category": spec.category,
            "subject": spec.subject, "current_value": current_value, "store_values": store_values,
            "ground_truth": gt_value, "evaluation_result": verdict,
            "lifecycle_state": metrics.get("lifecycle_state"), "cycles_run_so_far": metrics.get("cycles_run"),
            "store_size": metrics.get("store_size"),
        })

    wall_time_s = time.time() - t_start
    return {
        "condition": condition, "records": records,
        "final_metrics": backend.snapshot_metrics(),
        "wall_time_s": wall_time_s,
        "n_teach": sum(1 for s in specs if s.kind == "teach"),
        "n_query": sum(1 for s in specs if s.kind == "query"),
    }


def _percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


def summarize_condition(result: dict) -> dict:
    """Identical rate definitions to e8_long_running_1000.py::summarize_condition
    (Phase 9: "keep the locked primary metric"), applied here to the
    zero-model-call records produced by run_family_condition."""
    records = result["records"]
    verdict_counts = {}
    for r in records:
        verdict_counts[r["evaluation_result"]] = verdict_counts.get(r["evaluation_result"], 0) + 1
    n = len(records)
    n_evaluated = sum(1 for r in records if r["evaluation_result"] != "UNDETERMINED")

    contradictory_records = [r for r in records if r["category"] == "contradictory"]
    n_contra = len(contradictory_records)
    dangerous_errors = sum(1 for r in contradictory_records if r["evaluation_result"] == "INCORRECT")

    changing_records = [r for r in records if r["category"] == "changing"]

    return {
        "n_queries": n,
        "verdict_counts": verdict_counts,
        "correct_rate": (verdict_counts.get("CORRECT", 0) / n) if n else None,
        "incorrect_rate": (verdict_counts.get("INCORRECT", 0) / n) if n else None,
        "stale_rate": (verdict_counts.get("STALE", 0) / n) if n else None,
        "contradictory_rate": (verdict_counts.get("CONTRADICTORY", 0) / n) if n else None,
        "unknown_rate": (verdict_counts.get("UNKNOWN", 0) / n) if n else None,
        "undetermined_rate": (verdict_counts.get("UNDETERMINED", 0) / n) if n else None,
        "retrieval_accuracy": (verdict_counts.get("CORRECT", 0) / n_evaluated) if n_evaluated else None,
        "dangerous_contradiction_error_rate": (dangerous_errors / n_contra) if n_contra else None,
        "n_contradictory_category_queries": n_contra,
        "n_dangerous_contradiction_errors": dangerous_errors,
        "changing_category_accuracy": (
            sum(1 for r in changing_records if r["evaluation_result"] == "CORRECT") / len(changing_records)
        ) if changing_records else None,
        "cycles_run": result["final_metrics"].get("cycles_run"),
        "rollback_count": result["final_metrics"].get("rollback_count"),
        "final_store_size": result["final_metrics"].get("store_size"),
        "wall_time_s": result["wall_time_s"],
    }


def run_seed_workload(family_name: str, seed: int, conditions=None) -> dict:
    conditions = conditions or CONDITIONS
    config = FAMILIES[family_name]
    specs, ledger, empirical = build_family_workload(config, seed=seed)
    wl_hash = workload_hash(specs)

    summaries = {}
    raw_results = {}
    for condition in conditions:
        assert workload_hash(specs) == wl_hash, f"workload mutated before condition {condition!r}"
        result = run_family_condition(condition, specs, ledger)
        assert workload_hash(specs) == wl_hash, f"condition {condition!r} mutated the shared workload"
        summaries[condition] = summarize_condition(result)
        raw_results[condition] = result

    return {
        "family": family_name, "seed": seed, "workload_hash": wl_hash,
        "empirical_workload_metadata": empirical, "purpose": config.purpose,
        "conditions": conditions, "summaries": summaries, "raw": raw_results,
    }


def main(workloads=None, seeds=None, conditions=None, phase_label="phase1"):
    workloads = workloads or PHASE1_WORKLOADS
    seeds = seeds if seeds is not None else SEEDS
    conditions = conditions or CONDITIONS
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    t0 = time.time()
    for family in workloads:
        for seed in seeds:
            key = f"{family}_seed{seed}"
            run = run_seed_workload(family, seed, conditions)
            all_results[key] = run
            print(f"{key}: " + ", ".join(
                f"{c}={run['summaries'][c]['dangerous_contradiction_error_rate']}"
                for c in conditions
            ))
    elapsed = time.time() - t0

    manifest = {
        "phase_label": phase_label, "workloads": workloads, "seeds": seeds, "conditions": conditions,
        "n_runs": len(workloads) * len(seeds), "wall_time_s": elapsed, "timestamp": time.time(),
        "methodology": "zero-model-call: backend.read() used as model_output through the existing 6-label evaluator",
    }

    out_path = RAW_DIR / f"generalization_sweep_{phase_label}.json"
    # Save raw (minus the bulky per-interaction 'raw' records, kept in a separate per-run file) + summaries.
    summary_only = {
        k: {kk: vv for kk, vv in v.items() if kk != "raw"} for k, v in all_results.items()
    }
    out_path.write_text(json.dumps({"manifest": manifest, "runs": summary_only}, indent=2, default=str))

    raw_dir_detail = RAW_DIR / phase_label
    raw_dir_detail.mkdir(parents=True, exist_ok=True)
    for key, run in all_results.items():
        (raw_dir_detail / f"{key}.json").write_text(json.dumps(run, indent=2, default=str))

    print(f"\n{phase_label}: {len(all_results)} (workload,seed) runs, {elapsed:.1f}s total -> {out_path}")
    return all_results


if __name__ == "__main__":
    main()
