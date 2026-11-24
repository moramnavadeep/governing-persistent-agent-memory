"""
E8 -- 1,000-Interaction Long-Running-Agent Pilot
(AISOS_1000_INTERACTION_EXPERIMENT_PROMPT.md).

Named E8 (not "E1") to avoid colliding with the existing symbolic-workload
E1 (aisos/experiments/e1_consistency_longevity.py); both remain valid,
separately reported experiments testing the same underlying hypothesis at
different fidelity (symbolic surrogate vs. a real local LLM).

Runs the same deterministic 1,000-interaction workload (aisos/agent/
workload_1000.py) through all four conditions -- no_sleep_naive,
continuous_mvcc, periodic_aisos, aisos_full -- using the SAME real local
model (Ollama llama3:latest) for every query in every condition, so the
only thing that differs between conditions is the memory-maintenance
backend (aisos/agent/condition_backends.py).

This is explicitly a PRELIMINARY / pilot run: 1 seed, 1,000 interactions.
No claim of statistical significance is made or should be inferred from
its output alone (Section 12, 16, 19).
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import psutil
import requests

from aisos.agent.condition_backends import make_backend
from aisos.agent.evaluator import evaluate
from aisos.agent.model_backend import OllamaBackend
from aisos.agent.workload_1000 import build_workload, save_workload, workload_hash

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "results" / "raw" / "e8_long_running_1000"
CONDITIONS = ["no_sleep_naive", "continuous_mvcc", "periodic_aisos", "aisos_full"]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown (not a git repo checkpoint or git unavailable)"


def _build_prompt(subject: str, predicate: str, query: str, current_value, store_values: list[str]) -> str:
    parts = []
    if len(set(store_values)) > 1:
        parts.append(f"Memory about {subject} {predicate} is currently inconsistent: {sorted(set(store_values))}.")
    elif current_value is not None:
        parts.append(f"Memory: {subject} {predicate} = {current_value}.")
    else:
        parts.append(f"No memory is available about {subject} {predicate}.")
    parts.append(f"Question: {query}")
    parts.append("Answer in one short sentence, using only the memory above. If you don't know, say so.")
    return "\n".join(parts)


def _classify_failure(exc: Exception) -> str:
    """Confirmatory-experiment Section 24: every failure must be
    classified, never silently retried. This is deliberately a
    classification function only -- callers record the classification and
    move on to the next interaction; they do not retry automatically."""
    if isinstance(exc, requests.Timeout):
        return "MODEL_TIMEOUT"
    if isinstance(exc, requests.ConnectionError):
        return "OLLAMA_ERROR"
    if isinstance(exc, requests.RequestException):
        return "MODEL_ERROR"
    if isinstance(exc, (KeyError, ValueError, TypeError)):
        return "SERIALIZATION_ERROR"
    return "UNKNOWN"


def run_condition(condition: str, specs, ledger, model: OllamaBackend, process: psutil.Process) -> dict:
    backend = make_backend(condition)
    records = []
    resource_samples = []
    failures = []
    model_calls = 0
    t_start = time.time()

    for spec in specs:
        if spec.context_reset_before:
            reset_applied = backend.apply_context_reset()
        else:
            reset_applied = False

        if spec.interaction_id % 50 == 0:
            # Sampled on every interaction (teach or query), not gated
            # behind the query branch below -- an earlier version checked
            # this only after a query was processed, and every 50-boundary
            # in this workload happens to land on a teach event (every
            # session starts with at least its noise teaches), so it never
            # fired even once across the whole pilot run. See
            # docs/research_log.md's 2026-08-31 (night) entry.
            resource_samples.append({
                "interaction_id": spec.interaction_id,
                "cpu_percent": process.cpu_percent(interval=None),
                "rss_mb": process.memory_info().rss / (1024 * 1024),
            })

        if spec.kind == "teach":
            backend.write(spec.subject, spec.value, spec.source, spec.confidence, spec.interaction_id)
            backend.maybe_maintain(spec.interaction_id)
            continue

        # kind == "query"
        current_value = backend.read(spec.subject)
        store_values = backend.read_all_values(spec.subject)
        prompt = _build_prompt(spec.subject, spec.predicate, spec.query_text, current_value, store_values)

        t0 = time.time()
        try:
            output = model.generate(prompt, system="You are a careful assistant answering only from provided memory.")
        except Exception as exc:
            failure_kind = _classify_failure(exc)
            failures.append({
                "interaction_id": spec.interaction_id, "failure_kind": failure_kind,
                "error": str(exc), "retried": False,
            })
            print(f"  [{condition}] interaction {spec.interaction_id}: {failure_kind} ({exc}) -- "
                  f"recorded, not retried, continuing")
            backend.maybe_maintain(spec.interaction_id)
            continue
        latency_s = time.time() - t0
        model_calls += 1

        gt_fact = ledger.current_fact(spec.subject, spec.predicate, spec.interaction_id)
        gt_value = gt_fact.value if gt_fact else None
        all_known = [f.value for f in ledger.all_facts(spec.subject, spec.predicate, up_to_interaction=spec.interaction_id)]
        verdict = evaluate(output, gt_value, all_known, store_values)

        metrics = backend.snapshot_metrics()
        backend.maybe_maintain(spec.interaction_id)

        records.append({
            "interaction_id": spec.interaction_id, "session_id": spec.session_id,
            "context_reset_applied": reset_applied, "category": spec.category,
            "subject": spec.subject, "predicate": spec.predicate, "query": spec.query_text,
            "model": model.model_id, "current_value_before_query": current_value,
            "store_values_before_query": store_values, "ground_truth": gt_value,
            "agent_response": output, "evaluation_result": verdict, "latency_s": latency_s,
            "lifecycle_state": metrics.get("lifecycle_state"), "sleep_pressure": metrics.get("sleep_pressure"),
            "cycles_run_so_far": metrics.get("cycles_run"), "store_size": metrics.get("store_size"),
        })

        if spec.interaction_id % 50 == 0:
            print(f"  [{condition}] interaction {spec.interaction_id}/1000, "
                  f"verdict={verdict}, latency={latency_s:.2f}s")

    wall_time_s = time.time() - t_start
    return {
        "condition": condition,
        "records": records,
        "resource_samples": resource_samples,
        "failures": failures,
        "lifecycle_trace": backend.lifecycle_trace(),
        "scheduler_trace": backend.scheduler_trace(),
        "maintenance_events": [
            {"interaction_id": e.interaction_id, "kind": e.kind, "detail": e.detail}
            for e in backend.maintenance_events()
        ],
        "final_metrics": backend.snapshot_metrics(),
        "model_calls": model_calls,
        "wall_time_s": wall_time_s,
    }


def _percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


def summarize_condition(result: dict) -> dict:
    """Rate definitions fixed here BEFORE the confirmatory experiment runs
    (confirmatory-prompt Section 15/32: "do not redefine after observing
    results"). All six general rates share one denominator (len(records),
    i.e. all queries including UNDETERMINED) so they sum to 1.0 exactly --
    a direct executable check of "sum = total interactions"
    (adversarial-audit Section 10).

    dangerous_contradiction_error_rate is defined narrowly and specifically
    (not "any incorrect answer"): the fraction of *contradictory-category*
    queries answered with a losing, lower-authority value instead of the
    established one. This is the precise, pre-registered operationalization
    of "dangerous error" this project's own prior audit already
    identified: in the completed pilot, 100% of no_sleep_naive/
    continuous_mvcc's INCORRECT verdicts were contradictory-category
    (docs/experimental_validity_audit.md). safe_abstention_rate (UNKNOWN)
    and stale_rate are reported separately, never folded into the same
    number as dangerous_contradiction_error_rate, per Section 20's
    explicit separation requirement.
    """
    records = result["records"]
    evaluated = [r for r in records if r["evaluation_result"] != "UNDETERMINED"]
    verdict_counts = {}
    for r in records:
        verdict_counts[r["evaluation_result"]] = verdict_counts.get(r["evaluation_result"], 0) + 1
    latencies = [r["latency_s"] for r in records]
    n = len(records)
    n_evaluated = len(evaluated)

    contradictory_records = [r for r in records if r["category"] == "contradictory"]
    n_contra = len(contradictory_records)
    dangerous_errors = sum(1 for r in contradictory_records if r["evaluation_result"] == "INCORRECT")

    return {
        "n_queries": n,
        "n_evaluated": n_evaluated,
        "verdict_counts": verdict_counts,
        "correct_rate": (verdict_counts.get("CORRECT", 0) / n) if n else None,
        "incorrect_rate": (verdict_counts.get("INCORRECT", 0) / n) if n else None,
        "stale_rate": (verdict_counts.get("STALE", 0) / n) if n else None,
        "contradictory_rate": (verdict_counts.get("CONTRADICTORY", 0) / n) if n else None,
        "unknown_rate": (verdict_counts.get("UNKNOWN", 0) / n) if n else None,
        "undetermined_rate": (verdict_counts.get("UNDETERMINED", 0) / n) if n else None,
        "retrieval_accuracy": (verdict_counts.get("CORRECT", 0) / n_evaluated) if n_evaluated else None,
        "safe_abstention_rate": (verdict_counts.get("UNKNOWN", 0) / n) if n else None,
        "dangerous_contradiction_error_rate": (dangerous_errors / n_contra) if n_contra else None,
        "n_contradictory_category_queries": n_contra,
        "n_dangerous_contradiction_errors": dangerous_errors,
        "contradiction_rate": (verdict_counts.get("CONTRADICTORY", 0) / n) if n else None,  # kept for backward compat with pilot summary readers
        "task_success_rate": (verdict_counts.get("CORRECT", 0) / n) if n else None,  # kept for backward compat
        "latency_p50_s": _percentile(latencies, 0.5),
        "latency_p95_s": _percentile(latencies, 0.95),
        "latency_p99_s": _percentile(latencies, 0.99),
        "cycles_run": result["final_metrics"].get("cycles_run"),
        "rollback_count": result["final_metrics"].get("rollback_count"),
        "final_store_size": result["final_metrics"].get("store_size"),
        "model_calls": result["model_calls"],
        "n_failures": len(result.get("failures", [])),
        "wall_time_s": result["wall_time_s"],
    }


def _ollama_version() -> str:
    try:
        out = subprocess.check_output(["ollama", "--version"], stderr=subprocess.STDOUT).decode().strip()
        return out
    except Exception as e:
        return f"unknown ({e})"


def _dependency_versions() -> dict:
    import matplotlib
    import yaml
    return {
        "requests": requests.__version__, "psutil": psutil.__version__,
        "matplotlib": matplotlib.__version__, "PyYAML": yaml.__version__,
    }


def main(
    seed: int = 0,
    conditions: list[str] | None = None,
    run_id_suffix: str = "",
    max_interactions: int | None = None,
    out_dir: Path | None = None,
    experiment_id: str = "e8_long_running_1000",
):
    """conditions: subset of CONDITIONS to run (default: all four). Used to
    re-run a single condition after a harness fix without re-running the
    other three, which are unaffected and already saved -- see
    docs/research_log.md's 2026-08-31 (night) entry for why aisos_full
    specifically was re-run this way.

    max_interactions: truncates the workload to its first N interactions
    (confirmatory-experiment Section 27's 25-interaction dry run). Does
    NOT change how the workload is generated -- it is the same
    deterministic 1,000-interaction sequence, just sliced, so the dry run
    genuinely exercises the real workload's first sessions rather than a
    separately-generated toy one.

    out_dir/experiment_id: let a caller (e.g. the multi-seed confirmatory
    orchestrator) redirect output away from the pilot's RAW_DIR without
    touching or overwriting the pilot's already-saved results.
    """
    conditions = conditions or CONDITIONS
    raw_dir = out_dir or RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"seed{seed}_{int(time.time())}{run_id_suffix}"

    specs, ledger = build_workload(seed=seed)
    if max_interactions is not None:
        specs = specs[:max_interactions]
    save_workload(specs, raw_dir / f"workload_seed{seed}.json")

    model = OllamaBackend(model="llama3:latest")
    try:
        probe = model.generate("Reply with exactly one word: OK")
    except Exception as e:
        print(f"Ollama not reachable ({e}). Aborting -- no fabricated results.")
        sys.exit(1)
    print(f"Ollama reachable ({probe!r}). Running {len(specs)} interactions x {len(conditions)} condition(s)...")

    wl_hash = workload_hash(specs)
    process = psutil.Process()
    all_summaries = {}
    for condition in conditions:
        print(f"\n=== Condition: {condition} ===")
        # Every condition iterates this exact same `specs` list -- hashed
        # once before the loop and asserted unchanged after, so a bug that
        # let a condition mutate the shared workload in place (e.g. an
        # in-place sort or a stray .pop()) would fail loudly here instead
        # of silently producing an unfair comparison (adversarial-audit
        # Section 6/14).
        assert workload_hash(specs) == wl_hash, f"workload was mutated before condition {condition!r} ran"
        result = run_condition(condition, specs, ledger, model, process)
        assert workload_hash(specs) == wl_hash, f"condition {condition!r} mutated the shared workload"
        out_path = raw_dir / f"{condition}_{run_id}.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"  wrote {out_path}")
        summary = summarize_condition(result)
        all_summaries[condition] = summary
        print(f"  summary: {summary}")

    manifest = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "timestamp": time.time(),
        "git_commit": _git_commit(),
        "workload_hash": wl_hash,
        "model": model.model_id,
        "model_version": "llama3:latest (Ollama local)",
        "model_parameters": {"temperature": model.temperature, "host": model.host, "timeout_s": model.timeout_s},
        "ollama_version": _ollama_version(),
        "dependency_versions": _dependency_versions(),
        "workload_version": f"workload_1000.py seed={seed}" + (f" (truncated to {max_interactions})" if max_interactions else ""),
        "seed": seed,
        "conditions": conditions,
        "interaction_count": len(specs),
        "session_count": len({s.session_id for s in specs}),
        "reset_count": sum(1 for s in specs if s.context_reset_before),
        "python_version": sys.version,
        "platform": platform.platform(),
        "status": "PRELIMINARY -- see docs/reproducibility.md",
        "output_paths": [str(raw_dir / f"{c}_{run_id}.json") for c in conditions],
    }
    (raw_dir / f"manifest_{run_id}.json").write_text(json.dumps(manifest, indent=2))

    # Merge into the aggregate summary rather than overwriting it, so a
    # partial re-run (e.g. one corrected condition) does not silently
    # destroy the other, unaffected conditions' already-saved results.
    # Keyed by experiment_id/seed so the confirmatory multi-seed orchestrator
    # (which calls main() once per seed with its own out_dir) never collides
    # with the pilot's summary file or with another seed's same-named
    # conditions in one flat dict.
    summary_path = (out_dir or (ROOT / "results")) / f"{experiment_id}_seed{seed}_summary.json" if out_dir \
        else ROOT / "results" / f"{experiment_id}_summary.json"
    existing = {"manifests": [], "summaries": {}}
    if summary_path.exists():
        loaded = json.loads(summary_path.read_text())
        if "manifest" in loaded and "manifests" not in loaded:
            loaded = {"manifests": [loaded["manifest"]], "summaries": loaded["summaries"]}
        existing.update(loaded)
    existing["manifests"].append(manifest)
    existing["summaries"].update(all_summaries)
    summary_path.write_text(json.dumps(existing, indent=2, default=str))
    print(f"\nCondition(s) {conditions} complete. Aggregate summary -> {summary_path}")
    print(json.dumps(all_summaries, indent=2, default=str))
    return all_summaries


if __name__ == "__main__":
    main()
