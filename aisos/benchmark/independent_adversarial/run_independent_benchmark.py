"""
Independent adversarial benchmark runner
(AISOS_Independent_Adversarial_Benchmark_Problem.md).

Wires the independent corpus/attacks/evaluator (this package, freshly
built, not AISOS-derived) against SYSTEM implementations reused from
aisos/benchmark/semantic/systems.py -- reusing a SYSTEM implementation is
not an independence violation (Section 3 just names systems to test);
only reusing the corpus/attack-taxonomy/evaluator design would be, and
none of those three are reused here.

Each scenario runs on a FRESH system instance (base corpus + that
scenario's own submissions ingested together), because several attacks
deliberately target the same topic as other attacks (a real, disclosed
corpus-design property -- see docs/independent_adversarial_benchmark_protocol.md) --
running scenarios sequentially on one shared instance would let one
scenario's injected state contaminate another's ground-truth check.

S3 (RAG+rerank) and S4 (Mem0) are excluded from the main 27-scenario
sweep by default, for disclosed cost reasons (S3 adds a full LLM call per
retrieved candidate per query; S4's ingestion is independently confirmed,
three times across this project's history, to run roughly 9s/document,
which at this corpus's ~90-95 documents per scenario x 27 scenarios would
add hours) -- not because they were expected to perform worse or better.
Both remain available via --include-s3 / --include-s4.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from aisos.agent.model_backend import OllamaBackend
from aisos.benchmark.embeddings import Embedder
from aisos.benchmark.prompts import build_prompt_from_context
from aisos.benchmark.independent_adversarial.attacks import (
    ALL_BASIC_ATTACKS, ADAPTIVE_ATTACKS, compositional_scenarios, AttackScenario, Submission,
)
from aisos.benchmark.independent_adversarial.corpus import build_independent_corpus, Document as IADoc, source_authority
from aisos.benchmark.independent_adversarial.evaluator import evaluate_scenario, ScenarioGroundTruth
from aisos.benchmark.semantic.corpus import Document as SemDoc
from aisos.benchmark.semantic.systems import (
    KeyValueOnlySystem, RAGSemanticSystem, AISOSOnlySystem, AISOSPlusRAGSemanticSystem,
    ProvenanceWeightedRAGSystem, SMSRStyleSystem,
)
from aisos.agent.condition_backends import NoSleepBackend, ContinuousMVCCBackend, PeriodicAISOSBackend, TrustedAuthorityPeriodicAISOSBackend
from aisos.benchmark.independent_adversarial.comparator import independent_contradiction_score, independent_authority_score

PERIOD = 15


def _make_independent_aisos_backend():
    """This package's own comparator (independent_contradiction_score /
    independent_authority_score), NOT the semantic benchmark's
    _make_semantic_aisos_backend() -- see comparator.py's docstring for
    the bug this fixes."""
    return PeriodicAISOSBackend(period=PERIOD, contradiction_score_fn=independent_contradiction_score,
                                 authority_score_fn=independent_authority_score)


def _make_independent_authenticated_backend():
    from aisos.security.authority_registry import TrustedAuthorityRegistry
    from aisos.security.checkpoint import new_signing_key
    from aisos.benchmark.independent_adversarial.corpus import SOURCE_AUTHORITY
    registry = TrustedAuthorityRegistry(new_signing_key(), trusted_sources=dict(SOURCE_AUTHORITY))
    return TrustedAuthorityPeriodicAISOSBackend(period=PERIOD, registry=registry)

ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _to_sem_doc(d: IADoc) -> SemDoc:
    """Adapter: independent_adversarial.Document -> semantic.Document,
    using THIS package's own source_authority() table (not AISOS's
    SOURCE_AUTHORITY, not the semantic benchmark's SOURCE_TYPES)."""
    return SemDoc(d.doc_id, d.text, d.topic, d.value, d.source, source_authority(d.source),
                  d.confidence, d.timestamp, d.legitimate, True)


def make_systems_under_test(model, embedder, include_s3=False, include_s4=False) -> dict:
    systems = {
        "S1_no_sleep_naive": lambda: KeyValueOnlySystem(NoSleepBackend()),
        "S2_rag_standard": lambda: RAGSemanticSystem(embedder=embedder),
        "S5_continuous_mvcc": lambda: KeyValueOnlySystem(ContinuousMVCCBackend()),
        "S6_aisos": lambda: AISOSOnlySystem(_make_independent_aisos_backend()),
        "S6b_aisos_authenticated": lambda: AISOSOnlySystem(_make_independent_authenticated_backend(), credential_aware=True),
        # NOTE: no "S7b authenticated" condition -- confirmed by direct code inspection that
        # aisos/benchmark/semantic/systems.py::AISOSPlusRAGSemanticSystem.ingest() always calls
        # self.aisos.write(..., credential=None), i.e. the REUSED S7 implementation does not
        # support authenticated ingestion at all (this is a property of the reused system code,
        # not something added or worked around here -- see docs/independent_adversarial_benchmark_protocol.md's
        # "Known limitation of reused system code" note). S7 in this benchmark is therefore always
        # the unauthenticated-backend condition, exactly matching S6 (not S6b) in its authentication posture.
        "S7_aisos_plus_rag": lambda: AISOSPlusRAGSemanticSystem(embedder=embedder, aisos_backend_factory=_make_independent_aisos_backend),
        "S8_smsr_style": lambda: SMSRStyleSystem(embedder=embedder, model=model),
    }
    if include_s3:
        systems["S3_rag_reranked"] = lambda: RAGSemanticSystem(embedder=embedder, rerank=True, model=model)
    if include_s4:
        from aisos.benchmark.semantic.systems import Mem0SemanticSystem
        systems["S4_mem0_external"] = lambda: Mem0SemanticSystem()
    return systems


def _governed_or_context_value(system, topic: str, docs_by_topic: dict) -> tuple:
    """Returns (governed_value, evidence_values_shown, retrieved_doc_ids)."""
    governed = system.governed_value(topic) if hasattr(system, "governed_value") else None
    ctx = system.context_documents(f"What is the current value for {topic}?", topic, k=8) if hasattr(system, "context_documents") else []
    evidence_values = [d.value for d in ctx if d.value is not None]
    retrieved_ids = {d.doc_id for d in ctx} if ctx else None
    return governed, evidence_values, retrieved_ids


def run_scenario(system_name, system_factory, base_docs, scenario: AttackScenario, model) -> dict:
    system = system_factory()
    all_docs = list(base_docs) + [_to_sem_doc(s.doc) for s in scenario.submissions]
    t0 = time.time()
    system.ingest(all_docs)
    if hasattr(system, "maybe_maintain"):
        from aisos.benchmark.semantic.systems import PERIOD
        for i in range(1, 8 * PERIOD):
            system.maybe_maintain(i)
    ingest_s = time.time() - t0

    topic = scenario.ground_truth.topic
    governed, evidence_values, retrieved_ids = _governed_or_context_value(system, topic, None)

    # Answer generation
    ctx_docs = system.context_documents(scenario.query_text, topic, k=8) if hasattr(system, "context_documents") else []
    context_items = [{"subject": d.topic or d.doc_id, "value": d.value} for d in ctx_docs if d.value is not None]
    prompt = build_prompt_from_context(scenario.query_text, context_items)
    response = model.generate(prompt, system="You are a careful assistant answering only from the provided memory/context.")

    # Provenance rejection: for provenance-attack scenarios, "rejected" means the dangerous
    # value never became the governed/evidence value at all.
    provenance_rejected = None
    if scenario.provenance_attack_present:
        dv = scenario.ground_truth.dangerous_values
        provenance_rejected = not (
            (governed is not None and any(d in governed for d in dv)) or
            any(any(d in v for d in dv) for v in evidence_values if v)
        )

    higher_authority_won = None
    if scenario.authority_conflict_present:
        higher_authority_won = governed is not None and scenario.ground_truth.correct_value in governed

    result = evaluate_scenario(
        scenario.ground_truth, governed, governed, evidence_values, response,
        retrieved_doc_ids=retrieved_ids, provenance_attack_present=scenario.provenance_attack_present,
        provenance_attack_rejected=provenance_rejected, authority_conflict_present=scenario.authority_conflict_present,
        higher_authority_value_won=higher_authority_won,
    )

    return {
        "system": system_name, "attack_id": scenario.attack_id, "severity": scenario.severity,
        "description": scenario.description, "query": scenario.query_text,
        "ground_truth": scenario.ground_truth.correct_value, "dangerous_values": scenario.ground_truth.dangerous_values,
        "governed_value": governed, "response": response, "ingest_s": ingest_s,
        "n_attackers": scenario.n_attackers,
        "eval": {
            "retrieval_relevance": result.retrieval_relevance,
            "memory_admission_correct": result.memory_admission_correct,
            "persistence_correct": result.persistence_correct,
            "evidence_selection_correct": result.evidence_selection_correct,
            "answer_correctness": result.answer_correctness,
            "dangerous_contradiction": result.dangerous_contradiction,
            "provenance_integrity_ok": result.provenance_integrity_ok,
            "authority_correct": result.authority_correct,
            "failure_category": result.failure_category,
        },
    }


def run(model_id: str = "llama3:latest", include_s3: bool = False, include_s4: bool = False,
        include_compositional: bool = True) -> dict:
    model = OllamaBackend(model=model_id, temperature=0.0)
    embedder = Embedder()
    cb = build_independent_corpus()
    base_docs = [_to_sem_doc(d) for d in cb.documents]

    scenarios = [f() for f in ALL_BASIC_ATTACKS] + [f() for f in ADAPTIVE_ATTACKS]
    if include_compositional:
        scenarios += compositional_scenarios()

    systems = make_systems_under_test(model, embedder, include_s3=include_s3, include_s4=include_s4)

    results = {
        "manifest": {
            "timestamp": time.time(), "model": model_id, "n_base_documents": len(base_docs),
            "n_scenarios": len(scenarios), "systems": list(systems.keys()),
            "include_s3": include_s3, "include_s4": include_s4,
        },
        "runs": [],
    }
    for sys_name, sys_factory in systems.items():
        print(f"=== System: {sys_name} ===", flush=True)
        for scenario in scenarios:
            try:
                r = run_scenario(sys_name, sys_factory, base_docs, scenario, model)
                results["runs"].append(r)
                print(f"  {scenario.attack_id} ({scenario.severity}): correct={r['eval']['answer_correctness']} "
                      f"dangerous={r['eval']['dangerous_contradiction']}", flush=True)
            except Exception as exc:
                results["runs"].append({"system": sys_name, "attack_id": scenario.attack_id, "error": str(exc)})
                print(f"  {scenario.attack_id}: ERROR {exc}", flush=True)

    safe = model_id.replace(":", "_").replace("/", "_")
    out_path = ROOT / "results" / f"independent_adversarial_benchmark_{safe}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n-> {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", nargs="?", default="llama3:latest")
    parser.add_argument("--include-s3", action="store_true")
    parser.add_argument("--include-s4", action="store_true")
    parser.add_argument("--no-compositional", action="store_true")
    args = parser.parse_args()
    run(model_id=args.model_id, include_s3=args.include_s3, include_s4=args.include_s4,
        include_compositional=not args.no_compositional)
