"""
System adapters for the semantic/enterprise RAG-favorable benchmark
(AISOS_v3_Next_Semantic_RAG_Enterprise_Benchmark_Prompt.md Section 4).

Unlike the prior atomic-fact benchmark (aisos/benchmark/systems.py), these
adapters operate over a real document corpus and report retrieval-quality
metrics (ranked doc-ID lists) separately from governance/answer metrics,
per Section 9's explicit "retrieval relevance" vs. "memory governance"
distinction.

**S1 (no_sleep_naive) and S5 (continuous_mvcc) have no semantic retrieval
mechanism at all** -- they are pure key-value stores, exactly as in every
prior milestone. For this benchmark, they are given the query's
ground-truth topic key directly (they cannot discover it from free text),
and their retrieval-quality metrics are reported as N/A -- an honest
design choice, not a fabricated capability, matching Section 9's
instruction not to conflate "found the right topic" with "resolved the
right value once given the topic."

**S6/S6b (AISOS alone, no RAG) have the same limitation**: AISOS governs
which value is trusted for a known subject key, but has no mechanism of
its own for discovering which subject a free-text query is about. It is
therefore also given the ground-truth topic key directly, and its
retrieval-quality metrics are N/A by design -- this measures AISOS's
governance quality in isolation, not a retrieval capability it was never
built to have.

**S2/S3/S4/S7 perform genuine semantic retrieval** (embedding-based, or
Mem0's own retrieval) and their Recall@k/Precision@k/MRR/NDCG@k are
computed from real ranked document-ID lists.
"""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional

import chromadb

from aisos.agent.condition_backends import PeriodicAISOSBackend, TrustedAuthorityPeriodicAISOSBackend
from aisos.benchmark.embeddings import Embedder
from aisos.benchmark.semantic.corpus import Document

PERIOD = 15


class SemanticSystem(ABC):
    condition_name: str
    has_retrieval: bool  # False for S1/S5/S6/S6b -- see module docstring

    @abstractmethod
    def ingest(self, documents: list[Document]) -> None: ...

    def retrieve(self, query_text: str, k: int) -> list[str]:
        """Ranked doc_ids, most relevant first. Empty list if has_retrieval
        is False (N/A, not a failure)."""
        return []

    @abstractmethod
    def governed_value(self, topic: str) -> Optional[str]:
        """The value this system currently trusts for `topic`, or None."""
        ...

    @abstractmethod
    def context_documents(self, query_text: str, topic: Optional[str], k: int) -> list[Document]:
        """Documents/claims actually handed to the LLM to answer a query."""
        ...

    def maybe_maintain(self, tick: int) -> None:
        pass

    def snapshot_metrics(self) -> dict:
        return {}


class KeyValueOnlySystem(SemanticSystem):
    """S1/S5 adapter: exact topic->value store, last-write-wins (S1) or
    continuous-MVCC (S5, reusing the existing baseline unmodified)."""
    has_retrieval = False

    def __init__(self, backend):
        self._backend = backend
        self._docs_by_topic: dict[str, list[Document]] = {}

    def ingest(self, documents: list[Document]) -> None:
        for i, d in enumerate(documents):
            if d.topic is None or d.value is None:
                continue  # indirect/no-claim documents assert nothing -- must not overwrite a real value
            self._backend.write(d.topic, d.value, d.source, d.confidence, i)
            self._docs_by_topic.setdefault(d.topic, []).append(d)

    def governed_value(self, topic: str) -> Optional[str]:
        return self._backend.read(topic)

    def context_documents(self, query_text: str, topic: Optional[str], k: int) -> list[Document]:
        if topic is None:
            return []
        v = self.governed_value(topic)
        docs = self._docs_by_topic.get(topic, [])
        matching = [d for d in docs if d.value == v]
        return matching[-1:] if matching else []

    def snapshot_metrics(self) -> dict:
        return self._backend.snapshot_metrics()


class AISOSOnlySystem(SemanticSystem):
    """S6/S6b adapter: AISOS governs topic->value, no retrieval of its own."""
    has_retrieval = False

    def __init__(self, backend, credential_aware: bool = False):
        self._backend = backend
        self._credential_aware = credential_aware
        self._docs_by_topic: dict[str, list[Document]] = {}
        self._tick = 0

    def ingest(self, documents: list[Document]) -> None:
        for d in documents:
            if d.topic is None or d.value is None:
                continue  # indirect/no-claim documents assert nothing -- must not overwrite a real value
            self._tick += 1
            credential = None
            if self._credential_aware and d.legitimate and hasattr(self._backend, "issue_credential"):
                credential = self._backend.issue_credential(d.source)
            try:
                self._backend.write(d.topic, d.value, d.source, d.confidence, self._tick, credential=credential)
            except TypeError:
                self._backend.write(d.topic, d.value, d.source, d.confidence, self._tick)
            self._docs_by_topic.setdefault(d.topic, []).append(d)
        for i in range(1, 8 * PERIOD):
            self._backend.maybe_maintain(self._tick + i)
        self._tick += 8 * PERIOD

    def governed_value(self, topic: str) -> Optional[str]:
        return self._backend.read(topic)

    def context_documents(self, query_text: str, topic: Optional[str], k: int) -> list[Document]:
        if topic is None:
            return []
        v = self.governed_value(topic)
        docs = self._docs_by_topic.get(topic, [])
        matching = [d for d in docs if d.value == v]
        return matching[-1:] if matching else []

    def maybe_maintain(self, tick: int) -> None:
        self._backend.maybe_maintain(tick)

    def snapshot_metrics(self) -> dict:
        return self._backend.snapshot_metrics()


class RAGSemanticSystem(SemanticSystem):
    """S2/S3 adapter: real chromadb + nomic-embed-text retrieval over the
    raw document corpus. No governance -- context_documents() returns the
    top-k retrieved documents as-is, whatever their content."""
    has_retrieval = True

    def __init__(self, embedder: Optional[Embedder] = None, rerank: bool = False, model=None):
        self.embedder = embedder or Embedder()
        self._client = chromadb.Client()
        self._collection = self._client.create_collection(f"sem_{uuid.uuid4().hex[:8]}")
        self._docs_by_id: dict[str, Document] = {}
        self.rerank = rerank
        self.model = model
        self._n = 0

    def ingest(self, documents: list[Document]) -> None:
        batch = 64
        for start in range(0, len(documents), batch):
            chunk = documents[start:start + batch]
            embeddings = [self.embedder.embed(d.text) for d in chunk]
            self._collection.add(
                ids=[d.doc_id for d in chunk], embeddings=embeddings, documents=[d.text for d in chunk],
                metadatas=[{"topic": d.topic or "", "value": d.value or ""} for d in chunk],
            )
            for d in chunk:
                self._docs_by_id[d.doc_id] = d
            self._n += len(chunk)

    def retrieve(self, query_text: str, k: int) -> list[str]:
        if self._n == 0:
            return []
        q_emb = self.embedder.embed(query_text)
        n = min(k, self._n)
        result = self._collection.query(query_embeddings=[q_emb], n_results=n)
        ids = result["ids"][0]
        if self.rerank and self.model is not None and ids:
            ids = self._llm_rerank(query_text, ids)
        return ids

    def _llm_rerank(self, query_text: str, doc_ids: list[str]) -> list[str]:
        scored = []
        for did in doc_ids:
            text = self._docs_by_id[did].text
            prompt = (
                f"Query: {query_text}\nCandidate passage: {text}\n"
                "On a scale of 0 to 10, how relevant is this passage to answering the query? "
                "Respond with ONLY a single integer 0-10, nothing else."
            )
            try:
                resp = self.model.generate(prompt).strip()
                score = int("".join(c for c in resp if c.isdigit())[:2] or "0")
            except Exception:
                score = 0
            scored.append((score, did))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [did for _, did in scored]

    def governed_value(self, topic: str) -> Optional[str]:
        return None  # no governance concept

    def context_documents(self, query_text: str, topic: Optional[str], k: int) -> list[Document]:
        ids = self.retrieve(query_text, k)
        return [self._docs_by_id[i] for i in ids if i in self._docs_by_id]

    def snapshot_metrics(self) -> dict:
        return {"store_size": self._n, "embedder_backend": self.embedder.backend_used}


class ProvenanceWeightedRAGSystem(SemanticSystem):
    """System B for the write-time-gating-vs-retrieval-time-weighting
    experiment (AISOS_v3_Complete_Novelty_Elevation_Master_Prompt.md
    Priority 5; docs/write_time_vs_retrieval_time_governance.md). Same
    embedding/vector-store pipeline as S2 (RAGSemanticSystem) -- this is
    NOT a strawman: it is standard RAG retrieval, plus a re-ranking step
    that WEIGHTS each candidate's score by its own provenance/authority
    metadata (already present on every Document, populated from
    SOURCE_TYPES during corpus construction) before selecting the top-k
    context. This models what "Utility Under Attack" (arXiv:2608.21230)
    calls retrieval-time provenance weighting, in contrast to AISOS's
    (S6/S7) write-time gating: a low-authority document is RANKED LOWER
    here, but is never rejected outright, and can still surface if its
    semantic similarity dominates the (1-alpha) authority term -- unlike
    AISOS, where a low-authority candidate never becomes governed state
    at all, regardless of how semantically relevant it looks.

    alpha=0.5 (equal weight to semantic rank and authority) is fixed
    before any result is examined -- not tuned post-hoc to make this
    condition look better or worse; see the governing document's own
    "do not assume the result before testing" instruction.
    """
    has_retrieval = True
    ALPHA = 0.5  # weight on semantic-similarity rank vs. (1-ALPHA) on authority; fixed a priori

    def __init__(self, embedder: Optional[Embedder] = None):
        self.embedder = embedder or Embedder()
        self._client = chromadb.Client()
        self._collection = self._client.create_collection(f"semB_{uuid.uuid4().hex[:8]}")
        self._docs_by_id: dict[str, Document] = {}
        self._n = 0

    def ingest(self, documents: list[Document]) -> None:
        batch = 64
        for start in range(0, len(documents), batch):
            chunk = documents[start:start + batch]
            embeddings = [self.embedder.embed(d.text) for d in chunk]
            self._collection.add(
                ids=[d.doc_id for d in chunk], embeddings=embeddings, documents=[d.text for d in chunk],
                metadatas=[{"topic": d.topic or "", "value": d.value or ""} for d in chunk],
            )
            for d in chunk:
                self._docs_by_id[d.doc_id] = d
            self._n += len(chunk)

    def retrieve(self, query_text: str, k: int) -> list[str]:
        if self._n == 0:
            return []
        q_emb = self.embedder.embed(query_text)
        # Over-fetch a wider candidate pool than k so authority weighting
        # has real candidates to re-rank among, not just k already-final choices.
        n = min(max(k * 3, 10), self._n)
        result = self._collection.query(query_embeddings=[q_emb], n_results=n)
        ids = result["ids"][0]
        distances = result["distances"][0] if result.get("distances") else [0.0] * len(ids)
        if not ids:
            return []
        max_d = max(distances) or 1.0
        scored = []
        for rank, (did, dist) in enumerate(zip(ids, distances)):
            similarity = 1.0 - (dist / max_d) if max_d > 0 else 1.0
            authority = self._docs_by_id.get(did).authority if did in self._docs_by_id else 0.4
            combined = self.ALPHA * similarity + (1.0 - self.ALPHA) * authority
            scored.append((combined, did))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [did for _, did in scored[:k]]

    def governed_value(self, topic: str) -> Optional[str]:
        return None  # retrieval-time weighting has no persistent governance state, by design

    def context_documents(self, query_text: str, topic: Optional[str], k: int) -> list[Document]:
        ids = self.retrieve(query_text, k)
        return [self._docs_by_id[i] for i in ids if i in self._docs_by_id]

    def snapshot_metrics(self) -> dict:
        return {"store_size": self._n, "alpha": self.ALPHA}


class SMSRStyleSystem(SemanticSystem):
    """System E for the closest-prior-art comparison
    (AISOS_v3_Complete_Novelty_Elevation_Master_Prompt.md Priority 4;
    docs/closest_prior_art_comparison.md) -- a FAITHFUL, MECHANISM-LEVEL
    reimplementation of SMSR (arXiv:2606.12703, "Certified Defence Against
    Runtime Memory Poisoning in Persistent LLM Agent Systems"), not a
    literal run of the paper's own released repository
    (github.com/tarun-ks/smsr). That repository's own evaluation harness
    is hardwired to the Anthropic API (Claude Haiku/Sonnet as both agent
    and LLM judge) -- running it as-is would violate this project's own
    "same model" fairness rule (no Anthropic API key is configured in
    this environment, and using a different model for SMSR than for AISOS
    would make the comparison invalid on its own terms, not merely
    inconvenient). This class reimplements SMSR's two published
    components against AISOS's own Ollama model stack and AISOS's own
    corpus/evaluator instead, so the SAME model, data, and evaluator are
    used across every condition in this experiment -- explicitly disclosed
    as a reimplementation, per this project's "do not fabricate or
    approximate a result and call it a direct comparison" rule.

    Component 1 (write-time signing): documents from a legitimate
    (`d.legitimate=True`) source are treated as "signed"; documents from
    an illegitimate source (`d.legitimate=False`, this corpus's own
    convention for injected/malicious documents) are treated as
    "unsigned" and excluded from the retrieval pool entirely -- SMSR's
    own reported behavior ("blocks unsigned injection outright"). This is
    a corpus-level proxy for HMAC verification (the real corpus has no
    actual per-document signing infrastructure to attach cryptographic
    signatures to), not a claim that real HMAC signing was implemented.

    Component 2 (query-time randomized ablation + majority vote): of the
    remaining ("signed") top-m candidates, N_RUNS independent trials each
    sample a random subset of size ceil(m/2) and ask the LLM to answer
    from that subset alone; the majority verdict across N_RUNS trials is
    returned. N_RUNS=3 here (the original paper uses up to 30 for a tight
    certified bound) -- reduced specifically to keep real-model call cost
    bounded for this session, disclosed as a reduced-N deviation from the
    paper's own protocol, not presented as a full certified-robustness
    reproduction.
    """
    has_retrieval = True
    N_RUNS = 3
    M_CANDIDATES = 6

    def __init__(self, embedder: Optional[Embedder] = None, model=None, rng_seed: int = 0):
        import random
        self.embedder = embedder or Embedder()
        self.model = model
        self._client = chromadb.Client()
        self._collection = self._client.create_collection(f"smsr_{uuid.uuid4().hex[:8]}")
        self._docs_by_id: dict[str, Document] = {}
        self._n = 0
        self._rng = random.Random(rng_seed)

    def ingest(self, documents: list[Document]) -> None:
        signed = [d for d in documents if d.legitimate]  # Component 1: unsigned (illegitimate) docs never enter the pool
        batch = 64
        for start in range(0, len(signed), batch):
            chunk = signed[start:start + batch]
            embeddings = [self.embedder.embed(d.text) for d in chunk]
            self._collection.add(
                ids=[d.doc_id for d in chunk], embeddings=embeddings, documents=[d.text for d in chunk],
                metadatas=[{"topic": d.topic or "", "value": d.value or ""} for d in chunk],
            )
            for d in chunk:
                self._docs_by_id[d.doc_id] = d
            self._n += len(chunk)

    def retrieve(self, query_text: str, k: int) -> list[str]:
        if self._n == 0:
            return []
        q_emb = self.embedder.embed(query_text)
        m = min(self.M_CANDIDATES, self._n)
        result = self._collection.query(query_embeddings=[q_emb], n_results=m)
        return result["ids"][0][:k]

    def governed_value(self, topic: str) -> Optional[str]:
        return None  # certified-robustness voting is a per-query answer mechanism, not a persistent trusted value

    def context_documents(self, query_text: str, topic: Optional[str], k: int) -> list[Document]:
        ids = self.retrieve(query_text, self.M_CANDIDATES)
        return [self._docs_by_id[i] for i in ids if i in self._docs_by_id]

    def answer_with_ablation_voting(self, query_text: str, prompt_builder) -> tuple[str, list[dict]]:
        """Component 2: N_RUNS trials, each over a random half-subset of
        the (already write-time-filtered) top-m candidates; majority
        verdict wins. Returns (majority_response, per_trial_records) so
        the caller can still compute standard evaluator/dangerous-rate
        metrics against the majority answer."""
        ids = self.retrieve(query_text, self.M_CANDIDATES)
        docs = [self._docs_by_id[i] for i in ids if i in self._docs_by_id]
        if not docs:
            return "", []
        subset_size = max(1, -(-len(docs) // 2))  # ceil(len/2)
        trials = []
        for _ in range(self.N_RUNS):
            subset = self._rng.sample(docs, min(subset_size, len(docs)))
            context_items = [{"subject": d.topic or d.doc_id, "value": d.value} for d in subset if d.value is not None]
            prompt = prompt_builder(query_text, context_items)
            response = self.model.generate(prompt, system="You are a careful assistant answering only from the provided memory/context.")
            trials.append({"subset_doc_ids": [d.doc_id for d in subset], "response": response})
        # Majority vote by exact response string; ties broken by first-seen order (deterministic given rng_seed)
        from collections import Counter
        counts = Counter(t["response"] for t in trials)
        majority_response = counts.most_common(1)[0][0]
        return majority_response, trials

    def snapshot_metrics(self) -> dict:
        return {"store_size": self._n, "n_runs": self.N_RUNS, "m_candidates": self.M_CANDIDATES}


class Mem0SemanticSystem(SemanticSystem):
    """S4 adapter: real local Mem0 (llama3 LLM + nomic-embed-text + Chroma,
    no cloud API key), reused from aisos/benchmark/mem0_backend.py's
    established configuration."""
    has_retrieval = True

    def __init__(self, user_id: Optional[str] = None):
        from mem0 import Memory
        from aisos.benchmark.mem0_backend import DEFAULT_CONFIG
        config = {**DEFAULT_CONFIG, "vector_store": {
            "provider": "chroma", "config": {"collection_name": f"sem_mem0_{uuid.uuid4().hex[:8]}"},
        }}
        self._memory = Memory.from_config(config)
        self.user_id = user_id or f"sem_bench_{uuid.uuid4().hex[:8]}"
        self._docs_by_id: dict[str, Document] = {}
        self._n = 0

    def ingest(self, documents: list[Document]) -> None:
        for d in documents:
            self._memory.add(d.text, user_id=self.user_id, metadata={
                "doc_id": d.doc_id, "topic": d.topic or "", "value": d.value or "",
            })
            self._docs_by_id[d.doc_id] = d
            self._n += 1

    def _search(self, query_text: str, k: int) -> list[dict]:
        # Mem0's search() takes `top_k`, not `limit` -- see the identical
        # fix and explanation in aisos/benchmark/mem0_backend.py.
        result = self._memory.search(query_text, filters={"user_id": self.user_id}, top_k=k)
        return result.get("results", [])

    def retrieve(self, query_text: str, k: int) -> list[str]:
        hits = self._search(query_text, k)
        out = []
        for h in hits:
            meta = h.get("metadata") or {}
            did = meta.get("doc_id")
            if did:
                out.append(did)
        return out

    def governed_value(self, topic: str) -> Optional[str]:
        return None

    def context_documents(self, query_text: str, topic: Optional[str], k: int) -> list[Document]:
        ids = self.retrieve(query_text, k)
        docs = [self._docs_by_id[i] for i in ids if i in self._docs_by_id]
        if docs:
            return docs
        # Metadata may not survive Mem0's own rewrite for every hit --
        # fall back to a synthetic Document carrying Mem0's own memory text.
        hits = self._search(query_text, k)
        return [Document(doc_id=h.get("id", "?"), text=h.get("memory", ""), topic=topic, value=None,
                          source="mem0", authority=0.5, confidence=0.5, timestamp=0.0) for h in hits]

    def snapshot_metrics(self) -> dict:
        return {"store_size": self._n}


class AISOSPlusRAGSemanticSystem(SemanticSystem):
    """S7 adapter: RAG retrieves candidate documents/topics; AISOS governs
    the trusted value for whichever topics retrieval surfaces. This is the
    architecturally correct composition per Section 6/9: retrieval finds
    evidence, governance decides what becomes trusted state."""
    has_retrieval = True

    def __init__(self, embedder: Optional[Embedder] = None, aisos_backend_factory=None):
        self.rag = RAGSemanticSystem(embedder=embedder)
        self.aisos = (aisos_backend_factory or (lambda: PeriodicAISOSBackend(period=PERIOD)))()
        self._docs_by_topic: dict[str, list[Document]] = {}
        self._tick = 0

    def ingest(self, documents: list[Document]) -> None:
        self.rag.ingest(documents)
        for d in documents:
            if d.topic is None or d.value is None:
                continue  # indirect/no-claim documents assert nothing -- must not overwrite a real value
            self._tick += 1
            try:
                self.aisos.write(d.topic, d.value, d.source, d.confidence, self._tick, credential=None)
            except TypeError:
                self.aisos.write(d.topic, d.value, d.source, d.confidence, self._tick)
            self._docs_by_topic.setdefault(d.topic, []).append(d)
        for i in range(1, 8 * PERIOD):
            self.aisos.maybe_maintain(self._tick + i)
        self._tick += 8 * PERIOD

    def retrieve(self, query_text: str, k: int) -> list[str]:
        return self.rag.retrieve(query_text, k)

    def governed_value(self, topic: str) -> Optional[str]:
        return self.aisos.read(topic)

    def context_documents(self, query_text: str, topic: Optional[str], k: int) -> list[Document]:
        """RAG finds which topics are relevant to the free-text query;
        AISOS's governed value (not RAG's raw retrieved value) is
        substituted for each -- 'safe memory' reaching the LLM."""
        raw = self.rag.context_documents(query_text, topic, k)
        seen_topics = []
        out = []
        for d in raw:
            if d.topic is None or d.topic in seen_topics:
                continue
            seen_topics.append(d.topic)
            governed = self.governed_value(d.topic)
            if governed is not None:
                docs_for_topic = self._docs_by_topic.get(d.topic, [])
                matching = [x for x in docs_for_topic if x.value == governed]
                out.append(matching[-1] if matching else d)
            else:
                out.append(d)
        return out

    def maybe_maintain(self, tick: int) -> None:
        self.aisos.maybe_maintain(tick)

    def snapshot_metrics(self) -> dict:
        return {"rag": self.rag.snapshot_metrics(), "aisos": self.aisos.snapshot_metrics()}


def _make_semantic_aisos_backend():
    """S6: uses this benchmark's own SOURCE_TYPES authority table (see
    comparator.py's module docstring for the bug this fixes -- AISOS's
    production SOURCE_AUTHORITY table doesn't recognize this corpus's
    source strings at all, silently collapsing every document to the same
    default authority)."""
    from aisos.benchmark.semantic.comparator import semantic_authority_score, semantic_contradiction_score
    return PeriodicAISOSBackend(
        period=PERIOD, contradiction_score_fn=semantic_contradiction_score, authority_score_fn=semantic_authority_score,
    )


def _make_semantic_authenticated_backend():
    """S6b: uses AISOS's real verified-authority path (credential-gated),
    with the registry's trusted_sources populated from this benchmark's
    own SOURCE_TYPES table instead of the production default (which,
    identically to S6's issue, doesn't recognize this corpus's source
    strings)."""
    from aisos.security.authority_registry import TrustedAuthorityRegistry
    from aisos.security.checkpoint import new_signing_key
    from aisos.benchmark.semantic.corpus import SOURCE_TYPES
    registry = TrustedAuthorityRegistry(new_signing_key(), trusted_sources=dict(SOURCE_TYPES))
    return TrustedAuthorityPeriodicAISOSBackend(period=PERIOD, registry=registry)


def make_systems(model=None) -> dict:
    from aisos.agent.condition_backends import NoSleepBackend, ContinuousMVCCBackend
    embedder = Embedder()
    return {
        "S1_no_sleep_naive": lambda: KeyValueOnlySystem(NoSleepBackend()),
        "S2_rag_standard": lambda: RAGSemanticSystem(embedder=embedder),
        "S3_rag_reranked": lambda: RAGSemanticSystem(embedder=embedder, rerank=True, model=model),
        "S4_mem0_external": lambda: Mem0SemanticSystem(),
        "S5_continuous_mvcc": lambda: KeyValueOnlySystem(ContinuousMVCCBackend()),
        "S6_aisos": lambda: AISOSOnlySystem(_make_semantic_aisos_backend()),
        "S6b_aisos_authenticated": lambda: AISOSOnlySystem(
            _make_semantic_authenticated_backend(), credential_aware=True),
        "S7_aisos_plus_rag": lambda: AISOSPlusRAGSemanticSystem(
            embedder=embedder, aisos_backend_factory=_make_semantic_aisos_backend),
    }
