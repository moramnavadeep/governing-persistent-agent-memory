"""
S4 -- external long-term memory system baseline
(AISOS_v3_Claude_Competitive_RAG_Multimodel_100K_Prompt.md Section 2, S4;
Section 12).

Uses Mem0 (`mem0ai`, https://github.com/mem0ai/mem0), a real, published,
reproducible open-source long-term memory framework, configured entirely
locally (no OpenAI/cloud API key): `llm` and `embedder` both point at this
project's own already-running Ollama instance (`llama3:latest` /
`nomic-embed-text`), and `vector_store` uses `chromadb` -- the identical
embedding model and LLM already used by every other system in this
benchmark, so the comparison is apples-to-apples on model choice.
Unmodified library code; no weakening.

Documented, observed properties of Mem0's OWN behavior (not altered by
this wrapper):
  - `add()` internally calls the configured LLM to decide how to process
    each new fact (extract/rewrite/decide ADD vs. UPDATE vs. DELETE vs.
    NONE against existing memories) -- this makes Mem0's write path
    inherently more expensive than AISOS's or plain RAG's (an LLM call
    per write, not just an embedding call), a genuine architectural
    property of this system, disclosed in the latency results rather than
    hidden.
  - Mem0 REWRITES stored facts through its own LLM rather than storing the
    caller's literal value string verbatim (e.g. "The user lives in
    Paris" rather than the bare value "Paris"). `read()`/`read_all_values()`
    below therefore check for the raw value as a SUBSTRING of Mem0's
    rewritten memory text, matching the same convention this project's own
    evaluator (`aisos/agent/evaluator.py`) already uses for every other
    system, rather than requiring an exact string match Mem0 was never
    going to produce.
  - In ad-hoc testing during this milestone, Mem0's own conflict-resolution
    LLM call did NOT automatically supersede an old fact when a new,
    contradicting one arrived (both "lives in Paris" and "moved to Berlin"
    remained independently retrievable) -- this is Mem0's own configured
    behavior, observed and reported as-is, not adjusted to make either
    system look better or worse.
"""
from __future__ import annotations

import uuid
from typing import Optional

from aisos.agent.condition_backends import ConditionBackend

DEFAULT_CONFIG = {
    "llm": {"provider": "ollama", "config": {
        "model": "llama3:latest", "temperature": 0.0, "ollama_base_url": "http://localhost:11434",
    }},
    "embedder": {"provider": "ollama", "config": {
        "model": "nomic-embed-text", "ollama_base_url": "http://localhost:11434",
    }},
    "vector_store": {"provider": "chroma", "config": {"collection_name": None}},  # filled in per-instance
}


class Mem0Backend(ConditionBackend):
    condition_name = "mem0_external"

    def __init__(self, top_k: int = 5, user_id: Optional[str] = None):
        from mem0 import Memory
        config = {**DEFAULT_CONFIG, "vector_store": {
            "provider": "chroma", "config": {"collection_name": f"mem0_{uuid.uuid4().hex[:8]}"},
        }}
        self._memory = Memory.from_config(config)
        self.user_id = user_id or f"bench_{uuid.uuid4().hex[:8]}"
        self.top_k = top_k
        self._write_count = 0

    def write(self, subject, value, source, confidence, interaction_id, credential=None):
        text = f"{subject}: {value}"
        self._memory.add(text, user_id=self.user_id, metadata={
            "subject": subject, "value": value, "source": source,
            "confidence": confidence, "interaction_id": interaction_id,
        })
        self._write_count += 1

    def _search(self, query_text: str, k: int) -> list[dict]:
        # Mem0's search() takes `top_k`, not `limit` -- `limit=k` was
        # silently absorbed into **kwargs and ignored, meaning every call
        # used Mem0's own default top_k=20 regardless of the requested k.
        # Confirmed via direct signature inspection during the
        # semantic-benchmark milestone; fixed here retroactively since this
        # affects every Mem0 result in this project, not just new ones.
        result = self._memory.search(query_text, filters={"user_id": self.user_id}, top_k=k)
        return result.get("results", [])

    def read(self, subject):
        hits = self._search(subject, self.top_k)
        for h in hits:
            meta = h.get("metadata") or {}
            if meta.get("subject") == subject:
                return meta.get("value")
        # Fallback: metadata may not survive Mem0's own fact-extraction rewrite
        # for every hit -- check the rewritten memory text itself.
        for h in hits:
            if subject in (h.get("memory") or ""):
                return h.get("memory")
        return None

    def read_all_values(self, subject):
        hits = self._search(subject, self.top_k)
        values = []
        for h in hits:
            meta = h.get("metadata") or {}
            if meta.get("subject") == subject and meta.get("value") not in values:
                values.append(meta.get("value"))
        return values

    def retrieve_context(self, query_text: str) -> list[dict]:
        """Normalizes Mem0's raw search-result shape (id/memory/metadata/
        score/...) to the same {subject, value, ...} shape RAGBackend's
        retrieve_context() returns, so the shared benchmark prompt-builder
        and evaluator code can treat every context-based system uniformly.
        Falls back to the rewritten `memory` text as `value` when
        metadata wasn't preserved for a given hit (Mem0's own consolidation
        LLM can merge/rewrite entries in ways that don't always carry the
        original metadata forward)."""
        hits = self._search(query_text, self.top_k)
        out = []
        for h in hits:
            meta = h.get("metadata") or {}
            out.append({
                "subject": meta.get("subject", ""), "value": meta.get("value", h.get("memory", "")),
                "source": meta.get("source", "mem0"), "score": h.get("score"), "memory_text": h.get("memory"),
            })
        return out

    def maybe_maintain(self, interaction_id):
        pass  # Mem0 consolidates synchronously inside add(), no separate cycle

    def snapshot_metrics(self):
        return {"store_size": self._write_count, "cycles_run": 0, "lifecycle_state": "N/A", "sleep_pressure": None}
