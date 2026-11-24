"""
Shared QA prompt builder for the competitive benchmark -- generalizes
aisos/experiments/e8_long_running_1000.py's `_build_prompt` (subject +
current_value + store_values) to also accept a free-text list of retrieved
context passages, so the SAME builder drives AISOS-family systems
(subject-keyed) and RAG-family systems (context-list-retrieved) with one
shared, unbiased prompt convention (Section 3's fairness rule: same
evaluation procedure for every system).
"""
from __future__ import annotations


def build_prompt_from_subject(subject: str, predicate: str, query: str, current_value, store_values: list[str]) -> str:
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


def build_prompt_from_context(query: str, context_items: list[dict]) -> str:
    """context_items: list of {"subject":..., "value":...} (or any dict
    with those two keys) -- the retrieved candidates a RAG-family system's
    retrieve_context() returned."""
    parts = []
    if not context_items:
        parts.append("No memory is available to answer this question.")
    else:
        distinct = list(dict.fromkeys(f"{c['subject']}: {c['value']}" for c in context_items))
        parts.append("Retrieved memory:")
        parts.extend(f"- {d}" for d in distinct)
    parts.append(f"Question: {query}")
    parts.append("Answer in one short sentence, using only the memory above. If you don't know, say so.")
    return "\n".join(parts)
