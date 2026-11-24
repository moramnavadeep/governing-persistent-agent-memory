"""
Ground-truth evaluator (AISOS_1000_INTERACTION_EXPERIMENT_PROMPT.md
Section 9): six explicit labels, never forcing an ambiguous case into
correct/incorrect.

    CORRECT        model output contains the current ground-truth value.
    STALE          model output contains a *previously* valid value for
                   the same key instead of the current one -- a specific,
                   diagnosable failure mode (Formal Theory / Implementation
                   Report already treat "hasn't caught up yet" as distinct
                   from "answered confidently and wrong").
    CONTRADICTORY  the backend's own store currently holds more than one
                   distinct live value for this subject -- the evaluator
                   cannot attribute a single ground truth to what the agent
                   was even given to work with, independent of what the
                   model said.
    UNKNOWN        the model hedges (explicit "I don't know"-style output).
    UNDETERMINED   no ground truth is defined for this query at this
                   interaction (e.g. a temporal fact queried outside its
                   validity window) -- there is nothing to be right or
                   wrong about.
    INCORRECT      none of the above; a confident, wrong answer.

Deterministic string matching only (Section 9: "prefer deterministic
evaluation whenever possible"). No semantic/LLM-as-judge fallback is
implemented in this pass -- see docs/implementation_traceability.md.
"""
from __future__ import annotations

HEDGE_MARKERS = (
    "i don't know", "i do not know", "not sure", "no information",
    "unknown", "cannot determine", "no memory", "not available",
)

EvaluatorVerdict = str


def evaluate(
    model_output: str,
    ground_truth_value: str | None,
    all_known_values: list[str],
    store_values: list[str],
) -> EvaluatorVerdict:
    if len(set(store_values)) > 1:
        return "CONTRADICTORY"
    if ground_truth_value is None:
        return "UNDETERMINED"
    out = model_output.lower()
    if ground_truth_value.lower() in out:
        return "CORRECT"
    if any(m in out for m in HEDGE_MARKERS):
        return "UNKNOWN"
    for v in all_known_values:
        if v is not None and v.lower() != ground_truth_value.lower() and v.lower() in out:
            return "STALE"
    return "INCORRECT"
