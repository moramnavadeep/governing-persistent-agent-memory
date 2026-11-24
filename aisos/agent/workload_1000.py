"""
Deterministic 1,000-interaction long-running-agent workload
(AISOS_1000_INTERACTION_EXPERIMENT_PROMPT.md Section 5).

20 sessions x 50 interactions. A context reset happens at the start of
every session after the first (Section 6). Each interaction is either a
"teach" (writes a fact into agent memory / the ground-truth ledger) or a
"query" (asks the agent about a subject/predicate and is scored against
ground truth). Seven fact categories, per Section 5:

    stable         -- taught once, true for the whole run, queried repeatedly.
    changing       -- value changes at known interaction numbers.
    contradictory  -- a low-authority observation conflicts with an
                      established high-authority fact (must not win).
    repeated       -- the identical (subject, value) taught more than once.
    irrelevant     -- noise facts, never queried directly; exist only to
                      load the memory store and test retrieval precision.
    temporal       -- valid only inside an explicit interaction window;
                      queries outside the window have no ground truth
                      (UNDETERMINED), not a wrong answer to penalize.
    cross_session  -- taught in one session, queried again >=2 sessions
                      (i.e. >=2 context resets) later.

Determinism: everything is derived from a single seeded random.Random;
build_workload() returns the exact same InteractionSpec sequence and
GroundTruthLedger for a given seed every time, and the workload is also
serialized to JSON (Section 5: "store the workload itself so it can be
replayed exactly").
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from aisos.agent.ground_truth import GroundTruthLedger

N_SESSIONS = 20
INTERACTIONS_PER_SESSION = 50
N_INTERACTIONS = N_SESSIONS * INTERACTIONS_PER_SESSION  # 1000

N_STABLE = 5
N_CHANGING = 5
N_CONTRADICTORY = 3
N_REPEATED = 3
N_IRRELEVANT = 10
N_TEMPORAL = 3
N_CROSS_SESSION = 3

SOURCE_AUTHORITY_ORDER = ("tool", "user", "user_correction")  # low -> high, matches memory/contradiction.py


@dataclass
class InteractionSpec:
    interaction_id: int
    session_id: int
    context_reset_before: bool
    kind: str            # "teach" | "query"
    category: str        # stable | changing | contradictory | repeated | irrelevant | temporal | cross_session
    subject: str
    predicate: str
    value: Optional[str] = None       # for "teach"
    query_text: Optional[str] = None  # for "query"
    source: str = "user"
    confidence: float = 0.85


def _q(subject: str, predicate: str) -> str:
    return f"What is the current {predicate} for {subject}?"


def build_workload(seed: int = 0) -> tuple[list[InteractionSpec], GroundTruthLedger]:
    rng = random.Random(seed)
    ledger = GroundTruthLedger()
    specs: list[InteractionSpec] = []

    stable = [f"stable_{i}" for i in range(N_STABLE)]
    changing = [f"changing_{i}" for i in range(N_CHANGING)]
    contradictory = [f"contra_{i}" for i in range(N_CONTRADICTORY)]
    repeated = [f"repeat_{i}" for i in range(N_REPEATED)]
    irrelevant = [f"noise_{i}" for i in range(N_IRRELEVANT)]
    temporal = [f"temporal_{i}" for i in range(N_TEMPORAL)]
    cross_session = [f"cross_{i}" for i in range(N_CROSS_SESSION)]

    predicate = "value"  # single predicate per synthetic subject keeps the schema uniform
    iid = 0

    def teach(session, subject, category, value, source="user", confidence=0.85, reset=False):
        nonlocal iid
        specs.append(InteractionSpec(
            interaction_id=iid, session_id=session, context_reset_before=reset,
            kind="teach", category=category, subject=subject, predicate=predicate,
            value=value, source=source, confidence=confidence,
        ))
        iid += 1

    def query(session, subject, category, reset=False):
        nonlocal iid
        specs.append(InteractionSpec(
            interaction_id=iid, session_id=session, context_reset_before=reset,
            kind="query", category=category, subject=subject, predicate=predicate,
            query_text=_q(subject, predicate),
        ))
        iid += 1

    # ---- fixed schedule of *when* things happen, expressed in sessions ----
    changing_change_session = {s: rng.choice(range(3, N_SESSIONS - 1)) for s in changing}
    contradiction_session = {s: rng.choice(range(2, N_SESSIONS)) for s in contradictory}
    cross_session_teach_session = {s: 0 for s in cross_session}  # taught in session 0
    cross_session_query_sessions = {
        s: sorted(rng.sample(range(3, N_SESSIONS), k=min(3, N_SESSIONS - 3))) for s in cross_session
    }
    temporal_window = {}
    for s in temporal:
        start_session = rng.choice(range(2, N_SESSIONS - 4))
        temporal_window[s] = (start_session, start_session + 2)  # inclusive-exclusive by session

    for session in range(N_SESSIONS):
        reset_this_session = session > 0
        session_events: list = []

        # -- teach events --
        if session == 0:
            for s in stable:
                session_events.append(("teach", s, "stable", f"stable_value_{s}", "user", 0.9))
            for s in cross_session:
                session_events.append(("teach", s, "cross_session", f"cross_value_{s}", "user", 0.9))
            for s in changing:
                session_events.append(("teach", s, "changing", f"{s}_v0", "user", 0.85))
            for s in repeated:
                session_events.append(("teach", s, "repeated", f"{s}_value", "user", 0.85))

        for s in changing:
            if changing_change_session[s] == session:
                session_events.append(("teach", s, "changing", f"{s}_v1", "user_correction", 0.95))

        for s in contradictory:
            if contradiction_session[s] == session:
                if session == 0:
                    session_events.append(("teach", s, "contradictory", f"{s}_true_value", "user", 0.9))
                session_events.append(("teach", s, "contradictory", f"{s}_FALSE_value", "tool", 0.6))
            elif session == 0:
                session_events.append(("teach", s, "contradictory", f"{s}_true_value", "user", 0.9))

        for s in repeated:
            if session > 0 and rng.random() < 0.3:
                session_events.append(("teach", s, "repeated", f"{s}_value", "user", 0.8))  # identical value again

        for s in temporal:
            win_start, win_end = temporal_window[s]
            if session == win_start:
                session_events.append(("teach", s, "temporal", f"{s}_active_value", "user", 0.9))
            if session == win_end:
                session_events.append(("teach", s, "temporal", f"{s}_expired", "user_correction", 0.9))

        n_noise = 3
        for _ in range(n_noise):
            s = rng.choice(irrelevant)
            session_events.append(("teach", s, "irrelevant", f"{s}_val_{session}", "tool", 0.5))

        # -- query events: fill the rest of the session's budget --
        query_targets = []
        query_targets += rng.sample(stable, k=min(2, len(stable)))
        query_targets += [s for s in changing]
        query_targets += [s for s in contradictory if contradiction_session[s] <= session]
        query_targets += [s for s in temporal]
        for s in cross_session:
            if session in cross_session_query_sessions[s]:
                query_targets.append(s)
        for cat_list, cat_name in ((stable, "stable"),):
            pass

        remaining = INTERACTIONS_PER_SESSION - len(session_events)
        # pad/trim query_targets to exactly fill the session
        while len(query_targets) < remaining:
            query_targets.append(rng.choice(stable + changing))
        query_targets = query_targets[:remaining]

        category_by_subject = {}
        for s in stable: category_by_subject[s] = "stable"
        for s in changing: category_by_subject[s] = "changing"
        for s in contradictory: category_by_subject[s] = "contradictory"
        for s in temporal: category_by_subject[s] = "temporal"
        for s in cross_session: category_by_subject[s] = "cross_session"

        for i, (kind, subj, cat, val, src, conf) in enumerate(session_events):
            first = reset_this_session and i == 0
            teach(session, subj, cat, val, src, conf, reset=first)
        for i, subj in enumerate(query_targets):
            first = reset_this_session and len(session_events) == 0 and i == 0
            query(session, subj, category_by_subject.get(subj, "stable"), reset=first)

    # ---- feed every teach event into the ground-truth ledger, using
    # interaction_id as the validity axis (Section 5: "known ground truth") ----
    for spec in specs:
        if spec.kind != "teach":
            continue
        if spec.category == "contradictory" and spec.source == "tool":
            continue  # a losing contradiction attempt must not move ground truth
        if spec.category == "irrelevant":
            continue  # noise has no ground truth by design
        ledger.add_fact(spec.subject, spec.predicate, spec.value, valid_from=spec.interaction_id, source=spec.source)

    ledger.assert_no_overlaps()
    return specs, ledger


def workload_hash(specs: list[InteractionSpec]) -> str:
    """Deterministic content hash of a built workload (adversarial-audit
    Section 6/14: "record the workload hash; verify all conditions use the
    same hash"). Used by e8_long_running_1000.py to prove, not merely
    assume, that every condition ran against the byte-identical
    interaction sequence -- since all four conditions iterate the SAME
    `specs` list from one build_workload() call, this is true by
    construction today, but the hash makes that a checkable fact in the
    saved manifest rather than an assumption a future refactor could
    silently break.
    """
    import hashlib
    import json as _json
    payload = _json.dumps([asdict(s) for s in specs], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_workload(specs: list[InteractionSpec], path: Path) -> None:
    path.write_text(__import__("json").dumps([asdict(s) for s in specs], indent=2))


def load_workload(path: Path) -> list[InteractionSpec]:
    import json
    data = json.loads(path.read_text())
    return [InteractionSpec(**d) for d in data]
