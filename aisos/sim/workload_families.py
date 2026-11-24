"""
W1-W9 Workload Families (AISOS_Post_Security_Closure_Generalization_
Milestone.md Phase 3).

Confirmed gap (see docs/generalization_experiment_plan.md Phase 1 audit):
no W1-W9 family concept existed anywhere in this codebase before this
module. `aisos/sim/workload.py::generate_workload()` is one parameterized
symbolic-event generator (duplicate/contradiction/drift rates only, no
authority-authenticity or category concept). `aisos/agent/workload_1000.py::
build_workload()` has the right SHAPE (sessions, context resets, 7
categories, a paired GroundTruthLedger) but every count is a hardcoded
module constant, not a parameter -- it is exactly ONE fixed workload, not a
family.

This module generalizes workload_1000.py's pattern into a single
parameterized generator (`build_family_workload`) driven by a `FamilyConfig`
dataclass, instantiated nine times (W1-W9) with different, pre-registered
parameter values (Phase 3: "choose workloads before seeing results" -- the
FAMILIES dict below is fixed BEFORE any experiment in this milestone runs).

Deliberately additive: aisos/agent/workload_1000.py is NOT modified (Phase
2: "do not redesign the mechanism" extends, in spirit, to not touching the
already-validated 1,000-interaction pilot workload either) -- this is a new
module reusing the same InteractionSpec-like shape and GroundTruthLedger,
not a fork of existing code.

Authority-authenticity modeling (Phase 4's explicit "authority strength" vs
"authority authenticity" distinction): each teach-type FamilySpec carries a
`credential_action` describing what the RUNNER should do at execution time
(runner-side, not generator-side, since credentials are runtime objects
bound to a specific backend's TrustedAuthorityRegistry, not serializable
workload data):
    "none"          -- bare self-declared source, no credential offered
                       (the ONLY behavior workload_1000.py ever exercised)
    "issue"         -- runner issues a genuine credential for this event's
                       claimed source and attaches it
    "spoof"         -- claims a HIGH-authority source string but the runner
                       deliberately does NOT attach any credential (tests
                       whether claimed authority alone, unauthenticated,
                       is treated as a security boundary -- it must not be)
    "replay_prior"  -- runner reuses (replays) the credential originally
                       issued for the FamilySpec whose interaction_id is in
                       `replay_source_interaction_id`
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from aisos.agent.ground_truth import GroundTruthLedger


@dataclass
class FamilySpec:
    interaction_id: int
    session_id: int
    context_reset_before: bool
    kind: str  # "teach" | "query"
    category: str
    subject: str
    predicate: str
    value: Optional[str] = None
    query_text: Optional[str] = None
    source: str = "user"
    confidence: float = 0.85
    is_attack: bool = False
    credential_action: str = "none"  # "none" | "issue" | "spoof" | "replay_prior"
    replay_source_interaction_id: Optional[int] = None


@dataclass
class FamilyConfig:
    """One row per required Phase-3 documentation field."""
    name: str
    purpose: str
    n_sessions: int = 10
    interactions_per_session: int = 40
    n_stable: int = 5
    n_legit_changing: int = 4        # authenticated legitimate corrections
    n_attack_changing: int = 4       # malicious low-authority contradiction attempts
    n_repeated: int = 2
    n_irrelevant: int = 6
    n_temporal: int = 2
    n_cross_session: int = 2
    rapid_alternation_subjects: int = 0   # W4: subjects that flip value repeatedly within one session
    rapid_alternation_flips: int = 0
    stale_gap_sessions: int = 0           # W5: sessions between supersession and a query of the OLD value's subject
    write_density_multiplier: float = 1.0  # W6/W7: scales noise-teach count per session
    authenticated_fraction: float = 0.5    # fraction of legit-changing events that get "issue" vs "none"
    spoofed_fraction: float = 0.0          # fraction of attack-changing events that additionally "spoof"
    replay_fraction: float = 0.0           # fraction of legit-changing events replayed later ("replay_prior")
    expected_failure_modes: str = ""
    temporal_spacing: str = "one session apart (workload_1000.py baseline pace)"

    def authority_distribution(self) -> dict:
        """Intended source-label mix -- documented per Phase 3, verified
        empirically against actual generated specs in the report rather
        than only asserted here."""
        return {
            "user (stable/base teaches)": "SOURCE_AUTHORITY=0.8",
            "user_correction (legit corrections)": f"SOURCE_AUTHORITY=1.0, n={self.n_legit_changing}",
            "tool (attack attempts)": f"SOURCE_AUTHORITY=0.6, n={self.n_attack_changing}",
        }


def _q(subject: str, predicate: str) -> str:
    return f"What is the current {predicate} for {subject}?"


def build_family_workload(config: FamilyConfig, seed: int = 0) -> tuple[list[FamilySpec], GroundTruthLedger, dict]:
    """Returns (specs, ledger, empirical_metadata). empirical_metadata
    reports the ACTUAL realized category/source counts, for the
    documentation table's "verified against generated specs" requirement.
    """
    rng = random.Random(seed)
    ledger = GroundTruthLedger()
    specs: list[FamilySpec] = []
    predicate = "value"
    iid = 0

    stable = [f"{config.name}_stable_{i}" for i in range(config.n_stable)]
    legit_changing = [f"{config.name}_legit_{i}" for i in range(config.n_legit_changing)]
    attack_changing = [f"{config.name}_attack_{i}" for i in range(config.n_attack_changing)]
    repeated = [f"{config.name}_repeat_{i}" for i in range(config.n_repeated)]
    irrelevant = [f"{config.name}_noise_{i}" for i in range(config.n_irrelevant)]
    temporal = [f"{config.name}_temporal_{i}" for i in range(config.n_temporal)]
    cross_session = [f"{config.name}_cross_{i}" for i in range(config.n_cross_session)]
    rapid = [f"{config.name}_rapid_{i}" for i in range(config.rapid_alternation_subjects)]

    def teach(session, subject, category, value, source="user", confidence=0.85, reset=False,
              is_attack=False, credential_action="none", replay_from=None):
        nonlocal iid
        specs.append(FamilySpec(
            interaction_id=iid, session_id=session, context_reset_before=reset,
            kind="teach", category=category, subject=subject, predicate=predicate,
            value=value, source=source, confidence=confidence, is_attack=is_attack,
            credential_action=credential_action, replay_source_interaction_id=replay_from,
        ))
        iid += 1
        return iid - 1

    def query(session, subject, category, reset=False):
        nonlocal iid
        specs.append(FamilySpec(
            interaction_id=iid, session_id=session, context_reset_before=reset,
            kind="query", category=category, subject=subject, predicate=predicate,
            query_text=_q(subject, predicate),
        ))
        iid += 1

    legit_change_session = {s: rng.choice(range(1, max(2, config.n_sessions - 1))) for s in legit_changing}
    attack_session = {s: rng.choice(range(1, config.n_sessions)) for s in attack_changing}
    cross_teach_session = {s: 0 for s in cross_session}
    cross_query_sessions = {
        s: sorted(rng.sample(range(1, config.n_sessions), k=min(2, max(1, config.n_sessions - 1))))
        for s in cross_session
    }
    temporal_window = {}
    for s in temporal:
        start = rng.choice(range(0, max(1, config.n_sessions - 2)))
        temporal_window[s] = (start, min(start + 2, config.n_sessions))

    # replay bookkeeping: which legit-changing subjects get replayed, and at what later session
    replay_targets = set(rng.sample(legit_changing, k=int(len(legit_changing) * config.replay_fraction))) if legit_changing else set()
    replay_source_iid: dict[str, int] = {}

    for session in range(config.n_sessions):
        reset_this_session = session > 0
        session_events: list = []

        if session == 0:
            for s in stable:
                session_events.append(("teach", s, "stable", f"{s}_v0", "user", 0.9, False, "none"))
            for s in cross_session:
                session_events.append(("teach", s, "cross_session", f"{s}_v0", "user", 0.9, False, "none"))
            for s in repeated:
                session_events.append(("teach", s, "repeated", f"{s}_v0", "user", 0.85, False, "none"))
            for s in legit_changing:
                session_events.append(("teach", s, "changing", f"{s}_v0", "user", 0.85, False, "none"))
            for s in attack_changing:
                session_events.append(("teach", s, "contradictory", f"{s}_true_value", "user", 0.9, False, "none"))
            for s in rapid:
                session_events.append(("teach", s, "changing", f"{s}_v0", "user", 0.85, False, "none"))

        for s in legit_changing:
            if legit_change_session[s] == session:
                use_credential = "issue" if rng.random() < config.authenticated_fraction else "none"
                idx = len(specs) + len(session_events)  # provisional; real iid assigned at teach() time
                session_events.append(("teach", s, "changing", f"{s}_v1", "user_correction", 0.95, False, use_credential))

        for s in attack_changing:
            if attack_session[s] == session:
                spoof = "spoof" if rng.random() < config.spoofed_fraction else "none"
                claimed_source = "user_correction" if spoof == "spoof" else "tool"
                session_events.append(("teach", s, "contradictory", f"{s}_FALSE_value", claimed_source, 0.6, True, spoof))

        for s in repeated:
            if session > 0 and rng.random() < 0.3:
                session_events.append(("teach", s, "repeated", f"{s}_v0", "user", 0.8, False, "none"))

        for s in temporal:
            win_start, win_end = temporal_window[s]
            if session == win_start:
                session_events.append(("teach", s, "temporal", f"{s}_active", "user", 0.9, False, "none"))
            if session == win_end:
                session_events.append(("teach", s, "temporal", f"{s}_expired", "user_correction", 0.9, False, "none"))

        # W4: rapid alternation -- multiple flips of the SAME subject within one session
        if rapid and config.rapid_alternation_flips and session == max(1, config.n_sessions // 2):
            for s in rapid:
                for flip in range(config.rapid_alternation_flips):
                    src = "tool" if flip % 2 == 0 else "user_correction"
                    conf = 0.6 if src == "tool" else 0.9
                    session_events.append(("teach", s, "changing", f"{s}_flip{flip}", src, conf, src == "tool", "none"))

        n_noise = max(1, int(2 * config.write_density_multiplier))
        for _ in range(n_noise):
            if not irrelevant:
                break
            s = rng.choice(irrelevant)
            session_events.append(("teach", s, "irrelevant", f"{s}_val_{session}", "tool", 0.5, False, "none"))

        query_targets = []
        query_targets += rng.sample(stable, k=min(2, len(stable))) if stable else []
        query_targets += list(legit_changing)
        query_targets += [s for s in attack_changing if attack_session[s] <= session]
        query_targets += list(temporal)
        query_targets += list(rapid)
        for s in cross_session:
            if session in cross_query_sessions[s]:
                query_targets.append(s)
        # W5: stale-gap queries -- query a superseded subject's OLD identity
        # long after its legit correction, to test STALE-vs-CORRECT recognition
        if config.stale_gap_sessions and legit_changing:
            for s in legit_changing:
                if session == legit_change_session[s] + config.stale_gap_sessions:
                    query_targets.append(s)

        budget = int(config.interactions_per_session * max(1.0, 1.0))
        remaining = budget - len(session_events)
        while len(query_targets) < max(0, remaining) and (stable or legit_changing):
            query_targets.append(rng.choice((stable or legit_changing) + list(legit_changing)))
        query_targets = query_targets[:max(0, remaining)]

        category_by_subject = {}
        for s in stable: category_by_subject[s] = "stable"
        for s in legit_changing: category_by_subject[s] = "changing"
        for s in attack_changing: category_by_subject[s] = "contradictory"
        for s in temporal: category_by_subject[s] = "temporal"
        for s in cross_session: category_by_subject[s] = "cross_session"
        for s in rapid: category_by_subject[s] = "changing"

        for i, ev in enumerate(session_events):
            kind, subj, cat, val, src, conf, is_atk, cred_action = ev
            first = reset_this_session and i == 0
            new_iid = teach(session, subj, cat, val, src, conf, first, is_atk, cred_action)
            if subj in replay_targets and cat == "changing" and not is_atk and cred_action == "issue":
                replay_source_iid[subj] = new_iid

        for i, subj in enumerate(query_targets):
            first = reset_this_session and len(session_events) == 0 and i == 0
            query(session, subj, category_by_subject.get(subj, "stable"), reset=first)

    # W-replay: append late-session replay-attempt teach events for targeted subjects
    for s in replay_targets:
        if s not in replay_source_iid:
            continue
        replay_session = config.n_sessions - 1
        teach(replay_session, s, "changing", f"{s}_v1", "user_correction", 0.95, False, False, "replay_prior",
              replay_from=replay_source_iid[s])

    for spec in specs:
        if spec.kind != "teach":
            continue
        if spec.category == "contradictory" and spec.is_attack:
            continue  # a losing attack attempt must not move ground truth
        if spec.category == "irrelevant":
            continue
        if spec.credential_action == "replay_prior":
            continue  # a replay attempt must not move ground truth either -- it should be rejected
        ledger.add_fact(spec.subject, spec.predicate, spec.value, valid_from=spec.interaction_id, source=spec.source)

    ledger.assert_no_overlaps()

    empirical = {
        "n_interactions": len(specs),
        "n_teach": sum(1 for s in specs if s.kind == "teach"),
        "n_query": sum(1 for s in specs if s.kind == "query"),
        "n_sessions": config.n_sessions,
        "n_resets": sum(1 for s in specs if s.context_reset_before),
        "category_counts": {c: sum(1 for s in specs if s.category == c) for c in
                             {s.category for s in specs}},
        "source_counts": {src: sum(1 for s in specs if s.kind == "teach" and s.source == src) for src in
                           {s.source for s in specs if s.kind == "teach"}},
        "n_attacks": sum(1 for s in specs if s.is_attack),
        "n_legit_corrections": sum(1 for s in specs if s.category == "changing" and not s.is_attack and s.credential_action != "replay_prior"),
        "n_spoof_attempts": sum(1 for s in specs if s.credential_action == "spoof"),
        "n_replay_attempts": sum(1 for s in specs if s.credential_action == "replay_prior"),
        "n_authenticated_issues": sum(1 for s in specs if s.credential_action == "issue"),
    }
    return specs, ledger, empirical


def workload_hash(specs: list[FamilySpec]) -> str:
    import hashlib
    import json as _json
    from dataclasses import asdict
    payload = _json.dumps([asdict(s) for s in specs], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


FAMILIES: dict[str, FamilyConfig] = {
    "W1": FamilyConfig(
        name="W1", purpose="Stable-facts baseline: near-zero contradiction/attack rate, confirms no condition introduces false instability against uncontested facts.",
        n_sessions=8, interactions_per_session=30, n_stable=10, n_legit_changing=0, n_attack_changing=0,
        n_repeated=3, n_irrelevant=6, n_temporal=0, n_cross_session=2,
        expected_failure_modes="None expected -- any non-CORRECT verdict on a stable-category query not explained by a genuine reset/staleness window is a bug, not an acceptable tradeoff.",
    ),
    "W2": FamilyConfig(
        name="W2", purpose="Legitimate corrections only (no attacks): measures the pure freshness cost of adopting genuine authenticated updates with zero adversarial pressure.",
        n_sessions=10, interactions_per_session=35, n_stable=4, n_legit_changing=8, n_attack_changing=0,
        n_repeated=2, n_irrelevant=5, n_temporal=2, n_cross_session=2, authenticated_fraction=0.7,
        expected_failure_modes="A legitimate correction adopted late (STALE verdicts persisting past the correction's teach interaction) would be the freshness-cost signature this workload is designed to surface.",
    ),
    "W3": FamilyConfig(
        name="W3", purpose="Malicious contradictions only (no legitimate corrections): pure safety stress test isolating dangerous_contradiction_error_rate from any freshness confound.",
        n_sessions=10, interactions_per_session=35, n_stable=4, n_legit_changing=0, n_attack_changing=8,
        n_repeated=2, n_irrelevant=5, n_temporal=0, n_cross_session=0, spoofed_fraction=0.25,
        expected_failure_modes="A dangerous_contradiction_error_rate above 0 on this workload directly falsifies H-G1 (safety generalization) for any condition that shows it.",
    ),
    "W4": FamilyConfig(
        name="W4", purpose="Rapid alternating updates: stresses recency/tie-breaking tie-break logic with fast same-subject oscillation mixing legitimate and attack sources within one session.",
        n_sessions=8, interactions_per_session=30, n_stable=3, n_legit_changing=2, n_attack_changing=2,
        n_repeated=1, n_irrelevant=4, n_temporal=0, n_cross_session=0,
        rapid_alternation_subjects=3, rapid_alternation_flips=6, temporal_spacing="sub-session, adjacent interaction ids (as fast as the workload can express)",
        expected_failure_modes="A pure-recency mechanism (or one where trust_tier/authority mis-orders) is expected to be maximally vulnerable here -- this workload specifically targets H-G4 (tie-breaking).",
    ),
    "W5": FamilyConfig(
        name="W5", purpose="Stale information: long gap between a legitimate supersession and a later query of the pre-supersession identity, testing STALE-vs-CORRECT recognition over time.",
        n_sessions=12, interactions_per_session=30, n_stable=4, n_legit_changing=5, n_attack_changing=1,
        n_repeated=2, n_irrelevant=5, n_temporal=2, n_cross_session=2, stale_gap_sessions=5,
        temporal_spacing="up to 5 sessions between supersession and the stale-recognition query",
        expected_failure_modes="A CORRECT (not STALE) verdict on the stale-gap query would indicate the store never actually held the outdated value long enough to test staleness recognition at all -- a workload-construction risk documented and checked, not a mechanism failure.",
    ),
    "W6": FamilyConfig(
        name="W6", purpose="High-frequency writes: heavy write density per session, stresses promotion latency and scheduler behavior under write pressure.",
        n_sessions=10, interactions_per_session=45, n_stable=5, n_legit_changing=4, n_attack_changing=4,
        n_repeated=4, n_irrelevant=10, n_temporal=1, n_cross_session=1, write_density_multiplier=3.0,
        expected_failure_modes="Delayed promotion (higher promotion latency, more interactions spent at lower trust tiers) is an EXPECTED cost here, not automatically a failure -- distinguished from a dangerous safety failure explicitly in the report.",
    ),
    "W7": FamilyConfig(
        name="W7", purpose="Low-frequency writes: sparse writes with long gaps between them, testing whether infrequent maintenance still promotes/protects correctly.",
        n_sessions=14, interactions_per_session=20, n_stable=4, n_legit_changing=3, n_attack_changing=3,
        n_repeated=1, n_irrelevant=2, n_temporal=1, n_cross_session=2, write_density_multiplier=0.4,
        expected_failure_modes="A candidate that never accumulates enough consolidation cycles to promote (perpetually low trust tier) despite being legitimate would be this workload's freshness-cost signature.",
    ),
    "W8": FamilyConfig(
        name="W8", purpose="Long context-reset sequences: many short sessions (heavy reset frequency), emphasizing cross-session recall robustness.",
        n_sessions=20, interactions_per_session=15, n_stable=3, n_legit_changing=2, n_attack_changing=2,
        n_repeated=1, n_irrelevant=2, n_temporal=1, n_cross_session=6,
        expected_failure_modes="A cross-session query answered from context_buffer noise rather than the durable store (i.e. surviving a reset only by luck) would be this workload's specific target.",
    ),
    "W9": FamilyConfig(
        name="W9", purpose="Mixed/adversarial workload: the combined stress test -- legitimate corrections, attacks, rapid alternation, staleness, mixed write density, long resets, spoofing, and replay all at once. Closest in spirit to the original workload_1000.py but deliberately maximally heterogeneous.",
        n_sessions=16, interactions_per_session=35, n_stable=5, n_legit_changing=6, n_attack_changing=6,
        n_repeated=3, n_irrelevant=8, n_temporal=3, n_cross_session=3,
        rapid_alternation_subjects=2, rapid_alternation_flips=4, stale_gap_sessions=4,
        write_density_multiplier=1.5, authenticated_fraction=0.6, spoofed_fraction=0.3, replay_fraction=0.3,
        expected_failure_modes="Any single failure mode from W1-W8 could appear here; this workload's purpose is confirming no NEW failure mode emerges purely from combining previously-isolated stressors.",
    ),
}
