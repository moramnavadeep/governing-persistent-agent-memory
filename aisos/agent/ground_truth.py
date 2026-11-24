"""
Ground-truth fact ledger (new master prompt Section 10).

"The evaluator must know what the correct current fact is." Unlike
aisos/metrics.py::oracle_value() (a highest-authority-write oracle derived
from an event stream, used by the existing E1/E3/E4/E7 experiments and left
untouched here so those results stay reproducible), this ledger states
ground truth directly, as a set of facts each with an explicit validity
interval over interaction indices -- the FACT F001/F002 example from the
master prompt:

    F001: (Project A, database, PostgreSQL), valid_from=1, valid_until=500
    F002: (Project A, database, MongoDB),     valid_from=501, valid_until=infinity

This is the ground truth a long-running-agent experiment (aisos/agent/
long_running_agent.py) evaluates responses against: "what is the correct
answer to (subject, predicate) as of interaction N," independent of
whatever the agent's memory currently contains.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Fact:
    subject: str
    predicate: str
    value: str
    valid_from: int
    valid_until: Optional[int] = None  # None = infinity; interval is [valid_from, valid_until)
    source: str = "ground_truth"
    fact_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def is_valid_at(self, interaction: int) -> bool:
        if interaction < self.valid_from:
            return False
        if self.valid_until is not None and interaction >= self.valid_until:
            return False
        return True

    def key(self) -> tuple[str, str]:
        return (self.subject, self.predicate)


class GroundTruthLedger:
    """Append-only within each (subject, predicate) key's validity timeline.

    add_fact() automatically closes any still-open-ended prior fact for the
    same key at the new fact's valid_from, so two facts for one key never
    overlap -- this is the ledger's own invariant, checked by
    assert_no_overlaps() rather than merely hoped for.
    """

    def __init__(self):
        self._facts: list[Fact] = []

    def add_fact(
        self,
        subject: str,
        predicate: str,
        value: str,
        valid_from: int,
        valid_until: Optional[int] = None,
        source: str = "ground_truth",
    ) -> Fact:
        key = (subject, predicate)
        for f in self._facts:
            if f.key() == key and f.valid_until is None:
                f.valid_until = valid_from
        fact = Fact(
            subject=subject, predicate=predicate, value=value,
            valid_from=valid_from, valid_until=valid_until, source=source,
        )
        self._facts.append(fact)
        return fact

    def current_fact(self, subject: str, predicate: str, at_interaction: int) -> Optional[Fact]:
        for f in self._facts:
            if f.key() == (subject, predicate) and f.is_valid_at(at_interaction):
                return f
        return None

    def all_facts(
        self, subject: Optional[str] = None, predicate: Optional[str] = None, up_to_interaction: Optional[int] = None,
    ) -> list[Fact]:
        """up_to_interaction: when given, excludes any fact whose
        valid_from is strictly after it -- an evaluator scoring a query at
        interaction N must never be handed a fact that had not been
        introduced yet at N (a "future fact" leak; found during the
        adversarial audit, docs/research_log.md 2026-09-01 entry -- see
        that entry for why it did not change any already-reported result,
        since every observed STALE case in the completed pilot happened to
        match a genuinely past value, verified directly against the raw
        records rather than assumed)."""
        out = self._facts
        if subject is not None:
            out = [f for f in out if f.subject == subject]
        if predicate is not None:
            out = [f for f in out if f.predicate == predicate]
        if up_to_interaction is not None:
            out = [f for f in out if f.valid_from <= up_to_interaction]
        return list(out)

    def keys(self) -> list[tuple[str, str]]:
        seen = []
        for f in self._facts:
            if f.key() not in seen:
                seen.append(f.key())
        return seen

    def assert_no_overlaps(self) -> None:
        """Ledger self-check: for every key, sort by valid_from and confirm
        no interval overlaps the next. Raises AssertionError naming the
        offending pair, so a bad add_fact() sequence (e.g. a valid_until
        passed explicitly that lands before the next fact's valid_from was
        auto-closed against) is caught immediately rather than silently
        producing an ambiguous "current fact" later.
        """
        for key in self.keys():
            facts = sorted(self.all_facts(*key), key=lambda f: f.valid_from)
            for a, b in zip(facts, facts[1:]):
                a_end = a.valid_until if a.valid_until is not None else float("inf")
                if a_end > b.valid_from:
                    raise AssertionError(
                        f"overlapping facts for {key}: {a.fact_id} valid [{a.valid_from},{a.valid_until}) "
                        f"overlaps {b.fact_id} valid [{b.valid_from},{b.valid_until})"
                    )

    def to_workload_events(self) -> list[dict]:
        """Flattens the ledger into the event-dict shape aisos/sim/workload.py
        and aisos/metrics.py::oracle_value() already consume (subject/value/
        source/confidence/ts/seq/kind), so an existing experiment harness
        can replay a ground-truth-defined fact sequence without a rewrite.
        subject becomes "{subject}:{predicate}" to fit the flat-subject
        model the rest of the codebase uses.
        """
        events = []
        for i, f in enumerate(sorted(self._facts, key=lambda f: f.valid_from)):
            events.append({
                "seq": i, "ts": float(f.valid_from), "subject": f"{f.subject}:{f.predicate}",
                "value": f.value, "source": f.source, "confidence": 0.95,
                "domain": "ground_truth", "kind": "fresh" if i == 0 else "contradiction",
            })
        return events
