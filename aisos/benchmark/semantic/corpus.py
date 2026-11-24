"""
Synthetic enterprise document corpus for the RAG-favorable semantic
benchmark (AISOS_v3_Next_Semantic_RAG_Enterprise_Benchmark_Prompt.md
Section 2). Deterministic, seeded, template-based generation (no LLM
calls) so corpora up to 50,000 documents can be built in seconds, matching
this project's established convention for large-scale workload generation
(aisos/sim/workload.py).

A CORE_TOPIC is a real-world enterprise subject (a policy, an incident, a
procedure) with a canonical current fact. Each core topic accumulates:
  - an authoritative ORIGINAL document stating the fact plainly
  - several PARAPHRASE documents (different wording/structure, same fact)
  - INDIRECT documents (mention the topic without stating the fact
    directly -- relevant for retrieval, insufficient alone for the answer)
  - for topics used in contradiction/poisoning/temporal workloads: CONFLICT
    documents (a different value, different source/authority/timestamp)
  - DISTRACTOR documents about OTHER topics, and generic FILLER documents
    with no tracked topic at all, padding the corpus to the target size.

Each document has explicit provenance metadata (source, authority,
confidence, timestamp, document ID) -- Section 2's explicit requirement.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

def safe_value(n: int, prefix: str = "V") -> str:
    """Fixed-width, prefixed value token (e.g. "V040"), NOT a bare decimal
    string. Two distinct safe_value() outputs can never be a substring of
    each other (unlike bare numbers, e.g. "5" is a literal substring of
    "50" and of every other number containing a 5) -- eliminating a real
    class of false-positive dangerous_answer/evaluator-verdict bugs found
    during this milestone's own smoke testing (a temporal-poisoning query
    whose CORRECT answer was "50" was flagged dangerous_answer=True on
    every run purely because the poisoned value "5" is a substring of
    "50"; a poison value of bare "0" collided with every other number
    containing a zero). The surrounding document text still reads
    naturally (e.g. "the vacation policy is V040 days"); only the token
    itself is encoded this way, matching this project's own established
    convention (VALUE_0, STABLE_V, TRUE_VALUE, etc.) of using
    collision-safe tokens rather than naturalistic numbers for anything
    the evaluator does substring matching against.
    """
    return f"{prefix}{n:03d}"


# A real bug, found and fixed before any benchmark result was accepted:
# an earlier version of this table ranked "policy_doc" (1.0) ABOVE
# "user_correction" (0.95) -- backwards from this project's own
# established convention (AISOS's production SOURCE_AUTHORITY table puts
# "user_correction" at the ceiling, 1.0, specifically so an explicit,
# verified correction can always supersede a stale document). With
# policy_doc ranked highest, R5/R9/R10's "changing" cases and the
# temporal-policy-revision experiment's legitimate correction could NEVER
# win against the original policy document it was correcting, regardless
# of recency or credential verification -- confirmed by direct
# reproduction (S6b's R5/R9/R10 "changing" queries all resolved to the
# OLD value). Fixed by putting "user_correction" at the ceiling, matching
# AISOS's own production convention exactly.
SOURCE_TYPES = {
    "user_correction": 1.0, "policy_doc": 0.9, "incident_report": 0.85,
    "engineering_wiki": 0.8, "meeting_notes": 0.5, "ticket": 0.4, "anonymous_edit": 0.1,
}

TOPIC_DOMAINS = [
    ("vacation_policy", "the company's annual vacation allowance", "days"),
    ("remote_work_policy", "the company's remote-work eligibility threshold", "days per month"),
    ("database_backup_procedure", "the production database backup interval", "hours"),
    ("incident_response_sla", "the P1 incident response SLA", "minutes"),
    ("code_review_requirement", "the minimum number of required code review approvals", "approvals"),
    ("expense_approval_limit", "the manager expense approval limit", "USD"),
    ("password_rotation_policy", "the mandatory password rotation interval", "days"),
    ("on_call_rotation_length", "the on-call rotation length", "days"),
    ("data_retention_policy", "the customer data retention period", "months"),
    ("deployment_freeze_window", "the pre-holiday deployment freeze window", "days"),
    ("api_rate_limit", "the public API rate limit", "requests per minute"),
    ("security_review_cadence", "the mandatory security review cadence", "months"),
    ("severity1_escalation_time", "the severity-1 escalation time", "minutes"),
    ("contractor_access_duration", "the maximum contractor system-access duration", "days"),
    ("backup_retention_period", "the backup retention period", "days"),
    ("vpn_session_timeout", "the VPN session timeout", "minutes"),
    ("incident_postmortem_deadline", "the incident postmortem submission deadline", "days"),
    ("staging_environment_refresh", "the staging environment data refresh interval", "days"),
    ("laptop_replacement_cycle", "the laptop replacement cycle", "months"),
    ("customer_sla_credit_threshold", "the customer SLA credit threshold", "USD"),
    ("penetration_test_frequency", "the mandatory penetration test frequency", "months"),
    ("log_retention_period", "the application log retention period", "days"),
    ("change_advisory_lead_time", "the change advisory board lead time", "days"),
    ("disaster_recovery_drill_frequency", "the disaster recovery drill frequency", "months"),
    ("api_key_expiry", "the API key expiry period", "days"),
    ("employee_referral_bonus", "the employee referral bonus amount", "USD"),
    ("badge_access_review_cycle", "the badge access review cycle", "months"),
]

FILLER_TOPICS = [
    "office seating chart", "coffee machine maintenance", "parking permit renewal",
    "team lunch schedule", "conference room booking", "printer toner supply",
    "holiday party planning", "gym membership reimbursement", "desk plant policy",
    "birthday celebration rotation", "kitchen cleaning rota", "bike storage rules",
]

ORIGINAL_TEMPLATES = [
    "Per official policy document {doc_id}, {topic_desc} is {value} {unit}, effective as documented by {source_label}.",
    "The current, approved value for {topic_desc} is {value} {unit}, as recorded in {source_label}.",
    "According to {source_label}, {topic_desc} has been set to {value} {unit}.",
]
PARAPHRASE_TEMPLATES = [
    "Just a reminder that {topic_desc} is {value} {unit} — please plan accordingly.",
    "For anyone asking, {topic_desc} currently stands at {value} {unit}.",
    "To confirm what was discussed: {topic_desc} = {value} {unit}.",
    "FAQ: Q: What is {topic_desc}? A: {value} {unit}.",
    "Team update — {topic_desc} remains {value} {unit} this quarter.",
]
INDIRECT_TEMPLATES = [
    "There was some discussion in the meeting about {topic_desc}, but no specific number was mentioned here.",
    "Several teams have asked questions related to {topic_desc}; see the linked policy document for specifics.",
    "This ticket references {topic_desc} as background context for an unrelated request.",
]
CONFLICT_TEMPLATES = [
    # Deliberately as confident and declarative as ORIGINAL_TEMPLATES/
    # PARAPHRASE_TEMPLATES -- a genuinely dangerous malicious or
    # low-authority document reads with the SAME conviction as a
    # legitimate one; only its provenance (source/authority metadata, not
    # surface-level hedging language) should differ. An earlier version of
    # these templates used hedged phrasing ("some believe...", "may have
    # changed..."), discovered during this milestone's own smoke testing
    # to make the embedding model rank them systematically LOWER for
    # direct factual queries than the confident legitimate documents --
    # making RAG "accidentally" resist poisoning for a reason having
    # nothing to do with governance (a template-wording artifact, not a
    # retrieval or safety property). Fixed before any real result was
    # accepted, per this project's 'investigate a surprising result before
    # accepting it' discipline.
    "Per {source_label}, {topic_desc} is {value} {unit}, effective immediately.",
    "{source_label} confirms: {topic_desc} is now {value} {unit}.",
    "Official update from {source_label} — {topic_desc} is set to {value} {unit}.",
]
FILLER_TEMPLATES = [
    "Minutes from the {filler_topic} discussion: no major changes this week.",
    "Reminder about {filler_topic} — please see the shared calendar for details.",
    "Ticket #{doc_id}: request related to {filler_topic}, resolved by facilities.",
    "Update on {filler_topic}: rescheduled to next week due to availability.",
]


@dataclass
class Document:
    doc_id: str
    text: str
    topic: Optional[str]       # the AISOS "subject" this document asserts a claim about (None = pure filler/distractor)
    value: Optional[str]       # the canonical value asserted (None if no claim)
    source: str
    authority: float
    confidence: float
    timestamp: float
    legitimate: bool = True
    relevant_for_topic: bool = True  # False for INDIRECT docs: topically relevant to retrieve, but insufficient alone to answer


def _fmt(template: str, **kw) -> str:
    return template.format(**kw)


class CorpusBuilder:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self._doc_counter = 0
        self.documents: list[Document] = []

    def _next_id(self) -> str:
        self._doc_counter += 1
        return f"doc_{self._doc_counter:07d}"

    def add_original(self, topic: str, topic_desc: str, unit: str, value: str,
                      source: str = "policy_doc", ts: float = 0.0, confidence: float = 0.95) -> Document:
        doc_id = self._next_id()
        text = _fmt(self.rng.choice(ORIGINAL_TEMPLATES), doc_id=doc_id, topic_desc=topic_desc,
                    value=value, unit=unit, source_label=source.replace("_", " "))
        doc = Document(doc_id, text, topic, value, source, SOURCE_TYPES.get(source, 0.4), confidence, ts, True, True)
        self.documents.append(doc)
        return doc

    def add_paraphrases(self, topic: str, topic_desc: str, unit: str, value: str, n: int,
                         source: str = "engineering_wiki", ts: float = 0.0, confidence: float = 0.8) -> list[Document]:
        out = []
        for _ in range(n):
            doc_id = self._next_id()
            text = _fmt(self.rng.choice(PARAPHRASE_TEMPLATES), topic_desc=topic_desc, value=value, unit=unit)
            doc = Document(doc_id, text, topic, value, source, SOURCE_TYPES.get(source, 0.4), confidence, ts, True, True)
            self.documents.append(doc)
            out.append(doc)
        return out

    def add_indirect(self, topic: str, topic_desc: str, n: int, source: str = "meeting_notes", ts: float = 0.0) -> list[Document]:
        out = []
        for _ in range(n):
            doc_id = self._next_id()
            text = _fmt(self.rng.choice(INDIRECT_TEMPLATES), topic_desc=topic_desc)
            doc = Document(doc_id, text, topic, None, source, SOURCE_TYPES.get(source, 0.4), 0.5, ts, True, False)
            self.documents.append(doc)
            out.append(doc)
        return out

    def add_conflict(self, topic: str, topic_desc: str, unit: str, value: str,
                      source: str = "anonymous_edit", ts: float = 0.0, confidence: float = 0.6,
                      legitimate: bool = False) -> Document:
        doc_id = self._next_id()
        text = _fmt(self.rng.choice(CONFLICT_TEMPLATES), topic_desc=topic_desc, value=value, unit=unit,
                    source_label=source.replace("_", " "))
        doc = Document(doc_id, text, topic, value, source, SOURCE_TYPES.get(source, 0.4), confidence, ts, legitimate, True)
        self.documents.append(doc)
        return doc

    def add_filler(self, n: int, ts: float = 0.0) -> list[Document]:
        out = []
        for _ in range(n):
            doc_id = self._next_id()
            filler_topic = self.rng.choice(FILLER_TOPICS)
            text = _fmt(self.rng.choice(FILLER_TEMPLATES), filler_topic=filler_topic, doc_id=doc_id)
            doc = Document(doc_id, text, None, None, "meeting_notes", 0.5, 0.5, ts, True, False)
            self.documents.append(doc)
            out.append(doc)
        return out


# Topic-index reservation table (must stay non-overlapping -- see
# aisos/benchmark/semantic/workloads.py and adversarial.py, whose default
# `idx`/`idx1`/`idx2`/`start_idx` parameters correspond exactly to these
# ranges). Only DECORATED_INDICES get a pre-built clean original+
# paraphrases+indirect document set from build_base_corpus() itself;
# every other reserved index is left untouched here so the owning
# workload function can construct its own document set (old/new value,
# true/false claim, temporal stages, etc.) from a blank slate without
# an unwanted extra "default" value silently coexisting alongside it.
DECORATED_INDICES = set(range(0, 3)) | set(range(10, 17))  # R1/R2/R3, R10


def build_base_corpus(target_size: int, seed: int = 0) -> tuple[CorpusBuilder, dict]:
    """Builds the shared base corpus: DECORATED_INDICES (R1/R2/R3's and
    R10's topics) each get a clean original + a few paraphrases + indirect
    mentions; every other TOPIC_DOMAINS entry is left with NO documents at
    all here (reserved for its owning workload/experiment to populate from
    scratch -- R4-R9 and the Section 11/13 adversarial experiments all
    need full control over exactly which values exist for their topic,
    which a shared default value would corrupt). Padded with filler
    documents to `target_size`. Returns (builder, topic_info) mapping
    every TOPIC_DOMAINS entry's name -> {"desc", "unit", "value"} --
    `value` is only meaningful (non-None) for DECORATED_INDICES.
    """
    cb = CorpusBuilder(seed=seed)
    topic_info = {}
    for i, (topic, desc, unit) in enumerate(TOPIC_DOMAINS):
        if i in DECORATED_INDICES:
            value = safe_value(10 + i * 3)  # distinct, collision-safe canonical value per topic
            cb.add_original(topic, desc, unit, value, ts=0.0)
            cb.add_paraphrases(topic, desc, unit, value, n=4, ts=0.0)
            cb.add_indirect(topic, desc, n=2, ts=0.0)
            topic_info[topic] = {"desc": desc, "unit": unit, "value": value}
        else:
            topic_info[topic] = {"desc": desc, "unit": unit, "value": None}

    n_so_far = len(cb.documents)
    n_filler = max(0, target_size - n_so_far)
    cb.add_filler(n_filler, ts=0.0)
    return cb, topic_info
