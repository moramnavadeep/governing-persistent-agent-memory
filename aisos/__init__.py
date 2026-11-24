"""
AISOS - AI Sleep Operating System
==================================

Reference implementation of the AISOS runtime/SDK described in:
  - AISOS Technical System Specification v1 (implementation-oriented spec)
  - AISOS v2: Research Proposal with Formal Lifecycle Model

This package implements:
  - The lifecycle finite state machine (Section 5.1 of the formal model)
  - Invariants I1-I6 (Section 5.2, 5.10) enforced by construction
  - Algorithms A1-A8 (Section 6 of the formal model)
  - A competently-implemented continuous-MVCC baseline (required by Section 9,
    Phase 2, for a fair test of H1 vs H0)
  - A synthetic multi-day workload generator and experiment harness (Section 7)

It is a *software reference model*, not a production system: LLM calls are
replaced by fast deterministic/stochastic surrogates so that the lifecycle,
concurrency, and safety mechanisms can be exercised and measured directly,
exactly as Section 1.2 of the v2 document says AISOS itself does not claim a
novel consolidation/forgetting algorithm -- it orchestrates pluggable ones.

SDK layer (new, additive)
--------------------------
The symbols re-exported below add a cross-agent task scheduler (Kernel), a
tool-management layer, and multi-LLM-provider backends on top of the
research core described above. Unlike the core, this layer is NEW
engineering scope -- it has not been subjected to the causal experiments,
ablations, or adversarial benchmarks the research core has (see the
companion papers and aisos/tests/). Every symbol below is safely
importable/constructible with zero live network calls, zero API keys, and
zero optional third-party packages installed -- provider backends only
touch the network inside generate()/_call(), never at construction, and
framework adapters (e.g. aisos.adapters.langchain_memory) are intentionally
NOT re-exported here since their dependency (langchain-core) is optional;
import them directly if you've installed that extra.
"""

__version__ = "0.1.0"

from aisos.core.adapter import AgentAdapterAPI, AISOSAgent
from aisos.kernel.kernel import AgentSession, Kernel, PriorityQueuePolicy, RoundRobinPolicy
from aisos.tools.registry import DEFAULT_REGISTRY, ToolRegistry, tool
from aisos.agent.model_backend import EchoBackend, GenerationResult, ModelBackend, OllamaBackend
from aisos.providers.anthropic_backend import AnthropicBackend
from aisos.providers.openai_compatible import OpenAICompatibleBackend
from aisos.providers.registry import backend_from_profile
from aisos.observability import (
    CallbackSink,
    Event,
    EventEmitter,
    EventType,
    InMemorySink,
    LoggingSink,
)
from aisos.runtime import AgentRuntime, LocalTransport, RuntimeConfig
from aisos.storage import (
    InMemoryStorage,
    SQLiteStorage,
    StorageBackend,
    create_snapshot,
    restore_snapshot,
)

__all__ = [
    "__version__",
    "AISOSAgent", "AgentAdapterAPI",
    "Kernel", "AgentSession", "RoundRobinPolicy", "PriorityQueuePolicy",
    "ToolRegistry", "tool", "DEFAULT_REGISTRY",
    "ModelBackend", "OllamaBackend", "EchoBackend", "GenerationResult",
    "OpenAICompatibleBackend", "AnthropicBackend", "backend_from_profile",
    "StorageBackend", "InMemoryStorage", "SQLiteStorage",
    "create_snapshot", "restore_snapshot",
    "AgentRuntime", "RuntimeConfig", "LocalTransport",
    "EventEmitter", "Event", "EventType",
    "InMemorySink", "LoggingSink", "CallbackSink",
]
