"""SDK runtime layer: the operational facade over a research AISOSAgent,
plus the seam a future out-of-process runtime would plug into.

    AgentRuntime   -- one agent + durable storage + events. Real, complete.
    LocalTransport -- in-process transport. Real, complete.
    RuntimeConfig  -- mode='local' works; 'subprocess'/'remote' are named
                      but deliberately unimplemented (they raise).

AISOS is library-first. There is no server here and none is planned for this
milestone; see aisos/runtime/seam.py for what a runtime would still have to
decide.
"""
from aisos.runtime.agent_runtime import AgentRuntime
from aisos.runtime.seam import AgentHandle, LocalTransport, RuntimeConfig, RuntimeTransport

__all__ = [
    "AgentRuntime",
    "RuntimeConfig", "RuntimeTransport", "LocalTransport", "AgentHandle",
]
