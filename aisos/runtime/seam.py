"""Runtime seam (Phase N) -- interfaces for a future out-of-process runtime.

READ THIS BEFORE ADDING ANYTHING HERE
    No remote runtime exists. There is no server, no scheduler daemon, no
    control plane and no network transport in AISOS, and this module does
    not add one. What it adds is the smallest set of names and shapes that
    lets one be added later WITHOUT redesigning the library: a config
    object, a transport protocol, and a handle type.

    The reason to write this now rather than later is narrow and specific.
    If application code calls AgentRuntime directly everywhere, moving an
    agent out of process later means changing every call site. If it calls
    through a transport, the local implementation is a pass-through today
    and a remote implementation is a drop-in tomorrow. That is the entire
    value being claimed here -- nothing about distribution, scaling or
    availability is implemented or promised.

    LocalTransport is therefore fully working and genuinely useful (it is
    what `aisos status` and the canonical example use). Anything remote is
    an unimplemented protocol, and calling for it raises NotImplementedError
    with a pointer to this docstring rather than failing obscurely.

WHAT A FUTURE RUNTIME WOULD HAVE TO DECIDE, AND HAS NOT
    * Isolation: thread, subprocess or container per agent.
    * Transport: in-process call, local socket, or HTTP.
    * Ownership of durable storage: does the runtime own the SQLite file, or
      does each agent?
    * Failure semantics when an agent process dies mid-cycle. The research
      core's copy-on-write store means an uncommitted cycle is simply lost,
      which is the correct behavior -- but nothing currently supervises or
      restarts such a process.
    None of these are decided here. Choosing them is the work of building
    the runtime, not of leaving a seam for it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from aisos.runtime.agent_runtime import AgentRuntime


@dataclass
class RuntimeConfig:
    """Where and how an agent runs.

    `mode` is "local" today. "subprocess" and "remote" are named so the
    vocabulary is stable, and are explicitly not implemented -- a transport
    that does not exist should say so, not be quietly absent.
    """

    mode: str = "local"
    #: Storage location a runtime should use for its agents. None means each
    #: AgentRuntime brings its own backend.
    storage_path: Optional[str] = None
    #: Reserved for a future remote transport. Unused by LocalTransport.
    endpoint: Optional[str] = None
    options: dict = field(default_factory=dict)

    SUPPORTED_MODES = ("local",)
    PLANNED_MODES = ("subprocess", "remote")

    def __post_init__(self) -> None:
        if self.mode in self.PLANNED_MODES:
            raise NotImplementedError(
                f"RuntimeConfig(mode={self.mode!r}) is a planned mode with no implementation. "
                "AISOS is library-first: only mode='local' exists today. See "
                "aisos/runtime/seam.py and docs/sdk/roadmap.md."
            )
        if self.mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"unknown runtime mode {self.mode!r}; supported: {list(self.SUPPORTED_MODES)}, "
                f"planned: {list(self.PLANNED_MODES)}"
            )


@dataclass
class AgentHandle:
    """A reference to a running agent.

    Locally this simply carries the AgentRuntime. Remotely it would carry an
    identifier and no object -- which is exactly why callers should go
    through the handle rather than holding an AgentRuntime directly.
    """

    agent_id: str
    runtime: Optional[AgentRuntime] = None

    @property
    def is_local(self) -> bool:
        return self.runtime is not None


@runtime_checkable
class RuntimeTransport(Protocol):
    """How a caller reaches an agent. One method per verb the SDK exposes."""

    def spawn(self, agent_id: str, **kwargs: Any) -> AgentHandle: ...
    def remember(self, handle: AgentHandle, subject: str, value: str, **kwargs: Any) -> Any: ...
    def sleep(self, handle: AgentHandle, **kwargs: Any) -> Any: ...
    def status(self, handle: AgentHandle) -> dict: ...
    def shutdown(self, handle: AgentHandle, **kwargs: Any) -> None: ...


class LocalTransport:
    """In-process transport: every call is a direct method call.

    This is a real, complete implementation, not a stub. It is also
    deliberately thin -- it adds no queuing, no retry and no supervision,
    because adding those in-process would be inventing semantics that a
    future real runtime would then have to match.
    """

    def __init__(self, config: Optional[RuntimeConfig] = None, storage_factory=None):
        self.config = config or RuntimeConfig()
        #: Callable taking agent_id and returning a StorageBackend, or None
        #: for non-durable agents. Injected so the transport does not decide
        #: the storage policy.
        self.storage_factory = storage_factory
        self._handles: dict[str, AgentHandle] = {}

    def spawn(self, agent_id: str, **kwargs: Any) -> AgentHandle:
        if agent_id in self._handles:
            raise ValueError(f"an agent named {agent_id!r} is already running")
        storage = kwargs.pop("storage", None)
        if storage is None and self.storage_factory is not None:
            storage = self.storage_factory(agent_id)
        runtime = AgentRuntime(agent_id=agent_id, storage=storage, **kwargs)
        handle = AgentHandle(agent_id=agent_id, runtime=runtime)
        self._handles[agent_id] = handle
        return handle

    def get(self, agent_id: str) -> Optional[AgentHandle]:
        return self._handles.get(agent_id)

    def list_agents(self) -> list[str]:
        return sorted(self._handles)

    def remember(self, handle: AgentHandle, subject: str, value: str, **kwargs: Any):
        return self._runtime(handle).remember(subject, value, **kwargs)

    def wake(self, handle: AgentHandle, **kwargs: Any) -> bool:
        return self._runtime(handle).wake(**kwargs)

    def sleep(self, handle: AgentHandle, **kwargs: Any):
        return self._runtime(handle).sleep(**kwargs)

    def status(self, handle: AgentHandle) -> dict:
        return self._runtime(handle).status()

    def shutdown(self, handle: AgentHandle, **kwargs: Any) -> None:
        self._runtime(handle).shutdown(**kwargs)
        self._handles.pop(handle.agent_id, None)

    def shutdown_all(self, **kwargs: Any) -> None:
        for handle in list(self._handles.values()):
            self.shutdown(handle, **kwargs)

    @staticmethod
    def _runtime(handle: AgentHandle) -> AgentRuntime:
        if handle.runtime is None:
            raise NotImplementedError(
                f"AgentHandle({handle.agent_id!r}) has no local runtime. Remote transports "
                "are not implemented -- see aisos/runtime/seam.py."
            )
        return handle.runtime
