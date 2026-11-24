"""Tool registration and lookup for the AISOS SDK.

This is new, greenfield functionality -- the research codebase this SDK
wraps has no tool/function-calling concept anywhere (confirmed by a
repo-wide search before this module was written). Kept deliberately small:
a name -> ToolSpec map, a decorator to populate it, and a call() entry point
that always returns a ToolResult (see aisos/tools/execution.py) rather than
raising, so a calling agent loop can branch on `.ok` without a try/except.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from aisos.observability.emitter import DEFAULT_EMITTER, EventEmitter
from aisos.observability.events import EventType
from aisos.tools.errors import ToolNotFoundError
from aisos.tools.execution import ToolResult, safe_execute
from aisos.tools.schema import ToolParameter, extract_parameters, parameters_to_json_schema


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]
    timeout_s: float = 10.0
    # Legacy shape, kept for backward compatibility with Phase 1 callers:
    # {param_name: {"required": bool}}. New code should prefer `params`
    # (typed) or json_schema() (function-calling ready).
    parameters: dict = field(default_factory=dict)
    params: list[ToolParameter] = field(default_factory=list)

    def json_schema(self) -> dict:
        """OpenAI-function / JSON-Schema compatible representation of this
        tool, suitable for handing to a function-calling model or, later,
        an MCP adapter."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": parameters_to_json_schema(self.params),
        }


class ToolRegistry:
    """A namespaced collection of callable tools. Most applications can use
    the module-level DEFAULT_REGISTRY (via the `tool` decorator below); a
    ToolRegistry instance is exposed separately for cases that need more than
    one independent tool namespace -- e.g. aisos.kernel.kernel.AgentSession
    accepts a per-session ToolRegistry so different agents in one Kernel can
    have different tool access."""

    def __init__(self, emitter: Optional[EventEmitter] = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.emitter = emitter if emitter is not None else DEFAULT_EMITTER

    def register(
        self,
        name: Optional[str] = None,
        description: str = "",
        timeout_s: float = 10.0,
        param_descriptions: Optional[dict] = None,
    ):
        """Usable as a decorator: @registry.register() or
        @registry.register(name="search", description="...", timeout_s=5,
        param_descriptions={"query": "what to search for"}).

        Parameter types are derived from the function's annotations (see
        aisos/tools/schema.py); `param_descriptions` supplies the per-
        parameter prose, because docstring parsing is too format-fragile
        to be trustworthy.
        """

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or fn.__name__
            if tool_name in self._tools:
                raise ValueError(f"tool {tool_name!r} is already registered")
            typed_params = extract_parameters(fn, param_descriptions)
            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=description or (fn.__doc__ or "").strip(),
                fn=fn,
                timeout_s=timeout_s,
                # Legacy dict preserved verbatim so Phase 1 callers and
                # tests that read .parameters keep working unchanged.
                parameters={p.name: {"required": p.required} for p in typed_params},
                params=typed_params,
            )
            return fn

        return deco

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(name) from None

    def list_tools(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters, "timeout_s": t.timeout_s}
            for t in self._tools.values()
        ]

    def json_schemas(self) -> list[dict]:
        """Every registered tool as an OpenAI-function / JSON-Schema dict --
        the form a function-calling model (or a future MCP adapter) needs."""
        return [t.json_schema() for t in self._tools.values()]

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        spec = self.get(name)  # raises ToolNotFoundError if absent -- deliberately NOT wrapped in a ToolResult,
        # since "the tool doesn't exist" is a caller programming error, not a runtime tool failure.
        # Argument NAMES are emitted, never argument VALUES: a tool argument
        # is arbitrary caller data and may well be a credential.
        self.emitter.emit(EventType.TOOL_STARTED, tool=name, args=sorted(kwargs))
        result = safe_execute(spec, kwargs)
        if result.ok:
            self.emitter.emit(
                EventType.TOOL_COMPLETED, tool=name, latency_s=round(result.latency_s, 6)
            )
        else:
            self.emitter.emit(
                EventType.TOOL_FAILED, tool=name, error=result.error,
                latency_s=round(result.latency_s, 6),
            )
        return result

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


DEFAULT_REGISTRY = ToolRegistry()


def tool(
    name: Optional[str] = None,
    description: str = "",
    timeout_s: float = 10.0,
    param_descriptions: Optional[dict] = None,
):
    """Convenience decorator bound to DEFAULT_REGISTRY, for the common case
    of one shared tool namespace per process:

        @tool(description="Look up a fact by subject.")
        def lookup(subject: str) -> str:
            ...
    """
    return DEFAULT_REGISTRY.register(
        name=name, description=description, timeout_s=timeout_s, param_descriptions=param_descriptions
    )
