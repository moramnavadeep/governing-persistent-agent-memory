"""Exceptions raised by the AISOS SDK tool-management layer (aisos/tools/).

Kept as a separate module (rather than nested inside registry.py/execution.py)
so callers can catch these without importing the registry/execution machinery
itself -- e.g. `from aisos.tools.errors import ToolTimeoutError`.
"""
from __future__ import annotations


class ToolError(Exception):
    """Base class for every error this module raises."""


class ToolNotFoundError(ToolError):
    def __init__(self, name: str):
        super().__init__(f"no tool registered under the name {name!r}")
        self.name = name


class ToolExecutionError(ToolError):
    """Wraps whatever exception the tool function itself raised, so a
    caller inspecting ToolResult.error always sees a stable, tool-scoped
    message rather than an arbitrary third-party traceback string."""

    def __init__(self, name: str, original: BaseException):
        super().__init__(f"tool {name!r} raised {type(original).__name__}: {original}")
        self.name = name
        self.original = original


class ToolTimeoutError(ToolError):
    def __init__(self, name: str, timeout_s: float):
        super().__init__(f"tool {name!r} did not return within {timeout_s}s")
        self.name = name
        self.timeout_s = timeout_s


class ToolArgumentError(ToolError):
    """The supplied keyword arguments do not match the tool's signature.

    Detected BEFORE the call is dispatched, by binding the arguments to the
    signature. Two reasons that matters: the caller gets "missing a required
    argument: query" instead of a TypeError raised from inside a worker
    thread, and a call that cannot possibly succeed never occupies a thread
    in the shared executor.
    """

    def __init__(self, name: str, detail: str):
        super().__init__(f"tool {name!r} was called with invalid arguments: {detail}")
        self.name = name
        self.detail = detail
