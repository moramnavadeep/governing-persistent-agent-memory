"""AISOS SDK tool-management layer: registration, lookup, and safe (timeout-
guarded) execution of callable tools. New in the SDK layer -- the underlying
AISOS research codebase has no tool/function-calling concept."""
from aisos.tools.errors import ToolError, ToolExecutionError, ToolNotFoundError, ToolTimeoutError
from aisos.tools.execution import ToolResult, safe_execute
from aisos.tools.registry import DEFAULT_REGISTRY, ToolRegistry, ToolSpec, tool

__all__ = [
    "ToolError", "ToolExecutionError", "ToolNotFoundError", "ToolTimeoutError",
    "ToolResult", "safe_execute",
    "DEFAULT_REGISTRY", "ToolRegistry", "ToolSpec", "tool",
]
