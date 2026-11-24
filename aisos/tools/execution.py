"""Safe execution of a registered tool call, with a real (best-effort)
timeout.

Timeout implementation note: this uses concurrent.futures.ThreadPoolExecutor
rather than signal.alarm(), because signal.alarm() does not exist on Windows
(this environment is win32) and the SDK must work there. The tradeoff,
documented here and in docs/sdk/tools.md, is that ThreadPoolExecutor can only
stop *waiting* on a hung thread -- it cannot forcibly kill it, so a tool
function that ignores its own timeout and blocks forever will leak a
background thread. This is an accepted v0.1 limitation: it is sufficient for
the common case (a tool making a network call or running a subprocess that
respects its own timeout), and a true kill would require subprocess-level
isolation, which is out of scope here.
"""
from __future__ import annotations

import inspect
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Optional

from aisos.tools.errors import ToolArgumentError, ToolExecutionError, ToolTimeoutError

# Module-level, lazily sized executor shared by every ToolRegistry instance in
# this process. A dedicated small pool (not the default which scales with CPU
# count) keeps a runaway tool call from starving unrelated work.
_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="aisos-tool")


@dataclass
class ToolResult:
    name: str
    ok: bool
    value: Any = None
    error: Optional[str] = None
    latency_s: float = 0.0


def safe_execute(spec, kwargs: dict) -> ToolResult:
    """Runs spec.fn(**kwargs) under spec.timeout_s, converting every failure
    mode (bad arguments, an exception raised inside the tool, or a timeout)
    into a ToolResult rather than letting it propagate -- callers that want
    to distinguish failure kinds inspect `.ok`/`.error`, they never need a
    try/except around this call itself."""
    t0 = time.perf_counter()

    # Bind first: a call that cannot possibly succeed should fail here, with
    # a message naming the actual problem, rather than raising a TypeError
    # from inside a worker thread (where it would be reported as a generic
    # "tool raised TypeError") and occupying a slot in the shared executor.
    try:
        inspect.signature(spec.fn).bind(**kwargs)
    except TypeError as exc:
        err = ToolArgumentError(spec.name, str(exc))
        return ToolResult(name=spec.name, ok=False, error=str(err),
                          latency_s=time.perf_counter() - t0)
    except (ValueError, AttributeError):
        # Some builtins and C functions have no introspectable signature.
        # Skipping validation for those is strictly better than refusing to
        # run them.
        pass

    future: Future = _EXECUTOR.submit(spec.fn, **kwargs)
    try:
        value = future.result(timeout=spec.timeout_s)
        return ToolResult(name=spec.name, ok=True, value=value, latency_s=time.perf_counter() - t0)
    except FutureTimeoutError:
        future.cancel()  # best-effort only -- see module docstring
        err = ToolTimeoutError(spec.name, spec.timeout_s)
        return ToolResult(name=spec.name, ok=False, error=str(err), latency_s=time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any tool-raised exception is wrapped, not re-raised
        err = ToolExecutionError(spec.name, exc)
        return ToolResult(name=spec.name, ok=False, error=str(err), latency_s=time.perf_counter() - t0)
