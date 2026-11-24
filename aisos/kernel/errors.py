"""Exceptions raised by the AISOS SDK kernel.

Phase A found the Kernel raising a bare KeyError for an unknown session
name, which is indistinguishable from a KeyError raised by the caller's own
dict and tells them nothing about what went wrong. These are the typed
replacements.
"""
from __future__ import annotations


class KernelError(Exception):
    """Base class for every error the kernel raises."""


class UnknownSessionError(KernelError, KeyError):
    """No session is registered under that name.

    Also subclasses KeyError so that pre-existing callers catching KeyError
    (the old behavior) keep working -- normalizing an exception type must
    not silently break someone's error handling.
    """

    def __init__(self, name: str, known=()):
        known_names = sorted(known)
        detail = f"registered sessions: {known_names}" if known_names else "no sessions are registered"
        # KeyError's own __str__ repr()s its argument, so build the message
        # via Exception rather than relying on KeyError formatting.
        KernelError.__init__(self, f"no agent session named {name!r} ({detail})")
        self.name = name
        self.known = known_names

    def __str__(self) -> str:
        return KernelError.__str__(self)


class DuplicateSessionError(KernelError, ValueError):
    """A session with that name is already registered.

    Also subclasses ValueError because register_agent() raised ValueError
    before this type existed.
    """

    def __init__(self, name: str):
        KernelError.__init__(self, f"a session named {name!r} is already registered")
        self.name = name

    def __str__(self) -> str:
        return KernelError.__str__(self)


class InvalidWorkError(KernelError, ValueError):
    """A submitted work item is malformed (e.g. an empty subject).

    Rejected at submit() time rather than at run_turn() time: a caller that
    submits nonsense should find out at the call that was wrong, not several
    turns later in a scheduling loop.
    """

    def __init__(self, detail: str):
        KernelError.__init__(self, f"invalid work item: {detail}")

    def __str__(self) -> str:
        return KernelError.__str__(self)
