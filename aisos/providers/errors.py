"""Normalized provider exceptions.

Phase A found that provider backends leaked `requests.exceptions.HTTPError`,
`KeyError` and `IndexError` through the SDK's public API, coupling callers
to the `requests` library and to each vendor's response shape. These
exceptions replace that: a caller catches `ProviderError` (or a specific
subclass) regardless of which backend or HTTP library is underneath.

The original exception is always preserved as `__cause__` (via
`raise ... from exc`) and, where an HTTP response existed, as
`.status_code` / `.body_snippet` -- normalizing the type must not destroy
the information needed to debug the actual failure.

Deliberately NOT included in Phase 2: retry/backoff behavior. We do not
claim reliability semantics that have not been tested (see
docs/sdk/phase2-proposal.md, Decision 6).
"""
from __future__ import annotations

from typing import Optional

_BODY_SNIPPET_CHARS = 500


class ProviderError(Exception):
    """Base class for every provider failure surfaced by aisos.providers."""

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        body_snippet: Optional[str] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.body_snippet = body_snippet


class AuthenticationError(ProviderError):
    """401/403 -- missing, malformed, or rejected credentials."""


class RateLimitError(ProviderError):
    """429 -- quota or rate limit exceeded."""

    def __init__(self, message: str, *, retry_after: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    """The request exceeded the backend's configured timeout_s."""


class ProviderConnectionError(ProviderError):
    """The endpoint was unreachable (DNS failure, refused connection, TLS)."""


class InvalidResponseError(ProviderError):
    """A 2xx response whose body could not be parsed into a GenerationResult
    -- malformed JSON, or missing the fields this provider's API contract
    says it returns."""


def _snippet(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    return text[:_BODY_SNIPPET_CHARS] + ("…" if len(text) > _BODY_SNIPPET_CHARS else "")


def _retry_after_seconds(headers) -> Optional[float]:
    if not headers:
        return None
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        # Retry-After may legally be an HTTP-date rather than seconds. We do
        # not parse that here -- returning None is honest ("unknown"), and
        # nothing in the SDK currently acts on this value anyway.
        return None


def raise_for_status(resp, *, provider: str) -> None:
    """Map an HTTP error response onto the taxonomy above.

    Called instead of `requests.Response.raise_for_status()`. A successful
    response returns None and the caller proceeds to parse the body.
    """
    status = getattr(resp, "status_code", None)
    # isinstance guard, not just a None check: a response object whose
    # status_code is not a real int (a loosely-built test double, or a
    # non-requests HTTP client) would otherwise raise an opaque TypeError
    # from deep inside this helper. Treat "no usable status" as "not an
    # error" and let body parsing decide -- a malformed body then surfaces
    # as InvalidResponseError, which is the accurate diagnosis.
    if not isinstance(status, int) or isinstance(status, bool) or status < 400:
        return

    body = _snippet(getattr(resp, "text", None))
    common = {"provider": provider, "status_code": status, "body_snippet": body}

    if status in (401, 403):
        raise AuthenticationError(
            f"{provider}: authentication failed (HTTP {status}). "
            "Check the API key and the api_key_env setting.",
            **common,
        )
    if status == 429:
        raise RateLimitError(
            f"{provider}: rate limit exceeded (HTTP 429).",
            retry_after=_retry_after_seconds(getattr(resp, "headers", None)),
            **common,
        )
    raise ProviderError(f"{provider}: request failed with HTTP {status}.", **common)


# ---------------------------------------------------------------------------
# Observability seam (SDK Phase E)
#
# Emission lives HERE rather than in each backend so that adding a provider
# cannot accidentally ship without error observability: a backend gets it by
# wrapping its request in provider_call(), and every ProviderError raised
# inside is reported exactly once, on its way out.
# ---------------------------------------------------------------------------
from contextlib import contextmanager  # noqa: E402  (kept beside its only user)


@contextmanager
def provider_call(provider: str, emitter=None, model: Optional[str] = None):
    """Emits PROVIDER_ERROR for any ProviderError raised inside, then
    re-raises it unchanged.

    The event carries the provider label, model id, exception type and HTTP
    status -- never the request body, never headers, and therefore never the
    Authorization header the request was built with.
    """
    from aisos.observability.emitter import DEFAULT_EMITTER
    from aisos.observability.events import EventType

    try:
        yield
    except ProviderError as exc:
        (emitter or DEFAULT_EMITTER).emit(
            EventType.PROVIDER_ERROR,
            provider=provider,
            model=model,
            error_type=type(exc).__name__,
            status_code=exc.status_code,
            error=str(exc),
        )
        raise
