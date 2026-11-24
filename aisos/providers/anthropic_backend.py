"""Anthropic Messages API backend. Kept as its own class rather than folded
into OpenAICompatibleBackend because the wire format genuinely differs:
POST {base_url}/messages (not /chat/completions), auth via `x-api-key` +
`anthropic-version` headers (not `Authorization: Bearer`), `system` is a
top-level request field (not a message with role="system"), and the
response body is `content: [{type, text}, ...]` blocks with
`usage.input_tokens`/`usage.output_tokens` -- different field names than
OpenAI's `usage.prompt_tokens`/`usage.completion_tokens`.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import requests

from aisos.agent.model_backend import GenerationResult, ModelBackend
from aisos.providers.errors import (
    InvalidResponseError,
    ProviderConnectionError,
    ProviderTimeoutError,
    provider_call,
    raise_for_status,
)

DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
_PROVIDER_LABEL = "anthropic"


@dataclass
class AnthropicBackend(ModelBackend):
    model: str = "claude-sonnet-4-20250514"
    base_url: str = "https://api.anthropic.com/v1"
    api_key_env: str = "ANTHROPIC_API_KEY"
    api_key: Optional[str] = None
    anthropic_version: str = DEFAULT_ANTHROPIC_VERSION
    max_tokens: int = 1024
    timeout_s: float = 60.0
    temperature: float = 0.0
    #: Optional EventEmitter. None means the process-wide default,
    #: which emits nothing unless the application attached a sink.
    emitter: Optional[object] = None

    @property
    def model_id(self) -> str:
        return f"anthropic:{self.model}"

    def generate(self, prompt: str, system: str = "") -> str:
        return self._call(prompt, system).text

    def generate_with_usage(self, prompt: str, system: str = "") -> GenerationResult:
        return self._call(prompt, system)

    def _resolved_api_key(self) -> Optional[str]:
        return self.api_key or os.environ.get(self.api_key_env)

    def _call(self, prompt: str, system: str = "") -> GenerationResult:
        with provider_call(_PROVIDER_LABEL, self.emitter, self.model):
            return self._call_impl(prompt, system)

    def _call_impl(self, prompt: str, system: str = "") -> GenerationResult:
        key = self._resolved_api_key()
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.anthropic_version,
        }
        if key:
            headers["x-api-key"] = key
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.base_url.rstrip('/')}/messages", json=payload, headers=headers, timeout=self.timeout_s
            )
        except requests.exceptions.Timeout as exc:
            raise ProviderTimeoutError(
                f"{_PROVIDER_LABEL}: request timed out after {self.timeout_s}s.", provider=_PROVIDER_LABEL
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ProviderConnectionError(
                f"{_PROVIDER_LABEL}: could not reach {self.base_url}: {exc}", provider=_PROVIDER_LABEL
            ) from exc
        latency_s = time.perf_counter() - t0

        raise_for_status(resp, provider=_PROVIDER_LABEL)

        try:
            data = resp.json()
            blocks = data["content"]
            text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise InvalidResponseError(
                f"{_PROVIDER_LABEL}: 2xx response did not contain a parseable "
                f"content block list ({type(exc).__name__}: {exc}).",
                provider=_PROVIDER_LABEL,
                status_code=getattr(resp, "status_code", None),
            ) from exc

        usage = data.get("usage") or {}
        return GenerationResult(
            text=text.strip(),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            latency_s=latency_s,
            model_id=self.model_id,
        )
