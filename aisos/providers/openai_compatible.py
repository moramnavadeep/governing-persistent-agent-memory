"""A generic ModelBackend for any server that speaks the OpenAI
chat-completions HTTP shape: OpenAI itself, Groq, vLLM's OpenAI-compatible
server, LM Studio, Together, Anyscale, and most self-hosted inference
servers. One class covers all of these via the `base_url` override, rather
than a bespoke class per vendor -- they share one request/response shape
(POST {base_url}/chat/completions, `choices[0].message.content`,
`usage.prompt_tokens`/`usage.completion_tokens`), so duplicating that shape
per vendor would only add maintenance surface without adding capability.

Anthropic does NOT fit this shape (different endpoint, different auth
header, `system` as a top-level field rather than a message, a
content-blocks response) and is implemented separately in
aisos/providers/anthropic_backend.py.

This module imports, but never modifies, aisos.agent.model_backend -- that
module is part of the frozen research codebase (aisos/agent/) that the
existing test suite and experiment scripts depend on unmodified.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
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


@dataclass
class OpenAICompatibleBackend(ModelBackend):
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: Optional[str] = None
    timeout_s: float = 60.0
    temperature: float = 0.0
    extra_headers: dict = field(default_factory=dict)
    #: Optional EventEmitter. None means the process-wide default,
    #: which emits nothing unless the application attached a sink.
    emitter: Optional[object] = None

    @property
    def model_id(self) -> str:
        return f"openai-compatible:{self.model}"

    @property
    def _provider_label(self) -> str:
        """Human-readable provider name for error messages. Derived from
        base_url so a Groq/vLLM/LM-Studio failure doesn't say "openai"."""
        host = self.base_url.split("//")[-1].split("/")[0]
        return f"openai-compatible[{host}]"

    def generate(self, prompt: str, system: str = "") -> str:
        return self._call(prompt, system).text

    def generate_with_usage(self, prompt: str, system: str = "") -> GenerationResult:
        return self._call(prompt, system)

    def _resolved_api_key(self) -> Optional[str]:
        # Never read at construction time -- only inside a real call -- so
        # that `OpenAICompatibleBackend(model="gpt-4o-mini")` is safely
        # constructible (and importable via `import aisos`) with zero
        # environment variables set. Matches OllamaBackend's existing
        # lazy-request pattern.
        return self.api_key or os.environ.get(self.api_key_env)

    def _call(self, prompt: str, system: str = "") -> GenerationResult:
        with provider_call(self._provider_label, self.emitter, self.model):
            return self._call_impl(prompt, system)

    def _call_impl(self, prompt: str, system: str = "") -> GenerationResult:
        key = self._resolved_api_key()
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature}
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.base_url.rstrip('/')}/chat/completions", json=payload, headers=headers, timeout=self.timeout_s
            )
        except requests.exceptions.Timeout as exc:
            raise ProviderTimeoutError(
                f"{self._provider_label}: request timed out after {self.timeout_s}s.",
                provider=self._provider_label,
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ProviderConnectionError(
                f"{self._provider_label}: could not reach {self.base_url}: {exc}",
                provider=self._provider_label,
            ) from exc
        latency_s = time.perf_counter() - t0

        raise_for_status(resp, provider=self._provider_label)

        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise InvalidResponseError(
                f"{self._provider_label}: 2xx response did not contain "
                f"choices[0].message.content ({type(exc).__name__}: {exc}).",
                provider=self._provider_label,
                status_code=getattr(resp, "status_code", None),
            ) from exc

        usage = data.get("usage") or {}
        return GenerationResult(
            text=text.strip(),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_s=latency_s,
            model_id=self.model_id,
        )
