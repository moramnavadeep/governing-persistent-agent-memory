"""
Model backend abstraction (new master prompt Section 8: "The model should
be configurable... Start with a practical open model that can run on
available hardware").

OllamaBackend is the reference backend: a small local open-weight model
(llama3, already pulled on this machine) served by Ollama's local HTTP
API -- fully offline, no API key, no per-call cost, and reproducible given
a fixed model tag and prompt.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class GenerationResult:
    """Token-efficiency milestone (docs/token_efficiency_protocol.md): the
    forensic audit for that milestone found OllamaBackend.generate() read
    Ollama's full JSON response (which already includes prompt_eval_count/
    eval_count -- exact, model-reported token counts, not an estimate) and
    discarded everything except the answer text. This dataclass is the
    smallest fair measurement extension: it exposes those fields without
    changing generate()'s own signature or return type, so none of the
    ~19 existing call sites across every benchmark suite need to change.

    input_tokens/output_tokens are None when the backend cannot report
    them (e.g. EchoBackend, or a raise_for_status()-surviving response
    missing the field) -- callers must treat None as "not available", never
    silently coerce it to 0, per the master prompt's "do not mix exact and
    estimated counts without labeling them" rule.
    """
    text: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    latency_s: float
    model_id: str


class ModelBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        ...

    def generate_with_usage(self, prompt: str, system: str = "") -> GenerationResult:
        """Default implementation for any backend that hasn't overridden
        this: calls the existing generate() and reports token counts as
        unavailable (None), never estimated silently. Overridden by
        OllamaBackend to report real, model-reported counts."""
        t0 = time.perf_counter()
        text = self.generate(prompt, system)
        return GenerationResult(text=text, input_tokens=None, output_tokens=None,
                                 latency_s=time.perf_counter() - t0, model_id=self.model_id)


@dataclass
class OllamaBackend(ModelBackend):
    model: str = "llama3:latest"
    host: str = "http://localhost:11434"
    timeout_s: float = 60.0
    temperature: float = 0.0  # deterministic-as-possible for reproducibility

    @property
    def model_id(self) -> str:
        return f"ollama:{self.model}"

    def generate(self, prompt: str, system: str = "") -> str:
        return self._call(prompt, system).text

    def generate_with_usage(self, prompt: str, system: str = "") -> GenerationResult:
        return self._call(prompt, system)

    def _call(self, prompt: str, system: str = "") -> GenerationResult:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        t0 = time.perf_counter()
        resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout_s)
        latency_s = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()
        # prompt_eval_count/eval_count are Ollama's own model-reported token
        # counts (from llama.cpp's tokenizer for the loaded model) -- exact
        # for this backend, not an estimate. Both are absent from the
        # response in some Ollama versions/error paths; .get() returns None
        # rather than a fabricated 0 in that case.
        return GenerationResult(
            text=data.get("response", "").strip(),
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            latency_s=latency_s,
            model_id=self.model_id,
        )


class EchoBackend(ModelBackend):
    """Deterministic no-model backend for unit tests / CI: returns the most
    relevant retrieved fact verbatim, or a fixed "I don't know" string. Lets
    the LongRunningAgent harness itself be tested without requiring Ollama
    to be installed and running -- a real dependency on external state that
    unit tests must not silently require.
    """

    def __init__(self):
        self._last_prompt = None

    @property
    def model_id(self) -> str:
        return "echo:none"

    def generate(self, prompt: str, system: str = "") -> str:
        self._last_prompt = prompt
        return "I don't know."
