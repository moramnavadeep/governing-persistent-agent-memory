"""Builds a ModelBackend from a named profile in configs/providers.yaml.

Reuses aisos.config.load_yaml (does not modify it) so provider profiles
follow the same loading convention as configs/default.yaml and
configs/models.yaml. Profiles hold base_url/model/the NAME of an
environment variable to read a key from -- never a literal API key -- so
configs/providers.yaml is safe to keep in version control.
"""
from __future__ import annotations

from typing import Any

from aisos.agent.model_backend import EchoBackend, ModelBackend, OllamaBackend
from aisos.config import load_yaml
from aisos.providers.anthropic_backend import AnthropicBackend
from aisos.providers.openai_compatible import OpenAICompatibleBackend

_KIND_BUILDERS = {
    "echo": lambda p: EchoBackend(),
    "ollama": lambda p: OllamaBackend(**_only(p, "model", "host", "timeout_s", "temperature")),
    "openai": lambda p: OpenAICompatibleBackend(
        **_only(p, "model", "base_url", "api_key_env", "api_key", "timeout_s", "temperature")
    ),
    "groq": lambda p: OpenAICompatibleBackend(
        base_url=p.get("base_url", "https://api.groq.com/openai/v1"),
        **_only(p, "model", "api_key_env", "api_key", "timeout_s", "temperature"),
    ),
    "vllm": lambda p: OpenAICompatibleBackend(
        **_only(p, "model", "base_url", "api_key_env", "api_key", "timeout_s", "temperature")
    ),
    "anthropic": lambda p: AnthropicBackend(
        **_only(p, "model", "base_url", "api_key_env", "api_key", "anthropic_version", "max_tokens", "timeout_s", "temperature")
    ),
}


def _only(profile: dict, *keys: str) -> dict:
    return {k: profile[k] for k in keys if k in profile}


def backend_from_profile(profile: str, config_path: str = "providers.yaml") -> ModelBackend:
    """Look up `profile` in configs/providers.yaml and construct the
    matching ModelBackend. Raises KeyError if the profile or its `kind`
    field is missing/unrecognized -- deliberately fails loudly rather than
    silently falling back to a default provider."""
    data: dict[str, Any] = load_yaml(config_path) or {}
    profiles = data.get("profiles", {})
    if profile not in profiles:
        raise KeyError(f"no provider profile named {profile!r} in {config_path} (known: {sorted(profiles)})")
    spec = dict(profiles[profile])
    kind = spec.pop("kind", None)
    if kind not in _KIND_BUILDERS:
        raise KeyError(f"provider profile {profile!r} has unknown kind {kind!r} (known: {sorted(_KIND_BUILDERS)})")
    return _KIND_BUILDERS[kind](spec)
