"""AISOS SDK multi-provider LLM backends. Each class implements the existing
aisos.agent.model_backend.ModelBackend interface -- this package only adds
new backends, it never modifies that module."""
from aisos.providers.anthropic_backend import AnthropicBackend
from aisos.providers.errors import (
    AuthenticationError,
    InvalidResponseError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from aisos.providers.openai_compatible import OpenAICompatibleBackend
from aisos.providers.registry import backend_from_profile

__all__ = [
    "OpenAICompatibleBackend", "AnthropicBackend", "backend_from_profile",
    "ProviderError", "AuthenticationError", "RateLimitError",
    "ProviderTimeoutError", "ProviderConnectionError", "InvalidResponseError",
]
