"""Anthropic (Claude) Provider."""

from __future__ import annotations

from agentfly.models.types import ProviderType
from agentfly.providers.base import Provider

ANTHROPIC_MODELS = [
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-fable-5",
]


class AnthropicProvider(Provider):
    """Anthropic Claude API Provider."""

    name = ProviderType.ANTHROPIC
    display_name = "Anthropic (Claude)"

    def list_models(self) -> list[str]:
        return ANTHROPIC_MODELS
