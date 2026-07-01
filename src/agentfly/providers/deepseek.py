"""DeepSeek Provider."""

from __future__ import annotations

from agentfly.models.types import ProviderType
from agentfly.providers.base import Provider


class DeepSeekProvider(Provider):
    """DeepSeek API Provider.

    DeepSeek tiers: pro(强) + flash(快). 用于 Claude 时:
    opus=sonnet=pro, haiku=flash.
    """

    name = ProviderType.DEEPSEEK
    display_name = "DeepSeek"

    def list_models(self) -> list[str]:
        return self.config.model_names

    def _build_test_request(self, model: str) -> dict:
        return {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }
