"""客制化 Provider — 支持任意 OpenAI/Anthropic 兼容 API."""

from __future__ import annotations

from agentfly.models.schema import ProviderConfig
from agentfly.models.types import ProviderType
from agentfly.providers.base import Provider


class CustomProvider(Provider):
    """客制化 Provider — 支持任意 OpenAI/Anthropic 兼容 API.

    候选/探测/取最快的骨架在 base.test_model, base._parse_stream_chunk 已兼容
    两种 SSE; 这里只负责把跑通的接口写入 api_type 缓存 (如 "anthropic,openai",
    速度快的在前).
    """

    name = ProviderType.CUSTOM
    display_name = "Custom"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._cache_dirty = False  # 本次是否改动了 api_type 缓存

    def list_models(self) -> list[str]:
        return self.config.model_names

    def _on_results(self, model: str, api_types: list[str]) -> None:
        """缓存跑通的接口 (base 测试结束时回调). 读写同锁 base._models_lock."""
        val = ",".join(api_types)
        with self._models_lock:
            if self.config.models.get(model) == val:
                return
            self.config.models[model] = val
            self._cache_dirty = True
