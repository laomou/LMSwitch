"""Provider.test_model 流式探测 + SSE 解析测试 (mock httpx)."""

from __future__ import annotations

import httpx
import pytest

from lmswitch.models.schema import ProviderConfig
from lmswitch.models.types import ProviderType
from lmswitch.providers.anthropic import AnthropicProvider
from lmswitch.providers.openai import OpenAIProvider


def _openai_provider():
    return OpenAIProvider(ProviderConfig(
        name=ProviderType.OPENAI, api_key="sk-x",
        endpoints={"openai": "https://api.openai.com"}, models=["gpt-4o"],
    ))


def _anthropic_provider():
    return AnthropicProvider(ProviderConfig(
        name=ProviderType.ANTHROPIC, api_key="k",
        endpoints={"anthropic": "https://api.anthropic.com"}, models=["claude-opus-4-8"],
    ))


class _FakeStream:
    """模拟 httpx 的 stream 响应上下文."""

    def __init__(self, status_code, lines):
        self.status_code = status_code
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_lines(self):
        yield from self._lines


def _patch_client(monkeypatch, *, stream=None, exc=None):
    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, *a, **k):
            if exc is not None:
                raise exc
            return stream

    monkeypatch.setattr("lmswitch.providers.base.httpx.Client", FakeClient)


class TestTestModel:
    """test_model 各状态分支."""

    def test_unauthorized(self, monkeypatch):
        _patch_client(monkeypatch, stream=_FakeStream(401, []))
        assert _openai_provider().test_model("gpt-4o").status == "unauthorized"

    def test_http_error(self, monkeypatch):
        _patch_client(monkeypatch, stream=_FakeStream(500, []))
        assert _openai_provider().test_model("gpt-4o").status == "error"

    def test_timeout(self, monkeypatch):
        _patch_client(monkeypatch, exc=httpx.TimeoutException("x"))
        assert _openai_provider().test_model("gpt-4o").status == "timeout"

    def test_connect_error(self, monkeypatch):
        _patch_client(monkeypatch, exc=httpx.ConnectError("x"))
        r = _openai_provider().test_model("gpt-4o")
        assert r.status == "error"
        assert "无法连接" in r.error_message

    def test_ok_with_metrics(self, monkeypatch):
        lines = [
            'data: {"choices":[{"delta":{"content":"hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            "data: [DONE]",
        ]
        _patch_client(monkeypatch, stream=_FakeStream(200, lines))
        r = _openai_provider().test_model("gpt-4o", provider_key="deepseek")
        assert r.status == "ok"
        assert r.provider == "deepseek"  # provider_key 透传
        assert r.latency_ms >= 0
        assert r.ttft_ms >= 0


class TestParseStreamChunk:
    """SSE 行解析 (OpenAI + Anthropic)."""

    def test_openai_content(self):
        line = 'data: {"choices":[{"delta":{"content":"hi"}}]}'
        assert _openai_provider()._parse_stream_chunk(line) == "hi"

    def test_openai_reasoning_fallback(self):
        line = 'data: {"choices":[{"delta":{"reasoning_content":"t"}}]}'
        assert _openai_provider()._parse_stream_chunk(line) == "t"

    def test_done(self):
        assert _openai_provider()._parse_stream_chunk("data: [DONE]") is None

    def test_bad_json(self):
        assert _openai_provider()._parse_stream_chunk("data: not-json") is None

    def test_non_data_line(self):
        assert _openai_provider()._parse_stream_chunk(": comment") is None

    def test_anthropic_text(self):
        line = 'data: {"type":"content_block_delta","delta":{"text":"hi"}}'
        assert _anthropic_provider()._parse_stream_chunk(line) == "hi"

    def test_anthropic_message_delta_none(self):
        line = 'data: {"type":"message_delta","delta":{}}'
        assert _anthropic_provider()._parse_stream_chunk(line) is None


class TestProviderMethods:
    """各 Provider 的端点 / 请求体 / 模型列表."""

    def test_openai(self):
        p = _openai_provider()
        assert p._test_endpoint() == "/v1/chat/completions"
        assert p._build_test_request("m")["model"] == "m"
        assert "gpt-4o" in p.list_models()

    def test_anthropic(self):
        p = _anthropic_provider()
        assert p._test_endpoint() == "/v1/messages"
        assert p._build_test_request("m")["max_tokens"] == 64
        assert p.list_models()  # 非空

    def test_deepseek(self):
        from lmswitch.providers.deepseek import DeepSeekProvider
        p = DeepSeekProvider(ProviderConfig(
            name=ProviderType.DEEPSEEK, api_key="k",
            endpoints={"openai": "http://x"}, models=["d1"]))
        assert p._test_endpoint() == "/v1/chat/completions"
        assert p._build_test_request("m")["messages"]
        assert p.list_models() == ["d1"]

    def test_custom_endpoint_depends_on_format(self):
        from lmswitch.providers.custom import CustomProvider
        oa = CustomProvider(ProviderConfig(
            name=ProviderType.CUSTOM, api_key="k",
            endpoints={"openai": "http://x"}, models=["c1"]))
        an = CustomProvider(ProviderConfig(
            name=ProviderType.CUSTOM, api_key="k",
            endpoints={"anthropic": "http://x"}, models=["c1"]))
        assert oa._test_endpoint() == "/v1/chat/completions"
        assert an._test_endpoint() == "/v1/messages"
        assert oa._build_test_request("m")["model"] == "m"
        assert oa.list_models() == ["c1"]


class TestAnthropicParseEdge:
    """Anthropic SSE 解析的非 data / 坏 JSON 行."""

    def test_non_data_line(self):
        assert _anthropic_provider()._parse_stream_chunk("event: ping") is None

    def test_bad_json(self):
        assert _anthropic_provider()._parse_stream_chunk("data: nope") is None
