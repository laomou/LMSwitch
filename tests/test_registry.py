"""AgentRegistry 测试 — 注册/查询 + entry_points 发现容错."""

from __future__ import annotations

from agentfly.agents.codex import Codex
from agentfly.agents.registry import AgentRegistry
from agentfly.models.types import AgentType


class TestRegistry:
    def test_register_get_names_list(self):
        reg = AgentRegistry()
        reg.register(Codex())
        assert reg.get("codex") is not None
        assert "codex" in reg.names()
        assert [a.name for a in reg.list()] == [AgentType.CODEX]

    def test_get_unknown_returns_none(self):
        assert AgentRegistry().get("nope") is None


class _BadEP:
    name = "bad"

    def load(self):
        raise RuntimeError("boom")


class _GoodEP:
    name = "codex"

    def load(self):
        return Codex


class TestDiscover:
    def test_skips_failing_entry_point(self, monkeypatch):
        monkeypatch.setattr("agentfly.agents.registry.entry_points",
                            lambda group=None: [_BadEP()])
        reg = AgentRegistry()
        reg.discover_from_entry_points()  # 加载失败不应抛
        assert reg.names() == []

    def test_loads_good_entry_point(self, monkeypatch):
        monkeypatch.setattr("agentfly.agents.registry.entry_points",
                            lambda group=None: [_GoodEP()])
        reg = AgentRegistry()
        reg.discover_from_entry_points()
        assert "codex" in reg.names()


def test_get_registry_singleton():
    from agentfly.agents.registry import get_registry

    assert get_registry() is get_registry()


def test_get_registry_has_builtins():
    """内置 Agent 直接注册, 不依赖 entry_points (源码 checkout 也能用)."""
    from agentfly.agents.registry import get_registry

    names = set(get_registry().names())
    assert {"claude", "cline", "codex", "droid", "opencode", "openclaw", "pi"} <= names
