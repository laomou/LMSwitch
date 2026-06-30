"""AgentRegistry 测试 — 注册/查询 + entry_points 发现容错."""

from __future__ import annotations

from lmswitch.agents.codex import Codex
from lmswitch.agents.registry import AgentRegistry
from lmswitch.models.types import AgentType


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
        monkeypatch.setattr("lmswitch.agents.registry.entry_points",
                            lambda group=None: [_BadEP()])
        reg = AgentRegistry()
        reg.discover_from_entry_points()  # 加载失败不应抛
        assert reg.names() == []

    def test_loads_good_entry_point(self, monkeypatch):
        monkeypatch.setattr("lmswitch.agents.registry.entry_points",
                            lambda group=None: [_GoodEP()])
        reg = AgentRegistry()
        reg.discover_from_entry_points()
        assert "codex" in reg.names()


def test_get_registry_singleton():
    from lmswitch.agents.registry import get_registry

    assert get_registry() is get_registry()
