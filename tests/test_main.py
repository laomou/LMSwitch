"""CLI 入口冒烟测试."""

from __future__ import annotations

from click.testing import CliRunner

from agentfly.main import cli


class TestCli:
    def test_help_lists_subcommands(self):
        r = CliRunner().invoke(cli, ["--help"])
        assert r.exit_code == 0
        for sub in ("doctor", "launch", "test", "provider"):
            assert sub in r.output

    def test_version(self):
        r = CliRunner().invoke(cli, ["--version"])
        assert r.exit_code == 0
        assert "agentfly" in r.output
