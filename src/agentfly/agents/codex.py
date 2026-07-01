"""Codex (OpenAI) Agent 适配器."""

from __future__ import annotations

from agentfly.agents.base import Agent, openai_base_url
from agentfly.models.schema import ResolvedConfig
from agentfly.models.types import AgentType


class Codex(Agent):
    """OpenAI Codex CLI 适配器.

    需要环境变量:
    - OPENAI_API_KEY
    - OPENAI_BASE_URL (以 /v1 结尾，重定向内置 openai provider)

    注意: 自 2026/02 起 Codex 仅支持 Responses API (/v1/responses)，
    仅支持 Chat Completions 的 provider 需经 LiteLLM 之类代理。
    """

    name = AgentType.CODEX
    display_name = "Codex"
    preferred_format = "openai"

    def env_vars(self, config: ResolvedConfig) -> dict[str, str]:
        env = {
            "OPENAI_API_KEY": config.provider.api_key,
        }
        if config.effective_api_base:
            env["OPENAI_BASE_URL"] = openai_base_url(config.effective_api_base)
        return env

    def launch_command(self, config: ResolvedConfig) -> list[str]:
        cmd = ["codex"]
        if config.agent.model:
            cmd.extend(["--model", config.agent.model])
        return cmd
