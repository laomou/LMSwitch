"""OpenCode Agent 适配器."""

from lmswitch.agents.base import Agent, openai_base_url
from lmswitch.models.schema import ResolvedConfig
from lmswitch.models.types import AgentType


class OpenCode(Agent):
    """OpenCode — 开源 AI 编程 Agent.

    OpenAI 兼容模式: OPENAI_API_KEY + OPENAI_BASE_URL(/v1)。
    模型在应用内通过 /models 选择 (OpenCode 不读环境变量指定模型)。
    """

    name = AgentType.OPENCODE
    display_name = "OpenCode"
    preferred_format = "openai"

    def env_vars(self, config: ResolvedConfig) -> dict[str, str]:
        env = {"OPENAI_API_KEY": config.provider.api_key}
        if config.effective_api_base:
            env["OPENAI_BASE_URL"] = openai_base_url(config.effective_api_base)
        return env

    def launch_command(self, config: ResolvedConfig) -> list[str]:
        return ["opencode"]
