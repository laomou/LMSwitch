"""枚举类型定义."""

from enum import Enum


class ProviderType(str, Enum):
    """服务提供商类型."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    CUSTOM = "custom"


class AgentType(str, Enum):
    """AI Agent 类型."""

    CLAUDE = "claude"
    CLINE = "cline"
    CODEX = "codex"
    DROID = "droid"
    OPENCODE = "opencode"
    OPENCLAW = "openclaw"
    PI = "pi"
