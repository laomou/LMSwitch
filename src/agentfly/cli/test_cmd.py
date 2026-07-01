"""[agentfly test] 测试模型可用性和速度 (stream 模式: TTFT + 吞吐)."""

from __future__ import annotations

import json
from typing import Any

import click

from agentfly.core.config import ensure_config_exists, save_config
from agentfly.core.resolver import ConfigResolver
from agentfly.models.schema import ProviderConfig, TestResult
from agentfly.models.types import ProviderType
from agentfly.providers.registry import get_provider

_STATUS_ICONS = {"ok": "✅", "timeout": "⏳", "error": "❌", "unauthorized": "❌"}
_COL_STATUS, _COL_LATENCY, _COL_TTFT, _COL_TPS = 14, 8, 8, 8


# ── tab 补全 ──

def _complete_providers(ctx: click.Context, param: click.Parameter, incomplete: str) -> list[Any]:
    config, _ = ensure_config_exists()
    return [
        click.shell_completion.CompletionItem(name)
        for name in config.providers if name.startswith(incomplete)
    ]


def _complete_models(ctx: click.Context, param: click.Parameter, incomplete: str) -> list[Any]:
    config, _ = ensure_config_exists()
    pc = config.providers.get(ctx.params.get("target", ""))
    if not pc:
        return []
    return [
        click.shell_completion.CompletionItem(m)
        for m in pc.model_names if m.startswith(incomplete)
    ]


# ── helpers ──

def _resolve(config, name: str) -> tuple[ProviderConfig, Any]:
    """解析 Provider 名称 → (ProviderConfig, Provider 实例)."""
    try:
        pc = ConfigResolver(config).get_provider(name)
    except KeyError:
        raise click.ClickException(f"Provider 未配置: {name}")
    p = get_provider(pc)
    if p is None:
        raise click.ClickException(f"不支持的 Provider: {pc.name.value}")
    return pc, p


def _base(pc: ProviderConfig) -> str:
    return pc.base_url


def _has_cached_api_type(pc: ProviderConfig) -> bool:
    return pc.name == ProviderType.CUSTOM and any(me.api_type for me in pc.models)


def _maybe_save_cache(config, pc: ProviderConfig) -> None:
    if _has_cached_api_type(pc):
        save_config(config)


# ── 输出 ──

def _icon(status: str) -> str:
    return _STATUS_ICONS.get(status, "❓")


def _pad(val: float) -> str:
    """格式化数值: ms/s 或 `-`."""
    if val <= 0:
        return "-"
    if val < 1000:
        return f"{val:.0f}ms"
    return f"{val / 1000:.1f}s"


def _header(model_w: int) -> str:
    return (
        f"{'Model':<{model_w}}  "
        f"{'Status':<{_COL_STATUS}}  "
        f"{'Total':<{_COL_LATENCY}}  "
        f"{'TTFT':<{_COL_TTFT}}  "
        f"{'TPS':<{_COL_TPS}}\n"
        f"{'-'*model_w}  "
        f"{'-'*_COL_STATUS}  "
        f"{'-'*_COL_LATENCY}  "
        f"{'-'*_COL_TTFT}  "
        f"{'-'*_COL_TPS}"
    )


def _row(r: TestResult, model_w: int) -> str:
    return (
        f"{r.model:<{model_w}}  "
        f"{_icon(r.status):<2}{r.status:<{_COL_STATUS - 2}} "
        f"{_pad(r.latency_ms):<{_COL_LATENCY}}  "
        f"{_pad(r.ttft_ms):<{_COL_TTFT}}  "
        f"{_pad(r.tokens_per_sec):<{_COL_TPS}}"
    )


def _print_table(results: list[TestResult]) -> None:
    if not results:
        return
    model_w = max(max(len(r.model) for r in results), 5)
    click.echo(f"Provider: {results[0].provider}")
    click.echo(_header(model_w))
    for r in results:
        click.echo(_row(r, model_w))


def _print_json(results: list[TestResult]) -> None:
    click.echo(json.dumps(
        [r.model_dump(mode="json") for r in results], indent=2, ensure_ascii=False,
    ))


# ── 单 Provider 测试 (流式输出) ──

def _test_provider(config, pc: ProviderConfig, p, provider_key: str) -> list[TestResult]:
    models = [m for m in (pc.model_names or p.list_models()) if m]
    if not models:
        return []

    model_w = max(len(m) for m in models)
    click.echo(f"Provider: {provider_key}")
    click.echo(_header(model_w))

    results: list[TestResult] = []
    for model in models:
        r = p.test_model(model, pc.api_key, _base(pc), provider_key=provider_key)
        results.append(r)
        click.echo(_row(r, model_w))

    _maybe_save_cache(config, pc)
    return results


# ── command ──

@click.command(name="test")
@click.argument("target", required=False, shell_complete=_complete_providers)
@click.argument("model_name", required=False, shell_complete=_complete_models)
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text", help="输出格式")
def test(target: str | None, model_name: str | None, fmt: str) -> None:
    """测试模型可用性和延迟 (stream 模式).

    \b
    示例:
      agentfly test                           # 全部 Provider 所有模型
      agentfly test deepseek                  # DeepSeek 所有模型
      agentfly test deepseek deepseek-chat    # 指定模型
      agentfly test deepseek:deepseek-chat    # 兼容写法
      agentfly test --format json             # JSON 输出
    """
    config, _ = ensure_config_exists()

    # 兼容 provider:model 语法
    if target and not model_name and ":" in target:
        target, model_name = target.split(":", 1)

    if target and model_name:
        # 单模型
        pc, p = _resolve(config, target)
        r = p.test_model(model_name, pc.api_key, _base(pc), provider_key=target)
        _print_table([r]) if fmt == "text" else _print_json([r])
        _maybe_save_cache(config, pc)
        return

    if target:
        # 单个 Provider → 流式输出
        pc, p = _resolve(config, target)
        results = _test_provider(config, pc, p, target)
        if fmt == "json":
            _print_json(results)
        return

    # 全部 Provider → 逐个流式输出
    for pk, pc in config.providers.items():
        p = get_provider(pc)
        if p is None:
            continue
        _test_provider(config, pc, p, pk)
        click.echo()
