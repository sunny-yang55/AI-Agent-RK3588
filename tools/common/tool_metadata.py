from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolMetadata:
    """
    Tool 元数据描述。

    v0.6.7:
    增强 Tool 自描述能力。
    """

    # 基础信息

    name: str

    description: str

    # 输入参数描述

    parameters: dict[str, Any] = field(default_factory=dict)

    # 执行入口

    handler: Callable | None = None

    # 能力标签

    capabilities: list[str] = field(default_factory=list)

    # ==========================
    # v0.6.7 Metadata Enhancement
    # ==========================

    # Tool版本

    version: str = "1.0.0"

    # 输入Schema

    input_schema: dict[str, Any] = field(default_factory=dict)

    # 输出Schema

    output_schema: dict[str, Any] = field(default_factory=dict)

    # 是否启用

    enabled: bool = True

    # 标签

    tags: list[str] = field(default_factory=list)

    # 扩展信息

    metadata: dict[str, Any] = field(default_factory=dict)
