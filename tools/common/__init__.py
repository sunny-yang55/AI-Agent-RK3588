"""
Common tool infrastructure — abstract base classes, result containers,
and status enums shared by all concrete tool packages.

Sub-packages (``vision``, ``speech``, ``robot``, ``rknn``, ``filesystem``,
etc.) build on these primitives so that every tool exposes a uniform
interface to the Agent executor.
"""

from .base_tool import BaseTool
from .result import ToolResult
from .status import ToolStatus

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolStatus",
]
