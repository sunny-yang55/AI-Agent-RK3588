"""
Tools — the capability layer of the AI-Agent platform.

Each sub-package provides concrete implementations of
:class:`tools.common.BaseTool` for a specific domain:

* :mod:`tools.vision`   — computer vision (object detection, OCR, etc.)
* :mod:`tools.speech`   — speech-to-text and text-to-speech
* :mod:`tools.robot`    — robot control and sensor interfaces
* :mod:`tools.llm`      — LLM orchestration and prompt-based reasoning
* :mod:`tools.utils`    — shared utility helpers

All tools share the same lifecycle (initialize → execute → cleanup)
and return standardised :class:`tools.common.ToolResult` instances.
"""

from .common import BaseTool, ToolResult, ToolStatus

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolStatus",
]
