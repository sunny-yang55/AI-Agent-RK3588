"""
Immutable result container returned by every tool execution.

Designed to be serialisable and introspectable so that any caller —
whether the Agent executor, a visualisation front-end, or logs on
RK3588 — can reliably determine what happened.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .status import ToolStatus


@dataclass(frozen=True)
class ToolResult:
    """Outcome of a single :meth:`BaseTool.execute` invocation.

    Attributes
    ----------
    tool_name : str
        The :attr:`BaseTool.name` of the tool that produced this result.
    status : ToolStatus
        Final lifecycle state after execution.
    data : Any
        Arbitrary output payload produced by the tool.
    error : str | None
        Human-readable error description (``None`` on success).
    duration_ms : float
        Wall-clock execution time in milliseconds.
    metadata : dict[str, Any]
        Tool-specific supplementary information (e.g. file paths,
        inference confidence, port names).
    started_at : float
        ``time.perf_counter()`` value captured before execution began.
    finished_at : float
        ``time.perf_counter()`` value captured after execution ended.
    """

    tool_name: str
    status: ToolStatus
    data: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def success(self) -> bool:
        """``True`` iff execution finished with ``SUCCESS``."""
        return self.status is ToolStatus.SUCCESS

    @property
    def timed_out(self) -> bool:
        """``True`` iff execution was terminated by timeout."""
        return self.status is ToolStatus.TIMEOUT

    @property
    def cancelled(self) -> bool:
        """``True`` iff execution was explicitly cancelled."""
        return self.status is ToolStatus.CANCELLED

    @property
    def is_terminal(self) -> bool:
        """``True`` once the tool has definitively finished."""
        return self.status.is_terminal

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result into a plain dictionary for logging/transport."""
        return {
            "tool_name": self.tool_name,
            "status": self.status.name,
            "data": self.data,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_error(
        cls,
        tool_name: str,
        error: str,
        *,
        duration_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Convenience constructor for immediate-failure results."""
        return cls(
            tool_name=tool_name,
            status=ToolStatus.ERROR,
            error=error,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
