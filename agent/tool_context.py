from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass
class ToolContext:
    """
    Tool 生命周期上下文。
    """

    step_id: str
    action: str

    params: dict[str, Any]

    result: Any = None

    error: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    start_time: float = field(default_factory=perf_counter)

    end_time: float | None = None

    @property
    def duration_ms(self) -> float:

        if self.end_time is None:
            return 0.0

        return (self.end_time - self.start_time) * 1000
