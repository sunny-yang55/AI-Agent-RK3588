"""
Tool lifecycle status enumeration.

Defines the canonical states a tool may occupy throughout its
initialisation, execution, and teardown phases.  Compatible with
both Windows and Ubuntu (RK3588) environments.
"""

from __future__ import annotations

from enum import Enum, auto


class ToolStatus(Enum):
    """Lifecycle states for any tool implementing :class:`BaseTool`.

    +---------------+--------------------------------------------------+
    | State         | Meaning                                          |
    +===============+==================================================+
    | IDLE          | Tool has been instantiated but not initialised.  |
    +---------------+--------------------------------------------------+
    | INITIALIZING  | :meth:`BaseTool.initialize` is in progress.      |
    +---------------+--------------------------------------------------+
    | READY         | Tool is initialised and waiting for a task.      |
    +---------------+--------------------------------------------------+
    | RUNNING       | :meth:`BaseTool.execute` is actively running.    |
    +---------------+--------------------------------------------------+
    | SUCCESS       | Execution completed without errors.              |
    +---------------+--------------------------------------------------+
    | ERROR         | Execution failed with an unhandled exception.    |
    +---------------+--------------------------------------------------+
    | TIMEOUT       | Execution exceeded its configured time budget.   |
    +---------------+--------------------------------------------------+
    | CANCELLED     | Execution was explicitly cancelled by the caller.|
    +---------------+--------------------------------------------------+
    """

    IDLE = auto()
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    SUCCESS = auto()
    ERROR = auto()
    TIMEOUT = auto()
    CANCELLED = auto()

    @property
    def is_terminal(self) -> bool:
        """Return ``True`` when the tool is no longer executing."""
        return self in _TERMINAL_STATES

    @property
    def is_ok(self) -> bool:
        """Return ``True`` when execution finished successfully."""
        return self is ToolStatus.SUCCESS

    @property
    def transient(self) -> bool:
        """Return ``True`` while the tool is actively transitioning."""
        return self in _TRANSIENT_STATES


_TERMINAL_STATES: frozenset[ToolStatus] = frozenset(
    {ToolStatus.SUCCESS, ToolStatus.ERROR, ToolStatus.TIMEOUT, ToolStatus.CANCELLED}
)
_TRANSIENT_STATES: frozenset[ToolStatus] = frozenset(
    {ToolStatus.INITIALIZING, ToolStatus.RUNNING}
)
