"""
Abstract base class for every tool in the system.

Subclass :class:`BaseTool` to implement concrete tools for Vision,
Speech, Robot control, RKNN inference, Filesystem operations, and any
future capability the platform requires.

Design principles
-----------------
* **Async-first** – ``initialize``, ``execute``, and ``cleanup`` are all
  coroutines.  Synchronous wrappers are provided for simple callers.
* **Lifecycle-aware** – tools track their state via :class:`ToolStatus`
  and guard against illegal transitions.
* **Platform-agnostic** – the interface avoids OS-specific assumptions
  so it works identically on Windows (development) and Ubuntu aarch64
  (RK3588 deployment).
* **Metadata-rich** – every tool declares its name, description,
  version, and advertised capabilities so that registries and
  auto-discovery can query them without importing sub-packages.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from .result import ToolResult
from .status import ToolStatus

logger = logging.getLogger(__name__)


def sanitize_result(obj):
    # 防止大对象进入Memory和JSON

    if isinstance(obj, np.ndarray):
        return {"type": "ndarray", "shape": obj.shape}

    if hasattr(obj, "image_id"):

        obj.image_id = str(obj.image_id)

    return obj


class BaseTool(ABC):
    """Abstract tool that all concrete tools must inherit from.

    Subclasses **must** implement :meth:`_execute_impl` and should
    typically also declare :attr:`name`, :attr:`description`, and
    :attr:`capabilities`.

    Lifecycle
    ---------
    1. ``tool = ConcreteTool(config=...)``       → ``IDLE``
    2. ``await tool.initialize()``               → ``READY``
    3. ``result = await tool.execute(**kwargs)`` → ``SUCCESS / ERROR / …``
    4. ``await tool.cleanup()``                  → ``IDLE``
    """

    # ------------------------------------------------------------------
    # Subclass overrides (metadata)
    # ------------------------------------------------------------------

    name: ClassVar[str] = ""
    """Human-readable tool identifier (e.g. ``"VisionDetector"``)."""

    description: ClassVar[str] = ""
    """Short paragraph explaining what the tool does and when to use it."""

    version: ClassVar[str] = "0.1.0"
    """Semantic version string for the tool implementation."""

    capabilities: ClassVar[frozenset[str]] = frozenset()
    """Labels describing what the tool is capable of (e.g. ``"object_detection"``).

    Used by registries and planners to select appropriate tools for a
    given task.
    """

    # ------------------------------------------------------------------
    # Instance state
    # ------------------------------------------------------------------

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise the tool with an optional configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any] | None
            Tool-specific settings.  Keys and semantics are defined by
            each subclass.
        """
        self._config: dict[str, Any] = config or {}
        self._status: ToolStatus = ToolStatus.IDLE
        self._started_at: float = 0.0

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def status(self) -> ToolStatus:
        """Current lifecycle status of the tool."""
        return self._status

    @property
    def config(self) -> dict[str, Any]:
        """Mutable copy of the tool configuration dictionary."""
        return dict(self._config)

    @property
    def is_ready(self) -> bool:
        """``True`` when the tool is initialised and ready to execute."""
        return self._status is ToolStatus.READY

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Async initialisation: load models, open connections, allocate resources.

        Guards against double-initialisation.  Sets status to ``READY``
        on success.
        """
        if self._status is ToolStatus.READY:
            logger.debug("%s already initialised.", self.name)
            return

        self._status = ToolStatus.INITIALIZING
        logger.info("%s initialising…", self.name)

        try:
            await self._initialize_impl()
            self._status = ToolStatus.READY
            logger.info("%s initialised.", self.name)
        except Exception:
            self._status = ToolStatus.ERROR
            logger.exception("%s failed to initialise.", self.name)
            raise

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool with the supplied keyword arguments.

        This is the primary public entry-point.  It delegates to
        :meth:`_execute_impl`, manages status transitions, timing, and
        error handling automatically.

        Returns
        -------
        ToolResult
            Always returns a result object even on failure — check
            :attr:`ToolResult.success`.
        """
        if not self.is_ready:
            await self.initialize()

        self._status = ToolStatus.RUNNING
        self._started_at = time.perf_counter()

        try:
            data = await self._execute_impl(**kwargs)
            finished_at = time.perf_counter()
            self._status = ToolStatus.SUCCESS
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                data=sanitize_result(data),
                duration_ms=(finished_at - self._started_at) * 1000,
                started_at=self._started_at,
                finished_at=finished_at,
            )
        except asyncio.TimeoutError:
            finished_at = time.perf_counter()
            self._status = ToolStatus.TIMEOUT
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.TIMEOUT,
                error="Execution timed out",
                duration_ms=(finished_at - self._started_at) * 1000,
                started_at=self._started_at,
                finished_at=finished_at,
            )
        except asyncio.CancelledError:
            finished_at = time.perf_counter()
            self._status = ToolStatus.CANCELLED
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.CANCELLED,
                error="Execution cancelled",
                duration_ms=(finished_at - self._started_at) * 1000,
                started_at=self._started_at,
                finished_at=finished_at,
            )
        except Exception as exc:
            finished_at = time.perf_counter()
            self._status = ToolStatus.ERROR
            logger.exception("%s execution failed.", self.name)
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                error=str(exc),
                duration_ms=(finished_at - self._started_at) * 1000,
                started_at=self._started_at,
                finished_at=finished_at,
            )

    async def cleanup(self) -> None:
        """Release resources acquired during :meth:`initialize`.

        After this call the tool returns to ``IDLE`` so it can be
        re-initialised — or safely discarded.
        """
        logger.info("%s shutting down…", self.name)
        try:
            await self._cleanup_impl()
        except Exception:
            logger.exception("%s cleanup raised an exception.", self.name)
        finally:
            self._status = ToolStatus.IDLE

    # ------------------------------------------------------------------
    # Sync convenience wrappers
    # ------------------------------------------------------------------

    def initialize_sync(self) -> None:
        """Blocking variant of :meth:`initialize`."""
        self._run_sync(self.initialize())

    def execute_sync(self, **kwargs: Any) -> ToolResult:
        """Blocking variant of :meth:`execute`."""
        return self._run_sync(self.execute(**kwargs))

    def cleanup_sync(self) -> None:
        """Blocking variant of :meth:`cleanup`."""
        self._run_sync(self.cleanup())

    @staticmethod
    def _run_sync(coro: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_input(self, **kwargs: Any) -> None:
        """Validate keyword arguments before execution.

        By default this is a no-op.  Subclasses should override it to
        raise :class:`TypeError` or :class:`ValueError` for invalid
        inputs.

        Raises
        ------
        TypeError
            If a required argument is missing or of wrong type.
        ValueError
            If an argument value is semantically invalid.
        """
        pass  # no-op — override in subclass

    # ------------------------------------------------------------------
    # Abstract methods — subclasses MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    async def _execute_impl(self, **kwargs: Any) -> Any:
        """Core tool logic.

        Subclasses implement the actual tool behaviour here.  The public
        :meth:`execute` wrapper handles status management, error
        handling, and timing automatically.

        Parameters
        ----------
        **kwargs : Any
            Tool-specific keyword arguments.

        Returns
        -------
        Any
            Arbitrary output data wrapped in :class:`ToolResult.data`.
        """
        ...

    # ------------------------------------------------------------------
    # Optional override hooks
    # ------------------------------------------------------------------

    async def _initialize_impl(self) -> None:
        """Override to load models, connect to hardware, etc.

        Called once by :meth:`initialize`.  The default is a no-op.
        """
        pass

    async def _cleanup_impl(self) -> None:
        """Override to release models, disconnect hardware, etc.

        Called once by :meth:`cleanup`.  The default is a no-op.
        """
        pass

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of this tool's metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "capabilities": sorted(self.capabilities),
            "status": self._status.name,
        }
