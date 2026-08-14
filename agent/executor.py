from __future__ import annotations

import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from capability import CapabilityManager
from tools.common.tool import TOOL_REGISTRY
from tools.common.tool_metadata import ToolMetadata

from .memory import MemoryManager
from .planner import PlanStep, StepStatus
from .tool_context import ToolContext

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    SEQUENTIAL = auto()
    PARALLEL = auto()
    DAG = auto()


@dataclass
class ExecutionContext:
    """Runtime context shared across one tool execution."""

    session_id: str | None = None

    goal: str | None = None

    tool_name: str | None = None

    start_time: float = 0.0

    retry_count: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    step_id: str
    status: StepStatus
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


ToolHandler = Callable[..., Any]
logger = logging.getLogger(__name__)


class ToolRegistry:

    def __init__(self) -> None:
        self._tools: dict[str, ToolMetadata] = {}

    def register(self, tool: ToolMetadata) -> None:

        self.validate(tool)

        self._tools[tool.name] = tool

        logger.info("Tool validated and registered: %s", tool.name)

    def validate(self, tool: ToolMetadata) -> bool:
        """
        Validate tool metadata before registration.

        Checks:
        1. tool name exists
        2. description exists
        3. handler is callable
        4. parameters format
        """

        if not tool.name:
            raise ValueError("Tool validation failed: name is empty")

        if not tool.description:
            raise ValueError(f"Tool validation failed: {tool.name} description empty")

        if tool.handler is not None:
            if not callable(tool.handler):
                raise ValueError(f"Tool validation failed: {tool.name} handler invalid")

        if not isinstance(tool.parameters, dict):
            raise ValueError(
                f"Tool validation failed: {tool.name} parameters must be dict"
            )

        return True

    def register_class_tool(self, tool_instance) -> None:

        from tools.common.class_tool_adapter import class_tool_adapter

        metadata = class_tool_adapter(tool_instance)

        self.register(metadata)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolMetadata | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolMetadata]:
        return list(self._tools.values())

    def describe_tools(self) -> list[dict]:

        tools = []

        for tool in self._tools.values():

            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "capabilities": tool.capabilities,
                    "parameters": tool.parameters,
                }
            )

        return tools

    def discover(self) -> list[dict]:

        return self.describe_tools()

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


class BaseExecutor(ABC):
    def __init__(
        self,
        memory: MemoryManager | None = None,
        tools: ToolRegistry | None = None,
        capabilities: CapabilityManager | None = None,
    ) -> None:

        self._memory = memory
        self._tools = tools or ToolRegistry()
        self._capabilities = capabilities

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    @property
    def capabilities(self) -> CapabilityManager | None:
        return self._capabilities

    @abstractmethod
    async def execute_step(self, step: PlanStep) -> ExecutionResult: ...

    @abstractmethod
    async def execute_steps(
        self,
        steps: list[PlanStep],
        mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
    ) -> list[ExecutionResult]: ...

    @abstractmethod
    def execute_step_sync(self, step: PlanStep) -> ExecutionResult: ...


class DefaultExecutor(BaseExecutor):
    def __init__(
        self,
        memory: MemoryManager | None = None,
        tools: ToolRegistry | None = None,
        capabilities: CapabilityManager | None = None,
        default_handler: ToolHandler | None = None,
    ) -> None:

        super().__init__(
            memory=memory,
            tools=tools,
            capabilities=capabilities,
        )

        self._default_handler = default_handler

    async def execute_step(self, step: PlanStep) -> ExecutionResult:
        start = time.perf_counter()

        context = ToolContext(
            step_id=step.step_id,
            action=step.action,
            params=dict(step.params),
        )

        ctx = ExecutionContext(
            tool_name=step.action,
            start_time=start,
        )
        step.status = StepStatus.IN_PROGRESS

        if self._memory:
            self._memory.short_term.store(
                key=f"step:{step.step_id}",
                value={
                    "action": step.action,
                    "description": step.description,
                    "status": "in_progress",
                },
            )
        # -----------------------------
        # Context Binding
        # -----------------------------
        params = dict(step.params)

        if self._memory:

            image = self._memory.working.retrieve("image")

            if image and "image" in params:

                params["image"] = image

            audio = self._memory.working.retrieve("audio")

            if audio and "audio" in params:

                params["audio"] = audio

            robot = self._memory.working.retrieve("robot")

            if robot and "robot" in params:

                params["robot"] = robot
        await self.before_execute(context)
        try:

            # ==================================
            # Capability Layer 优先
            # ==================================

            if self._capabilities and self._capabilities.get(step.action):

                output = self._capabilities.execute(
                    step.action,
                    **params,
                )

            else:

                # ==================================
                # 原 Tool 系统兼容执行
                # ==================================

                tool = self._tools.get(step.action)

                if tool is None:

                    if self._default_handler is None:
                        raise RuntimeError(f"No handler for action: {step.action}")

                    handler = self._default_handler

                    if asyncio.iscoroutinefunction(handler):

                        output = await handler(**params)

                    else:

                        output = handler(**params)

                else:

                    if tool.handler is not None:

                        if asyncio.iscoroutinefunction(tool.handler):

                            output = await tool.handler(**params)

                        else:

                            output = tool.handler(**params)

                    elif hasattr(tool, "execute"):

                        result = tool.execute(**params)

                        if asyncio.iscoroutine(result):

                            output = await result

                        else:

                            output = result

                    else:

                        raise RuntimeError(
                            f"Tool {step.action} has no executable interface"
                        )
            step.result = output

            if self._memory:
                self._memory.working.store(
                    key=step.action,
                    value=output,
                )

            step.status = StepStatus.COMPLETED
            step.error = None
            await self.after_execute(
                context,
                output,
            )
            if self._memory:
                self._memory.short_term.store(
                    key=f"step:{step.step_id}",
                    value={
                        "action": step.action,
                        "status": "completed",
                    },
                )
        except Exception as exc:

            step.status = StepStatus.FAILED
            step.error = str(exc)

            if self._memory:
                self._memory.short_term.store(
                    key=f"step:{step.step_id}",
                    value={
                        "action": step.action,
                        "error": str(exc),
                        "status": "failed",
                    },
                )

            logger.error(
                "Step %s (%s) failed:\n%s",
                step.step_id,
                step.action,
                traceback.format_exc(),
            )

        duration_ms = (time.perf_counter() - start) * 1000

        return ExecutionResult(
            step_id=step.step_id,
            status=step.status,
            output=step.result,
            error=step.error,
            duration_ms=duration_ms,
        )

    async def execute_steps(
        self,
        steps: list[PlanStep],
        mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
    ) -> list[ExecutionResult]:
        if mode == ExecutionMode.PARALLEL:
            tasks = [self.execute_step(s) for s in steps]
            return list(await asyncio.gather(*tasks))

        if mode == ExecutionMode.DAG:
            return await self._execute_dag(steps)

        results: list[ExecutionResult] = []
        for step in steps:
            result = await self.execute_step(step)
            results.append(result)
            if step.status == StepStatus.FAILED:
                break
        return results

    async def _execute_dag(self, steps: list[PlanStep]) -> list[ExecutionResult]:
        completed: set[str] = set()
        results: list[ExecutionResult] = []
        remaining = list(steps)

        while remaining:
            ready = [
                s for s in remaining if all(dep in completed for dep in s.depends_on)
            ]
            if not ready:
                failed = [s for s in remaining if s.status == StepStatus.FAILED]
                if failed:
                    break
                ready = [remaining[0]]

            tasks = [self.execute_step(s) for s in ready]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            for s in ready:
                if s.status == StepStatus.COMPLETED:
                    completed.add(s.step_id)
                remaining.remove(s)

            if any(r.status == StepStatus.FAILED for r in batch_results):
                break

        return results

    async def before_execute(self, context: ToolContext):

        logger.info(
            "[Tool] %s starting...",
            context.action,
        )

    async def after_execute(
        self,
        context: ToolContext,
        output,
    ):

        logger.info(
            "[Tool] %s finished.",
            context.action,
        )

    def execute_step_sync(self, step: PlanStep) -> ExecutionResult:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.execute_step(step))
        import concurrent.futures

        future = asyncio.run_coroutine_threadsafe(self.execute_step(step), loop)
        return future.result()


Executor = DefaultExecutor
