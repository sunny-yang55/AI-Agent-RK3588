from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from agent.responder import ResponseGenerator
from tools.common.tool_metadata import ToolMetadata
from tools.loader import load_tools

from .context import AgentContext
from .decision import DecisionMaker
from .executor import (
    BaseExecutor,
    DefaultExecutor,
    ExecutionMode,
    ExecutionResult,
    ToolRegistry,
)
from .llm_planner import LLMPlanner
from .memory import MemoryManager
from .planner import BasePlanner, Plan, RuleBasedPlanner, StepStatus

logger = logging.getLogger(__name__)


class AgentState(Enum):
    IDLE = auto()
    PLANNING = auto()
    EXECUTING = auto()
    WAITING = auto()
    COMPLETED = auto()
    FAILED = auto()


class BaseAgent:
    def __init__(
        self,
        planner: BasePlanner | None = None,
        executor: BaseExecutor | None = None,
        memory: MemoryManager | None = None,
    ) -> None:
        self._memory = memory or MemoryManager()
        self._planner = planner or RuleBasedPlanner(memory=self._memory)
        self._executor = executor or DefaultExecutor(memory=self._memory)
        self._responder = ResponseGenerator()

        # 自动加载工具
        self._load_builtin_tools()
        self._decision = DecisionMaker()
        self._context = AgentContext()
        self._agent_context = self._context

    @property
    def context(self) -> AgentContext:
        return self._context

    @property
    def memory(self) -> MemoryManager:
        return self._memory

    @property
    def planner(self) -> BasePlanner:
        return self._planner

    @property
    def executor(self) -> BaseExecutor:
        return self._executor

    @property
    def tools(self) -> ToolRegistry:
        return self._executor.tools

    def register_tool(self, tool, handler=None) -> None:
        """
        Register tool.

        Compatible with:
        1. New style:
        register_tool(ToolMetadata)

        2. Legacy style:
        register_tool(name, handler)
        """

        from tools.common.tool_metadata import ToolMetadata

        # 新接口
        if isinstance(tool, ToolMetadata):

            self.tools.register(tool)

            logger.info("Registered ToolMetadata: %s", tool.name)

            return

        # 兼容旧接口
        if isinstance(tool, str) and handler is not None:

            metadata = ToolMetadata(
                name=tool,
                description=f"legacy tool: {tool}",
                handler=handler,
                parameters={},
            )

            self.tools.register(metadata)

            logger.info("Registered legacy tool: %s", tool)

            return

        raise TypeError("register_tool expects ToolMetadata " "or (name, handler)")

    def register_class_tool(self, tool_instance) -> None:

        self._executor.tools.register_class_tool(tool_instance)

    def _load_builtin_tools(self):
        """
        Load builtin Agent tools.
        Sync tools from global registry
        into Executor ToolRegistry.
        """
        from tools.common.tool import TOOL_REGISTRY
        from tools.loader import load_tools

        load_tools()
        for name, metadata in TOOL_REGISTRY.items():

            self._executor.tools.register(metadata)

        logger.info("Loaded tools: %s", list(TOOL_REGISTRY.keys()))

    async def run(
        self, goal: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # ==========================
        # v1.1 InputEvent Support
        # ==========================

        if hasattr(goal, "data"):

            event = goal

            if isinstance(event.data, dict):

                goal = event.data.get("text", "")

            else:

                goal = str(event.data)

        request_id = uuid.uuid4().hex[:8]

        conversation_memory = None
        response_stream_callback = None

        if context:
            conversation_memory = context.get("conversation_memory")
            response_stream_callback = context.get("response_stream_callback")

        self._context.goal = goal
        self._agent_context.goal = goal

        # 保存上下文到 Working Memory
        if context:
            for k, v in context.items():
                if k == "response_stream_callback":
                    continue
                self._memory.working.store(k, v)

        logger.info("[REQ-%s] Agent starting for goal: %s", request_id, goal)
        # ==========================
        # Step 4.5 Memory Compression
        # ==========================

        try:

            compressed = self._memory.compress_memory()

            if compressed:

                logger.debug("[Memory] Compression completed")

        except Exception as e:

            logger.warning("[Memory Compression Error] %s", e)
        plan = self._planner.create_plan(goal, self.tools.describe_tools(), context)

        # 保存当前执行计划到Context
        self._context.current_plan = plan

        self._memory.short_term.store(
            key=f"plan:{plan.plan_id}",
            value={"goal": goal, "steps": len(plan.steps)},
        )
        # 保存工具执行结果
        execution_outputs = []
        direct_response = (
            len(plan.steps) == 1 and plan.steps[0].action == "generate_response"
        )

        # Ordinary voice questions used to call the LLM twice: once through the
        # generate_response tool and again in ResponseGenerator.  Answer them
        # directly with the richer responder prompt and stream the only call.
        if direct_response:
            step = plan.steps[0]
            step.status = StepStatus.IN_PROGRESS
            response = self._responder.generate(
                goal,
                [],
                self._memory,
                request_id=request_id,
                conversation_memory=conversation_memory,
                on_token=response_stream_callback,
            )
            step.result = {"response": response}
            step.status = StepStatus.COMPLETED
            execution_outputs.append(
                {"step_id": step.step_id, "output": step.result, "error": None}
            )

        while not direct_response and not plan.is_complete():
            step = self._decision.next_step(plan)

            if step is None:
                break

            pending = [step]
            if not pending:
                failed = [s for s in plan.steps if s.status == StepStatus.FAILED]
                if failed:
                    feedback = "; ".join(s.error or "unknown error" for s in failed)

                    plan = self._planner.revise_plan(plan, feedback)

                    # 更新Context中的最新计划
                    self._context.current_plan = plan

                    continue
                break

            results = await self._executor.execute_steps(
                pending, mode=ExecutionMode.SEQUENTIAL
            )
            for r in results:

                # 更新PlanStep状态
                for s in plan.steps:
                    if s.step_id == r.step_id:

                        s.status = r.status

                        if r.error:
                            s.error = r.error

                        break

                self._agent_context.update(
                    {
                        "step_id": r.step_id,
                        "action": next(
                            (s.action for s in plan.steps if s.step_id == r.step_id),
                            "",
                        ),
                        "status": r.status.name,
                        "output": r.output,
                        "error": r.error,
                    }
                )
                execution_outputs.append(
                    {
                        "step_id": r.step_id,
                        "output": r.output,
                        "error": r.error,
                    }
                )
        success = plan.is_success()
        if not direct_response:
            response = self._responder.generate(
                goal,
                self._agent_context.history,
                self._memory,
                request_id=request_id,
                conversation_memory=conversation_memory,
                on_token=response_stream_callback,
            )
        episode = {
            "goal": goal,
            "response": response,
            "plan_summary": plan.summary(),
            "history": self._context.history,
            "success": success,
            "duration_s": 0,
            # 新增
            "outputs": execution_outputs,
        }
        self._memory.episodic.store(key=plan.plan_id, value=episode)
        self._memory.short_term.store(key=f"episode:{plan.plan_id}", value=episode)

        logger.info(
            "[REQ-%s] Agent finished | goal=%s | success=%s", request_id, goal, success
        )

        return episode

    def run_sync(
        self, goal: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(goal, context))

        import concurrent.futures

        future = asyncio.run_coroutine_threadsafe(self.run(goal, context), loop)
        return future.result()

    def stop(self) -> None:
        logger.info("Agent stopping...")
        self._context.state = AgentState.FAILED

    def reset(self) -> None:
        self._memory.clear_all()
        self._context = AgentContext()
        logger.info("Agent reset.")


Agent = BaseAgent
