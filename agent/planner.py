from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Protocol

from .memory import MemoryManager

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class PlanStep:
    step_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    action: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str | None = None
    retries: int = 0
    max_retries: int = 3

    @property
    def is_ready(self) -> bool:
        return self.status == StepStatus.PENDING

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            StepStatus.COMPLETED,
            StepStatus.FAILED,
            StepStatus.SKIPPED,
        )


@dataclass
class Plan:
    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    goal: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=__import__("time").time)

    def pending_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == StepStatus.PENDING]

    def in_progress_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == StepStatus.IN_PROGRESS]

    def next_available(self) -> list[PlanStep]:
        completed_ids = {
            s.step_id for s in self.steps if s.status == StepStatus.COMPLETED
        }
        return [
            s
            for s in self.steps
            if s.status == StepStatus.PENDING
            and all(dep in completed_ids for dep in s.depends_on)
        ]

    def is_complete(self) -> bool:
        return all(s.is_terminal for s in self.steps)

    def is_success(self) -> bool:
        return all(s.status == StepStatus.COMPLETED for s in self.steps)

    def summary(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "total_steps": len(self.steps),
            "completed": sum(1 for s in self.steps if s.status == StepStatus.COMPLETED),
            "failed": sum(1 for s in self.steps if s.status == StepStatus.FAILED),
            "pending": sum(1 for s in self.steps if s.status == StepStatus.PENDING),
            "is_complete": self.is_complete(),
            "is_success": self.is_success(),
        }


class TaskDecomposer(Protocol):
    def __call__(self, goal: str, context: dict[str, Any]) -> list[dict[str, Any]]: ...


class BasePlanner(ABC):
    def __init__(self, memory: MemoryManager | None = None) -> None:
        self._memory = memory

    @abstractmethod
    def create_plan(self, goal: str, context: dict[str, Any] | None = None) -> Plan: ...

    @abstractmethod
    def revise_plan(self, plan: Plan, feedback: str) -> Plan: ...

    def evaluate_step(self, step: PlanStep, result: Any) -> StepStatus:
        if result is None:
            return StepStatus.FAILED
        return StepStatus.COMPLETED


class RuleBasedPlanner(BasePlanner):
    """Simple planner that decomposes goals via pattern-keyword heuristics."""

    STEP_PATTERNS: dict[str, list[dict[str, Any]]] = {
        "analyze": [
            {
                "action": "read_project",
                "description": "扫描项目目录结构",
            },
            {
                "action": "analyze_code",
                "description": "执行AST代码分析",
            },
            {
                "action": "generate_report",
                "description": "生成Markdown分析报告",
            },
        ],
        "build": [
            {
                "action": "setup_environment",
                "description": "Set up the working environment",
            },
            {"action": "implement_core", "description": "Implement the core logic"},
            {
                "action": "integrate_components",
                "description": "Integrate all components",
            },
            {
                "action": "validate_output",
                "description": "Validate and verify the output",
            },
        ],
        "fix": [
            {
                "action": "reproduce_issue",
                "description": "Reproduce and isolate the issue",
            },
            {"action": "diagnose_root_cause", "description": "Diagnose the root cause"},
            {"action": "apply_fix", "description": "Apply the fix"},
            {
                "action": "verify_fix",
                "description": "Verify the fix resolves the issue",
            },
        ],
        "deploy": [
            {"action": "validate_artifacts", "description": "Validate build artifacts"},
            {"action": "run_tests", "description": "Run test suite"},
            {
                "action": "deploy_to_target",
                "description": "Deploy to target environment",
            },
            {"action": "smoke_test", "description": "Run smoke tests post-deployment"},
        ],
        "default": [
            {
                "action": "understand_task",
                "description": "Understand the task requirements",
            },
            {"action": "execute_task", "description": "Execute the primary task"},
            {
                "action": "review_result",
                "description": "Review and validate the result",
            },
            {"action": "report_completion", "description": "Report completion status"},
        ],
    }

    # ==========================
    # Local Rule Planner
    # ==========================
    def _rule_plan(self, goal: str) -> Plan:
        """
        Generate plan for simple natural language tasks.
        Used by speech interaction.
        """

        goal_lower = goal.lower()

        steps = []

        # ---------
        # 1. 问候
        # ---------

        if any(
            k in goal_lower
            for k in [
                "你好",
                "您好",
                "hello",
                "hi",
            ]
        ):

            steps.append(
                PlanStep(
                    step_id="greeting",
                    description="生成问候回复",
                    action="generate_response",
                    params={"goal": goal},
                )
            )

        # ---------
        # 2. 时间查询
        # ---------

        elif any(
            k in goal_lower
            for k in [
                "时间",
                "几点",
            ]
        ):

            steps.append(
                PlanStep(
                    step_id="time",
                    description="回答时间查询",
                    action="generate_response",
                    params={"goal": goal},
                )
            )

        # ---------
        # 3. 视觉请求
        # ---------

        elif any(
            k in goal_lower
            for k in [
                "摄像头",
                "图像",
                "图片",
                "识别",
            ]
        ):

            steps.append(
                PlanStep(
                    step_id="vision",
                    description="调用视觉工具",
                    action="vision",
                    params={},
                )
            )

        # ---------
        # 4. 默认
        # ---------

        else:

            steps.append(
                PlanStep(
                    step_id="default",
                    description="处理用户请求",
                    action="generate_response",
                    params={"goal": goal},
                )
            )

        plan = Plan(goal=goal, steps=steps, metadata={"planner": "local_rule"})

        logger.info("Created local rule plan for goal: %s", goal)

        return plan

    def _llm_plan(self, goal: str, context: dict[str, Any] | None = None) -> Plan:
        """
        Temporary LLM planner fallback.

        Later replaced by real LLM adapter.
        """

        logger.info("Fallback planner used for goal: %s", goal)

        steps = [
            PlanStep(
                step_id="general_response",
                description="生成用户回复",
                action="generate_response",
                params={"goal": goal},
            )
        ]

        return Plan(goal=goal, steps=steps, metadata={"planner": "fallback"})

    # ==========================
    # Fast Planner
    # ==========================

    def is_fast_task(self, goal: str) -> bool:

        keywords = [
            "analyze",
            "analysis",
            "project",
            "项目",
            "分析",
            "扫描",
        ]

        text = goal.lower()

        return any(k in text for k in keywords)

    def build_fast_plan(self, goal: str) -> Plan:

        steps = [
            PlanStep(
                step_id="scan_project",
                action="read_project",
                description="扫描项目目录结构",
                params={"project_path": "."},
            ),
            PlanStep(
                step_id="analyze_code",
                action="analyze_code",
                description="AST分析Python代码",
                params={"project_path": "."},
                depends_on=["scan_project"],
            ),
            PlanStep(
                step_id="generate_report",
                action="generate_report",
                description="生成Markdown分析报告",
                params={"project_path": ".", "output": "ai_agent_analysis_report.md"},
                depends_on=["analyze_code"],
            ),
        ]

        return Plan(goal=goal, steps=steps)

    def _is_simple_task(self, goal: str) -> bool:
        """
        判断是否属于简单本地规则任务。
        这类任务不调用LLM，直接本地处理。
        """

        simple_keywords = [
            # greeting
            "你好",
            "您好",
            "hello",
            "hi",
            # time
            "时间",
            "几点",
            "现在几点",
            # vision
            "摄像头",
            "看一下",
            "识别",
            "图像",
            "图片",
        ]

        goal_lower = goal.lower()

        return any(keyword in goal_lower for keyword in simple_keywords)

    def create_plan(
        self, goal: str, tools=None, context: dict[str, Any] | None = None
    ) -> Plan:

        # ==========================
        # Fast path
        # ==========================

        if self.is_fast_task(goal):

            return self.build_fast_plan(goal)

        ctx = context or {}

        if self._memory:
            ctx.update(self._memory.contextualize(goal))
        # ==============================
        # Local Rule Planner 优先
        # ==============================

        if self._is_simple_task(goal):

            logger.info("Use local rule planner for task: %s", goal)

            return self._rule_plan(goal)

        # ==============================
        # Complex Task -> LLM Planner
        # ==============================

        logger.info("Use LLM planner for task: %s", goal)

        return self._llm_plan(goal, ctx)

        pattern_key = "default"
        for keyword in self.STEP_PATTERNS:
            if keyword != "default" and keyword in goal.lower():
                pattern_key = keyword
                break

        template = self.STEP_PATTERNS[pattern_key]
        steps = [
            PlanStep(
                step_id=f"{pattern_key}_{i}",
                description=tmpl.get("description", ""),
                action=tmpl.get("action", ""),
                params=tmpl.get("params", {}),
            )
            for i, tmpl in enumerate(template)
        ]

        plan = Plan(
            goal=goal, steps=steps, metadata={"context": ctx, "pattern": pattern_key}
        )
        logger.info(
            "Created plan %s for goal: %s (pattern: %s)",
            plan.plan_id,
            goal,
            pattern_key,
        )
        return plan

    def revise_plan(self, plan: Plan, feedback: str) -> Plan:
        remaining = [
            s for s in plan.steps if s.status in (StepStatus.PENDING, StepStatus.FAILED)
        ]
        if not remaining:
            logger.info("Plan %s already complete; nothing to revise.", plan.plan_id)
            return plan

        for step in remaining:
            if step.status == StepStatus.FAILED:
                step.status = StepStatus.PENDING
                step.retries += 1
                step.error = None

        diagnostic_step = PlanStep(
            step_id=f"revise_{len(plan.steps)}",
            description=f"Handle feedback: {feedback}",
            action="generate_response",
            params={"feedback": feedback},
        )
        plan.steps.insert(0, diagnostic_step)
        plan.metadata.setdefault("revisions", []).append(feedback)
        logger.info("Revised plan %s with feedback: %s", plan.plan_id, feedback)
        return plan


Planner = RuleBasedPlanner
