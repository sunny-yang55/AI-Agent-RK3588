from __future__ import annotations

import json
import logging
from typing import Any

from tools.llm.adapter import LLMAdapter

from .planner import (
    BasePlanner,
    Plan,
    PlanStep,
    StepStatus,
)

logger = logging.getLogger(__name__)


class LLMPlanner(BasePlanner):
    """
    LLM driven planner.

    Use LLM to generate dynamic execution plans.
    """

    def __init__(
        self,
        memory=None,
    ):
        super().__init__(memory)
        self.llm = LLMAdapter()

        super().__init__(memory)

        self.llm = LLMAdapter()

    def create_plan(
        self,
        goal: str,
        tools: list[dict],
        context: dict[str, Any] | None = None,
    ) -> Plan:

        tool_text = json.dumps(tools, ensure_ascii=False, indent=2)

        prompt = f"""
你是一名AI Agent Planner。你的任务是根据用户目标，选择合适工具生成执行计划。
当前可用工具：{tool_text}
规则：
1. action必须严格使用工具name。
2. 不允许创造不存在的工具。
3. 每一步params必须符合工具parameters定义。
4. 一个工具只能完成自己的职责。
5. 如果任务需要多个工具，可以生成多个步骤。

工具选择原则：请严格根据当前可用工具描述选择工具。
工具能力来自：{tool_text}
不要根据自己的知识创造工具。如果当前工具列表不存在某功能，不要生成对应 action。
如果任务需要多个工具，可以生成多个执行步骤。
返回格式：
[
 {{
   "action":"工具名称",
   "description":"执行说明",
   "params":{{}}
 }}
]
不要输出Markdown。
不要输出解释。
用户目标：

{goal}
"""
        print("\n===== PROMPT =====")
        print(prompt)
        response = self.llm.chat(prompt)
        print("\n===== LLM RAW RESPONSE =====")
        print(response)

        try:
            steps_json = json.loads(response)

        except Exception:

            logger.warning("LLM返回不是JSON，使用默认步骤")

            steps_json = []

        steps = []

        for index, item in enumerate(steps_json):

            steps.append(
                PlanStep(
                    step_id=f"llm_{index}",
                    action=item.get("action", "unknown"),
                    description=item.get("description", ""),
                    params=item.get("params", {}),
                    status=StepStatus.PENDING,
                )
            )

        plan = Plan(goal=goal, steps=steps, metadata={"planner": "llm"})

        return plan

    def revise_plan(
        self,
        plan: Plan,
        feedback: str,
    ) -> Plan:

        logger.warning(
            "Plan execution failed: %s",
            feedback,
        )

        return plan

    def next_step(
        self,
        goal: str,
        history: list[dict],
        tools: list[dict],
    ) -> dict:

        tool_text = json.dumps(
            tools,
            ensure_ascii=False,
            indent=2,
        )

        history_text = json.dumps(
            history,
            ensure_ascii=False,
            indent=2,
        )

        prompt = f"""
你是一名 AI Agent。
目标：
{goal}
已经完成：
{history_text}
可用工具：
{tool_text}
请决定下一步。
如果任务完成，请返回：
{{
  "action":"finish",
  "answer":"任务完成"
}}
否则返回：
{{
  "action":"工具名称",
  "params":{{}}
}}
不要输出Markdown。
不要输出解释。
"""

        response = self.llm.chat(prompt)

        try:
            return json.loads(response)

        except Exception:

            logger.warning("next_step parse failed")

            return {
                "action": "finish",
                "answer": "parse failed",
            }
