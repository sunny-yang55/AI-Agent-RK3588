from typing import Any


class CapabilityRouter:
    """
    根据任务选择Capability
    """

    def __init__(self, manager):

        self.manager = manager

    def route(self, goal: str, **kwargs) -> dict[str, Any]:

        goal_lower = goal.lower()

        if any(
            k in goal_lower for k in ["图片", "图像", "视觉", "识别", "检测", "目标"]
        ):

            capability = "vision"

        elif any(k in goal_lower for k in ["移动", "运动", "导航", "机器人"]):

            capability = "robot"

        elif any(k in goal_lower for k in ["语音", "声音", "说话"]):

            capability = "speech"

        else:

            capability = "llm"

        return {"capability": capability, "goal": goal, "kwargs": kwargs}

    def execute(self, goal: str, **kwargs):

        route = self.route(goal, **kwargs)

        capability = self.manager.get(route["capability"])

        if capability is None:

            return {"status": "error", "message": "Capability not found"}

        capability.initialize()

        result = capability.execute(**kwargs)

        capability.shutdown()

        return result
