"""Voice-safe planning bridge for the robot arm.

No phrase in this module can command a motor.  It only asks the already-owned
vision process for stable facts and produces a plan for operator review.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tools.robot_arm import RobotArmCoordinator


class RobotArmVoiceController:
    """Keep grasp intent local, factual, and disabled until calibrated."""

    def __init__(
        self,
        vision_service,
        speak: Callable[..., object],
        *,
        calibration_path: str | Path,
    ) -> None:
        self.service = vision_service
        self._speak = speak
        self.coordinator = RobotArmCoordinator(calibration_path)

    def handle(self, text: str) -> bool:
        if not self.coordinator.is_pick_request(text):
            return False
        if not self.service.is_running:
            self._speak("摄像头尚未打开，无法确认抓取目标。请先说打开摄像头。", allow_interrupt=True)
            return True
        try:
            plan = self.coordinator.prepare_pick(text, self.service.locate_workbench())
        except Exception as exc:
            self._speak(f"抓取规划失败：{exc}。为安全起见，本次没有执行动作。", allow_interrupt=True)
            return True
        self._speak(plan.message, allow_interrupt=True)
        return True
