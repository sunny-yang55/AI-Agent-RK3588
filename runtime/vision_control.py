"""Local voice-command bridge for the visual child process."""

from __future__ import annotations

from collections.abc import Callable

from tools.vision.session import (
    VisionCommand,
    classify_vision_command,
    is_broad_scene_query,
)


_WORKBENCH_TERMS = ("工作台", "桌上", "桌面", "物块", "方块", "正方体", "圆柱", "三棱锥")


def _is_workbench_request(text: str) -> bool:
    return any(term in text.replace(" ", "") for term in _WORKBENCH_TERMS)


class VisionVoiceController:
    """Handle visual controls locally so they never reach the LLM."""

    def __init__(self, service, speak: Callable[..., object], *, scene_describer=None) -> None:
        self.service = service
        self._speak = speak
        self._scene_describer = scene_describer

    def handle(self, text: str) -> bool:
        was_running = self.service.is_running
        command = classify_vision_command(text, active=was_running)
        if command is None:
            return False

        if command is VisionCommand.OPEN:
            if was_running:
                self._speak("摄像头已经打开。", allow_interrupt=True)
            elif self.service.start():
                self._speak("摄像头已打开。", allow_interrupt=True)
            else:
                reason = self.service.session.error or "未知错误"
                self._speak(f"摄像头打开失败：{reason}", allow_interrupt=True)
            return True

        if command is VisionCommand.DESCRIBE:
            if not was_running:
                self._speak(
                    "摄像头尚未打开，请先说打开摄像头。",
                    allow_interrupt=True,
                )
                return True
            try:
                details = self._describe_details(text)
                message = self._describe_scene(text, details)
            except Exception as exc:
                message = f"视觉识别失败：{exc}"
            self._speak(message, allow_interrupt=True)
            return True

        if was_running:
            stopped = self.service.stop()
            message = "摄像头已关闭。" if stopped else "摄像头关闭失败，请检查设备。"
        else:
            self.service.stop()
            message = "摄像头已经关闭。"
        self._speak(message, allow_interrupt=True)
        return True

    def _describe_details(self, text: str) -> dict:
        details_method = getattr(self.service, "describe_details", None)
        if details_method is not None:
            return dict(details_method(text))
        return {"summary": self.service.describe(text), "detections": [], "workbench_objects": []}

    def _describe_scene(self, text: str, details: dict) -> str:
        """Use cloud narration only for broad scene questions and fail locally."""
        describer = self._scene_describer
        if describer is None or _is_workbench_request(text):
            return str(details["summary"])
        if not (describer.is_available and is_broad_scene_query(text)):
            return str(details["summary"])
        try:
            image_jpeg = self.service.snapshot()
            return describer.describe(image_jpeg, text, details)
        except Exception as exc:
            # A network/API failure must never take down local perception.
            print(f"[Vision] Online scene description unavailable: {exc}")
            return str(details["summary"])

    def close(self) -> None:
        """Silently release vision resources during runtime shutdown."""
        if self.service.is_running:
            self.service.stop()
