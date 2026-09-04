"""Local voice-command bridge for the visual child process."""

from __future__ import annotations

from collections.abc import Callable

from tools.vision.session import VisionCommand, classify_vision_command


class VisionVoiceController:
    """Handle visual controls locally so they never reach the LLM."""

    def __init__(self, service, speak: Callable[..., object]) -> None:
        self.service = service
        self._speak = speak

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
            if not was_running and not self.service.start():
                reason = self.service.session.error or "未知错误"
                self._speak(f"摄像头打开失败：{reason}", allow_interrupt=True)
                return True
            try:
                message = self.service.describe(text)
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

    def close(self) -> None:
        """Silently release vision resources during runtime shutdown."""
        if self.service.is_running:
            self.service.stop()
