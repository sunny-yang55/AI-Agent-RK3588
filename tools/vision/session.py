"""Voice-facing vision session commands and lifecycle state."""

from __future__ import annotations

import re
import threading
from enum import Enum


class VisionCommand(str, Enum):
    OPEN = "open"
    CLOSE = "close"
    DESCRIBE = "describe"


class VisionSessionState(str, Enum):
    CLOSED = "closed"
    STARTING = "starting"
    ACTIVE = "active"
    STOPPING = "stopping"
    ERROR = "error"


_CLOSE_PHRASES = (
    "关闭摄像头",
    "关掉摄像头",
    "停止摄像头",
    "结束视觉",
    "关闭视觉",
    "不看了",
    "别看了",
)
_OPEN_PHRASES = (
    "打开摄像头",
    "开启摄像头",
    "启动摄像头",
    "打开视觉",
    "开启视觉",
)
_DESCRIBE_PHRASES = (
    "前面有什么",
    "画面里有什么",
    "看到了什么",
    "看到什么",
    "这是什么",
    "识别一下",
    "看下前面",
    "有什么东西",
    "有什么物体",
    "有哪些东西",
    "有哪些物块",
)
_ACTIVE_VISUAL_FOLLOWUPS = (
    "没有看到",
    "没看到",
    "没有看见",
    "没看见",
    "看到了吗",
    "看到吗",
    "看见了吗",
    "看见吗",
    "有没有看到",
    "有没有看见",
    "能看到吗",
    "能看见吗",
    "看一下",
    "看一看",
    "帮我看",
    "你看看",
)


def _normalize(text: str) -> str:
    text = re.sub(r"<\|.*?\|>", "", text)
    return re.sub(r"[。！？!?，,；;：:\s]", "", text).strip()


def classify_vision_command(
    text: str,
    *,
    active: bool = False,
) -> VisionCommand | None:
    """Classify explicit local vision controls without invoking the LLM."""
    clean = _normalize(text).replace("小安", "")
    if any(phrase in clean for phrase in _CLOSE_PHRASES):
        return VisionCommand.CLOSE
    if active and any(phrase in clean for phrase in _DESCRIBE_PHRASES):
        return VisionCommand.DESCRIBE
    if active and any(phrase in clean for phrase in _ACTIVE_VISUAL_FOLLOWUPS):
        return VisionCommand.DESCRIBE
    workbench_terms = (
        "桌上", "桌面", "工作台", "物块", "方块",
        "正方体", "圆柱", "三棱锥", "红色", "黄色", "蓝色", "绿色",
    )
    if active and any(term in clean for term in workbench_terms) and any(
        word in clean for word in ("有没", "有什么", "有哪些", "看到", "看见")
    ):
        return VisionCommand.DESCRIBE
    if any(phrase in clean for phrase in _OPEN_PHRASES):
        return VisionCommand.OPEN
    if active and clean in {"关掉", "关闭", "结束", "停下"}:
        return VisionCommand.CLOSE
    return None


class VisionSession:
    """Thread-safe lifecycle state independent of camera, GUI, and detector."""

    def __init__(self) -> None:
        self._state = VisionSessionState.CLOSED
        self._error: str | None = None
        self._lock = threading.RLock()

    @property
    def state(self) -> VisionSessionState:
        with self._lock:
            return self._state

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    @property
    def is_active(self) -> bool:
        return self.state in {
            VisionSessionState.STARTING,
            VisionSessionState.ACTIVE,
        }

    def request_start(self) -> bool:
        with self._lock:
            if self._state in {
                VisionSessionState.STARTING,
                VisionSessionState.ACTIVE,
            }:
                return False
            self._state = VisionSessionState.STARTING
            self._error = None
            return True

    def mark_active(self) -> None:
        self._transition(VisionSessionState.STARTING, VisionSessionState.ACTIVE)

    def request_stop(self) -> bool:
        with self._lock:
            if self._state in {
                VisionSessionState.CLOSED,
                VisionSessionState.STOPPING,
            }:
                return False
            self._state = VisionSessionState.STOPPING
            return True

    def mark_closed(self) -> None:
        with self._lock:
            self._state = VisionSessionState.CLOSED
            self._error = None

    def mark_error(self, error: Exception | str) -> None:
        with self._lock:
            self._state = VisionSessionState.ERROR
            self._error = str(error)

    def _transition(
        self,
        expected: VisionSessionState,
        target: VisionSessionState,
    ) -> None:
        with self._lock:
            if self._state is not expected:
                raise RuntimeError(
                    f"invalid vision transition: {self._state.value} -> {target.value}"
                )
            self._state = target
