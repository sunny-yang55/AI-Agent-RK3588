"""Vision capability package.

Platform-neutral interfaces must remain importable when optional development
backends (for example Ultralytics/PyTorch) are not installed on RK3588.
"""

from .factory import VisionFactory
from .manager import VisionPipelineManager
from .register import register_vision_tool
from .session import (
    VisionCommand,
    VisionSession,
    VisionSessionState,
    classify_vision_command,
)
from .service import OpenCVVisionWindow, VisionService
from .process_service import ProcessVisionService
from .tool import VisionTool


try:
    # The legacy YOLO backend is optional and is not part of the RK3588
    # production runtime. Importing it still registers the backend on
    # development machines where Ultralytics is installed.
    from .yolo import yolo_vision as _yolo_vision  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "ultralytics":
        raise


__all__ = [
    "VisionFactory",
    "VisionPipelineManager",
    "register_vision_tool",
    "VisionCommand",
    "VisionSession",
    "VisionSessionState",
    "classify_vision_command",
    "OpenCVVisionWindow",
    "ProcessVisionService",
    "VisionService",
    "VisionTool",
]
