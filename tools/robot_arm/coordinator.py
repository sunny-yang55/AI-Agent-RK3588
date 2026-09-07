"""Convert stable workbench detections into non-executing, safe pick plans."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .calibration import CalibrationError, CameraRobotCalibration


_COLORS = {"红色": "red", "黄色": "yellow", "蓝色": "blue", "绿色": "green"}
_SHAPES = {
    "正方体": "cube",
    "圆柱体": "cylinder",
    "圆柱": "cylinder",
    "三棱锥": "triangular_pyramid",
    "三轮锥": "triangular_pyramid",
    "三菱锥": "triangular_pyramid",
    "三轮车": "triangular_pyramid",
}


@dataclass(frozen=True)
class PickPlan:
    """A reviewable plan only; it is not a movement command."""

    status: str
    message: str
    label: str | None = None
    camera_pixel: tuple[int, int] | None = None
    roi_pixel: tuple[int, int] | None = None
    robot_xy_mm: tuple[float, float] | None = None
    vision_verified: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RobotArmCoordinator:
    """Planning gate shared by voice, vision, and a future arm adapter.

    It intentionally has no serial dependency.  `prepare_pick` is safe to
    call on the RK3588 even before an arm is connected.
    """

    def __init__(
        self,
        calibration_path: str | Path,
        *,
        expected_image_size: tuple[int, int] = (1280, 720),
        workspace_x: tuple[float, float] = (-150.0, 150.0),
        workspace_y: tuple[float, float] = (100.0, 350.0),
    ) -> None:
        self.calibration_path = Path(calibration_path)
        self.expected_image_size = expected_image_size
        self.workspace_x = workspace_x
        self.workspace_y = workspace_y

    @staticmethod
    def target_from_text(text: str) -> tuple[str | None, str | None]:
        normalized = re.sub(r"[，。！？!?、\s]", "", text.replace("圆珠体", "圆柱体"))
        color = next((value for phrase, value in _COLORS.items() if phrase in normalized), None)
        shape = next((value for phrase, value in _SHAPES.items() if phrase in normalized), None)
        return color, shape

    @staticmethod
    def is_pick_request(text: str) -> bool:
        clean = text.replace(" ", "")
        return any(word in clean for word in ("抓取", "夹取", "拿取", "去拿", "抓一下", "夹一下"))

    def prepare_pick(self, text: str, objects: list[dict[str, Any]]) -> PickPlan:
        color, shape = self.target_from_text(text)
        if not color or not shape:
            return PickPlan(
                "rejected",
                "请明确说出颜色和形状，例如“抓取红色正方体”。",
            )
        matches = [
            item for item in objects
            if item.get("color") == color
            and item.get("shape") == shape
            and bool(item.get("shape_verified"))
        ]
        label = f"{color}_{shape}"
        if not matches:
            return PickPlan(
                "rejected",
                f"没有找到已稳定确认的{self._zh(color, shape)}，为避免误抓，本次不执行。",
                label=label,
            )
        if len(matches) != 1:
            return PickPlan(
                "rejected",
                f"检测到{len(matches)}个已确认的{self._zh(color, shape)}，目标不唯一，请先指定位置。",
                label=label,
            )
        target = matches[0]
        camera_pixel = self._pair(target.get("center_pixel"))
        roi_pixel = self._pair(target.get("center_roi"))
        if camera_pixel is None:
            return PickPlan("rejected", "视觉结果缺少全画面像素坐标，本次不执行。", label=label)
        try:
            calibration = CameraRobotCalibration.load(
                self.calibration_path,
                expected_image_size=self.expected_image_size,
            )
            robot_xy = calibration.pixel_to_robot(*camera_pixel)
        except CalibrationError as exc:
            return PickPlan(
                "needs_calibration",
                f"已找到{self._zh(color, shape)}，全画面像素坐标为{camera_pixel}；{exc}",
                label=label,
                camera_pixel=camera_pixel,
                roi_pixel=roi_pixel,
                vision_verified=True,
            )
        if not self._in_workspace(*robot_xy):
            return PickPlan(
                "rejected",
                f"目标换算为机械臂坐标{robot_xy}mm，超出安全工作区，本次不执行。",
                label=label,
                camera_pixel=camera_pixel,
                roi_pixel=roi_pixel,
                robot_xy_mm=robot_xy,
                vision_verified=True,
            )
        return PickPlan(
            "ready_for_confirmation",
            f"已找到{self._zh(color, shape)}，全画面像素坐标为{camera_pixel}，"
            f"预测机械臂坐标为{robot_xy}毫米。标定与工作区检查通过；"
            "当前为安全规划模式，等待您确认后才允许悬停动作。",
            label=label,
            camera_pixel=camera_pixel,
            roi_pixel=roi_pixel,
            robot_xy_mm=robot_xy,
            vision_verified=True,
        )

    def _in_workspace(self, x: float, y: float) -> bool:
        return self.workspace_x[0] <= x <= self.workspace_x[1] and self.workspace_y[0] <= y <= self.workspace_y[1]

    @staticmethod
    def _pair(value: Any) -> tuple[int, int] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        return (int(value[0]), int(value[1]))

    @staticmethod
    def _zh(color: str, shape: str) -> str:
        colors = {value: key for key, value in _COLORS.items()}
        shapes = {"cube": "正方体", "cylinder": "圆柱体", "triangular_pyramid": "三棱锥"}
        return f"{colors[color]}{shapes[shape]}"
