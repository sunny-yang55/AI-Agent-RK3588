"""Full-camera pixel to robot-XY calibration with strict validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class CalibrationError(ValueError):
    """A calibration artifact is absent, incompatible, or unsafe to use."""


@dataclass(frozen=True)
class CameraRobotCalibration:
    """One fixed-camera mapping from **full-frame** pixels to arm XY in mm."""

    image_width: int
    image_height: int
    transform: np.ndarray
    transform_type: str
    rmse_mm: float
    max_error_mm: float
    point_count: int

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_image_size: tuple[int, int] = (1280, 720),
        max_rmse_mm: float = 3.0,
        max_error_mm: float = 5.0,
    ) -> "CameraRobotCalibration":
        artifact_path = Path(path)
        if not artifact_path.is_file():
            raise CalibrationError(f"机械臂标定文件不存在：{artifact_path}")
        try:
            data: dict[str, Any] = json.loads(artifact_path.read_text(encoding="utf-8"))
            width, height = (int(value) for value in data["image_size"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CalibrationError(f"机械臂标定文件格式错误：{exc}") from exc
        if (width, height) != expected_image_size:
            raise CalibrationError(
                "标定画面尺寸不匹配："
                f"文件为{width}x{height}，当前摄像头为"
                f"{expected_image_size[0]}x{expected_image_size[1]}；必须重新标定。"
            )
        report = data.get("report") or {}
        rmse = float(report.get("rmse_mm", float("inf")))
        worst = float(report.get("max_error_mm", float("inf")))
        points = len(data.get("pixel_pts") or [])
        if points < 9:
            raise CalibrationError(f"标定点不足：{points}个；全画面标定至少需要9个点。")
        if rmse > max_rmse_mm or worst > max_error_mm:
            raise CalibrationError(
                f"标定精度不合格：RMSE={rmse:.2f}mm，最大误差={worst:.2f}mm；"
                f"要求分别不高于{max_rmse_mm:.1f}mm和{max_error_mm:.1f}mm。"
            )
        transform_type = str(data.get("transform_type", ""))
        if transform_type == "homography":
            transform = np.asarray(data.get("homography"), dtype=np.float64)
            valid_shape = (3, 3)
        elif transform_type == "affine":
            transform = np.asarray(data.get("affine"), dtype=np.float64)
            valid_shape = (2, 3)
        else:
            raise CalibrationError("标定文件必须包含 affine 或 homography 变换矩阵。")
        if transform.shape != valid_shape or not np.isfinite(transform).all():
            raise CalibrationError("机械臂标定矩阵无效。")
        return cls(width, height, transform, transform_type, rmse, worst, points)

    def pixel_to_robot(self, x: int | float, y: int | float) -> tuple[float, float]:
        """Transform a full-frame pixel, never an ROI-relative pixel."""
        point = np.asarray([float(x), float(y), 1.0], dtype=np.float64)
        if self.transform_type == "homography":
            result = self.transform @ point
            if abs(result[2]) < 1e-9:
                raise CalibrationError("标定变换在该像素点无效。")
            return (round(float(result[0] / result[2]), 2), round(float(result[1] / result[2]), 2))
        result = self.transform @ point
        return (round(float(result[0]), 2), round(float(result[1]), 2))
