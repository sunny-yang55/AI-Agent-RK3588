"""Fixed overhead workbench ROI and colored-block perception."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


COLOR_RANGES = {
    "red": (((0, 80, 60), (10, 255, 255)), ((170, 80, 60), (180, 255, 255))),
    "yellow": (((18, 80, 60), (38, 255, 255)),),
    "blue": (((90, 80, 50), (135, 255, 255)),),
}
COLOR_ZH = {"red": "红色", "yellow": "黄色", "blue": "蓝色"}


@dataclass(frozen=True)
class WorkbenchROI:
    x: int
    y: int
    width: int
    height: int

    def clipped(self, image: np.ndarray) -> "WorkbenchROI":
        image_height, image_width = image.shape[:2]
        x = min(max(0, self.x), image_width - 1)
        y = min(max(0, self.y), image_height - 1)
        width = min(max(1, self.width), image_width - x)
        height = min(max(1, self.height), image_height - y)
        return WorkbenchROI(x, y, width, height)


@dataclass(frozen=True)
class ColoredBlockDetection:
    color: str
    color_zh: str
    center_pixel: tuple[int, int]
    center_roi: tuple[int, int]
    box: tuple[int, int, int, int]
    angle_deg: float
    area_pixels: float
    confidence: float


def load_workbench_roi(path: str | Path) -> WorkbenchROI | None:
    config_path = Path(path)
    if not config_path.is_file():
        return None
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return WorkbenchROI(*(int(data[key]) for key in ("x", "y", "width", "height")))


def save_workbench_roi(path: str | Path, roi: WorkbenchROI) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(roi.__dict__, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class ColorBlockDetector:
    def __init__(self, roi: WorkbenchROI | None = None, *, min_area: float = 500.0):
        self.roi = roi
        self.min_area = min_area

    def detect(self, image: np.ndarray) -> list[ColoredBlockDetection]:
        import cv2

        roi = (self.roi or WorkbenchROI(0, 0, image.shape[1], image.shape[0])).clipped(image)
        crop = image[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), dtype=np.uint8)
        detections = []
        for color, ranges in COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, np.asarray(lower), np.asarray(upper))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.min_area:
                    continue
                (cx, cy), (width, height), angle = cv2.minAreaRect(contour)
                if width < height:
                    angle += 90.0
                x, y, w, h = cv2.boundingRect(contour)
                rectangle_area = max(1.0, float(width * height))
                detections.append(
                    ColoredBlockDetection(
                        color=color,
                        color_zh=COLOR_ZH[color],
                        center_pixel=(round(cx + roi.x), round(cy + roi.y)),
                        center_roi=(round(cx), round(cy)),
                        box=(x + roi.x, y + roi.y, x + w + roi.x, y + h + roi.y),
                        angle_deg=round(angle, 2),
                        area_pixels=area,
                        confidence=min(1.0, area / rectangle_area),
                    )
                )
        return sorted(detections, key=lambda item: item.area_pixels, reverse=True)


def summarize_colored_blocks(detections: list[ColoredBlockDetection]) -> str:
    if not detections:
        return "工作台上暂时没有检测到彩色物块。"
    counts = {}
    for item in detections:
        counts[item.color_zh] = counts.get(item.color_zh, 0) + 1
    parts = [f"{count}个{color}物块" for color, count in counts.items()]
    return "我在工作台上看到" + "、".join(parts) + "。"


def draw_workbench_detections(
    image: np.ndarray,
    roi: WorkbenchROI,
    detections: list[ColoredBlockDetection],
) -> np.ndarray:
    import cv2

    output = image.copy()
    clipped = roi.clipped(image)
    cv2.rectangle(output, (clipped.x, clipped.y),
                  (clipped.x + clipped.width, clipped.y + clipped.height), (255, 255, 0), 2)
    colors = {"red": (0, 0, 255), "yellow": (0, 255, 255), "blue": (255, 0, 0)}
    for item in detections:
        x1, y1, x2, y2 = item.box
        color = colors[item.color]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.circle(output, item.center_pixel, 5, color, -1)
        cv2.putText(output, f"{item.color} {item.center_pixel}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return output
