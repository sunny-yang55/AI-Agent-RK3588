"""Fixed overhead workbench ROI and colored-block perception."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


COLOR_RANGES = {
    "red": (((0, 80, 60), (10, 255, 255)), ((170, 80, 60), (180, 255, 255))),
    "yellow": (((18, 80, 60), (38, 255, 255)),),
    "blue": (((90, 80, 50), (135, 255, 255)),),
    "green": (((38, 60, 45), (90, 255, 255)),),
}
COLOR_ZH = {"red": "红色", "yellow": "黄色", "blue": "蓝色", "green": "绿色"}
SHAPE_ZH = {"cube": "正方体", "cylinder": "圆柱体", "triangular_pyramid": "三棱锥"}


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
    shape: str
    shape_zh: str
    center_pixel: tuple[int, int]
    center_roi: tuple[int, int]
    box: tuple[int, int, int, int]
    angle_deg: float
    area_pixels: float
    confidence: float
    shape_verified: bool = False


@dataclass(frozen=True)
class ColoredBlockCrop:
    """One padded, axis-aligned crop of a colour-segmented workbench block."""

    color: str
    image: np.ndarray
    box: tuple[int, int, int, int]
    area_pixels: float


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


def _color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    import cv2

    if color not in COLOR_RANGES:
        raise ValueError(f"unsupported block colour: {color}")
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.asarray(lower), np.asarray(upper))
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def extract_single_colored_block_crop(
    image: np.ndarray,
    color: str,
    *,
    min_area: float = 500.0,
    padding: int = 16,
) -> ColoredBlockCrop:
    """Crop exactly one labelled colour target or reject an ambiguous frame.

    Labelled shape samples must contain one target.  Refusing frames with no
    target or more than one target prevents unrelated objects and background
    pixels from silently entering the training set.
    """
    import cv2

    if image is None or image.size == 0:
        raise ValueError("empty workbench ROI")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = _color_mask(hsv, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one {color} block, found {len(candidates)}")
    contour = candidates[0]
    x, y, width, height = cv2.boundingRect(contour)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + width + padding)
    y2 = min(image.shape[0], y + height + padding)
    return ColoredBlockCrop(
        color=color,
        image=image[y1:y2, x1:x2].copy(),
        box=(x1, y1, x2, y2),
        area_pixels=float(cv2.contourArea(contour)),
    )


def classify_workbench_shapes(
    image: np.ndarray,
    detections: list[ColoredBlockDetection],
    classifier,
    *,
    padding: int = 16,
) -> list[ColoredBlockDetection]:
    """Attach reviewed-model shapes to detected colour blocks.

    A failed per-block prediction remains unverified and is deliberately not
    used for shape-specific speech or robot targeting.
    """
    classified = []
    for item in detections:
        x1, y1, x2, y2 = item.box
        crop_x1, crop_y1 = max(0, x1 - padding), max(0, y1 - padding)
        crop_x2 = min(image.shape[1], x2 + padding)
        crop_y2 = min(image.shape[0], y2 + padding)
        try:
            shape = classifier.predict(image[crop_y1:crop_y2, crop_x1:crop_x2])
            classified.append(
                replace(item, shape=shape, shape_zh=SHAPE_ZH[shape], shape_verified=True)
            )
        except (ValueError, KeyError):
            classified.append(item)
    return classified


class ColorBlockDetector:
    def __init__(self, roi: WorkbenchROI | None = None, *, min_area: float = 500.0):
        self.roi = roi
        self.min_area = min_area

    def detect(self, image: np.ndarray) -> list[ColoredBlockDetection]:
        import cv2

        roi = (self.roi or WorkbenchROI(0, 0, image.shape[1], image.shape[0])).clipped(image)
        crop = image[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        detections = []
        for color in COLOR_RANGES:
            mask = _color_mask(hsv, color)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.min_area:
                    continue
                (cx, cy), (width, height), angle = cv2.minAreaRect(contour)
                if width < height:
                    angle += 90.0
                x, y, w, h = cv2.boundingRect(contour)
                perimeter = float(cv2.arcLength(contour, True))
                vertices = len(cv2.approxPolyDP(contour, 0.04 * perimeter, True))
                circularity = (
                    4.0 * np.pi * area / (perimeter * perimeter)
                    if perimeter > 0 else 0.0
                )
                extent = area / max(1.0, float(w * h))
                aspect = width / max(1.0, height)
                if vertices == 3:
                    shape, shape_zh = "triangular_pyramid", "三棱锥"
                elif vertices >= 6 or (
                    0.70 <= aspect <= 1.43
                    and circularity >= 0.68
                    and extent < 0.88
                ):
                    shape, shape_zh = "cylinder", "圆柱体"
                else:
                    shape, shape_zh = "cube", "正方体"
                rectangle_area = max(1.0, float(width * height))
                detections.append(
                    ColoredBlockDetection(
                        color=color,
                        color_zh=COLOR_ZH[color],
                        shape=shape,
                        shape_zh=shape_zh,
                        center_pixel=(round(cx + roi.x), round(cy + roi.y)),
                        center_roi=(round(cx), round(cy)),
                        box=(x + roi.x, y + roi.y, x + w + roi.x, y + h + roi.y),
                        angle_deg=round(angle, 2),
                        area_pixels=area,
                        confidence=min(1.0, area / rectangle_area),
                    )
                )
        return sorted(detections, key=lambda item: item.area_pixels, reverse=True)


def summarize_colored_blocks(
    detections: list[ColoredBlockDetection], *, interactive: bool = False
) -> str:
    if not detections:
        answer = "我暂时没有在工作台上发现彩色物块。"
        return answer + "您可以调整物块位置后让我再看一次。" if interactive else answer
    counts = {}
    for item in detections:
        label = (
            f"{item.color_zh}{item.shape_zh}"
            if item.shape_verified else f"{item.color_zh}物块"
        )
        counts[label] = counts.get(label, 0) + 1
    parts = [
        f"{count}个{label}"
        for label, count in counts.items()
    ]
    answer = "我在工作台上看到" + "、".join(parts) + "。"
    if interactive:
        return answer + "需要我定位其中某一个，还是继续查看其他物品？"
    return answer


def answer_workbench_query(
    text: str,
    detections: list[ColoredBlockDetection],
) -> str:
    # Correct the common ASR substitutions only inside the workbench channel.
    # Do not pass these aliases to the general LLM conversation.
    normalized_text = (
        text.replace("圆珠体", "圆柱体")
        .replace("圆住体", "圆柱体")
        .replace("三轮车", "三棱锥")
        .replace("三棱椎", "三棱锥")
        .replace("三菱锥", "三棱锥")
        .replace("三轮锥", "三棱锥")
    )
    requested_colors = {color for color, label in COLOR_ZH.items() if label in normalized_text}
    shape_labels = {
        "cube": "正方体",
        "cylinder": "圆柱体",
        "triangular_pyramid": "三棱锥",
    }
    shape_aliases = {
        "正方体": "cube",
        "圆柱体": "cylinder",
        "圆柱": "cylinder",
        "三棱锥": "triangular_pyramid",
        "三轮锥": "triangular_pyramid",
        "三菱锥": "triangular_pyramid",
    }
    requested_shapes = {
        shape for phrase, shape in shape_aliases.items() if phrase in normalized_text
    }
    wants_location = any(
        word in normalized_text
        for word in ("定位", "位置", "坐标", "在哪", "哪里", "什么地方", "哪个地方")
    )
    if wants_location:
        matches = [
            item for item in detections
            if not requested_colors or item.color in requested_colors
        ]
        if not requested_colors:
            return "请告诉我需要定位哪一种颜色和形状的物块，例如“红色正方体在哪里”。"
        target = "、".join(COLOR_ZH[color] for color in sorted(requested_colors))
        if not matches:
            return f"我暂时没有看到{target}物块，因此不能给出坐标。您可以调整摆放后让我重新查看。"
        positions = "；".join(
            f"({item.center_roi[0]}, {item.center_roi[1]})" for item in matches
        )
        if requested_shapes:
            shape_text = "、".join(shape_labels[shape] for shape in requested_shapes)
            verified_matches = [
                item for item in matches
                if item.shape_verified and item.shape in requested_shapes
            ]
            if verified_matches:
                positions = "；".join(
                    f"({item.center_roi[0]}, {item.center_roi[1]})" for item in verified_matches
                )
                return (
                    f"我已经找到{len(verified_matches)}个{target}{shape_text}，"
                    f"工作台像素坐标为{positions}。"
                    "坐标原点是标定板左上角，尚未换算为机械臂坐标。"
                    "需要我继续定位其他物块吗？"
                )
            return (
                f"当前还不能可靠区分{target}物块是否为{shape_text}，"
                f"但检测到{len(matches)}个{target}物块，坐标为{positions}。"
                "为避免抓错，我建议先按颜色定位，或让我重新确认一次。"
            )
        return (
            f"我已经找到{len(matches)}个{target}物块，工作台像素坐标为{positions}。"
            "坐标原点是标定板左上角，尚未换算为机械臂坐标。"
            "还需要我继续定位其他物块吗？"
        )
    if requested_shapes:
        requested_color_objects = [
            item for item in detections
            if not requested_colors or item.color in requested_colors
        ]
        verified_matches = [
            item for item in requested_color_objects
            if item.shape_verified and item.shape in requested_shapes
        ]
        if verified_matches:
            return summarize_colored_blocks(verified_matches, interactive=True).replace("我在工作台上", "")
        if requested_color_objects:
            color_text = "、".join(
                sorted({item.color_zh for item in requested_color_objects})
            )
            return f"我看到了{color_text}物块，但形状还不够稳定。为了避免误判，您可以让我再确认一次。"
        target_color = "".join(COLOR_ZH[color] for color in sorted(requested_colors))
        return f"我暂时没有看到{target_color}物块。需要我重新查看工作台吗？"
    if not requested_colors and not requested_shapes:
        return summarize_colored_blocks(detections, interactive=True)
    matches = [
        item for item in detections
        if (not requested_colors or item.color in requested_colors)
        and (not requested_shapes or item.shape in requested_shapes)
    ]
    if matches:
        return summarize_colored_blocks(matches, interactive=True).replace("我在工作台上", "")
    target = "".join(COLOR_ZH[color] for color in requested_colors)
    target += "".join(shape_labels[shape] for shape in requested_shapes)
    if requested_colors and not requested_shapes:
        target += "物块"
    return f"我暂时没有看到{target}。您可以调整它的位置后让我再确认。"


def select_stable_workbench_snapshot(
    history: list[list[ColoredBlockDetection]],
) -> list[ColoredBlockDetection]:
    """Choose the most frequent recent color/shape scene, newest on ties."""
    if not history:
        return []
    signatures = [
        tuple(sorted((item.color, item.shape) for item in frame))
        for frame in history
    ]
    winning_count = max(Counter(signatures).values())
    winning = {
        signature for signature, count in Counter(signatures).items()
        if count == winning_count
    }
    for index in range(len(history) - 1, -1, -1):
        if signatures[index] in winning:
            return list(history[index])
    return []


def stabilize_verified_workbench_shapes(
    history: list[list[ColoredBlockDetection]],
    *,
    minimum_votes: int = 4,
    max_center_distance: float = 40.0,
) -> list[ColoredBlockDetection]:
    """Return newest objects with only temporally agreed model shapes verified.

    Objects are matched by colour and nearby ROI centre, so multiple blocks of
    the same colour do not share a vote. A shape must appear in at least four
    of the last five frames; otherwise only the safe colour-only result stays
    available for speech and robot integrations.
    """
    if not history:
        return []
    newest = history[-1]
    stable = []
    for current in newest:
        votes: Counter[str] = Counter()
        for frame in history:
            candidates = [
                item for item in frame
                if item.color == current.color and item.shape_verified
            ]
            if not candidates:
                continue
            nearest = min(
                candidates,
                key=lambda item: (
                    (item.center_roi[0] - current.center_roi[0]) ** 2
                    + (item.center_roi[1] - current.center_roi[1]) ** 2
                ),
            )
            distance = (
                (nearest.center_roi[0] - current.center_roi[0]) ** 2
                + (nearest.center_roi[1] - current.center_roi[1]) ** 2
            ) ** 0.5
            if distance <= max_center_distance:
                votes[nearest.shape] += 1
        if votes:
            shape, count = votes.most_common(1)[0]
            if count >= minimum_votes:
                stable.append(
                    replace(current, shape=shape, shape_zh=SHAPE_ZH[shape], shape_verified=True)
                )
                continue
        stable.append(replace(current, shape_verified=False))
    return stable


def is_workbench_query(text: str) -> bool:
    return any(
        phrase in text
        for phrase in ("工作台", "桌面", "桌上", "物块", "方块", "正方体", "圆柱", "圆珠体", "三棱锥", "三轮锥", "三菱锥", "三轮车", "红色", "黄色", "蓝色", "绿色")
    )


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
    colors = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "blue": (255, 0, 0),
        "green": (0, 180, 0),
    }
    for item in detections:
        x1, y1, x2, y2 = item.box
        color = colors[item.color]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.circle(output, item.center_pixel, 5, color, -1)
        cv2.putText(output, f"{item.color} {item.shape} {item.center_pixel}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return output
