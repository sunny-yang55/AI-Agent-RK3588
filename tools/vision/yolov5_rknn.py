"""RKNNLite YOLOv5 detector based on Rockchip's official model-zoo layout."""

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)

COCO_ZH = {
    "person": "人", "bicycle": "自行车", "car": "汽车", "motorcycle": "摩托车",
    "airplane": "飞机", "bus": "公交车", "train": "火车", "truck": "卡车",
    "boat": "船", "traffic light": "交通灯", "bird": "鸟", "cat": "猫",
    "dog": "狗", "horse": "马", "backpack": "背包", "umbrella": "雨伞",
    "handbag": "手提包", "suitcase": "行李箱", "bottle": "瓶子",
    "cup": "杯子", "bowl": "碗", "banana": "香蕉", "apple": "苹果",
    "orange": "橙子", "chair": "椅子", "couch": "沙发", "bed": "床",
    "dining table": "餐桌", "toilet": "马桶", "tv": "电视",
    "laptop": "笔记本电脑", "mouse": "鼠标", "remote": "遥控器",
    "keyboard": "键盘", "cell phone": "手机", "microwave": "微波炉",
    "oven": "烤箱", "sink": "水槽", "refrigerator": "冰箱",
    "book": "书", "clock": "时钟", "vase": "花瓶", "scissors": "剪刀",
}

MEASURE_WORDS = {
    "人": "个", "自行车": "辆", "汽车": "辆", "摩托车": "辆",
    "飞机": "架", "公交车": "辆", "火车": "列", "卡车": "辆", "船": "艘",
    "雨伞": "把", "椅子": "把", "电视": "台", "笔记本电脑": "台",
    "键盘": "个", "鼠标": "个", "手机": "部", "书": "本", "剪刀": "把",
}

VISUAL_QUERY_ALIASES = {
    "本子": "书",
    "书本": "书",
    "手机": "手机",
    "电话": "手机",
    "笔记本电脑": "笔记本电脑",
}

ANCHORS = np.asarray(
    (
        ((10, 13), (16, 30), (33, 23)),
        ((30, 61), (62, 45), (59, 119)),
        ((116, 90), (156, 198), (373, 326)),
    ),
    dtype=np.float32,
)


@dataclass(frozen=True)
class YoloDetection:
    class_id: int
    label: str
    label_zh: str
    confidence: float
    box: tuple[int, int, int, int]


class TemporalDetectionStabilizer:
    """Suppress one-frame count changes while retaining recent boxes."""

    def __init__(self, window_size: int = 5) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self._history = deque(maxlen=window_size)

    def update(self, detections: list[YoloDetection]) -> list[YoloDetection]:
        self._history.append(list(detections))
        if len(self._history) < 3:
            return list(detections)
        labels = {item.label for frame in self._history for item in frame}
        selected = []
        for label in labels:
            counts = [sum(item.label == label for item in frame) for frame in self._history]
            stable_count = int(statistics.median(counts))
            candidates = [
                item
                for frame in reversed(self._history)
                for item in frame
                if item.label == label
            ]
            selected.extend(sorted(candidates, key=lambda item: item.confidence, reverse=True)[:stable_count])
        return sorted(selected, key=lambda item: item.confidence, reverse=True)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[index] + areas[order[1:]] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = order[np.where(iou <= threshold)[0] + 1]
    return np.asarray(keep, dtype=np.int64)


def _as_branch(output: np.ndarray) -> np.ndarray:
    output = np.asarray(output)
    if output.ndim != 4:
        raise ValueError(f"YOLO output must be 4D, got {output.shape}")
    if output.shape[1] == 255:
        return output.reshape(3, 85, output.shape[2], output.shape[3])
    if output.shape[-1] == 255:
        output = output.transpose(0, 3, 1, 2)
        return output.reshape(3, 85, output.shape[2], output.shape[3])
    raise ValueError(f"Unsupported YOLO output shape: {output.shape}")


def postprocess_yolov5(
    outputs: list[np.ndarray],
    *,
    confidence_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    input_size: int = 640,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(outputs) != 3:
        raise ValueError(f"YOLOv5 expects 3 output branches, got {len(outputs)}")
    all_boxes, all_classes, all_scores = [], [], []
    for branch_index, raw in enumerate(outputs):
        data = _as_branch(raw)
        grid_h, grid_w = data.shape[2:]
        col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
        grid = np.stack((col, row), axis=0)[None, ...]
        stride = np.asarray((input_size / grid_w, input_size / grid_h)).reshape(1, 2, 1, 1)
        anchor = ANCHORS[branch_index].reshape(3, 2, 1, 1)
        xy = (data[:, :2] * 2.0 - 0.5 + grid) * stride
        wh = np.square(data[:, 2:4] * 2.0) * anchor
        boxes = np.concatenate((xy - wh / 2.0, xy + wh / 2.0), axis=1)
        objectness = data[:, 4:5]
        class_scores = data[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = np.max(class_scores, axis=1) * objectness[:, 0]
        mask = scores >= confidence_threshold
        all_boxes.append(boxes.transpose(0, 2, 3, 1)[mask])
        all_classes.append(class_ids[mask])
        all_scores.append(scores[mask])
    boxes = np.concatenate(all_boxes) if all_boxes else np.empty((0, 4))
    classes = np.concatenate(all_classes) if all_classes else np.empty((0,), dtype=np.int64)
    scores = np.concatenate(all_scores) if all_scores else np.empty((0,))
    kept = []
    for class_id in np.unique(classes):
        indices = np.where(classes == class_id)[0]
        kept.extend(indices[_nms(boxes[indices], scores[indices], nms_threshold)].tolist())
    kept_array = np.asarray(kept, dtype=np.int64)
    return boxes[kept_array], classes[kept_array], scores[kept_array]


class RKNNYOLOv5Detector:
    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        confidence: float = 0.25,
        nms: float = 0.45,
        core: str | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        self.model_path = Path(
            model_path
            or root / "models/vision/yolov5s_relu-rk3588-fp.rknn"
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Vision RKNN model not found: {self.model_path}")
        from rknnlite.api import RKNNLite

        core_name = core or os.getenv("VISION_NPU_CORE", "1")
        masks = {
            "0": RKNNLite.NPU_CORE_0,
            "1": RKNNLite.NPU_CORE_1,
            "2": RKNNLite.NPU_CORE_2,
        }
        if core_name not in masks:
            raise ValueError("VISION_NPU_CORE must be 0, 1 or 2")
        self.confidence = confidence
        self.nms = nms
        self.last_latency_ms: float | None = None
        self._rknn = RKNNLite(verbose=False)
        code = self._rknn.load_rknn(str(self.model_path))
        if code != 0:
            raise RuntimeError(f"RKNNLite.load_rknn failed with code {code}")
        code = self._rknn.init_runtime(core_mask=masks[core_name])
        if code != 0:
            self._rknn.release()
            raise RuntimeError(f"RKNNLite.init_runtime failed with code {code}")

    @staticmethod
    def _letterbox(image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        import cv2

        height, width = image.shape[:2]
        scale = min(640 / width, 640 / height)
        resized_w, resized_h = round(width * scale), round(height * scale)
        resized = cv2.resize(image, (resized_w, resized_h))
        pad_x, pad_y = (640 - resized_w) / 2, (640 - resized_h) / 2
        left, top = int(round(pad_x - 0.1)), int(round(pad_y - 0.1))
        right, bottom = int(round(pad_x + 0.1)), int(round(pad_y + 0.1))
        output = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return cv2.cvtColor(output, cv2.COLOR_BGR2RGB), scale, float(left), float(top)

    def detect(self, image: np.ndarray) -> list[YoloDetection]:
        input_image, scale, pad_x, pad_y = self._letterbox(image)
        # RKNN Toolkit 2.3.2 exports this model with a static four-dimensional
        # NHWC input. RKNNLite does not add the batch dimension automatically.
        input_batch = np.expand_dims(input_image, axis=0)
        started = time.perf_counter()
        outputs = self._rknn.inference(inputs=[input_batch], data_format=["nhwc"])
        self.last_latency_ms = (time.perf_counter() - started) * 1000
        if outputs is None:
            raise RuntimeError("RKNNLite inference returned no outputs")
        boxes, classes, scores = postprocess_yolov5(
            outputs,
            confidence_threshold=self.confidence,
            nms_threshold=self.nms,
        )
        height, width = image.shape[:2]
        detections = []
        for box, class_id, score in zip(boxes, classes, scores):
            x1 = int(np.clip((box[0] - pad_x) / scale, 0, width - 1))
            y1 = int(np.clip((box[1] - pad_y) / scale, 0, height - 1))
            x2 = int(np.clip((box[2] - pad_x) / scale, 0, width - 1))
            y2 = int(np.clip((box[3] - pad_y) / scale, 0, height - 1))
            label = COCO_LABELS[int(class_id)]
            detections.append(
                YoloDetection(
                    int(class_id), label, COCO_ZH.get(label, label), float(score),
                    (x1, y1, x2, y2),
                )
            )
        return sorted(detections, key=lambda item: item.confidence, reverse=True)

    def close(self) -> None:
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None


def summarize_detections(
    detections: list[YoloDetection],
    *,
    minimum_confidence: float = 0.30,
) -> str:
    spoken = [item for item in detections if item.confidence >= minimum_confidence]
    if not spoken:
        return "当前画面中没有检测到已知物体。"
    counts: dict[str, int] = {}
    for detection in spoken:
        counts[detection.label_zh] = counts.get(detection.label_zh, 0) + 1
    parts = [
        f"{count}{MEASURE_WORDS.get(label, '个')}{label}"
        for label, count in counts.items()
    ]
    return "我看到" + "、".join(parts) + "。"


def answer_visual_query(text: str, detections: list[YoloDetection]) -> str:
    """Answer supported object-presence questions from grounded detections."""
    targets = []
    for phrase, label_zh in VISUAL_QUERY_ALIASES.items():
        if phrase in text and label_zh not in targets:
            targets.append(label_zh)
    if not targets:
        return summarize_detections(detections)
    visible = {
        item.label_zh
        for item in detections
        if item.confidence >= 0.30
    }
    found = [label for label in targets if label in visible]
    missing = [label for label in targets if label not in visible]
    parts = []
    if found:
        parts.append("看到了" + "、".join(found))
    if missing:
        parts.append("暂时没有看到" + "、".join(missing))
    return "，".join(parts) + "。"


def draw_detections(image: np.ndarray, detections: list[YoloDetection]) -> np.ndarray:
    """Draw the latest detections without modifying the captured frame."""
    import cv2

    annotated = image.copy()
    for item in detections:
        x1, y1, x2, y2 = item.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{item.label} {item.confidence:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated
