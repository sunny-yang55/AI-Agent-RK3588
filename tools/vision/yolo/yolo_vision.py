"""
YOLO Vision
Minimal implementation.
Sprint B1.5.4
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from tools.vision.registry import VisionRegistry
from tools.vision.result import (
    BoundingBox,
    Detection,
    DetectionResult,
)
from tools.vision.vision_base import VisionBase


class YOLOVision(VisionBase):
    """
    YOLO Vision implementation.
    """

    name = "yolo_vision"
    description = "YOLO object detector"
    version = "0.1.0"
    capabilities = frozenset(
        {
            "object_detection",
            "yolo",
        }
    )

    def __init__(
        self,
        model_path: str | Path = "models/vision/yolov8n.pt",
        confidence: float = 0.25,
    ):
        """
        Parameters
        ----------
        model_path
            YOLO model path.
        """
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.model: YOLO | None = None

    async def initialize(self) -> None:
        """
        Load YOLO model.
        """
        if self.model is None:

            self.model = YOLO(str(self.model_path))

    async def detect(
        self,
        image: Any,
        **kwargs,
    ) -> DetectionResult:
        """
        Run YOLO inference.
        """
        if self.model is None:
            raise RuntimeError("YOLO model has not been initialized.")

        results = self.model.predict(
            source=image,
            conf=self.confidence,
            verbose=False,
        )

        return self._convert_results(
            image,
            results,
        )

    def _convert_results(
        self,
        image,
        results,
    ) -> DetectionResult:
        """
        Convert YOLO results into unified DetectionResult.
        """

        detections = []

        result = results[0]

        boxes = result.boxes

        for idx, box in enumerate(boxes):

            cls_id = int(box.cls[0])

            confidence = float(box.conf[0])

            xyxy = box.xyxy[0].tolist()

            label = result.names[cls_id]

            detection = Detection(
                id=idx,
                label=label,
                confidence=confidence,
                bbox=BoundingBox(
                    xmin=int(xyxy[0]),
                    ymin=int(xyxy[1]),
                    xmax=int(xyxy[2]),
                    ymax=int(xyxy[3]),
                ),
            )

            detections.append(detection)

        return DetectionResult(
            image_id="input_image",
            timestamp=time.time(),
            detections=detections,
            metadata={"model": "YOLO", "source": "ultralytics"},
        )

    async def cleanup(self) -> None:
        """
        Release model.
        """

        self.model = None


VisionRegistry.register(
    "yolo",
    YOLOVision,
)
