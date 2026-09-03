#!/usr/bin/env python3
"""Capture one frame and validate RKNN YOLOv5 detection."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = ROOT / "venv/bin/python"
PROJECT_VENV = ROOT / "venv"
if PROJECT_PYTHON.is_file() and Path(sys.prefix).resolve() != PROJECT_VENV.resolve():
    os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
sys.path.insert(0, str(ROOT))

from tools.vision.camera import OpenCVCameraSource  # noqa: E402
from tools.vision.yolov5_rknn import (  # noqa: E402
    RKNNYOLOv5Detector,
    summarize_detections,
)


def main() -> int:
    import cv2

    camera = OpenCVCameraSource()
    detector = None
    try:
        camera.open()
        for _ in range(5):
            frame = camera.read()
        detector = RKNNYOLOv5Detector()
        detections = detector.detect(frame.image)
        annotated = frame.image.copy()
        for item in detections:
            x1, y1, x2, y2 = item.box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, f"{item.label} {item.confidence:.2f}", (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
            )
        output_dir = ROOT / "reports/vision-camera"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / time.strftime("detection-%Y%m%d-%H%M%S.jpg")
        cv2.imwrite(str(output), annotated)
        print(f"[Detection] latency={detector.last_latency_ms:.2f}ms")
        print(f"[Detection] objects={len(detections)}")
        for item in detections:
            print(f"[Detection] {item.label_zh} {item.confidence:.3f} box={item.box}")
        print(f"[Detection] summary={summarize_detections(detections)}")
        print(f"[Detection] image={output}")
        return 0
    finally:
        if detector is not None:
            detector.close()
        camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
