#!/usr/bin/env python3
"""Calibrate the workbench ROI and preview colored-block detections."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--config", default="config/workbench_roi.json")
    return parser.parse_args()


def main() -> int:
    import cv2

    from tools.vision.camera import OpenCVCameraSource
    from tools.vision.workbench import (
        ColorBlockDetector,
        WorkbenchROI,
        draw_workbench_detections,
        load_workbench_roi,
        save_workbench_roi,
    )

    args = parse_args()
    config_path = ROOT / args.config
    camera = OpenCVCameraSource()
    try:
        camera.open()
        frame = camera.read().image
        roi = load_workbench_roi(config_path)
        if args.calibrate or roi is None:
            print("[Workbench] Drag around the white board, then press Enter")
            selected = cv2.selectROI("Select Workbench ROI", frame, False, False)
            cv2.destroyWindow("Select Workbench ROI")
            x, y, width, height = (int(value) for value in selected)
            if width <= 0 or height <= 0:
                print("[Workbench] ROI selection cancelled")
                return 1
            roi = WorkbenchROI(x, y, width, height)
            save_workbench_roi(config_path, roi)
            print(f"[Workbench] ROI saved: {roi}")
        detector = ColorBlockDetector(roi)
        print("[Workbench] q/Esc=quit, c=recalibrate")
        while True:
            frame = camera.read().image
            detections = detector.detect(frame)
            display = draw_workbench_detections(frame, roi, detections)
            cv2.imshow("XiaoAn Vision - Workbench", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                selected = cv2.selectROI("Select Workbench ROI", frame, False, False)
                cv2.destroyWindow("Select Workbench ROI")
                x, y, width, height = (int(value) for value in selected)
                if width > 0 and height > 0:
                    roi = WorkbenchROI(x, y, width, height)
                    detector = ColorBlockDetector(roi)
                    save_workbench_roi(config_path, roi)
            if detections:
                details = ", ".join(
                    f"{item.color}@{item.center_pixel} angle={item.angle_deg:.1f}"
                    for item in detections
                )
                print(f"\r[Workbench] {details}", end="", flush=True)
        print()
        return 0
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
