#!/usr/bin/env python3
"""Capture labelled workbench ROI samples for the three-shape classifier.

Example:
    venv/bin/python scripts/capture_workbench_shape_samples.py green cube

Place exactly one labelled object on the white board, press ``s`` to save a
frame, and press ``q`` or Esc to finish.  Capture 20-30 independently posed
samples per colour/shape class.  The images never leave the RK3588 unless the
operator copies them elsewhere.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COLORS = ("red", "yellow", "blue", "green")
SHAPES = ("cube", "cylinder", "triangular_pyramid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("color", choices=COLORS, help="ground-truth colour label")
    parser.add_argument("shape", choices=SHAPES, help="ground-truth shape label")
    parser.add_argument("--output", default="datasets/workbench_shapes")
    parser.add_argument(
        "--split", choices=("train", "val", "test"), default="train",
        help="dataset split; capture each split in a separate session",
    )
    return parser.parse_args()


def main() -> int:
    import cv2

    from tools.vision.camera import OpenCVCameraSource
    from tools.vision.workbench import WorkbenchROI, load_workbench_roi

    args = parse_args()
    roi = load_workbench_roi(ROOT / "config/workbench_roi.json")
    if roi is None:
        print("[Shapes] Missing config/workbench_roi.json; calibrate the workbench first.")
        return 2
    label = f"{args.color}_{args.shape}"
    output_dir = ROOT / args.output / label / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    camera = OpenCVCameraSource()
    saved = 0
    try:
        camera.open()
        print(f"[Shapes] label={label} split={args.split}; s=save, q/Esc=quit")
        while True:
            image = camera.read().image
            clipped = roi.clipped(image)
            preview = image.copy()
            cv2.rectangle(
                preview,
                (clipped.x, clipped.y),
                (clipped.x + clipped.width, clipped.y + clipped.height),
                (255, 255, 0),
                2,
            )
            cv2.putText(
                preview,
                f"label={label} saved={saved}",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("XiaoAn Shape Samples", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                crop = image[
                    clipped.y : clipped.y + clipped.height,
                    clipped.x : clipped.x + clipped.width,
                ]
                stamp = time.strftime("%Y%m%d-%H%M%S")
                target = output_dir / f"{label}-{stamp}-{saved:03d}.jpg"
                if cv2.imwrite(str(target), crop):
                    saved += 1
                    print(f"[Shapes] saved {target.relative_to(ROOT)}")
                else:
                    print("[Shapes] failed to save image")
        print(f"[Shapes] complete: {saved} samples in {output_dir.relative_to(ROOT)}")
        return 0
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
