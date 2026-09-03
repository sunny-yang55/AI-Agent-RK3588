#!/usr/bin/env python3
"""Convert Rockchip's optimized YOLOv5 ONNX model for RK3588."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"ONNX model not found: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    try:
        print("[VisionModel] Configuring RK3588 FP model")
        code = rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform="rk3588",
        )
        if code != 0:
            raise RuntimeError(f"RKNN config failed with code {code}")

        print("[VisionModel] Loading ONNX")
        code = rknn.load_onnx(model=str(args.input))
        if code != 0:
            raise RuntimeError(f"RKNN load_onnx failed with code {code}")

        print("[VisionModel] Building; this may take several minutes")
        code = rknn.build(do_quantization=False)
        if code != 0:
            raise RuntimeError(f"RKNN build failed with code {code}")

        print(f"[VisionModel] Exporting: {args.output}")
        code = rknn.export_rknn(str(args.output))
        if code != 0:
            raise RuntimeError(f"RKNN export failed with code {code}")
    finally:
        rknn.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
