#!/usr/bin/env python3
"""Convert the two-input SenseVoice time-100 ONNX to non-quantized RKNN."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="rk3588")
    return parser.parse_args()


def require_success(step: str, code: int) -> None:
    if code != 0:
        raise RuntimeError(f"{step} failed with RKNN return code {code}")


def main() -> int:
    args = parse_args()
    if not args.onnx.is_file() or args.onnx.stat().st_size == 0:
        raise FileNotFoundError(args.onnx)

    from rknn.api import RKNN

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=True)
    try:
        rknn.config(target_platform=args.target)
        require_success("rknn.load_onnx", rknn.load_onnx(model=str(args.onnx)))
        require_success("rknn.build", rknn.build(do_quantization=False))
        require_success("rknn.export_rknn", rknn.export_rknn(str(args.output)))
    finally:
        rknn.release()

    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise RuntimeError(f"RKNN export did not create a valid file: {args.output}")
    print(f"RKNN: {args.output} ({args.output.stat().st_size} bytes)")
    print("RKNN export: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
