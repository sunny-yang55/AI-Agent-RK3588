#!/usr/bin/env python3
"""Create a reproducible ONNX input/output baseline on Windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

EXPECTED_INPUT_NAME = "speech"
EXPECTED_OUTPUT_NAME = "logits"
EXPECTED_INPUT_SHAPE = (1, 30, 560)
EXPECTED_OUTPUT_SHAPE = (1, 30, 25055)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="Existing float32 speech.npy")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Generate deterministic random input for conversion validation only",
    )
    parser.add_argument("--input-out", type=Path, required=True)
    parser.add_argument("--logits-out", type=Path, required=True)
    return parser.parse_args()


def load_input(args: argparse.Namespace) -> np.ndarray:
    if args.input is not None and args.random:
        raise ValueError("use either --input or --random, not both")
    if args.input is None and not args.random:
        raise ValueError("provide a real --input speech.npy or explicitly use --random")

    if args.input is not None:
        speech = np.load(args.input, allow_pickle=False)
    else:
        rng = np.random.default_rng(seed=20260811)
        speech = rng.standard_normal(EXPECTED_INPUT_SHAPE).astype(np.float32)

    speech = np.asarray(speech, dtype=np.float32)
    if speech.shape != EXPECTED_INPUT_SHAPE:
        raise ValueError(
            f"input shape mismatch: expected {EXPECTED_INPUT_SHAPE}, got {speech.shape}"
        )
    if not np.isfinite(speech).all():
        raise ValueError("input contains NaN or Inf")
    return speech


def main() -> int:
    args = parse_args()
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)

    import onnxruntime as ort

    session = ort.InferenceSession(
        str(args.onnx), providers=["CPUExecutionProvider"]
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != EXPECTED_INPUT_NAME:
        raise RuntimeError(f"unexpected ONNX inputs: {[item.name for item in inputs]}")
    if tuple(inputs[0].shape) != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(f"unexpected ONNX input shape: {inputs[0].shape}")
    if len(outputs) != 1 or outputs[0].name != EXPECTED_OUTPUT_NAME:
        raise RuntimeError(f"unexpected ONNX outputs: {[item.name for item in outputs]}")

    speech = load_input(args)
    logits = session.run([EXPECTED_OUTPUT_NAME], {EXPECTED_INPUT_NAME: speech})[0]
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape != EXPECTED_OUTPUT_SHAPE:
        raise RuntimeError(
            f"output shape mismatch: expected {EXPECTED_OUTPUT_SHAPE}, got {logits.shape}"
        )
    if not np.isfinite(logits).all():
        raise RuntimeError("ONNX output contains NaN or Inf")

    args.input_out.parent.mkdir(parents=True, exist_ok=True)
    args.logits_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.input_out, speech, allow_pickle=False)
    np.save(args.logits_out, logits, allow_pickle=False)

    print(f"ONNX: {args.onnx}")
    print(f"input: {speech.shape} {speech.dtype} -> {args.input_out}")
    print(f"logits: {logits.shape} {logits.dtype} -> {args.logits_out}")
    print("ONNX baseline: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
