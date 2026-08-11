#!/usr/bin/env python3
"""Run SenseVoice RKNN on RK3588 and compare it with ONNX logits."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

EXPECTED_INPUT_SHAPE = (1, 30, 560)
EXPECTED_OUTPUT_SHAPE = (1, 30, 25055)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--golden", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--core", choices=("0", "1", "2"), default="0")
    return parser.parse_args()


def check_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise RuntimeError(f"file is empty: {path}")


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left64 = left.astype(np.float64, copy=False).ravel()
    right64 = right.astype(np.float64, copy=False).ravel()
    denominator = np.linalg.norm(left64) * np.linalg.norm(right64)
    if denominator == 0:
        return 1.0 if np.array_equal(left64, right64) else 0.0
    return float(np.dot(left64, right64) / denominator)


def compare(logits: np.ndarray, golden: np.ndarray) -> None:
    if logits.shape != golden.shape:
        raise RuntimeError(
            f"golden shape mismatch: RKNN={logits.shape}, ONNX={golden.shape}"
        )
    delta = logits.astype(np.float64) - golden.astype(np.float64)
    rknn_tokens = np.argmax(logits, axis=-1)
    onnx_tokens = np.argmax(golden, axis=-1)
    token_agreement = float(np.mean(rknn_tokens == onnx_tokens))

    print(f"cosine similarity: {cosine_similarity(logits, golden):.8f}")
    print(f"mean absolute error: {np.mean(np.abs(delta)):.8f}")
    print(f"max absolute error: {np.max(np.abs(delta)):.8f}")
    print(f"token Top-1 agreement: {token_agreement * 100:.2f}%")


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    check_file(args.model)
    check_file(args.input)

    speech = np.load(args.input, allow_pickle=False).astype(np.float32, copy=False)
    if speech.shape != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(
            f"input shape mismatch: expected {EXPECTED_INPUT_SHAPE}, got {speech.shape}"
        )
    if not np.isfinite(speech).all():
        raise RuntimeError("input contains NaN or Inf")

    from rknnlite.api import RKNNLite

    core_mask = {
        "0": RKNNLite.NPU_CORE_0,
        "1": RKNNLite.NPU_CORE_1,
        "2": RKNNLite.NPU_CORE_2,
    }[args.core]

    rknn = RKNNLite(verbose=True)
    try:
        ret = rknn.load_rknn(str(args.model))
        if ret != 0:
            raise RuntimeError(f"RKNNLite.load_rknn failed with code {ret}")

        ret = rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"RKNNLite.init_runtime failed with code {ret}")

        durations_ms: list[float] = []
        logits = None
        for index in range(args.runs):
            started = time.perf_counter()
            outputs = rknn.inference(inputs=[speech])
            durations_ms.append((time.perf_counter() - started) * 1000)
            if not outputs:
                raise RuntimeError(f"inference run {index + 1} returned no outputs")
            logits = np.asarray(outputs[0], dtype=np.float32)
            print(f"run {index + 1}/{args.runs}: {durations_ms[-1]:.2f} ms")
    finally:
        rknn.release()

    assert logits is not None
    if logits.shape != EXPECTED_OUTPUT_SHAPE:
        raise RuntimeError(
            f"output shape mismatch: expected {EXPECTED_OUTPUT_SHAPE}, got {logits.shape}"
        )
    if not np.isfinite(logits).all():
        raise RuntimeError("RKNN output contains NaN or Inf")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, logits, allow_pickle=False)
    print(f"output: {logits.shape} {logits.dtype} -> {args.output}")
    print(f"latency median: {statistics.median(durations_ms):.2f} ms")

    if args.golden is not None:
        check_file(args.golden)
        golden = np.load(args.golden, allow_pickle=False).astype(
            np.float32, copy=False
        )
        compare(logits, golden)

    print("RKNN inference: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
