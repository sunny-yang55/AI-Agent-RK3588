#!/usr/bin/env python3
"""Run real-audio features on RK3588, compare logits, and decode CTC tokens."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import numpy as np

INPUT_SHAPE = (1, 96, 560)
OUTPUT_SHAPE = (1, 100, 25055)
BLANK_ID = 0
QUERY_FRAMES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--core", choices=("0", "1", "2"), default="0")
    return parser.parse_args()


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left64 = left.astype(np.float64, copy=False).ravel()
    right64 = right.astype(np.float64, copy=False).ravel()
    denominator = np.linalg.norm(left64) * np.linalg.norm(right64)
    if denominator == 0:
        return 1.0 if np.array_equal(left64, right64) else 0.0
    return float(np.dot(left64, right64) / denominator)


def ctc_ids(logits: np.ndarray, valid_total_frames: int) -> list[int]:
    frame_ids = np.argmax(logits[0, :valid_total_frames, :], axis=-1).tolist()
    result: list[int] = []
    previous: int | None = None
    for token_id in frame_ids:
        if token_id != previous and token_id != BLANK_ID:
            result.append(int(token_id))
        previous = token_id
    return result


def decode_tokens(token_ids: list[int], token_list: list[str]) -> tuple[str, str]:
    pieces = [token_list[token_id] for token_id in token_ids]
    raw = "".join(pieces).replace("▁", " ").strip()
    clean = re.sub(r"<\|[^|]+\|>", "", raw).strip()
    return raw, clean


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    speech = np.load(args.input_dir / "speech.npy", allow_pickle=False).astype(
        np.float32, copy=False
    )
    lengths = np.load(
        args.input_dir / "speech_lengths.npy", allow_pickle=False
    ).astype(np.float32, copy=False)
    golden = np.load(args.golden, allow_pickle=False).astype(np.float32, copy=False)
    token_list = json.loads(args.tokens.read_text(encoding="utf-8"))
    if speech.shape != INPUT_SHAPE or lengths.shape != (1,):
        raise RuntimeError(f"invalid inputs: speech={speech.shape}, lengths={lengths.shape}")
    if golden.shape != OUTPUT_SHAPE:
        raise RuntimeError(f"invalid golden shape: {golden.shape}")
    if not isinstance(token_list, list) or len(token_list) != OUTPUT_SHAPE[-1]:
        raise RuntimeError("tokens.json must be a 25055-item list")

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
            outputs = rknn.inference(inputs=[speech, lengths])
            durations_ms.append((time.perf_counter() - started) * 1000)
            if not outputs:
                raise RuntimeError(f"inference run {index + 1} returned no outputs")
            logits = np.asarray(outputs[0], dtype=np.float32)
            print(f"run {index + 1}/{args.runs}: {durations_ms[-1]:.2f} ms")
    finally:
        rknn.release()

    assert logits is not None
    if logits.shape != OUTPUT_SHAPE:
        raise RuntimeError(f"unexpected RKNN output: {logits.shape}")
    delta = logits.astype(np.float64) - golden.astype(np.float64)
    rknn_frame_ids = np.argmax(logits, axis=-1)
    onnx_frame_ids = np.argmax(golden, axis=-1)
    token_agreement = float(np.mean(rknn_frame_ids == onnx_frame_ids))
    valid_total_frames = int(lengths[0]) + QUERY_FRAMES
    token_ids = ctc_ids(logits, valid_total_frames)
    raw_text, clean_text = decode_tokens(token_ids, token_list)
    official_text = (args.input_dir / "official_text.txt").read_text(
        encoding="utf-8"
    ).strip()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.text_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, logits, allow_pickle=False)
    args.text_out.write_text(clean_text + "\n", encoding="utf-8")

    print(f"output: {logits.shape} {logits.dtype} -> {args.output}")
    print(f"latency median: {statistics.median(durations_ms):.2f} ms")
    print(f"cosine similarity: {cosine_similarity(logits, golden):.8f}")
    print(f"mean absolute error: {np.mean(np.abs(delta)):.8f}")
    print(f"max absolute error: {np.max(np.abs(delta)):.8f}")
    print(f"frame Top-1 agreement: {token_agreement * 100:.2f}%")
    print(f"token ids: {token_ids}")
    print(f"RKNN raw: {raw_text}")
    print(f"RKNN clean: {clean_text}")
    print(f"official clean: {official_text}")
    if token_agreement < 0.99:
        raise RuntimeError("RKNN/ONNX frame Top-1 agreement is below 99%")
    if clean_text != official_text:
        raise RuntimeError("RKNN text does not match official FunASR text")
    print("RKNN real-audio inference: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
