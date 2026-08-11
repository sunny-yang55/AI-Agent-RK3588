#!/usr/bin/env python3
"""Run the time-100 ONNX and verify its decoded text against official FunASR."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

INPUT_SHAPE = (1, 96, 560)
OUTPUT_SHAPE = (1, 100, 25055)
BLANK_ID = 0
QUERY_FRAMES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--logits-out", type=Path, required=True)
    parser.add_argument("--text-out", type=Path, required=True)
    return parser.parse_args()


def ctc_ids(logits: np.ndarray, valid_total_frames: int) -> list[int]:
    frame_ids = np.argmax(logits[0, :valid_total_frames, :], axis=-1).tolist()
    collapsed: list[int] = []
    previous: int | None = None
    for token_id in frame_ids:
        if token_id != previous and token_id != BLANK_ID:
            collapsed.append(int(token_id))
        previous = token_id
    return collapsed


def clean_text(text: str) -> str:
    return re.sub(r"<\|[^|]+\|>", "", text).strip()


def main() -> int:
    args = parse_args()
    speech = np.load(args.input_dir / "speech.npy", allow_pickle=False).astype(
        np.float32, copy=False
    )
    lengths = np.load(
        args.input_dir / "speech_lengths.npy", allow_pickle=False
    ).astype(np.float32, copy=False)
    if speech.shape != INPUT_SHAPE:
        raise RuntimeError(f"unexpected speech shape: {speech.shape}")
    if lengths.shape != (1,):
        raise RuntimeError(f"unexpected length shape: {lengths.shape}")
    valid_audio_frames = int(lengths[0])
    if not 1 <= valid_audio_frames <= INPUT_SHAPE[1]:
        raise RuntimeError(f"invalid speech length: {lengths[0]}")

    import onnxruntime as ort
    from funasr import AutoModel

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_contract = [(item.name, tuple(item.shape), item.type) for item in session.get_inputs()]
    expected_inputs = [
        ("speech", INPUT_SHAPE, "tensor(float)"),
        ("speech_lengths", (1,), "tensor(float)"),
    ]
    if input_contract != expected_inputs:
        raise RuntimeError(f"unexpected ONNX inputs: {input_contract}")

    logits = session.run(
        ["logits"], {"speech": speech, "speech_lengths": lengths}
    )[0]
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape != OUTPUT_SHAPE:
        raise RuntimeError(f"unexpected logits shape: {logits.shape}")
    if not np.isfinite(logits).all():
        raise RuntimeError("ONNX logits contain NaN or Inf")

    token_ids = ctc_ids(logits, valid_audio_frames + QUERY_FRAMES)
    auto_model = AutoModel(model=str(args.model_root), device="cpu")
    tokenizer = auto_model.kwargs["tokenizer"]
    onnx_raw = tokenizer.decode(token_ids)
    onnx_clean = clean_text(onnx_raw)
    official_clean = (args.input_dir / "official_text.txt").read_text(
        encoding="utf-8"
    ).strip()

    args.logits_out.parent.mkdir(parents=True, exist_ok=True)
    args.text_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.logits_out, logits, allow_pickle=False)
    args.text_out.write_text(onnx_clean + "\n", encoding="utf-8")

    print(f"ONNX inputs: {input_contract}")
    print(f"logits: {logits.shape} {logits.dtype}")
    print(f"valid total frames: {valid_audio_frames + QUERY_FRAMES}")
    print(f"token ids: {token_ids}")
    print(f"ONNX raw: {onnx_raw}")
    print(f"ONNX clean: {onnx_clean}")
    print(f"official clean: {official_clean}")
    if onnx_clean != official_clean:
        raise RuntimeError("ONNX text does not match official FunASR text")
    print("ONNX real-audio baseline: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
