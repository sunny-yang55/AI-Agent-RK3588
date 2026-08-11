#!/usr/bin/env python3
"""Create same-source SenseVoice features and the official CPU reference text."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

MAX_AUDIO_FRAMES = 96
FEATURE_SIZE = 560
EXPECTED_FUNASR_VERSION = "1.1.3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def clean_text(text: str) -> str:
    return re.sub(r"<\|[^|]+\|>", "", text).strip()


def main() -> int:
    args = parse_args()
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    if not (args.model_root / "am.mvn").is_file():
        raise FileNotFoundError(args.model_root / "am.mvn")

    import funasr
    from funasr import AutoModel

    version = getattr(funasr, "__version__", "unknown")
    if version != EXPECTED_FUNASR_VERSION:
        raise RuntimeError(
            f"FunASR {EXPECTED_FUNASR_VERSION} is required, found {version}"
        )

    auto_model = AutoModel(model=str(args.model_root), device="cpu")
    model = auto_model.model
    frontend = auto_model.kwargs["frontend"]
    funcs = model.inference.__globals__

    audio_list = funcs["load_audio_text_image_video"](
        str(args.audio),
        fs=frontend.fs,
        audio_fs=16000,
        data_type="sound",
        tokenizer=auto_model.kwargs.get("tokenizer"),
    )
    extracted, extracted_lengths = funcs["extract_fbank"](
        audio_list,
        data_type="sound",
        frontend=frontend,
    )
    features = extracted.detach().cpu().numpy().astype(np.float32, copy=False)
    valid_frames = int(extracted_lengths.reshape(-1)[0])

    if features.shape != (1, valid_frames, FEATURE_SIZE):
        raise RuntimeError(
            f"unexpected feature shape: {features.shape}, length={valid_frames}"
        )
    if not 1 <= valid_frames <= MAX_AUDIO_FRAMES:
        raise RuntimeError(
            f"audio produces {valid_frames} LFR frames; v1.2 accepts 1..{MAX_AUDIO_FRAMES}"
        )
    if not np.isfinite(features).all():
        raise RuntimeError("features contain NaN or Inf")

    padded = np.zeros((1, MAX_AUDIO_FRAMES, FEATURE_SIZE), dtype=np.float32)
    padded[:, :valid_frames, :] = features
    length_input = np.asarray([valid_frames], dtype=np.float32)

    official = auto_model.generate(
        input=str(args.audio), language="zh", use_itn=True
    )
    if not isinstance(official, list) or not official:
        raise RuntimeError(f"unexpected official result: {official!r}")
    official_raw = str(official[0]["text"])
    official_clean = clean_text(official_raw)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "speech.npy", padded, allow_pickle=False)
    np.save(args.output_dir / "speech_lengths.npy", length_input, allow_pickle=False)
    (args.output_dir / "official_text.txt").write_text(
        official_clean + "\n", encoding="utf-8"
    )
    metadata = {
        "audio": str(args.audio),
        "funasr_version": version,
        "frontend": {
            "fs": int(frontend.fs),
            "lfr_m": int(frontend.lfr_m),
            "lfr_n": int(frontend.lfr_n),
        },
        "valid_audio_frames": valid_frames,
        "query_frames": 4,
        "valid_total_frames": valid_frames + 4,
        "official_raw": official_raw,
        "official_clean": official_clean,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"audio: {args.audio}")
    print(f"features: {features.shape} -> padded {padded.shape}")
    print(f"valid audio frames: {valid_frames}")
    print(f"valid total frames: {valid_frames + 4}")
    print(f"official raw: {official_raw}")
    print(f"official clean: {official_clean}")
    print(f"artifacts: {args.output_dir}")
    print("real-audio preparation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
