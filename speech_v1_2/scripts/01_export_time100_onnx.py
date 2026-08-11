#!/usr/bin/env python3
"""Export SenseVoice encoder+CTC with internal query frames and variable valid length."""

from __future__ import annotations

import argparse
from pathlib import Path

MAX_AUDIO_FRAMES = 96
FEATURE_SIZE = 560
QUERY_IDS = (3, 1, 2, 14)  # zh, event, emotion, withitn
TOTAL_FRAMES = MAX_AUDIO_FRAMES + len(QUERY_IDS)
VOCAB_SIZE = 25055


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required = ("config.yaml", "am.mvn", "tokens.json", "model.pt")
    missing = [name for name in required if not (args.model_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing model assets: {missing}")

    import funasr
    import torch
    import torch.nn as nn
    from funasr import AutoModel

    class SenseVoiceTime100Wrapper(nn.Module):
        def __init__(self, sensevoice_model: nn.Module) -> None:
            super().__init__()
            self.model = sensevoice_model
            self.register_buffer(
                "query_ids", torch.tensor(QUERY_IDS, dtype=torch.int64)
            )

        def forward(
            self, speech: torch.Tensor, speech_lengths: torch.Tensor
        ) -> torch.Tensor:
            # Float32 is the portable RKNN boundary type; SenseVoice uses int64.
            lengths = speech_lengths.to(dtype=torch.int64)
            queries = self.model.embed(self.query_ids.unsqueeze(0))
            queries = queries.repeat(speech.shape[0], 1, 1)
            encoder_input = torch.cat((queries, speech), dim=1)
            encoder_lengths = lengths + len(QUERY_IDS)

            encoder_out, _ = self.model.encoder(encoder_input, encoder_lengths)
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]
            return self.model.ctc.ctc_lo(encoder_out)

    version = getattr(funasr, "__version__", "unknown")
    if version != "1.1.3":
        raise RuntimeError(
            f"FunASR 1.1.3 is required for the validated export, found {version}"
        )

    auto_model = AutoModel(model=str(args.model_root), device="cpu")
    model = auto_model.model
    model.eval()
    wrapper = SenseVoiceTime100Wrapper(model).eval()

    dummy_speech = torch.zeros(
        (1, MAX_AUDIO_FRAMES, FEATURE_SIZE), dtype=torch.float32
    )
    dummy_lengths = torch.tensor([MAX_AUDIO_FRAMES], dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        probe = wrapper(dummy_speech, dummy_lengths)
    expected_output = (1, TOTAL_FRAMES, VOCAB_SIZE)
    if tuple(probe.shape) != expected_output:
        raise RuntimeError(
            f"unexpected PyTorch output: expected {expected_output}, got {tuple(probe.shape)}"
        )

    torch.onnx.export(
        wrapper,
        (dummy_speech, dummy_lengths),
        str(args.output),
        input_names=["speech", "speech_lengths"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes=None,
        do_constant_folding=True,
    )

    print(f"FunASR: {version}")
    print(f"model root: {args.model_root}")
    print(f"query ids: {QUERY_IDS}")
    print(f"speech input: {(1, MAX_AUDIO_FRAMES, FEATURE_SIZE)} float32")
    print("length input: (1,) float32")
    print(f"logits output: {expected_output} float32")
    print(f"ONNX: {args.output} ({args.output.stat().st_size} bytes)")
    print("ONNX export: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
