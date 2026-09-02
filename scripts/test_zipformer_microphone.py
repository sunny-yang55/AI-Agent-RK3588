#!/usr/bin/env python3
"""Real-microphone acceptance test for the Zipformer RKNN prototype."""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import numpy as np
import webrtcvad


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speech.asr.zipformer_rknn_asr import ZipformerRKNNASR
from speech.audio.alsa_pcm_stream import ALSAStreamingMicrophone


def redraw(message: str) -> None:
    print(f"\r\033[2K{message}", end="", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zipformer RKNN microphone test")
    parser.add_argument("--device", default=None, help="ALSA device; default auto")
    parser.add_argument("--end-silence", type=float, default=0.55)
    parser.add_argument("--start-voice-ms", type=int, default=180)
    parser.add_argument("--pre-roll-ms", type=int, default=300)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--vad-mode", type=int, choices=range(4), default=2)
    parser.add_argument("--once", action="store_true", help="exit after one utterance")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame_ms = 30
    required_voice = max(1, args.start_voice_ms // frame_ms)
    required_silence = max(1, round(args.end_silence * 1000 / frame_ms))
    pre_roll_count = max(1, args.pre_roll_ms // frame_ms)
    vad = webrtcvad.Vad(args.vad_mode)
    model_dir = PROJECT_ROOT / "models/speech/zipformer"

    print("\n===== Zipformer RKNN 实时识别测试 =====")
    print("加载模型中…", flush=True)
    with ZipformerRKNNASR(model_dir) as recognizer:
        print("✓ 模型加载完成")
        with ALSAStreamingMicrophone(args.device, frame_ms=frame_ms) as microphone:
            print(f"✓ 麦克风: {microphone.label or microphone.device}")
            print("按 Ctrl+C 退出\n")
            pre_roll: collections.deque[np.ndarray] = collections.deque(maxlen=pre_roll_count)
            speech_started = False
            voice_run = 0
            silence_run = 0
            utterance_started = 0.0
            first_partial_at: float | None = None
            last_partial = ""
            redraw("● 等待说话…")
            try:
                for frame in microphone.frames():
                    voiced = vad.is_speech(frame.pcm, microphone.sample_rate)
                    if not speech_started:
                        pre_roll.append(frame.waveform)
                        voice_run = voice_run + 1 if voiced else 0
                        if voice_run < required_voice:
                            continue
                        speech_started = True
                        utterance_started = time.monotonic()
                        silence_run = 0
                        recognizer.reset()
                        redraw("🎤 检测到讲话，正在识别…")
                        for waveform in pre_roll:
                            partial = recognizer.accept_waveform(waveform)
                            if partial:
                                last_partial = partial
                                first_partial_at = first_partial_at or time.monotonic()
                                redraw(f"◌ 正在识别… {partial}")
                        pre_roll.clear()
                        continue

                    partial = recognizer.accept_waveform(frame.waveform)
                    if partial and partial != last_partial:
                        last_partial = partial
                        first_partial_at = first_partial_at or time.monotonic()
                        redraw(f"◌ 正在识别… {partial}")
                    silence_run = 0 if voiced else silence_run + 1
                    elapsed = time.monotonic() - utterance_started
                    if silence_run < required_silence and elapsed < args.max_seconds:
                        continue

                    final_started = time.monotonic()
                    final_text = recognizer.finalize().strip()
                    finished = time.monotonic()
                    print("\r\033[2K", end="")
                    print(f"[识别结果] {final_text or '（空）'}")
                    first_ms = (
                        (first_partial_at - utterance_started) * 1000
                        if first_partial_at is not None else -1
                    )
                    print(
                        f"[耗时] 语音 {elapsed:.2f}s｜首个动态结果 "
                        f"{first_ms:.0f}ms｜收尾 {1000 * (finished-final_started):.0f}ms｜"
                        f"末段NPU {recognizer.last_inference_ms:.0f}ms\n"
                    )
                    if args.once:
                        return 0
                    speech_started = False
                    voice_run = 0
                    silence_run = 0
                    first_partial_at = None
                    last_partial = ""
                    pre_roll.clear()
                    redraw("● 等待说话…")
            except KeyboardInterrupt:
                print("\n✓ 测试已退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
