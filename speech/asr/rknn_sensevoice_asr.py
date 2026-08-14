"""SenseVoice-small RKNN backend for RK3588.

Runtime assets live under models/speech/sensevoice. No FunASR, PyTorch,
torchaudio or ONNX Runtime dependency is used at runtime.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import threading
import time
from pathlib import Path

import numpy as np

from speech.asr.recognizer import SpeechRecognizer
import voice_ui as ui

MAX_AUDIO_FRAMES = 96
FEATURE_SIZE = 560
OUTPUT_CLASSES = 25055
QUERY_FRAMES = 4
BLANK_ID = 0


class RKNNSenseVoiceASR(SpeechRecognizer):
    """Persistent RKNNLite SenseVoice recognizer for the main voice loop."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        cmvn_path: str | Path | None = None,
        tokens_path: str | Path | None = None,
        core: str | None = None,
    ) -> None:
        super().__init__()
        root = Path(__file__).resolve().parents[2]
        self.model_path = Path(
            model_path
            or os.getenv(
                "SENSEVOICE_RKNN_MODEL",
                root / "models/speech/sensevoice/sensevoice_time100_fp.rknn",
            )
        )
        self.config_path = Path(
            config_path
            or os.getenv("SENSEVOICE_FRONTEND_CONFIG", root / "models/speech/sensevoice/config.yaml")
        )
        self.cmvn_path = Path(
            cmvn_path
            or os.getenv("SENSEVOICE_CMVN", root / "models/speech/sensevoice/am.mvn")
        )
        self.tokens_path = Path(
            tokens_path
            or os.getenv("SENSEVOICE_TOKENS", root / "models/speech/sensevoice/tokens.json")
        )
        for path, label in (
            (self.model_path, "RKNN model"),
            (self.config_path, "frontend config"),
            (self.cmvn_path, "CMVN"),
            (self.tokens_path, "tokens"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")

        self.tokens = json.loads(self.tokens_path.read_text(encoding="utf-8"))
        if not isinstance(self.tokens, list) or len(self.tokens) != OUTPUT_CLASSES:
            raise RuntimeError(f"tokens must contain {OUTPUT_CLASSES} entries")

        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:
            raise RuntimeError("rknn-toolkit-lite2 is not installed in this environment") from exc

        selected_core = core or os.getenv("SENSEVOICE_NPU_CORE", "0")
        core_masks = {
            "0": RKNNLite.NPU_CORE_0,
            "1": RKNNLite.NPU_CORE_1,
            "2": RKNNLite.NPU_CORE_2,
        }
        if selected_core not in core_masks:
            raise ValueError("SENSEVOICE_NPU_CORE must be 0, 1 or 2")

        self._rknn = RKNNLite(verbose=False)
        code = self._rknn.load_rknn(str(self.model_path))
        if code != 0:
            raise RuntimeError(f"RKNNLite.load_rknn failed with code {code}")
        code = self._rknn.init_runtime(core_mask=core_masks[selected_core])
        if code != 0:
            self._rknn.release()
            raise RuntimeError(f"RKNNLite.init_runtime failed with code {code}")
        self._lock = threading.Lock()
        self.last_latency_ms: float | None = None
        atexit.register(self.close)
        ui.debug(f"[SenseVoice RKNN] 已加载: {self.model_path.name} (NPU core {selected_core})")

    def close(self) -> None:
        rknn = getattr(self, "_rknn", None)
        if rknn is not None:
            rknn.release()
            self._rknn = None

    @staticmethod
    def _decode(logits: np.ndarray, valid_frames: int, tokens: list[str]) -> str:
        frame_ids = np.argmax(logits[0, : valid_frames + QUERY_FRAMES], axis=-1)
        result: list[int] = []
        previous: int | None = None
        for value in frame_ids.tolist():
            token_id = int(value)
            if token_id != previous and token_id != BLANK_ID:
                result.append(token_id)
            previous = token_id
        raw = "".join(tokens[token_id] for token_id in result).replace("▁", " ").strip()
        return re.sub(r"<\|[^|]+\|>", "", raw).strip()

    def transcribe(self, audio: dict[str, object]) -> str:
        if not isinstance(audio, dict) or "data" not in audio:
            raise TypeError("audio must be a dictionary containing waveform data")
        sample_rate = int(audio.get("sample_rate", 16000))
        channels = int(audio.get("channels", 1))
        if sample_rate != 16000 or channels != 1:
            raise RuntimeError(
                f"SenseVoice RKNN requires 16 kHz mono audio; got {sample_rate} Hz, {channels} channel(s)"
            )
        waveform = np.asarray(audio["data"], dtype=np.float32).reshape(-1)
        if not np.isfinite(waveform).all():
            raise RuntimeError("audio contains NaN or Inf")

        from speech.asr.lightweight_frontend import extract_waveform_features

        _, features, _ = extract_waveform_features(
            waveform, self.config_path, self.cmvn_path
        )
        valid_frames = int(features.shape[0])
        if features.shape[1:] != (FEATURE_SIZE,):
            raise RuntimeError(f"unexpected frontend output: {features.shape}")
        if not 1 <= valid_frames <= MAX_AUDIO_FRAMES:
            raise RuntimeError(
                f"audio produces {valid_frames} frames; static RKNN accepts 1..{MAX_AUDIO_FRAMES}. "
                "Keep one utterance within 5 seconds."
            )

        speech = np.zeros((1, MAX_AUDIO_FRAMES, FEATURE_SIZE), dtype=np.float32)
        speech[0, :valid_frames] = features
        lengths = np.asarray([valid_frames], dtype=np.float32)
        with self._lock:
            if self._rknn is None:
                raise RuntimeError("RKNN runtime has already been closed")
            started = time.perf_counter()
            outputs = self._rknn.inference(inputs=[speech, lengths])
            self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        if not outputs:
            raise RuntimeError("RKNN inference returned no outputs")
        logits = np.asarray(outputs[0], dtype=np.float32)
        expected = (1, MAX_AUDIO_FRAMES + QUERY_FRAMES, OUTPUT_CLASSES)
        if logits.shape != expected:
            raise RuntimeError(f"unexpected RKNN output: {logits.shape}, expected {expected}")
        text = self._decode(logits, valid_frames, self.tokens)
        ui.debug(f"[SenseVoice RKNN] frames={valid_frames}, NPU={self.last_latency_ms:.2f} ms")
        return text
