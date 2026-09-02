"""Streaming Zipformer transducer inference using three RKNN models."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import kaldifeat
import torch
from rknnlite.api import RKNNLite


class ZipformerRKNNASR:
    sample_rate = 16000
    segment_frames = 103
    offset_frames = 96
    context_size = 2
    blank_id = 0
    unk_id = 2

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self.tokens = self._read_tokens(self.model_dir / "vocab.txt")
        self.encoder = self._load("encoder-epoch-99-avg-1.rknn")
        self.decoder = self._load("decoder-epoch-99-avg-1.rknn")
        self.joiner = self._load("joiner-epoch-99-avg-1.rknn")
        self._lock = threading.Lock()
        self.reset()

    @staticmethod
    def _read_tokens(path: Path) -> dict[int, str]:
        tokens: dict[int, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.rsplit(maxsplit=1)
            if len(parts) == 2:
                tokens[int(parts[1])] = parts[0]
        if not tokens:
            raise RuntimeError(f"empty Zipformer vocabulary: {path}")
        return tokens

    def _load(self, name: str) -> RKNNLite:
        runtime = RKNNLite()
        path = self.model_dir / name
        load_code = runtime.load_rknn(str(path))
        if load_code != 0:
            raise RuntimeError(f"load_rknn failed ({load_code}): {path}")
        init_code = runtime.init_runtime()
        if init_code != 0:
            runtime.release()
            raise RuntimeError(f"init_runtime failed ({init_code}): {path}")
        return runtime

    @staticmethod
    def _new_encoder_cache() -> list[np.ndarray]:
        values: list[np.ndarray] = []
        for _ in range(5):
            values.append(np.zeros((2, 1), dtype=np.int64))
        for _ in range(5):
            values.append(np.zeros((2, 1, 256), dtype=np.float32))
        for length in (192, 96, 48, 24, 96):
            values.append(np.zeros((2, length, 1, 192), dtype=np.float32))
        for _ in range(2):
            for length in (192, 96, 48, 24, 96):
                values.append(np.zeros((2, length, 1, 96), dtype=np.float32))
        for _ in range(2):
            for _layer in range(5):
                values.append(np.zeros((2, 1, 256, 30), dtype=np.float32))
        return values

    @staticmethod
    def _new_fbank() -> kaldifeat.OnlineFbank:
        options = kaldifeat.FbankOptions()
        options.frame_opts.samp_freq = 16000
        options.mel_opts.num_bins = 80
        options.mel_opts.high_freq = -400
        options.frame_opts.dither = 0
        options.frame_opts.snip_edges = False
        return kaldifeat.OnlineFbank(options)

    def reset(self) -> None:
        self.fbank = self._new_fbank()
        self._processed_frames = 0
        self._hyp = [self.blank_id] * self.context_size
        self._encoder_cache = self._new_encoder_cache()
        self._last_text = ""
        self.last_inference_ms = 0.0

    @property
    def text(self) -> str:
        return self._last_text

    def _available_frames(self) -> int:
        return self.fbank.num_frames_ready - self._processed_frames

    def accept_waveform(self, waveform: np.ndarray) -> str:
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return self.text
        with self._lock:
            self.fbank.accept_waveform(
                sampling_rate=self.sample_rate,
                waveform=torch.from_numpy(np.ascontiguousarray(samples)),
            )
            self._process_ready_segments()
            return self.text

    def finalize(self) -> str:
        with self._lock:
            available = self._available_frames()
            if available > 0 and available < self.segment_frames:
                missing = self.segment_frames - available
                padding = np.zeros((missing + 3) * 160, dtype=np.float32)
                self.fbank.accept_waveform(
                    sampling_rate=self.sample_rate,
                    waveform=torch.from_numpy(padding),
                )
            self._process_ready_segments()
            return self.text

    def _process_ready_segments(self) -> None:
        while self._available_frames() >= self.segment_frames:
            frames = [
                self.fbank.get_frame(self._processed_frames + index).reshape(-1)
                for index in range(self.segment_frames)
            ]
            features = torch.stack(frames).cpu().numpy().astype(np.float32)
            features = np.ascontiguousarray(features.reshape(1, 103, 80))
            started = time.perf_counter()
            outputs = self.encoder.inference(inputs=[features, *self._encoder_cache])
            if outputs is None or len(outputs) < 2:
                raise RuntimeError("Zipformer encoder returned no cache outputs")
            encoder_out = np.asarray(outputs[0]).squeeze(0)
            new_cache: list[np.ndarray] = []
            for output_index, value in enumerate(outputs[1:]):
                array = np.asarray(value)
                if output_index + 1 > 10 and array.ndim == 4:
                    array = array.transpose(0, 2, 3, 1)
                new_cache.append(np.ascontiguousarray(array))
            self._encoder_cache = new_cache
            self._decode_encoder_output(encoder_out)
            self.last_inference_ms = (time.perf_counter() - started) * 1000
            self._processed_frames += self.offset_frames
            self._last_text = self._tokens_to_text(self._hyp[self.context_size:])

    def _decode_encoder_output(self, encoder_out: np.ndarray) -> None:
        for frame in encoder_out:
            decoder_input = np.asarray([self._hyp[-self.context_size:]], dtype=np.int64)
            decoder_out = self.decoder.inference(inputs=[decoder_input])
            if not decoder_out:
                raise RuntimeError("Zipformer decoder returned no output")
            joiner_out = self.joiner.inference(
                inputs=[
                    np.ascontiguousarray(np.asarray(frame).reshape(1, -1)),
                    np.ascontiguousarray(np.asarray(decoder_out[0])),
                ]
            )
            if not joiner_out:
                raise RuntimeError("Zipformer joiner returned no output")
            token = int(np.asarray(joiner_out[0]).argmax())
            if token not in {self.blank_id, self.unk_id}:
                self._hyp.append(token)

    def _tokens_to_text(self, ids: list[int]) -> str:
        text = "".join(self.tokens.get(token, "") for token in ids)
        return " ".join(text.replace("▁", " ").split())

    def close(self) -> None:
        for runtime in (self.encoder, self.decoder, self.joiner):
            try:
                runtime.release()
            except Exception:
                pass

    def __enter__(self) -> "ZipformerRKNNASR":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
