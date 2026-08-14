#!/usr/bin/env python3
"""NumPy/SciPy implementation of the SenseVoice Kaldi/LFR/CMVN frontend."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.io import wavfile


@dataclass(frozen=True)
class FrontendConfig:
    fs: int = 16000
    n_mels: int = 80
    frame_length: float = 25.0
    frame_shift: float = 10.0
    lfr_m: int = 7
    lfr_n: int = 6
    window: str = "hamming"
    dither: float = 0.0
    preemphasis: float = 0.97
    remove_dc_offset: bool = True
    snip_edges: bool = True
    low_freq: float = 20.0
    high_freq: float = 0.0


def _find_mapping(value: object, key: str) -> dict[str, object] | None:
    if isinstance(value, dict):
        candidate = value.get(key)
        if isinstance(candidate, dict):
            return candidate
        for child in value.values():
            found = _find_mapping(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_mapping(child, key)
            if found is not None:
                return found
    return None


def load_config(path: Path) -> FrontendConfig:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = _find_mapping(document, "frontend_conf") or {}

    def pick(name: str, default: object, *aliases: str) -> object:
        for candidate in (name, *aliases):
            if candidate in values:
                return values[candidate]
        return default

    default = FrontendConfig()
    return FrontendConfig(
        fs=int(pick("fs", default.fs, "sample_rate")),
        n_mels=int(pick("n_mels", default.n_mels, "num_mel_bins")),
        frame_length=float(pick("frame_length", default.frame_length)),
        frame_shift=float(pick("frame_shift", default.frame_shift)),
        lfr_m=int(pick("lfr_m", default.lfr_m)),
        lfr_n=int(pick("lfr_n", default.lfr_n)),
        window=str(pick("window", default.window, "window_type")),
        dither=float(pick("dither", default.dither)),
        preemphasis=float(pick("preemphasis_coefficient", default.preemphasis)),
        remove_dc_offset=bool(pick("remove_dc_offset", default.remove_dc_offset)),
        snip_edges=bool(pick("snip_edges", default.snip_edges)),
        low_freq=float(pick("low_freq", default.low_freq)),
        high_freq=float(pick("high_freq", default.high_freq)),
    )


def read_wav(path: Path, expected_fs: int) -> np.ndarray:
    sample_rate, samples = wavfile.read(path)
    if sample_rate != expected_fs:
        raise RuntimeError(f"expected {expected_fs} Hz WAV, found {sample_rate} Hz")
    if samples.ndim != 1:
        raise RuntimeError(f"expected mono WAV, found shape {samples.shape}")
    if samples.dtype == np.float32 or samples.dtype == np.float64:
        waveform = samples.astype(np.float32, copy=False)
    elif samples.dtype == np.int16:
        waveform = samples.astype(np.float32) / 32768.0
    elif samples.dtype == np.int32:
        waveform = samples.astype(np.float32) / 2147483648.0
    elif samples.dtype == np.uint8:
        waveform = (samples.astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError(f"unsupported WAV dtype: {samples.dtype}")
    if not np.isfinite(waveform).all():
        raise RuntimeError("WAV contains NaN or Inf")
    return np.ascontiguousarray(waveform)


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _window(name: str, size: int) -> np.ndarray:
    key = name.lower()
    phase = 2.0 * np.pi * np.arange(size, dtype=np.float64) / (size - 1)
    if key == "hamming":
        result = 0.54 - 0.46 * np.cos(phase)
    elif key in ("hanning", "hann"):
        result = 0.5 - 0.5 * np.cos(phase)
    elif key == "povey":
        result = (0.5 - 0.5 * np.cos(phase)) ** 0.85
    elif key == "rectangular":
        result = np.ones(size, dtype=np.float64)
    elif key == "blackman":
        result = 0.42 - 0.5 * np.cos(phase) + 0.08 * np.cos(2.0 * phase)
    else:
        raise RuntimeError(f"unsupported Kaldi window type: {name}")
    return result.astype(np.float32)


def _mel(value_hz: np.ndarray | float) -> np.ndarray:
    return 1127.0 * np.log1p(np.asarray(value_hz, dtype=np.float64) / 700.0)


def _mel_banks(config: FrontendConfig, fft_size: int) -> np.ndarray:
    # Kaldi excludes the Nyquist bin and constructs triangles in mel space.
    num_bins = fft_size // 2
    nyquist = 0.5 * config.fs
    high_freq = config.high_freq if config.high_freq > 0 else nyquist + config.high_freq
    if not 0 <= config.low_freq < high_freq <= nyquist:
        raise RuntimeError(
            f"invalid mel range: low={config.low_freq}, high={high_freq}, nyquist={nyquist}"
        )
    mel_low = float(_mel(config.low_freq))
    mel_high = float(_mel(high_freq))
    mel_delta = (mel_high - mel_low) / (config.n_mels + 1)
    fft_mels = _mel(np.arange(num_bins, dtype=np.float64) * config.fs / fft_size)
    banks = np.zeros((config.n_mels, num_bins), dtype=np.float32)
    for index in range(config.n_mels):
        left = mel_low + index * mel_delta
        center = left + mel_delta
        right = center + mel_delta
        up = (fft_mels - left) / mel_delta
        down = (right - fft_mels) / mel_delta
        banks[index] = np.maximum(0.0, np.minimum(up, down)).astype(np.float32)
    return banks


def kaldi_fbank(waveform: np.ndarray, config: FrontendConfig) -> np.ndarray:
    frame_size = int(config.fs * config.frame_length / 1000.0)
    frame_shift = int(config.fs * config.frame_shift / 1000.0)
    if config.dither != 0.0:
        raise RuntimeError(
            "non-zero dither cannot reproduce a saved golden feature deterministically"
        )
    if not config.snip_edges:
        raise RuntimeError("the lightweight frontend currently requires snip_edges=true")
    if waveform.size < frame_size:
        raise RuntimeError("WAV is shorter than one analysis frame")
    frame_count = 1 + (waveform.size - frame_size) // frame_shift
    shape = (frame_count, frame_size)
    strides = (waveform.strides[0] * frame_shift, waveform.strides[0])
    frames = np.lib.stride_tricks.as_strided(waveform, shape=shape, strides=strides).copy()

    # FunASR scales normalized audio to Kaldi's int16 amplitude convention.
    frames *= np.float32(32768.0)
    if config.remove_dc_offset:
        frames -= np.mean(frames, axis=1, keepdims=True, dtype=np.float32)
    if config.preemphasis != 0.0:
        first = frames[:, :1] * np.float32(1.0 - config.preemphasis)
        rest = frames[:, 1:] - np.float32(config.preemphasis) * frames[:, :-1]
        frames = np.concatenate((first, rest), axis=1)
    frames *= _window(config.window, frame_size)[None, :]

    fft_size = _next_power_of_two(frame_size)
    spectrum = np.fft.rfft(frames, n=fft_size, axis=1)[:, : fft_size // 2]
    power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag).astype(
        np.float32
    )
    energies = power @ _mel_banks(config, fft_size).T
    return np.log(np.maximum(energies, np.finfo(np.float32).eps)).astype(np.float32)


def apply_lfr(features: np.ndarray, m: int, n: int) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] == 0:
        raise RuntimeError(f"invalid FBank shape: {features.shape}")
    left = (m - 1) // 2
    padded = np.concatenate((np.repeat(features[:1], left, axis=0), features), axis=0)
    output_frames = int(math.ceil(features.shape[0] / n))
    required = (output_frames - 1) * n + m
    if padded.shape[0] < required:
        padded = np.concatenate(
            (padded, np.repeat(padded[-1:], required - padded.shape[0], axis=0)), axis=0
        )
    return np.stack(
        [padded[index * n : index * n + m].reshape(-1) for index in range(output_frames)]
    ).astype(np.float32)


def load_cmvn(path: Path, expected_size: int) -> tuple[np.ndarray, np.ndarray]:
    text = path.read_text(encoding="utf-8")
    sections = re.findall(r"\[([^\]]+)\]", text, flags=re.DOTALL)
    # Kaldi's am.mvn contains a short Splice index block before the actual
    # AddShift and Rescale blocks.  Select vectors by feature width instead of
    # assuming that the first two bracketed blocks are CMVN parameters.
    vectors: list[np.ndarray] = []
    for section in sections:
        numbers = np.fromstring(section.replace("\n", " "), sep=" ", dtype=np.float32)
        if numbers.size == expected_size:
            vectors.append(numbers)
    if len(vectors) != 2:
        sizes = [
            int(np.fromstring(section.replace("\n", " "), sep=" ").size)
            for section in sections
        ]
        raise RuntimeError(
            f"could not find exactly two {expected_size}-value CMVN vectors in "
            f"{path}; bracketed block sizes={sizes}"
        )
    return vectors[0], vectors[1]


def extract_features(
    wav_path: Path, config_path: Path, cmvn_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, FrontendConfig]:
    config = load_config(config_path)
    waveform = read_wav(wav_path, config.fs)
    fbank, cmvn, config = extract_waveform_features(waveform, config_path, cmvn_path)
    return waveform, fbank, cmvn, config


def extract_waveform_features(
    waveform: np.ndarray, config_path: Path, cmvn_path: Path
) -> tuple[np.ndarray, np.ndarray, FrontendConfig]:
    """Extract model-ready features directly from normalized mono waveform.

    This is the production entry used by ``speech/asr/rknn_sensevoice_asr.py``;
    the WAV-based validation entry above remains unchanged for reproducibility.
    """

    config = load_config(config_path)
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if not np.isfinite(waveform).all():
        raise RuntimeError("waveform contains NaN or Inf")
    waveform = np.ascontiguousarray(waveform)
    fbank = kaldi_fbank(waveform, config)
    lfr = apply_lfr(fbank, config.lfr_m, config.lfr_n)
    shift, scale = load_cmvn(cmvn_path, lfr.shape[1])
    cmvn = ((lfr + shift[None, :]) * scale[None, :]).astype(np.float32)
    return fbank, cmvn, config
