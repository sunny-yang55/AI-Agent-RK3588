# v1.1-rk3588-speech

本目录只负责先跑通第一台 RK3588 的 SenseVoice NPU 最小闭环，不包含
麦克风、VAD、Agent、LLM、TTS 和批量镜像部署。

## 已知模型契约

| 项目 | 当前值 |
|---|---|
| ONNX | `sensevoice_fixed.onnx` |
| ONNX 大小 | `937562722` bytes |
| 输入 | `speech`, float32, `[1, 30, 560]` |
| 输出 | `logits`, float32, `[1, 30, 25055]` |
| RKNN Toolkit2 | `2.3.2` |
| 目标芯片 | `RK3588` |
| 第一轮量化 | 关闭 |

`[1, 30, 560]` 当前只作为转换和 NPU 一致性验证契约。它能覆盖多长的真实
语音，需要等实际 FBank/LFR 特征接入后再确认。

## 目录

```text
speech_v1_1/
├── scripts/
│   ├── 00_check_env.py
│   ├── 01_make_onnx_baseline.py
│   ├── 02_convert_onnx_to_rknn.py
│   └── 03_run_rknn_lite.py
├── models/       # ONNX/RKNN，不提交 Git
└── artifacts/    # speech.npy、logits 和日志，不提交 Git
```

## 1. Windows 生成同源基线

在 `E:\AI-Agent-RK3588` 中执行：

```powershell
python -m pip install -r speech_v1_1\requirements-windows.txt

python speech_v1_1\scripts\01_make_onnx_baseline.py `
  --onnx E:\AI-Agent\sensevoice_rknn\sensevoice_fixed.onnx `
  --random `
  --input-out speech_v1_1\artifacts\speech.npy `
  --logits-out speech_v1_1\artifacts\onnx_logits.npy
```

这里的随机输入只用于判断 ONNX 与 RKNN 数值是否一致，不代表真实识别效果。
后续接真实音频时，用真实前处理生成的 `speech.npy` 覆盖它即可。

## 2. 传输到第一台 RK3588

```powershell
scp E:\AI-Agent\sensevoice_rknn\sensevoice_fixed.onnx yarce@192.168.1.128:/home/yarce/AI-Agent-RK3588/speech_v1_1/models/
scp speech_v1_1\artifacts\speech.npy yarce@192.168.1.128:/home/yarce/AI-Agent-RK3588/speech_v1_1/artifacts/
scp speech_v1_1\artifacts\onnx_logits.npy yarce@192.168.1.128:/home/yarce/AI-Agent-RK3588/speech_v1_1/artifacts/
```

## 3. RK3588 环境检查

```bash
cd /home/yarce/AI-Agent-RK3588
python3 speech_v1_1/scripts/00_check_env.py
```

必须至少看到：

- `machine: aarch64`
- `rknn.api.RKNN: OK`
- `rknnlite.api.RKNNLite: OK`

Toolkit2 用于转换，Toolkit-Lite2 用于板端 NPU 推理。两者应使用相同的
`2.3.2` 版本。

## 4. 非量化 ONNX 转 RKNN

转换时间可能较长，完整日志必须保留：

```bash
python3 speech_v1_1/scripts/02_convert_onnx_to_rknn.py \
  --onnx speech_v1_1/models/sensevoice_fixed.onnx \
  --output speech_v1_1/models/sensevoice_fixed_fp.rknn \
  2>&1 | tee speech_v1_1/artifacts/rknn_build.log
```

通过标准：脚本显示 `RKNN export: PASS`，并生成非空 `.rknn` 文件。

## 5. RK3588 NPU 推理与对比

```bash
python3 speech_v1_1/scripts/03_run_rknn_lite.py \
  --model speech_v1_1/models/sensevoice_fixed_fp.rknn \
  --input speech_v1_1/artifacts/speech.npy \
  --golden speech_v1_1/artifacts/onnx_logits.npy \
  --output speech_v1_1/artifacts/rknn_logits.npy \
  --runs 3 \
  2>&1 | tee speech_v1_1/artifacts/rknn_infer.log
```

第一轮通过标准：

- RKNN 输出形状为 `[1, 30, 25055]`；
- 输出中没有 NaN/Inf；
- token Top-1 agreement 达到 `99%` 以上；
- 连续 3 次推理不崩溃。

如果构建失败，保留完整 `rknn_build.log`，从第一个 `E RKNN` 开始定位，
不要只截取最后一行。
