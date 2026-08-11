# v1.2-real-audio

本阶段只验证第一台 RK3588 上的固定短语音：同一段真实音频必须在官方 FunASR、
新 ONNX 和 RK3588 RKNN 三条路径得到相同中文文本。暂不接麦克风、VAD、Agent、
LLM、TTS，也不制作批量镜像。

## 已确认的模型语义

| 项目 | 值 |
|---|---|
| FunASR | 固定 `1.1.3`，本阶段不要升级 |
| 声学前处理 | 16 kHz、80维 FBank、LFR `(7,6)`、`am.mvn` CMVN |
| 单个声学帧 | 560维，约前进 60 ms |
| 最大声学帧 | 96帧，约 5.76 s |
| 查询帧 | `[zh,event,emotion,withitn] = [3,1,2,14]` |
| ONNX输入1 | `speech`, float32, `[1,96,560]` |
| ONNX输入2 | `speech_lengths`, float32, `[1]` |
| ONNX输出 | `logits`, float32, `[1,100,25055]` |
| CTC blank | `0` |

官方 `example/zh.mp3` 产生 94 个声学帧；加 4 个查询帧后有效总长度为 98，
可由本模型完整处理。补齐到 96 的两个零帧由 `speech_lengths=94` 屏蔽。

## 1. Windows：保持已验证环境

不要执行 FunASR 的自动升级提示。先确认版本：

```powershell
python -c "import funasr; print(funasr.__version__)"
```

必须输出 `1.1.3`。模型目录：

```powershell
$modelRoot = "C:\Users\PJYang\.cache\modelscope\models\iic--SenseVoiceSmall\snapshots\master"
```

## 2. Windows：导出带查询帧的 TIME=100 ONNX

```powershell
python speech_v1_2\scripts\01_export_time100_onnx.py `
  --model-root $modelRoot `
  --output speech_v1_2\models\sensevoice_time100.onnx
```

新包装器在模型内部拼接四个查询帧，并保留独立的有效长度输入。注意力掩码始终
保持100帧静态宽度，只把有效长度之后的位置标记为padding，以兼容ONNX和RKNN
的固定输入图。导出脚本还会用94帧有效长度立即实跑一次ONNX，只有该回归检查
通过才会输出 `ONNX export: PASS`。不要覆盖 v1.1 的 `sensevoice_fixed.onnx`。

## 3. Windows：生成真实音频特征与官方文本

```powershell
$audio = Join-Path $modelRoot "example\zh.mp3"

python speech_v1_2\scripts\02_prepare_real_audio.py `
  --model-root $modelRoot `
  --audio $audio `
  --output-dir speech_v1_2\artifacts
```

预期看到：

```text
valid audio frames: 94
valid total frames: 98
official clean: 开放时间早上9点至下午5点。
real-audio preparation: PASS
```

## 4. Windows：验证 ONNX 语义

```powershell
python speech_v1_2\scripts\03_make_onnx_baseline.py `
  --onnx speech_v1_2\models\sensevoice_time100.onnx `
  --input-dir speech_v1_2\artifacts `
  --model-root $modelRoot `
  --logits-out speech_v1_2\artifacts\onnx_logits.npy `
  --text-out speech_v1_2\artifacts\onnx_text.txt
```

只有同时看到以下两项，才传到 RK3588：

```text
ONNX clean: 开放时间早上9点至下午5点。
ONNX real-audio baseline: PASS
```

## 5. 传到第一台 RK3588

```powershell
scp speech_v1_2\models\sensevoice_time100.onnx yarce@192.168.1.128:/home/yarce/AI-Agent-RK3588/speech_v1_2/models/
scp "$modelRoot\tokens.json" yarce@192.168.1.128:/home/yarce/AI-Agent-RK3588/speech_v1_2/models/
scp speech_v1_2\artifacts\speech.npy yarce@192.168.1.128:/home/yarce/AI-Agent-RK3588/speech_v1_2/artifacts/
scp speech_v1_2\artifacts\speech_lengths.npy yarce@192.168.1.128:/home/yarce/AI-Agent-RK3588/speech_v1_2/artifacts/
scp speech_v1_2\artifacts\onnx_logits.npy yarce@192.168.1.128:/home/yarce/AI-Agent-RK3588/speech_v1_2/artifacts/
scp speech_v1_2\artifacts\official_text.txt yarce@192.168.1.128:/home/yarce/AI-Agent-RK3588/speech_v1_2/artifacts/
```

## 6. RK3588：转换和真实文本验收

```bash
cd /home/yarce/AI-Agent-RK3588

python3 speech_v1_2/scripts/04_convert_onnx_to_rknn.py \
  --onnx speech_v1_2/models/sensevoice_time100.onnx \
  --output speech_v1_2/models/sensevoice_time100_fp.rknn \
  2>&1 | tee speech_v1_2/artifacts/rknn_build.log

python3 speech_v1_2/scripts/05_run_rknn_real_audio.py \
  --model speech_v1_2/models/sensevoice_time100_fp.rknn \
  --input-dir speech_v1_2/artifacts \
  --tokens speech_v1_2/models/tokens.json \
  --golden speech_v1_2/artifacts/onnx_logits.npy \
  --output speech_v1_2/artifacts/rknn_logits.npy \
  --text-out speech_v1_2/artifacts/rknn_text.txt \
  --runs 3 \
  2>&1 | tee speech_v1_2/artifacts/rknn_infer.log
```

## 验收标准

- ONNX 与 RKNN 输出均为 `[1,100,25055]`，无 NaN/Inf；
- RKNN/ONNX frame Top-1 agreement 不低于 99%；
- RKNN clean、ONNX clean、official clean 三者都是
  `开放时间早上9点至下午5点。`；
- 连续 3 次 NPU 推理不崩溃，并记录中位延迟。

若 RKNN 转换器不支持 `speech_lengths` 输入或图中的长度转换，保留完整
`rknn_build.log`，从第一条 `E RKNN` 开始处理；不要回退成把长度硬编码为 100。
