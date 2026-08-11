# AI-Agent-RK3588

RK3588 迁移与部署仓库。

当前唯一目标：在第一台 NanoPC-T6（RK3588）上跑通
`v1.2-real-audio`，再考虑仓库整理、镜像制作和剩余实验箱部署。

当前阶段代码位于 [`speech_v1_2/`](speech_v1_2/README.md)。
`speech_v1_1/` 保留为固定随机张量的 ONNX/RKNN 数值一致性基线。

## v1.1 最小验收链路

```text
Windows ONNXRuntime 基线
        -> ONNX 非量化转换为 RKNN
        -> RK3588 NPU 固定输入推理
        -> ONNX/RKNN logits 对比
```

模型与推理产物体积较大，不提交 Git，通过 SCP 传到开发板。

## v1.2 真实音频链路

```text
16 kHz 音频
        -> FunASR 1.1.3 同源 FBank/LFR/CMVN
        -> 96 帧声学特征 + 有效长度
        -> ONNX 内部拼接 4 个 SenseVoice 查询帧
        -> RK3588 NPU 推理
        -> CTC 贪心解码
```

v1.2 详情与逐步命令见 [`speech_v1_2/README.md`](speech_v1_2/README.md)。
