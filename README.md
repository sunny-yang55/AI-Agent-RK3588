# AI-Agent-RK3588

RK3588 迁移与部署仓库。

当前唯一目标：在第一台 NanoPC-T6（RK3588）上跑通
`v1.1-rk3588-speech`，再考虑镜像制作和剩余 20 台实验箱部署。

当前阶段代码位于 [`speech_v1_1/`](speech_v1_1/README.md)。

## v1.1 最小验收链路

```text
Windows ONNXRuntime 基线
        -> ONNX 非量化转换为 RKNN
        -> RK3588 NPU 固定输入推理
        -> ONNX/RKNN logits 对比
```

模型与推理产物体积较大，不提交 Git，通过 SCP 传到开发板。
