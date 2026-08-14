# AI-Agent-RK3588 语音助手开发与部署手册

## 1. 文档范围

本文记录 `v1.3.14-voice-ui-freeze` 的架构、开发过程、配置、常见问题、验证方法和 Linux 操作。语音模块在本版后暂时冻结，后续工作应优先保证视觉与机械臂模块通过能力层接入，避免直接侵入稳定语音链路。

## 2. 运行链路

```text
USB 麦克风
  → ALSA 采集
  → VAD 起止点检测
  → SenseVoice RKNN（RK3588 NPU）
  → 文本清洗/实体纠错/上下文补全
  → Qwen 兼容 API
  → Edge-TTS 在线合成
  → Piper/espeak 离线回退
  → ALSA/pygame 播放
```

播报期间另有一条受限监听链路，只接受带“小安”的停止、退出或休眠命令，避免扬声器回声造成误打断。

## 3. 主要模块

| 路径 | 职责 |
| --- | --- |
| `main.py` | 程序入口与 Agent 初始化 |
| `runtime/runtime_manager.py` | 语音循环、多轮上下文、LLM 和 TTS 调度 |
| `voice_ui.py` | 用户模式界面与调试日志分流 |
| `speech/audio/alsa_microphone.py` | USB 麦克风选择、ALSA 录音与 VAD |
| `speech/asr/chunked_rknn_sensevoice_asr.py` | 长语音分段和去重 |
| `speech/asr/rknn_sensevoice_asr.py` | RKNN 模型推理与文本解码 |
| `tools/speech/speech_tool.py` | 持续对话、本地控制命令、打断监听 |
| `speech/tts/edge_tts_backend.py` | 在线自然语音、分段、超时和播放 |
| `speech/tts/piper_tts_backend.py` | 离线 TTS 回退 |
| `speech/tts/tts_engine.py` | TTS 后端选择和故障回退 |
| `tools/common/context_resolver.py` | 省略主语的多轮上下文补全 |
| `tools/llm/adapter.py` | LLM API、流式输出与回答长度约束 |
| `run_rk3588.sh` | 环境加载、用户/调试模式和启动入口 |

## 4. 核心知识与技术

### 4.1 ALSA 与 VAD

ALSA 是 Linux 音频设备接口。程序优先使用稳定的 `plughw:CARD=...,DEV=...` 名称连接 USB 麦克风，避免声卡编号变化。VAD 根据音频能量判断开口和静音，控制 3 秒开口等待、10 秒最长录音和约 0.7 秒尾部静音。

### 4.2 SenseVoice 与 RKNN

SenseVoice 模型先从原框架导出 ONNX，再由 RKNN Toolkit 2 转换为 RK3588 的 `.rknn`。板端使用 RKNN Lite Runtime 调用 NPU。静态形状模型出现 dynamic range 查询警告不代表推理失败；应以输出形状、数值对齐和实际识别结果判断。

已完成的模型一致性验证包括余弦相似度、MAE、最大绝对误差和 token Top-1 一致率。产品镜像只应携带最终 `.rknn` 和运行所需词表/归一化文件，不携带多份 894 MB 的中间 ONNX。

### 4.3 多轮对话

普通对话采用持续监听，不要求每轮说唤醒词。原始识别文本用于用户界面；实体纠错、记忆改写和上下文补全只作为内部 LLM 输入。例如用户说“它有哪些专业”，界面保持原话，内部可补全为“安徽工程大学有哪些专业”。

### 4.4 TTS 与打断

Edge-TTS 音色自然，但依赖网络，可能出现首段等待或 `No audio was received`。当前实现对单段合成设置硬超时和有限重试，失败段由 Piper 补播，避免整篇重复。Piper 离线可靠但音色较机械，仅作为兜底。

打断监听必须在真实播放开始后启动，否则会在 Edge 合成等待期抢占麦克风。停止事件会同时通知播放器与监听线程，防止上一轮状态污染下一轮。

## 5. 重要配置

当前配置示例位于 `config/.env.qwen` 或对应模板。真实文件不得提交 Git。

```text
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
AI_AGENT_TTS_BACKEND=edge
AI_AGENT_EDGE_VOICE=zh-CN-XiaoxiaoNeural
AI_AGENT_ALWAYS_LISTEN=1
AI_AGENT_UI_MODE=user
```

常用开关：

| 变量 | 作用 |
| --- | --- |
| `AI_AGENT_UI_MODE=user` | 简洁用户界面 |
| `AI_AGENT_UI_MODE=debug` | 在终端显示完整技术日志 |
| `AI_AGENT_ALWAYS_LISTEN=1` | 普通多轮对话无需唤醒词 |
| `AI_AGENT_BARGE_IN_ENABLED=1` | 允许“小安停止”打断 |
| `AI_AGENT_EDGE_TIMEOUT` | 单段在线 TTS 超时 |
| `AI_AGENT_EDGE_RETRIES` | Edge 单段重试次数 |
| `AI_AGENT_MAX_RESPONSE_TOKENS` | LLM 最大输出 token 保护值 |

## 6. 开发中遇到的问题与处理

| 问题 | 根因 | 处理 |
| --- | --- | --- |
| 普通多轮对话仍要求唤醒词 | 唤醒门控仍在正常输入路径 | 默认持续监听；唤醒词仅用于播报打断 |
| TTS 文字出现后等待很久 | 在线合成、分段并行逻辑和网络抖动 | 首段就绪即播、单段硬超时、有限重试 |
| 播报只到第二句或段落提前切换 | 播放完成判断不准确 | 使用解码音频实际时长和播放状态判断 |
| Edge 返回 No audio | 在线服务偶发无音频 | 失败段局部回退 Piper，避免整篇重播 |
| 播报监听误打断 | ASR 听到扬声器回声 | 打断必须包含“小安”且播放开始后才监听 |
| 板端本地运行一度比 SSH 慢 | 并非算力/网络基础问题，主要是不同运行状态与音频线程表现 | 对比路由、DNS、负载、温度、进程和日志；统一启动入口 |
| ALSA 无数据 | USB 设备偶发未返回 PCM | 墙钟超时、设备重新探测和短暂恢复等待 |
| 界面日志刷屏 | VAD 噪声触发后仍显示识别状态 | 用户模式隐藏无效识别过程，完整日志写文件 |
| 上下文改写显示成用户原话 | UI 使用了内部 resolved text | 分离 display text 与 LLM input text |

## 7. 安装、测试与运行

```bash
cd /home/ztl/AI-Agent-RK3588
source /home/ztl/AI-Agent-RK3588/venv/bin/activate
python -m unittest discover -s tests -v
./run_rk3588.sh
```

完整调试模式：

```bash
cd /home/ztl/AI-Agent-RK3588
AI_AGENT_UI_MODE=debug ./run_rk3588.sh
```

查看最近日志：

```bash
tail -n 100 logs/voice-debug.log
tail -f logs/voice-debug.log
```

## 8. Linux 常用命令

| 命令 | 作用 |
| --- | --- |
| `pwd` | 显示当前目录 |
| `ls -lah` | 显示文件、权限和大小 |
| `cd /path` | 切换目录 |
| `du -sh DIR` | 查看目录总大小 |
| `du -ah DIR | sort -h | tail` | 查找大文件 |
| `df -h` | 查看磁盘空间 |
| `free -h` | 查看内存 |
| `chmod +x FILE` | 添加执行权限 |
| `source venv/bin/activate` | 激活 Python 虚拟环境 |
| `python -m unittest discover -s tests -v` | 运行测试 |
| `pgrep -af PATTERN` | 查找相关进程 |
| `fuser -v /dev/snd/*` | 查看音频设备占用 |
| `aplay -l`、`arecord -l` | 列出播放/录音设备 |
| `tar -xzf FILE.tar.gz` | 解压 gzip tar 包 |
| `sha256sum FILE` | 计算文件校验值 |
| `git status --short` | 查看工作区修改 |
| `git diff` | 查看未提交变化 |
| `git log --oneline -10` | 查看最近提交 |
| `git tag --list` | 查看版本 Tag |
| `journalctl -u SERVICE -f` | 实时查看 systemd 服务日志 |

Linux 命令区分大小写；`grep` 不能写成 `GREP`。环境变量名包含下划线，读取方式如 `echo "$AI_AGENT_ENV"`，不能写成 `$AI-AGENT_ENV`。

## 9. 维护要求

1. 语音冻结代码变更必须先新增回归测试。
2. 视觉和机械臂通过 `capability/` 与 `embodied/` 接入，不直接修改 ASR/TTS 内部实现。
3. 密钥、日志、虚拟环境、模型中间文件和系统镜像不得普通提交 Git。
4. 每个发布版本记录代码 Tag、模型 SHA-256、依赖版本和板端驱动版本。
5. 产品镜像发布前必须在第二张存储介质上做恢复验证。

## 10. 当前已知限制

- Edge-TTS 依赖外网，网络异常时离线音色自然度下降。
- 麦克风距离、回声和环境噪声仍会影响 ASR。
- LLM 生成的时效性事实可能不准确，需要增加联网检索或权威知识库后才能用于高准确性场景。
