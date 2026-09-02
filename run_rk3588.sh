#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")" && pwd)"
cd "$project_root"

venv_activate="${AI_AGENT_VENV_ACTIVATE:-$project_root/venv/bin/activate}"
if [[ ! -f "$venv_activate" ]]; then
    echo "[启动失败] 找不到虚拟环境: $venv_activate" >&2
    exit 1
fi

source "$venv_activate"
export AI_AGENT_ENV="${AI_AGENT_ENV:-.env.qwen}"
export AI_AGENT_UI_MODE="${AI_AGENT_UI_MODE:-user}"
export RKNN_LOG_LEVEL="${RKNN_LOG_LEVEL:-0}"
# Product defaults. Values explicitly exported by an operator still win.
export AI_AGENT_TTS_MODE="${AI_AGENT_TTS_MODE:-whole}"
export AI_AGENT_STREAM_TTS="${AI_AGENT_STREAM_TTS:-0}"
export AI_AGENT_EDGE_CHUNK_CHARS="${AI_AGENT_EDGE_CHUNK_CHARS:-45}"
export AI_AGENT_EDGE_TIMEOUT="${AI_AGENT_EDGE_TIMEOUT:-6}"
export AI_AGENT_EDGE_RETRIES="${AI_AGENT_EDGE_RETRIES:-0}"
export AI_AGENT_TTS_FAILOVER_RETRIES="${AI_AGENT_TTS_FAILOVER_RETRIES:-0}"
export AI_AGENT_TTS_OFFLINE_FALLBACK="${AI_AGENT_TTS_OFFLINE_FALLBACK:-0}"
export AI_AGENT_LLM_TIMEOUT="${AI_AGENT_LLM_TIMEOUT:-20}"
export AI_AGENT_LLM_RETRIES="${AI_AGENT_LLM_RETRIES:-1}"
export AI_AGENT_LLM_RETRY_DELAY="${AI_AGENT_LLM_RETRY_DELAY:-0.5}"
zipformer_runtime="$project_root/vendor/zipformer-runtime-site"
if [[ -d "$zipformer_runtime/torch" ]]; then
    export PYTHONPATH="$zipformer_runtime${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="$zipformer_runtime/torch/lib:$zipformer_runtime/torch.libs:$zipformer_runtime/kaldifeat/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
mkdir -p logs
if [[ "$AI_AGENT_UI_MODE" == "debug" ]]; then
    exec python main.py
else
    exec python main.py 2>>logs/voice-debug.log
fi
