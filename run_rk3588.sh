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
mkdir -p logs
if [[ "$AI_AGENT_UI_MODE" == "debug" ]]; then
    exec python main.py
else
    exec python main.py 2>>logs/voice-debug.log
fi
