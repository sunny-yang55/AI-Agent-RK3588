#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="$project_root/venv/bin/python"
audio_check=1
[[ "${1:-}" == "--no-audio" ]] && audio_check=0

failures=0
check_file() {
    if [[ -f "$1" ]]; then echo "[OK] $1"; else echo "[FAIL] $1"; failures=$((failures + 1)); fi
}

check_file "$project_root/VERSION"
check_file "$python_bin"
check_file "$project_root/models/speech/sensevoice/sensevoice_time100_fp.rknn"
check_file "$project_root/models/speech/sensevoice/config.yaml"
check_file "$project_root/models/speech/sensevoice/am.mvn"
check_file "$project_root/models/speech/sensevoice/tokens.json"
check_file "$project_root/models/speech/tts/zh_CN-huayan-medium.onnx"
check_file "$project_root/models/speech/tts/zh_CN-huayan-medium.onnx.json"

if [[ -f "$project_root/models/manifest.sha256" ]]; then
    (cd "$project_root" && sha256sum -c models/manifest.sha256) || failures=$((failures + 1))
else
    echo "[FAIL] models/manifest.sha256"
    failures=$((failures + 1))
fi

if [[ -x "$python_bin" ]]; then
    "$python_bin" - <<'PY' || failures=$((failures + 1))
import edge_tts, numpy, openai, pygame, scipy, yaml
from rknnlite.api import RKNNLite
from speech.asr.lightweight_frontend import extract_waveform_features
print("[OK] Python runtime imports")
PY
fi

if (( audio_check )); then
    command -v arecord >/dev/null && arecord -l || failures=$((failures + 1))
    command -v aplay >/dev/null && aplay -l || failures=$((failures + 1))
fi

if (( failures > 0 )); then
    echo "[健康检查失败] $failures 项"
    exit 1
fi
echo "[健康检查通过]"
