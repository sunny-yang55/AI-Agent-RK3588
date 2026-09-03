#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cache="${XDG_CACHE_HOME:-$HOME/.cache}/ai-agent-rknn-convert"
env_dir="$cache/venv"
onnx="$cache/yolov5s_relu.onnx"
output="$root/models/vision/yolov5s_relu-rk3588-fp.rknn"
url="https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov5/yolov5s_relu.onnx"

mkdir -p "$cache" "$root/models/vision"

if [[ ! -x "$env_dir/bin/python" ]]; then
    python3 -m venv "$env_dir"
fi

"$env_dir/bin/python" -m pip install --quiet --upgrade pip
"$env_dir/bin/python" -m pip install --quiet "rknn-toolkit2==2.3.2"

if [[ ! -s "$onnx" ]]; then
    curl --fail --location --retry 3 --output "$onnx" "$url"
fi

"$env_dir/bin/python" "$root/scripts/convert_vision_model.py" "$onnx" "$output"
sha256sum "$onnx" "$output"
echo "[VisionModel] Ready: $output"
