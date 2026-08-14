#!/usr/bin/env bash
set -euo pipefail

current_root="${AI_AGENT_CURRENT_ROOT:-/home/ztl/AI-Agent-RK3588}"
legacy_root="${AI_AGENT_LEGACY_ROOT:-/home/ztl/AI-Agent}"
staging_root="${AI_AGENT_STAGING_ROOT:-/home/ztl/AI-Agent-RK3588-unified}"
package_root="$(cd "$(dirname "$0")/.." && pwd)"

fail() { echo "[准备失败] $*" >&2; exit 1; }
need_file() { [[ -f "$1" ]] || fail "缺少文件: $1"; }

[[ -d "$current_root" ]] || fail "找不到当前项目: $current_root"
[[ ! -e "$staging_root" ]] || fail "目标已存在，请先人工检查并改名: $staging_root"

need_file "$current_root/speech_v1_2/models/sensevoice_time100_fp.rknn"
need_file "$current_root/speech_v1_3/reference/config.yaml"
need_file "$current_root/speech_v1_3/reference/am.mvn"
need_file "$current_root/speech_v1_3/reference/tokens.json"
need_file "$current_root/models/tts/zh_CN-huayan-medium.onnx"

mkdir -p "$staging_root"

# Copy only maintained runtime/source directories. Historical conversion trees,
# caches, reports and intermediate ONNX files are intentionally excluded.
for item in .git agent capability config core embodied remote_asr runtime speech tests tools; do
    [[ -e "$current_root/$item" ]] || fail "当前项目缺少: $item"
    cp -a "$current_root/$item" "$staging_root/"
done

for item in main.py voice_ui.py; do
    need_file "$current_root/$item"
    cp -a "$current_root/$item" "$staging_root/"
done

mkdir -p \
    "$staging_root/models/speech/sensevoice" \
    "$staging_root/models/speech/tts" \
    "$staging_root/models/vision" \
    "$staging_root/vendor/wheels" \
    "$staging_root/logs" \
    "$staging_root/data"

cp -a "$current_root/speech_v1_2/models/sensevoice_time100_fp.rknn" \
    "$staging_root/models/speech/sensevoice/"
cp -a "$current_root/speech_v1_3/reference/config.yaml" \
    "$current_root/speech_v1_3/reference/am.mvn" \
    "$current_root/speech_v1_3/reference/tokens.json" \
    "$staging_root/models/speech/sensevoice/"
cp -a "$current_root/models/tts/zh_CN-huayan-medium.onnx" \
    "$staging_root/models/speech/tts/"

if [[ -f "$current_root/models/tts/zh_CN-huayan-medium.onnx.json" ]]; then
    cp -a "$current_root/models/tts/zh_CN-huayan-medium.onnx.json" \
        "$staging_root/models/speech/tts/"
elif [[ -f "$current_root/models/tts/huayan-download/model.onnx.json" ]]; then
    cp -a "$current_root/models/tts/huayan-download/model.onnx.json" \
        "$staging_root/models/speech/tts/zh_CN-huayan-medium.onnx.json"
else
    fail "找不到 Piper 模型 JSON"
fi

if [[ -f "$current_root/yolov8n.pt" ]]; then
    cp -a "$current_root/yolov8n.pt" "$staging_root/models/vision/yolov8n.pt"
fi

lite_wheel="$(find /home/ztl -maxdepth 4 -type f \
    -name 'rknn_toolkit_lite2-2.3.2-cp310-*.whl' -print -quit)"
if [[ -n "$lite_wheel" ]]; then
    cp -a "$lite_wheel" "$staging_root/vendor/wheels/"
else
    legacy_python="$legacy_root/venv/bin/python"
    [[ -x "$legacy_python" ]] || fail "找不到 RKNN wheel，也找不到旧环境 Python"
    "$legacy_python" "$package_root/scripts/export_installed_rknn.py" \
        "$staging_root/vendor/rknn-runtime-site"
fi

# Overlay path-independent code, deployment files and consolidated manuals.
cp -a "$package_root/overlay/." "$staging_root/"
cp -a "$package_root/scripts" "$package_root/systemd" "$staging_root/"
chmod +x "$staging_root/scripts/"*.sh
chmod +x "$staging_root/run_rk3588.sh"

# Keep the private LLM settings, but make local model paths independent from
# the historical directory layout.
if [[ -f "$staging_root/config/.env.qwen" ]]; then
    if grep -q '^PIPER_MODEL=' "$staging_root/config/.env.qwen"; then
        sed -i \
            's#^PIPER_MODEL=.*#PIPER_MODEL=models/speech/tts/zh_CN-huayan-medium.onnx#' \
            "$staging_root/config/.env.qwen"
    else
        printf '\nPIPER_MODEL=models/speech/tts/zh_CN-huayan-medium.onnx\n' \
            >> "$staging_root/config/.env.qwen"
    fi
fi

(
    cd "$staging_root"
    find models -type f ! -name manifest.sha256 -print0 \
        | sort -z | xargs -0 sha256sum > models/manifest.sha256
)

cat > "$staging_root/MIGRATION_STATE.txt" <<EOF
state=prepared
prepared_at=$(date --iso-8601=seconds)
source=$current_root
legacy_environment=$legacy_root/venv
target=$staging_root
EOF

echo "[准备完成] 新的统一项目已生成：$staging_root"
echo "当前项目没有被修改或删除。"
echo "下一步先检查目录，再运行 scripts/02_activate_unified_project.sh"
