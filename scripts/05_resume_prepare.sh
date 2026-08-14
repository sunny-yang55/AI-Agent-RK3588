#!/usr/bin/env bash
set -euo pipefail

current_root="${AI_AGENT_CURRENT_ROOT:-/home/ztl/AI-Agent-RK3588}"
legacy_root="${AI_AGENT_LEGACY_ROOT:-/home/ztl/AI-Agent}"
staging_root="${AI_AGENT_STAGING_ROOT:-/home/ztl/AI-Agent-RK3588-unified}"
package_root="$(cd "$(dirname "$0")/.." && pwd)"

fail() { echo "[续跑失败] $*" >&2; exit 1; }
need_file() { [[ -f "$1" ]] || fail "缺少文件: $1"; }

[[ -d "$staging_root" ]] || fail "找不到半成品目录: $staging_root"
[[ ! -f "$staging_root/MIGRATION_STATE.txt" ]] || fail "准备已完成，不需要续跑"
need_file "$staging_root/models/speech/sensevoice/sensevoice_time100_fp.rknn"
need_file "$staging_root/models/speech/sensevoice/config.yaml"
need_file "$staging_root/models/speech/sensevoice/am.mvn"
need_file "$staging_root/models/speech/sensevoice/tokens.json"
need_file "$staging_root/models/speech/tts/zh_CN-huayan-medium.onnx"
need_file "$staging_root/models/speech/tts/zh_CN-huayan-medium.onnx.json"

mkdir -p "$staging_root/vendor/wheels"
lite_wheel="$(find /home/ztl -maxdepth 4 -type f \
    -name 'rknn_toolkit_lite2-2.3.2-cp310-*.whl' -print -quit)"
if [[ -n "$lite_wheel" ]]; then
    cp -a "$lite_wheel" "$staging_root/vendor/wheels/"
else
    legacy_python="$legacy_root/venv/bin/python"
    [[ -x "$legacy_python" ]] || fail "找不到旧环境 Python: $legacy_python"
    [[ ! -e "$staging_root/vendor/rknn-runtime-site" ]] || \
        fail "RKNN 导出目录已存在，请先人工检查"
    "$legacy_python" "$package_root/scripts/export_installed_rknn.py" \
        "$staging_root/vendor/rknn-runtime-site"
fi

cp -a "$package_root/overlay/." "$staging_root/"
cp -a "$package_root/scripts" "$package_root/systemd" "$staging_root/"
chmod +x "$staging_root/scripts/"*.sh "$staging_root/run_rk3588.sh"

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
rknn_source=installed-runtime-export
EOF

echo "[续跑完成] $staging_root"
echo "当前正式项目没有被修改或删除。"
