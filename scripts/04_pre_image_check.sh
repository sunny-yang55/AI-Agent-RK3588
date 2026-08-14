#!/usr/bin/env bash
set -euo pipefail

root="${AI_AGENT_FINAL_ROOT:-/home/ztl/AI-Agent-RK3588}"
cd "$root"

./scripts/healthcheck.sh
./venv/bin/python -m unittest discover -s tests -v

if grep -RIlE '(API_KEY|TOKEN|SECRET)=|sk-|Bearer [A-Za-z0-9]' config \
    --exclude='*.example' 2>/dev/null | grep -q .; then
    echo "[镜像阻止] config 中可能存在真实密钥，请在制作公共/批量镜像前移除。" >&2
    exit 1
fi

find . -type d -name __pycache__ -prune -print
find logs -type f -size +0 -print 2>/dev/null || true

echo "[镜像前检查通过]"
echo "仍需人工完成：清理 SSH host/user key、machine-id、shell history 和网络唯一配置。"
