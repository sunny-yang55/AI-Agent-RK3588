#!/usr/bin/env bash
set -euo pipefail

final_root="${AI_AGENT_FINAL_ROOT:-/home/ztl/AI-Agent-RK3588}"
legacy_root="${AI_AGENT_LEGACY_ROOT:-/home/ztl/AI-Agent}"
state_file="$final_root/MIGRATION_STATE.txt"

[[ -f "$state_file" ]] || { echo "缺少迁移状态文件" >&2; exit 1; }
grep -q '^state=activated$' "$state_file" || { echo "统一版本尚未激活" >&2; exit 1; }
"$final_root/scripts/healthcheck.sh"

echo "该脚本不会永久删除文件，只会把旧目录改名为 quarantine。"
read -r -p "确认语音实测已经通过，输入 QUARANTINE：" answer
[[ "$answer" == "QUARANTINE" ]] || { echo "已取消"; exit 1; }

stamp="$(date +%Y%m%d-%H%M%S)"
if [[ -d "$legacy_root" ]]; then
    mv "$legacy_root" "${legacy_root}-quarantine-${stamp}"
    echo "已隔离旧环境: ${legacy_root}-quarantine-${stamp}"
fi

echo "保留激活时生成的 AI-Agent-RK3588-backup-* 至少完成一次重启测试。"
echo "确认镜像恢复测试成功后，再人工删除 quarantine 和 backup。"
