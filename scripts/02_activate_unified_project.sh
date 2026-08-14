#!/usr/bin/env bash
set -euo pipefail

final_root="${AI_AGENT_FINAL_ROOT:-/home/ztl/AI-Agent-RK3588}"
staging_root="${AI_AGENT_STAGING_ROOT:-/home/ztl/AI-Agent-RK3588-unified}"
stamp="$(date +%Y%m%d-%H%M%S)"
backup_root="${final_root}-backup-${stamp}"
failed_root="${final_root}-failed-${stamp}"

[[ -d "$final_root" ]] || { echo "找不到当前项目: $final_root" >&2; exit 1; }
[[ -d "$staging_root" ]] || { echo "找不到统一项目: $staging_root" >&2; exit 1; }
[[ -f "$staging_root/MIGRATION_STATE.txt" ]] || { echo "统一项目未完成准备" >&2; exit 1; }

if pgrep -af "python.*${final_root}/main.py" >/dev/null; then
    echo "检测到 AI-Agent 正在运行，请先正常退出。" >&2
    exit 1
fi

echo "即将切换目录："
echo "  当前版本 -> $backup_root"
echo "  统一版本 -> $final_root"
echo "旧版本只改名保留，不会删除。"
read -r -p "输入 ACTIVATE 继续：" answer
[[ "$answer" == "ACTIVATE" ]] || { echo "已取消"; exit 1; }

mv "$final_root" "$backup_root"
mv "$staging_root" "$final_root"

rollback() {
    status=$?
    if [[ $status -ne 0 ]]; then
        echo "[切换失败] 正在恢复旧版本..." >&2
        [[ -d "$final_root" ]] && mv "$final_root" "$failed_root"
        [[ -d "$backup_root" ]] && mv "$backup_root" "$final_root"
        echo "旧版本已恢复；失败版本保留在: $failed_root" >&2
    fi
    exit $status
}
trap rollback EXIT

"$final_root/scripts/install_runtime.sh" "$final_root"
(
    cd "$final_root"
    ./venv/bin/python -m unittest discover -s tests -v
    ./scripts/healthcheck.sh --no-audio
)

sed -i 's/^state=prepared$/state=activated/' "$final_root/MIGRATION_STATE.txt"
printf 'activated_at=%s\nbackup=%s\n' \
    "$(date --iso-8601=seconds)" "$backup_root" >> "$final_root/MIGRATION_STATE.txt"

trap - EXIT
echo "[切换完成] 新项目: $final_root"
echo "[可回滚备份] $backup_root"
echo "现在运行 ./run_rk3588.sh 完成真实麦克风、LLM、TTS 回归。"
