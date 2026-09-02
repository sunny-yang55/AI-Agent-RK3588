#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runtime_site="$project_root/vendor/zipformer-runtime-site"

if [[ ! -x "$project_root/venv/bin/python" ]]; then
    echo "[失败] 缺少项目虚拟环境: $project_root/venv" >&2
    exit 1
fi
if [[ ! -d "$runtime_site/torch" || ! -d "$runtime_site/kaldifeat" ]]; then
    echo "[失败] Zipformer运行时不完整: $runtime_site" >&2
    exit 1
fi

export PYTHONPATH="$runtime_site${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$runtime_site/torch/lib:$runtime_site/torch.libs:$runtime_site/kaldifeat/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$project_root/venv/bin/python" \
    "$project_root/scripts/test_zipformer_microphone.py" "$@"
