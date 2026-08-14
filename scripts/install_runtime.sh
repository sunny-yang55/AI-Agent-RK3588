#!/usr/bin/env bash
set -euo pipefail

project_root="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${AI_AGENT_PYTHON:-python3.10}"
venv_root="$project_root/venv"

command -v "$python_bin" >/dev/null || {
    echo "找不到 $python_bin" >&2
    exit 1
}

[[ ! -e "$venv_root" ]] || {
    echo "虚拟环境已存在，拒绝覆盖: $venv_root" >&2
    exit 1
}

"$python_bin" -m venv "$venv_root"
"$venv_root/bin/python" -m pip install \
    -r "$project_root/requirements-rk3588-runtime.lock"

wheel="$(find "$project_root/vendor/wheels" -maxdepth 1 -type f \
    -name 'rknn_toolkit_lite2-2.3.2-cp310-*.whl' -print -quit)"
if [[ -n "$wheel" ]]; then
    "$venv_root/bin/python" -m pip install "$wheel"
elif [[ -d "$project_root/vendor/rknn-runtime-site" ]]; then
    target_site="$("$venv_root/bin/python" -c \
        'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    cp -a "$project_root/vendor/rknn-runtime-site/." "$target_site/"
else
    echo "缺少 RKNN Lite wheel 或已导出的运行包" >&2
    exit 1
fi
"$venv_root/bin/python" -c \
    'from rknnlite.api import RKNNLite; print("RKNN Lite import OK")'
"$venv_root/bin/python" -m pip check
"$venv_root/bin/python" -m pip freeze > "$project_root/requirements-installed.txt"

echo "[环境完成] $venv_root"
