#!/usr/bin/env python3
"""Report the minimum environment required by v1.1-rk3588-speech."""

from __future__ import annotations

import importlib
import platform
import sys


def probe(module_name: str, symbol: str | None = None) -> bool:
    try:
        module = importlib.import_module(module_name)
        if symbol is not None:
            getattr(module, symbol)
    except Exception as exc:  # noqa: BLE001 - environment probe must report all failures
        print(f"{module_name}{'.' + symbol if symbol else ''}: FAIL ({exc})")
        return False

    version = getattr(module, "__version__", None)
    suffix = f" (version={version})" if version else ""
    print(f"{module_name}{'.' + symbol if symbol else ''}: OK{suffix}")
    return True


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"system: {platform.system()}")
    print(f"machine: {platform.machine()}")

    numpy_ok = probe("numpy")
    toolkit_ok = probe("rknn.api", "RKNN")
    lite_ok = probe("rknnlite.api", "RKNNLite")

    if platform.machine().lower() not in {"aarch64", "arm64"}:
        print("warning: this is not an ARM64 RK3588 environment")

    if not toolkit_ok:
        print("missing: RKNN Toolkit2 2.3.2 is required for ONNX conversion")
    if not lite_ok:
        print("missing: RKNN Toolkit-Lite2 2.3.2 is required for board inference")

    return 0 if numpy_ok and toolkit_ok and lite_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
