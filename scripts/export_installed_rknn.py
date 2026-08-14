#!/usr/bin/env python3
"""Export an installed RKNN Lite distribution for same-platform deployment."""

from __future__ import annotations

import importlib.metadata
import shutil
import sys
import sysconfig
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: export_installed_rknn.py DESTINATION", file=sys.stderr)
        return 2
    destination = Path(sys.argv[1]).resolve()
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    distribution = importlib.metadata.distribution("rknn-toolkit-lite2")
    files = distribution.files or []
    copied = 0
    destination.mkdir(parents=True, exist_ok=False)
    for entry in files:
        source = Path(distribution.locate_file(entry)).resolve()
        try:
            relative = source.relative_to(purelib)
        except ValueError:
            continue
        if not source.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    if copied == 0:
        raise RuntimeError("installed rknn-toolkit-lite2 contained no exportable files")
    print(f"exported {copied} RKNN Lite files from {purelib} to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
