#!/usr/bin/env python3
"""Hardware smoke test for the background vision preview service."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_VENV = PROJECT_ROOT / "venv"
PROJECT_PYTHON = PROJECT_VENV / "bin/python"
if PROJECT_PYTHON.is_file() and Path(sys.prefix).resolve() != PROJECT_VENV.resolve():
    os.execv(
        str(PROJECT_PYTHON),
        [str(PROJECT_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="automatically close after this many seconds; 0 means unlimited",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration < 0:
        print("[VisionService] --duration must be zero or positive", file=sys.stderr)
        return 2

    from tools.vision.process_service import ProcessVisionService

    service = ProcessVisionService()
    if not service.start():
        print(f"[VisionService] Start failed: {service.session.error}", file=sys.stderr)
        return 3

    camera_format = service.camera_format
    print(f"[VisionService] Active: {camera_format}")
    print("[VisionService] Window controls: q/Esc/close; terminal: Ctrl+C")
    started_at = time.monotonic()
    try:
        while service.is_running:
            if args.duration and time.monotonic() - started_at >= args.duration:
                print("[VisionService] Duration reached; stopping")
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[VisionService] Ctrl+C received; stopping")
    finally:
        stopped = service.stop()

    print(
        f"[VisionService] Closed: state={service.session.state.value}, "
        f"released={stopped}"
    )
    return 0 if stopped and service.session.error is None else 3


if __name__ == "__main__":
    raise SystemExit(main())
