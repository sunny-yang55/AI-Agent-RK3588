#!/usr/bin/env python3
"""RK3588 UVC camera diagnostic for the vision M1 milestone."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WINDOW_NAME = "XiaoAn Vision - Camera Diagnostic"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DiagnosticError(RuntimeError):
    """Raised for diagnostic-only failures outside CameraSource."""


@dataclass
class DiagnosticStats:
    started_at: float
    frames: int = 0
    read_failures: int = 0

    def average_fps(self, now: float) -> float:
        elapsed = max(0.0, now - self.started_at)
        if elapsed == 0:
            return 0.0
        return self.frames / elapsed


def duration_reached(duration: float, started_at: float, now: float) -> bool:
    return duration > 0 and now - started_at >= duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default=(
            "/dev/v4l/by-id/"
            "usb-Ruision_USB_FHD_Camera_20220623-c6ec643-video-index0"
        ),
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="stop automatically after this many seconds; 0 means unlimited",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="capture frames without creating a GUI window",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=10.0,
        help="seconds between console progress reports",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=Path("reports/vision-camera"),
    )
    return parser.parse_args()


def load_cv2() -> Any:
    try:
        return importlib.import_module("cv2")
    except ModuleNotFoundError as exc:
        raise DiagnosticError("OpenCV is required for the diagnostic window") from exc


def load_camera_api() -> Any:
    """Load board runtime dependencies only when capture actually starts."""
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        return importlib.import_module("tools.vision.camera")
    except ModuleNotFoundError as exc:
        raise DiagnosticError(f"failed to import camera runtime: {exc}") from exc


def save_screenshot(frame: Any, directory: Path, cv: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    filename = directory / time.strftime("camera-%Y%m%d-%H%M%S.jpg")
    if not cv.imwrite(str(filename), frame):
        raise DiagnosticError(f"failed to save screenshot: {filename}")
    print(f"[Camera] Screenshot saved: {filename.resolve()}")


def validate_args(args: argparse.Namespace) -> None:
    if args.duration < 0:
        raise ValueError("--duration must be zero or positive")
    if args.report_interval <= 0:
        raise ValueError("--report-interval must be positive")
    if args.max_consecutive_failures <= 0:
        raise ValueError("--max-consecutive-failures must be positive")


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    camera_api = load_camera_api()
    config = camera_api.CameraConfig(
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
        buffer_size=1,
    )
    camera = camera_api.OpenCVCameraSource(config)
    cv = None if args.no_window else load_cv2()
    window_created = False
    rolling_times: deque[float] = deque(maxlen=60)
    consecutive_failures = 0
    started_at = time.monotonic()
    stats = DiagnosticStats(started_at=started_at)
    next_report_at = started_at + args.report_interval
    exit_reason = "completed"

    try:
        camera_format = camera.open()
        print(
            "[Camera] Negotiated: "
            f"{camera_format.width}x{camera_format.height}, "
            f"{camera_format.fps:.2f} FPS, "
            f"FOURCC={camera_format.fourcc!r}, "
            f"backend={camera_format.backend}"
        )

        if cv is not None:
            cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
            window_created = True
            print("[Camera] Controls: q/Esc=quit, s=save screenshot")
        else:
            print("[Camera] Headless capture mode enabled")

        while True:
            now = time.monotonic()
            if duration_reached(args.duration, started_at, now):
                exit_reason = "duration reached"
                break

            try:
                camera_frame = camera.read()
            except camera_api.CameraReadError as exc:
                stats.read_failures += 1
                consecutive_failures += 1
                print(
                    "[Camera] Frame read failed "
                    f"({consecutive_failures}/{args.max_consecutive_failures}): {exc}",
                    file=sys.stderr,
                )
                if consecutive_failures >= args.max_consecutive_failures:
                    raise
                time.sleep(0.02)
                continue

            consecutive_failures = 0
            stats.frames += 1
            frame = camera_frame.image
            now = camera_frame.monotonic_at
            rolling_times.append(now)

            rolling_fps = 0.0
            if len(rolling_times) >= 2:
                rolling_elapsed = rolling_times[-1] - rolling_times[0]
                if rolling_elapsed > 0:
                    rolling_fps = (len(rolling_times) - 1) / rolling_elapsed

            if now >= next_report_at:
                print(
                    f"[Camera] frames={stats.frames}, "
                    f"fps={stats.average_fps(now):.2f}, "
                    f"read_failures={stats.read_failures}"
                )
                next_report_at = now + args.report_interval

            if cv is None:
                continue

            overlay = (
                f"Frame {camera_frame.sequence} | "
                f"{frame.shape[1]}x{frame.shape[0]} | {rolling_fps:.1f} FPS"
            )
            cv.putText(
                frame,
                overlay,
                (20, 38),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv.LINE_AA,
            )
            cv.imshow(WINDOW_NAME, frame)

            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                exit_reason = "window key"
                break
            if key in (ord("s"), ord("S")):
                save_screenshot(frame, args.screenshot_dir, cv)

            if cv.getWindowProperty(WINDOW_NAME, cv.WND_PROP_VISIBLE) < 1:
                exit_reason = "window closed"
                break

    except KeyboardInterrupt:
        exit_reason = "Ctrl+C"
        print("\n[Camera] Ctrl+C received")
    except (camera_api.CameraError, DiagnosticError) as exc:
        exit_reason = f"camera error: {exc}"
        print(f"[Camera] Error: {exc}", file=sys.stderr)
        return 3
    finally:
        camera.close()
        if window_created and cv is not None:
            cv.destroyAllWindows()
            cv.waitKey(1)
        finished_at = time.monotonic()
        print(
            "[Camera] Summary: "
            f"reason={exit_reason}, "
            f"elapsed={finished_at - started_at:.2f}s, "
            f"frames={stats.frames}, "
            f"fps={stats.average_fps(finished_at):.2f}, "
            f"read_failures={stats.read_failures}"
        )
        print("[Camera] Capture released; windows destroyed")

    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except ValueError as exc:
        print(f"[Camera] Invalid argument: {exc}", file=sys.stderr)
        return 2
    except DiagnosticError as exc:
        print(f"[Camera] Error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
