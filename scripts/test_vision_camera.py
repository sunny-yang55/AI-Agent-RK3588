#!/usr/bin/env python3
"""RK3588 UVC camera diagnostic for the vision M1 milestone."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2


DEFAULT_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-Ruision_USB_FHD_Camera_20220623-c6ec643-video-index0"
)
WINDOW_NAME = "XiaoAn Vision - Camera Diagnostic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=Path("reports/vision-camera"),
    )
    return parser.parse_args()


def set_capture_property(
    capture: cv2.VideoCapture,
    prop: int,
    value: float,
    name: str,
) -> None:
    if not capture.set(prop, value):
        print(f"[Camera] Warning: driver rejected {name}={value}")


def print_negotiated_format(capture: cv2.VideoCapture) -> None:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    print(
        "[Camera] Negotiated: "
        f"{width}x{height}, {fps:.2f} FPS, FOURCC={fourcc!r}"
    )


def save_screenshot(frame, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    filename = directory / time.strftime("camera-%Y%m%d-%H%M%S.jpg")
    if not cv2.imwrite(str(filename), frame):
        raise RuntimeError(f"Failed to save screenshot: {filename}")
    print(f"[Camera] Screenshot saved: {filename.resolve()}")


def run(args: argparse.Namespace) -> int:
    device = Path(args.device)
    if not device.exists():
        print(f"[Camera] Device does not exist: {device}", file=sys.stderr)
        return 2

    capture = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
    if not capture.isOpened():
        print(f"[Camera] Failed to open: {device}", file=sys.stderr)
        return 3

    window_created = False
    frame_times: deque[float] = deque(maxlen=60)
    consecutive_failures = 0
    frame_count = 0

    try:
        set_capture_property(
            capture,
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*args.fourcc),
            "FOURCC",
        )
        set_capture_property(capture, cv2.CAP_PROP_FRAME_WIDTH, args.width, "width")
        set_capture_property(capture, cv2.CAP_PROP_FRAME_HEIGHT, args.height, "height")
        set_capture_property(capture, cv2.CAP_PROP_FPS, args.fps, "fps")
        set_capture_property(capture, cv2.CAP_PROP_BUFFERSIZE, 1, "buffer-size")
        print_negotiated_format(capture)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        window_created = True
        print("[Camera] Controls: q/Esc=quit, s=save screenshot")

        while True:
            ok, frame = capture.read()
            now = time.monotonic()
            if not ok or frame is None:
                consecutive_failures += 1
                print(
                    f"[Camera] Frame read failed ({consecutive_failures}/30)",
                    file=sys.stderr,
                )
                if consecutive_failures >= 30:
                    raise RuntimeError("Camera failed for 30 consecutive frames")
                time.sleep(0.02)
                continue

            consecutive_failures = 0
            frame_count += 1
            frame_times.append(now)
            measured_fps = 0.0
            if len(frame_times) >= 2:
                elapsed = frame_times[-1] - frame_times[0]
                if elapsed > 0:
                    measured_fps = (len(frame_times) - 1) / elapsed

            overlay = (
                f"Frame {frame_count} | {frame.shape[1]}x{frame.shape[0]} "
                f"| {measured_fps:.1f} FPS"
            )
            cv2.putText(
                frame,
                overlay,
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("[Camera] Quit requested from window")
                break
            if key in (ord("s"), ord("S")):
                save_screenshot(frame, args.screenshot_dir)

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("[Camera] Window close requested")
                break

    except KeyboardInterrupt:
        print("\n[Camera] Ctrl+C received")
    finally:
        capture.release()
        if window_created:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        print("[Camera] Capture released; windows destroyed")

    return 0


def main() -> int:
    args = parse_args()
    if len(args.fourcc) != 4:
        print("[Camera] --fourcc must contain exactly four characters", file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
