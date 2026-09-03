"""Process-isolated camera preview service for Qt-safe GUI ownership."""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

from .camera import CameraConfig, CameraFormat, CameraSource, OpenCVCameraSource
from .service import OpenCVVisionWindow, VisionDisplay
from .session import VisionSession, VisionSessionState


def _vision_process_main(
    connection: Connection,
    config: CameraConfig,
    detection_enabled: bool = False,
    detection_interval: int = 5,
    camera_factory: Callable[[], CameraSource] | None = None,
    display_factory: Callable[[], VisionDisplay] | None = None,
    detector_factory: Callable[[], Any] | None = None,
) -> None:
    """Run camera and HighGUI together on the child process main thread."""
    camera = None
    display = None
    detector = None
    latest_detections = []
    failed = False
    try:
        camera = camera_factory() if camera_factory else OpenCVCameraSource(config)
        display = display_factory() if display_factory else OpenCVVisionWindow()
        if detection_enabled:
            if detector_factory is not None:
                detector = detector_factory()
            else:
                from .yolov5_rknn import RKNNYOLOv5Detector

                detector = RKNNYOLOv5Detector()
        camera_format = camera.open()
        connection.send(
            {
                "event": "active",
                "format": {
                    "width": camera_format.width,
                    "height": camera_format.height,
                    "fps": camera_format.fps,
                    "fourcc": camera_format.fourcc,
                    "backend": camera_format.backend,
                },
            }
        )
        while True:
            frame = camera.read()
            display_image = frame.image
            if detector is not None and (
                frame.sequence == 1 or frame.sequence % max(1, detection_interval) == 0
            ):
                latest_detections = detector.detect(frame.image)
            stop_requested = False
            while connection.poll():
                message = connection.recv()
                command = message.get("command")
                if command == "stop":
                    stop_requested = True
                    break
                if command == "describe":
                    from .yolov5_rknn import summarize_detections

                    connection.send(
                        {
                            "event": "description",
                            "summary": summarize_detections(latest_detections),
                            "detections": [item.__dict__ for item in latest_detections],
                        }
                    )
            if detector is not None:
                from .yolov5_rknn import draw_detections

                display_image = draw_detections(frame.image, latest_detections)
            if stop_requested or not display.show(display_image, sequence=frame.sequence):
                break
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        failed = True
        try:
            connection.send({"event": "error", "error": str(exc)})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if display is not None:
            try:
                display.close()
            except Exception as exc:
                failed = True
                try:
                    connection.send({"event": "error", "error": str(exc)})
                except (EOFError, BrokenPipeError, OSError):
                    pass
        if camera is not None:
            try:
                camera.close()
            except Exception as exc:
                failed = True
                try:
                    connection.send({"event": "error", "error": str(exc)})
                except (EOFError, BrokenPipeError, OSError):
                    pass
        if detector is not None:
            try:
                detector.close()
            except Exception as exc:
                failed = True
                try:
                    connection.send({"event": "error", "error": str(exc)})
                except (EOFError, BrokenPipeError, OSError):
                    pass
        if not failed:
            try:
                connection.send({"event": "closed"})
            except (EOFError, BrokenPipeError, OSError):
                pass
        connection.close()


class ProcessVisionService:
    """Control a Qt-safe visual child process from the voice runtime."""

    def __init__(
        self,
        config: CameraConfig | None = None,
        *,
        session: VisionSession | None = None,
        process_context=None,
        detection_enabled: bool = True,
        detection_interval: int = 5,
    ) -> None:
        self.config = config or CameraConfig()
        self.session = session or VisionSession()
        self._context = process_context or multiprocessing.get_context("spawn")
        self.detection_enabled = detection_enabled
        self.detection_interval = max(1, detection_interval)
        self._process = None
        self._connection = None
        self._camera_format: CameraFormat | None = None

    @property
    def camera_format(self) -> CameraFormat | None:
        self._refresh()
        return self._camera_format

    @property
    def is_running(self) -> bool:
        self._refresh()
        return bool(self._process and self._process.is_alive())

    def start(self, *, timeout: float = 8.0) -> bool:
        if not self.session.request_start():
            return self.session.state is VisionSessionState.ACTIVE
        parent, child = self._context.Pipe(duplex=True)
        self._connection = parent
        self._process = self._context.Process(
            target=_vision_process_main,
            args=(
                child,
                self.config,
                self.detection_enabled,
                self.detection_interval,
            ),
            name="vision-preview",
            daemon=True,
        )
        self._process.start()
        child.close()
        if not parent.poll(max(0.0, timeout)):
            self.session.mark_error("camera process startup timed out")
            self._cleanup_failed_start()
            return False
        self._handle_message(parent.recv())
        started = self.session.state is VisionSessionState.ACTIVE
        if not started:
            self._cleanup_failed_start()
        return started

    def stop(self, *, timeout: float = 5.0) -> bool:
        self.session.request_stop()
        process = self._process
        connection = self._connection
        if process and process.is_alive() and connection is not None:
            try:
                connection.send({"command": "stop"})
            except (EOFError, BrokenPipeError, OSError):
                pass
            process.join(max(0.0, timeout))
        self._refresh()
        if process and process.is_alive():
            process.terminate()
            process.join(1.0)
            self.session.mark_error("camera process shutdown timed out")
            return False
        if self.session.state is not VisionSessionState.ERROR:
            self.session.mark_closed()
        if connection is not None:
            connection.close()
        return self.session.state is VisionSessionState.CLOSED

    def describe(self, *, timeout: float = 3.0) -> str:
        """Request the latest stable detection summary from the child."""
        if not self.is_running or self._connection is None:
            raise RuntimeError("vision service is not running")
        self._connection.send({"command": "describe"})
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if not self._connection.poll(remaining):
                break
            message = self._connection.recv()
            if message.get("event") == "description":
                return str(message["summary"])
            self._handle_message(message)
        raise TimeoutError("vision description timed out")

    def _refresh(self) -> None:
        connection = self._connection
        if connection is not None:
            try:
                while connection.poll():
                    self._handle_message(connection.recv())
            except (EOFError, OSError):
                pass
        if (
            self._process is not None
            and not self._process.is_alive()
            and self.session.state in {
                VisionSessionState.ACTIVE,
                VisionSessionState.STOPPING,
            }
        ):
            self.session.mark_closed()

    def _cleanup_failed_start(self) -> None:
        """Reap a failed child without overwriting the reported error."""
        process = self._process
        if process is not None:
            process.join(1.0)
            if process.is_alive():
                process.terminate()
                process.join(1.0)
        if self._connection is not None:
            self._connection.close()

    def _handle_message(self, message: dict) -> None:
        event = message.get("event")
        if event == "active":
            self._camera_format = CameraFormat(**message["format"])
            self.session.mark_active()
        elif event == "error":
            self.session.mark_error(message.get("error", "unknown camera process error"))
        elif event == "closed" and self.session.state is not VisionSessionState.ERROR:
            self.session.mark_closed()
