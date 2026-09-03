"""Background camera and preview-window lifecycle for a vision session."""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable
from typing import Any, Protocol

from .camera import CameraFormat, CameraSource, OpenCVCameraSource
from .session import VisionSession, VisionSessionState


class VisionDisplay(Protocol):
    def show(self, image: Any, *, sequence: int) -> bool:
        """Show one frame and return false when the user requests closure."""

    def close(self) -> None:
        """Release display resources."""


class OpenCVVisionWindow:
    """Minimal OpenCV preview window; OpenCV is imported lazily."""

    def __init__(self, title: str = "XiaoAn Vision") -> None:
        self.title = title
        self._cv = importlib.import_module("cv2")
        self._created = False

    def show(self, image: Any, *, sequence: int) -> bool:
        cv = self._cv
        if not self._created:
            cv.namedWindow(self.title, cv.WINDOW_NORMAL)
            self._created = True
        frame = image.copy() if hasattr(image, "copy") else image
        if hasattr(frame, "shape") and len(frame.shape) >= 2:
            overlay = f"Frame {sequence} | {frame.shape[1]}x{frame.shape[0]}"
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
        cv.imshow(self.title, frame)
        key = cv.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return False
        return cv.getWindowProperty(self.title, cv.WND_PROP_VISIBLE) >= 1

    def close(self) -> None:
        if not self._created:
            return
        try:
            self._cv.destroyWindow(self.title)
            self._cv.waitKey(1)
        except Exception:
            # Clicking the native window close button may destroy it before
            # the worker reaches cleanup. Resource release remains idempotent.
            pass
        finally:
            self._created = False


class VisionService:
    """Own a camera and preview window on one stoppable worker thread."""

    def __init__(
        self,
        *,
        camera_factory: Callable[[], CameraSource] = OpenCVCameraSource,
        display_factory: Callable[[], VisionDisplay] = OpenCVVisionWindow,
        session: VisionSession | None = None,
    ) -> None:
        self.session = session or VisionSession()
        self._camera_factory = camera_factory
        self._display_factory = display_factory
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._camera_format: CameraFormat | None = None
        self._lock = threading.RLock()

    @property
    def camera_format(self) -> CameraFormat | None:
        with self._lock:
            return self._camera_format

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

    def start(self, *, timeout: float = 5.0) -> bool:
        """Start capture and wait until the camera opens or fails."""
        with self._lock:
            if not self.session.request_start():
                return self.session.state is VisionSessionState.ACTIVE
            self._stop_event.clear()
            self._ready_event.clear()
            self._camera_format = None
            self._thread = threading.Thread(
                target=self._run,
                name="vision-camera",
                daemon=True,
            )
            self._thread.start()
        if not self._ready_event.wait(timeout=max(0.0, timeout)):
            self.session.mark_error("camera startup timed out")
            self._stop_event.set()
            return False
        return self.session.state is VisionSessionState.ACTIVE

    def stop(self, *, timeout: float = 5.0) -> bool:
        """Request closure and wait for camera/window resource release."""
        self.session.request_stop()
        self._stop_event.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))
        if thread and thread.is_alive():
            self.session.mark_error("camera shutdown timed out")
            return False
        if self.session.state is not VisionSessionState.ERROR:
            self.session.mark_closed()
        return True

    def _run(self) -> None:
        camera: CameraSource | None = None
        display: VisionDisplay | None = None
        failed = False
        try:
            camera = self._camera_factory()
            display = self._display_factory()
            camera_format = camera.open()
            with self._lock:
                self._camera_format = camera_format
            self.session.mark_active()
            self._ready_event.set()
            while not self._stop_event.is_set():
                frame = camera.read()
                if not display.show(frame.image, sequence=frame.sequence):
                    break
        except Exception as exc:
            failed = True
            self.session.mark_error(exc)
            self._ready_event.set()
        finally:
            if display is not None:
                try:
                    display.close()
                except Exception as exc:
                    if not failed:
                        failed = True
                        self.session.mark_error(exc)
            if camera is not None:
                try:
                    camera.close()
                except Exception as exc:
                    if not failed:
                        failed = True
                        self.session.mark_error(exc)
            if not failed and self.session.state is not VisionSessionState.ERROR:
                self.session.mark_closed()
            self._ready_event.set()
