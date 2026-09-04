"""Camera acquisition abstractions for the RK3588 vision pipeline."""

from __future__ import annotations

import glob
import importlib
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


DEFAULT_CAMERA_DEVICE = "auto"


def discover_camera_devices() -> list[str]:
    """Return stable USB capture candidates, excluding RKISP/HDMI nodes."""
    candidates = []
    override = os.getenv("VISION_CAMERA_DEVICE", "").strip()
    if override:
        candidates.append(override)
    candidates.extend(sorted(glob.glob("/dev/v4l/by-id/*-video-index0")))
    for sysfs_path in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            if "/usb" not in os.path.realpath(sysfs_path + "/device"):
                continue
        except OSError:
            continue
        candidates.append("/dev/" + os.path.basename(sysfs_path))
    unique = []
    targets = set()
    for candidate in candidates:
        target = os.path.realpath(candidate)
        if target not in targets:
            targets.add(target)
            unique.append(candidate)
    return unique


class CameraError(RuntimeError):
    """Base class for camera failures."""


class CameraOpenError(CameraError):
    """Raised when a camera cannot be opened."""


class CameraNotOpenError(CameraError):
    """Raised when reading from a closed camera."""


class CameraReadError(CameraError):
    """Raised when the camera fails to return a frame."""


@dataclass(frozen=True)
class CameraConfig:
    device: str = DEFAULT_CAMERA_DEVICE
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    fourcc: str = "MJPG"
    buffer_size: int | None = 1

    def __post_init__(self) -> None:
        if not self.device:
            raise ValueError("camera device must not be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        if self.fps <= 0:
            raise ValueError("camera fps must be positive")
        if len(self.fourcc) != 4:
            raise ValueError("camera fourcc must contain exactly four characters")
        if self.buffer_size is not None and self.buffer_size <= 0:
            raise ValueError("camera buffer_size must be positive or None")


@dataclass(frozen=True)
class CameraFormat:
    width: int
    height: int
    fps: float
    fourcc: str
    backend: str


@dataclass(frozen=True)
class CameraFrame:
    sequence: int
    captured_at: float
    monotonic_at: float
    image: Any


class CameraSource(ABC):
    """Synchronous frame source independent of a detector or GUI."""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Return whether the source is ready to read frames."""

    @property
    @abstractmethod
    def negotiated_format(self) -> CameraFormat | None:
        """Return the driver-negotiated format after opening."""

    @abstractmethod
    def open(self) -> CameraFormat:
        """Open the source. Repeated calls must be safe."""

    @abstractmethod
    def read(self) -> CameraFrame:
        """Read one frame or raise CameraReadError."""

    @abstractmethod
    def close(self) -> None:
        """Release the source. Repeated calls must be safe."""

    def __enter__(self) -> CameraSource:
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class OpenCVCameraSource(CameraSource):
    """V4L2 camera source implemented with OpenCV."""

    def __init__(
        self,
        config: CameraConfig | None = None,
        *,
        cv_module: Any | None = None,
        capture_factory: Callable[[str, int], Any] | None = None,
    ) -> None:
        self.config = config or CameraConfig()
        self._cv = cv_module
        self._capture_factory = capture_factory
        self._capture: Any | None = None
        self._negotiated_format: CameraFormat | None = None
        self._sequence = 0
        self._active_device: str | None = None

    @property
    def is_open(self) -> bool:
        return self._capture is not None and bool(self._capture.isOpened())

    @property
    def negotiated_format(self) -> CameraFormat | None:
        return self._negotiated_format

    @property
    def active_device(self) -> str | None:
        return self._active_device

    def _get_cv(self) -> Any:
        if self._cv is None:
            try:
                self._cv = importlib.import_module("cv2")
            except ModuleNotFoundError as exc:
                raise CameraOpenError(
                    "OpenCV is not installed in the active Python environment"
                ) from exc
        return self._cv

    def _set_property(self, prop: int, value: float) -> bool:
        return bool(self._capture.set(prop, value))

    @staticmethod
    def _decode_fourcc(value: float) -> str:
        integer = int(value)
        return "".join(chr((integer >> (8 * index)) & 0xFF) for index in range(4))

    def open(self) -> CameraFormat:
        if self.is_open and self._negotiated_format is not None:
            return self._negotiated_format

        self.close()
        cv = self._get_cv()
        factory = self._capture_factory or cv.VideoCapture
        candidates = (
            discover_camera_devices()
            if self.config.device == "auto"
            else [self.config.device]
        )
        if not candidates:
            raise CameraOpenError("no USB video capture device was discovered")
        capture = None
        for device in candidates:
            candidate = factory(device, cv.CAP_V4L2)
            if candidate.isOpened():
                capture = candidate
                self._active_device = device
                break
            candidate.release()
        if capture is None:
            raise CameraOpenError(
                "failed to open camera candidates: " + ", ".join(candidates)
            )
        self._capture = capture

        try:
            self._set_property(
                cv.CAP_PROP_FOURCC,
                cv.VideoWriter_fourcc(*self.config.fourcc),
            )
            self._set_property(cv.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._set_property(cv.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._set_property(cv.CAP_PROP_FPS, self.config.fps)
            if self.config.buffer_size is not None:
                self._set_property(cv.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

            backend = "V4L2"
            get_backend_name = getattr(capture, "getBackendName", None)
            if callable(get_backend_name):
                backend = str(get_backend_name())

            self._negotiated_format = CameraFormat(
                width=int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
                height=int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)),
                fps=float(capture.get(cv.CAP_PROP_FPS)),
                fourcc=self._decode_fourcc(capture.get(cv.CAP_PROP_FOURCC)),
                backend=backend,
            )
            self._sequence = 0
            return self._negotiated_format
        except Exception:
            self.close()
            raise

    def read(self) -> CameraFrame:
        if not self.is_open:
            raise CameraNotOpenError("camera is not open")

        ok, image = self._capture.read()
        if not ok or image is None:
            device = self._active_device or self.config.device
            raise CameraReadError(f"failed to read frame from {device}")

        self._sequence += 1
        return CameraFrame(
            sequence=self._sequence,
            captured_at=time.time(),
            monotonic_at=time.monotonic(),
            image=image,
        )

    def close(self) -> None:
        capture = self._capture
        self._capture = None
        self._negotiated_format = None
        self._active_device = None
        if capture is not None:
            capture.release()
