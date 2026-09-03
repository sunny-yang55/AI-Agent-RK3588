"""Camera acquisition abstractions for the RK3588 vision pipeline."""

from __future__ import annotations

import importlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


DEFAULT_CAMERA_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-Ruision_USB_FHD_Camera_20220623-c6ec643-video-index0"
)


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

    @property
    def is_open(self) -> bool:
        return self._capture is not None and bool(self._capture.isOpened())

    @property
    def negotiated_format(self) -> CameraFormat | None:
        return self._negotiated_format

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
        capture = factory(self.config.device, cv.CAP_V4L2)
        self._capture = capture

        if not capture.isOpened():
            self.close()
            raise CameraOpenError(f"failed to open camera: {self.config.device}")

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
            raise CameraReadError(f"failed to read frame from {self.config.device}")

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
        if capture is not None:
            capture.release()
