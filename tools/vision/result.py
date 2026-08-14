from dataclasses import dataclass, field
from typing import Any


def safe_image_id(image):
    """
    防止图片数据进入memory和json
    """

    if image is None:
        return "none"

    if isinstance(image, str):
        return image

    if hasattr(image, "filename"):
        return image.filename

    return "image_object"


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class Detection:
    id: int
    label: str
    confidence: float
    bbox: BoundingBox

    track_id: int | None = None

    mask: Any = None

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:

    image_id: str

    timestamp: float

    detections: list[Detection]

    metadata: dict[str, Any] = field(default_factory=dict)
