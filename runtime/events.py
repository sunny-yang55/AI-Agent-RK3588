import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class InputType(Enum):

    TEXT = "TEXT"

    AUDIO = "AUDIO"

    IMAGE = "IMAGE"

    SENSOR = "SENSOR"


@dataclass
class InputEvent:

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    type: InputType = InputType.TEXT

    data: object = None

    timestamp: float = field(default_factory=time.time)

    metadata: dict = field(default_factory=dict)
