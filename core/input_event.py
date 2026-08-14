import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


@dataclass
class AudioData:

    samples: Any = None

    sample_rate: int = 16000

    channels: int = 1

    text: str = ""


class InputType(Enum):

    TEXT = auto()

    AUDIO = auto()

    IMAGE = auto()

    VIDEO = auto()

    ROBOT_STATE = auto()


@dataclass
class AgentInput:

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    type: InputType = InputType.TEXT

    data: Any = None

    timestamp: float = field(default_factory=time.time)

    metadata: dict = field(default_factory=dict)

    def summary(self):

        return {
            "id": self.id,
            "type": self.type.name,
            "timestamp": self.timestamp,
            "data": self.data,
            "metadata": self.metadata,
        }
