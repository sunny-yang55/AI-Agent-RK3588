from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict


class CapabilityStatus(Enum):

    CREATED = "created"

    INITIALIZED = "initialized"

    RUNNING = "running"

    ERROR = "error"

    STOPPED = "stopped"


class Capability(ABC):
    """
    Capability生命周期基类
    """

    name: str = "base"

    description: str = ""

    def __init__(self):

        self.status = CapabilityStatus.CREATED

        self.metadata = {}

    def initialize(self) -> bool:
        """
        初始化资源

        例如：
        - 打开摄像头
        - 加载模型
        - 建立ROS连接
        """

        self.status = CapabilityStatus.INITIALIZED

        return True

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        执行能力
        """

        pass

    def health_check(self) -> Dict[str, Any]:

        return {
            "name": self.name,
            "status": self.status.value,
            "healthy": self.status != CapabilityStatus.ERROR,
        }

    def shutdown(self):
        """
        释放资源
        """

        self.status = CapabilityStatus.STOPPED

    def info(self):

        return {
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
        }
