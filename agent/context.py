import json
from dataclasses import asdict, is_dataclass
from enum import Enum


def sanitize_memory_data(obj):
    """
    Memory安全转换

    目标:
    任何对象 -> JSON安全结构
    禁止:
    - numpy array
    - 图片矩阵
    - 自定义对象字符串爆炸
    """

    # None
    if obj is None:
        return None

    # 基础类型
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Enum
    if isinstance(obj, Enum):
        return obj.name

    # dataclass
    if is_dataclass(obj):
        return sanitize_memory_data(asdict(obj))

    # dict
    if isinstance(obj, dict):

        return {str(k): sanitize_memory_data(v) for k, v in obj.items()}

    # list / tuple
    if isinstance(obj, (list, tuple)):

        return [sanitize_memory_data(v) for v in obj]

    # numpy
    try:

        import numpy as np

        if isinstance(obj, np.ndarray):

            return {
                "type": "ndarray",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
            }

    except ImportError:
        pass

    # 普通对象
    if hasattr(obj, "__dict__"):

        return {k: sanitize_memory_data(v) for k, v in obj.__dict__.items()}

    # 最后兜底

    return str(obj)


from enum import Enum, auto


class ContextState(Enum):
    IDLE = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()


class AgentContext:
    """
    Agent运行上下文
    """

    def __init__(self):

        import uuid

        from .agent import AgentState

        self.session_id = str(uuid.uuid4())

        self.state = AgentState.IDLE

        self.goal = ""

        self.current_plan = None

        self.history = []

        self.last_result = None

    def update(self, result):

        # 重点:
        # 不允许直接str(result)

        record = sanitize_memory_data(result)

        self.last_result = record

        self.history.append(record)

    def get_last_result(self):

        return self.last_result
