from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Generic, TypeVar

from agent.memory_compressor import MemoryCompressor

T = TypeVar("T")


def sanitize_memory_data(obj):
    """
    Memory数据保护层

    防止保存:
    - numpy.ndarray
    - PIL Image
    - tensor
    - 模型输出对象
    - 自定义class对象
    """

    import numpy as np

    # numpy图片矩阵
    if isinstance(obj, np.ndarray):

        return {
            "type": "numpy.ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }

    # dict
    if isinstance(obj, dict):

        return {k: sanitize_memory_data(v) for k, v in obj.items()}

    # list
    if isinstance(obj, list):

        return [sanitize_memory_data(v) for v in obj]

    # tuple
    if isinstance(obj, tuple):

        return [sanitize_memory_data(v) for v in obj]

    # 基础类型
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # dataclass / 自定义对象
    if hasattr(obj, "__dict__"):

        return sanitize_memory_data(obj.__dict__)

    # 最后保险
    return str(obj)


class MemoryType(Enum):
    WORKING = auto()
    SHORT_TERM = auto()
    LONG_TERM = auto()
    EPISODIC = auto()


@dataclass
class MemoryEntry:
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    ttl: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


class BaseMemory(ABC, Generic[T]):
    def __init__(self, capacity: int | None = None) -> None:
        self._capacity = capacity

    @abstractmethod
    def store(
        self, key: str, value: T, ttl: float | None = None, **metadata: Any
    ) -> None: ...

    @abstractmethod
    def retrieve(self, key: str) -> T | None: ...

    @abstractmethod
    def forget(self, key: str) -> bool: ...

    @abstractmethod
    def contains(self, key: str) -> bool: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def snapshot(self) -> dict[str, T]: ...

    @property
    def capacity(self) -> int | None:
        return self._capacity

    def __len__(self) -> int:
        raise NotImplementedError

    def __contains__(self, key: str) -> bool:
        return self.contains(key)


class WorkingMemory(BaseMemory[Any]):
    """Volatile scratchpad for the current task context."""

    def __init__(self, capacity: int = 256) -> None:
        super().__init__(capacity=capacity)
        self._store: OrderedDict[str, MemoryEntry] = OrderedDict()

    def store(
        self, key: str, value: Any, ttl: float | None = None, **metadata: Any
    ) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        elif self._capacity is not None and len(self._store) >= self._capacity:
            self._store.popitem(last=False)
        self._store[key] = MemoryEntry(key=key, value=value, ttl=ttl, metadata=metadata)

    def retrieve(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return entry.value

    def forget(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    def contains(self, key: str) -> bool:
        entry = self._store.get(key)
        if entry is None:
            return False
        if entry.is_expired():
            del self._store[key]
            return False
        return True

    def clear(self) -> None:
        self._store.clear()

    def snapshot(self) -> dict[str, Any]:
        return {k: e.value for k, e in self._store.items() if not e.is_expired()}

    def __len__(self) -> int:
        return len(self._store)


class ShortTermMemory(BaseMemory[Any]):
    """Bounded FIFO buffer for recent conversation / action history."""

    def __init__(self, capacity: int = 1024) -> None:
        super().__init__(capacity=capacity)
        self._buffer: deque[MemoryEntry] = deque(maxlen=capacity)

    def store(
        self, key: str, value: Any, ttl: float | None = None, **metadata: Any
    ) -> None:
        self._buffer.append(
            MemoryEntry(key=key, value=value, ttl=ttl, metadata=metadata)
        )

    def retrieve(self, key: str) -> Any | None:
        for entry in reversed(self._buffer):
            if entry.key == key and not entry.is_expired():
                return entry.value
        return None

    def forget(self, key: str) -> bool:
        for i, entry in enumerate(self._buffer):
            if entry.key == key:
                del self._buffer[i]
                return True
        return False

    def contains(self, key: str) -> bool:
        return any(
            entry.key == key and not entry.is_expired() for entry in self._buffer
        )

    def clear(self) -> None:
        self._buffer.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            entry.key: entry.value for entry in self._buffer if not entry.is_expired()
        }

    def recent(self, n: int = 10) -> list[MemoryEntry]:
        items = list(self._buffer)
        return items[-n:]

    def __len__(self) -> int:
        return len(self._buffer)


class LongTermMemory(BaseMemory[Any]):
    """Persistent, file-backed memory for stable knowledge."""

    def __init__(
        self, storage_path: Path | str | None = None, capacity: int | None = None
    ) -> None:
        super().__init__(capacity=capacity)
        self._storage_path = (
            Path(storage_path) if storage_path else Path("logs/long_term_memory.json")
        )
        self._store: dict[str, MemoryEntry] = {}
        self._load()

    def _load(self) -> None:
        if self._storage_path.exists():
            try:
                raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
                for k, v in raw.items():
                    self._store[k] = MemoryEntry(
                        key=k,
                        value=v["value"],
                        timestamp=v.get("timestamp", time.time()),
                        ttl=v.get("ttl"),
                        metadata=v.get("metadata", {}),
                    )
            except (json.JSONDecodeError, KeyError):
                pass

    def _persist(self) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            k: {
                "value": e.value,
                "timestamp": e.timestamp,
                "ttl": e.ttl,
                "metadata": e.metadata,
            }
            for k, e in self._store.items()
        }
        self._storage_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def store(
        self, key: str, value: Any, ttl: float | None = None, **metadata: Any
    ) -> None:
        self._store[key] = MemoryEntry(key=key, value=value, ttl=ttl, metadata=metadata)
        self._persist()

    def retrieve(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            del self._store[key]
            self._persist()
            return None
        return entry.value

    def forget(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            self._persist()
            return True
        return False

    def contains(self, key: str) -> bool:
        entry = self._store.get(key)
        if entry is None:
            return False
        if entry.is_expired():
            del self._store[key]
            self._persist()
            return False
        return True

    def clear(self) -> None:
        self._store.clear()
        self._persist()

    def snapshot(self) -> dict[str, Any]:
        return {k: e.value for k, e in self._store.items() if not e.is_expired()}

    def __len__(self) -> int:
        return len(self._store)


class EpisodicMemory(BaseMemory[dict[str, Any]]):
    """Stores full episode traces (plan -> execution -> outcome)."""

    def __init__(
        self, storage_path: Path | str | None = None, capacity: int | None = None
    ) -> None:
        super().__init__(capacity=capacity)
        self._storage_path = (
            Path(storage_path) if storage_path else Path("logs/episodic_memory.json")
        )
        self._episodes: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self._storage_path.exists():
            try:
                self._episodes = json.loads(
                    self._storage_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                self._episodes = []

    def _persist(self) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(
            json.dumps(self._episodes, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def store(
        self, key: str, value: dict[str, Any], ttl: float | None = None, **metadata: Any
    ) -> None:
        episode = {
            "id": key,
            "timestamp": time.time(),
            "data": value,
            "metadata": metadata,
        }
        self._episodes.append(sanitize_memory_data(episode))
        if self._capacity is not None and len(self._episodes) > self._capacity:
            self._episodes = self._episodes[-self._capacity :]
        self._persist()

    def retrieve(self, key: str) -> dict[str, Any] | None:
        for ep in reversed(self._episodes):
            if ep["id"] == key:
                return ep["data"]
        return None

    def forget(self, key: str) -> bool:
        before = len(self._episodes)
        self._episodes = [ep for ep in self._episodes if ep["id"] != key]
        if len(self._episodes) != before:
            self._persist()
            return True
        return False

    def contains(self, key: str) -> bool:
        return any(ep["id"] == key for ep in self._episodes)

    def clear(self) -> None:
        self._episodes.clear()
        self._persist()

    def snapshot(self) -> dict[str, Any]:
        return {ep["id"]: ep["data"] for ep in self._episodes}

    def recent(self, n: int = 10) -> list[dict[str, Any]]:
        return self._episodes[-n:]

    def search(self, query: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for ep in self._episodes:
            if query.lower() in str(ep).lower():
                results.append(ep)
        return results

    def __len__(self) -> int:
        return len(self._episodes)


class MemoryManager:
    """Unified interface over all memory subsystems."""

    def __init__(
        self,
        working_capacity: int = 256,
        short_term_capacity: int = 1024,
        long_term_path: Path | str | None = None,
        episodic_path: Path | str | None = None,
    ) -> None:
        self.working = WorkingMemory(capacity=working_capacity)
        self.short_term = ShortTermMemory(capacity=short_term_capacity)
        self.long_term = LongTermMemory(storage_path=long_term_path)
        self.episodic = EpisodicMemory(storage_path=episodic_path)
        self.compressor = MemoryCompressor()

    def contextualize(self, query: str, scope: str = "all") -> dict[str, Any]:
        context: dict[str, Any] = {}
        if scope in ("all", "working"):
            context["working"] = self.working.snapshot()
        if scope in ("all", "recent"):
            context["recent"] = [e.value for e in self.short_term.recent(n=20)]
        if scope in ("all", "episodes"):
            context["episodes"] = self.episodic.recent(n=5)
        return context

    def compress_memory(self):

        episodes = self.episodic.recent(n=500)

        if not self.compressor.should_compress(episodes):

            logger.debug("[Memory] Compression skipped")

            return None

        summary = self.compressor.compress(episodes)

        # 保存长期摘要

        timestamp = int(time.time())

        self.long_term.store(f"compressed_summary_{int(time.time())}", summary)

        # 清理旧episodic

        keep = 50

        if len(self.episodic) > keep:

            self.episodic._episodes = self.episodic._episodes[-keep:]

            self.episodic._persist()

        logger.debug("[Memory] Compression finished")

        return summary

    def clear_all(self) -> None:
        self.working.clear()
        self.short_term.clear()
        self.long_term.clear()
        self.episodic.clear()
