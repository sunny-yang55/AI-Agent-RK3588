"""
AI-Agent — Core Agent Framework.

A modular, extensible agent architecture providing planning, execution,
and memory management primitives for building AI agents.
"""

from .agent import Agent, AgentContext, AgentState, BaseAgent
from .executor import (
    BaseExecutor,
    DefaultExecutor,
    ExecutionMode,
    ExecutionResult,
    Executor,
    ToolRegistry,
)
from .memory import (
    BaseMemory,
    EpisodicMemory,
    LongTermMemory,
    MemoryEntry,
    MemoryManager,
    MemoryType,
    ShortTermMemory,
    WorkingMemory,
)
from .planner import (
    BasePlanner,
    Plan,
    Planner,
    PlanStep,
    RuleBasedPlanner,
    StepStatus,
)

from .llm_planner import LLMPlanner

__version__ = "0.7.0"
__all__ = [
    # Agent
    "Agent",
    "AgentContext",
    "AgentState",
    "BaseAgent",
    # Planner
    "LLMPlanner",
    "BasePlanner",
    "Plan",
    "Planner",
    "PlanStep",
    "RuleBasedPlanner",
    "StepStatus",
    # Executor
    "BaseExecutor",
    "DefaultExecutor",
    "ExecutionMode",
    "ExecutionResult",
    "Executor",
    "ToolRegistry",
    # Memory
    "BaseMemory",
    "EpisodicMemory",
    "LongTermMemory",
    "MemoryEntry",
    "MemoryManager",
    "MemoryType",
    "ShortTermMemory",
    "WorkingMemory",
]
