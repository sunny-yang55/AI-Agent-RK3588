from agent.agent import BaseAgent
from agent.memory import MemoryManager


def create_agent():

    memory = MemoryManager()

    agent = BaseAgent(memory=memory)

    # =====================
    # Agent response tool
    # =====================

    from tools.common.tool_metadata import ToolMetadata

    agent.tools.register(
        ToolMetadata(
            name="generate_response",
            description="Generate final response",
            handler=lambda **kw: agent._responder.generate(
                goal=kw.get("goal", ""),
                tool_results=kw.get("tool_results", ""),
                memory=memory,
            ),
        )
    )

    return agent
