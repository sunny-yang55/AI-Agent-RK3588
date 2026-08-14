import logging

logger = logging.getLogger(__name__)


def register_builtin_tools(agent, memory):

    agent.register_tool(
        "generate_response",
        lambda **kw: agent._responder.generate(
            goal=kw.get("goal", ""),
            tool_results=kw.get("tool_results", ""),
            memory=memory,
        ),
    )

    agent.register_tool(
        "identify_requirements", lambda **kw: {"identified": True, "params": kw}
    )

    agent.register_tool(
        "propose_approach", lambda **kw: {"approach": "default", "params": kw}
    )

    agent.register_tool(
        "handle_feedback", lambda **kw: {"feedback_handled": True, "params": kw}
    )

    logger.info("Builtin agent tools registered")
