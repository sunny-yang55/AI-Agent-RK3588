from agent.executor import ToolRegistry
from tools.common.tool_metadata import ToolMetadata


def dummy_tool():
    return "ok"


def test_valid_tool():

    registry = ToolRegistry()

    tool = ToolMetadata(
        name="test_tool",
        description="validation test tool",
        handler=dummy_tool,
        parameters={},
    )

    registry.register(tool)

    assert "test_tool" in registry


def test_invalid_tool():

    registry = ToolRegistry()

    try:

        tool = ToolMetadata(name="", description="", handler=None, parameters={})

        registry.register(tool)

    except ValueError:

        print("Validation rejected invalid tool")
        return

    raise AssertionError("Invalid tool was accepted")


if __name__ == "__main__":

    test_valid_tool()
    test_invalid_tool()

    print("Tool Validation Test OK")
