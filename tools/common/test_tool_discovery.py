from agent.executor import ToolRegistry
from tools.common.tool_discovery import ToolDiscovery
from tools.common.tool_metadata import ToolMetadata


def test_discovery():

    registry = ToolRegistry()

    registry.register(
        ToolMetadata(name="test_tool", description="test", capabilities=["test"])
    )

    discovery = ToolDiscovery(registry)

    tools = discovery.discover()

    assert len(tools) == 1

    result = discovery.find_by_capability("test")

    assert len(result) == 1


if __name__ == "__main__":

    test_discovery()

    print("Tool Discovery Test OK")
