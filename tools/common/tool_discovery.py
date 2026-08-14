from typing import Any


class ToolDiscovery:
    """
    Tool Discovery Service

    v0.6.8

    负责从 ToolRegistry 中发现工具信息。
    """

    def __init__(self, registry):

        self.registry = registry

    def discover(self) -> list[dict[str, Any]]:
        """
        返回系统所有工具描述
        """

        return self.registry.describe_tools()

    def find_by_capability(self, capability: str):
        """
        根据 capability 查找工具
        """

        results = []

        for tool in self.registry.list_tools():

            if capability in tool.capabilities:

                results.append(tool)

        return results

    def count(self) -> int:
        """
        当前工具数量
        """

        return len(self.registry)
