"""
Vision Tool
Agent bridge for vision pipeline.
Sprint B1.8.2
"""

from tools.common.base_tool import BaseTool


class VisionTool(BaseTool):
    """
    Vision capability exposed to Agent.
    """

    name = "vision"

    description = "Object detection and visual perception tool"

    version = "0.1.0"

    capabilities = frozenset(
        [
            "object_detection",
            "vision",
        ]
    )

    def __init__(
        self,
        manager,
        config=None,
    ):

        super().__init__(config)

        self.manager = manager

    async def _initialize_impl(self):

        await self.manager.initialize()

    async def _cleanup_impl(self):

        await self.manager.cleanup()

    async def _execute_impl(
        self,
        image,
        **kwargs,
    ):

        result = await self.manager.process(image)

        return result
