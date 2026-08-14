"""
Vision Agent Tool Registration

Sprint B1.8.2
"""

from tools.common.tool import TOOL_REGISTRY
from tools.common.tool_metadata import ToolMetadata

from .tool import VisionTool


def register_vision_tool(pipeline):

    vision_tool = VisionTool(pipeline)

    metadata = ToolMetadata(
        name=vision_tool.name,
        description=vision_tool.description,
        capabilities=list(vision_tool.capabilities),
        parameters={
            "image": {
                "type": "image",
                "required": True,
            }
        },
        handler=vision_tool.execute,
    )

    TOOL_REGISTRY[vision_tool.name] = metadata

    return vision_tool
