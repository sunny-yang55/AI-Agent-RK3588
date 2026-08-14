"""
Class Tool Adapter

Convert BaseTool class instance
into framework ToolMetadata.
"""

import inspect

from .tool_metadata import ToolMetadata


def class_tool_adapter(tool_instance):

    async def handler(**kwargs):

        result = await tool_instance.execute(**kwargs)

        return result

    parameters = {}

    if hasattr(tool_instance, "detect"):

        signature = inspect.signature(tool_instance.detect)

        for name, param in signature.parameters.items():

            if name == "self":
                continue

            if param.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                continue

            parameters[name] = {
                "type": str(param.annotation),
                "required": param.default is inspect.Parameter.empty,
            }

    metadata = ToolMetadata(
        name=tool_instance.name,
        description=tool_instance.description,
        parameters=parameters,
        handler=handler,
    )

    return metadata
