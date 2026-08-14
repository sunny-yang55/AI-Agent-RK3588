"""
Tool decorator
"""

import inspect

from .tool_metadata import ToolMetadata

TOOL_REGISTRY = {}


def tool(name=None, description="", parameters=None):

    def decorator(func):

        tool_name = name or func.__name__

        # 自动解析函数参数
        if parameters is None:

            signature = inspect.signature(func)

            params = {}

            for param_name, param in signature.parameters.items():

                params[param_name] = {
                    "type": str(param.annotation),
                    "required": (param.default is inspect.Parameter.empty),
                }

            tool_parameters = params

        else:

            tool_parameters = parameters

        metadata = ToolMetadata(
            name=tool_name,
            description=description,
            parameters=tool_parameters,
            handler=func,
        )

        TOOL_REGISTRY[tool_name] = metadata

        func.tool_metadata = metadata

        return func

    return decorator


import voice_ui as ui

ui.debug("==========模型工具加载中==========")
# print("registry id =", id(TOOL_REGISTRY))
