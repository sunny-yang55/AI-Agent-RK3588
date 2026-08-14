"""
Automatic Tool Discovery
"""

import importlib
import pkgutil

from tools.common.tool import TOOL_REGISTRY


def load_tools():

    import tools

    for module in pkgutil.walk_packages(tools.__path__, tools.__name__ + "."):

        name = module.name

        if "test" in name:
            continue

        try:

            module_obj = importlib.import_module(name)

            # 自动注册 Vision Tool
            if name.endswith("tools.vision.register"):

                from tools.vision.register import register_vision_tool

                try:

                    from tools.vision.manager import VisionPipelineManager

                    pipeline = VisionPipelineManager("config/vision.yaml").create()

                    register_vision_tool(pipeline)

                    # print("Vision tool registered")

                except Exception as e:

                    print("Vision register failed:", e)

        except Exception as e:

            import voice_ui as ui
            ui.debug(f"Tool load failed: {name} {e}")

    # print("loader registry id =", id(TOOL_REGISTRY))

    # print("loader registry =", TOOL_REGISTRY.keys())

    return TOOL_REGISTRY
