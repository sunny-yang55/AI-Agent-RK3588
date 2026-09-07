"""Unit tests for the online multimodal boundary; no network is used."""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TOOLS_PACKAGE = types.ModuleType("tools")
TOOLS_PACKAGE.__path__ = [str(ROOT / "tools")]
VISION_PACKAGE = types.ModuleType("tools.vision")
VISION_PACKAGE.__path__ = [str(ROOT / "tools/vision")]
sys.modules.setdefault("tools", TOOLS_PACKAGE)
sys.modules.setdefault("tools.vision", VISION_PACKAGE)


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tools/vision" / f"{name.rsplit('.', 1)[-1]}.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load("tools.vision.session")
module = load("tools.vision.online_describer")
OnlineVisionDescriber = module.OnlineVisionDescriber
should_use_online_scene_description = module.should_use_online_scene_description


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="我看到一个白色杯子。"))]
        )


class OnlineVisionDescriberTests(unittest.TestCase):
    def test_broad_scene_question_uses_online_route(self):
        self.assertTrue(should_use_online_scene_description("看一下桌上有什么东西"))
        self.assertTrue(should_use_online_scene_description("这是什么"))
        self.assertFalse(should_use_online_scene_description("有没有红色物块"))

    def test_missing_configuration_is_not_available(self):
        with patch.dict(os.environ, {"VISION_LLM_ENABLED": "0"}, clear=True):
            describer = OnlineVisionDescriber(root=Path("/missing"))
        self.assertFalse(describer.is_available)
        self.assertEqual(describer.status, "disabled")

    def test_default_environment_matches_text_runtime(self):
        with patch.dict(os.environ, {}, clear=True):
            describer = OnlineVisionDescriber(root=Path("/missing"))
        self.assertEqual(describer.root, Path("/missing"))
        # The source-level assertion guards against accidentally restoring the
        # old, nonexistent .env.rk3588 default.
        source = (ROOT / "tools/vision/online_describer.py").read_text(encoding="utf-8")
        self.assertIn('os.getenv("AI_AGENT_ENV", ".env.qwen")', source)

    def test_image_and_fact_bounded_prompt_are_sent(self):
        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        env = {
            "VISION_LLM_ENABLED": "1",
            "VISION_LLM_BASE_URL": "https://example.invalid/v1",
            "VISION_LLM_API_KEY": "test-key",
            "VISION_LLM_MODEL": "vision-test",
        }
        with patch.dict(os.environ, env, clear=True):
            describer = OnlineVisionDescriber(client=client, root=Path("/missing"))
            answer = describer.describe(
                b"jpeg", "桌上有什么", {
                    "summary": "我在工作台上看到1个红色物块。",
                    "workbench_objects": [{"color_zh": "红色", "shape_zh": "正方体", "center_pixel": (12, 34)}],
                },
            )
        self.assertEqual(answer, "我看到一个白色杯子。")
        text = completions.kwargs["messages"][1]["content"][0]["text"]
        self.assertIn("红色", text)
        self.assertNotIn("正方体", text)
        self.assertIn("data:image/jpeg;base64", completions.kwargs["messages"][1]["content"][1]["image_url"]["url"])


if __name__ == "__main__":
    unittest.main()
