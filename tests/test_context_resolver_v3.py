import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "build-v1.3.4"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT))

from tools.common.context_resolver import ContextResolver


class Memory:
    def __init__(self, focus="安徽工程大学"):
        self.focus = focus

    def get_focus(self):
        return self.focus


class ContextResolverV3Tests(unittest.TestCase):
    def setUp(self):
        self.resolver = ContextResolver()
        self.memory = Memory()

    def test_explicit_new_place_is_not_prefixed(self):
        text = "介绍一下安徽合肥"
        self.assertEqual(self.resolver.resolve(text, self.memory), text)

    def test_explicit_new_company_is_not_prefixed(self):
        text = "介绍一下长鑫存储"
        self.assertEqual(self.resolver.resolve(text, self.memory), text)

    def test_elliptical_followup_inherits_focus(self):
        self.assertEqual(
            self.resolver.resolve("有哪些专业", self.memory),
            "安徽工程大学有哪些专业",
        )

    def test_location_followup_inherits_focus(self):
        self.assertEqual(
            self.resolver.resolve("在哪里", self.memory),
            "安徽工程大学在哪里",
        )


if __name__ == "__main__":
    unittest.main()
