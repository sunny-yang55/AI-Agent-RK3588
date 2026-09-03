"""Static safety checks for the isolated RKNN conversion workflow."""

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class VisionModelConversionTests(unittest.TestCase):
    def test_conversion_is_fp_and_targets_rk3588(self):
        source = (ROOT / "scripts/convert_vision_model.py").read_text(encoding="utf-8")
        self.assertIn('target_platform="rk3588"', source)
        self.assertIn("do_quantization=False", source)

    def test_conversion_script_has_valid_syntax(self):
        source = (ROOT / "scripts/convert_vision_model.py").read_text(encoding="utf-8")
        ast.parse(source)

    def test_build_uses_an_isolated_environment(self):
        source = (ROOT / "scripts/build_vision_model.sh").read_text(encoding="utf-8")
        self.assertIn('env_dir="$cache/venv"', source)
        self.assertNotIn('"$root/venv/bin/pip"', source)

    def test_service_diagnostic_bootstraps_project_venv(self):
        source = (ROOT / "scripts/test_vision_service.py").read_text(encoding="utf-8")
        self.assertIn("Path(sys.prefix).resolve()", source)
        self.assertIn("os.execv", source)


if __name__ == "__main__":
    unittest.main()
