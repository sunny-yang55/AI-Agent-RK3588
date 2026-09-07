"""Source-level guardrails for the labelled shape sample capture tool."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WorkbenchShapeCaptureScriptTests(unittest.TestCase):
    def test_script_declares_fixed_shape_labels_and_safe_roi_capture(self):
        source = (ROOT / "scripts/capture_workbench_shape_samples.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('COLORS = ("red", "yellow", "blue", "green")', source)
        self.assertIn('SHAPES = ("cube", "cylinder", "triangular_pyramid")', source)
        self.assertIn('label = f"{args.color}_{args.shape}"', source)
        self.assertIn('choices=("train", "val", "test")', source)
        self.assertIn('label / args.split', source)
        self.assertIn('load_workbench_roi', source)
        self.assertIn('extract_single_colored_block_crop', source)
        self.assertIn('datasets/workbench_shapes_cropped', source)
        self.assertIn('save refused', source)
        self.assertIn('key == ord("s")', source)
        self.assertIn('q/Esc=quit', source)


if __name__ == "__main__":
    unittest.main()
