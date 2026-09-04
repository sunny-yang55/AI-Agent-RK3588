"""Tests for YOLOv5 postprocessing and Chinese summaries."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/vision/yolov5_rknn.py"
SPEC = importlib.util.spec_from_file_location("yolov5_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
yolov5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = yolov5
SPEC.loader.exec_module(yolov5)

YoloDetection = yolov5.YoloDetection
postprocess_yolov5 = yolov5.postprocess_yolov5
summarize_detections = yolov5.summarize_detections


class YOLOv5PostprocessTests(unittest.TestCase):
    def test_empty_outputs_return_empty_arrays(self):
        outputs = [
            np.zeros((1, 255, size, size), dtype=np.float32)
            for size in (80, 40, 20)
        ]
        boxes, classes, scores = postprocess_yolov5(outputs)
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(classes.size, 0)
        self.assertEqual(scores.size, 0)

    def test_one_high_confidence_detection(self):
        outputs = [
            np.zeros((1, 255, size, size), dtype=np.float32)
            for size in (80, 40, 20)
        ]
        branch = outputs[0].reshape(3, 85, 80, 80)
        branch[0, :4, 10, 20] = 0.5
        branch[0, 4, 10, 20] = 0.9
        branch[0, 5, 10, 20] = 0.8
        boxes, classes, scores = postprocess_yolov5(outputs)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(int(classes[0]), 0)
        self.assertAlmostEqual(float(scores[0]), 0.72, places=5)

    def test_summary_groups_chinese_labels(self):
        detections = [
            YoloDetection(0, "person", "人", 0.9, (0, 0, 10, 10)),
            YoloDetection(0, "person", "人", 0.8, (20, 0, 30, 10)),
            YoloDetection(39, "bottle", "瓶子", 0.7, (0, 20, 10, 30)),
        ]
        self.assertEqual(summarize_detections(detections), "我看到2个人、1个瓶子。")

    def test_summary_uses_natural_chinese_measure_words(self):
        detections = [
            YoloDetection(63, "laptop", "笔记本电脑", 0.9, (0, 0, 10, 10)),
            YoloDetection(73, "book", "书", 0.8, (0, 0, 10, 10)),
        ]
        self.assertEqual(summarize_detections(detections), "我看到1台笔记本电脑、1本书。")

    def test_empty_summary_is_explicit(self):
        self.assertEqual(summarize_detections([]), "当前画面中没有检测到已知物体。")

    def test_summary_filters_low_confidence_candidates(self):
        detections = [
            YoloDetection(0, "person", "人", 0.29, (0, 0, 10, 10)),
            YoloDetection(39, "bottle", "瓶子", 0.75, (0, 0, 10, 10)),
        ]
        self.assertEqual(summarize_detections(detections), "我看到1个瓶子。")

    def test_summary_speaks_detection_at_point_three(self):
        detections = [
            YoloDetection(67, "cell phone", "手机", 0.30, (0, 0, 10, 10)),
        ]
        self.assertEqual(summarize_detections(detections), "我看到1部手机。")

    def test_detector_adds_static_batch_dimension(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("np.expand_dims(input_image, axis=0)", source)
        self.assertIn('data_format=["nhwc"]', source)

    def test_diagnostic_detects_venv_by_prefix_not_symlink_target(self):
        script = (MODULE_PATH.parents[2] / "scripts/test_vision_detection.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Path(sys.prefix).resolve()", script)
        self.assertNotIn("Path(sys.executable).resolve()", script)


if __name__ == "__main__":
    unittest.main()
