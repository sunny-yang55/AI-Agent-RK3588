"""Tests for workbench ROI and colored-block perception."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.vision.workbench import (
    ColorBlockDetector,
    WorkbenchROI,
    answer_workbench_query,
    load_workbench_roi,
    is_workbench_query,
    save_workbench_roi,
    select_stable_workbench_snapshot,
    summarize_colored_blocks,
)


class WorkbenchVisionTests(unittest.TestCase):
    def test_roi_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roi.json"
            expected = WorkbenchROI(100, 50, 800, 600)
            save_workbench_roi(path, expected)
            self.assertEqual(load_workbench_roi(path), expected)

    def test_detects_four_colors_inside_roi_only(self):
        image = np.full((400, 600, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (120, 100), (190, 170), (0, 0, 255), -1)
        cv2.rectangle(image, (260, 100), (330, 170), (0, 255, 255), -1)
        cv2.rectangle(image, (400, 100), (470, 170), (255, 0, 0), -1)
        cv2.rectangle(image, (400, 240), (470, 310), (0, 255, 0), -1)
        cv2.rectangle(image, (10, 10), (80, 80), (0, 0, 255), -1)
        detector = ColorBlockDetector(WorkbenchROI(100, 80, 400, 250))
        detections = detector.detect(image)
        self.assertEqual(
            {item.color for item in detections}, {"red", "yellow", "blue", "green"}
        )
        self.assertEqual(len(detections), 4)

    def test_reports_global_and_roi_centers(self):
        image = np.full((300, 400, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (150, 100), (210, 160), (0, 0, 255), -1)
        detection = ColorBlockDetector(WorkbenchROI(100, 50, 200, 200)).detect(image)[0]
        self.assertEqual(detection.center_pixel, (180, 130))
        self.assertEqual(detection.center_roi, (80, 80))

    def test_summary_groups_colors(self):
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (70, 70), (0, 0, 255), -1)
        cv2.rectangle(image, (100, 20), (150, 70), (0, 0, 255), -1)
        summary = summarize_colored_blocks(ColorBlockDetector().detect(image))
        self.assertEqual(summary, "我在工作台上看到2个红色物块。")

    def test_classifies_top_view_circle_and_triangle(self):
        image = np.full((250, 400, 3), 255, dtype=np.uint8)
        cv2.circle(image, (100, 120), 45, (0, 255, 0), -1)
        triangle = np.asarray([[250, 170], [300, 70], [350, 170]], dtype=np.int32)
        cv2.fillPoly(image, [triangle], (255, 0, 0))
        detections = ColorBlockDetector().detect(image)
        shapes = {(item.color, item.shape) for item in detections}
        self.assertIn(("green", "cylinder"), shapes)
        self.assertIn(("blue", "triangular_pyramid"), shapes)

    def test_workbench_queries_are_routed_to_color_channel(self):
        for text in ("桌面有什么", "桌上有什么", "看到绿色物块了吗", "有没有红色方块"):
            self.assertTrue(is_workbench_query(text))
        self.assertFalse(is_workbench_query("前面有什么"))

    def test_specific_query_only_reports_matching_objects(self):
        image = np.full((220, 400, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (80, 80), (0, 0, 255), -1)
        cv2.circle(image, (180, 55), 32, (0, 255, 0), -1)
        detections = ColorBlockDetector().detect(image)
        self.assertEqual(
            answer_workbench_query("有没有绿色物块", detections),
            "看到1个绿色物块。",
        )
        self.assertEqual(
            answer_workbench_query("有没有红色三棱锥", detections),
            "看到了红色物块，但目前还不能可靠确认它的形状。",
        )

    def test_stable_snapshot_ignores_one_frame_shape_flip(self):
        image = np.full((180, 220, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (40, 40), (100, 100), (0, 255, 0), -1)
        cube = ColorBlockDetector().detect(image)
        circle = np.full((180, 220, 3), 255, dtype=np.uint8)
        cv2.circle(circle, (70, 70), 30, (0, 255, 0), -1)
        cylinder = ColorBlockDetector().detect(circle)
        stable = select_stable_workbench_snapshot([cube, cube, cylinder])
        self.assertEqual(stable[0].color, "green")


if __name__ == "__main__":
    unittest.main()
