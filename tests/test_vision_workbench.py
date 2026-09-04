"""Tests for workbench ROI and colored-block perception."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.vision.workbench import (
    ColorBlockDetector,
    WorkbenchROI,
    load_workbench_roi,
    save_workbench_roi,
    summarize_colored_blocks,
)


class WorkbenchVisionTests(unittest.TestCase):
    def test_roi_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roi.json"
            expected = WorkbenchROI(100, 50, 800, 600)
            save_workbench_roi(path, expected)
            self.assertEqual(load_workbench_roi(path), expected)

    def test_detects_red_yellow_and_blue_inside_roi_only(self):
        image = np.full((400, 600, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (120, 100), (190, 170), (0, 0, 255), -1)
        cv2.rectangle(image, (260, 100), (330, 170), (0, 255, 255), -1)
        cv2.rectangle(image, (400, 100), (470, 170), (255, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (80, 80), (0, 0, 255), -1)
        detector = ColorBlockDetector(WorkbenchROI(100, 80, 400, 150))
        detections = detector.detect(image)
        self.assertEqual({item.color for item in detections}, {"red", "yellow", "blue"})
        self.assertEqual(len(detections), 3)

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


if __name__ == "__main__":
    unittest.main()
