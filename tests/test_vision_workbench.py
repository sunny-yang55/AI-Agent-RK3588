"""Tests for workbench ROI and colored-block perception."""

import tempfile
import unittest
from dataclasses import replace
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
    stabilize_verified_workbench_shapes,
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
            "看到1个绿色物块。需要我定位其中某一个，还是继续查看其他物品？",
        )
        self.assertEqual(
            answer_workbench_query("有没有红色三棱锥", detections),
            "我看到了红色物块，但形状还不够稳定。为了避免误判，您可以让我再确认一次。",
        )

    def test_asr_shape_alias_keeps_answer_grounded(self):
        image = np.full((160, 200, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (40, 40), (100, 100), (0, 0, 255), -1)
        detections = ColorBlockDetector().detect(image)
        self.assertEqual(
            answer_workbench_query("有没有红色三轮锥", detections),
            "我看到了红色物块，但形状还不够稳定。为了避免误判，您可以让我再确认一次。",
        )

    def test_color_location_returns_roi_pixel_coordinates(self):
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (120, 70), (180, 130), (0, 0, 255), -1)
        detections = ColorBlockDetector(WorkbenchROI(100, 50, 150, 120)).detect(image)
        answer = answer_workbench_query("定位红色物块", detections)
        self.assertIn("(50, 50)", answer)
        self.assertIn("标定板左上角", answer)

    def test_location_without_color_asks_for_target(self):
        self.assertEqual(
            answer_workbench_query("物块坐标在哪", []),
            "请告诉我需要定位哪一种颜色和形状的物块，例如“红色正方体在哪里”。",
        )

    def test_shape_location_never_claims_unreliable_shape(self):
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (80, 80), (0, 255, 0), -1)
        cv2.circle(image, (160, 50), 30, (0, 255, 0), -1)
        answer = answer_workbench_query(
            "绿色正方体在哪里", ColorBlockDetector().detect(image)
        )
        self.assertIn("不能可靠区分", answer)
        self.assertIn("2个绿色物块", answer)

    def test_verified_shape_can_be_used_for_location(self):
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (80, 80), (0, 0, 255), -1)
        item = ColorBlockDetector().detect(image)[0]
        verified = type(item)(**{**item.__dict__, "shape": "cube", "shape_zh": "正方体", "shape_verified": True})
        answer = answer_workbench_query("红色正方体在哪里", [verified])
        self.assertIn("我已经找到1个红色正方体", answer)
        self.assertIn("需要我继续定位", answer)
        self.assertIn("工作台像素坐标", answer)

    def test_asr_tricycle_alias_can_locate_verified_triangular_pyramid(self):
        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        triangle = np.asarray([[20, 100], [60, 30], [100, 100]], dtype=np.int32)
        cv2.fillPoly(image, [triangle], (0, 0, 255))
        item = ColorBlockDetector().detect(image)[0]
        verified = replace(
            item,
            shape="triangular_pyramid",
            shape_zh="三棱锥",
            shape_verified=True,
        )
        answer = answer_workbench_query("红色三轮车在什么地方", [verified])
        self.assertIn("红色三棱锥", answer)
        self.assertIn("工作台像素坐标", answer)

    def test_stable_snapshot_ignores_one_frame_shape_flip(self):
        image = np.full((180, 220, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (40, 40), (100, 100), (0, 255, 0), -1)
        cube = ColorBlockDetector().detect(image)
        circle = np.full((180, 220, 3), 255, dtype=np.uint8)
        cv2.circle(circle, (70, 70), 30, (0, 255, 0), -1)
        cylinder = ColorBlockDetector().detect(circle)
        stable = select_stable_workbench_snapshot([cube, cube, cylinder])
        self.assertEqual(stable[0].color, "green")

    def test_verified_shape_needs_four_consistent_nearby_votes(self):
        image = np.full((180, 220, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (40, 40), (100, 100), (0, 0, 255), -1)
        cube = replace(ColorBlockDetector().detect(image)[0], shape_verified=True)
        flipped = replace(cube, shape="cylinder", shape_zh="圆柱体")
        stable = stabilize_verified_workbench_shapes(
            [[cube], [cube], [cube], [cube], [flipped]]
        )
        self.assertTrue(stable[0].shape_verified)
        self.assertEqual(stable[0].shape, "cube")

    def test_shape_with_insufficient_votes_is_not_verified(self):
        image = np.full((180, 220, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (40, 40), (100, 100), (0, 0, 255), -1)
        cube = replace(ColorBlockDetector().detect(image)[0], shape_verified=True)
        cylinder = replace(cube, shape="cylinder", shape_zh="圆柱体")
        stable = stabilize_verified_workbench_shapes(
            [[cube], [cube], [cylinder], [cylinder], [cylinder]]
        )
        self.assertFalse(stable[0].shape_verified)


if __name__ == "__main__":
    unittest.main()
