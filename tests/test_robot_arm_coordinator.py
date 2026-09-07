"""Safety tests for visual full-frame coordinate robot planning."""

import json
import tempfile
import unittest
from pathlib import Path

from tools.robot_arm import CameraRobotCalibration, RobotArmCoordinator
from runtime.robot_arm_control import RobotArmVoiceController


def _target(**changes):
    item = {
        "color": "red",
        "shape": "cube",
        "shape_verified": True,
        "center_pixel": (500, 300),
        "center_roi": (124, 42),
    }
    item.update(changes)
    return item


class RobotArmCoordinatorTests(unittest.TestCase):
    def _calibration_file(self, directory: str) -> Path:
        path = Path(directory) / "calibration.json"
        path.write_text(json.dumps({
            "image_size": [1280, 720],
            "pixel_pts": [[0, 0]] * 9,
            "report": {"rmse_mm": 1.2, "max_error_mm": 2.1},
            "transform_type": "affine",
            "affine": [[0.1, 0, -100], [0, 0.1, 100]],
        }), encoding="utf-8")
        return path

    def test_requires_exact_verified_unique_target(self):
        coordinator = RobotArmCoordinator("missing.json")
        self.assertEqual(coordinator.prepare_pick("抓取红色正方体", [_target(shape_verified=False)]).status, "rejected")
        self.assertEqual(coordinator.prepare_pick("抓取红色正方体", [_target(), _target()]).status, "rejected")

    def test_uses_full_camera_pixel_and_preserves_roi_pixel(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = RobotArmCoordinator(self._calibration_file(directory)).prepare_pick("抓取红色正方体", [_target()])
        self.assertEqual(plan.status, "ready_for_confirmation")
        self.assertEqual(plan.camera_pixel, (500, 300))
        self.assertEqual(plan.roi_pixel, (124, 42))
        self.assertEqual(plan.robot_xy_mm, (-50.0, 130.0))

    def test_rejects_legacy_camera_resolution_before_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._calibration_file(directory)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["image_size"] = [640, 480]
            path.write_text(json.dumps(data), encoding="utf-8")
            plan = RobotArmCoordinator(path).prepare_pick("抓取红色正方体", [_target()])
        self.assertEqual(plan.status, "needs_calibration")
        self.assertIn("尺寸不匹配", plan.message)

    def test_parses_common_shape_asr_alias(self):
        self.assertEqual(
            RobotArmCoordinator.target_from_text("夹取红色三轮锥"),
            ("red", "triangular_pyramid"),
        )

    def test_calibration_refuses_low_quality_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._calibration_file(directory)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["report"] = {"rmse_mm": 23.5, "max_error_mm": 47.0}
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "精度不合格"):
                CameraRobotCalibration.load(path)

    def test_voice_pick_intent_never_moves_and_reports_missing_calibration(self):
        class Vision:
            is_running = True

            @staticmethod
            def locate_workbench():
                return [_target()]

        spoken = []
        controller = RobotArmVoiceController(
            Vision(), lambda text, **_: spoken.append(text), calibration_path="missing.json"
        )
        self.assertTrue(controller.handle("抓取红色正方体"))
        self.assertIn("标定文件不存在", spoken[0])
